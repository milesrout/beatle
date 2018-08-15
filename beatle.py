#!/usr/bin/env python3

import argparse
import os
import pathlib
import readline
import sys

from utils import ApeError, pformat, to_sexpr
import parser
import scanner
import importer
import macroexpander
import typechecker
import evaluator

PHASES = ['SCAN', 'PARSE', 'IMPORTS', 'MACROS', 'TYPES', 'EVAL']
PROMPT = "\U0001F98D "

def add_phase_arguments(parser):
    phases = parser.add_argument_group(title='phases of compilation')

    # These options *set* the phases, so they are mutually exclusive.
    ph_mutex = phases.add_mutually_exclusive_group()
    ph_mutex.add_argument('-p', '--phases', nargs='+', metavar="PHASE", choices=PHASES, default=PHASES)
    ph_mutex.add_argument('--scan', action='store_const', dest='phases', const=PHASES[:1],
                          help='shorthand for --phase ' + ' '.join(PHASES[:1]))
    ph_mutex.add_argument('--parse', action='store_const', dest='phases', const=PHASES[:2],
                          help='shorthand for --phase ' + ' '.join(PHASES[:2]))
    ph_mutex.add_argument('--imports', action='store_const', dest='phases', const=PHASES[:3],
                          help='shorthand for --phase ' + ' '.join(PHASES[:3]))
    ph_mutex.add_argument('--macros', action='store_const', dest='phases', const=PHASES[:4],
                          help='shorthand for --phase ' + ' '.join(PHASES[:4]))
    ph_mutex.add_argument('--types', action='store_const', dest='phases', const=PHASES[:5],
                          help='shorthand for --phase ' + ' '.join(PHASES[:5]))
    ph_mutex.add_argument('--eval', action='store_const', dest='phases', const=PHASES[:6],
                          help='shorthand for --phase ' + ' '.join(PHASES[:6]))

    # The rest of the options *add to* the phases, so any combination can
    # be added.
    # opt_flags = phases.add_mutually_exclusive_group()
    # opt_flags.add_argument('-O1', action='append_const', dest='phases',
    #                        const='BASIC_OPT',
    #                        help='basic optimisations')
    # opt_flags.add_argument('-O2', action='append_const', dest='phases',
    #                       const='ADV_OPT',
    #                       help='advanced optimisations')

def parse_args():
    parser = argparse.ArgumentParser(description='ApeVM Beatle Compiler')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='enables verbose output mode. the more \'-v\'s, the more output.')
    parser.add_argument('-s', '--stacktrace', action='store_true', default=False,
                        help='enable printing of stacktraces on errors that '
                             'are potentially user errors')
    parser.add_argument('--tokens', type=argparse.FileType('r'), default=open('tokens.json'),
                        help='a file with a list of tokens and their regexes')
    parser.add_argument('--keywords', type=argparse.FileType('r'), default=open('keywords.json'),
                        help='a file with a list of literal tokens (keywords)')

    subparsers = parser.add_subparsers(metavar='COMMAND', help='the following commands are built into beatle')

    # Unfortunately you can't set this in the add_subparsers command anymore
    subparsers.required = True

    repl_parser = subparsers.add_parser('repl', aliases=['r'], help='type beatle code directly into a read-eval-print loop')
    add_phase_arguments(repl_parser)

    com_parser = subparsers.add_parser('compile', aliases=['c'], help='compile a single file')
    com_parser.add_argument('input', metavar='INPUT_FILE', default=sys.stdin, nargs='?', type=argparse.FileType('r'))
    com_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=None)
    add_phase_arguments(com_parser)

    # build_parser = subparsers.add_parser('build', aliases=['b'], help='build a file along with all its dependencies')
    # build_parser.add_argument('input', metavar='INPUT_FILE', default=sys.stdin, nargs='?', type=argparse.FileType('r'))

    repl_parser.set_defaults(func=cmd_repl)
    com_parser.set_defaults(func=cmd_compile)
    # build_parser.set_defaults(func=cmd_build)

    return parser.parse_args()

def validate_phases(args):
    if args.verbose >= 2:
        print('args:')
        print(pformat(vars(args)))

    if 'PARSE' in args.phases and 'SCAN' not in args.phases:
        print('Unworkable --phase arguments: '
              'PARSE phase requires SCAN phase')
        return False

    if 'IMPORTS' in args.phases and 'PARSE' not in args.phases:
        print('Unworkable --phase arguments: '
              'IMPORTS phase requires SCAN and PARSE phases')
        return False

    if 'MACROS' in args.phases and 'IMPORTS' not in args.phases:
        print('Unworkable --phase arguments: '
              'MACROS phase requires SCAN, PARSE and IMPORTS phases')
        return False

    if 'TYPES' in args.phases and 'MACROS' not in args.phases:
        print('Unworkable --phase arguments: '
              'TYPES phase requires SCAN, PARSE, IMPORTS and MACROS phases')
        return False

    if 'EVAL' in args.phases and 'TYPES' not in args.phases:
        print('Unworkable --phase arguments: '
              'EVAL phase requires SCAN, PARSE, IMPORTS, MACROS and TYPES phases')
        return False

    return True


def scan(input_text, args, regexp=None):
    if regexp is None:
        regexp = scanner.Regexp.from_files(args.keywords, args.tokens)
    tokens = scanner.scan(args.verbose, regexp, input_text)

    if args.verbose >= 2:
        print('tokens:')
        print(pformat(tokens))

    return tokens, regexp


def parse(tokens, args, input_text, initial_production):
    ast = parser.any_input(tokens, input_text, initial_production=initial_production)

    if args.verbose >= 1:
        print('ast:')
        print(to_sexpr(ast, indent=None if args.verbose == 1 else 4))

    return ast


def imports(ast, args, input_text, regexp):
    try:
        base_search_path = os.path.dirname(args.input.name)
    except AttributeError:
        base_search_path = os.path.abspath('.')
    ast = importer.process(ast, base_search_path, (args, regexp, scan, parse))

    if args.verbose >= 1:
        print('import-expanded ast:')
        print(to_sexpr(ast, indent=None if args.verbose == 1 else 4))

    return ast


def macros(ast, args, input_text):
    ast = macroexpander.process(ast)

    if args.verbose >= 1:
        print('macro-expanded ast:')
        print(to_sexpr(ast, indent=None if args.verbose == 1 else 4))

    return ast


def types(ast, args, input_text):
    ast = typechecker.infer(ast, input_text)

    if args.verbose >= 1:
        print('type-annotated ast:')
        print(to_sexpr(ast, indent=None if args.verbose == 1 else 4))

    return ast


def evaluate(ast, args, input_text):
    result = evaluator.evaluate(ast)

    if args.verbose >= 1:
        print('bytecode:')
        print(to_sexpr(result, indent=None if args.verbose == 1 else 4))

    return result


# simple usage:
# ./beatle.py compile ./examples/stack.b            --> creates stack.bo which contains bytecode and interface
# ./beatle.py compile ./examples/queue.b            --> creates queue.bo which contains bytecode and interface
# ./beatle.py compile ./examples/two-stack-queue.b  --> reads stack.bo and queue.bo and creates two-stack-queue.bo which contains bytecode and interface

# real usage:
# ./beatle.py build ./examples/two-stack-queue.b    --> does all of the above

# repl usage:
# ./beatle.py repl

def cmd_repl(args):
    if not validate_phases(args):
        sys.exit(1)

    if 'SCAN' in args.phases:
        regexp = scanner.Regexp.from_files(args.keywords, args.tokens)

    readline.read_init_file(pathlib.Path.home() / '.inputrc')

    while True:
        try:
            input_text = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        try:
            print(process_phases(input_text, args, regexp=regexp, initial_production='single_input'))
        except ApeError:
            pass

def cmd_build(args):
    raise

def cmd_compile(args):
    if not validate_phases(args):
        sys.exit(1)

    input_text = args.input.read()

    if args.verbose >= 2:
        print('file:')
        print(input_text)

    try:
        process_phases(input_text, args, regexp=None, initial_production='file_input')
    except ApeError:
        sys.exit(1)

def process_phases(input_text, args, regexp=None, initial_production='file_input'):
    if 'SCAN' not in args.phases:
        return

    try:
        tokens, regexp = scan(input_text, args, regexp=regexp)

        if 'PARSE' not in args.phases:
            return tokens

        ast = parse(tokens, args, input_text, initial_production=initial_production)

        if 'IMPORTS' not in args.phases:
            return ast

        ast = imports(ast, args, input_text, regexp)

        if 'MACROS' not in args.phases:
            return ast

        ast = macros(ast, args, input_text)

        if 'TYPES' not in args.phases:
            return ast

        ast = types(ast, args, input_text)

        if 'EVAL' not in args.phases:
            return ast

        result = evaluate(ast, args, input_text)
        return result

    except ApeError as exc:
        print(exc.format_with_context(input_text=input_text, stacktrace=args.stacktrace))
        raise

def main():
    args = parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
