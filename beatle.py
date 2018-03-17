#!/usr/bin/env python3

import argparse
import os
import sys

from utils import ApeError, ApeSyntaxError, pformat, to_json
import parser
import scanner
import imports
import macros
import typechecker
import codegen

def parse_args():
    parser = argparse.ArgumentParser(description='ApeVM Beatle Compiler')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='enables verbose output mode. the more \'-v\'s, the more output.')
    parser.add_argument('-s', '--stacktrace', action='store_true', default=False,
                        help='enable printing of stacktraces on errors that '
                             'are potentially user errors')

    subparsers = parser.add_subparsers(metavar='COMMAND', help='the following commands are built into beatle')

    com_parser = subparsers.add_parser('compile', aliases=['c'], help='compile a single file')
    com_parser.add_argument('input', metavar='INPUT_FILE', default=sys.stdin, nargs='?', type=argparse.FileType('r'))
    com_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=None)
    com_parser.add_argument('--tokens', type=argparse.FileType('r'), default=open('tokens.json'),
                            help='a file with a list of tokens and their regexes')
    com_parser.add_argument('--keywords', type=argparse.FileType('r'), default=open('keywords.json'),
                            help='a file with a list of literal tokens (keywords)')

    PHASES = ['SCAN', 'PARSE', 'IMPORTS', 'MACROS', 'TYPES', 'CODEGEN']
    phases = com_parser.add_argument_group(title='phases of compilation')

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
    ph_mutex.add_argument('--gen', action='store_const', dest='phases', const=PHASES[:6],
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

    build_parser = subparsers.add_parser('build', aliases=['b'], help='build a file along with all its dependencies')
    build_parser.add_argument('input', metavar='INPUT_FILE', default=sys.stdin, nargs='?', type=argparse.FileType('r'))

    return parser.parse_args()

# simple usage:
# ./beatle.py compile ./examples/stack.b            --> creates stack.bo which contains bytecode and interface
# ./beatle.py compile ./examples/queue.b            --> creates queue.bo which contains bytecode and interface
# ./beatle.py compile ./examples/two-stack-queue.b  --> reads stack.bo and queue.bo and creates two-stack-queue.bo which contains bytecode and interface

# real usage:
# ./beatle.py build ./examples/two-stack-queue.b    --> does all of the above

def main():
    args = parse_args()
    verbosity = args.verbose

    if verbosity >= 2:
        print('args:')
        print(pformat(vars(args)))

    if 'ADV_OPT' in args.phases and 'BASIC_OPT' not in args.phases:
        args.phases.append('BASIC_OPT')

    if 'PARSE' in args.phases and 'SCAN' not in args.phases:
        print('Unworkable --phase arguments: PARSE phase requires SCAN phase')

    if 'IMPORTS' in args.phases and 'PARSE' not in args.phases:
        print('Unworkable --phase arguments: '
              'IMPORTS phase requires SCAN and PARSE phases')

    if 'MACROS' in args.phases and 'IMPORTS' not in args.phases:
        print('Unworkable --phase arguments: '
              'MACROS phase requires SCAN, PARSE and IMPORTS phases')

    if 'TYPES' in args.phases and 'MACROS' not in args.phases:
        print('Unworkable --phase arguments: '
              'TYPES phase requires SCAN, PARSE, IMPORTS and MACROS phases')

    if 'CODEGEN' in args.phases and 'TYPES' not in args.phases:
        print('Unworkable --phase arguments: '
              'CODEGEN phase requires SCAN, PARSE, IMPORTS, MACROS and TYPES phases')

    input_text = args.input.read()

    if verbosity >= 2:
        print('file:')
        print(input_text)

    if 'SCAN' not in args.phases:
        return

    try:
        tokens = scanner.scan(args.keywords, args.tokens, input_text)
    except ApeError as exc:
        print('ERROR SCANNING')
        print(exc.format_with_context(input_text, stacktrace=args.stacktrace))
        return

    if verbosity >= 2:
        print('tokens:')
        print(pformat(tokens))

    if 'PARSE' not in args.phases:
        return

    try:
        ast = parser.file_input(tokens)
    except ApeSyntaxError as exc:
        print('ERROR PARSING')
        print(exc.format_with_context(input_text, stacktrace=args.stacktrace))
        return

    if verbosity >= 1:
        print('ast:')
        print(to_json(ast, indent=None if verbosity == 1 else 4))

    if 'IMPORTS' not in args.phases:
        return

    base_search_path = os.path.dirname(args.input.name)
    print(base_search_path)
    try:
        ast = imports.process(ast, base_search_path)
    except ApeError as exc:
        print('IMPORT ERROR')
        print(exc.format_with_context(input_text, stacktrace=args.stacktrace))
        return

    if 'MACROS' not in args.phases:
        return

    try:
        ast = macros.process(ast)
    except ApeError as exc:
        print('MACRO EXPANSION ERROR')
        print(exc.format_with_context(input_text, stacktrace=args.stacktrace))
        return

    if verbosity >= 1:
        print('macro-expanded ast:')
        print(to_json(ast, indent=None if verbosity == 1 else 4))

    if 'TYPES' not in args.phases:
        return

    try:
        ast = typechecker.infer(ast)
    except ApeError as exc:
        print('TYPE ERROR')
        print(exc.format_with_context(input_text, stacktrace=args.stacktrace))
        return

    if verbosity >= 1:
        print('type-annotated ast:')
        print(to_json(ast, indent=None if verbosity == 1 else 4))

    if 'CODEGEN' not in args.phases:
        return

    try:
        bytecode = codegen.generate(ast)
    except ApeError as exc:
        print('ERROR GENERATING CODE')
        print(exc.format_with_context(input_text, stacktrace=args.stacktrace))
        return

    if verbosity >= 1:
        print('bytecode:')
        print(to_json(bytecode, indent=None if verbosity == 1 else 4))

if __name__ == '__main__':
    main()
