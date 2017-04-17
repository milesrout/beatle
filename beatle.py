#!/usr/bin/env python3

import argparse
import collections
import functools
import itertools
import json
import operator
import pprint
import re
import sys

from collections import namedtuple
from itertools import groupby

from utils import *
import parser
import scanner

def parse_args():
    parser = argparse.ArgumentParser(description='ApeVM Beatle Compiler')
    parser.add_argument('input', metavar='INPUT_FILE', type=argparse.FileType('r'),
                        default=sys.stdin, nargs='?',
                        help='the input file')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=None,
                        help='the output file')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='enables verbose output mode. the more \'-v\'s, the more output.')
    parser.add_argument('--tokens', type=argparse.FileType('r'), default=open('tokens.json'),
                        help='a file with a list of tokens and their regexes')
    parser.add_argument('--keywords', type=argparse.FileType('r'), default=open('keywords.json'),
                        help='a file with a list of literal tokens (keywords)')

    PHASES = ['SCAN', 'PARSE']
    phases = parser.add_argument_group(title='phases of compilation')

    # These options *set* the phases, so they are mutually exclusive.
    ph_mutex = phases.add_mutually_exclusive_group()
    ph_mutex.add_argument('-p', '--phases', nargs='+',
                          choices=PHASES, default=PHASES,
                          help='which phases to do (SCAN, PARSE)')
    ph_mutex.add_argument('--scan', action='store_const',
                          dest='phases', const=['SCAN'],
                          help='shorthand for --phase SCAN')
    ph_mutex.add_argument('--parse', action='store_const',
                          dest='phases', const=['SCAN', 'PARSE'],
                          help='shorthand for --phase SCAN PARSE')

    # The rest of the options *add to* the phases, so any combination can
    # be added.
    #opt_flags = phases.add_mutually_exclusive_group()
    #opt_flags.add_argument('-O1', action='append_const', dest='phases',
    #                       const='BASIC_OPT',
    #                       help='basic optimisations')
    #opt_flags.add_argument('-O2', action='append_const', dest='phases',
    #                       const='ADV_OPT',
    #                       help='advanced optimisations')

    return parser.parse_args()


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
        print(exc.message)
        raise

    if verbosity >= 2:
        print('tokens:')
        print(pformat(tokens))

    if 'PARSE' not in args.phases:
        return

    try:
        ast = parser.file_input(tokens)
    except ApeError as exc:
        print('ERROR PARSING')
        print(exc.message)
        raise

    if verbosity >= 1:
        print('ast:')
        print(to_json(ast))

if __name__ == '__main__':
    main()
