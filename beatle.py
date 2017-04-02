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

    return parser.parse_args()


def main():
    args = parse_args()
    verbosity = args.verbose

    if verbosity >= 2:
        print('args:')
        print(pformat(vars(args)))

    input_text = args.input.read()

    if verbosity >= 2:
        print('file:')
        print(input_text)

    tokens = scanner.scan(args.keywords, args.tokens, input_text)

    if verbosity >= 2:
        print('tokens:')
        print(pformat(tokens))

    ast = parser.single_input(tokens)

    if verbosity >= 1:
        print('ast:')
        print(pformat(ast))

if __name__ == '__main__':
    main()
