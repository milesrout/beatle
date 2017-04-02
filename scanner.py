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

def make_regex(keywords_file, tokens_file):
    keywords_list = json.load(keywords_file, object_pairs_hook=collections.OrderedDict)

    keywords = { k.lower():k for k in keywords_list }

    tokens = json.load(tokens_file, object_pairs_hook=collections.OrderedDict)

    # this ensures that keywords come before the ID token
    keywords.update(tokens)
    tokens = keywords

    patterns = (f'(?P<{tok}>{pat})' for tok,pat in tokens.items())
    regex = '|'.join(patterns)
    return re.compile(regex)

@compose(list)
def lex(pattern, string):
    i = 0
    last = 0
    while True:
        to_match = string[i:]
        if to_match == '':
            return
        match = pattern.match(to_match)
        if match is None:
            raise RuntimeError('cannot match at {i}: {bad}'.format(
                i=i, bad=string[i:].encode('utf-8')))
        j = match.start() # relative to i
        if j != 0:
            raise RuntimeError('unlexable gap between {i} and {j}: {gap}'.format(
                i=i, j=i+j, gap=string[i:i+j]))
        groups = sorted(match.groupdict().items(), key=operator.itemgetter(0))
        group = next((k, v) for (k, v) in groups if v is not None)
        tok = Token(*group)
        yield tok
        i += match.end()

def is_newline(tok):
    return tok.type == 'space' and tok.string == '\n'

@compose(list)
def split_into_physical_lines(tokens):
    """Split the list of tokens on newline tokens"""
    for k, group in groupby(tokens, is_newline):
        if k:
            for i in range(len(list(group)) - 1):
                yield []
        else:
            yield list(group)

def is_space(tok):
    return tok.type == 'space'

def count_indent(ch):
    if ch.string == ' ':
        return 1
    elif ch.string == '\t':
        return 8
    else:
        raise 'ARGH'

def separate_indent_prefix(lines, regexp=re.compile(r'(\s*)(.*)')):
    for line in lines:
        line1, line2 = itertools.tee(line)
        prefix = list(itertools.takewhile(is_space, line1))
        rest = list(itertools.dropwhile(is_space, line2))
        level = sum(count_indent(tok) for tok in prefix)
        yield Line(indent=level, content=rest)

def flatten(indent, logical_line):
    return Line(indent=indent,
                content=list(itertools.chain.from_iterable(logical_line)))

def join_continuation_backslashes(lines):
    """Merge explicit continuation lines"""
    indent_level = None
    logical_line = []
    for line in lines:
        if len(line.content) != 0 and line.content[-1].type == 'backslash':
            # Always keep the initial indent level
            if indent_level is None:
                indent_level = line.indent
            logical_line.append(line.content[:-1])
        else:
            logical_line.append(line.content)
            yield flatten(indent_level or line.indent, logical_line)
            logical_line = []
            indent_level = None
    if len(logical_line) != 0:
        raise RuntimeError('cannot end with a continuation line: {}'.format(
            logical_line))

def join_implicit_continuation_lines(lines):
    """Merge implicit continuation lines"""
    bracket_stack = []
    logical_line = []
    for line in lines:
        for token in line.content:
            if token.type in ['lparen', 'lbrack', 'lbrace']:
                bracket_stack.append(token.type)
            elif token.type in ['rparen', 'rbrack', 'rbrace']:
                if token.type[1:] != bracket_stack[-1][1:]:
                    raise RuntimeError('mismatched brackets: {l} and {r}'.format(
                        l=bracket_stack[-1], r=token.type))
                bracket_stack.pop()
        logical_line.append(line)
        if len(bracket_stack) == 0:
            yield flatten(logical_line[0].indent, (l.content for l in logical_line))
            logical_line = []

@precompose(list)
def create_indentation_tokens(lines):
    yield lines[0].content
    for i, j in pairs_upto(len(lines)):
        if lines[i].indent < lines[j].indent:
            yield [Token('indent', ' '*lines[j].indent)] + lines[j].content
        elif lines[i].indent > lines[j].indent:
            yield [Token('dedent', ' '*lines[j].indent)] + lines[j].content
        else:
            yield lines[j].content

@precompose(iter)
def create_newline_tokens(lines):
    for line in lines:
        yield from line
        yield Token('newline', '\n')

def remove_remaining_whitespace(tokens):
    return (t for t in tokens if t.type != 'space')

def add_eof_token(tokens):
    return itertools.chain(tokens, (Token('EOF', ''),))

@compose(list)
def scan(keywords, tokens, input_text, add_eof=False):
    regex = make_regex(keywords, tokens)
    tokens = lex(regex, input_text)

    # Some of these return complex objects, not just
    # tokens, despite what 'tokens = step(tokens)'
    # might indicate.
    lexing_steps = [
        split_into_physical_lines,
        separate_indent_prefix,
        join_continuation_backslashes,
        join_implicit_continuation_lines,
        create_indentation_tokens,
        create_newline_tokens,
        remove_remaining_whitespace,
    ]

    if add_eof:
        lexing_steps.append(add_eof_token)

    for step in lexing_steps:
        tokens = step(tokens)

    return tokens
