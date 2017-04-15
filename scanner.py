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

def is_newline(tok) -> bool:
    return tok.type == 'space' and tok.string == '\n'

def is_space(tok) -> bool:
    return tok.type == 'space'

def flatten(indent, pos, group):
    return IndentLine(indent=indent, pos=pos,
            content=list(itertools.chain.from_iterable(group)))

def join(iterables):
    return (x for y in iterables for x in y)

def count_indent(ch):
    if ch.string == ' ':
        return 1
    elif ch.string == '\t':
        return 8
    else:
        raise 'ARGH'

class Scanner:
    def make_regex(self, keywords_file, tokens_file):
        keywords_list = json.load(keywords_file, object_pairs_hook=collections.OrderedDict)

        keywords = { k.lower():f'\\b{k}\\b' for k in keywords_list }

        tokens = json.load(tokens_file, object_pairs_hook=collections.OrderedDict)

        # this ensures that keywords come before the ID token
        keywords.update(tokens)
        tokens = keywords

        patterns = (f'(?P<{tok}>{pat})' for tok,pat in tokens.items())
        regex = '|'.join(patterns)
        return re.compile(regex)

    @compose(list)
    def lex(self):
        i = 0
        last = 0
        while True:
            to_match = self.input_text[i:]
            if to_match == '':
                return
            match = self.pattern.match(to_match)
            if match is None:
                raise RuntimeError('cannot match at {i}: {bad}'.format(
                    i=i, bad=self.input_text[i:].encode('utf-8')))
            j = match.start() # relative to i
            if j != 0:
                raise RuntimeError('unlexable gap between {i} and {j}: {gap}'.format(
                    i=i, j=i+j, gap=self.input_text[i:i+j]))
            groups = filter(operator.itemgetter(1), match.groupdict().items())
            k, v = max(groups, key=operator.itemgetter(0))
            tok = self.make_token(type=k, string=v, pos=i)
            if tok.type != 'comment':
                yield tok
            i += match.end()

    @compose(list)
    def split_into_physical_lines(self, tokens):
        """Split the list of tokens on newline tokens"""
        pos = 0
        for k, group in groupby(tokens, is_newline):
            if k:
                for i in range(len(list(group)) - 1):
                    pos += 1
                    yield PhysicalLine(pos, [])
            else:
                g = list(group)
                pos = g[0].pos
                yield PhysicalLine(pos, g)
                pos = g[-1].pos + len(g[-1].string)
        yield PhysicalLine(pos, [])

    def separate_indent_prefix(self, physical_lines, regexp=re.compile(r'(\s*)(.*)')):
        for pos, line in physical_lines:
            line1, line2 = itertools.tee(line)
            prefix = list(itertools.takewhile(is_space, line1))
            rest = list(itertools.dropwhile(is_space, line2))
            level = sum(count_indent(tok) for tok in prefix)
            yield IndentLine(indent=level, pos=pos, content=rest)

    def join_continuation_backslashes(self, lines):
        """Merge explicit continuation lines"""
        initial_line = None
        group = []
        for line in lines:
            if len(line.content) != 0 and line.content[-1].type == 'backslash':
                group.append(line.content[:-1])
                if initial_line is None:
                    initial_line = line
            else:
                group.append(line.content)
                if initial_line is not None:
                    yield flatten(initial_line.indent, initial_line.pos, group)
                else:
                    yield line
                initial_line = None
                group = []
        if len(group) != 0:
            raise RuntimeError('cannot end with a continuation line: {}'.format(
                group))

    @compose(itertools.chain.from_iterable)
    def add_virtual_newlines(self, lines):
        for i, line in enumerate(lines):
            previous_line = lines[i-1] if i != 0 else None
            if previous_line is not None:
                if previous_line.indent < line.indent:
                    token = self.make_token('virtual_indent', '', line.pos)
                    yield [token]
                elif previous_line.indent > line.indent:
                    token = self.make_token('virtual_dedent', '', line.pos)
                    yield [token]
            yield line.content
            next_line = lines[i+1] if i < len(lines) - 1 else None
            if next_line is not None:
                pos = next_line.pos - 1
                yield [self.make_token('virtual_newline', '\n', pos)]

    @compose(list)
    def join_implicit_continuation_lines(self, lines):
        """Merge implicit continuation lines"""
        bracket_stack = []
        logical_line = []
        for line in lines:
            for token in line.content:
                if token.type in ['lparen', 'lbrack', 'lbrace']:
                    bracket_stack.append(token.type)
                elif token.type in ['rparen', 'rbrack', 'rbrace']:
                    if token.type[1:] != bracket_stack[-1][1:]:
                        raise ApeError('mismatched brackets: {l} and {r}'.format(
                            l=bracket_stack[-1], r=token.type))
                    bracket_stack.pop()
            logical_line.append(line)
            if len(bracket_stack) == 0:
                if len(logical_line) == 1:
                    yield logical_line[0]
                else:
                    yield IndentLine(
                        logical_line[0].indent,
                        logical_line[0].pos,
                        list(self.add_virtual_newlines(logical_line)))
                logical_line = []
        if len(bracket_stack) != 0:
            raise ApeError(f'mismatched bracket: {bracket_stack[-1]}')

    def create_indentation_tokens(self, lines):
        yield LogicalLine(pos=lines[0].pos,
                          content=lines[0].content)
        for previous, line in nviews(lines, 2):
            spaces = ' ' * line.indent
            if previous.indent < line.indent:
                token = self.make_token('indent', spaces, line.pos)
                content = [token] + line.content
            elif previous.indent > line.indent:
                token = self.make_token('dedent', spaces, line.pos)
                content = [token] + line.content
            else:
                content = line.content
            yield LogicalLine(pos=line.pos,
                              content=content)

    def add_eof_line(self, tokens):
        return itertools.chain(tokens, [LogicalLine(pos=len(self.input_text), content=[])])

    def create_newline_tokens(self, lines):
        for a, b in nviews(lines, 2):
            yield from a.content
            yield self.make_token('newline', '\n', pos=b.pos - 1)

    def remove_remaining_whitespace(self, tokens):
        return (t for t in tokens if t.type != 'space')

    def add_eof_token(self, tokens):
        eof = self.make_token('EOF', '', pos=len(self.input_text))
        return itertools.chain(tokens, (eof,))

    @staticmethod
    @compose(list)
    def scan(*args, **kwds):
        scanner = Scanner(*args, **kwds)
        return scanner()

    def __init__(self, keywords, tokens, input_text):
        self.pattern = self.make_regex(keywords, tokens)
        self.input_text = input_text
        class Token(namedtuple('Token', 'type string pos')):
            '''Essentially one giant hack'''
            def __repr__(this):
                return f'({this.line}:{this.col}:{this.type}:{this.string!r})'
            def __iter__(this):
                yield this.type
                yield this.string
            @property
            def line(this):
                return self.input_text[:this.pos].count('\n')
            @property
            def col(this):
                before = self.input_text[:this.pos].rfind('\n')
                if before == -1:
                    return this.pos
                else:
                    return this.pos - before 
        self.make_token = Token

    def __call__(self):
        tokens = self.lex()

        # Some of these return complex objects, not just
        # tokens, despite what 'tokens = step(tokens)'
        # might indicate.
        lexing_steps = [
            self.split_into_physical_lines,
            self.separate_indent_prefix,
            self.join_continuation_backslashes,
            self.join_implicit_continuation_lines,
            self.create_indentation_tokens,
            self.add_eof_line,
            self.create_newline_tokens,
            self.remove_remaining_whitespace,
            self.add_eof_token,
        ]

        for step in lexing_steps:
            tokens = step(tokens)

        return tokens

scan = Scanner.scan
