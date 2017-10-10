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

VirtualLevel = namedtuple('VirtualLevel', 'token indent')

def token_is_newline(tok) -> bool:
    return tok.type == 'space' and tok.string == '\n'

def is_space(tok) -> bool:
    return tok.type == 'space'

def flatten(indent, pos, group):
    return IndentLine(indent=indent, pos=pos,
            content=list(itertools.chain.from_iterable(group)))

def count_indent(ch):
    if ch.string == ' ':
        return 1
    elif ch.string == '\t':
        return 8
    else:
        raise 'ARGH'

class Scanner:
    @staticmethod
    def make_regex(keywords_file, tokens_file):
        """Create a regex from the contents of a keywords array and tokens dict

        The keywords array and tokens dict are obtained from files."""

        keywords_list = json.load(keywords_file, object_pairs_hook=collections.OrderedDict)
        keywords = { k.lower():f'\\b{k}\\b' for k in keywords_list }

        tokens = json.load(tokens_file, object_pairs_hook=collections.OrderedDict)

        # this ensures that keywords come before the ID token
        keywords.update(tokens)
        tokens = keywords

        patterns = (f'(?P<{tok}>{pat})' for tok,pat in tokens.items())
        regex = '|'.join(patterns)
        return re.compile(regex)

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

    def split_into_physical_lines(self, tokens):
        pos = 0
        for is_newline, group in groupby(tokens, token_is_newline):
            if is_newline:
                for i in range(len(list(group)) - 1):
                    pos += 1
                    yield PhysicalLine(pos, [])
            else:
                g = list(group)
                pos = g[0].pos
                yield PhysicalLine(pos, g)
                pos = g[-1].pos + len(g[-1].string)
        yield PhysicalLine(pos, [])

    def separate_indent_prefix(self, physical_lines):
        for pos, line in physical_lines:
            line1, line2 = itertools.tee(line)
            prefix = list(itertools.takewhile(is_space, line1))
            rest = list(itertools.filterfalse(is_space, line2))
            level = sum(count_indent(tok) for tok in prefix)
            yield IndentLine(indent=level, pos=pos, content=rest)

    def remove_blank_lines(self, indented_lines):
        for line in indented_lines:
            if not all(is_space(tok) for tok in line.content):
                yield line

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

    @compose(list)
    def add_blank_line(self, lines):
        return itertools.chain(lines, [IndentLine(indent=0, pos=len(self.input_text), content=[])])

    def parse_indentation(self, lines):
        stack = [0]
        indent = [0]
        for line in lines:
            if stack[-1] is not None:
                total = stack[-1] + indent[-1]
                if line.indent > total:
                    yield self.make_token('indent', ' ' * (line.indent - total), line.pos, virtual=len(stack) - 1)
                    indent[-1] += line.indent - total
                elif line.indent < total:
                    # we cannot dedent by more than the indent at this level.
                    amount = min(total - line.indent, indent[-1])
                    if amount != 0:
                        yield self.make_token('dedent', ' ' * amount, line.pos, virtual=len(stack) - 1)
                        indent[-1] -= total - line.indent
            for token in line.content:
                if token.type in ['lbrack', 'lbrace', 'lparen']:
                    if stack[-1] is None:
                        stack[-1] = token.col - 1
                    indent.append(0)
                    stack.append(None)
                elif token.type in ['rbrack', 'rbrace', 'rparen']:
                    yield self.make_token('newline', '\n', pos=token.pos, virtual=len(stack) - 1)
                    if indent[-1] > 0:
                        yield self.make_token('dedent', ' ' * indent[-1], token.pos, virtual=len(stack) - 1)
                    stack.pop()
                    indent.pop()
                elif stack[-1] is None:
                    stack[-1] = token.col - 1
                yield token
            yield self.make_token('newline', '\n', pos=line.pos, virtual=len(stack) - 1)

    def split_dedent_tokens(self, tokens):
        indent_stack = []
        for token in tokens:
            if token.type == 'indent':
                indent_stack.append(token)
            elif token.type == 'dedent':
                matching = indent_stack.pop()
                diff = len(token.string) - len(matching.string)
                while diff != 0:
                    if diff < 0:
                        indent_stack.append(self.make_token(
                            type=matching.type,
                            string=' ' * -diff,
                            pos=matching.pos,
                            virtual=matching.virtual))
                        break
                    else:
                        yield self.make_token(
                            type='dedent',
                            string=matching.string,
                            pos=token.pos,
                            virtual=matching.virtual)
                        token = self.make_token('dedent', diff * ' ', token.pos)
                        matching = indent_stack.pop()
                        diff = len(token.string) - len(matching.string)
            yield token
        if len(indent_stack) != 0:
            raise ApeSyntaxError(f'mismatched indents somehow: {indent_stack[-1]}')

    def add_eof_token(self, tokens):
        eof = self.make_token('EOF', '', pos=len(self.input_text))
        return itertools.chain(tokens, (eof,))

    def check_indents(self, tokens):
        t1, t2 = itertools.tee(tokens, 2)
        total = 0
        totals = collections.defaultdict(lambda: 0)
        for token in t1:
            if token.type == 'indent':
                total += len(token.string)
                totals[token.virtual] = totals[token.virtual] + len(token.string)
            if token.type == 'dedent':
                total -= len(token.string)
                totals[token.virtual] = totals[token.virtual] - len(token.string)
        if total != 0:
            raise RuntimeError(f'Something weird is going on: {total} {totals}')
        return t2

    def __init__(self, keywords, tokens, input_text):
        self.pattern = self.make_regex(keywords, tokens)
        self.input_text = input_text

        class Token:
            def __init__(self, type, string, pos, virtual):
                self.type = type
                self.string = string
                self.pos = pos
                self.virtual = virtual

            def virtual_repr(this):
                if this.virtual:
                    return ':' + ('*' * this.virtual)
                return ''

            def extra_repr(this):
                if this.type in ['dedent', 'indent']:
                    return f':{len(this.string)}'
                if this.type in variable_content_tokens:
                    return f':{this.string!r}'
                return ''

            def __repr__(this):
                base = f'{this.line}:{this.col}:{this.type}'
                return f'({base}{this.extra_repr()}{this.virtual_repr()})'

            def __iter__(this):
                yield this.type
                yield this.string

            @property
            def line(this):
                return self.input_text.count('\n', 0, this.pos) + 1

            @property
            def col(this):
                return this.pos - self.input_text.rfind('\n', 0, this.pos)

        self.make_token_impl = Token

    def make_token(self, type, string, pos, virtual=0):
        return self.make_token_impl(type, string, pos, virtual)

    def scan(self):
        tokens = self.lex()

        # Some of these return complex objects, not just
        # tokens, despite what 'tokens = step(tokens)'
        # might indicate.
        lexing_steps = [
            self.split_into_physical_lines,
            self.separate_indent_prefix,
            self.remove_blank_lines,
            self.join_continuation_backslashes,
            self.add_blank_line,
            self.parse_indentation,
            self.split_dedent_tokens,
            self.add_eof_token,
            self.check_indents,
        ]

        for step in lexing_steps:
            tokens = step(tokens)

        return tokens

def scan(*args, **kwds):
    scanner = Scanner(*args, **kwds)
    return list(scanner.scan())
