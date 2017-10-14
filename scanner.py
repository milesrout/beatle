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

def is_newline(tok) -> bool:
    return tok.type == 'space' and tok.string == '\n'

def is_space(tok) -> bool:
    return tok.type == 'space'

def flatten(indent, pos, group):
    return IndentLine(indent=indent, pos=pos,
            content=list(itertools.chain.from_iterable(group)))

def count_indent(token):
    if all(x == ' ' for x in token.string):
        return len(token.string)
    raise ApeSyntaxError(msg='Tab characters are not valid as indentation', pos=token.pos)

class Scanner:
    def make_regex(self, keywords_file, tokens_file):
        """Create a regex from the contents of a keywords array and tokens dict

        The keywords array and tokens dict are obtained from files."""

        keywords_list = json.load(keywords_file, object_pairs_hook=collections.OrderedDict)
        keywords = { k.lower():f'\\b{k}\\b' for k in keywords_list }

        tokens = json.load(tokens_file, object_pairs_hook=collections.OrderedDict)

        # this ensures that keywords come before the ID token
        keywords.update(tokens)
        tokens = keywords

        token_types = list(tokens.keys())

        patterns = [f'({pat})' for pat in tokens.values()]
        regex = '|'.join(patterns)
        return token_types, re.compile(regex)

    def lex(self):
        i = 0
        while i < len(self.input_text):
            match = self.pattern.match(self.input_text, i)
            if match is None:
                bad = self.input_text[i:]
                raise self.Error(f'cannot match at {i}: {bad}', pos=i)
            start, end = match.span()
            if start != i:
                gap = self.input_text[i:start]
                raise self.Error(f'unlexable gap between {i} and {start}: {gap}', pos=i)
            token_type = self.token_types[match.lastindex - 1]
            if token_type != 'comment':
                yield self.Token(type=token_type, string=match.group(match.lastindex), pos=i)
            i = end

    def split_into_physical_lines(self, tokens):
        line = []
        pos = -1
        indent = 0
        for token in tokens:
            if len(line) == 0 and pos == -1:
                pos = token.pos
            if is_newline(token):
                yield IndentLine(indent=indent, pos=pos, content=line)
                pos = -1
                line = []
                indent = 0
            elif is_space(token):
                if len(line) == 0:
                    indent += count_indent(token)
            else:
                line.append(token)
        yield IndentLine(indent=0, pos=len(self.input_text) - 1, content=[])

    def remove_blank_lines(self, indented_lines):
        for line in indented_lines:
            if not all(is_space(tok) for tok in line.content):
                yield line

    def join_continuation_backslashes(self, lines):
        """Merge explicit continuation lines"""
        initial = None
        group = []
        for line in lines:
            if len(line.content) != 0 and line.content[-1].type == 'backslash':
                group.extend(line.content)
                group.pop()
                initial = initial or line
            else:
                if initial is None:
                    yield line
                else:
                    group.extend(line.content)
                    yield IndentLine(indent=initial.indent, pos=initial.pos, content=group)
                    initial = None
                    group = []
        if len(group) != 0:
            raise self.Error('cannot end with backslash-continuation line');

    def add_blank_line(self, lines):
        return itertools.chain(lines, [IndentLine(indent=0, pos=len(self.input_text), content=[])])

    def parse_indentation(self, lines):
        stack = [0]
        indent = [0]
        for line in lines:
            if stack[-1] is not None:
                total = stack[-1] + indent[-1]
                if line.indent > total:
                    yield self.Token('indent', ' ' * (line.indent - total), line.pos, virtual=len(stack) - 1)
                    indent[-1] += line.indent - total
                elif line.indent < total:
                    # we cannot dedent by more than the indent at this level.
                    amount = min(total - line.indent, indent[-1])
                    if amount != 0:
                        yield self.Token('dedent', ' ' * amount, line.pos, virtual=len(stack) - 1)
                        indent[-1] -= total - line.indent
            for token in line.content:
                if token.type in ['lbrack', 'lbrace', 'lparen']:
                    if stack[-1] is None:
                        stack[-1] = token.col - 1
                    indent.append(0)
                    stack.append(None)
                elif token.type in ['rbrack', 'rbrace', 'rparen']:
                    yield self.Token('newline', '\n', pos=token.pos, virtual=len(stack) - 1)
                    if indent[-1] > 0:
                        yield self.Token('dedent', ' ' * indent[-1], token.pos, virtual=len(stack) - 1)
                    stack.pop()
                    indent.pop()
                elif stack[-1] is None:
                    stack[-1] = token.col - 1
                yield token
            yield self.Token('newline', '\n', pos=line.pos, virtual=len(stack) - 1)

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
                        indent_stack.append(self.Token(
                            type=matching.type,
                            string=' ' * -diff,
                            pos=matching.pos,
                            virtual=matching.virtual))
                        break
                    else:
                        yield self.Token(
                            type='dedent',
                            string=matching.string,
                            pos=token.pos,
                            virtual=matching.virtual)
                        token = self.Token('dedent', diff * ' ', token.pos)
                        matching = indent_stack.pop()
                        diff = len(token.string) - len(matching.string)
            yield token
        if len(indent_stack) != 0:
            raise ApeSyntaxError(f'mismatched indents somehow: {indent_stack[-1]}')

    def add_eof_token(self, tokens):
        eof = self.Token('EOF', '', pos=len(self.input_text))
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
        self.token_types, self.pattern = self.make_regex(keywords, tokens)
        self.input_text = input_text

        def col(pos):
            return pos - self.input_text.rfind('\n', 0, pos)

        def line(pos):
            return self.input_text.count('\n', 0, pos) + 1

        class Error(ApeSyntaxError):
            def __init__(this, msg, pos):
                super().__init__(
                    msg=msg,
                    line=line(pos),
                    col=col(pos))

        class Token:
            def __init__(this, type, string, pos, virtual=0):
                this.type = type
                this.string = string
                this.pos = pos
                this.virtual = virtual

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

        self.Token = Token
        self.Error = Error

    def scan(self):
        tokens = self.lex()

        # Some of these return complex objects, not just
        # tokens, despite what 'tokens = step(tokens)'
        # might indicate.
        lexing_steps = [
            self.split_into_physical_lines,
            self.remove_blank_lines,
            self.join_continuation_backslashes,
            self.add_blank_line,
            self.parse_indentation,
            self.split_dedent_tokens,
            self.add_eof_token,
            #self.check_indents,
        ]

        for step in lexing_steps:
            tokens = step(tokens)

        return tokens

def scan(*args, **kwds):
    scanner = Scanner(*args, **kwds)
    return list(scanner.scan())
