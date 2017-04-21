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
    def make_regex(self, keywords_file, tokens_file):
        keywords_list = json.load(keywords_file, object_pairs_hook=collections.OrderedDict)

        keywords = { k.lower():f'\\b{k}\\b' for k in keywords_list }

        tokens = json.load(tokens_file, object_pairs_hook=collections.OrderedDict)

        # this ensures that keywords come before the ID token
        keywords.update(tokens)
        tokens = keywords

        patterns = (f'(?P<{tok}>{pat})' for tok,pat in tokens.items())
        regex = '|'.join(patterns)
        print(regex)
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
        """Split the list of tokens on newline tokens"""
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

    def create_indentation_tokens(self, lines):
        yield LogicalLine(pos=lines[0].pos,
                          content=lines[0].content)
        for previous, line in nviews(lines, 2):
            spaces = ' ' * abs(line.indent - previous.indent)
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


    def annotate_and_split_control_tokens(self, tokens):
        indent = 0
        reduction = 0
        stack = [VirtualLevel(None, 0)]
        for token in tokens:
            if token.type in ['lbrack', 'lbrace', 'lparen']:
                stack.append(VirtualLevel(token, indent))
                yield token
            elif token.type == 'indent':
                indent += len(token.string)
                yield self.make_token(token.type, token.string, token.pos, len(stack) - 1)
            elif token.type == 'newline':
                yield self.make_token(token.type, token.string, token.pos, len(stack) - 1)
            elif token.type == 'dedent':
                amount = len(token.string)
                if amount > reduction:
                    amount -= reduction
                    reduction = 0
                else:
                    reduction -= amount
                    continue
                indent -= amount
                yield self.make_token(type='dedent',
                                      string=amount * ' ',
                                      pos=token.pos,
                                      virtual=len(stack) - 1)
            elif token.type in ['rparen', 'rbrack', 'rbrace']:
                # this ensures that you can write things like
                #     (def foo():
                #         bar())
                # instead of having to write
                #     (def foo():
                #         bar()
                #     )
                # (which is still legal).
                yield self.make_token('newline', '\n', token.pos, len(stack) - 1)

                matching = stack.pop()
                if token.type[1:] != matching.token.type[1:]:
                    raise ApeSyntaxError('mismatched brackets: {l} and {r}'.format(
                        l=stack[-1].token, r=token))
                diff = indent - matching.indent
                if diff != 0:
                    if diff > 0:
                        yield self.make_token('dedent', ' ' * diff,
                            pos=token.pos, virtual=len(stack))
                        indent = matching.indent
                        reduction += diff
                    else:
                        raise ApeSyntaxError(
                            line=token.line,
                            col=token.col,
                            msg=f'mismatched indent levels: {matching} and {token}')
                yield token
            else:
                yield token

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

    @staticmethod
    @compose(list)
    def scan(*args, **kwds):
        scanner = Scanner(*args, **kwds)
        return scanner()

    def __init__(self, keywords, tokens, input_text):
        self.pattern = self.make_regex(keywords, tokens)
        self.input_text = input_text
        class Token(namedtuple('Token', 'type string pos virtual')):
            '''Essentially one giant hack'''
            def extra_repr_content(this):
                if this.type in ['dedent', 'indent']:
                    if this.virtual:
                        v = 'X' * this.virtual
                        return f'{len(this.string)}:{v}'
                    return f'{len(this.string)}'
                if this.type == 'newline' and this.virtual:
                    v = 'X' * this.virtual
                    return f'{v}'
                if this.type in variable_content_tokens:
                    v = 'X' * this.virtual
                    return f'{this.string!r}:{v}'
                v = 'X' * this.virtual
                return f'{v}'
            def __repr__(this):
                extra = this.extra_repr_content()
                if extra is not None:
                    return f'({this.line}:{this.col}:{this.type}:{extra})'
                return f'({this.line}:{this.col}:{this.type})'
            def __iter__(this):
                yield this.type
                yield this.string
            @property
            def line(this):
                return self.input_text[:this.pos].count('\n') + 1
            @property
            def col(this):
                before = self.input_text[:this.pos].rfind('\n')
                return this.pos - before
        self.make_token_impl = Token

    def make_token(self, type, string, pos, virtual=0):
        return self.make_token_impl(type, string, pos, virtual)

    def __call__(self):
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
            self.create_indentation_tokens,
            self.add_eof_line,
            self.create_newline_tokens,
            self.annotate_and_split_control_tokens,
            self.split_dedent_tokens,
            self.add_eof_token,
        ]

        for step in lexing_steps:
            tokens = step(tokens)

        return tokens

scan = Scanner.scan
