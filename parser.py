import collections
import functools
import itertools
import re
import sys

from collections import namedtuple

from astnodes import *
from utils import *

class Parser:
    """A parser for the Beatle programming language"""

    compound_tokens = 'if while for try with def class at async'.split()
    ops = 'plus minus times matmul div mod and or xor lsh rsh exp'.split()
    augassign_tokens = [f'aug_{op}' for op in ops]
    comparison_op_tokens = 'lt gt eq ge le ne in is'.split()
    bases = 'decimal hexadecimal octal binary'.split()
    int_tokens = [f'{base}_int' for base in bases]
    float_tokens = 'pointfloat expfloat'.split()
    string_tokens = [f'{x}_string' for x in 's d sss ddd'.split()]

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.index = 0

    def current_token(self):
        return self.tokens[self.index]

    def get_token(self, type=None):
        tok = self.tokens[self.index]
        if type is not None and tok.type != type:
            raise ApeSyntaxError(f"expected '{type}', got {tok}")
        self.index += 1
        return tok

    def expect_get(self, *expected):
        actual = self.current_token()
        if actual.type not in expected:
            friendly = '|'.join(expected)
            raise ApeSyntaxError(f"expected '{friendly}', got {actual}")
        else:
            self.index += 1
            return actual

    def expect(self, expected):
        actual = self.current_token()
        if actual.type != expected:
            raise ApeSyntaxError(f"expected '{expected}', got {actual}")
        else:
            self.index += 1

    def raise_unexpected(self):
        tok = self.current_token()
        rest = self.tokens[self.index-5:]
        raise ApeSyntaxError(f'unexpected token: {tok} ...context: {rest}')

    def accept(self, *acceptable):
        actual = self.current_token().type
        return actual in acceptable

    def accept_next(self, acceptable):
        actual = self.current_token().type
        if actual == acceptable:
            self.index += 1
        return actual == acceptable

    @compose(list)
    def file_input(self):
        """Returns a list of statement objects"""
        while not self.accept_next('EOF'):
            if self.accept_next('newline'):
                continue
            yield self.stmt()

    def single_input(self):
        """Returns a list of statement objects"""
        if self.accept(*self.compound_tokens):
            stmt = self.compound_stmt()
            self.expect('newline')
            return stmt
        elif self.accept_next('newline'):
            return
        else:
            return self.simple_stmt()

    def stmt(self):
        if self.current_token().type in self.compound_tokens:
            return self.compound_stmt()
        else:
            return self.simple_stmt()

    def simple_stmt(self):
        stmts = []
        stmts.append(self.small_stmt())
        while self.accept_next('semicolon'):
            if self.accept_next('newline'):
                return stmts
            stmts.append(self.small_stmt())
        self.expect('newline')
        return stmts

    def testlist(self):
        raise NotImplementedError

    def small_stmt(self):
        if self.accept_next('del'):
            return self.del_stmt()
        if self.accept_next('pass'):
            return self.pass_stmt()
        if self.accept_next('assert'):
            return self.assert_stmt()
        if self.accept_next('global'):
            return self.global_stmt()
        if self.accept_next('nonlocal'):
            return self.nonlocal_stmt()
        if self.accept_next('break'):
            return self.break_stmt()
        if self.accept_next('continue'):
            return self.continue_stmt()
        if self.accept_next('return'):
            return self.return_stmt()
        if self.accept_next('raise'):
            return self.raise_expr()
        if self.accept_next('yield'):
            return self.yield_expr()
        if self.accept('import', 'from'):
            return self.import_stmt()
        return self.expr_stmt()

    def del_stmt(self):
        return DelStatement(self.exprlist())

    def pass_stmt(self):
        return PassStatement()

    def assert_stmt(self):
        exprs = [self.test()]
        while self.accept_next('comma'):
            exprs.append(self.test())
        return AssertStatement(exprs)

    def global_stmt(self):
        names = [self.name()]
        while self.accept_next('comma'):
            names.append(self.name())
        return GlobalStatement(names)

    def nonlocal_stmt(self):
        names = [self.name()]
        while self.accept_next('comma'):
            names.append(self.name())
        return NonlocalStatement(names)

    def break_stmt(self):
        return BreakStatement()

    def continue_stmt(self):
        return ContinueStatement()

    def return_stmt(self):
        if self.accept('semicolon', 'newline'):
            return ReturnStatement(None)
        return ReturnStatement(self.testlist())

    def raise_expr(self):
        if self.accept('semicolon', 'newline'):
            return RaiseStatement(exc=None, original=None)
        exc = self.test()
        if self.accept_next('from'):
            return RaiseStatement(exc, original=self.test())
        return RaiseStatement(exc, original=None)

    def yield_expr(self):
        if self.accept_next('from'):
            return YieldExpression(self.test())
        return YieldExpression(self.testlist())

    def import_stmt(self):
        if self.accept_next('from'):
            return self.import_from()
        elif self.accept_next('import'):
            return self.import_name()
        raise LogicError(f'{import_stmt} called in the wrong context')

    def import_name(self):
        return ImportStatement(names=self.dotted_as_names())

    def import_from(self):
        dots = self.import_dots()
        name = None
        if self.accept('id'):
            name = self.dotted_name()
        self.expect('import')
        if self.accept_next('asterisk'):
            return FromImportStatement(name=name, dots=dots, what=None)
        if self.accept_next('lparen'):
            what = self.import_as_names()
            self.expect('rparen')
            return FromImportStatement(name=name, dots=dots, what=what)
        return FromImportStatement(name=name, dots=dots, what=self.import_as_names())

    @compose(sum)
    def import_dots(self):
        while self.accept('dot', 'ellipsis'):
            if self.accept_next('ellipsis'):
                yield 3
            if self.accept_next('dot'):
                yield 1

    def import_as_name(self):
        name = self.name()
        if self.accept_next('as'):
            return ImportName(name, alias=self.name())
        return ImportName(name, alias=None)

    def dotted_as_name(self):
        name = self.dotted_name()
        if self.accept_next('as'):
            return ImportName(name, alias=self.name())
        return ImportName(name, alias=None)

    @compose(list)
    def import_as_names(self):
        yield self.import_as_name()
        while self.accept_next('comma'):
            if not self.accept('id'):
                return
            yield self.import_as_name()

    @compose(list)
    def dotted_as_names(self):
        yield self.dotted_as_name()
        while self.accept_next('comma'):
            yield self.dotted_as_name()

    @compose(list)
    def dotted_name(self):
        yield self.name()
        while self.accept_next('dot'):
            yield self.name()

    def expr_stmt(self):
        tlse = self.testlist_star_expr()
        if self.accept_next('colon'):
            annotation = self.test()
            if self.accept_next('equal'):
                return AnnotatedAssignment(
                    type='equal',
                    assignee=tlse,
                    expr=self.test(),
                    annotation=annotation)
            return AnnotatedExpression(tlse, annotation)
        if self.accept(*self.augassign_tokens):
            augtype = self.get_token().type
            if self.accept_next('yield'):
                expr = self.yield_expr()
            else:
                expr = self.testlist()
            return AugmentedAssignment(augtype, expr)
        exprs = [tlse]
        while self.accept_next('equal'):
            if self.accept_next('yield'):
                exprs.append(self.yield_expr())
            else:
                exprs.append(self.testlist_star_expr())
        if len(exprs) == 1:
            return tlse
        else:
            return Assignment(exprs)

    def testlist_star_expr(self):
        exprs = []
        if self.accept_next('asterisk'):
            exprs.append(StarExpr(self.expr()))
        else:
            exprs.append(self.test())
        while self.accept_next('comma'):
            if self.accept('equal', 'colon', *self.augassign_tokens):
                return exprs
            if self.accept_next('asterisk'):
                exprs.append(StarExpr(self.expr()))
            else:
                exprs.append(self.test())
        return exprs

    def compound_stmt(self):
        if self.accept_next('if'):
            return self.if_stmt()
        elif self.accept_next('while'):
            return self.while_stmt()
        elif self.accept_next('for'):
            return self.for_stmt()
        elif self.accept_next('try'):
            return self.try_stmt()
        elif self.accept_next('with'):
            return self.with_stmt()
        elif self.accept_next('def'):
            return self.funcdef()
        elif self.accept_next('class'):
            return self.classdef()
        elif self.accept_next('at'):
            return self.decorated()
        elif self.accept_next('async'):
            return self.async_stmt()
        self.raise_unexpected()

    def if_stmt(self):
        def cond_suite():
            cond = self.test()
            self.expect('colon')
            suite = self.suite()
            return (cond, suite)
        branches = [cond_suite()]
        while self.accept_next('elif'):
            branches.append(cond_suite())
        if self.accept_next('else'):
            branches.append((None, self.suite()))
        return IfElifElseStatement(branches)

    def async_funcdef(self):
        self.expect('def')
        return AsyncFunctionStatement(self.funcdef())

    def async_stmt(self):
        if self.accept_next('def'):
            return AsyncFunctionStatement(self.funcdef())
        if self.accept_next('with'):
            return AsyncWithStatement(self.with_stmt())
        if self.accept_next('for'):
            return AsyncForStatement(self.for_stmt())
        self.raise_unexpected()

    def while_stmt(self):
        cond = self.test()
        self.expect('colon')
        suite = self.suite()
        if self.accept_next('else'):
            self.expect('colon')
            alt = self.suite()
            return WhileStatement(cond, suite, alt)
        return WhileStatement(cond, suite, alt=None)

    def for_stmt(self):
        assignees = self.exprlist()
        self.expect('in')
        iterables = self.testlist()
        self.expect('colon')
        body = self.suite()
        if self.expect_next('else'):
            self.expect('colon')
            alt = self.suite()
            return ForStatement(assignees, iterables, body, alt)
        return ForStatement(assignees, iterables, body, alt=None)

    def try_stmt(self):
        self.expect('colon')
        try_body = self.suite()
        excepts = []
        elses = []
        finallies = []
        while self.accept('except', 'else', 'finally'):
            if self.accept_next('except'):
                excepts.append(self.except_block())
            elif self.accept_next('else'):
                elses.append(self.else_block())
            elif self.accept_next('finally'):
                finallies.append(self.finally_block())
        return TryStatement(body=try_body,
                            excepts=excepts,
                            elses=elses,
                            finallies=finallies)
    def except_block(self):
        if self.accept_next('colon'):
            return ExceptBlock(body=self.suite())
        test = self.test()
        if self.accept_next('as'):
            name = self.name()
            self.expect('colon')
            return ExceptBlock(test=test, name=name, body=self.suite())
        return ExceptBlock(test=test, name=None, body=self.suite())

    def else_block(self):
        self.expect('colon')
        return self.suite()

    def finally_block(self):
        self.expect('colon')
        return self.suite()

    def with_stmt(self):
        items = [self.with_item()]
        while self.accept_next('comma'):
            items.append(self.with_item())
        self.expect('colon')
        return WithStatement(items, self.suite())

    def with_item(self):
        expr = self.test()
        if self.accept_next('as'):
            return WithItem(expr=expr, assignee=self.expr())
        return WithItem(expr=expr, assignee=None)

    def classdef(self):
        name = self.name()
        if self.accept_next('lparen'):
            if self.accept_next('rparen'):
                bases = []
            else:
                bases = self.arglist()
        self.expect('colon')
        return ClassDef(name, bases, body=self.suite())

    def decorator(self):
        name = self.dotted_name()
        if self.accept_next('lparen'):
            if self.accept_next('rparen'):
                args = []
            else:
                args = self.arglist()
        else:
            args = None
        self.expect('newline')
        return Decorator(name, args)

    def decorated(self):
        decorators = self.decorators()
        if self.accept_next('class'):
            defn = self.classdef()
        elif self.accept_next('async'):
            defn = self.async_funcdef()
        elif self.accept_next('def'):
            defn = self.funcdef()
        else:
            raise_unexpected()
        return Decorated(decorators=decorators, defn=defn)

    @compose(list)
    def decorators(self):
        while self.accept_next('at'):
            yield self.decorator()

    def funcdef(self):
        name = self.get_token('id')
        params = self.parameters()
        if self.accept_next('arrow'):
            return_annotation = self.test()
        else:
            return_annotation = None
        self.expect('colon')
        suite = self.suite()
        return FuncDef(name, params, suite, return_annotation)

    def parameters(self):
        self.expect('lparen')
        if self.accept_next('rparen'):
            return []
        params = self.typedparamslist()
        self.expect('rparen')
        return params

    def kwparam(self):
        param = StarStarKwparam(self.tfpdef())
        self.accept_next('comma')
        return param

    def varparams(self):
        if self.accept('id'):
            params = [StarParam(self.tfpdef())]
        else:
            params = [EndOfPosParams()]
        while self.accept_next('comma'):
            if self.accept_next('double_asterisk'):
                return params + [self.kwparam()]
            if self.accept('id'):
                params.append(self.dtfpdef())
            else:
                return params

    def dtfpdef(self):
        name, annotation = self.tfpdef()
        if self.accept_next('equal'):
            return Param(name, annotation, self.test())
        else:
            return Param(name, annotation, None)

    def typedparamslist(self):
        if self.accept_next('double_asterisk'):
            return self.kwparam()
        elif self.accept_next('asterisk'):
            return self.varparams()
        params = [self.dtfpdef()]
        while self.accept_next('comma'):
            if self.accept('id'):
                params.append(self.dtfpdef())
                continue
            if self.accept_next('asterisk'):
                return params + self.varparams()
            if self.accept_next('double_asterisk'):
                return params + [self.kwparam()]
            break
        return params

    def tfpdef(self):
        name = self.get_token('id')
        if self.accept_next('colon'):
            annotation = self.test()
        else:
            annotation = None
        return (name, annotation)

    def suite(self):
        if self.accept_next('newline'):
            self.expect('indent')
            stmts = []
            while not self.accept_next('dedent'):
                stmts.append(self.stmt())
            return Statements(stmts)
        return self.simple_stmt()

    def test(self):
        if self.accept_next('lambda'):
            return self.lambdef()
        expr = self.or_test()
        if self.accept_next('if'):
            cond = self.or_expr()
            self.expect('else')
            alt = self.test()
            return IfElseExpr(expr, cond, alt)
        return expr

    def or_test(self):
        exprs = [self.and_test()]
        while self.accept_next('or'):
            exprs.append(self.and_test())
        if len(exprs) == 1:
            return exprs[0]
        return LogicalOrExpression(exprs)

    def and_test(self):
        exprs = [self.not_test()]
        while self.accept_next('and'):
            exprs.append(self.not_test())
        if len(exprs) == 1:
            return exprs[0]
        return LogicalAndExpression(exprs)

    def not_test(self):
        if self.accept_next('not'):
            return LogicalNotExpression(self.not_test())
        return self.comparison()

    def comparison(self):
        exprs = [self.expr()]
        while self.accept('is', 'not', *self.comparison_op_tokens):
            if self.accept_next('is'):
                if self.accept_next('not'):
                    exprs.append('is not')
                else:
                    exprs.append('is')
            elif self.accept_next('not'):
                self.expect('in')
                exprs.append('not in')
            elif self.accept(*self.comparison_op_tokens):
                exprs.append(self.get_token().type)
            exprs.append(self.expr())
        if len(exprs) == 1:
            return exprs[0]
        return ComparisonChain(exprs)

    def expr(self):
        exprs = [self.bitxor_expr()]
        while self.accept_next('bit_or'):
            exprs.append(self.xor_expr())
        if len(exprs) == 1:
            return exprs[0]
        return BitOrExpression(exprs)

    def bitxor_expr(self):
        exprs = [self.bitand_expr()]
        while self.accept_next('bit_xor'):
            exprs.append(self.bitand_expr())
        if len(exprs) == 1:
            return exprs[0]
        return BitXorExpression(exprs)

    def bitand_expr(self):
        exprs = [self.bitshift_expr()]
        while self.accept_next('bit_and'):
            exprs.append(self.bitshift_expr())
        if len(exprs) == 1:
            return exprs[0]
        return BitAndExpression(exprs)

    def bitshift_expr(self):
        expr = self.arith_expr()
        if self.accept('bit_lsh', 'bit_rsh', 'bit_asr'):
            shift_type = self.get_token().type
            return BitShiftExpression(shift_type, expr, self.bitshift_expr())
        return expr

    def arith_expr(self):
        term = self.term()
        if self.accept('plus', 'minus'):
            return ArithExpression(self.get_token().type, term, self.arith_expr())
        return term

    def term(self):
        factor = self.factor()
        if self.accept('asterisk', 'at', 'div', 'mod', 'truediv'):
            return TermExpression(self.get_token().type, factor, self.term())
        return factor

    def factor(self):
        if self.accept('plus', 'minus', 'tilde'):
            return FactorExpression(self.get_token().type, self.factor())
        return self.power()

    def power(self):
        atom_expr = self.atom_expr()
        if self.accept_next('double_asterisk'):
            return PowerExpression(atom_expr, self.factor())
        return atom_expr

    def atom_expr(self):
        prepend_await = self.accept_next('await')
        atom = self.atom()
        trailers = []
        while self.accept('lparen', 'lbrack', 'dot'):
            trailers.append(self.trailer())
        if len(trailers) != 0:
            atom = AtomExpression(atom, trailers)
        if prepend_await:
            return AwaitExpression(atom)
        return atom

    def trailer(self):
        if self.accept_next('lparen'):
            return CallTrailer(self.call_trailer())
        if self.accept_next('lbrack'):
            return IndexTrailer(self.index_trailer())
        if self.accept_next('dot'):
            return AttrTrailer(self.attr_trailer())
        self.raise_unexpected()

    def call_trailer(self):
        if self.accept_next('rparen'):
            return []
        return self.arglist()

    def arglist(self):
        args = [self.argument()]
        while self.accept_next('comma'):
            if self.accept_next('rparen'):
                return args
            args.append(self.argument())
        self.expect('rparen')
        return args

 ##############################################################################
 #                XXX: DOES THIS APPLY TO THIS PARSER?                        #
 #                                                                            #
 # The reason that keywords are test nodes instead of NAME is that using NAME #
 # results in an ambiguity. ast.c makes sure it's a NAME.                     #
 # "test '=' test" is really "keyword '=' test", but we have no such token.   #
 # These need to be in a single rule to avoid grammar that is ambiguous       #
 # to our LL(1) parser. Even though 'test' includes '*expr' in star_expr,     #
 # we explicitly match '*' here, too, to give it proper precedence.           #
 # Illegal combinations and orderings are blocked in ast.c:                   #
 # multiple (test comp_for) arguments are blocked; keyword unpackings         #
 # that precede iterable unpackings are blocked; etc.                         #
 ##############################################################################
    def argument(self):
        if self.accept_next('asterisk'):
            return StarArg(self.test())
        if self.accept_next('double_asterisk'):
            return StarStarKwarg(self.test())
        keyword = self.test()
        if self.accept_next('equal'):
            return KeywordArg(keyword, self.test())
        if self.accept('async', 'for'):
            return CompForArg(self.comp_for())
        return keyword

    def index_trailer(self):
        if self.accept_next('rbrack'):
            return []
        raise NotImplementedError

    def attr_trailer(self):
        raise NotImplementedError

    def atom(self):
        if self.accept_next('lparen'):
            raise NotImplementedError
        if self.accept_next('lbrack'):
            raise NotImplementedError
        if self.accept_next('lbrace'):
            raise NotImplementedError
        if self.accept('id'):
            return self.name()
        if self.accept(*self.int_tokens):
            return self.int_number()
        if self.accept(*self.float_tokens):
            return self.float_number()
        if self.accept(*self.string_tokens):
            return self.string()
        if self.accept_next('ellipsis'):
            return EllipsisExpression()
        if self.accept_next('None'):
            return NoneExpression()
        if self.accept_next('True'):
            return TrueExpression()
        if self.accept_next('False'):
            return FalseExpression()
        self.raise_unexpected()

    def int_number(self):
        return IntExpression(*self.expect_get(*self.int_tokens))

    def float_number(self):
        return FloatExpression(*self.expect_get(*self.float_tokens))

    def name(self):
        return IdExpression(self.expect_get('id').string)

    def string(self):
        return StringExpression(*self.expect_get(*self.string_tokens))

def single_input(tokens):
    return Parser(tokens).single_input()

def file_input(tokens):
    return Parser(tokens).file_input()

def eval_input(tokens):
    return Parser(tokens).eval_input()
