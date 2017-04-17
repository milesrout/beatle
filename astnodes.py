import re
from collections import namedtuple

from utils import *

#### PLACE THESE SENSIBLY

class EmptyListExpression(Expression):
    def __repr__(self):
        return 'EmptyListExpression'

class EmptyDictExpression(Expression):
    def __repr__(self):
        return 'EmptyDictExpression'

class EmptyTupleExpression(Expression):
    def __repr__(self):
        return 'EmptyTupleExpression'

class DictPair(Expression):
    def __init__(self, key_expr, value_expr):
        self.key_expr = key_expr 
        self.value_expr = value_expr

class Comprehension(Expression):
    def __init__(self, expr, rest):
        self.expr = expr
        self.rest = rest

class Literal(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class SetComprehension(Expression):
    def __init__(self, expr, rest):
        self.expr = expr
        self.rest = rest

class DictComprehension(Expression):
    def __init__(self, expr, rest):
        self.expr = expr
        self.rest = rest

class ListComprehension(Expression):
    def __init__(self, expr, rest):
        self.expr = expr
        self.rest = rest

class GeneratorExpression(Expression):
    def __init__(self, expr, rest):
        self.expr = expr
        self.rest = rest

class SetLiteral(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class DictLiteral(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class ListLiteral(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class TupleLiteral(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

#### ENDPLACE

class StarParam(Expression):
    def __init__(self, name):
        self.name = name

class StarStarKwparam(Expression):
    def __init__(self, name):
        self.name = name

class Statements(Expression):
    def __init__(self, stmts):
        self.stmts = stmts

class RaiseStatement(Expression):
    def __init__(self, exprs, original):
        self.exprs = exprs
        self.original = original

class YieldExpression(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class DelStatement(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class AssertStatement(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class GlobalStatement(Expression):
    def __init__(self, names):
        self.names = names

class NonlocalStatement(Expression):
    def __init__(self, names):
        self.names = names

class IfBranch(Expression):
    def __init__(self, cond, suite):
        self.cond = cond
        self.suite = suite

class ElifBranch(Expression):
    def __init__(self, cond, suite):
        self.cond = cond
        self.suite = suite

class ElseBranch(Expression):
    def __init__(self, suite):
        self.suite = suite

class IfElifElseStatement(Expression):
    def __init__(self, branches):
        self.branches = branches

class WhileStatement(Expression):
    def __init__(self, cond, body, alt):
        self.cond = cond
        self.body = body
        self.alt = alt

class ForStatement(Expression):
    def __init__(self, assignees, iterables, body, alt):
        self.assignees = assignees
        self.iterables = iterables
        self.body = body
        self.alt = alt

class TryStatement(Expression):
    def __init__(self, body, excepts, elses, finallies):
        self.body = body
        self.excepts = excepts
        self.elses = elses
        self.finallies = finallies

class FunctionDefinition(Expression):
    def __init__(self, name, params, suite, return_annotation):
        self.name = name
        self.params = params
        self.suite = suite
        self.return_annotation = return_annotation

class FunctionExpression(Expression):
    def __init__(self, params, suite, return_annotation):
        self.params = params
        self.suite = suite
        self.return_annotation = return_annotation

class ClassDefinition(Expression):
    def __init__(self, name, bases, body):
        self.name = name
        self.bases = bases
        self.body = body

class WithStatement(Expression):
    def __init__(self, items, body):
        self.items = items
        self.body = body

class AsyncFunctionStatement(Expression):
    def __init__(self, defn):
        self.defn = defn

class AsyncForStatement(Expression):
    def __init__(self, for_stmt):
        self.for_stmt = for_stmt

class AsyncWithStatement(Expression):
    def __init__(self, with_stmt):
        self.with_stmt = with_stmt

class ExceptBlock(Expression):
    def __init__(self, test, name, body):
        self.test = test
        self.name = name
        self.body = body

class WithItem(Expression):
    def __init__(self, expr, assignee):
        self.expr = expr
        self.assignee = assignee

class Decorator(Expression):
    def __init__(self, name, args):
        self.name = name
        self.args = args

class Decorated(Expression):
    def __init__(self, decorators, defn):
        self.decorators = decorators
        self.defn = defn

class ImportName(Expression):
    def __init__(self, name, alias):
        self.name = name
        self.alias = alias

class ImportStatement(Expression):
    def __init__(self, names):
        self.names = names

class FromImportStatement(Expression):
    def __init__(self, name, dots, what):
        self.name = name
        self.dots = dots
        self.what = what

class ChainedAssignment(Expression):
    def __init__(self, assignees):
        self.assignees = assignees

class AnnotatedAssignment(Expression):
    def __init__(self, type, assignee, expr, annotation):
        self.type = type
        self.assignee = assignee
        self.expr = expr
        self.annotation = annotation

class AnnotatedExpression(Expression):
    def __init__(self, expr, annotation):
        self.expr = expr
        self.annotation = annotation

class AugmentedAssignment(Expression):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class StarExpr(Expression):
    def __init__(self, expr):
        self.expr = expr

class IfElseExpr(Expression):
    def __init__(self, expr, cond, alt):
        self.expr = expr
        self.cond = cond
        self.alt = alt

class LogicalOrExpression(Expression):
    def __init__(self, expr):
        self.expr = expr

class LogicalAndExpression(Expression):
    def __init__(self, expr):
        self.expr = expr

class LogicalNotExpression(Expression):
    def __init__(self, expr):
        self.expr = expr

class BitOrExpression(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class BitXorExpression(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class BitAndExpression(Expression):
    def __init__(self, exprs):
        self.exprs = exprs

class BitShiftExpression(Expression):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class ArithExpression(Expression):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class UnaryExpression(Expression):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class PowerExpression(Expression):
    def __init__(self, expr, exponent):
        self.expr = expr
        self.exponent = exponent

class AtomExpression(Expression):
    def __init__(self, atom, trailers):
        self.atom = atom 
        self.trailers = trailers

class AwaitExpression(Expression):
    def __init__(self, expr):
        self.expr = expr

class IntExpression(Expression):
    def __init__(self, base, expr):
        self.base = base 
        self.expr = expr

class FloatExpression(Expression):
    def __init__(self, format, expr):
        self.format = format 
        self.expr = expr

class IdExpression(Expression):
    def __init__(self, name):
        self.name = name

class StringExpression(Expression):
    def __init__(self, type, unparsed):
        self.type = type 
        self.unparsed = unparsed

class EllipsisExpression(Expression):
    pass

class NoneExpression(Expression):
    pass

class TrueExpression(Expression):
    pass

class FalseExpression(Expression):
    pass

class CallTrailer(Expression):
    def __init__(self, args):
        self.args = args

class IndexTrailer(Expression):
    def __init__(self, indices):
        self.indices = indices

class Index(Expression):
    def __init__(self, idx):
        self.idx = idx

class Slice(Expression):
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step

class AttrTrailer(Expression):
    def __init__(self, name):
        self.name = name

class StarArg(Expression):
    def __init__(self, name):
        self.name = name

class StarStarKwarg(Expression):
    def __init__(self, name):
        self.name = name

class KeywordArg(Expression):
    def __init__(self, name, expr):
        self.name = name 
        self.expr = expr

class CompForArg(Expression):
    def __init__(self, keyword, comp):
        self.keyword = keyword
        self.comp = comp

class CompForClause(Expression):
    def __init__(self, is_async, exprs, iterable):
        self.is_async = is_async 
        self.exprs = exprs 
        self.iterable = iterable

class CompIfClause(Expression):
    def __init__(self, test):
        self.test = test

class EndOfPosParams(Expression):
    pass

class PassStatement(Expression):
    pass

class BreakStatement(Expression):
    pass

class ContinueStatement(Expression):
    pass

class ReturnStatement(Expression):
    def __init__(self, expr):
        self.expr = expr

class Param(Expression):
    def __init__(self, name, annotation, default):
        self.name = name
        self.annotation = annotation
        self.default = default
    def __repr__(self):
        return f'Param({self.name}, {self.annotation}, {self.default})'
    def __str__(self):
        if self.default is None and self.annotation is None:
            return self.name.string
        elif self.default is None:
            return f'{self.name.string}: {self.annotation}'
        elif self.annotation is None:
            return f'{self.name.string}={self.default}'
        else:
            return f'{self.name.string}: {self.annotation} = {self.default}'

class ComparisonChain(Expression):
    def __init__(self, chain):
        """A chain of comparisons
        
        chain is something like ('0', 'lt', 'x', 'le', '1'), which
        gets translated to (('0', 'lt', 'x') && ('x', 'le' '1'))
        """
        self.input = chain
        self.result = list(nviews(chain, 3))[::2]

    def __repr__(self):
        return repr(self.result)

