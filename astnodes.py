import re
from collections import namedtuple

from utils import *

class StarParam:
    def __init__(self, name):
        self.name = name

class StarStarKwparam:
    def __init__(self, name):
        self.name = name

class Statements:
    def __init__(self, stmts):
        self.stmts = stmts

class RaiseStatement:
    def __init__(self, exprs, original):
        self.exprs = exprs
        self.original = original

class DelStatement:
    def __init__(self, exprs):
        self.exprs = exprs

class AssertStatement:
    def __init__(self, exprs):
        self.exprs = exprs

class GlobalStatement:
    def __init__(self, names):
        self.names = names

class NonlocalStatement:
    def __init__(self, names):
        self.names = names

class IfElifElseStatement:
    def __init__(self, branches):
        self.branches = branches

class WhileStatement:
    def __init__(self, cond, body, alt):
        self.cond = cond
        self.body = body
        self.alt = alt

class ForStatement:
    def __init__(self, assignees, iterables, body, alt):
        self.assignees = assignees
        self.iterables = iterables
        self.body = body
        self.alt = alt

class TryStatement:
    def __init__(self, body, excepts, elses, finallies):
        self.body = body
        self.excepts = excepts
        self.elses = elses
        self.finallies = finallies

class FuncDef:
    def __init__(self, name, params, suite, return_annotation):
        self.name = name
        self.params = params
        self.suite = suite
        self.return_annotation = return_annotation

class ClassDef:
    def __init__(self, name, bases, body):
        self.name = name
        self.bases = bases
        self.body = body

class AsyncFunctionStatement:
    def __init__(self, defn):
        self.defn = defn

class AsyncForStatement:
    def __init__(self, for_stmt):
        self.for_stmt = for_stmt

class AsyncWithStatement:
    def __init__(self, with_stmt):
        self.with_stmt = with_stmt

class ExceptBlock:
    def __init__(self, test, name, body):
        self.test = test
        self.name = name
        self.body = body

class WithItem:
    def __init__(self, expr, assignee):
        self.expr = expr
        self.assignee = assignee

class Decorator:
    def __init__(self, name, args):
        self.name = name
        self.args = args

class Decorated:
    def __init__(self, decorators, defn):
        self.decorators = decorators
        self.defn = defn

class ImportName:
    def __init__(self, name, alias):
        self.name = name
        self.alias = alias

class ImportStatement:
    def __init__(self, names):
        self.names = names

class FromImportStatement:
    def __init__(self, name, dots, what):
        self.name = name
        self.dots = dots
        self.what = what

class ChainedAssignment:
    def __init__(self, assignees):
        self.assignees = assignees

class AnnotatedAssignment:
    def __init__(self, type, assignee, expr, annotation):
        self.type = type
        self.assignee = assignee
        self.expr = expr
        self.annotation = annotation

class AnnotatedExpression:
    def __init__(self, expr, annotation):
        self.expr = expr
        self.annotation = annotation

class AugmentedAssignment:
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class StarExpr:
    def __init__(self, expr):
        self.expr = expr

class IfElseExpr:
    def __init__(self, expr, cond, alt):
        self.expr = expr
        self.cond = cond
        self.alt = alt

class LogicalOrExpression:
    def __init__(self, expr):
        self.expr = expr

class LogicalAndExpression:
    def __init__(self, expr):
        self.expr = expr

class LogicalNotExpression:
    def __init__(self, expr):
        self.expr = expr

class BitOrExpression:
    def __init__(self, exprs):
        self.exprs = exprs

class BitXorExpression:
    def __init__(self, exprs):
        self.exprs = exprs

class BitAndExpression:
    def __init__(self, exprs):
        self.exprs = exprs

class BitShiftExpression:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class ArithExpression:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class TermExpression:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class FactorExpression:
    def __init__(self, expr):
        self.expr = expr

class PowerExpression:
    def __init__(self, expr):
        self.expr = expr

class AtomExpression:
    def __init__(self, atom, trailers):
        self.atom = atom 
        self.trailers = trailers

class AwaitExpression:
    def __init__(self, expr):
        self.expr = expr

class IntExpression:
    def __init__(self, base, expr):
        self.base = base 
        self.expr = expr

class FloatExpression:
    def __init__(self, format, expr):
        self.format = format 
        self.expr = expr

class IdExpression:
    def __init__(self, name):
        self.name = name

class StringExpression:
    def __init__(self, type, unparsed):
        self.type = type 
        self.unparsed = unparsed

class EllipsisExpression:
    pass

class NoneExpression:
    pass

class TrueExpression:
    pass

class FalseExpression:
    pass

class CallTrailer:
    def __init__(self, args):
        self.args = args

class IndexTrailer:
    def __init__(self, indices):
        self.indices = indices

class AttrTrailer:
    def __init__(self, name):
        self.name = name

class StarArg:
    def __init__(self, name):
        self.name = name

class StarStarKwarg:
    def __init__(self, name):
        self.name = name

class KeywordArg:
    def __init__(self, name, expr):
        self.name = name 
        self.expr = expr

class CompForArg:
    def __init__(self, comp):
        self.comp = comp

class CompForClause:
    def __init__(self, is_async, exprs, iterable):
        self.is_async = is_async 
        self.exprs = exprs 
        self.iterable = iterable

class CompIfClause:
    def __init__(self, test):
        self.test = test

class EndOfPosParams:
    pass

class PassStatement:
    pass

class BreakStatement:
    pass

class ContinueStatement:
    pass

class Param:
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

class ComparisonChain:
    def __init__(self, chain):
        """A chain of comparisons
        
        chain is something like ('0', 'lt', 'x', 'le', '1'), which
        gets translated to (('0', 'lt', 'x') && ('x', 'le' '1'))
        """
        self.input = chain
        self.result = list(nviews(chain, 3))[::2]

    def __repr__(self):
        return repr(self.result)

