import re
from collections import namedtuple
from functools import reduce

from utils import *

class EmptyListExpression(Expression):
    def __init__(self, pos):
        self.pos = pos
    def __repr__(self):
        return 'EmptyListExpression'

class EmptyDictExpression(Expression):
    def __init__(self, pos):
        self.pos = pos
    def __repr__(self):
        return 'EmptyDictExpression'

class EmptyTupleExpression(Expression):
    def __init__(self, pos):
        self.pos = pos
    def __repr__(self):
        return 'EmptyTupleExpression'

# This is only used inside the parser, and is not present in the final AST
class Comprehension(Expression):
    def __init__(self, expr, rest, pos):
        self.expr = expr
        self.rest = rest
        self.pos = pos

# This is only used inside the parser, and is not present in the final AST
class Literal(Expression):
    def __init__(self, exprs, pos, *, trailing_comma):
        self.exprs = exprs
        self.trailing_comma = trailing_comma
        self.pos = pos

class SetComprehension(Expression):
    def __init__(self, expr, rest, pos):
        self.expr = expr
        self.rest = rest
        self.pos = pos

class DictComprehension(Expression):
    def __init__(self, expr, rest, pos):
        self.expr = expr
        self.rest = rest
        self.pos = pos

class ListComprehension(Expression):
    def __init__(self, expr, rest, pos):
        self.expr = expr
        self.rest = rest
        self.pos = pos

class GeneratorExpression(Expression):
    def __init__(self, expr, rest, pos):
        self.expr = expr
        self.rest = rest
        self.pos = pos

class SetLiteral(Expression):
    def __init__(self, exprs, pos):
        self.exprs = exprs
        self.pos = pos
    def is_gen(self):
        return any(e.is_gen() for e in self.exprs)

class DictLiteral(Expression):
    def __init__(self, exprs, pos):
        self.exprs = exprs
        self.pos = pos
    def is_gen(self):
        return any(e.is_gen() for e in self.exprs)
class DictPair(Expression):
    def __init__(self, key_expr, value_expr):
        self.key_expr = key_expr 
        self.value_expr = value_expr
        self.pos = key_expr.pos

class ListLiteral(Expression):
    def __init__(self, exprs, pos):
        self.exprs = exprs
        self.pos = pos
    def is_gen(self):
        return any(e.is_gen() for e in self.exprs)

class TupleLiteral(Expression):
    def __init__(self, exprs, pos):
        self.exprs = exprs
        self.pos = pos
    def is_gen(self):
        return any(e.is_gen() for e in self.exprs)

class Quasiquote(Expression):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

class Unquote(Expression):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

class UnquoteSplice(Expression):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

class Lazy(Expression):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

class StarParam(Expression):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos

class StarStarKwparam(Expression):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos

class Statements(Expression):
    #def __new__(self, stmts):
    #    if len(stmts) == 1:
    #        return stmts[0]
    #    return super().__new__(self)
    def __init__(self, stmts):
        self.stmts = stmts
        self.pos = stmts[0].pos
    def is_gen(self):
        return any(s.is_gen() for s in self.stmts)

class RaiseStatement(Expression):
    def __init__(self, expr, original, pos):
        self.expr = expr
        self.original = original
        self.pos = pos
    def is_gen(self):
        if self.expr is not None:
            if self.original is not None:
                return self.expr.is_gen() or self.original.is_gen()
            return self.expr.is_gen()
        return None

class YieldExpression(Expression):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos
    def is_gen(self):
        return True

class DelStatement(Expression):
    def __init__(self, exprs, pos):
        self.exprs = exprs
        self.pos = pos
    def is_gen(self):
        return any(e.is_gen() for e in self.exprs)

class AssertStatement(Expression):
    def __init__(self, exprs, pos):
        self.exprs = exprs
        self.pos = pos
    def is_gen(self):
        return any(e.is_gen() for e in self.exprs)

class GlobalStatement(Expression):
    def __init__(self, names, pos):
        self.names = names
        self.pos = pos

class NonlocalStatement(Expression):
    def __init__(self, names, pos):
        self.names = names
        self.pos = pos

class IfBranch(Expression):
    def __init__(self, cond, body, pos):
        self.cond = cond
        self.body = body
        self.pos = pos

class ElifBranch(Expression):
    def __init__(self, cond, body, pos):
        self.cond = cond
        self.body = body
        self.pos = pos

class ElseBranch(Expression):
    def __init__(self, body, pos):
        self.body = body
        self.pos = pos

class IfElifElseStatement(Expression):
    def __init__(self, if_branch, elif_branches, else_branch, pos):
        self.if_branch = if_branch
        self.elif_branches = elif_branches
        self.else_branch = else_branch
        self.pos = pos

class MatchStatement(Expression):
    def __init__(self, expr, cases, pos):
        self.expr = expr
        self.cases = cases
        self.pos = pos

class MatchCase(Expression):
    def __init__(self, pattern, body, pos):
        self.pattern = pattern
        self.body = body
        self.pos = pos

class WhileStatement(Expression):
    def __init__(self, cond, body, alt, pos):
        self.cond = cond
        self.body = body
        self.alt = alt
        self.pos = pos

class ForStatement(Expression):
    def __init__(self, assignees, iterables, body, alt, pos):
        self.assignees = assignees
        self.iterables = iterables
        self.body = body
        self.alt = alt
        self.pos = pos

class TryStatement(Expression):
    def __init__(self, body, excepts, elses, finallies, pos):
        self.body = body
        self.excepts = excepts
        self.elses = elses
        self.finallies = finallies
        self.pos = pos

class MacroDefinition(Expression):
    def __init__(self, name, params, body, return_annotation, pos):
        self.name = name
        self.params = params
        self.body = body
        self.return_annotation = return_annotation
        self.pos = pos

class FunctionDefinition(Expression):
    def __init__(self, name, params, body, return_annotation, pos):
        self.name = name
        self.params = params
        self.body = body
        self.return_annotation = return_annotation
        self.pos = pos

class FunctionExpression(Expression):
    def __init__(self, params, body, return_annotation, pos):
        self.params = params
        self.body = body
        self.return_annotation = return_annotation
        self.pos = pos
    def is_gen(self):
        return any(p.is_gen() for p in self.params)

class LambdaExpression(Expression):
    def __init__(self, args, body, pos):
        self.args = args
        self.body = body
        self.pos = pos

class InterfaceDefinition(Expression):
    def __init__(self, name, body, pos):
        self.name = name
        self.body = body
        self.pos = pos

class ClassDefinition(Expression):
    def __init__(self, name, bases, body, pos):
        self.name = name
        self.bases = bases
        self.body = body
        self.pos = pos

class WithStatement(Expression):
    def __init__(self, items, body, pos):
        self.items = items
        self.body = body
        self.pos = pos

class AsyncFunctionStatement(Expression):
    def __init__(self, defn, pos):
        self.defn = defn
        self.pos = pos

class AsyncForStatement(Expression):
    def __init__(self, for_stmt, pos):
        self.for_stmt = for_stmt
        self.pos = pos

class AsyncWithStatement(Expression):
    def __init__(self, with_stmt, pos):
        self.with_stmt = with_stmt
        self.pos = pos

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
    def __init__(self, decorators, defn, pos):
        self.decorators = decorators
        self.defn = defn
        self.pos = pos

class ImportName(Expression):
    def __init__(self, name, alias):
        self.name = name
        self.alias = alias

class ImportStatement(Expression):
    def __init__(self, names, pos):
        self.names = names
        self.pos = pos

class FromImportStatement(Expression):
    def __init__(self, name, dots, what, pos):
        self.name = name
        self.dots = dots
        self.what = what
        self.pos = pos

class ChainedAssignment(Expression):
    def __init__(self, assignees):
        self.assignees = assignees
        self.pos = assignees[0].pos
    def is_gen(self):
        return any(s.is_gen() for s in self.assignees)

class AnnotatedAssignment(Expression):
    def __init__(self, type, assignee, expr, annotation, pos):
        self.type = type
        self.assignee = assignee
        self.expr = expr
        self.annotation = annotation
        self.pos = pos

class AnnotatedExpression(Expression):
    def __init__(self, expr, annotation, pos):
        self.expr = expr
        self.annotation = annotation
        self.pos = pos

class AugmentedAssignment(Expression):
    def __init__(self, assignee, op, expr, pos):
        self.assignee = assignee
        self.op = op
        self.expr = expr
        self.pos = pos

class StarStarExpr(Expression):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

class StarExpr(Expression):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

class IfElseExpr(Expression):
    def __init__(self, expr, cond, alt):
        self.expr = expr
        self.cond = cond
        self.alt = alt
        self.pos = expr.pos

class LogicalOrExpressions(Expression):
    def __init__(self, exprs):
        self.exprs = exprs
        self.pos = exprs[0].pos

class LogicalAndExpressions(Expression):
    def __init__(self, exprs):
        self.exprs = exprs
        self.pos = exprs[0].pos

class LogicalNotExpression(Expression):
    def __init__(self, expr):
        self.expr = expr
        self.pos = expr.pos

class BitOrExpression(Expression):
    def __init__(self, exprs):
        self.exprs = exprs
        self.pos = exprs[0].pos

class BitXorExpression(Expression):
    def __init__(self, exprs):
        self.exprs = exprs
        self.pos = exprs[0].pos

class BitAndExpression(Expression):
    def __init__(self, exprs):
        self.exprs = exprs
        self.pos = exprs[0].pos

class BitShiftExpression(Expression):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.pos = left.pos

class ArithExpression(Expression):
    def __init__(self, op, left, right):
        self.op = op.type
        self.left = left
        self.right = right
        self.pos = left.pos
    def is_gen(self):
        return self.left.is_gen() or self.right.is_gen()

class UnaryExpression(Expression):
    def __init__(self, op, expr):
        self.op = op.type
        self.expr = expr
        self.pos = op.pos

class PowerExpression(Expression):
    def __init__(self, expr, exponent):
        self.expr = expr
        self.exponent = exponent
        self.pos = expr.pos

class AtomExpression(Expression):
    def __new__(self, atom, trailers):
        for trailer in trailers:
            atom = trailer.fix(atom)
        return atom

class CallExpression(Expression):
    def __init__(self, atom, args, pos):
        self.atom = atom 
        self.args = args
        self.pos = pos
    def is_gen(self):
        return self.atom.is_gen() or any(e.is_gen() for e in self.args)

class IndexExpression(Expression):
    def __init__(self, atom, indices, pos):
        self.atom = atom 
        self.indices = indices
        self.pos = pos

class AttrExpression(Expression):
    def __init__(self, atom, name, pos):
        self.atom = atom 
        self.name = name
        self.pos = pos

class AwaitExpression(Expression):
    def __init__(self, expr):
        self.expr = expr
        self.pos = expr.pos

class IntExpression(Expression):
    def __init__(self, token):
        self.base = token.type
        self.value = token.string
        self.pos = token.pos
    def is_gen(self):
        return False

class FloatExpression(Expression):
    def __init__(self, token):
        self.format = token.type
        self.value = token.string
        self.pos = token.pos
    def is_gen(self):
        return False

class IdExpression(Expression):
    def __init__(self, name):
        self.name = name.string
        self.pos = name.pos
    def is_gen(self):
        return False

class StringExpression(Expression):
    def __init__(self, unparsed):
        'Unparsed is a list of strings'
        self.unparsed = unparsed
        self.pos = unparsed[0].pos
    def is_gen(self):
        return False

class EllipsisExpression(Expression):
    def __init__(self, pos):
        self.pos = pos

class NoneExpression(Expression):
    def __init__(self, pos):
        self.pos = pos

class TrueExpression(Expression):
    def __init__(self, pos):
        self.pos = pos

class FalseExpression(Expression):
    def __init__(self, pos):
        self.pos = pos

class AttrTrailer(Expression):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
    def fix(self, atom):
        return AttrExpression(atom, self.name, self.pos)

class CallTrailer(Expression):
    def __init__(self, args, pos):
        self.args = args
        self.pos = pos
    def fix(self, atom):
        return CallExpression(atom, self.args, self.pos)

class IndexTrailer(Expression):
    def __init__(self, indices, pos):
        self.indices = indices
        self.pos = pos
    def fix(self, atom):
        return IndexExpression(atom, self.indices, self.pos)

class Index(Expression):
    def __init__(self, idx):
        self.idx = idx

class Slice(Expression):
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step

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
    def __init__(self, pos):
        self.pos = pos

class PassStatement(Expression):
    def __init__(self, pos):
        self.pos = pos
    def is_gen(self):
        return False

class BreakStatement(Expression):
    def __init__(self, pos):
        self.pos = pos

class ContinueStatement(Expression):
    def __init__(self, pos):
        self.pos = pos

class ReturnStatement(Expression):
    def __init__(self, expr, pos):
        self.pos = pos
        self.expr = expr

class TypeNameExpression(Expression):
    def __init__(self, name):
        self.name = name.name
        self.pos = name.pos

class TypeFunctionExpression(Expression):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
        self.pos = t1.pos

class TypeTupleExpression(Expression):
    def __init__(self, exprs, pos):
        self.exprs = exprs
        self.pos = pos

class TypeForallExpression(Expression):
    def __init__(self, tvars, expr, pos):
        self.tvars = tvars
        self.expr = expr
        self.pos = pos

class TypeCallExpression(Expression):
    def __init__(self, atom, args):
        self.atom = atom
        self.args = args
        self.pos = atom.pos

class NameDeclaration(Expression):
    def __init__(self, name, annotation, pos):
        self.name = name
        self.annotation = annotation
        self.pos = pos

class TypeDeclaration(Expression):
    def __init__(self, name, args, pos):
        self.name = name
        self.args = args
        self.pos = pos

class LawDeclaration(Expression):
    def __init__(self, names, expr, pos):
        self.names = names
        self.expr = expr
        self.pos = pos

class Param(Expression):
    def __init__(self, name, annotation, default):
        self.name = name
        self.annotation = annotation
        self.default = default
    def __repr__(self):
        return f'Param({self.name}, {self.annotation}, {self.default})'
    def __str__(self):
        if self.default is None and self.annotation is None:
            return self.name
        elif self.default is None:
            return f'{self.name}: {self.annotation}'
        elif self.annotation is None:
            return f'{self.name}={self.default}'
        else:
            return f'{self.name}: {self.annotation} = {self.default}'
    def is_gen(self):
        return self.default is not None and self.default.is_gen()

class Comparison(Expression):
    def __init__(self, op, a, b, pos):
        self.op = op
        self.a = a
        self.b = b
        self.pos = pos
    def is_gen(self):
        return self.a.is_gen() or self.b.is_gen()

class ComparisonChain(Expression):
    def __new__(self, chain, pos):
        """A chain of comparisons
        
        chain is something like ('0', 'lt', 'x', 'le', '1'), which
        gets translated to (('0', 'lt', 'x') && ('x', 'le' '1'))
        """
        split = list(nviews(chain, 3))[::2]
        combined = functools.reduce((lambda a, b: LogicalAndExpressions([a, b])),
            (Comparison(op, a, b, pos) for a, op, b in split))
        return combined

