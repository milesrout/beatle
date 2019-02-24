from collections import namedtuple

class EmptyList:
    def __str__(self):
        return '[]'

    def __repr__(self):
        return 'EmptyList()'

class EmptyDict:
    def __str__(self):
        return '{}'

    def __repr__(self):
        return 'EmptyDict()'

class EmptyTuple:
    def __str__(self):
        return 'set()'

    def __repr__(self):
        return 'EmptyTuple()'

class SetComp:
    def __init__(self, expr, clauses):
        self.expr = expr
        self.clauses = clauses

class DictComp:
    def __init__(self, expr, rest):
        self.expr = expr
        self.rest = rest

class ListComp:
    def __init__(self, expr, rest):
        self.expr = expr
        self.rest = rest

class GenExpr:
    def __init__(self, expr, rest):
        self.expr = expr
        self.rest = rest

class SetLit:
    def __init__(self, exprs):
        self.exprs = exprs

class DictLit:
    def __init__(self, exprs):
        self.exprs = exprs

class DictKV:
    def __init__(self, key_expr, value_expr):
        self.key_expr = key_expr
        self.value_expr = value_expr

class ListLit:
    def __init__(self, exprs):
        self.exprs = exprs

Tuple = namedtuple('Tuple', 'exprs')

Quote = namedtuple('Quote', 'cls, args')

Quasiquote = namedtuple('Quasiquote', 'expr')

Unquote = namedtuple('Unquote', 'expr')

UnquoteSplice = namedtuple('UnquoteSplice', 'expr')

#class Lazy(Expression):
#    def __init__(self, expr):
#        self.expr = expr
#
#class StarParam(Expression):
#    def __init__(self, name):
#        self.name = name
#
#class StarStarKwparam(Expression):
#    def __init__(self, name):
#        self.name = name

class Statements:
    def __new__(cls, stmts):
        if len(stmts) == 1:
            return stmts[0]
        return super().__new__(cls)

    def __init__(self, stmts):
        self.stmts = stmts
        self.pos = stmts[0].pos

class Raise:
    def __init__(self, expr, original):
        self.expr = expr
        self.original = original
#    def is_gen(self):
#        if self.expr is not None:
#            if self.original is not None:
#                return self.expr.is_gen() or self.original.is_gen()
#            return self.expr.is_gen()
#        return None

Yield = namedtuple('Yield', 'expr')

#class DelStatement(Expression):
#    def __init__(self, exprs):
#        self.exprs = exprs
#    def is_gen(self):
#        return any(e.is_gen() for e in self.exprs)
#
#class AssertStatement(Expression):
#    def __init__(self, exprs):
#        self.exprs = exprs
#    def is_gen(self):
#        return any(e.is_gen() for e in self.exprs)
#
#class GlobalStatement(Expression):
#    def __init__(self, names):
#        self.names = names
#
#class NonlocalStatement(Expression):
#    def __init__(self, names):
#        self.names = names

class IfBranch:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

class ElifBranch:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

class ElseBranch:
    def __init__(self, body):
        self.body = body

class IfElifElse:
    def __init__(self, if_branch, elif_branches, else_branch):
        self.if_branch = if_branch
        self.elif_branches = elif_branches
        self.else_branch = else_branch

#class MatchStatement(Expression):
#    def __init__(self, expr, cases):
#        self.expr = expr
#        self.cases = cases
#
#class MatchCase(Expression):
#    def __init__(self, pattern, body):
#        self.pattern = pattern
#        self.body = body
#
#class WhileStatement(Expression):
#    def __init__(self, cond, body, alt):
#        self.cond = cond
#        self.body = body
#        self.alt = alt
#
#class ForStatement(Expression):
#    def __init__(self, assignees, iterables, body, alt):
#        self.assignees = assignees
#        self.iterables = iterables
#        self.body = body
#        self.alt = alt
#
#class TryStatement(Expression):
#    def __init__(self, body, excepts, elses, finallies):
#        self.body = body
#        self.excepts = excepts
#        self.elses = elses
#        self.finallies = finallies
#
#class MacroDefinition(Expression):
#    def __init__(self, name, params, body, return_annotation):
#        self.name = name
#        self.params = params
#        self.body = body
#        self.return_annotation = return_annotation
#

FunctionDefinition = namedtuple('FunctionDefinition', 'name params defaults body')

Function = namedtuple('Function', 'params defaults body')

#class LambdaExpression(Expression):
#    def __init__(self, args, body):
#        self.args = args
#        self.body = body

NamespaceDefn = namedtuple('NamespaceDefn', 'name, key, expr')

NamespaceRefDefn = namedtuple('NamespaceRefDefn', 'name, key')

SignatureDefn = namedtuple('SignatureDefn', 'name, body')

#class ModuleDefinition(Expression):
#    def __init__(self, name, bases, body):
#        self.name = name
#        self.bases = bases
#        self.body = body
#
#class WithStatement(Expression):
#    def __init__(self, items, body):
#        self.items = items
#        self.body = body
#
#class ExceptBlock(Expression):
#    def __init__(self, test, name, body):
#        self.test = test
#        self.name = name
#        self.body = body
#
#class WithItem(Expression):
#    def __init__(self, expr, assignee):
#        self.expr = expr
#        self.assignee = assignee
#
#class Decorator(Expression):
#    def __init__(self, name, args):
#        self.name = name
#        self.args = args
#
#class Decorated(Expression):
#    def __init__(self, decorators, defn):
#        self.decorators = decorators
#        self.defn = defn
#
#class ImportName(Expression):
#    def __init__(self, name, alias):
#        self.name = name
#        self.alias = alias
#
#class ImportStatement(Expression):
#    def __init__(self, names):
#        self.names = names
#
#class FromImportStatement(Expression):
#    def __init__(self, name, dots, what):
#        self.name = name
#        self.dots = dots
#        self.what = what

Assignment = namedtuple('Assignment', 'targets expr')

#class ChainedAssignment(Expression):
#    def __init__(self, assignees):
#        self.assignees = assignees
#        self.pos = assignees[0].pos
#    def is_gen(self):
#        return any(s.is_gen() for s in self.assignees)
VariableDeclaration = namedtuple('VariableDeclaration', 'assignee, expr, annotation')

#class AnnotatedExpression(Expression):
#    def __init__(self, expr, annotation):
#        self.expr = expr
#        self.annotation = annotation
#
#class AugmentedAssignment(Expression):
#    def __init__(self, assignee, op, expr):
#        self.assignee = assignee
#        self.op = op
#        self.expr = expr
StarStar = namedtuple('StarStar', 'expr')

Star = namedtuple('Star', 'expr')

IfElse = namedtuple('IfElse', 'expr, cond, alt')

LogicalOr = namedtuple('LogicalOr', 'exprs')

LogicalAnd = namedtuple('LogicalAnd', 'exprs')

LogicalNot = namedtuple('LogicalNot', 'expr')

#class BitOrExpression(Expression):
#    def __init__(self, exprs):
#        self.exprs = exprs
#        self.pos = exprs[0].pos
#
#class BitXorExpression(Expression):
#    def __init__(self, exprs):
#        self.exprs = exprs
#        self.pos = exprs[0].pos
#
#class BitAndExpression(Expression):
#    def __init__(self, exprs):
#        self.exprs = exprs
#        self.pos = exprs[0].pos
#
#class BitShiftExpression(Expression):
#    def __init__(self, op, left, right):
#        self.op = op
#        self.left = left
#        self.right = right
#        self.pos = left.pos
Arith = namedtuple('Arith', 'op, left, right')

#Arith = namedtuple('Arith', 'op left right')

Unary = namedtuple('Unary', 'op, expr')

#class PowerExpression(Expression):
#    def __init__(self, expr, exponent):
#        self.expr = expr
#        self.exponent = exponent
#        self.pos = expr.pos
#
#class AtomExpression(Expression):
#    def __new__(self, atom, trailers):
#        for trailer in trailers:
#            atom = trailer.fix(atom)
#        return atom

Call = namedtuple('Call', 'f args')

#class IndexExpression(Expression):
#    def __init__(self, atom, indices):
#        self.atom = atom
#        self.indices = indices
#
#class AttrExpression(Expression):
#    def __init__(self, atom, name):
#        self.atom = atom
#        self.name = name
#
#class AwaitExpression(Expression):
#    def __init__(self, expr):
#        self.expr = expr
#        self.pos = expr.pos

Int = namedtuple('Int', 'base value')

#class IntExpression(Expression):
#    def __init__(self, token):
#        self.base = token.type
#        self.value = token.string
#        self.pos = token.pos
#    def is_gen(self):
#        return False
#

Float = namedtuple('Float', 'format value')

Id = namedtuple('Id', 'name')

String = namedtuple('String', 'string')

#class StringExpression(Expression):
#    def __init__(self, unparsed):
#        'Unparsed is a list of strings'
#        self.unparsed = unparsed
#        self.pos = unparsed[0].pos
#    def is_gen(self):
#        return False
#
#class EllipsisExpression(Expression):
#    def __init__(self):

NoneExpr = namedtuple('NoneExpr', '')

Bool = namedtuple('Bool', 'value')

#class TrueExpression(Expression):
#    def __init__(self):
#
#class FalseExpression(Expression):
#    def __init__(self):
#
#class AttrTrailer(Expression):
#    def __init__(self, name):
#        self.name = name
#    def fix(self, atom):
#        return AttrExpression(atom, self.name, self.pos)
#
#class CallTrailer(Expression):
#    def __init__(self, args):
#        self.args = args
#    def fix(self, atom):
#        return CallExpression(atom, self.args, self.pos)
#
#class IndexTrailer(Expression):
#    def __init__(self, indices):
#        self.indices = indices
#    def fix(self, atom):
#        return IndexExpression(atom, self.indices, self.pos)
#
#class Index(Expression):
#    def __init__(self, idx):
#        self.idx = idx
#
#class Slice(Expression):
#    def __init__(self, start, end, step):
#        self.start = start
#        self.end = end
#        self.step = step
#
#class StarArg(Expression):
#    def __init__(self, name):
#        self.name = name
#
#class StarStarKwarg(Expression):
#    def __init__(self, name):
#        self.name = name
#
#class KeywordArg(Expression):
#    def __init__(self, name, expr):
#        self.name = name
#        self.expr = expr
#
#class CompForArg(Expression):
#    def __init__(self, keyword, comp):
#        self.keyword = keyword
#        self.comp = comp
#
#class CompForClause(Expression):
#    def __init__(self, exprs, iterable):
#        self.exprs = exprs
#        self.iterable = iterable
#
#class CompIfClause(Expression):
#    def __init__(self, test):
#        self.test = test
#
#class EndOfPosParams(Expression):
#    def __init__(self):
#
class Pass:
    pass

#class BreakStatement(Expression):
#    def __init__(self):
#
#class ContinueStatement(Expression):
#    def __init__(self):

class Return:
    def __init__(self, expr):
        self.expr = expr

#class TypeNameExpression(Expression):
#    def __init__(self, name):
#        self.name = name.name
#        self.pos = name.pos
#
#class TypeFunctionExpression(Expression):
#    def __init__(self, t1, t2):
#        self.t1 = t1
#        self.t2 = t2
#        self.pos = t1.pos
#
#class TypeTupleExpression(Expression):
#    def __init__(self, exprs):
#        self.exprs = exprs
#
#class TypeForallExpression(Expression):
#    def __init__(self, tvars, expr):
#        self.tvars = tvars
#        self.expr = expr
#
#class TypeCallExpression(Expression):
#    def __init__(self, atom, args):
#        self.atom = atom
#        self.args = args
#        self.pos = atom.pos
#
#class NameDeclaration(Expression):
#    def __init__(self, name, annotation):
#        self.name = name
#        self.annotation = annotation
#
#class TypeDeclaration(Expression):
#    def __init__(self, name, args):
#        self.name = name
#        self.args = args
#
#class LawDeclaration(Expression):
#    def __init__(self, names, expr):
#        self.names = names
#        self.expr = expr
#
#class Param(Expression):
#    def __init__(self, name, annotation, default):
#        self.name = name
#        self.annotation = annotation
#        self.default = default
#    def __repr__(self):
#        return f'Param({self.name}, {self.annotation}, {self.default})'
#    def __str__(self):
#        if self.default is None and self.annotation is None:
#            return self.name
#        elif self.default is None:
#            return f'{self.name}: {self.annotation}'
#        elif self.annotation is None:
#            return f'{self.name}={self.default}'
#        else:
#            return f'{self.name}: {self.annotation} = {self.default}'
#    def is_gen(self):
#        return self.default is not None and self.default.is_gen()

Comparison = namedtuple('Comparison', 'op a b')

#class Comparison(Expression):
#    def __init__(self, op, a, b):
#        self.op = op
#        self.a = a
#        self.b = b
#    def is_gen(self):
#        return self.a.is_gen() or self.b.is_gen()
#
#class ComparisonChain(Expression):
#    def __new__(self, chain):
#        """A chain of comparisons
#
#        chain is something like ('0', 'lt', 'x', 'le', '1'), which
#        gets translated to (('0', 'lt', 'x') && ('x', 'le' '1'))
#        """
#        split = list(nviews(chain, 3))[::2]
#        combined = functools.reduce((lambda a, b: LogicalAndExpressions([a, b])),
#            (Comparison(op, a, b) for a, op, b in split))
#        return combined
#
# # These are only used inside the parser, and are not present in the final AST
#
#class Comprehension(Expression):
#    def __init__(self, expr, rest):
#        self.expr = expr
#        self.rest = rest
#
#class Literal(Expression):
#    def __init__(self, exprs, pos, *, trailing_comma):
#        self.exprs = exprs
#        self.trailing_comma = trailing_comma
