import re
from collections import namedtuple

from utils import *

Param = namedtuple('Param', 'name annotation default')
StarParam = namedtuple('StarParam', 'name')
StarStarKwparam = namedtuple('StarStarKwparam', 'name')
EndOfPosParams = namedtuple('EndOfPosParams', [])

Statements = namedtuple('Statements', 'stmts')
PassStatement = namedtuple('PassStatement', [])
BreakStatement = namedtuple('BreakStatement', [])
ContinueStatement = namedtuple('ContinueStatement', [])
RaiseStatement = namedtuple('RaiseStatement', 'exprs original')
DelStatement = namedtuple('DelStatement', 'exprs')
AssertStatement = namedtuple('AssertStatement', 'exprs')
GlobalStatement = namedtuple('GlobalStatement', 'names')
NonlocalStatement = namedtuple('NonlocalStatement', 'names')
IfElifElseStatement = namedtuple('IfStatement', 'branches')
WhileStatement = namedtuple('WhileStatement', 'cond body alt')
ForStatement = namedtuple('ForStatement', 'assignees iterables body alt')
TryStatement = namedtuple('TryStatement', 'body excepts elses finallies')
FuncDef = namedtuple('FuncDef', 'name params suite return_annotation')
ClassDef = namedtuple('ClassDef', 'name bases body')
AsyncFunctionStatement = namedtuple('AsyncFunctionStatement', 'defn')
AsyncForStatement = namedtuple('AsyncForStatement', 'for_stmt')
AsyncWithStatement = namedtuple('AsyncWithStatement', 'with_stmt')

ExceptBlock = namedtuple('ExceptBlock', 'test name body')
WithItem = namedtuple('WithItem', 'expr assignee')

Decorator = namedtuple('Decorator', 'name args')
Decorated = namedtuple('Decorated', 'decorators defn')

ImportName = namedtuple('ImportName', 'name alias')
ImportStatement = namedtuple('ImportStatement', 'names')
FromImportStatement = namedtuple('FromImportStatement', 'name dots what')

AnnotatedAssignment = namedtuple('AnnotatedAssignment', 'type assignee expr annotation')
AnnotatedExpression = namedtuple('AnnotatedExpression', 'expr annotation')
AugmentedAssignment = namedtuple('AugmentedAssignment', 'op expr')

StarExpr = namedtuple('StarExpr', 'expr')
IfElseExpr = namedtuple('IfElseExpr', 'expr cond alt')
LogicalOrExpression = namedtuple('LogicalOrExpression', 'expr')
LogicalAndExpression = namedtuple('LogicalAndExpression', 'expr')
LogicalNotExpression = namedtuple('LogicalNotExpression', 'expr')
BitOrExpression = namedtuple('BitOrExpression', 'exprs')
BitXorExpression = namedtuple('BitXorExpression', 'exprs')
BitAndExpression = namedtuple('BitAndExpression', 'exprs')
BitShiftExpression = namedtuple('BitShiftExpression', 'op left right')
ArithExpression = namedtuple('ArithExpression', 'op left right')
TermExpression = namedtuple('TermExpression', 'op left right')
FactorExpression = namedtuple('FactorExpression', 'expr')
PowerExpression = namedtuple('PowerExpression', 'expr')
AtomExpression = namedtuple('AtomExpression', 'atom trailers')
AwaitExpression = namedtuple('AwaitExpression', 'expr')

IntExpression = namedtuple('IntExpression', 'base expr')
FloatExpression = namedtuple('FloatExpression', 'format expr')
IdExpression = namedtuple('IdExpression', 'name')
StringExpression = namedtuple('StringExpression', 'type unparsed')
EllipsisExpression = namedtuple('EllipsisExpression', [])
NoneExpression = namedtuple('NoneExpression', [])
TrueExpression = namedtuple('TrueExpression', [])
FalseExpression = namedtuple('FalseExpression', [])

CallTrailer = namedtuple('CallTrailer', 'args')
IndexTrailer = namedtuple('IndexTrailer', 'indices')
AttrTrailer = namedtuple('AttrTrailer', 'name')

StarArg = namedtuple('StarArg', 'name')
StarStarKwarg = namedtuple('StarStarKwarg', 'name')
KeywordArg = namedtuple('KeywordArg', 'name expr')
CompForArg = namedtuple('CompForArg', 'comp')

CompForClause = namedtuple('CompForClause', 'is_async exprs iterable')
CompIfClause = namedtuple('CompIfClause', 'test')

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

