import re
from collections import namedtuple

from utils import *

StarParam = namedtuple('StarParam', 'name')
StarStarKwparam = namedtuple('StarStarKwparam', 'name')

PassStatement = namedtuple('PassStatement', [])
IfElifElseStatement = namedtuple('IfStatement', 'branches')
Statements = namedtuple('Statements', 'stmts')
FuncDef = namedtuple('FuncDef', 'name params suite return_annotation')

StarExpr = namedtuple('StarExpr', 'expr')
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

class Param:
    def __init__(self, fpdef, default):
        self.name, self.annotation = fpdef
        self.default = default
    def __repr__(self):
        return f'Param(({self.name}, {self.annotation}), {self.default})'
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

