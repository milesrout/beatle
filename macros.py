from utils import *
from astnodes import *

class MacroExpander:
    def __init__(self):
        self.env = {}

    def expand(self, ast: Expression):
        return ast

    def handle(self, ast: MacroDefinition):
        pass

    @compose(Statements)
    @compose(list)
    def process(self, ast: Statements):
        for stmt in ast.stmts:
            if isinstance(stmt, MacroDefinition):
                self.handle(stmt)
            else:
                yield self.expand(stmt)

def process(ast):
    me = MacroExpander()
    return me.process(ast)
