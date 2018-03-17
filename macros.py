from utils import compose, overloadmethod
from astpass import AstPass, DeepAstPass
import astnodes as E

class MacroExpander(DeepAstPass):
    pass

class MacroProcessor(AstPass):
    def __init__(self):
        self.env = {}

    @overloadmethod(use_as_default=True)
    def do_pass(self, ast):
        return ast

    # This should be sufficient to catch all 'toplevel' statements for now.
    # Macros should be restricted to the top level in the parser. In future
    # it might be desirable to allow macros to be defined more freely.
    @do_pass.on(E.Statements)
    @compose(E.Statements)
    @compose(list)
    def _(self, ast):
        for stmt in ast.stmts:
            print(stmt)
            yield self.expand(self.do_pass(stmt))

    @do_pass.on(E.MacroDefinition)
    def _(self, ast):
        self.env[ast.name] = (ast.params, ast.body)
        return E.NoneExpression(pos=ast.pos)

    expand = MacroExpander().visit

def process(ast):
    mp = MacroProcessor()
    return mp.visit(ast)
