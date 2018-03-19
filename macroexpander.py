from utils import overloadmethod
from astpass import DeepAstPass
import astnodes as E

# MacroExpander should look recursively at all forms it is given.
# If a form is a CallExpression where the CAR of the form is the name of a
# macro, it should be replaced with the result of running that macro on the
# content of that call expression (APPLIED).
# Otherwise, default DeepAstPass behaviour.

# SOON:
# When a macro is defined, its body should be wrapped in a FunctionExpression
# operating on expressions.
# When a macro is expanded, a new CallExpression should be created where the
# atom is the above FunctionExpression and the arguments are a representation
# of the expressions passed to the macro. This CallExpression should be
# type-checked and evaluated using the direct AST interpreter.
# This obviously precludes any kind of 'define-for-macro' functionality, which
# will have to wait for the proper bytecode-based macros.

# FUTURE:
# When a macro is defined, its body should be wrapped in a FunctionExpression
# operating on expressions, then processed by all subsequent steps in the chain
# (type-checking, bytecode generation and optimisation).
# When a macro is expanded, this function should be executed on the expressions
# passed to the macro. The resulting forms should then be substituted for the
# macro invocation.

class MacroExpander(DeepAstPass):
    def __init__(self, processor):
        self.processor = processor

    @overloadmethod(use_as_default=True)
    def visit(self, ast):
        return self.do_visit(ast)

    @visit.on(E.CallExpression)
    def _(self, ast):
        if isinstance(ast.atom, E.IdExpression):
            if ast.atom.name in self.processor.env:
                # === apply the macro here ===
                pass
        return ast

# MacroProcessor should look recursively at all the forms it is given
# If a form is a MacroDefinition form, it should add it to its
# macro environment and replace the form with a NoneExpression.
# Otherwise, it should expand all the macros in the form.

class MacroProcessor(DeepAstPass):
    def __init__(self):
        self.env = {}
        self.expand = MacroExpander(self).visit

    @overloadmethod(use_as_default=True)
    def visit(self, ast):
        return self.expand(self.do_visit(ast))

    @visit.on(E.MacroDefinition)
    def _(self, ast):
        # === define the macro properly here ===
        self.env[ast.name.name] = (ast.params, ast.body)
        return E.NoneExpression(pos=ast.pos)

def process(ast):
    mp = MacroProcessor()
    return mp.visit(ast)
