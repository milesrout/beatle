from utils import overloadmethod, ApeSyntaxError, to_sexpr
from astpass import DeepAstPass
import cstnodes as E

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

    @visit.on(E.ControlStructureExpression)
    def visit_ControlStructureExpression(self, ast):
        if ast.components[0].name in self.processor.env:
            # === apply the macro here ===
            return self.processor.env[ast.components[0].name](ast)
        return ast

    @visit.on(E.CallExpression)
    def visit_CallExpression(self, ast):
        if isinstance(ast.atom, E.IdExpression):
            if ast.atom.name in self.processor.env:
                # === apply the macro here ===
                return self.processor.env[ast.atom.name](ast)
        return ast

# ControlStructureExpander are used for expanding built-in control structures.
# In the future, as the capacities of the language are extended to include
# things like unwind-protect, tagbody/go, etc., it will be possible to remove
# some of these and implement these control structures using macros.

def if_expand(ast):
    ifs = [E.IfBranch(c.params[0], c.body, c.pos) for c in ast.components if c.name == 'if']
    elifs = [E.ElifBranch(c.params[0], c.body, c.pos) for c in ast.components if c.name == 'elif']
    elses = [E.ElseBranch(c.body, c.pos) for c in ast.components if c.name == 'else']
    if len(elses) == 0:
        elses = [None]
    if len(ifs) > 1:
        raise ApeSyntaxError('Too many if branches', [i.pos for i in ifs])
    if len(ifs) > 1:
        raise ApeSyntaxError('Too many else branches', [i.pos for i in elses])
    return E.IfElifElseStatement(ifs[0], elifs, elses[0], ast.pos)

def for_expand(ast):
    fors = [c for c in ast.components if c.name == 'for']
    elses = [c for c in ast.components if c.name == 'else']
    if len(fors) > 1:
        raise ApeSyntaxError('Too many for branches', [f.pos for f in fors])
    if len(elses) > 1:
        raise ApeSyntaxError('Too many else branches', [e.pos for e in elses])
    return E.ForStatement(fors[0].params[0], fors[0].params[2], fors[0].body, elses[0].body if len(elses) == 1 else None, ast.pos)

def while_expand(ast):
    whiles = [(c.params[0], c.body) for c in ast.components if c.name == 'while']
    elses = [c.body for c in ast.components if c.name == 'else']
    if len(elses) == 0:
        elses = [None]
    if len(whiles) > 1:
        raise ApeSyntaxError('Too many while branches', [w.pos for w in whiles])
    if len(elses) > 1:
        raise ApeSyntaxError('Too many else branches', [e.pos for e in elses])
    return E.WhileStatement(*whiles[0], elses[0], ast.pos)

def with_expand(ast):
    return E.WithStatement(
        [E.WithItem(ast.components[0].params[0],
                    ast.components[0].params[2] if len(ast.components[0].params) >= 3 else None)],
        ast.components[0].body,
        ast.pos)

def except_expand(comp):
    if len(comp.params) == 0:
        return E.ExceptBlock(test=None, name=None, body=comp.body)
    if len(comp.params) == 1:
        return E.ExceptBlock(test=comp.params[0], name=None, body=comp.body)
    if len(comp.params) == 3:
        return E.ExceptBlock(test=comp.params[0], name=comp.params[2], body=comp.body)
    raise ApeSyntaxError('Invalid except block', comp.pos)

def try_expand(ast):
    trys = [c for c in ast.components if c.name == 'try']
    excepts = [except_expand(c) for c in ast.components if c.name == 'except']
    elses = [c for c in ast.components if c.name == 'else']
    finallys = [c for c in ast.components if c.name == 'finally']
    if len(trys) > 1:
        raise ApeSyntaxError('Too many try branches', [e.pos for e in trys])
    if len(elses) > 1:
        raise ApeSyntaxError('Too many else branches', [e.pos for e in elses])
    if len(finallys) > 1:
        raise ApeSyntaxError('Too many finally branches', [e.pos for e in finallys])
    return E.TryStatement(trys[0].body, excepts, [e.body for e in elses], [f.body for f in finallys], ast.pos)

# MacroProcessor should look recursively at all the forms it is given
# If a form is a MacroDefinition form, it should add it to its
# macro environment and replace the form with a NoneExpression.
# Otherwise, it should expand all the macros in the form.

class MacroProcessor(DeepAstPass):
    def __init__(self):
        self.env = {
            'if': if_expand,
            'for': for_expand,
            'while': while_expand,
            'with': with_expand,
            'try': try_expand
        }
        self.expand = MacroExpander(self).visit

    @overloadmethod(use_as_default=True)
    def visit(self, ast):
        new = self.do_visit(ast)
        if ast is not new:
            return self.expand(new)
        else:
            return ast

    @visit.on(E.MacroDefinition)
    def _(self, ast):
        # === define the macro properly here ===
        self.env[ast.name.name] = (ast.params, ast.body)
        return E.NoneExpression(pos=ast.pos)

def process(ast):
    mp = MacroProcessor()
    return mp.visit(ast)
