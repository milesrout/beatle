from utils import overloadmethod, ApeInternalError
from astpass import DeepAstPass
import cstnodes as E

class Sentinel:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'Sentinel({self.value!r})'

def is_sentinel(x):
    return isinstance(x, Sentinel)

def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False

class IsGenPass(DeepAstPass):
    """A compiler pass to test for the presence of generators"""

    def true_args(self, args):
        return any(self.true_arg(a) for a in args if a is not args and (is_iterable(a) or is_sentinel(a)))

    def true_arg(self, arg):
        if is_iterable(arg):
            return self.true_args(arg)
        if is_sentinel(arg):
            return arg.value
        raise ApeInternalError('We should\'t ever get here')

    def override_do_visit_wrapper(self, ast, new):
        if isinstance(new, Sentinel):
            return new

        if isinstance(new, bool):
            return Sentinel(new)

        # no atomic expressions can yield
        if ast is new:
            return Sentinel(False)

        cls, args = new
        try:
            return Sentinel(self.true_args(args))
        except TypeError:
            raise

    def visit(self, ast):
        return self.my_visit(ast)

    @overloadmethod(use_as_default=True)
    def my_visit(self, ast):
        return super().visit(ast)

    @my_visit.on((E.YieldFromExpression, E.YieldExpression))
    def my_visit_Yield_Or_YieldFrom_Expression(self, ast):
        return Sentinel(True)

    @my_visit.on((E.FunctionExpression, E.FunctionDefinition))
    def my_visit_FunctionDefinition_Or_FunctionExpression(self, ast):
        return Sentinel(any(self.is_gen(p.default) for p in ast.params if p.default is not None))

    @my_visit.wrapper()
    def my_visit_wrapper(self, ast, new):
        return self.override_do_visit_wrapper(ast, new)

    def is_gen(self, ast):
        return self.visit(ast).value

def is_gen(ast):
    return IsGenPass().is_gen(ast)
