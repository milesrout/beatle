import functools

from astnodes import *
from utils import *

def overload(f):
    registry = {}
    @functools.wraps(f)
    def overloaded(x, *args, **kwds):
        for k, v in registry.items():
            if isinstance(x, k):
                return v(x, *args, **kwds)
        raise TypeError('no overload found for {}'.format(x.__class__))
    def on(t):
        def register(g):
            if registry.get(t) is None:
                registry[t] = g
            else:
                raise ValueError('can\'t overload on the same type twice')
        return register
    overloaded.on = on
    return overloaded

@overload
def annotate(ast):
    ...

@annotate.on(Statements)
def _(ast):
    return Statements(list(map(annotate, ast.stmts)))
