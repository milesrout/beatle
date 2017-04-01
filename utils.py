import functools
from collections import namedtuple

Token = namedtuple('Token', 'type string')
Line = namedtuple('Line', 'indent content')

def pairs_upto(n):
    return zip(range(n - 1), range(1, n))

def precompose(f):
    def outer(g):
        @functools.wraps(g)
        def inner(*args, **kwds):
            return g(f(*args, **kwds))
        return inner
    return outer

def compose(f):
    def outer(g):
        @functools.wraps(g)
        def inner(*args, **kwds):
            return f(g(*args, **kwds))
        return inner
    return outer

