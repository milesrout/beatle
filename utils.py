import functools
import pprint
from collections import namedtuple
from itertools import tee

pformat = pprint.PrettyPrinter(indent=4, compact=True).pformat

variable_content_tokens = [
    'id',
    'sstring',
    'dstring',
    'pointfloat',
    'expfloat',
    'decimal_int',
    'hexadecimal_int',
    'octal_int',
    'binary_int'
]

irrelevant_content_tokens = ['newline', 'indent', 'dedent', 'EOF']

class Token(namedtuple('Token', 'type string')):
    def __repr__(self):
        if self.type in variable_content_tokens:
            return f'{self.type}={self.string!r}'
        if self.type in irrelevant_content_tokens:
            return repr(self.type).upper()
        return repr(self.string)
Line = namedtuple('Line', 'indent content')

def pairs_upto(n):
    return zip(range(n - 1), range(1, n))

def nviews(iterable, n):
    (*teed,) = tee(iterable, n)
    for i in range(len(teed)):
        for j in range(len(teed) - i - 1):
            next(teed[i], None)
    return zip(*reversed(teed))

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

def tap(f):
    def outer(g):
        @functools.wraps(g)
        def inner(*args, **kwds):
            x = g(*args, **kwds)
            f(g.__name__, x)
            return x
        return inner
    return outer

class ApeError(Exception):
    pass

class ApeSyntaxError(ApeError):
    pass
