import functools
import json
import pprint
import traceback

from collections import namedtuple
from itertools import tee, zip_longest

class Type:
    pass

class Expression:
    pass

def to_json(x, indent=None):
    return MyJSONEncoder(indent=indent).encode(x)

class MyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Type):
            return (o.__class__.__name__, vars(o))
        if isinstance(o, Expression):
            return (o.__class__.__name__, vars(o))
        if o.__class__.__name__ == 'Token':
            return repr(o)
        return super().default(o)

pformat = pprint.PrettyPrinter(indent=4, compact=True).pformat

variable_content_tokens = [
    'id',
    'fs_string',
    'fd_string',
    's_string',
    'd_string',
    'fsss_string',
    'fddd_string',
    'sss_string',
    'ddd_string',
    'compound_string',
    'pointfloat',
    'expfloat',
    'decimal_int',
    'hexadecimal_int',
    'octal_int',
    'binary_int',
]

irrelevant_content_tokens = ['newline', 'indent', 'dedent', 'EOF']

PhysicalLine = namedtuple('PhysicalLine', 'pos tokens')
IndentLine = namedtuple('IndentLine', 'indent pos content')
LogicalLine = namedtuple('LogicalLine', 'pos content')

def pairs_upto(n):
    return zip(range(n - 1), range(1, n))

def nviews(iterable, n, *, with_nones=False):
    iterables = tee(iterable, n)
    for i in range(len(iterables)):
        for j in range(len(iterables) - i - 1):
            next(iterables[i], None)
    if with_nones:
        return zip_longest(*reversed(iterables), fillvalue=None)
    else:
        return zip(*reversed(iterables))

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

def overloadmethod(*, use_as_default=False, use_as_modifier=False):
    if use_as_modifier and use_as_default:
        raise ValueError('cannot use function both as the default and a modifier')

    def decorator(f):
        registry = {}
        @functools.wraps(f)
        def overloaded(self, x, *args, **kwds):
            for k, v in registry.items():
                if isinstance(x, k):
                    r = v(self, x, *args, **kwds)
                    if use_as_modifier:
                        f(self, x, r)
                    return r
            if use_as_default:
                return f(self, x, *args, **kwds)
            else:
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
    return decorator
def overload(*, use_as_default=False, use_as_modifier=False):
    if use_as_modifier and use_as_default:
        raise ValueError('cannot use function both as the default and a modifier')

    def decorator(f):
        registry = {}
        @functools.wraps(f)
        def overloaded(x, *args, **kwds):
            for k, v in registry.items():
                if isinstance(x, k):
                    r = v(x, *args, **kwds)
                    if use_as_modifier:
                        f(x, r)
                    return r
            if use_as_default:
                return f(x, *args, **kwds)
            else:
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
    return decorator

def format_exception(exc):
    tb = exc.__traceback__
    cls = exc.__class__
    return traceback.format_exception(cls, exc, tb)

class ApeError(Exception):
    def __init__(self, pos, msg):
        self.msg = msg
        self.pos = pos

    def format_with_context(self, input_text, stacktrace=False):
        linenumber = input_text.count('\n', 0, self.pos) + 1
        col = self.pos - input_text.rfind('\n', 0, self.pos)

        try:
            line = input_text.splitlines()[linenumber - 1]
        except IndexError:
            context = ''
        else:
            stripped = line.lstrip()
            wspace = len(line) - len(stripped)
            pointer = (' ' * (col - wspace - 1)) + '^'
            context = f'\n{stripped}\n{pointer}'
        if stacktrace:
            tb = ''.join(format_exception(self)[:-1])
        else: 
            tb = ''

        message = f'{linenumber}:{col}: {self.msg}'
        return '{}{}{}'.format(tb, message, context)

class ApeSyntaxError(ApeError):
    pass
