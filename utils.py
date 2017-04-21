import functools
import json
import pprint
import traceback

from collections import namedtuple
from itertools import tee, zip_longest

class Expression:
    pass

def to_json(x):
    return MyJSONEncoder(indent=None).encode(x)

class MyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Expression):
            return (o.__class__.__name__, vars(o))
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
    'pointfloat',
    'expfloat',
    'decimal_int',
    'hexadecimal_int',
    'octal_int',
    'binary_int'
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

def format_exception(exc):
    tb = exc.__traceback__
    cls = exc.__class__
    return traceback.format_exception(cls, exc, tb)

class ApeError(Exception):
    def __init__(self, line, col, msg):
        self.message = f'{line}:{col}: {msg}'
        self.line = line
        self.col = col

    def format_with_context(self, input_text, stacktrace=False):
        try:
            line = input_text.splitlines()[self.line - 1]
        except IndexError:
            context = ''
        else:
            stripped = line.lstrip()
            wspace = len(line) - len(stripped)
            pointer = (' ' * (self.col - wspace - 1)) + '^'
            context = f'\n{stripped}\n{pointer}'
        if stacktrace:
            tb = ''.join(format_exception(self)[:-1])
        else: 
            tb = ''
        return '{}{}{}'.format(tb, self.message, context)

class ApeSyntaxError(ApeError):
    pass
