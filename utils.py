import functools
import json
import pprint
import traceback

from itertools import tee, zip_longest

class Type:
    pass

class Expression:
    pass

class Ast:
    def __init__(self, node, type, pos):
        self.node = node
        self.type = type
        self.pos = pos

    def __str__(self):
        return f'({self.node} : {self.type})'

    def __repr__(self):
        return f'Ast({self.node!r}, {self.type!r}, {self.pos!r})'

def to_json(x, indent=None):
    return MyJSONEncoder(indent=indent).encode(x)

class MyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Type):
            return str(o)
        if isinstance(o, Ast):
            return (o.type.vtype, o.node)
        if o.__class__.__module__ == 'typednodes' or isinstance(o, Expression):
            return (o.__class__.__name__, {k:v for k,v in vars(o).items() if k != 'pos'})
        if o.__class__.__name__ in ['Token', 'Types']:
            return repr(o)
        return super().default(o)

# to print an s-expression
#
# e.g.
# (macro-definition
#   (id-expression name)
#   (statements
#     (...)
#     (...)
#     (...)))
#
# 1. Print opening bracket
# 2. Print first item in list directly
# 3. If the list contains any lists or the length condition holds
#    a. Increment the indentation level
#    b. For each item in the list:
#       i.   Print a newline
#       ii.  Print enough spaces to reach the current indentation level
#       iii. Print the *sub-expression*
# 4. Otherwise
#    a. For each item in the list:
#       i.  Print a space
#       ii. Print the *sub-expression*
# 5. Print a closing bracket

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

def nviews(iterable, n, *, with_nones=False):
    iterables = tee(iterable, n)
    for i in range(len(iterables)):
        for j in range(len(iterables) - i - 1):
            next(iterables[i], None)
    if with_nones:
        return zip_longest(*reversed(iterables), fillvalue=None)
    else:
        return zip(*reversed(iterables))

def tap(f):
    def outer(g):
        @functools.wraps(g)
        def inner(*args, **kwds):
            x = g(*args, **kwds)
            f(g.__name__, x)
            return x
        return inner
    return outer

def compose(f):
    def outer(g):
        @functools.wraps(g)
        def inner(*args, **kwds):
            return f(g(*args, **kwds))
        return inner
    return outer

def trace(show_counter=False, show_types=False):
    def outer_wrapper(f):
        counter = 0

        @functools.wraps(f)
        def inner_wrapper(*args):
            nonlocal counter
            counter += 1
            local_counter = counter

            if show_types:
                args_string = ', '.join(
                        '{}: {}'.format(arg, type(arg).__name__
                            ) for arg in args)
            else:
                args_string = ', '.join(map(str, args))

            ret = f(*args)

            if show_counter:
                print(local_counter, end=' ')
            print('{}({}) -> {}'.format(f.__name__, args_string, ret))

            return ret
        return inner_wrapper
    return outer_wrapper

def unzip(iterable):
    return tuple(zip(*iterable))

def overloadmethod(*, use_as_default=False, use_as_modifier=False, use_as_wrapper=False, error_function=None):
    if use_as_modifier + use_as_default + use_as_wrapper > 1:
        raise ValueError('use_as_{default,modifier,wrapper} are mutually exclusive - use .{default,modifier,wrapper} to add additional utility')

    def decorator(f):
        registry = {}
        default = None
        modifiers = []
        wrappers = []

        if use_as_default:
            default = f
        if use_as_modifier:
            modifiers.append(f)
        if use_as_wrapper:
            wrappers.append(f)

        @functools.wraps(f)
        def overloaded(self, x, *args, **kwds):
            def do_overload():
                for k, v in registry.items():
                    if isinstance(x, k):
                        return v(self, x, *args, **kwds)
                if default is not None:
                    return default(self, x, *args, **kwds)
                elif error_function is not None:
                    raise error_function(self, x, *args, **kwds)
                else:
                    raise TypeError('no overload found for {}'.format(x.__class__))

            r = do_overload()
            for modifier in modifiers:
                modifier(self, x, r)
            for wrapper in wrappers:
                r = wrapper(self, x, r)
            return r

        def on(t):
            def register(g):
                if registry.get(t) is None:
                    registry[t] = g
                else:
                    raise ValueError('can\'t overload on the same type twice')
            return register

        def add_modifier():
            def register(g):
                modifiers.append(g)
            return register

        def add_wrapper():
            def register(g):
                wrappers.append(g)
            return register

        def add_default():
            def register(g):
                nonlocal default
                if default is None:
                    default = g
                else:
                    raise ValueError('can\'t set two default functions')
            return register

        overloaded.on = on
        overloaded.default = add_default
        overloaded.wrapper = add_wrapper
        overloaded.modifier = add_modifier
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

def to_sexpr(o, indent=None):
    return fmt_sexpr(json.loads(MyJSONEncoder().encode(o)))

def fmt_sexpr(o, depth=1):
    if isinstance(o, dict):
        return fmt_sexpr(list(o.values()), depth)
    if isinstance(o, tuple):
        return fmt_sexpr(list(o), depth)
    if isinstance(o, list):
        if len(o) == 0:
            return '()'
        if len(o) == 1:
            return fmt_sexpr(o[0], depth)
        return fmt_sexpr_list(o, depth)
    return str(o)

def fmt_sexpr_short(o, depth=1):
    if isinstance(o, dict):
        return fmt_sexpr_short(list(o.values()))
    if isinstance(o, tuple):
        return fmt_sexpr_short(list(o))
    if isinstance(o, list):
        if len(o) == 0:
            return '()'
        if len(o) == 1:
            return fmt_sexpr_short(o[0])
        return fmt_sexpr_list_short(o)
    return str(o)

@compose(''.join)
def fmt_sexpr_list_long(o, depth):
    yield '('
    yield fmt_sexpr(o[0], depth + 1)
    for item in o[1:]:
        yield '\n'
        yield (2 * depth * ' ')
        yield fmt_sexpr(item, depth + 1)
    yield ')'

@compose(''.join)
def fmt_sexpr_list_short(o):
    yield '('
    yield fmt_sexpr_short(o[0])
    for item in o[1:]:
        yield ' '
        yield fmt_sexpr_short(item)
    yield ')'

def fmt_sexpr_list(o, depth):
    result = fmt_sexpr_list_short(o)
    if max(map(len, result.splitlines())) > 80:
        return fmt_sexpr_list_long(o, depth)
    return result

def format_exception(exc):
    tb = exc.__traceback__
    cls = exc.__class__
    return traceback.format_exception(cls, exc, tb)

class ApeError(Exception):
    header = None
    def __init__(self, pos, msg, input_text=None):
        self.msg = msg
        self.input_text = input_text
        try:
            iter(pos)
            self.pos = list(pos)
        except TypeError:
            self.pos = [pos]

    def format_context(self, input_text, pos):
        linenumber = input_text.count('\n', 0, pos) + 1
        col = pos - input_text.rfind('\n', 0, pos)

        try:
            line = input_text.splitlines()[linenumber - 1]
        except IndexError:
            context = ''
        else:
            stripped = line.lstrip()
            wspace = len(line) - len(stripped)
            pointer = (' ' * (col - wspace - 1)) + '^'
            context = f'\n{stripped}\n{pointer}'

        return context

    def format_with_context(self, input_text, stacktrace=False):
        if self.input_text is not None:
            input_text = self.input_text
        linenumber = input_text.count('\n', 0, self.pos[0]) + 1
        col = self.pos[0] - input_text.rfind('\n', 0, self.pos[0])

        context = '\n'.join(self.format_context(input_text, p) for p in self.pos)

        if stacktrace:
            tb = ''.join(format_exception(self)[:-1])
        else:
            tb = ''

        header = self.header + '\n' if self.header is not None else ''
        message = f'{header}{linenumber}:{col}: {self.msg}'
        return '{}{}{}'.format(tb, message, context)

class ApeScanError(ApeError):
    header = 'ERROR SCANNING INTO TOKENS'
class ApeSyntaxError(ApeError):
    header = 'ERROR PARSING INTO AST'
class ApeImportError(ApeError):
    header = 'ERROR IMPORTING MODULES'
class ApeMacroError(ApeError):
    header = 'ERROR EXPANDING MACROS'
class ApeTypeError(ApeError):
    header = 'ERROR CHECKING TYPES'
class ApeCodegenError(ApeError):
    header = 'ERROR GENERATING CODE'
class ApeInternalError(ApeError):
    header = 'ERROR INTERALLY'
class ApeNotImplementedError(ApeInternalError):
    header = 'UNIMPLEMENTED FUNCTIONALITY'
