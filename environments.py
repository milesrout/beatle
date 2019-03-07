import collections
import contextlib
import cstnodes as E

printout_depth = 0

@contextlib.contextmanager
def _depth():
    global printout_depth
    printout_depth += 1
    yield
    printout_depth -= 1

def _printout(*args, **kwds):
    print(printout_depth * ' ', *args, **kwds)

class Namespace:
    def __init__(self, base_env):
        self.base = collections.ChainMap(base_env)
        self.env = self.base.new_child()

    def printout(self):
        for k, v in self.env.items():
            if isinstance(v, Object):
                _printout(f'{k!s:<15} {v.value!s}')
            elif isinstance(v, Namespace):
                _printout(f'{k!s}:')
                with _depth():
                    v.printout()
            else:
                _printout(f'{k!s:<15} {v!s}')

    def __getitem__(self, key):
        if isinstance(key, E.DottedNameExpression):
            raise
        return self.env[key]

    def __setitem__(self, key, value):
        if isinstance(key, E.DottedNameExpression):
            raise
        self.env[key] = value

    def get(self, key, alt=None):
        if isinstance(key, E.DottedNameExpression):
            raise
        return self.env.get(key, alt)

    def update(self, others):
        self.env.update(others)

    @contextlib.contextmanager
    def clean_subenv(self):
        old = self.env
        self.env = self.base.new_child()
        yield
        self.env = old

    @contextlib.contextmanager
    def subenv(self):
        old = self.env
        self.env = self.env.new_child()
        yield
        self.env = old

class Object:
    def __init__(self, value):
        self.value = value

def add_namespace(base_env, name, env):
    for part in name.parts[:-1]:
        base_env = base_env[part.name] = Namespace({})
    base_env[name.parts[-1].name] = Namespace(env)
