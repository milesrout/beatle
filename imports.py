from utils import to_json, compose, overloadmethod, ApeError
from astnodes import Statements, ImportStatement, FromImportStatement
import pathlib

class ImportHandler:
    def __init__(self, base_search_path):
        self.base_search_path = base_search_path
        self.loaded_modules = []

    def expand_error(self, ast):
        try:
            return ApeError(pos=ast.pos, msg='no overload found for {}'.format(ast.__class__))
        except Exception:
            return ApeError(pos=0, msg='no overload found for {}'.format(ast.__class__))

    def load_module(self, name):
        print(name)
        p = pathlib.Path(self.base_search_path)
        for part in name:
            p /= part.name
        with p.with_suffix('.b').open('r') as f:
            print(f.read())

    @overloadmethod(error_function=expand_error)
    def expand(self, ast):
        ...

    @expand.on(ImportStatement)
    def _(self, ast):
        for name in ast.names:
            self.load_module(name.name)
            print('N', end=' ')
            for part in name.name:
                print(to_json(part), end=' ')
            print()
        return ast

    @expand.on(FromImportStatement)
    def _(self, ast):
        print('F', to_json(ast))
        self.load_module(ast.name)
        return ast

    @compose(list)
    def do_process(self, ast: Statements):
        for stmt in ast.stmts:
            if isinstance(stmt, Statements):
                yield from self.do_process(stmt)
            if isinstance(stmt, (ImportStatement, FromImportStatement)):
                yield self.expand(stmt)
            else:
                yield stmt

    def process(self, ast):
        return Statements(self.do_process(ast))

def process(ast, base_search_path):
    ih = ImportHandler(base_search_path)
    ast = ih.process(ast)
    print(ih.loaded_modules)
    return ast
