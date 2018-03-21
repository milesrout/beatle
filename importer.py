from utils import compose, overloadmethod, ApeError, ApeImportError, ApeInternalError
import astnodes as E
from astpass import DeepAstPass
import pathlib

class ImportHandler:
    def __init__(self, base_search_path, dirty_hack):
        self.expand = ImportExpander(self).visit
        self.base_search_path = base_search_path
        self.loaded_modules = {}
        self.args, self.regexp, self.scan, self.parse = dirty_hack

    def load_module(self, name, pos):
        p = pathlib.Path(self.base_search_path)
        for part in name.parts:
            p /= part.name
        p = p.with_suffix('.b').resolve()

        key = str(p)

        module = self.loaded_modules.get(key, None)
        if module is not None:
            return E.NamespaceReferenceDefinition(name, key, pos)

        with p.open('r') as f:
            input_text = f.read()
            if self.args.verbose >= 1:
                print(f'=== Starting to process {key} ===')
            tokens, _ = self.scan(input_text, self.args, self.regexp)
            module = self.parse(tokens, self.args, input_text, initial_production='file_input')
            if self.args.verbose >= 1:
                print(f'=== Finished processing {key} ===')
            self.loaded_modules[key] = module
            return E.NamespaceDefinition(name, key, module, pos)

    def process(self, ast):
        return self.expand(ast)

class ImportExpander(DeepAstPass):
    def __init__(self, handler):
        self.handler = handler

    def expand_error(self, ast):
        try:
            return ApeInternalError(pos=ast.pos, msg='no overload found for {}'.format(ast.__class__))
        except Exception:
            return ApeInternalError(pos=0, msg='no overload found for {}'.format(ast.__class__))

    @overloadmethod(use_as_default=True, error_function=expand_error)
    def visit(self, ast):
        return super().visit(ast)

    @visit.on(E.ImportStatement)
    @compose(E.Statements)
    @compose(list)
    def _(self, ast):
        for name in ast.names:
            yield self.handler.load_module(name.name, ast.pos)

    @visit.on(E.FromImportStatement)
    def _(self, ast):
        raise
        self.handler.load_module(ast.name, ast.pos)
        return E.NoneExpression()

def process(ast, base_search_path, dirty_hack):
    try:
        ih = ImportHandler(base_search_path, dirty_hack)
        ast = ih.process(ast)
        return ast
    except ApeError as exc:
        raise ApeImportError(msg=exc.msg, pos=exc.pos) from exc
