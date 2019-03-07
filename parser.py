import contextlib
import collections

import cstnodes as E
from utils import ApeNotImplementedError, ApeSyntaxError, compose

class Parser:
    """A parser for the Beatle programming language"""

    small_expr_tokens = ('del pass assert global nonlocal break continue '
                         'return raise import from').split()
    base_compound_tokens = ('def match'  # ('if while for try with def match '
                            'module signature at').split()
    ops = 'plus minus asterisk matmul div mod and or xor lsh rsh exp'.split()
    augassign_tokens = [f'aug_{op}' for op in ops]
    comparison_op_tokens = 'lt gt eq ge le ne in is'.split()
    bases = 'decimal hexadecimal octal binary'.split()
    int_tokens = [f'{base}_int' for base in bases]
    float_tokens = 'pointfloat expfloat'.split()
    string_tokens = [f'{x}_string' for x in 'fs fd fsss fddd s d sss ddd compound'.split()]

    # for each element of this list, if we are parsing a compound statement and we encounter
    # something that is the *first* element of the list, then we accept follow-on statements
    # as long as they are an element of the list.
    # so the first items of these lists must be unique across the whole list, but remaining items
    # maybe hbe shared (as 'else' is).
    # they also aren't lists, obviously, they're dictionaries, which means that
    # they aren't really ordered. order should be enforced later.

    base_linked_control_structures = {
        'do': {
            'do': [],
        },
        #'class': {
        #    'class': [['EXPR']],
        #},
        'if': {
            'if': [['TEST']],
            'elif': [['TEST']],
            'else': []
        },
        'while': {
            'while': [['TEST']],
            'else': [],
        },
        'for': {
            'for': [['EXPRLIST', 'in', 'TESTLIST']],
            'else': [],
        },
        'try': {
            'try': [],
            'except': [[], ['TEST'], ['as', 'NAME']],
            'else': [],
            'finally': [],
        },
        'with': {
            'with': [['TEST'], ['as', 'EXPR']],
        }
    }

    def acceptable_lcs_successors(self, initial):
        return [k for k in self.linked_control_structures[initial].keys() if k != initial]

    def calculate_compound_stmt_tokens(self):
        self.compound_stmt_tokens = set.union(*(set.union({x for opt in L.values() for y in opt for x in y if x.islower()}, set(L.keys()))
                                                for L in self.linked_control_structures.values()))

    @property
    @compose(list)
    def compound_tokens(self):
        for lcs in self.linked_control_structures.values():
            for token in lcs:
                yield token

    def __init__(self, tokens, input_text):
        self.tokens = list(tokens)
        self.index = 0
        self.virtuals = 0
        self.brackets = 0
        self.linked_control_structures = collections.ChainMap(self.base_linked_control_structures)
        self.calculate_compound_stmt_tokens()
        # Prepopulate these with the built-in tags
        self.unary_tags = ['cons']
        self.nullary_tags = ['nil']

        class Error(ApeSyntaxError):
            def __init__(self, msg, pos):
                super().__init__(msg, pos, input_text=input_text)

        self.Error = Error

    @contextlib.contextmanager
    def show_virtuals(self):
        old = self.virtuals
        self.virtuals = self.brackets
        yield
        self.virtuals = old

    def current_token(self):
        # cache for efficiency - sucks to write code like this but worth it
        # to make parsing big files significantly faster (hundreds of ms)
        V = self.virtuals
        for i in range(self.index, len(self.tokens)):
            v = self.tokens[i].virtual
            if v is None or v <= V:
                tok = self.tokens[i]
                if tok.type == 'id' and tok.string in self.compound_stmt_tokens:
                    return tok.__class__(type=tok.string, string=tok.string, pos=tok.pos, virtual=tok.virtual)
                return tok

    def consume_token(self):
        token = self.tokens[self.index]
        if token.type in ['lparen', 'lbrace', 'lbrack']:
            self.brackets += 1
        elif token.type in ['rparen', 'rbrace', 'rbrack']:
            self.brackets -= 1
        self.index += 1

    def next_token(self):
        while self.tokens[self.index].virtual > self.virtuals:
            self.consume_token()
        self.consume_token()

    def get_token(self):
        tok = self.current_token()
        self.next_token()
        return tok

    def expect_get(self, *expected):
        actual = self.current_token()
        if actual.type not in expected:
            friendly = '|'.join(expected)
            raise self.Error(
                actual.pos,
                f"{self.virtuals} - expected '{friendly}', got {actual}")
        self.next_token()
        return actual

    def expect_get_many(self, *expected):
        result = [self.get_token()]
        if result[0].type not in expected:
            friendly = '|'.join(expected)
            raise self.Error(
                result[0].pos,
                f"{self.virtuals} - expected '{friendly}', got {result[0]}")
        while self.accept(*expected):
            result.append(self.current_token())
            self.next_token()
        return result

    def expect(self, expected):
        actual = self.current_token()
        if actual.type != expected:
            lhs = self.tokens[self.index - 5:self.index]
            rhs = self.tokens[self.index + 1:self.index + 6]
            raise self.Error(
                actual.pos,
                f"{self.virtuals} - expected '{expected}', got {actual} "
                f"...context: {lhs} >>>> {actual} <<<< {rhs}")
        else:
            self.next_token()

    def accept(self, *acceptable):
        actual = self.current_token().type
        return actual in acceptable

    def accept_next(self, acceptable):
        actual = self.current_token().type
        if actual == acceptable:
            self.next_token()
        return actual == acceptable

    def raise_unexpected(self):
        tok = self.current_token()
        lhs = self.tokens[self.index - 5:self.index]
        rhs = self.tokens[self.index + 1:self.index + 6]
        raise self.Error(
            tok.pos,
            f'{self.virtuals} - unexpected token: {tok} '
            f'...context: {lhs} >>>> {tok} <<<< {rhs}')

    @compose(E.Statements)
    @compose(list)
    def file_input(self):
        while not self.accept_next('EOF'):
            if self.accept_next('newline'):
                continue
            yield self.stmt()

    def single_input(self):
        stmt = self.toplevel_stmt()
        if stmt is not None:
            self.expect('newline')
            return stmt
        elif self.accept_next('newline'):
            return
        else:
            return self.simple_stmt()

    def inner_stmt(self):
        stmt = self.compound_stmt()
        if stmt is not None:
            return stmt
        return self.simple_stmt()

    def stmt(self):
        stmt = self.toplevel_stmt()
        if stmt is not None:
            return stmt
        return self.simple_stmt()

    @compose(E.Statements)
    @compose(list)
    def simple_stmt(self):
        yield self.small_stmt()
        while self.accept_next('semicolon'):
            if self.accept_next('newline'):
                return
            yield self.small_stmt()
        self.expect('newline')

    @compose(list)
    def testlist(self):
        yield self.test()
        while self.accept_next('comma'):
            if self.accept('semicolon', 'newline', 'equal'):
                return
            yield self.test()

    @compose(list)
    def exprlist(self, *terminators):
        if self.accept_next('asterisk'):
            yield self.star_expr()
        else:
            yield self.expr()
        while self.accept_next('comma'):
            if self.accept(terminators):
                return
            if self.accept_next('asterisk'):
                yield self.star_expr()
            else:
                yield self.expr()

    def small_expr(self):
        with self.show_virtuals():
            return self.small_stmt()

    def small_stmt(self):
        pos = self.current_token().pos
        if self.accept_next('del'):
            return self.del_stmt(pos)
        if self.accept_next('pass'):
            return self.pass_stmt(pos)
        if self.accept_next('assert'):
            return self.assert_stmt(pos)
        if self.accept_next('global'):
            return self.global_stmt(pos)
        if self.accept_next('type'):
            return self.type_defn(pos)
        if self.accept_next('nonlocal'):
            return self.nonlocal_stmt(pos)
        if self.accept_next('break'):
            return self.break_stmt(pos)
        if self.accept_next('continue'):
            return self.continue_stmt(pos)
        if self.accept_next('return'):
            return self.return_stmt(pos)
        if self.accept_next('raise'):
            return self.raise_expr(pos)
        if self.accept_next('yield'):
            return self.yield_expr(pos, 'semicolon', 'newline')
        if self.accept('import', 'from'):
            return self.import_stmt(pos)
        return self.expr_stmt()

    def del_stmt(self, pos):
        return E.DelStatement(self.exprlist('semicolon', 'newline'), pos)

    def pass_stmt(self, pos):
        return E.PassStatement(pos)

    def assert_stmt(self, pos):
        exprs = [self.test()]
        while self.accept_next('comma'):
            exprs.append(self.test())
        return E.AssertStatement(exprs, pos)

    def global_stmt(self, pos):
        names = [self.name()]
        while self.accept_next('comma'):
            names.append(self.name())
        return E.GlobalStatement(names, pos)

    def nonlocal_stmt(self, pos):
        names = [self.name()]
        while self.accept_next('comma'):
            names.append(self.name())
        return E.NonlocalStatement(names, pos)

    def break_stmt(self, pos):
        return E.BreakStatement(pos)

    def continue_stmt(self, pos):
        return E.ContinueStatement(pos)

    def return_stmt(self, pos):
        if self.accept('semicolon', 'newline'):
            return E.ReturnStatement(None, pos)
        exprs = self.testlist()
        return E.ReturnStatement(E.TupleLiteral(exprs, exprs[0].pos), pos)
        # return ReturnStatement(self.testlist('newline'))

    def raise_expr(self, pos):
        if self.accept('semicolon', 'newline'):
            return E.RaiseStatement(exc=None, original=None, pos=pos)
        exc = self.test()
        if self.accept_next('from'):
            return E.RaiseStatement(exc, original=self.test(), pos=pos)
        return E.RaiseStatement(exc, original=None, pos=pos)

    def yield_expr(self, pos, *terminators):
        if self.accept_next('from'):
            return E.YieldFromExpression(expr=self.test(), pos=pos)
        if self.accept(*terminators):
            return E.YieldExpression(expr=E.TupleLiteral([], pos=pos), pos=pos)
        exprs = self.testlist()
        return E.YieldExpression(expr=E.TupleLiteral(exprs, exprs[0].pos), pos=pos)

    def import_stmt(self, pos):
        if self.accept_next('from'):
            return self.import_from(pos)
        elif self.accept_next('import'):
            return self.import_name(pos)
        raise RuntimeError(f'import_stmt called in the wrong context')

    def import_name(self, pos):
        return E.ImportStatement(names=self.dotted_as_names(), pos=pos)

    def import_from(self, pos):
        dots = self.import_dots()
        name = None
        if self.accept('id'):
            name = self.dotted_name()
        self.expect('import')
        if self.accept_next('asterisk'):
            return E.FromImportStatement(name=name, dots=dots, what=[], pos=pos)
        if self.accept_next('lparen'):
            what = self.import_as_names()
            self.expect('rparen')
            return E.FromImportStatement(name=name, dots=dots, what=what, pos=pos)
        return E.FromImportStatement(name=name, dots=dots, what=self.import_as_names(), pos=pos)

    @compose(sum)
    def import_dots(self):
        while self.accept('dot', 'ellipsis'):
            if self.accept_next('ellipsis'):
                yield 3
            if self.accept_next('dot'):
                yield 1

    def import_as_name(self):
        name = self.dotted_name()
        if self.accept_next('as'):
            return E.ImportName(name, alias=self.name())
        return E.ImportName(name, alias=None)

    def dotted_as_name(self):
        name = self.dotted_name()
        if self.accept_next('as'):
            return E.ImportName(name, alias=self.name())
        return E.ImportName(name, alias=None)

    @compose(list)
    def import_as_names(self):
        yield self.import_as_name()
        while self.accept_next('comma'):
            if not self.accept('id'):
                return
            yield self.import_as_name()

    @compose(list)
    def dotted_as_names(self):
        yield self.dotted_as_name()
        while self.accept_next('comma'):
            yield self.dotted_as_name()

    @compose(E.DottedNameExpression)
    @compose(list)
    def dotted_name(self):
        yield self.name()
        while self.accept_next('dot'):
            yield self.name()

    # a -> b? -> c
    # a -> b -> c
    # a -> (b -> c)
    # a[b] -> c -> d
    # (a[b]) -> (c -> d)
    # (a -> b) -> [a] -> [b]
    # ((a -> b) -> (([a]) -> ([b])))

    def toplevel_type_expr(self):
        pos = self.current_token().pos
        if self.accept_next('forall'):
            return self.forall_type_expr(pos)
        return E.TypeForallExpression([], self.type_expr(), pos=pos)

    def type_expr(self, *terminators):
        pos = self.current_token().pos
        exprs = [self.type_tagged_expr('bit_or', *terminators)]
        while self.accept_next('bit_or'):
            exprs.append(self.type_tagged_expr('bit_or', *terminators))
        if len(exprs) == 1:
            return exprs[0]
        return E.TypeDisjunctionExpression(exprs, pos)

    def type_tagged_expr(self, *terminators):
        pos = self.current_token().pos
        if self.accept_next('exclamation'):
            tag = self.name().name
            if tag in self.nullary_tags:
                return E.TypeTaggedExpression(tag, None, pos)
            if tag in self.unary_tags:
                return E.TypeTaggedExpression(tag, self.type_func_expr(), pos)
            if not self.accept(*terminators):
                self.unary_tags.append(tag)
                t = self.type_func_expr()
                return E.TypeTaggedExpression(tag, t, pos)
            else:
                self.nullary_tags.append(tag)
                return E.TypeTaggedExpression(tag, None, pos)
        return self.type_func_expr()

    def type_func_expr(self):
        expr = self.type_atom_expr()
        if self.accept_next('arrow'):
            return E.TypeFunctionExpression(expr, self.type_func_expr())
        return expr

    def type_atom_expr(self):
        expr = self.type()
        if self.accept_next('question'):
            return E.TypeMaybeExpression(expr, expr.pos)
        if self.accept_next('lbrack'):
            return E.TypeCallExpression(expr, self.type_call_args())
        return expr

    def type(self):
        pos = self.current_token().pos
        if self.accept_next('lparen'):
            exprs = [self.type_expr('comma', 'rparen')]
            if not self.accept('comma'):
                self.expect('rparen')
                return exprs[0]
            while self.accept_next('comma'):
                if self.accept_next('rparen'):
                    return E.TypeTupleExpression(exprs, pos=pos)
                exprs.append(self.type_expr())
            self.expect('rparen')
            return E.TypeTupleExpression(exprs, pos=pos)
        if self.accept_next('lbrack'):
            return self.type_list()
        return E.TypeNameExpression(self.dotted_name())

    def forall_type_expr(self, pos):
        tvars = [self.name()]
        while self.accept_next('comma'):
            tvars.append(self.name())
        self.expect('dot')
        return E.TypeForallExpression(tvars, self.type_expr(), pos)

    def new_type_expr(self):
        name = self.name()
        if self.accept_next('lbrack'):
            return (name, self.new_type_call_args())
        return (name, [])

    def type_defn(self, pos):
        name, args = self.new_type_expr()
        self.expect('equal')
        expr = self.type_expr()
        return E.TypeDefinition(name, args, expr, pos)

    def type_decl(self, pos):
        name, args = self.new_type_expr()
        return E.TypeDeclaration(name, args, pos)

    @compose(list)
    def new_type_call_args(self):
        yield self.name()
        while self.accept_next('comma'):
            if self.accept_next('rbrack'):
                return
            yield self.name()
        self.expect('rbrack')

    @compose(list)
    def type_call_args(self):
        yield self.type_expr()
        while self.accept_next('comma'):
            if self.accept_next('rbrack'):
                return
            yield self.type_expr()
        self.expect('rbrack')

    def signature_decl(self):
        pos = self.current_token().pos
        if self.accept_next('type'):
            return self.type_decl(pos)
        if self.accept_next('law'):
            return self.law_decl(pos)
        return self.name_declaration()

    def law_decl(self, pos):
        if self.accept_next('forall'):
            names = [self.name()]
            while self.accept_next('comma'):
                names.append(self.name())
            self.expect('dot')
            return E.LawDeclaration(names, self.law_expr(), pos)
        return E.LawDeclaration([], self.law_expr(), pos)

    def law_expr(self):
        return self.test()

    def name_declaration(self):
        name = self.name()
        self.expect('colon')
        annotation = self.toplevel_type_expr()
        return E.NameDeclaration(name, annotation, pos=name.pos)

    def expr_stmt(self):
        tlse = self.testlist_star_expr()
        colon_pos = self.current_token().pos
        if self.accept_next('colon'):
            annotation = self.type_expr()
            equal_pos = self.current_token().pos
            if self.accept_next('equal'):
                return E.AnnotatedAssignment(
                    assignee=tlse,
                    expr=self.test(),
                    annotation=annotation,
                    pos=equal_pos)
            return E.AnnotatedExpression(tlse, annotation, pos=colon_pos)
        if self.accept(*self.augassign_tokens):
            augtype = self.get_token().type
            pos = self.current_token().pos
            if self.accept_next('yield'):
                expr = self.yield_expr(pos, 'semicolon', 'newline')
            else:
                expr = self.testlist()
                expr = E.TupleLiteral(expr, pos=expr[0].pos)
            return E.AugmentedAssignment(tlse, augtype, expr, pos)
        exprs = [tlse]
        while self.accept_next('equal'):
            pos = self.current_token().pos
            if self.accept_next('yield'):
                exprs.append(self.yield_expr(pos, 'equal', 'semicolon', 'newline'))
            else:
                exprs.append(self.testlist_star_expr())
        if len(exprs) == 1:
            return tlse
        else:
            return E.ChainedAssignment(exprs)

    def testlist_star_expr(self):
        exprs = []
        if self.accept_next('asterisk'):
            exprs.append(E.StarExpr(self.expr()))
        else:
            exprs.append(self.test())
        while self.accept_next('comma'):
            if self.accept('equal', 'colon', *self.augassign_tokens):
                return exprs
            if self.accept_next('asterisk'):
                exprs.append(E.StarExpr(self.expr()))
            else:
                exprs.append(self.test())
        if len(exprs) != 1:
            return E.TupleLiteral(exprs, pos=exprs[0].pos)
        return exprs[0]

    def compound_expr(self):
        with self.show_virtuals():
            return self.compound_stmt()

    def toplevel_stmt(self):
        pos = self.current_token().pos
        if self.accept_next('macro'):
            return self.macro_def(pos)
        if self.accept_next('control_structure'):
            self.expect('lparen')
            return self.control_structure(pos)
        return self.compound_stmt()

    def control_structure(self, pos):
        arm_pos = self.current_token().pos
        arms = [self.control_structure_arm(arm_pos)]
        if arms[0][0] in self.linked_control_structures:
            raise ApeSyntaxError(pos=arm_pos, msg='There is already a control '
                                                  'structure initiated by '
                                                  f"'{arms[0][0]}'")
        while self.accept_next('lparen'):
            arms.append(self.control_structure_arm(arm_pos))
        self.linked_control_structures[arms[0][0]] = dict(arms)
        self.expect('newline')
        self.calculate_compound_stmt_tokens()
        return E.NoneExpression(pos)

    def control_structure_arm(self, pos):
        name = self.expect_get('id', *self.compound_stmt_tokens).string
        arm = [[]]
        while not self.accept_next('rparen'):
            if self.accept('id'):
                arm[-1].append(self.get_token().string)
            elif self.accept(*self.compound_stmt_tokens):
                arm[-1].append(self.get_token().string)
            elif self.accept_next('bit_and'):
                arm.append([])
            elif self.accept_next('lt'):
                arm[-1].append(self.expect_get('id').string.upper())
                self.expect('gt')
            else:
                self.raise_unexpected()
        return name, arm
        raise ApeNotImplementedError(pos, 'control_structure directive')

    def compound_stmt(self):
        if self.accept(*self.compound_stmt_tokens):
            return self.linked_control_structure_stmt(self.current_token().string)
        return self.other_compound_stmt()

    @compose(E.ControlStructureExpression)
    @compose(list)
    def linked_control_structure_stmt(self, initial):
        linking_structure = self.linked_control_structures[self.current_token().string]

        while True:
            pos = self.current_token().pos
            name = self.get_token().string
            options = linking_structure[name]
            args = []

            for option in options:
                for i, param in enumerate(option):
                    if param == 'TEST':
                        args.append(self.test())
                    elif param == 'NAME':
                        args.append(self.name())
                    elif param == 'EXPR':
                        args.append(self.expr())
                    elif param == 'EXPRLIST':
                        if i + 1 < len(option):
                            terminators = (option[i + 1],)
                        else:
                            terminators = ('colon', 'exclamation')
                        args.append(E.Expressions(self.exprlist(*terminators)))
                    elif param == 'TESTLIST':
                        args.append(E.Expressions(self.testlist()))
                    else:
                        args.append([E.SymbolExpression(self.expect_get(param))])
                if self.accept('colon', 'exclamation'):
                    break
            yield E.ControlStructureLinkExpression(name, args, self.suite(), pos)
            if not self.accept(*self.acceptable_lcs_successors(initial)):
                break

    def other_compound_stmt(self):
        pos = self.current_token().pos
        if self.accept_next('def'):
            return self.func_def(pos)
        elif self.accept_next('signature'):
            return self.signature_def(pos)
        elif self.accept_next('module'):
            return self.module_def(pos)
        elif self.accept_next('at'):
            return self.decorated(pos)
        elif self.accept_next('match'):
            return self.match_stmt(pos)
        return None
        #return self.expr_stmt()
        ##self.raise_unexpected()

    def match_case(self, pos):
        tlse = self.testlist_star_expr()
        suite = self.suite()
        return E.MatchCase(tlse, suite, pos)

    @compose(list)
    def match_cases(self):
        if self.accept_next('exclamation'):
            return []
        self.expect('colon')
        self.expect('newline')
        self.expect('indent')
        pos = self.current_token().pos
        while self.accept_next('case'):
            yield self.match_case(pos)
        self.expect('dedent')

    def match_stmt(self, pos):
        expr = self.test()
        cases = self.match_cases()
        return E.MatchStatement(expr, cases, pos)

    def signature_suite(self):
        if self.accept_next('exclamation'):
            return []
        self.expect('colon')
        if self.accept_next('newline'):
            self.expect('indent')
            exprs = []
            while not self.accept_next('dedent'):
                if self.accept_next('newline'):
                    continue
                exprs.append(self.signature_decl())
            return exprs
        return self.signature_decl()

    def signature_def(self, pos):
        name = self.name()
        body = self.signature_suite()
        return E.SignatureDefinition(name, body, pos)

    def module_def(self, pos):
        name = self.name()
        params = self.parameters()
        if self.accept_next('arrow'):
            typ = self.type_expr()
        else:
            typ = None
        body = self.suite()
        return E.ModuleDefinition(name, params, typ, body, pos)

    def decorator(self, pos):
        name = self.dotted_name()
        if self.accept_next('lparen'):
            if self.accept_next('rparen'):
                args = E.Expressions([])
            else:
                args = E.Expressions(self.arglist())
        else:
            args = None
        self.expect('newline')
        return E.Decorator(name, args, pos)

    def decorated(self, pos):
        decorators = self.decorators(pos)
        pos0 = self.current_token().pos
        if self.accept_next('module'):
            defn = self.module_def(pos0)
        elif self.accept_next('def'):
            defn = self.func_def(pos0)
        else:
            self.raise_unexpected()
        return E.Decorated(decorators=decorators, defn=defn, pos=pos)

    @compose(list)
    def decorators(self, pos):
        yield self.decorator(pos)
        while self.accept('at'):
            pos = self.get_token().pos
            yield self.decorator(pos)

    def macro_def(self, pos):
        name = self.name()
        params = self.parameters()
        suite = self.suite()
        return E.MacroDefinition(name, params, suite, pos)

    def func_def(self, pos):
        name = self.name()
        params = self.parameters()
        if self.accept_next('arrow'):
            return_annotation = self.type_expr()
        else:
            return_annotation = None
        suite = self.suite()
        return E.FunctionDefinition(name, params, suite, return_annotation, pos)

    def parameters(self):
        self.expect('lparen')
        if self.accept_next('rparen'):
            return []
        params = self.typedparamslist()
        self.expect('rparen')
        return params

    def vkwparam(self, pos):
        param = E.StarStarKwParams(*self.vfpdef(), pos)
        self.accept_next('comma')
        return param

    def vvarparams(self, pos):
        if self.accept('id'):
            params = [E.StarVarParams(*self.vfpdef(), pos)]
        else:
            params = [E.EndOfPosParams(pos)]
        while self.accept_next('comma'):
            pos = self.current_token().pos
            if self.accept_next('double_asterisk'):
                return params + [self.vkwparam(pos)]
            if self.accept('id'):
                params.append(self.dvfpdef())
            else:
                return params
        return params

    def tkwparam(self, pos):
        param = E.StarStarKwParams(*self.tfpdef(), pos)
        self.accept_next('comma')
        return param

    def tvarparams(self, pos):
        if self.accept('id'):
            params = [E.StarVarParams(*self.tfpdef(), pos)]
        else:
            params = [E.EndOfPosParams(pos)]
        while self.accept_next('comma'):
            pos = self.current_token().pos
            if self.accept_next('double_asterisk'):
                return params + [self.tkwparam(pos)]
            if self.accept('id'):
                params.append(self.dtfpdef())
            else:
                return params
        return params

    def dtfpdef(self):
        name, annotation = self.tfpdef()
        if self.accept_next('equal'):
            return E.Param(name, annotation, self.test())
        else:
            return E.Param(name, annotation, default=None)

    def dvfpdef(self):
        name = self.vfpdef()
        if self.accept_next('equal'):
            return E.Param(name, annotation=None, default=self.test())
        else:
            return E.Param(name, annotation=None, default=None)

    def varparamslist(self):
        pos = self.current_token().pos
        if self.accept_next('double_asterisk'):
            return self.vkwparam(pos)
        elif self.accept_next('asterisk'):
            return self.vvarparams(pos)
        params = [self.dvfpdef()]
        while self.accept_next('comma'):
            if self.accept_next('div'):
                params.append(E.EndOfPosOnlyParams(pos))
                continue
            if self.accept('id'):
                params.append(self.dvfpdef())
                continue
            pos = self.current_token().pos
            if self.accept_next('asterisk'):
                return params + self.vvarparams(pos)
            if self.accept_next('double_asterisk'):
                return params + [self.vkwparam(pos)]
            if not self.accept('rparen'):
                self.raise_unexpected()
            break
        return params

    def vfpdef(self):
        return self.name()

    def typedparamslist(self):
        pos = self.current_token().pos
        if self.accept_next('double_asterisk'):
            return self.tkwparam(pos)
        elif self.accept_next('asterisk'):
            return self.tvarparams(pos)
        params = [self.dtfpdef()]
        while self.accept_next('comma'):
            if self.accept_next('div'):
                params.append(E.EndOfPosOnlyParams(pos))
                continue
            if self.accept('id'):
                params.append(self.dtfpdef())
                continue
            pos = self.current_token().pos
            if self.accept_next('asterisk'):
                return params + self.tvarparams(pos)
            if self.accept_next('double_asterisk'):
                return params + [self.tkwparam(pos)]
            if not self.accept('rparen'):
                self.raise_unexpected()
            break
        return params

    def tfpdef(self):
        name = self.name()
        if self.accept_next('colon'):
            annotation = self.type_expr()
        else:
            annotation = None
        return (name, annotation)

    def suite(self):
        if self.accept_next('exclamation'):
            return E.Statements([])
        self.expect('colon')
        if self.accept_next('newline'):
            self.expect('indent')
            stmts = []
            while not self.accept_next('dedent'):
                if self.accept_next('newline'):
                    continue
                stmts.append(self.inner_stmt())
            return E.Statements(stmts)
        return self.simple_stmt()

    def lambda_expr(self):
        pos = self.current_token().pos
        if self.accept_next('def'):
            return self.def_expr(pos)
        self.expect('lambda')
        return self.lambdef(pos)

    def def_expr(self, pos):
        with self.show_virtuals():
            return self._def_expr(pos)

    def _def_expr(self, pos):
        params = self.parameters()
        if self.accept_next('arrow'):
            return_annotation = self.type_expr()
        else:
            return_annotation = None
        suite = self.suite()
        return E.FunctionExpression(params, suite, return_annotation, pos)

    def lambdef(self, pos):
        if self.accept_next('colon'):
            return E.LambdaExpression(params=[], body=self.test(), pos=pos)
        params = self.varparamslist()
        self.expect('colon')
        return E.LambdaExpression(params=params, body=self.test(), pos=pos)

    def lambda_nocond(self, pos):
        if self.accept_next('colon'):
            return E.LambdaExpression(params=[], body=self.test_nocond(), pos=pos)
        params = self.varparamslist()
        self.expect('colon')
        return E.LambdaExpression(params=params, body=self.test_nocond(), pos=pos)

    def test_nocond(self):
        pos = self.current_token().pos
        if self.accept('lambda'):
            return self.lambda_nocond(pos)
        return self.or_test()

    def test(self):
        if self.accept('lambda', 'def'):
            return self.lambda_expr()
        if self.accept(*self.compound_tokens):
            return self.compound_expr()
        if self.accept(*self.small_expr_tokens):
            return self.small_expr()
        expr = self.or_test()
        if self.accept_next('if'):
            cond = self.or_test()
            self.expect('else')
            alt = self.test()
            return E.IfElseExpr(expr, cond, alt)
        return expr

    def or_test(self):
        exprs = [self.and_test()]
        while self.accept_next('or'):
            exprs.append(self.and_test())
        if len(exprs) == 1:
            return exprs[0]
        return E.LogicalOrExpressions(exprs)

    def and_test(self):
        exprs = [self.not_test()]
        while self.accept_next('and'):
            exprs.append(self.not_test())
        if len(exprs) == 1:
            return exprs[0]
        return E.LogicalAndExpressions(exprs)

    def not_test(self):
        if self.accept_next('not'):
            return E.LogicalNotExpression(self.not_test())
        return self.comparison()

    def comparison(self):
        exprs = [self.expr()]
        while self.accept('is', 'not', *self.comparison_op_tokens):
            if self.accept_next('is'):
                if self.accept_next('not'):
                    exprs.append('is not')
                else:
                    exprs.append('is')
            elif self.accept_next('not'):
                self.expect('in')
                exprs.append('not in')
            elif self.accept(*self.comparison_op_tokens):
                exprs.append(self.get_token().type)
            exprs.append(self.expr())
        if len(exprs) == 1:
            return exprs[0]
        return E.ComparisonChain(exprs, pos=exprs[0].pos)

    def expr(self):
        exprs = [self.bitxor_expr()]
        while self.accept_next('bit_or'):
            exprs.append(self.xor_expr())
        if len(exprs) == 1:
            return exprs[0]
        return E.BitOrExpression(exprs)

    def bitxor_expr(self):
        exprs = [self.bitand_expr()]
        while self.accept_next('bit_xor'):
            exprs.append(self.bitand_expr())
        if len(exprs) == 1:
            return exprs[0]
        return E.BitXorExpression(exprs)

    def bitand_expr(self):
        exprs = [self.bitshift_expr()]
        while self.accept_next('bit_and'):
            exprs.append(self.bitshift_expr())
        if len(exprs) == 1:
            return exprs[0]
        return E.BitAndExpression(exprs)

    def bitshift_expr(self):
        expr = self.arith_expr()
        if self.accept('bit_lsh', 'bit_rsh', 'bit_asr'):
            shift_type = self.get_token().type
            return E.BitShiftExpression(shift_type, expr, self.bitshift_expr())
        return expr

    def arith_expr(self):
        term = self.term()
        if self.accept('plus', 'minus'):
            return E.ArithExpression(self.get_token().type, term, self.arith_expr())
        return term

    def term(self):
        factor = self.factor()
        if self.accept('asterisk', 'at', 'div', 'mod', 'truediv'):
            return E.ArithExpression(self.get_token().type, factor, self.term())
        return factor

    def factor(self):
        if self.accept('plus', 'minus'):
            op = self.get_token()
            return E.UnaryExpression(op.type, self.factor(), op.pos)
        if self.accept('tilde'):
            pos = self.get_token().pos
            return E.Lazy(self.factor(), pos=pos)
        return self.power()

    def power(self):
        atom_expr = self.atom_expr()
        if self.accept_next('double_asterisk'):
            return E.PowerExpression(atom_expr, self.factor())
        return atom_expr

    def atom_expr(self):
        atom = self.quasiatom()
        trailers = []
        while self.accept('lparen', 'lbrack', 'dot'):
            trailers.append(self.trailer())
        if len(trailers) != 0:
            atom = E.AtomExpression(atom, trailers)
        return atom

    def quasiatom(self):
        pos = self.current_token().pos
        if self.accept_next('exclamation'):
            tag = self.name().name
            if tag in self.nullary_tags:
                return E.TaggedExpression(tag, None, pos=pos)
            else:
                if tag not in self.unary_tags:
                    raise ApeSyntaxError(msg=f'Unexpected and undefined tag in tag expression: {tag}', pos=pos)
                return E.TaggedExpression(tag, self.quasiatom(), pos=pos)
        if self.accept_next('backslash'):
            return E.Quasiquote(self.quasiatom(), pos=pos)
        if self.accept_next('unquote'):
            return E.Unquote(self.quasiatom(), pos=pos)
        if self.accept_next('unquote_splicing'):
            return E.UnquoteSplice(self.quasiatom(), pos=pos)
        return self.atom()

    def trailer(self):
        pos = self.current_token().pos
        if self.accept_next('lparen'):
            return E.CallTrailer(self.call_trailer(), pos=pos)
        if self.accept_next('lbrack'):
            return E.IndexTrailer(self.index_trailer(), pos=pos)
        if self.accept_next('dot'):
            return E.AttrTrailer(self.attr_trailer(), pos=pos)
        self.raise_unexpected()

    def call_trailer(self):
        if self.accept_next('rparen'):
            return []
        return self.arglist()

    def arglist(self):
        args = [self.argument()]
        while self.accept_next('comma'):
            if self.accept_next('rparen'):
                return args
            args.append(self.argument())
        self.expect('rparen')
        return args

##############################################################################
#                XXX: DOES THIS APPLY TO THIS PARSER?                        #
#                                                                            #
# The reason that keywords are test nodes instead of NAME is that using NAME #
# results in an ambiguity. ast.c makes sure it's a NAME.                     #
# "test '=' test" is really "keyword '=' test", but we have no such token.   #
# These need to be in a single rule to avoid grammar that is ambiguous       #
# to our LL(1) parser. Even though 'test' includes '*expr' in star_expr,     #
# we explicitly match '*' here, too, to give it proper precedence.           #
# Illegal combinations and orderings are blocked in ast.c:                   #
# multiple (test comp_for) arguments are blocked; keyword unpackings         #
# that precede iterable unpackings are blocked; etc.                         #
##############################################################################
    def argument(self):
        if self.accept_next('asterisk'):
            return E.StarArg(self.test())
        if self.accept_next('double_asterisk'):
            return E.StarStarKwarg(self.test())
        keyword = self.test()
        if self.accept_next('equal'):
            return E.KeywordArg(keyword, self.test())
        if self.accept('for'):
            return E.CompForArg(keyword, self.comp_for())
        return E.PlainArg(keyword)

    @compose(list)
    def index_trailer(self):
        while not self.accept_next('rbrack'):
            yield self.subscript()

    def subscript(self):
        "subscript: test | [test] ':' [test] [':' [test]]"
        idx_or_start = None
        if not self.accept('colon'):
            idx_or_start = self.test()
            if not self.accept('colon'):
                return E.Index(idx=idx_or_start)
        self.expect('colon')
        end, step = self.rest_of_slice()
        return E.Slice(start=idx_or_start, end=end, step=step)

    def rest_of_slice(self):
        end, step = None, None
        if not self.accept('colon', 'rbrack', 'comma'):
            end = self.test()
        if self.accept_next('colon'):
            step = self.sliceop()
        return end, step

    def sliceop(self):
        if not self.accept('rbrack', 'comma'):
            return self.test()

    def attr_trailer(self):
        return self.name()

    def star_expr(self):
        self.expect('asterisk')
        return E.StarExpr(self.expr())

    def star_expr_or_test(self):
        if self.accept('asterisk'):
            pos = self.get_token().pos
            return E.StarExpr(self.expr(), pos)
        else:
            return self.test()

    def starstar_expr_or_pair(self):
        if self.accept('double_asterisk'):
            pos = self.get_token().pos
            return E.StarStarExpr(self.expr(), pos)
        else:
            k = self.test()
            self.expect('colon')
            v = self.test()
            return E.DictPair(k, v)

    def testlist_comp(self, pos, *terminators):
        exprs = [self.star_expr_or_test()]
        if self.accept('for'):
            return E.Comprehension(exprs[0], self.comp_for(), pos=pos)
        while self.accept_next('comma'):
            if self.accept(*terminators):
                return E.Literal(exprs, trailing_comma=True, pos=pos)
            exprs.append(self.star_expr_or_test())
        return E.Literal(exprs, trailing_comma=False, pos=pos)

    def rest_of_dictmaker(self, pos, first):
        exprs = [first]
        if self.accept('for'):
            return E.DictComprehension(first, self.comp_for(), pos=pos)
        while self.accept_next('comma'):
            if self.accept('rbrace'):
                break
            exprs.append(self.starstar_expr_or_pair())
        return E.DictLiteral(exprs, pos)

    def rest_of_setmaker(self, pos, first):
        exprs = [first]
        if self.accept('for'):
            return E.SetComprehension(first, self.comp_for(), pos=pos)
        while self.accept_next('comma'):
            if self.accept('rbrace'):
                break
            exprs.append(self.star_expr_or_test())
        return E.SetLiteral(exprs, pos=pos)

    def dictorsetmaker(self, pos):
        pos1 = self.current_token().pos
        if self.accept_next('double_asterisk'):
            return self.rest_of_dictmaker(pos, E.StarStarExpr(self.expr(), pos=pos1))
        if self.accept_next('asterisk'):
            return self.rest_of_setmaker(pos, E.StarArg(self.expr(), pos=pos1))
        expr = self.test()
        if self.accept_next('colon'):
            return self.rest_of_dictmaker(pos, E.DictPair(expr, self.test()))
        return self.rest_of_setmaker(pos, expr)

    def gen_expr_or_tuple_literal(self, pos, expr):
        if isinstance(expr, E.Literal):
            if len(expr.exprs) > 1 or expr.trailing_comma:
                return E.TupleLiteral(expr.exprs, pos=pos)
            return expr.exprs[0]
        else:
            return E.GeneratorExpression(expr.expr, expr.rest, pos=pos)

    def list_comp_or_list_literal(self, pos, expr):
        if isinstance(expr, E.Literal):
            return E.ListLiteral(expr.exprs, pos)
        else:
            return E.ListComprehension(expr.expr, expr.rest, pos)

    def atom(self):
        if self.accept('lparen'):
            pos = self.get_token().pos
            if self.accept_next('rparen'):
                return E.EmptyTupleExpression(pos)
            if self.accept_next('yield'):
                expr = self.yield_expr(pos, 'rparen')
            else:
                expr = self.gen_expr_or_tuple_literal(pos, self.testlist_comp(pos, 'rparen'))
            self.expect('rparen')
            return expr
        if self.accept('lbrack'):
            pos = self.get_token().pos
            if self.accept_next('rbrack'):
                return E.EmptyListExpression(pos)
            expr = self.testlist_comp(pos, 'rbrack')
            self.expect('rbrack')
            return self.list_comp_or_list_literal(pos, expr)
        if self.accept('lbrace'):
            pos = self.get_token().pos
            if self.accept_next('colon'):
                self.expect('rbrace')
                return E.EmptyDictExpression(pos)
            if self.accept_next('rbrace'):
                return E.EmptySetExpression(pos)
            dictorset = self.dictorsetmaker(pos)
            self.expect('rbrace')
            return dictorset
        if self.accept('id'):
            return self.name()
        if self.accept(*self.int_tokens):
            return self.int_number()
        if self.accept(*self.float_tokens):
            return self.float_number()
        if self.accept(*self.string_tokens):
            if self.current_token().line in range(59, 62):
                pass  # self.raise_unexpected()
            return self.string()
        if self.accept('ellipsis'):
            return E.EllipsisExpression(self.get_token().pos)
        if self.accept('none'):
            return E.NoneExpression(self.get_token().pos)
        if self.accept('true'):
            return E.TrueExpression(self.get_token().pos)
        if self.accept('false'):
            return E.FalseExpression(self.get_token().pos)
        self.raise_unexpected()

    @compose(list)
    def comp_for(self):
        yield self.comp_for_clause()
        while self.accept('if', 'for'):
            if self.accept_next('if'):
                yield self.comp_if_clause()
            else:
                yield self.comp_for_clause()

    def comp_for_clause(self):
        self.expect('for')
        exprs = self.exprlist('in')
        self.expect('in')
        iterable = self.or_test()
        return E.CompForClause(exprs=exprs, iterable=iterable)

    def comp_if_clause(self):
        return E.CompIfClause(test=self.test_nocond())

    def int_number(self):
        return E.IntExpression(self.expect_get(*self.int_tokens))

    def float_number(self):
        return E.FloatExpression(self.expect_get(*self.float_tokens))

    def name(self):
        return E.IdExpression(self.expect_get('id'))

    def string(self):
        return E.StringExpression(
            list(self.expect_get_many(*self.string_tokens)))

def single_input(tokens, input_text):
    return Parser(tokens, input_text).single_input()

def file_input(tokens, input_text):
    return Parser(tokens, input_text).file_input()

def eval_input(tokens, input_text):
    return Parser(tokens, input_text).eval_input()

def any_input(tokens, input_text, initial_production='file_input'):
    production = getattr(Parser(tokens, input_text), initial_production)
    return production()
