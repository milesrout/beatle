from utils import overloadmethod
import cstnodes as E

class AstPass:
    """A compiler pass over the abstract syntax tree"""

    def visit(self, ast):
        result = self.do_visit(ast)
        if not hasattr(result, 'pos'):
            try:
                result.pos = ast.pos
            except AttributeError as exc:
                print('Error while trying to assign pos:', exc)
        return result

class DeepAstPass(AstPass):
    """A compiler pass over all syntax tree nodes"""
    def visit(self, ast):
        return self.do_visit(ast)

    def visit_maybe_all(self, L):
        return [self.visit_all(x) if isinstance(x, list) else self.visit(x) for x in L]

    def visit_all(self, list):
        return [self.visit(x) for x in list]

    def visit_maybe(self, maybe):
        return self.visit(maybe) if maybe is not None else None

    @overloadmethod()
    def do_visit(self, ast):
        ...

    @do_visit.default()
    def do_visit_default(self, *args, **kwds):
        print(args, kwds)
        raise

    @do_visit.wrapper()
    def do_visit_wrapper(self, ast, new):
        return self.override_do_visit_wrapper(ast, new)

    def override_do_visit_wrapper(self, ast, new):
        if ast is new:
            return ast

        cls, args = new
        try:
            return cls(*args)
        except TypeError:
            print(cls.__name__)
            raise

    @do_visit.on(E.EmptyListExpression)
    def do_visit_EmptyListExpression(self, ast):
        return ast

    @do_visit.on(E.EmptyDictExpression)
    def do_visit_EmptyDictExpression(self, ast):
        return ast

    @do_visit.on(E.EmptySetExpression)
    def do_visit_EmptySetExpression(self, ast):
        return ast

    @do_visit.on(E.EmptyTupleExpression)
    def do_visit_EmptyTupleExpression(self, ast):
        return ast

    @do_visit.on(E.SetComprehension)
    def do_visit_SetComprehension(self, ast):
        return E.SetComprehension, (self.visit(ast.expr), self.visit_all(ast.rest), ast.pos)

    @do_visit.on(E.DictComprehension)
    def do_visit_DictComprehension(self, ast):
        return E.DictComprehension, (self.visit(ast.expr), self.visit_all(ast.rest), ast.pos)

    @do_visit.on(E.ListComprehension)
    def do_visit_ListComprehension(self, ast):
        return E.ListComprehension, (self.visit(ast.expr), self.visit_all(ast.rest), ast.pos)

    @do_visit.on(E.GeneratorExpression)
    def do_visit_GeneratorExpression(self, ast):
        return E.GeneratorExpression, (self.visit(ast.expr), self.visit_all(ast.rest), ast.pos)

    @do_visit.on(E.SetLiteral)
    def do_visit_SetLiteral(self, ast):
        return E.SetLiteral, (self.visit_all(ast.exprs), ast.pos)

    @do_visit.on(E.DictLiteral)
    def do_visit_DictLiteral(self, ast):
        return E.DictLiteral, (self.visit_all(ast.exprs), ast.pos)

    @do_visit.on(E.DictPair)
    def do_visit_DictPair(self, ast):
        return E.DictPair, (self.visit(ast.key_expr), self.visit(ast.value_expr))

    @do_visit.on(E.ListLiteral)
    def do_visit_ListLiteral(self, ast):
        return E.ListLiteral, (self.visit_all(ast.exprs), ast.pos)

    @do_visit.on(E.TupleLiteral)
    def do_visit_TupleLiteral(self, ast):
        return E.TupleLiteral, (self.visit_all(ast.exprs), ast.pos)

    @do_visit.on(E.TaggedExpression)
    def do_visit_TaggedExpression(self, ast):
        return E.TaggedExpression, (ast.tag, self.visit_maybe(ast.expr), ast.pos)

    @do_visit.on(E.Quasiquote)
    def do_visit_Quasiquote(self, ast):
        return E.Quasiquote, (self.visit(ast.expr), ast.pos)

    @do_visit.on(E.Unquote)
    def do_visit_Unquote(self, ast):
        return E.Unquote, (self.visit(ast.expr), ast.pos)

    @do_visit.on(E.UnquoteSplice)
    def do_visit_UnquoteSplice(self, ast):
        return E.UnquoteSplice, (self.visit(ast.expr), ast.pos)

    @do_visit.on(E.Lazy)
    def do_visit_Lazy(self, ast):
        return E.Lazy, (self.visit(ast.expr), ast.pos)

    @do_visit.on(E.Expressions)
    def do_visit_Expressions(self, ast):
        return E.Expressions, (self.visit_all(ast.exprs),)

    @do_visit.on(E.Statements)
    def do_visit_Statements(self, ast):
        return E.Statements, (self.visit_all(ast.stmts),)

    @do_visit.on(E.RaiseStatement)
    def do_visit_RaiseStatement(self, ast):
        expr = self.visit_maybe(ast.expr)
        original = self.visit_maybe(ast.original)
        return E.RaiseStatement, (expr, original, ast.pos)

    @do_visit.on(E.YieldFromExpression)
    def do_visit_YieldFromExpression(self, ast):
        return E.YieldFromExpression, (self.visit(ast.expr), ast.pos)

    @do_visit.on(E.YieldExpression)
    def do_visit_YieldExpression(self, ast):
        return E.YieldExpression, (self.visit(ast.expr), ast.pos)

    @do_visit.on(E.DelStatement)
    def do_visit_DelStatement(self, ast):
        return E.DelStatement, (self.visit_all(ast.exprs), ast.pos)

    @do_visit.on(E.AssertStatement)
    def do_visit_AssertStatement(self, ast):
        return E.AssertStatement, (self.visit_all(ast.exprs), ast.pos)

    @do_visit.on(E.GlobalStatement)
    def do_visit_GlobalStatement(self, ast):
        return E.GlobalStatement, (self.visit_all(ast.names), ast.pos)

    @do_visit.on(E.NonlocalStatement)
    def do_visit_NonlocalStatement(self, ast):
        return E.NonlocalStatement, (self.visit_all(ast.names), ast.pos)

    @do_visit.on(E.IfBranch)
    def do_visit_IfBranch(self, ast):
        return E.IfBranch, (self.visit(ast.cond), self.visit(ast.body), ast.pos)

    @do_visit.on(E.ElifBranch)
    def do_visit_ElifBranch(self, ast):
        return E.ElifBranch, (self.visit(ast.cond), self.visit(ast.body), ast.pos)

    @do_visit.on(E.ElseBranch)
    def do_visit_ElseBranch(self, ast):
        return E.ElseBranch, (self.visit(ast.body), ast.pos)

    @do_visit.on(E.DoStatement)
    def do_visit_DoStatement(self, ast):
        return E.DoStatement, (self.visit(ast.body), ast.pos)

    @do_visit.on(E.IfElifElseStatement)
    def do_visit_IfElifElseStatement(self, ast):
        i = self.visit(ast.if_branch)
        eis = self.visit_all(ast.elif_branches)
        e = self.visit_maybe(ast.else_branch)
        return E.IfElifElseStatement, (i, eis, e, ast.pos)

    @do_visit.on(E.MatchStatement)
    def do_visit_MatchStatement(self, ast):
        return E.MatchStatement, (self.visit(ast.expr), self.visit_all(ast.cases), ast.pos)

    @do_visit.on(E.MatchCase)
    def do_visit_MatchCase(self, ast):
        return E.MatchCase, (self.visit(ast.pattern), self.visit(ast.body), ast.pos)

    @do_visit.on(E.WhileStatement)
    def do_visit_WhileStatement(self, ast):
        alt = self.visit_maybe(ast.alt)
        return E.WhileStatement, (self.visit(ast.cond), self.visit(ast.body), alt, ast.pos)

    @do_visit.on(E.ForStatement)
    def do_visit_ForStatement(self, ast):
        asn = self.visit(ast.assignees)
        its = self.visit(ast.iterables)
        b = self.visit(ast.body)
        alt = self.visit_maybe(ast.alt)
        return E.ForStatement, (asn, its, b, alt, ast.pos)

    @do_visit.on(E.TryStatement)
    def do_visit_TryStatement(self, ast):
        b = self.visit(ast.body)
        exs = self.visit_all(ast.excepts)
        els = self.visit_all(ast.elses)
        fis = self.visit_all(ast.finallies)
        return E.TryStatement, (b, exs, els, fis, ast.pos)

    @do_visit.on(E.MacroDefinition)
    def do_visit_MacroDefinition(self, ast):
        return E.MacroDefinition, (self.visit(ast.name), self.visit_all(ast.params), self.visit(ast.body), ast.pos)

    @do_visit.on(E.FunctionDefinition)
    def do_visit_FunctionDefinition(self, ast):
        return E.FunctionDefinition, (self.visit(ast.name), self.visit_all(ast.params), self.visit(ast.body), self.visit_maybe(ast.return_annotation), ast.pos)

    @do_visit.on(E.FunctionExpression)
    def do_visit_FunctionExpression(self, ast):
        return E.FunctionExpression, (self.visit_all(ast.params), self.visit(ast.body), self.visit_maybe(ast.return_annotation), ast.pos)

    @do_visit.on(E.LambdaExpression)
    def do_visit_LambdaExpression(self, ast):
        return E.LambdaExpression, (self.visit_all(ast.params), self.visit(ast.body), ast.pos)

    @do_visit.on(E.NamespaceDefinition)
    def do_visit_NamespaceDefinition(self, ast):
        return E.NamespaceDefinition, (self.visit(ast.name), ast.key, self.visit(ast.expr), ast.pos)

    @do_visit.on(E.NamespaceReferenceDefinition)
    def do_visit_NamespaceReferenceDefinition(self, ast):
        return E.NamespaceReferenceDefinition, (self.visit(ast.name), ast.key, ast.pos)

    @do_visit.on(E.ControlStructureExpression)
    def do_visit_ControlStructureExpression(self, ast):
        return E.ControlStructureExpression, (self.visit_all(ast.components),)

    @do_visit.on(E.ControlStructureLinkExpression)
    def do_visit_ControlStructureLinkExpression(self, ast):
        L = [self.visit_all(p) if isinstance(p, list) else self.visit(p) for p in ast.params]
        return E.ControlStructureLinkExpression, (ast.name, L, self.visit(ast.body), ast.pos)

    @do_visit.on(E.SignatureDefinition)
    def do_visit_SignatureDefinition(self, ast):
        return E.SignatureDefinition, (self.visit(ast.name), self.visit_all(ast.body), ast.pos)

    @do_visit.on(E.ModuleDefinition)
    def do_visit_ModuleDefinition(self, ast):
        return E.ModuleDefinition, (self.visit(ast.name), self.visit_all(ast.params), self.visit(ast.typ), self.visit(ast.body), ast.pos)

    @do_visit.on(E.WithStatement)
    def do_visit_WithStatement(self, ast):
        return E.WithStatement, (self.visit_all(ast.items), self.visit(ast.body), ast.pos)

    @do_visit.on(E.ExceptBlock)
    def do_visit_ExceptBlock(self, ast):
        return E.ExceptBlock, (self.visit_maybe(ast.test), self.visit_maybe(ast.name), self.visit(ast.body))

    @do_visit.on(E.WithItem)
    def do_visit_WithItem(self, ast):
        return E.WithItem, (self.visit(ast.expr), self.visit_maybe(ast.assignee))

    @do_visit.on(E.Decorator)
    def do_visit_Decorator(self, ast):
        return E.Decorator, (self.visit(ast.name), self.visit_maybe(ast.args), ast.pos)

    @do_visit.on(E.Decorated)
    def do_visit_Decorated(self, ast):
        return E.Decorated, (self.visit_all(ast.decorators), self.visit(ast.defn), ast.pos)

    @do_visit.on(E.ImportName)
    def do_visit_ImportName(self, ast):
        return E.ImportName, (self.visit(ast.name), self.visit_maybe(ast.alias))

    @do_visit.on(E.ImportStatement)
    def do_visit_ImportStatement(self, ast):
        return E.ImportStatement, (self.visit_all(ast.names), ast.pos)

    @do_visit.on(E.FromImportStatement)
    def do_visit_FromImportStatement(self, ast):
        name = self.visit_maybe(ast.name)
        what = self.visit_all(ast.what)
        return E.FromImportStatement, (name, ast.dots, what, ast.pos)

    @do_visit.on(E.ChainedAssignment)
    def do_visit_ChainedAssignment(self, ast):
        return E.ChainedAssignment, (self.visit_all(ast.assignees),)

    @do_visit.on(E.AnnotatedAssignment)
    def do_visit_AnnotatedAssignment(self, ast):
        return E.AnnotatedAssignment, (self.visit(ast.assignee), self.visit(ast.expr), self.visit(ast.annotation), ast.pos)

    @do_visit.on(E.AnnotatedExpression)
    def do_visit_AnnotatedExpression(self, ast):
        return E.AnnotatedExpression, (self.visit(ast.expr), self.visit(ast.annotation), ast.pos)

    @do_visit.on(E.AugmentedAssignment)
    def do_visit_AugmentedAssignment(self, ast):
        print(ast.assignee, ast.expr)
        return E.AugmentedAssignment, (self.visit(ast.assignee), ast.op, self.visit(ast.expr), ast.pos)

    @do_visit.on(E.StarStarExpr)
    def do_visit_StarStarExpr(self, ast):
        return E.StarStarExpr, (self.visit(ast.expr), ast.pos)

    @do_visit.on(E.StarExpr)
    def do_visit_StarExpr(self, ast):
        return E.StarExpr, (self.visit(ast.expr), ast.pos)

    @do_visit.on(E.IfElseExpr)
    def do_visit_IfElseExpr(self, ast):
        return E.IfElseExpr, (self.visit(ast.expr), self.visit(ast.cond), self.visit(ast.alt), ast.pos)

    @do_visit.on(E.LogicalOrExpressions)
    def do_visit_LogicalOrExpressions(self, ast):
        return E.LogicalOrExpressions, (self.visit_all(ast.exprs),)

    @do_visit.on(E.LogicalAndExpressions)
    def do_visit_LogicalAndExpressions(self, ast):
        return E.LogicalAndExpressions, (self.visit_all(ast.exprs),)

    @do_visit.on(E.LogicalNotExpression)
    def do_visit_LogicalNotExpression(self, ast):
        return E.LogicalNotExpression, (self.visit(ast.expr),)

    @do_visit.on(E.BitOrExpression)
    def do_visit_BitOrExpression(self, ast):
        return E.BitOrExpression, (self.visit_all(ast.exprs),)

    @do_visit.on(E.BitXorExpression)
    def do_visit_BitXorExpression(self, ast):
        return E.BitXorExpression, (self.visit_all(ast.exprs),)

    @do_visit.on(E.BitAndExpression)
    def do_visit_BitAndExpression(self, ast):
        return E.BitAndExpression, (self.visit_all(ast.exprs),)

    @do_visit.on(E.BitShiftExpression)
    def do_visit_BitShiftExpression(self, ast):
        return E.BitShiftExpression, (ast.op, self.visit(ast.left), self.visit(ast.right))

    @do_visit.on(E.ArithExpression)
    def do_visit_ArithExpression(self, ast):
        return E.ArithExpression, (ast.op, self.visit(ast.left), self.visit(ast.right))

    @do_visit.on(E.UnaryExpression)
    def do_visit_UnaryExpression(self, ast):
        return E.UnaryExpression, (ast.op, self.visit(ast.expr), ast.pos)

    @do_visit.on(E.PowerExpression)
    def do_visit_PowerExpression(self, ast):
        return E.PowerExpression, (ast.op, self.visit(ast.expr), self.visit(ast.exponent))

    @do_visit.on(E.CallExpression)
    def do_visit_CallExpression(self, ast):
        return E.CallExpression, (self.visit(ast.atom), self.visit_all(ast.args), ast.pos)

    @do_visit.on(E.IndexExpression)
    def do_visit_IndexExpression(self, ast):
        return E.IndexExpression, (self.visit(ast.atom), self.visit_all(ast.indices), ast.pos)

    @do_visit.on(E.AttrExpression)
    def do_visit_AttrExpression(self, ast):
        return E.AttrExpression, (self.visit(ast.atom), self.visit(ast.name), ast.pos)

    @do_visit.on(E.IntExpression)
    def do_visit_IntExpression(self, ast):
        return ast

    @do_visit.on(E.FloatExpression)
    def do_visit_FloatExpression(self, ast):
        return ast

    @do_visit.on(E.DottedNameExpression)
    def do_visit_DottedNameExpression(self, ast):
        return ast

    @do_visit.on(E.IdExpression)
    def do_visit_IdExpression(self, ast):
        return ast

    @do_visit.on(E.SymbolExpression)
    def do_visit_SymbolExpression(self, ast):
        return ast

    @do_visit.on(E.StringExpression)
    def do_visit_StringExpression(self, ast):
        return ast

    @do_visit.on(E.EllipsisExpression)
    def do_visit_EllipsisExpression(self, ast):
        return ast

    @do_visit.on(E.NoneExpression)
    def do_visit_NoneExpression(self, ast):
        return ast

    @do_visit.on(E.TrueExpression)
    def do_visit_TrueExpression(self, ast):
        return ast

    @do_visit.on(E.FalseExpression)
    def do_visit_FalseExpression(self, ast):
        return ast

    @do_visit.on(E.AttrTrailer)
    def do_visit_AttrTrailer(self, ast):
        raise RuntimeError('We should never get here')

    @do_visit.on(E.CallTrailer)
    def do_visit_CallTrailer(self, ast):
        raise RuntimeError('We should never get here')

    @do_visit.on(E.IndexTrailer)
    def do_visit_IndexTrailer(self, ast):
        raise RuntimeError('We should never get here')

    @do_visit.on(E.Index)
    def do_visit_Index(self, ast):
        return E.Index, (self.visit(ast.idx),)

    @do_visit.on(E.Slice)
    def do_visit_Slice(self, ast):
        return E.Slice, (self.visit_maybe(ast.start), self.visit_maybe(ast.end), self.visit_maybe(ast.step))

    @do_visit.on(E.StarArg)
    def do_visit_StarArg(self, ast):
        return E.StarArg, (self.visit(ast.name),)

    @do_visit.on(E.StarStarKwarg)
    def do_visit_StarStarKwarg(self, ast):
        return E.StarStarKwarg, (self.visit(ast.name),)

    @do_visit.on(E.PlainArg)
    def do_visit_PlainArg(self, ast):
        return E.PlainArg, (self.visit(ast.expr),)

    @do_visit.on(E.KeywordArg)
    def do_visit_KeywordArg(self, ast):
        return E.KeywordArg, (self.visit(ast.name), self.visit(ast.expr))

    @do_visit.on(E.CompForArg)
    def do_visit_CompForArg(self, ast):
        return E.CompForArg, (self.visit(ast.keyword), self.visit(ast.comp))

    @do_visit.on(E.CompForClause)
    def do_visit_CompForClause(self, ast):
        return E.CompForClause, (self.visit_all(ast.exprs), self.visit(ast.iterable))

    @do_visit.on(E.CompIfClause)
    def do_visit_CompIfClause(self, ast):
        return E.CompIfClause, (self.visit(ast.test),)

    @do_visit.on(E.PassStatement)
    def do_visit_PassStatement(self, ast):
        return ast

    @do_visit.on(E.BreakStatement)
    def do_visit_BreakStatement(self, ast):
        return ast

    @do_visit.on(E.ContinueStatement)
    def do_visit_ContinueStatement(self, ast):
        return ast

    @do_visit.on(E.ReturnStatement)
    def do_visit_ReturnStatement(self, ast):
        return E.ReturnStatement, (self.visit_maybe(ast.expr), ast.pos)

    @do_visit.on(E.TypeNameExpression)
    def do_visit_TypeNameExpression(self, ast):
        return E.TypeNameExpression, (self.visit(ast.name),)

    @do_visit.on(E.TypeFunctionExpression)
    def do_visit_TypeFunctionExpression(self, ast):
        return E.TypeFunctionExpression, (self.visit(ast.t1), self.visit(ast.t2))

    @do_visit.on(E.TypeTupleExpression)
    def do_visit_TypeTupleExpression(self, ast):
        return E.TypeTupleExpression, (self.visit_all(ast.exprs), ast.pos)

    @do_visit.on(E.TypeForallExpression)
    def do_visit_TypeForallExpression(self, ast):
        return E.TypeForallExpression, (self.visit_all(ast.tvars), self.visit(ast.expr), ast.pos)

    @do_visit.on(E.TypeMaybeExpression)
    def do_visit_TypeMaybeExpression(self, ast):
        return E.TypeMaybeExpression, (self.visit(ast.t), ast.pos)

    @do_visit.on(E.TypeDisjunctionExpression)
    def do_visit_TypeDisjunctionExpression(self, ast):
        return E.TypeDisjunctionExpression, (self.visit_all(ast.exprs), ast.pos)

    @do_visit.on(E.TypeTaggedExpression)
    def do_visit_TypeTaggedExpression(self, ast):
        return E.TypeTaggedExpression, (ast.tag, self.visit_maybe(ast.t), ast.pos)

    @do_visit.on(E.TypeCallExpression)
    def do_visit_TypeCallExpression(self, ast):
        return E.TypeCallExpression, (self.visit(ast.atom), self.visit_all(ast.args))

    @do_visit.on(E.NameDeclaration)
    def do_visit_NameDeclaration(self, ast):
        return E.NameDeclaration, (self.visit(ast.name), self.visit(ast.annotation), ast.pos)

    @do_visit.on(E.TypeDefinition)
    def do_visit_TypeDefinition(self, ast):
        return E.TypeDefinition, (self.visit(ast.name), self.visit_all(ast.args), self.visit(ast.expr), ast.pos)

    @do_visit.on(E.TypeDeclaration)
    def do_visit_TypeDeclaration(self, ast):
        return E.TypeDeclaration, (self.visit(ast.name), self.visit_all(ast.args), ast.pos)

    @do_visit.on(E.LawDeclaration)
    def do_visit_LawDeclaration(self, ast):
        return E.LawDeclaration, (self.visit_all(ast.names), self.visit(ast.expr), ast.pos)

    @do_visit.on(E.Param)
    def do_visit_Param(self, ast):
        return E.Param, (ast.name, self.visit_maybe(ast.annotation), self.visit_maybe(ast.default))

    @do_visit.on(E.EndOfPosParams)
    def do_visit_EndOfPosParams(self, ast):
        return ast

    @do_visit.on(E.StarVarParams)
    def do_visit_StarVarParams(self, ast):
        return E.StarVarParams, (self.visit(ast.name), self.visit_maybe(ast.annotation), ast.pos)

    @do_visit.on(E.StarStarKwParams)
    def do_visit_StarStarKwParams(self, ast):
        return E.StarStarKwParams, (self.visit(ast.name), self.visit_maybe(ast.annotation), ast.pos)

    @do_visit.on(E.Comparison)
    def do_visit_Comparison(self, ast):
        return E.Comparison, (ast.op, self.visit(ast.a), self.visit(ast.b), ast.pos)
