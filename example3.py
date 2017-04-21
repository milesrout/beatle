
class Foo:
    def expr_stmt(self):
        tlse = self.testlist_star_expr()
        if self.accept_next('colon'):
            annotation = self.test()
            if self.accept_next('equal'):
                return AnnotatedAssignment(
                    type='equal',
                    assignee=tlse,
                    expr=self.test(),
                    annotation=annotation)
            return AnnotatedExpression(tlse, annotation)
        if self.accept(*self.augassign_tokens):
            augtype = self.get_token().type
            if self.accept_next('yield'):
                expr = self.yield_expr('semicolon', 'newline')
            else:
                expr = self.testlist()
            return AugmentedAssignment(augtype, expr)
        exprs = [tlse]
        while self.accept_next('equal'):
            if self.accept_next('yield'):
                exprs.append(self.yield_expr('equal', 'semicolon', 'newline'))
            else:
                exprs.append(self.testlist_star_expr())
        if len(exprs) == 1:
            return tlse
        else:
            return ChainedAssignment(exprs)

    def testlist_star_expr(self):
        exprs = []
        if self.accept_next('asterisk'):
            exprs.append(StarExpr(self.expr()))
        else:
            exprs.append(self.test())
        while self.accept_next('comma'):
            if self.accept('equal', 'colon', *self.augassign_tokens):
                return exprs
            if self.accept_next('asterisk'):
                exprs.append(StarExpr(self.expr()))
            else:
                exprs.append(self.test())
        return exprs
    pass

