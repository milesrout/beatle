@infer.on(E.CallExpression)
def _(self, ast):
    ef, tf = self.infer(ast.atom)

    ta, es = unzip(for arg in ast.args:
                      ea, ta = self.infer(arg)
                      self.unify_others(tf, ta, ast.pos)
                      (ta, ea))

    es, ts = [], []
    for arg in ast.args:
        ea, ta = self.infer(arg)
        self.unify_others(tf, ta, ast.pos)
        ts.append(ta)
        es.append(ea)

    tt = TupleType([t.vtype for t in ts])
    tv = self.fresh()
    self.unify(tf.vtype, FunctionType(tt, tv), ast.pos)
    return T.Call(ef, es), ta.but(tv)
