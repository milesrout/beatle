import collections
import functools
import itertools

from astnodes import *
from utils import *

class ApeInferenceError(ApeError): pass
class ApeUnificationError(ApeError): pass

class PrimitiveType(Type):
    def apply(self, subst):
        return self
    def ftv(self):
        return set()

class BoolType(PrimitiveType):
    def __str__(self):
        return 'bool'
Bool = BoolType()

class IntType(PrimitiveType):
    def __str__(self):
        return 'int'
Int = IntType()

class StringType(PrimitiveType):
    def __str__(self):
        return 'string'
String = StringType()

class TupleType(Type):
    def __init__(self, ts):
        self.ts = ts
    def __str__(self):
        return '(' + ' ✗ '.join(map(str, self.ts)) + ')'
    def apply(self, subst):
        return TupleType([t.apply(subst) for t in self.ts])
    def ftv(self):
        if len(self.ts) == 0:
            return set()
        return set.union(*[t.ftv() for t in self.ts])
    def __eq__(self, other):
        if isinstance(other, TupleType):
            if len(self.ts) == len(other.ts):
                return all(t == s for t, s in zip(self.ts, other.ts))
        return False
Unit = TupleType([])

class TypeVariable(Type):
    def __init__(self, tvar):
        self.tvar = tvar
    def __str__(self):
        return self.tvar
    def apply(self, subst):
        return subst.get(self.tvar, self)
    def ftv(self):
        return {self.tvar}

class TypeConstant(Type):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def apply(self, subst):
        return self
    def ftv(self):
        return set()

class ListType(Type):
    def __init__(self, t):
        self.t = t
    def __str__(self):
        return f'[{self.t}]'
    def apply(self, subst):
        return ListType(self.t.apply(subst))
    def ftv(self):
        return self.t.ftv()

class FunctionType(Type):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
    def __str__(self):
        return f'({self.t1} → {self.t2})'
    def apply(self, subst):
        return FunctionType(self.t1.apply(subst), self.t2.apply(subst))
    def ftv(self):
        return set.union(self.t1.ftv(), self.t2.ftv())

class TypeScheme:
    def __init__(self, tvars, t):
        self.tvars = tvars
        self.t = t
    def __str__(self):
        v = ','.join(map(str, self.tvars))
        return f'∀{v}.{self.t}'
    def apply(self, subst):
        s = {k: v for k, v in subst.items() if k not in self.tvars}
        return TypeScheme(self.tvars, self.t.apply(s))
    def ftv(self):
        return self.t.ftv() - set(self.tvars)

#print(TypeScheme(['a'], FunctionType(TypeVariable('a'), TypeVariable('a'))).apply({'a': 'b'}))
#print(TypeScheme(['a', 'b'], FunctionType(
#    FunctionType(TypeVariable('a'), TypeVariable('b')),
#    FunctionType(ListType(TypeVariable('a')), ListType(TypeVariable('b')))
#)))

BASE_ENVIRONMENT = {
    'print': TypeScheme([], FunctionType(String, Unit)),
    'str': TypeScheme(['a'], FunctionType(TypeVariable('a'), String)),
}

BUILTIN_OPERATORS = {
    'plus': FunctionType(TupleType([Int, Int]), Int)
}

class Infer:
    def __init__(self):
        self.env = collections.ChainMap(BASE_ENVIRONMENT)
        self.unifiers = []
        self.fresh_vars = (f'a{i}' for i in itertools.count(1))

    def fresh(self):
        return TypeVariable(next(self.fresh_vars))

    def instantiate(self, scm):
        return scm.t.apply({tvar: self.fresh() for tvar in scm.tvars})

    def lookup(self, name):
        scm = self.env.get(name, None)
        if scm is None:
            raise ApeSyntaxError(f'Unbound variable: {name}', pos=0)
        return self.instantiate(scm)

    @overloadmethod
    def infer(self, ast, t):
        ast.type = t

    @infer.on(EmptyTupleExpression)
    def _(self, ast):
        return Unit

    @infer.on(EmptyListExpression)
    def _(self, ast):
        tv = self.fresh()
        return TypeScheme([tv.name], ListType(tv))

    @infer.on(StringExpression)
    def _(self, ast):
        return String

    @infer.on(IntExpression)
    def _(self, ast):
        return Int

    @infer.on(IfElseExpr)
    def _(self, ast):
        t1 = self.infer(ast.cond)
        t2 = self.infer(ast.expr)
        t3 = self.infer(ast.alt)
        self.unify(t1, Bool)
        self.unify(t2, t3)
        return t2

    @infer.on(ArithExpression)
    def _(self, ast):
        t1 = self.infer(ast.left)
        t2 = self.infer(ast.right)
        tv = self.fresh()
        u1 = FunctionType(TupleType([t1, t2]), tv)
        u2 = BUILTIN_OPERATORS[ast.op]
        self.unify(u1, u2)
        return tv

    @infer.on(IfElifElseStatement)
    def _(self, ast):
        tc1 = self.infer(ast.if_branch.cond)
        tcs = [self.infer(b.cond) for b in ast.elif_branches]

        self.unify(tc1, Bool)
        for tc in tcs:
            self.unify(tc1, Bool)

        ts1 = self.infer(ast.if_branch.suite)
        tss = [self.infer(b.suite) for b in ast.elif_branches]

        if ast.else_branch is not None:
            ts2 = self.infer(ast.else_branch.suite)
            self.unify(ts1, ts2)

        for ts in tss:
            self.unify(ts1, ts)

        return ts1

    @infer.on(TupleLiteral)
    def _(self, ast):
        ts = [self.infer(expr) for expr in ast.exprs]
        return TupleType(ts)

    @infer.on(CallExpression)
    def _(self, ast):
        t1 = self.infer(ast.atom)
        ts = TupleType([self.infer(arg) for arg in ast.args])
        tv = self.fresh()
        self.unify(t1, FunctionType(ts, tv))
        return tv

    @infer.on(ChainedAssignment)
    def _(self, ast):
        assignees = ast.assignees[:-1]
        expr = ast.assignees[-1]

        t1 = self.infer(expr)
        ts = []
        for a in assignees:
            if isinstance(a, IdExpression):
                t = self.fresh()
                self.env[a.name] = TypeScheme([], t)
                ts.append(t)

        for t in ts:
            self.unify(t1, t)

        return Unit

    @infer.on(Comparison)
    def _(self, ast):
        t1 = self.infer(ast.a)
        t2 = self.infer(ast.b)
        self.unify(t1, t2)
        return Bool

    @infer.on(IdExpression)
    def _(self, ast):
        return self.lookup(ast.name)

    @infer.on(Statements)
    def _(self, ast):
        ts = [self.infer(stmt) for stmt in ast.stmts]
        for t in ts:
            self.unify(t, Unit)
        return Unit

    @infer.on(LogicalOrExpressions)
    def _(self, ast):
        ts = [self.infer(expr) for expr in ast.exprs]
        for t in ts:
            self.unify(t, Bool)
        return Bool

    @compose(list)
    def infer_params(self, params):
        for p in params:
            tv = self.fresh()
            self.env[p.name] = TypeScheme([], tv)

            if p.annotation:
                pass # we don't parse types properly yet
                # ta = self.understand_type_language(p.annotation)
                # self.unify(tv, ta)

            if p.default:
                td = self.infer(p.default)
                self.unify(tv, td)

            yield tv

    @infer.on(PassStatement)
    def _(self, ast):
        return Unit

    @infer.on(FunctionExpression)
    def _(self, ast):
        old = self.env
        self.env = self.env.new_child()
        try:
            params = self.infer_params(ast.params)
            t = self.infer(ast.suite)
            if ast.return_annotation:
                pass # we don't parse types properly yet
                # tr = self.understand_type_language(ast.return_annotation)
                # self.unify(t, tr)
            return FunctionType(TupleType(params), t)
        finally:
            self.env = old

    @infer.on(FunctionDefinition)
    def _(self, ast):
        old = self.env
        self.env = self.env.new_child()
        try:
            params = self.infer_params(ast.params)
            t = self.infer(ast.suite)
            if ast.return_annotation:
                pass # we don't parse types properly yet
                # tr = self.understand_type_language(ast.return_annotation)
                # self.unify(t, tr)
            return FunctionType(TupleType(params), t)
        finally:
            self.env = old

    def unify(self, t1, t2):
        if isinstance(t1, TypeScheme) or isinstance(t2, TypeScheme):
            raise RuntimeError('What???')
        self.unifiers.append(Constraint(t1, t2))

def infer(ast):
    i = Infer()
    try:
        t = i.infer(ast)
        s = solve({}, i.unifiers)
        return ast.apply(s)
    except TypeError as exc:
        raise ApeInferenceError(line=0, col=0, msg=str(exc)) from exc

def bind(tvar, t):
    if isinstance(t, TypeVariable) and t.tvar == tvar:
        return {}
    if tvar in t.ftv():
        raise RuntimeError('infinite type!')
    return {tvar: t}

def unifies(t1, t2):
    if t1 == t2:
        return {}
    if isinstance(t1, TypeVariable):
        return bind(t1.tvar, t2)
    if isinstance(t2, TypeVariable):
        return bind(t2.tvar, t1)
    if isinstance(t1, TupleType) and len(t1.ts) == 1:
        return unifies(t1.ts[0], t2)
    if isinstance(t2, TupleType) and len(t2.ts) == 1:
        return unifies(t1, t2.ts[0])
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        return unify_many([t1.t1, t1.t2], [t2.t1, t2.t2])
    raise ApeUnificationError(line=0, col=0, msg=f'Cannot unify {t1} and {t2}')

def unify_many(t1x, t2x):
    if len(t1x) == 0 and len(t2x) == 0:
        return {}
    [t1, *t1s], [t2, *t2s] = t1x, t2x
    su1 = unifies(t1, t2)
    su2 = unify_many(
        [t.apply(su1) for t in t1s],
        [t.apply(su1) for t in t2s])
    return compose_subst(su2, su1)

class Constraint:
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
    def __repr__(self):
        return f'{self.t1} ~ {self.t2}'
    def apply(self, subst):
        return Constraint(self.t1.apply(subst), self.t2.apply(subst))
    def ftv(self):
        return set.union(self.t1.ftv(), self.t2.ftv())
    def __iter__(self):
        yield self.t1
        yield self.t2

def compose_subst(su1, su2):
    return {**su1, **su2}

def solve(su, cs):
    if len(cs) == 0:
        return su

    [(t1, t2), *cs0] = cs
    su1 = unifies(t1, t2)

    return solve(compose_subst(su1, su), [c.apply(su1) for c in cs0])

