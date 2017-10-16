import collections
import contextlib
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

class FloatType(PrimitiveType):
    def __str__(self):
        return 'float'
Float = FloatType()

class IntType(PrimitiveType):
    def __str__(self):
        return 'int'
Int = IntType()

class ErrorType(PrimitiveType):
    def __str__(self):
        return 'error'
Error = ErrorType()

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

class TypeConstructor:
    def __init__(self, args: int):
        self.args = args

class TypeConstant(Type):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def apply(self, subst):
        return self
    def ftv(self):
        return set()

class TypeCall(Type):
    def __init__(self, con, ts):
        self.con = con
        self.ts = ts
    def __str__(self):
        params = ', '.join(map(str, self.ts))
        return f'{self.con}[{params}]'
    def apply(self, subst):
        return TypeCall(self.con.apply(subst), [t.apply(subst) for t in self.ts])
    def ftv(self):
        if len(self.ts) == 0:
            return self.con.ftv()
        return set.union(self.con.ftv(), *[t.ftv() for t in self.ts])
    def __eq__(self, other):
        if isinstance(other, TypeCall):
            if self.con == other.con:
                if len(self.ts) == len(other.ts):
                    return all(t == s for t, s in zip(self.ts, other.ts))
        return False

class ListType(Type):
    def __init__(self, t):
        self.t = t
    def __str__(self):
        return f'[{self.t}]'
    def apply(self, subst):
        return ListType(self.t.apply(subst))
    def ftv(self):
        return self.t.ftv()
    def __eq__(self, other):
        if isinstance(other, ListType):
            return self.t == other.t
        return False

class SetType(Type):
    def __init__(self, t):
        self.t = t
    def __str__(self):
        return f'{{{self.t}}}'
    def apply(self, subst):
        return SetType(self.t.apply(subst))
    def ftv(self):
        return self.t.ftv()
    def __eq__(self, other):
        if isinstance(other, SetType):
            return self.t == other.t
        return False

class DictType(Type):
    def __init__(self, k, v):
        self.k = k
        self.v = v
    def __str__(self):
        return f'{{{self.k}: {self.v}}}'
    def apply(self, subst):
        return DictType(self.k.apply(subst), self.v.apply(subst))
    def ftv(self):
        return set.union(self.k.ftv(), self.v.ftv())
    def __eq__(self, other):
        if isinstance(other, DictType):
            return self.k == other.k and self.v == other.v
        return False

class InterfaceType:
    def __init__(self, types, names):
        self.types = types
        self.names = names
    def __str__(self):
        return 'interface({self.types}; {self.names})'

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
    'set': TypeScheme(['a'], FunctionType(Unit, SetType(TypeVariable('a')))),
    'RuntimeError': TypeScheme([], FunctionType(String, Error)),
}

BASE_TYPE_ENVIRONMENT = {
    'int': Int,
    'float': Float,
    'bool': Bool,
    'error': Error,
}

BUILTIN_OPERATORS = {
    'plus': TypeScheme(['a'], FunctionType(TupleType([TypeVariable('a'), TypeVariable('a')]), TypeVariable('a'))),
}

class Infer:
    def __init__(self):
        self.env = collections.ChainMap(BASE_ENVIRONMENT)
        self.type_env = collections.ChainMap(BASE_TYPE_ENVIRONMENT)
        self.unifiers = []
        self.fresh_vars = (f'a{i}' for i in itertools.count(1))

    ######

    def fresh(self):
        return TypeVariable(next(self.fresh_vars))

    def instantiate(self, scm):
        return scm.t.apply({tvar: self.fresh() for tvar in scm.tvars})

    def lookup(self, name, pos):
        scm = self.env.get(name, None)
        if scm is None:
            raise ApeSyntaxError(msg=f'Unbound variable: {name}', pos=pos)
        return self.instantiate(scm)

    ######

    def parse_toplevel_type(self, ast):
        if not isinstance(ast, TypeForallExpression):
            raise ApeError(pos=ast.pos, msg='Unexpected type expression')
        with self.subenv():
            names = [tvar.name for tvar in ast.tvars]
            self.type_env.update({name: TypeVariable(name) for name in names})
            return TypeScheme(names, self.parse_type(ast.expr))

    @overloadmethod()
    def parse_type(self):
        ...

    @parse_type.on(TypeNameExpression)
    def _(self, ast):
        return self.type_env[ast.name]

    @parse_type.on(TypeTupleExpression)
    def _(self, ast):
        return TupleType([self.parse_type(t) for t in ast.exprs])

    @parse_type.on(TypeFunctionExpression)
    def _(self, ast):
        return FunctionType(self.parse_type(ast.t1), self.parse_type(ast.t2))

    @parse_type.on(TypeCallExpression)
    def _(self, ast):
        con = self.parse_type(ast.atom)
        return TypeCall(con, [self.parse_type(t) for t in ast.args])

    ######

    def infer_error(self, ast):
        return ApeError(pos=ast.pos, msg='no overload found for {}'.format(ast.__class__))

    @overloadmethod(use_as_modifier=True, error_function=infer_error)
    def infer(self, ast, t):
        ast.type = t

    @infer.on(EmptyListExpression)
    def _(self, ast):
        tv = self.fresh()
        return ListType(tv)

    @infer.on(EmptyDictExpression)
    def _(self, ast):
        tk = self.fresh()
        tv = self.fresh()
        return DictType(tk, tv)

    @infer.on(EmptyTupleExpression)
    def _(self, ast):
        return Unit

    @infer.on(SetLiteral)
    def _(self, ast):
        tv = self.fresh()
        ts = SetType(tv)
        for expr in ast.exprs:
            if isinstance(expr, StarExpr):
                self.unify(ts, self.infer(expr.expr), expr.pos)
            else:
                self.unify(tv, self.infer(expr), expr.pos)
        return ts

    @infer.on(DictLiteral)
    def _(self, ast):
        tK, tV = self.fresh(), self.fresh()
        tD = DictType(tK, tV)
        for expr in ast.exprs:
            if isinstance(expr, DictPair):
                self.unify(tK, self.infer(expr.key_expr), expr.key_expr.pos)
                self.unify(tV, self.infer(expr.value_expr), expr.value_expr.pos)
            else:
                self.unify(tD, self.infer(expr.expr), expr.expr.pos)
        return tD

    @infer.on(ListLiteral)
    def _(self, ast):
        tv = self.fresh()
        ts = ListType(tv)
        for expr in ast.exprs:
            if isinstance(expr, StarExpr):
                self.unify(ts, self.infer(expr.expr), expr.pos)
            else:
                self.unify(tv, self.infer(expr), expr.pos)
        return ts

    @infer.on(TupleLiteral)
    def _(self, ast):
        ts = [self.infer(expr) for expr in ast.exprs]
        return TupleType(ts)

    @infer.on(Lazy)
    def _(self, ast):
        return FunctionType(Unit, self.infer(ast))

    @infer.on(RaiseStatement)
    def _(self, ast):
        t1 = self.infer(ast.expr)
        self.unify(t1, Error, ast.expr.pos)

        if ast.original is not None:
            t2 = self.infer(ast.original)
            self.unify(t2, Error, ast.original.pos)

        return Unit

    @infer.on(NoneExpression)
    def _(self, ast):
        return Unit

    @infer.on(StringExpression)
    def _(self, ast):
        return String

    @infer.on(FloatExpression)
    def _(self, ast):
        return Float

    @infer.on(IntExpression)
    def _(self, ast):
        return Int

    @infer.on(IfElseExpr)
    def _(self, ast):
        t1 = self.infer(ast.cond)
        t2 = self.infer(ast.expr)
        t3 = self.infer(ast.alt)
        self.unify(t1, Bool, ast.cond.pos)
        self.unify(t2, t3, ast.expr.pos)
        return t2

    @infer.on(ArithExpression)
    def _(self, ast):
        t1 = self.infer(ast.left)
        t2 = self.infer(ast.right)
        tv = self.fresh()
        u1 = FunctionType(TupleType([t1, t2]), tv)
        u2 = self.instantiate(BUILTIN_OPERATORS[ast.op])
        self.unify(u1, u2, ast.pos)
        return tv

    @infer.on(IfElifElseStatement)
    def _(self, ast):
        tc1 = self.infer(ast.if_branch.cond)
        ts1 = self.infer(ast.if_branch.suite)

        self.unify(tc1, Bool, ast.pos)

        for br in ast.elif_branches:
            t = self.infer(br.cond)
            self.unify(t, Bool, br.cond.pos)
            s = self.infer(br.suite)
            self.unify(s, ts1, br.suite.pos)

        if ast.else_branch is not None:
            ts2 = self.infer(ast.else_branch.suite)
            self.unify(ts1, ts2, ast.else_branch.pos)

        return ts1

    @infer.on(CallExpression)
    def _(self, ast):
        t1 = self.infer(ast.atom)
        ts = TupleType([self.infer(arg) for arg in ast.args])
        tv = self.fresh()
        self.unify(t1, FunctionType(ts, tv), ast.pos)
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
                ts.append((t, a.pos))

        for t, pos in ts:
            self.unify(t1, t, pos)

        return Unit

    @infer.on(Comparison)
    def _(self, ast):
        t1 = self.infer(ast.a)
        t2 = self.infer(ast.b)
        self.unify(t1, t2, ast.pos)
        return Bool

    @infer.on(IdExpression)
    def _(self, ast):
        return self.lookup(ast.name, ast.pos)

    @infer.on(Statements)
    def _(self, ast):
        for stmt in ast.stmts:
            self.unify(self.infer(stmt), Unit, stmt.pos)
        return Unit

    @infer.on(LogicalOrExpressions)
    def _(self, ast):
        for expr in ast.exprs:
            self.unify(self.infer(expr), Bool, expr.pos)
        return Bool

    @compose(list)
    def infer_params(self, params):
        for p in params:
            if isinstance(p, EndOfPosParams):
                continue
            tv = self.fresh()
            self.env[p.name] = TypeScheme([], tv)

            if isinstance(p, Param):
                if p.annotation:
                    ta = self.parse_type(p.annotation)
                    self.unify(tv, ta, p.annotation.pos)

                if p.default:
                    td = self.infer(p.default)
                    self.unify(tv, td, p.default.pos)

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
                tr = self.parse_type(p.return_annotation)
                self.unify(t, tr, p.return_annotation.pos)
            return FunctionType(TupleType(params), t)
        finally:
            self.env = old

    @infer.on(InterfaceDefinition)
    def _(self, ast):
        with self.subenv():
            types = {}
            names = {}
            for decl in ast.body:
                if isinstance(decl, TypeDeclaration):
                    if len(decl.args) == 0:
                        # declaration of type
                    else:
                        # declaration of type *constructor*
                    params = [expr.name for expr in decl.args]
                    tvars = [TypeVariable(name) for name in params]
                    types[decl.name.name] = self.type_env[decl.name.name] = TypeScheme(params, TypeCall(TypeConstant(decl.name.name), tvars))
                elif isinstance(decl, NameDeclaration):
                    scm = self.parse_toplevel_type(decl.annotation)
                    name = decl.name.name
                    names[name] = self.env[name] = scm
                elif isinstance(decl, LawDeclaration):
                    with self.subenv():
                        self.env.update({name.name: TypeScheme([], self.fresh()) for name in decl.names})
                        t = self.infer(decl.expr)
                        self.unify(t, Bool, decl.pos)
                else: raise RuntimeError()
        self.type_env[ast.name.name] = InterfaceType(types, names)

        return Unit

    @infer.on(FunctionDefinition)
    def _(self, ast):
        with self.subenv():
            params = self.infer_params(ast.params)
            t = self.infer(ast.suite)
            if ast.return_annotation:
                tr = self.parse_type(p.return_annotation)
                self.unify(t, tr, p.return_annotation.pos)
            typ = FunctionType(TupleType(params), t)
        self.env[ast.name] = TypeScheme([], typ)
        return Unit

    @contextlib.contextmanager
    def subenv(self):
        old, oldT = self.env, self.type_env
        self.env = self.env.new_child()
        self.type_env = self.type_env.new_child()
        yield
        self.env, self.type_env = old, oldT

    def unify(self, t1, t2, pos):
        self.unifiers.append(Constraint(t1, t2, pos))

def infer(ast):
    i = Infer()
    t = i.infer(ast)
    s = solve({}, i.unifiers)
    return ast

def bind(tvar, t):
    if isinstance(t, TypeVariable) and t.tvar == tvar:
        return {}
    if tvar in t.ftv():
        raise RuntimeError('infinite type!')
    return {tvar: t}

def unifies(t1, t2, pos):
    if t1 == t2:
        return {}

    if isinstance(t1, TypeVariable):
        return bind(t1.tvar, t2)
    if isinstance(t2, TypeVariable):
        return bind(t2.tvar, t1)

    # one-element tuples are transparent
    if isinstance(t1, TupleType) and len(t1.ts) == 1:
        return unifies(t1.ts[0], t2, pos)
    if isinstance(t2, TupleType) and len(t2.ts) == 1:
        return unifies(t1, t2.ts[0], pos)
    if isinstance(t1, TupleType) and isinstance(t2, TupleType):
        if len(t1.ts) == len(t2.ts):
            return unify_many(t1.ts, t2.ts, pos)

    if isinstance(t1, SetType) and isinstance(t2, SetType):
        return unifies(t1.t, t2.t, pos)
    if isinstance(t1, ListType) and isinstance(t2, ListType):
        return unifies(t1.t, t2.t, pos)

    if isinstance(t1, DictType) and isinstance(t2, DictType):
        return unify_many([t1.k, t1.v], [t2.k, t2.v], pos)
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        return unify_many([t1.t1, t1.t2], [t2.t1, t2.t2], pos)
    raise ApeUnificationError(pos=pos, msg=f'Cannot unify {t1} and {t2}')

def unify_many(t1x, t2x, pos):
    if len(t1x) == 0 and len(t2x) == 0:
        return {}

    [t1, *t1s], [t2, *t2s] = t1x, t2x
    su1 = unifies(t1, t2, pos)
    su2 = unify_many(
        [t.apply(su1) for t in t1s],
        [t.apply(su1) for t in t2s], pos)
    return compose_subst(su2, su1)

class Constraint:
    def __init__(self, t1, t2, pos):
        self.t1 = t1
        self.t2 = t2
        self.pos = pos
    def __repr__(self):
        return f'{self.t1} ~ {self.t2}'
    def apply(self, subst):
        return Constraint(self.t1.apply(subst), self.t2.apply(subst), self.pos)
    def ftv(self):
        return set.union(self.t1.ftv(), self.t2.ftv())

def compose_subst(su1, su2):
    return {**su1, **su2}

def solve(su, cs):
    if len(cs) == 0:
        return su

    [c, *cs0] = cs
    su1 = unifies(c.t1, c.t2, c.pos)

    return solve(compose_subst(su1, su), [c.apply(su1) for c in cs0])

