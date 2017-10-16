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

class VoidType(PrimitiveType):
    """This should not be instantiated ANYWHERE!
    
    This type exists to be unified with type variables that are required to be
    unconstrained."""
    def __str__(self):
        return 'void'
Void = VoidType()

class CoroutineType(Type):
    def __init__(self, ytype, stype, rtype):
        self.ytype = ytype
        self.stype = stype
        self.rtype = rtype
    def __str__(self):
        if self.stype == Unit:
            if self.rtype == Unit:
                return f'generator[{self.ytype}]'
            else:
                return f'generator[{self.ytype}, {self.rtype}]'
        else:
            if self.rtype == Unit:
                return f'coroutine[{self.ytype}, {self.stype}]'
            else:
                return f'coroutine[{self.ytype}, {self.stype}, {self.rtype}]'
    def apply(self, subst):
        return CoroutineType(
            self.ytype.apply(subst),
            self.stype.apply(subst),
            self.rtype.apply(subst))
    def ftv(self):
        return set.union(self.ytype.ftv(), self.stype.ftv(), self.rtype.ftv())

class TupleType(Type):
    def __init__(self, ts):
        self.ts = ts
    def __str__(self):
        if len(self.ts) == 0:
            return 'unit'
        if len(self.ts) == 1:
            return str(self.ts[0])
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

        class Types:
            def __init__(this, vtype=None, *, rtype=None, ytype=None, stype=None, ctype=None):
                this.vtype = vtype or self.fresh()
                this.rtype = rtype or self.fresh()
                this.ytype = ytype or self.fresh()
                this.stype = stype or self.fresh()
                this.ctype = ctype or self.fresh()

            def but(this, vtype=None, *, rtype=None, ytype=None, stype=None, ctype=None):
                return Types(vtype=vtype or this.vtype,
                             rtype=rtype or this.rtype,
                             ytype=ytype or this.ytype,
                             stype=stype or this.stype,
                             ctype=ctype or this.ctype)

            @staticmethod
            def void(vtype=None, *, rtype=None, ytype=None, stype=None, ctype=None):
                return Types(vtype=vtype or Void,
                             rtype=rtype or Void,
                             ytype=ytype or Void,
                             stype=stype or Void,
                             ctype=ctype or Void)

        self.Types = Types

    ######

    def fresh(self):
        return TypeVariable(next(self.fresh_vars))

    def instantiate(self, scm):
        return scm.t.apply({tvar: self.fresh() for tvar in scm.tvars})

    def generalise(self, t):
        ftv = t.ftv() - set.union(*(ty.ftv() for ty in self.env.values()))
        return TypeScheme(list(ftv), t)

    def lookup_env(self, name, pos):
        scm = self.env.get(name, None)
        if scm is None:
            raise ApeSyntaxError(msg=f'Unbound variable: {name}', pos=pos)
        return self.instantiate(scm)

    def add_env(self, name, t):
        sub = solve({}, self.unifiers)
        self.env.update({name: ty.apply(sub) for name, ty in self.env.items()})
        self.env[name] = self.generalise(t.apply(sub))

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
        ast.type = t.vtype

    @infer.on(EmptyListExpression)
    def _(self, ast):
        tv = self.fresh()
        return self.Types(ListType(tv))

    @infer.on(EmptyDictExpression)
    def _(self, ast):
        tk = self.fresh()
        tv = self.fresh()
        return self.Types(DictType(tk, tv))

    @infer.on(EmptyTupleExpression)
    def _(self, ast):
        return self.Types(Unit)

    @infer.on(SetLiteral)
    def _(self, ast):
        tv = self.Types(self.fresh())
        ts = tv.but(SetType(tv.vtype))
        for expr in ast.exprs:
            if isinstance(expr, StarExpr):
                self.unify_all(ts, self.infer(expr.expr), expr.pos)
            else:
                self.unify_all(tv, self.infer(expr), expr.pos)
        return ts

    @infer.on(DictLiteral)
    def _(self, ast):
        tk, tv = self.fresh(), self.fresh()
        tD = self.Types(DictType(tk, tv))
        tK, tV = tD.but(tk), tD.but(tv)
        for expr in ast.exprs:
            if isinstance(expr, DictPair):
                self.unify_all(tK, self.infer(expr.key_expr), expr.key_expr.pos)
                self.unify_all(tV, self.infer(expr.value_expr), expr.value_expr.pos)
            else:
                self.unify_all(tD, self.infer(expr.expr), expr.expr.pos)
        return tD

    @infer.on(ListLiteral)
    def _(self, ast):
        tv = self.Types(self.fresh())
        ts = tv.but(ListType(tv.vtype))
        for expr in ast.exprs:
            if isinstance(expr, StarExpr):
                self.unify_all(ts, self.infer(expr.expr), expr.pos)
            else:
                self.unify_all(tv, self.infer(expr), expr.pos)
        return ts

    @infer.on(TupleLiteral)
    def _(self, ast):
        ts = [self.infer(expr) for expr in ast.exprs]
        tt = self.Types(TupleType([t.vtype for t in ts]))
        for t, expr in zip(ts, ast.exprs):
            self.unify_others(tt, t, expr.pos)

        return tt

    @infer.on(Lazy)
    def _(self, ast):
        #return self.Types(FunctionType(Unit, self.infer(ast)))
        raise NotImplementedError('Haven\'t figured out laziness yet')

    @infer.on(RaiseStatement)
    def _(self, ast):
        t1 = self.infer(ast.expr)
        self.unify(t1.vtype, t1.ctype, ast.expr.pos)

        if ast.original is not None:
            t2 = self.infer(ast.original)
            self.unify_all(t1, t2, ast.original.pos)

        return t1.but(vtype=self.fresh())

    @infer.on(NoneExpression)
    def _(self, ast):
        return self.Types(Unit)

    @infer.on(StringExpression)
    def _(self, ast):
        return self.Types(String)

    @infer.on(FloatExpression)
    def _(self, ast):
        return self.Types(Float)

    @infer.on(IntExpression)
    def _(self, ast):
        return self.Types(Int)

    @infer.on(IfElseExpr)
    def _(self, ast):
        t1 = self.infer(ast.cond)
        t2 = self.infer(ast.expr)
        t3 = self.infer(ast.alt)
        self.unify_all(t1, self.Types(Bool), ast.cond.pos)
        self.unify_all(t2, t3, ast.expr.pos)
        return t2

    @infer.on(ArithExpression)
    def _(self, ast):
        t1 = self.infer(ast.left)
        t2 = self.infer(ast.right)
        self.unify_others(t1, t2, ast.pos)

        tv = self.fresh()
        u1 = FunctionType(TupleType([t1.vtype, t2.vtype]), tv)
        u2 = self.instantiate(BUILTIN_OPERATORS[ast.op])
        self.unify(u1, u2, ast.pos)

        return t1.but(tv)

    @infer.on(ReturnStatement)
    def _(self, ast):
        t = self.infer(ast.expr)
        self.unify(t.vtype, t.rtype, ast.pos)
        return t

    @infer.on(PassStatement)
    def _(self, ast):
        return self.Types(Unit)

    @infer.on(IfElifElseStatement)
    def _(self, ast):
        tc = self.infer(ast.if_branch.cond)
        self.unify(tc.vtype, Bool, ast.pos)

        ts = self.infer(ast.if_branch.body)
        self.unify_others(tc, ts, ast.pos)

        for br in ast.elif_branches:
            c = self.infer(br.cond)
            self.unify_all(c, tc, br.cond.pos)

            s = self.infer(br.body)
            self.unify_all(s, ts, br.body.pos)

        if ast.else_branch is not None:
            s = self.infer(ast.else_branch.body)
            self.unify_all(s, ts, ast.else_branch.pos)

        return ts

    @infer.on(CallExpression)
    def _(self, ast):
        ta = self.infer(ast.atom)
        ts = [self.infer(arg) for arg in ast.args]
        for t in ts:
            self.unify_others(t, ta, ast.pos)

        tt = TupleType([t.vtype for t in ts])

        tv = self.fresh()
        self.unify(ta.vtype, FunctionType(tt, tv), ast.pos)
        return ta.but(tv)

    @infer.on(ChainedAssignment)
    def _(self, ast):
        assignees = ast.assignees[:-1]
        expr = ast.assignees[-1]

        te = self.infer(expr)
        ts = []
        for a in assignees:
            if isinstance(a, IdExpression):
                t = self.fresh()
                self.env[a.name] = TypeScheme([], t)
                ts.append((t, a.pos))
            else: raise ApeSyntaxError(pos=a.pos, msg=f'Cannot assign to {a.__class__.__name__}')

        for t, pos in ts:
            self.unify(te.vtype, t, pos)

        return te.but(Unit)

    @infer.on(Comparison)
    def _(self, ast):
        t1 = self.infer(ast.a)
        t2 = self.infer(ast.b)
        self.unify_all(t1, t2, ast.pos)
        return t1.but(Bool)

    @infer.on(IdExpression)
    def _(self, ast):
        return self.Types(self.lookup_env(ast.name, ast.pos))

    @infer.on(YieldExpression)
    def _(self, ast):
        t = self.infer(ast.expr)
        self.unify(t.vtype, t.ytype, ast.pos)
        return t.but(t.stype)

    @infer.on(Statements)
    def _(self, ast):
        t = self.Types()
        for i, stmt in enumerate(ast.stmts):
            if i == len(ast.stmts) - 1:
                self.unify_all(self.infer(stmt), t, stmt.pos)
            else:
                self.unify_all(self.infer(stmt), t.but(Unit), stmt.pos)
        return t

    @infer.on(LogicalOrExpressions)
    def _(self, ast):
        t = self.Types(Bool)
        for expr in ast.exprs:
            self.unify_all(self.infer(expr), t, expr.pos)
        return t

    def infer_params(self, params):
        # default parameters are evaluated at definition site
        tD = self.Types()
        ts = []
        env = {}
        for p in params:
            if isinstance(p, Param):
                tv = self.fresh()
                env[p.name] = TypeScheme([], tv)
                if p.annotation:
                    ta = self.parse_type(p.annotation)
                    self.unify(tv, ta, p.annotation.pos)
                if p.default:
                    td = self.infer(p.default)
                    self.unify(tv, td.vtype, p.default.pos)
                    self.unify_others(tD, td, p.default.pos)
                ts.append(tv)
            else: raise NotImplementedError(f'{p.__class__.__name__} is not supported yet')

        # can't have these in the environment when type-inferring defaults
        # so add the parameters to the environment at the end.
        self.env.update(env)
        return ts, tD

    def infer_function(self, ast):
        with self.subenv():
            params, default_effects = self.infer_params(ast.params)

            t = self.fresh()
            tb = self.infer(ast.body)
            self.unify(tb.vtype, t, ast.pos)
            self.unify(tb.rtype, t, ast.pos)

            if ast.body.is_gen():
                y, s = self.fresh(), self.fresh()
                self.unify(tb.ytype, y, ast.pos)
                self.unify(tb.stype, s, ast.pos)
                tr = CoroutineType(y, s, t)
            else:
                tr = t

            if ast.return_annotation:
                ta = self.parse_type(p.return_annotation)
                self.unify(tr, ta, p.return_annotation.pos)
            tf = FunctionType(TupleType(params), tr)
        return default_effects, tf

    @infer.on(FunctionExpression)
    def _(self, ast):
        default_effects, tf = self.infer_function(ast)
        return default_effects.but(tf)

    @infer.on(FunctionDefinition)
    def _(self, ast):
        default_effects, tf = self.infer_function(ast)
        self.add_env(ast.name, tf)
        return default_effects.but(Unit)

    @infer.on(InterfaceDefinition)
    def _(self, ast):
        with self.subenv():
            types = {}
            names = {}
            for decl in ast.body:
                if isinstance(decl, TypeDeclaration):
                    if len(decl.args) == 0:
                        pass# declaration of type
                    else:
                        pass# declaration of type *constructor*
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
                        law_type = self.Types.void(Bool)
                        self.unify_all(t, law_type, decl.pos)
                else: raise RuntimeError()

        self.type_env[ast.name.name] = InterfaceType(types, names)
        return self.Types(Unit)

    @contextlib.contextmanager
    def subenv(self):
        old, oldT = self.env, self.type_env
        self.env = self.env.new_child()
        self.type_env = self.type_env.new_child()
        yield
        self.env, self.type_env = old, oldT

    def unify_others(self, T1, T2, pos):
        self.unify(T1.rtype, T2.rtype, pos)
        self.unify(T1.ytype, T2.ytype, pos)
        self.unify(T1.stype, T2.stype, pos)
        self.unify(T1.ctype, T2.ctype, pos)

    def unify_all(self, T1, T2, pos):
        self.unify(T1.vtype, T2.vtype, pos)
        self.unify(T1.rtype, T2.rtype, pos)
        self.unify(T1.ytype, T2.ytype, pos)
        self.unify(T1.stype, T2.stype, pos)
        self.unify(T1.ctype, T2.ctype, pos)

    def unify(self, t1, t2, pos):
        self.unifiers.append(Constraint(t1, t2, pos))

def infer(ast):
    i = Infer()
    t = i.infer(ast)
    s = solve({}, i.unifiers)
    for k, v in i.env.items():
        print(k, v.apply(s))
    return back_substitute(ast, s)

def back_substitute(ast, subst):
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

