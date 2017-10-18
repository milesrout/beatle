import collections
import contextlib
import functools
import itertools

from astnodes import *
from utils import *

#⭑: 2b51

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

class AnyType(PrimitiveType):
    """This should not be instantiated ANYWHERE!
    
    This type exists to be unified with type variables that are required to be
    unconstrained."""
    def __str__(self):
        return 'any'
Any = AnyType()

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

class TypeCall(Type):
    def __init__(self, con, ts):
        self.con = con
        self.ts = ts
    def __str__(self):
        params = ', '.join(map(str, self.ts))
        return f'{self.con}[{params}]'
    def apply(self, subst):
        return TypeCall(self.con, [t.apply(subst) for t in self.ts])
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
    def kind(self):
        return Star

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
    def kind(self):
        return Star

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
    def kind(self):
        return Star

class SignatureType(Type):
    def __init__(self, types, names):
        self.types = types
        self.names = names
    def __str__(self):
        types = '\n'.join(f'\t\t\t\t{k}:{v}' for k,v in self.types.items())
        names = '\n'.join(f'\t\t\t\t{k}:{v}' for k,v in self.names.items())
        return f'signature(\n{types}\n{names}\n\t\t\t)'
    def kind(self):
        return Star

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
    def kind(self):
        return Star

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

class Nullary:
    def __init__(self, t):
        self.t = t
    def __str__(self):
        return str(self.t)
    def apply(self, subst):
        return Nullary(self.t.apply(subst))
    def ftv(self):
        return self.t.ftv()

class TypeConstructor:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
    def __str__(self):
        return f'{self.name}'
    def apply(self, subst):
        return self
    def ftv(self):
        return set()
    #def apply(self, subst):
    #    s = {k:v for k, v in subst.items() if k.WHAT not in self.args}
    #    return TypeConstructor(args, self.t.apply(s))
    #def ftv(self):
    #    return self.t.ftv()

class Kind:
    pass

class KindConstant(Kind):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def apply(self, subst):
        return self
    def fkv(self):
        return set()
Star = KindConstant('⭑')

class KindVariable(Kind):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def apply(self, subst):
        return subst.get(self.name, self)
    def fkv(self):
        return {self.name}

class ArrowKind(Kind):
    def __init__(self, ks, k):
        self.ks = ks
        self.k = k
    def __str__(self):
        ks = ', '.join(map(str, self.ks))
        if len(self.ks) == 1:
            return f'({ks} → {self.k})'
        return f'(({ks}) → {self.k})'
    def apply(self, subst):
        ks = [k.apply(subst) for k in self.ks]
        return ArrowKind(ks, self.k.apply(subst))
    def fkv(self):
        return set.union(self.k.fkv(), *[k.fkv() for k in self.ks])

BASE_ENVIRONMENT = {
    'print': TypeScheme([], FunctionType(String, Unit)),
    'str':   TypeScheme(['a'], FunctionType(TypeVariable('a'), String)),
    'set':   TypeScheme(['a'], FunctionType(Unit, SetType(TypeVariable('a')))),
    'cons':  TypeScheme(['a'], FunctionType(TupleType([TypeVariable('a'), ListType(TypeVariable('a'))]),
                                            ListType(TypeVariable('a')))),
    'RuntimeError': TypeScheme([], FunctionType(String, Error)),
}

BASE_TYPE_ENVIRONMENT = {
    'list':  TypeConstructor('list', 1),
    'int':   Nullary(Int),
    'float': Nullary(Float),
    'bool':  Nullary(Bool),
    'error': Nullary(Error),
}

generic_unary_op =  TypeScheme(['a'], FunctionType(TypeVariable('a'), TypeVariable('a')))
generic_binary_op = TypeScheme(['a'], FunctionType(TupleType([TypeVariable('a'), TypeVariable('a')]), TypeVariable('a')))

UNARY_OPERATORS = {
    'plus':  generic_unary_op,
    'minus': generic_unary_op,
}

BINARY_OPERATORS = {
    'plus':     generic_binary_op,
    'minus':    generic_binary_op,
    'asterisk': generic_binary_op,
    'at':       generic_binary_op,
    'div':      generic_binary_op,
    'mod':      generic_binary_op,
    'truediv':  generic_binary_op,
}

class Infer:
    def __init__(self):
        self.env = collections.ChainMap(BASE_ENVIRONMENT)
        self.type_env = collections.ChainMap(BASE_TYPE_ENVIRONMENT)
        self.unifiers = []
        self.kind_unifiers = []
        self.fresh_vars = (f'a{i}' for i in itertools.count(1))
        self.fresh_kind_vars = (f'k{i}' for i in itertools.count(1))

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

            @staticmethod
            def any(vtype=None, *, rtype=None, ytype=None, stype=None, ctype=None):
                return Types(vtype=vtype or Any,
                             rtype=rtype or Any,
                             ytype=ytype or Any,
                             stype=stype or Any,
                             ctype=ctype or Any)

        self.Types = Types

    def infer_error(self, ast):
        try:
            return ApeError(pos=ast.pos, msg='no overload found for {}'.format(ast.__class__))
        except:
            return ApeError(pos=0, msg='no overload found for {}'.format(ast.__class__))

    ######

    def fresh(self):
        return TypeVariable(next(self.fresh_vars))

    def fresh_kind(self):
        return KindVariable(next(self.fresh_kind_vars))

    def instantiate(self, scm):
        return scm.t.apply({tvar: self.fresh() for tvar in scm.tvars})

    def generalise(self, t):
        ftv = t.ftv() - set.union(*(ty.ftv() for ty in self.env.values()))
        return TypeScheme(list(ftv), t)

    def lookup_name(self, name, pos):
        scm = self.env.get(name, None)
        if scm is None:
            raise ApeSyntaxError(msg=f'Unbound variable: {name}', pos=pos)
        return self.instantiate(scm)

    def lookup_type_con(self, name, pos):
        con = self.type_env.get(name, None)
        if con is None:
            raise ApeSyntaxError(msg=f'Unbound type: {name}', pos=pos)
        if isinstance(con, Nullary):
            return con.t, Star
        return con, con.kind

    def solve(self):
        subst = solve({}, self.unifiers)
        self.env.update({name: scm.apply(subst) for name, scm in self.env.items()})
        return subst

    def add_name(self, name, t):
        subst = self.solve()
        self.env[name] = self.generalise(t.apply(subst))

    ######

    def parse_toplevel_type(self, ast):
        if not isinstance(ast, TypeForallExpression):
            raise ApeError(pos=ast.pos, msg='Unexpected type expression')
        with self.subenv():
            names = [tvar.name for tvar in ast.tvars]
            for name in names:
                self.type_env[name] = Nullary(TypeVariable(name))
            t, k = self.parse_type(ast.expr)
            self.unify_kind(k, Star, ast.pos)
            return TypeScheme(names, t)

    @overloadmethod(error_function=infer_error)
    def infer_kind(self):
        ...

    @infer_kind.on(TypeNameExpression)
    def _(self, ast):
        _, k = self.lookup_type_con(ast.name, ast.pos)
        return k

    @infer_kind.on(TypeTupleExpression)
    def _(self, ast):
        for expr in ast.exprs:
            self.unify_kind(self.infer_kind(expr), Star, ast.pos)
        return Star

    @infer_kind.on(TypeFunctionExpression)
    def _(self, ast):
        self.unify_kind(self.infer_kind(ast.t1), Star, ast.t1.pos)
        self.unify_kind(self.infer_kind(ast.t2), Star, ast.t2.pos)
        return Star

    @infer_kind.on(TypeCallExpression)
    def _(self, ast):
        kc = self.infer_kind(ast.atom)
        kt = [self.infer_kind(t) for t in ast.args]
        kv = self.fresh_kind()
        self.unify_kind(kc, ArrowKind(kt, kv), ast.pos)
        return kv

    @overloadmethod(error_function=infer_error)
    def infer_type(self):
        ...

    @infer_type.on(TypeNameExpression)
    def _(self, ast):
        t, _ = self.lookup_type_con(ast.name, ast.pos)
        return t

    @infer_type.on(TypeTupleExpression)
    def _(self, ast):
        return TupleType([self.infer_type(expr) for expr in ast.exprs])

    @infer_type.on(TypeFunctionExpression)
    def _(self, ast):
        return FunctionType(self.infer_type(ast.t1), self.infer_type(ast.t2))

    @infer_type.on(TypeCallExpression)
    def _(self, ast):
        tc = self.infer_type(ast.atom)
        ts = [self.infer_type(t) for t in ast.args]
        return TypeCall(tc, ts)

    def parse_type(self, ast):
        t, k = self.infer_type(ast), self.infer_kind(ast)
        return t, k

    ######

    @overloadmethod(use_as_modifier=True, error_function=infer_error)
    def infer(self, ast, t):
        ast.type = t.vtype

    @infer.on(EmptyListExpression)
    def _(self, ast):
        tv = self.fresh()
        return self.Types.any(ListType(tv))

    @infer.on(EmptyDictExpression)
    def _(self, ast):
        tk = self.fresh()
        tv = self.fresh()
        return self.Types.any(DictType(tk, tv))

    @infer.on(EmptyTupleExpression)
    def _(self, ast):
        return self.Types.any(Unit)

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
        return self.Types.any(Unit)

    @infer.on(StringExpression)
    def _(self, ast):
        return self.Types.any(String)

    @infer.on(FloatExpression)
    def _(self, ast):
        return self.Types.any(Float)

    @infer.on(IntExpression)
    def _(self, ast):
        return self.Types.any(Int)

    @infer.on(IfElseExpr)
    def _(self, ast):
        t1 = self.infer(ast.cond)
        t2 = self.infer(ast.expr)
        t3 = self.infer(ast.alt)
        self.unify_all(t1, self.Types(Bool), ast.cond.pos)
        self.unify_others(t1, t2, ast.expr.pos)
        self.unify_others(t1, t3, ast.expr.pos)
        self.unify(t2.vtype, t3.vtype, ast.expr.pos)
        return t2

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

    @infer.on(UnaryExpression)
    def _(self, ast):
        t = self.infer(ast.expr)

        tv = self.fresh()
        u1 = FunctionType(t.vtype, tv)
        u2 = self.instantiate(UNARY_OPERATORS[ast.op])
        self.unify(u1, u2, ast.pos)

        return t.but(tv)

    @infer.on(ArithExpression)
    def _(self, ast):
        t1 = self.infer(ast.left)
        t2 = self.infer(ast.right)
        self.unify_others(t1, t2, ast.pos)

        tv = self.fresh()
        u1 = FunctionType(TupleType([t1.vtype, t2.vtype]), tv)
        u2 = self.instantiate(BINARY_OPERATORS[ast.op])
        self.unify(u1, u2, ast.pos)

        return t1.but(tv)

    @infer.on(ReturnStatement)
    def _(self, ast):
        t = self.infer(ast.expr)
        self.unify(t.vtype, t.rtype, ast.pos)
        return t

    @infer.on(PassStatement)
    def _(self, ast):
        return self.Types.any(Unit)

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
                self.add_name(a.name, t)
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
        return self.Types.any(self.lookup_name(ast.name, ast.pos))

    @infer.on(YieldExpression)
    def _(self, ast):
        t = self.infer(ast.expr)
        self.unify(t.vtype, t.ytype, ast.pos)
        return t.but(t.stype)

    @infer.on(AnnotatedAssignment)
    def _(self, ast):
        t, k = self.parse_type(ast.annotation)
        raise

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
                    ta, ka = self.parse_type(p.annotation)
                    self.unify(tv, ta, p.annotation.pos)
                    self.unify_kind(ka, Star, p.annotation.pos)
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
                ta, ka = self.parse_type(ast.return_annotation)
                self.unify(tr, ta, ast.return_annotation.pos)
                self.unify_kind(ka, Star, ast.return_annotation.pos)
            tf = FunctionType(TupleType(params), tr)
        return default_effects, tf

    @infer.on(FunctionExpression)
    def _(self, ast):
        default_effects, tf = self.infer_function(ast)
        return default_effects.but(tf)

    @infer.on(FunctionDefinition)
    def _(self, ast):
        default_effects, tf = self.infer_function(ast)
        self.add_name(ast.name, tf)
        return default_effects.but(Unit)

    @infer.on(SignatureDefinition)
    def _(self, ast):
        with self.subenv():
            types = {}
            names = {}
            for decl in ast.body:
                if isinstance(decl, TypeDeclaration):
                    name = decl.name.name
                    if len(decl.args) == 0:
                        types[name] = self.type_env[name] = Nullary(self.fresh())
                    else:
                        types[name] = self.type_env[name] = TypeConstructor(name, ArrowKind([Star for _ in decl.args], Star))
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
            self.type_env[ast.name.name] = Nullary(SignatureType(types, names))
        return self.Types.any(Unit)

    def print_env(self):
        print('self.type_env:')
        for k, v in self.type_env.items():
            print(f'\t{k!s:<15} {v!s}')
        print('self.env:')
        for k, v in self.env.items():
            print(f'\t{k!s:<15} {v!s}')

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
        if t1 is not Any and t2 is not Any:
            self.unifiers.append(Constraint(t1, t2, pos))

    def unify_kind(self, k1, k2, pos):
        self.kind_unifiers.append(KindConstraint(k1, k2, pos))

######

def bind_kind(name, k, pos):
    if isinstance(k, KindVariable) and k.name == name:
        return {}
    if name in k.fkv():
        raise ApeUnificationError(msg='infinite type!', pos=pos)
    return {name: k}

def bind(tvar, t, pos):
    if isinstance(t, TypeVariable) and t.tvar == tvar:
        return {}
    if tvar in t.ftv():
        raise ApeUnificationError(msg='infinite type!', pos=pos)
    return {tvar: t}

def unifies_kind(k1, k2, pos):
    if k1 == k2:
        return {}

    if isinstance(k1, KindVariable):
        return bind_kind(k1.name, k2, pos)
    if isinstance(k2, KindVariable):
        return bind_kind(k2.name, k1, pos)

    if isinstance(k1, ArrowKind) and isinstance(k2, ArrowKind):
        if len(k1.ks) == len(k2.ks):
            return unify_many_kinds([k1.k, *k1.ks], [k2.k, *k2.ks], pos)

    raise ApeUnificationError(pos=pos, msg=f'Cannot unify {k1} and {k2}')

def unifies(t1, t2, pos):
    if isinstance(t1, TypeConstructor) and isinstance(t2, TypeConstructor):
        if t1.name == t2.name:
            return unifies_kind(t1.kind, t2.kind, pos)
        raise ApeUnificationError(pos=pos, msg=f'Cannot unify distinct type constructors {t1.name} and {t2.name}')

    if not isinstance(t1, Type) or not isinstance(t2, Type):
        raise ApeUnificationError(pos=pos, msg=f'Can only unify types, not {t1} and {t2}')

    if t1 == t2:
        return {}

    if isinstance(t1, TypeVariable):
        return bind(t1.tvar, t2, pos)
    if isinstance(t2, TypeVariable):
        return bind(t2.tvar, t1, pos)

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
    if isinstance(t1, TypeCall) and isinstance(t2, TypeCall):
        return unify_many([t1.con, *t1.ts], [t2.con, *t2.ts], pos)
    raise ApeUnificationError(pos=pos, msg=f'Cannot unify {t1} and {t2}')

def unify_many_kinds(k1x, k2x, pos):
    if len(k1x) == 0 and len(k2x) == 0:
        return {}

    [k1, *k1s], [k2, *k2s] = k1x, k2x
    su1 = unifies_kind(k1, k2, pos)
    su2 = unify_many_kinds(
        [t.apply(su1) for t in k1s],
        [t.apply(su1) for t in k2s], pos)
    return compose_subst(su2, su1)

def unify_many(t1x, t2x, pos):
    if len(t1x) == 0 and len(t2x) == 0:
        return {}

    [t1, *t1s], [t2, *t2s] = t1x, t2x
    su1 = unifies(t1, t2, pos)
    su2 = unify_many(
        [t.apply(su1) for t in t1s],
        [t.apply(su1) for t in t2s], pos)
    return compose_subst(su2, su1)

class KindConstraint:
    def __init__(self, k1, k2, pos):
        self.k1 = k1
        self.k2 = k2
        self.pos = pos
    def __repr__(self):
        uni = f'{self.k1} ~ {self.k2}'
        return f'{uni:<25} at {self.pos}'
    def apply(self, subst):
        return KindConstraint(self.k1.apply(subst), self.k2.apply(subst), self.pos)
    def fkv(self):
        return set.union(self.k1.ftv(), self.k2.ftv())

class Constraint:
    def __init__(self, t1, t2, pos):
        self.t1 = t1
        self.t2 = t2
        self.pos = pos
    def __repr__(self):
        uni = f'{self.t1} ~ {self.t2}'
        return f'{uni:<25} at {self.pos}'
    def apply(self, subst):
        return Constraint(self.t1.apply(subst), self.t2.apply(subst), self.pos)
    def ftv(self):
        return set.union(self.t1.ftv(), self.t2.ftv())

def compose_subst(su1, su2):
    return {**su1, **su2}

def solve_kind(su, cs):
    if len(cs) == 0:
        return su

    [c, *cs0] = cs
    su1 = unifies_kind(c.k1, c.k2, c.pos)

    return solve_kind(compose_subst(su1, su), [c.apply(su1) for c in cs0])

def solve(su, cs):
    if len(cs) == 0:
        return su

    [c, *cs0] = cs
    su1 = unifies(c.t1, c.t2, c.pos)

    return solve(compose_subst(su1, su), [c.apply(su1) for c in cs0])

######

def infer(ast):
    i = Infer()
    t = i.infer(ast)
    s = solve({}, i.unifiers)
    s1 = solve_kind({}, i.kind_unifiers)
    return ast
    #print(len(i.unifiers), len(i.kind_unifiers))
    #for uni in i.unifiers:
    #    print(uni)
    #for uni in i.kind_unifiers:
    #    print(uni)
    #i.solve()
    #for k, v in i.env.items():
    #    print(k, v)
    #return back_substitute(ast, s)

def back_substitute(ast, subst):
    return ast

