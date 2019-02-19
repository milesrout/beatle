import collections
import contextlib
import itertools

import typednodes as T
import cstnodes as E
from is_gen import is_gen
from utils import Type, Ast, ApeInternalError, ApeNotImplementedError, ApeTypeError, overloadmethod, unzip

#⭑: 2b51

class ApeInferenceError(ApeTypeError):
    pass

class ApeUnificationError(ApeTypeError):
    pass

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
        types = '\n'.join(f'\t\t\t\t{k}:{v}' for k, v in self.types.items())
        names = '\n'.join(f'\t\t\t\t{k}:{v}' for k, v in self.names.items())
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

printout_depth = 0

@contextlib.contextmanager
def depth():
    global printout_depth
    printout_depth += 1
    yield
    printout_depth -= 1

def printout(*args, **kwds):
    print(printout_depth * ' ', *args, **kwds)

class Namespace:
    def __init__(self, env):
        if isinstance(env, Namespace):
            raise
        self.env = env

    def printout(self):
        for k, v in self.env.items():
            if isinstance(v, Object):
                printout(f'{k!s:<15} {v.value!s}')
            elif isinstance(v, Namespace):
                printout(f'{k!s}:')
                with depth():
                    v.printout()
            else:
                printout(f'{k!s:<15} {v!s}')

    def __getitem__(self, key):
        if isinstance(key, E.DottedNameExpression):
            raise
        return self.env[key]

    def __setitem__(self, key, value):
        if isinstance(key, E.DottedNameExpression):
            raise
        self.env[key] = value

    def get(self, key, alt=None):
        if isinstance(key, E.DottedNameExpression):
            raise
        return self.env.get(key, alt)

    def update(self, others):
        self.env.update(others)

class Object:
    def __init__(self, value):
        self.value = value

BASE_ENVIRONMENT = {
    # Want to add 'next', 'send', 'throw', etc. for generators.
    # But we need linear types before we can do that. Then we can say
    # iter:   Iterable(y,r)         ~> r + (y, Iterator(y,r)),
    # next:   Iterator(y,r)         ~> r + (y, Iterator(y,r)),
    # start:  Generator(y,s,r)      ~> r + (y, Coroutine(y,s,r)), and
    # send:   (s, Coroutine(y,s,r)) ~> r + (y, Coroutine(y,s,r)),
    # where ~> is the LinearArrowType, Iterable(y,r) = Generator(y,(),r),
    # and Iterator(y,r) = Coroutine(y,(),r).
    'print': Object(TypeScheme([], FunctionType(Any, Unit))),
    'str':   Object(TypeScheme(['a'], FunctionType(TypeVariable('a'), String))),
    'set':   Object(TypeScheme(['a'], FunctionType(Unit, SetType(TypeVariable('a'))))),
    'cons':  Object(TypeScheme(['a'], FunctionType(TupleType([TypeVariable('a'), ListType(TypeVariable('a'))]),
                                                   ListType(TypeVariable('a'))))),
    'RuntimeError': Object(TypeScheme([], FunctionType(String, Error))),
}

BASE_TYPE_ENVIRONMENT = {
    'list':  Object(TypeConstructor('list', 1)),
    'int':   Object(Nullary(Int)),
    'float': Object(Nullary(Float)),
    'bool':  Object(Nullary(Bool)),
    'error': Object(Nullary(Error)),
}

generic_unary_op = TypeScheme(['a'], FunctionType(TypeVariable('a'), TypeVariable('a')))
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

class TypeChecker:
    def __init__(self, input_text):
        self.env = Namespace(collections.ChainMap(BASE_ENVIRONMENT).new_child())
        self.type_env = Namespace(collections.ChainMap(BASE_TYPE_ENVIRONMENT).new_child())
        self.subst = {}
        self.unifiers = []
        self.kind_unifiers = []
        self.fresh_vars = (f'a{i}' for i in itertools.count(1))
        self.fresh_kind_vars = (f'k{i}' for i in itertools.count(1))
        self.asts = []

        class Error:
            def __init__(this, msg, pos):
                super().__init__(msg, pos, self.input_text)

        class Types:
            """A collection of the different types an expression can have

            vtype: the type of the _V_alue of the expression
            rtype: the type that an expression may early _R_eturn while being evaluated
            ytype: the type that an expression may _Y_ield while being evaluated
            stype: the type that an expression may be _S_ent while being evaluated
            ctype: the type of _C_ondition that an expression may throw while being evaluated
            """
            def __init__(this, vtype=None, *, rtype=None, ytype=None, stype=None, ctype=None):
                this.vtype = vtype or self.fresh()
                this.rtype = rtype or self.fresh()
                this.ytype = ytype or self.fresh()
                this.stype = stype or self.fresh()
                this.ctype = ctype or self.fresh()

            def __repr__(this):
                return f'Types({this.vtype}, {this.rtype}, {this.ytype}, {this.stype}, {this.ctype})'

            def apply(this, subst):
                vtype = this.vtype.apply(subst) if this.vtype is not None else None
                rtype = this.rtype.apply(subst) if this.rtype is not None else None
                ytype = this.ytype.apply(subst) if this.ytype is not None else None
                stype = this.stype.apply(subst) if this.stype is not None else None
                ctype = this.ctype.apply(subst) if this.ctype is not None else None
                return Types(vtype=vtype,
                             rtype=rtype,
                             ytype=ytype,
                             stype=stype,
                             ctype=ctype)

            def also(this, vtype=None, *, rtype=None, ytype=None, stype=None, ctype=None):
                new = this.but(vtype=vtype, rtype=rtype, ytype=ytype, stype=stype, ctype=ctype)
                self.unify_all(this, new)
                return new

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

        self.Error = Error
        self.Types = Types

    ######

    def fresh(self):
        return TypeVariable(next(self.fresh_vars))

    def fresh_kind(self):
        return KindVariable(next(self.fresh_kind_vars))

    def instantiate(self, scm):
        return scm.t.apply({tvar: self.fresh() for tvar in scm.tvars})

    def generalise(self, t):
        ftv = t.ftv() - set.union(*(ty.value.ftv() for ty in self.env.env.values()))
        return TypeScheme(list(ftv), t)

    def lookup_name(self, name, pos):
        scm = self.env.get(name, None)
        if scm is None:
            raise ApeTypeError(msg=f'Unbound variable: {name}', pos=pos)
        return self.instantiate(scm.value)

    def lookup_type_con(self, name, pos):
        if isinstance(name, E.DottedNameExpression):
            if len(name.parts) == 1:
                name = name.parts[0]
            else:
                raise NotImplementedError()
        if isinstance(name, E.IdExpression):
            name = name.name
        con = self.type_env.get(name, None)
        if con is None:
            raise ApeTypeError(msg=f'Unbound type: {name}', pos=pos)
        if isinstance(con.value, Nullary):
            return con.value.t, Star
        return con.value, con.value.kind

    def solve(self):
        self.subst = solve(self.subst, self.unifiers)
        self.env.update({name: Object(scm.value.apply(self.subst)) for name, scm in self.env.env.items()})
        return self.subst

    def add_name(self, name, t):
        subst = self.solve()
        self.env[name] = Object(self.generalise(t.apply(subst)))

    def add_type_namespace(self, name, env):
        environment = self.type_env
        for part in name.parts[:-1]:
            environment[part.name] = Namespace({})
            environment = environment[part.name]
        environment[name.parts[-1].name] = Namespace(env)

    def add_namespace(self, name, env):
        environment = self.env
        for part in name.parts[:-1]:
            environment[part.name] = Namespace({})
            environment = environment[part.name]
        environment[name.parts[-1].name] = Namespace(env)

    ######

    def infer_error(self, ast):
        try:
            return ApeInternalError(pos=ast.pos, msg='no overload found for {}'.format(ast.__class__))
        except Exception:
            return ApeInternalError(pos=0, msg='no overload found for {}'.format(ast.__class__))

    ######

    def parse_toplevel_type(self, ast):
        if not isinstance(ast, E.TypeForallExpression):
            raise ApeInternalError(pos=ast.pos, msg='Unexpected type expression')
        with self.subenv():
            names = [tvar.name for tvar in ast.tvars]
            for name in names:
                self.type_env[name] = Object(Nullary(TypeVariable(name)))
            t, k = self.parse_type(ast.expr)
            self.unify_kind(k, Star, ast.pos)
            return TypeScheme(names, t)

    @overloadmethod(error_function=infer_error)
    def infer_kind(self):
        ...

    @infer_kind.on(E.TypeNameExpression)
    def infer_kind_TypeNameExpression(self, ast):
        _, k = self.lookup_type_con(ast.name, ast.pos)
        return k

    @infer_kind.on(E.TypeTupleExpression)
    def infer_kind_TypeTupleExpression(self, ast):
        for expr in ast.exprs:
            self.unify_kind(self.infer_kind(expr), Star, ast.pos)
        return Star

    @infer_kind.on(E.TypeFunctionExpression)
    def infer_kind_TypeFunctionExpression(self, ast):
        self.unify_kind(self.infer_kind(ast.t1), Star, ast.t1.pos)
        self.unify_kind(self.infer_kind(ast.t2), Star, ast.t2.pos)
        return Star

    @infer_kind.on(E.TypeCallExpression)
    def infer_kind_TypeCallExpression(self, ast):
        kc = self.infer_kind(ast.atom)
        kt = [self.infer_kind(t) for t in ast.args]
        kv = self.fresh_kind()
        self.unify_kind(kc, ArrowKind(kt, kv), ast.pos)
        return kv

    @overloadmethod(error_function=infer_error)
    def infer_type(self):
        ...

    @infer_type.on(E.TypeNameExpression)
    def infer_type_TypeNameExpression(self, ast):
        t, _ = self.lookup_type_con(ast.name, ast.pos)
        return t

    @infer_type.on(E.TypeTupleExpression)
    def infer_type_TypeTupleExpression(self, ast):
        return TupleType([self.infer_type(expr) for expr in ast.exprs])

    @infer_type.on(E.TypeFunctionExpression)
    def infer_type_TypeFunctionExpression(self, ast):
        return FunctionType(self.infer_type(ast.t1), self.infer_type(ast.t2))

    @infer_type.on(E.TypeCallExpression)
    def infer_type_TypeCallExpression(self, ast):
        tc = self.infer_type(ast.atom)
        ts = [self.infer_type(t) for t in ast.args]
        return TypeCall(tc, ts)

    def parse_type(self, ast):
        t, k = self.infer_type(ast), self.infer_kind(ast)
        return t, k

    ######

    @overloadmethod(use_as_wrapper=True, error_function=infer_error)
    def infer(self, original, ast_and_type):
        if isinstance(ast_and_type, self.Types):
            print('You forgot to return the typed node as well as the type itself:', ast_and_type, original)
            raise
        ast, t = ast_and_type
        ret = Ast(ast, t, original.pos)
        self.asts.append((ret, self.env))
        return ret, t

    @infer.on(E.EmptyListExpression)
    def infer_EmptyListExpression(self, ast):
        tv = self.fresh()
        return T.EmptyList(), self.Types(ListType(tv))

    @infer.on(E.EmptyDictExpression)
    def infer_EmptyDictExpression(self, ast):
        tk = self.fresh()
        tv = self.fresh()
        return T.EmptyDict(), self.Types(DictType(tk, tv))

    @infer.on(E.EmptyTupleExpression)
    def infer_EmptyTupleExpression(self, ast):
        return T.EmptyTuple(), self.Types(Unit)

    @infer.on(E.SetLiteral)
    def infer_SetLiteral(self, ast):
        v = self.fresh()
        tS = self.Types(SetType(v))
        tV = tS.but(v)

        exprs = []
        for expr in ast.exprs:
            if isinstance(expr, E.StarExpr):
                es, ts = self.infer(expr.expr)
                self.unify_all(tS, ts, expr.pos)
                exprs.append(T.Star(es, expr.pos))
            else:
                ev, tv = self.infer(expr)
                self.unify_all(tV, tv, expr.pos)
                exprs.append(ev)

        return T.SetLit(exprs, ast.pos), tS

    @infer.on(E.DictLiteral)
    def infer_DictLiteral(self, ast):
        k, v = self.fresh(), self.fresh()
        tD = self.Types(DictType(k, v))
        tK, tV = tD.but(k), tD.but(v)

        exprs = []
        for expr in ast.exprs:
            if isinstance(expr, E.DictPair):
                ek, tk = self.infer(expr.key_expr)
                self.unify_all(tK, tk, expr.key_expr.pos)
                ev, tv = self.infer(expr.value_expr)
                self.unify_all(tV, tv, expr.value_expr.pos)
                exprs.append(T.DictKV(ek, ev, expr.pos))
            elif isinstance(expr, E.StarStarExpr):
                ee, te = self.infer(expr.expr)
                self.unify_all(tD, te, expr.expr.pos)
                exprs.append(T.StarStar(ee, expr.pos))
            else:
                raise ApeInternalError(pos=ast.pos, msg=f'Unexpected {expr.__class__.__name__} in {ast.__class__.__name__}')

        return T.DictLit(exprs, ast.pos), tD

    @infer.on(E.ListLiteral)
    def infer_ListLiteral(self, ast):
        v = self.fresh()
        tL = self.Types(ListType(v))
        tV = tL.but(v)

        exprs = []
        for expr in ast.exprs:
            if isinstance(expr, E.StarExpr):
                el, tl = self.infer(expr.expr)
                self.unify_all(tL, tl, expr.pos)
                exprs.append(T.Star(el, expr.pos))
            else:
                ev, tv = self.infer(expr)
                self.unify_all(tV, tv, [ast.pos, expr.pos])
                exprs.append(ev)

        return T.ListLit(exprs, ast.pos), tL

    @infer.on(E.TupleLiteral)
    def infer_TupleLiteral(self, ast):
        ets = [self.infer(expr) for expr in ast.exprs]
        es, ts = unzip(ets) if len(ets) > 0 else [], []

        tt = self.Types(TupleType([t.vtype for t in ts]))
        for e, t in ets:
            self.unify_others(tt, t, e.pos)

        return T.Tuple(es), tt

    @infer.on(E.RaiseStatement)
    def infer_RaiseStatement(self, ast):
        e1, t1 = self.infer(ast.expr)
        self.unify(t1.vtype, t1.ctype, ast.expr.pos)

        if ast.original is not None:
            e2, t2 = self.infer(ast.original)
            self.unify_all(t1, t2, ast.original.pos)
        else:
            e2 = None

        return T.Raise(e1, e2, ast.pos), t1.but(vtype=self.fresh())

    @infer.on(E.NoneExpression)
    def infer_NoneExpression(self, ast):
        return T.NoneExpr(ast.pos), self.Types(Unit)

    @infer.on(E.StringExpression)
    def infer_StringExpression(self, ast):
        return T.String(ast.unparsed, ast.pos), self.Types(String)

    @infer.on(E.FloatExpression)
    def infer_FloatExpression(self, ast):
        return T.Float(ast.format, ast.value, ast.pos), self.Types(Float)

    @infer.on(E.IntExpression)
    def infer_IntExpression(self, ast):
        return T.Int(ast.base, ast.value, ast.pos), self.Types(Int)

    @infer.on(E.IfElseExpr)
    def infer_IfElseExpr(self, ast):
        t1 = self.infer(ast.cond)
        t2 = self.infer(ast.expr)
        t3 = self.infer(ast.alt)
        self.unify_all(t1, self.Types(Bool), ast.cond.pos)
        self.unify_others(t1, t2, ast.expr.pos)
        self.unify_others(t1, t3, ast.expr.pos)
        self.unify(t2.vtype, t3.vtype, ast.expr.pos)
        return t2

    @infer.on(E.IfElifElseStatement)
    def infer_IfElifElseStatement(self, ast):
        eic, tic = self.infer(ast.if_branch.cond)
        self.unify(tic.vtype, Bool, ast.pos)

        eib, tib = self.infer(ast.if_branch.body)
        self.unify_others(tic, tib, ast.pos)

        elifs = []
        for br in ast.elif_branches:
            eeic, teic = self.infer(br.cond)
            self.unify_all(teic, tic, br.cond.pos)

            eeib, teib = self.infer(br.body)
            self.unify_all(teib, tib, br.body.pos)

            elifs.append(T.ElifBranch(eeic, eeib, br.pos))

        if ast.else_branch is not None:
            eeb, teb = self.infer(ast.else_branch.body)
            self.unify_all(teb, tib, ast.else_branch.pos)
            eelse = T.ElseBranch(eeb, ast.else_branch.pos)
        else:
            eelse = None

        return T.IfElifElse(T.IfBranch(eic, eib, ast.if_branch.pos), elifs, eelse, ast.pos), tib

    @infer.on(E.UnaryExpression)
    def infer_UnaryExpression(self, ast):
        e, t = self.infer(ast.expr)

        tv = self.fresh()
        u1 = FunctionType(t.vtype, tv)
        u2 = self.instantiate(UNARY_OPERATORS[ast.op])
        self.unify(u1, u2, ast.pos)

        return T.Unary(ast.op, e), t.but(tv)

    @infer.on(E.ArithExpression)
    def infer_ArithExpression(self, ast):
        e1, t1 = self.infer(ast.left)
        e2, t2 = self.infer(ast.right)
        self.unify_others(t1, t2, ast.pos)

        tv = self.fresh()
        u1 = FunctionType(TupleType([t1.vtype, t2.vtype]), tv)
        u2 = self.instantiate(BINARY_OPERATORS[ast.op])
        self.unify(u1, u2, ast.pos)

        return T.Arith(ast.op, e1, e2), t1.but(tv)

    @infer.on(E.ReturnStatement)
    def infer_ReturnStatement(self, ast):
        e, t = self.infer(ast.expr)
        self.unify(t.vtype, t.rtype, ast.pos)
        return T.Return(e, ast.pos), t

    @infer.on(E.PassStatement)
    def infer_PassStatement(self, ast):
        return T.Pass(ast.pos), self.Types(Unit)

    @infer.on(E.CallExpression)
    def infer_CallExpression(self, ast):
        ef, tf = self.infer(ast.atom)

        es, ts = [], []
        for arg in ast.args:
            ea, ta = self.infer(arg)
            self.unify_others(tf, ta, ast.pos)
            es.append(ea)
            ts.append(ta)

        tt = TupleType([t.vtype for t in ts])
        tv = self.fresh()
        self.unify(tf.vtype, FunctionType(tt, tv), ast.pos)
        return T.Call(ef, es), tf.but(tv)

    @infer.on(E.ChainedAssignment)
    def infer_ChainedAssignment(self, ast):
        assignees = ast.assignees[:-1]
        expr = ast.assignees[-1]

        ee, te = self.infer(expr)
        targets = []
        for a in assignees:
            if isinstance(a, E.IdExpression):
                t = self.fresh()
                self.add_name(a.name, t)
                self.unify(te.vtype, t, a.pos)
                targets.append(a.name)
            else:
                raise ApeTypeError(pos=a.pos, msg=f'Cannot assign to {a.__class__.__name__}')

        return T.Assignment(targets, ee), te.but(Unit)

    @infer.on(E.Comparison)
    def infer_Comparison(self, ast):
        e1, t1 = self.infer(ast.a)
        e2, t2 = self.infer(ast.b)
        self.unify_all(t1, t2, ast.pos)
        return T.Comparison(ast.op, e1, e2), t1.but(Bool)

    @infer.on(E.IdExpression)
    def infer_IdExpression(self, ast):
        return T.Id(ast.name, ast.pos), self.Types(self.lookup_name(ast.name, ast.pos))

    @infer.on(E.TrueExpression)
    def infer_TrueExpression(self, ast):
        return T.Bool(True, ast.pos), self.Types(Bool)

    @infer.on(E.FalseExpression)
    def infer_FalseExpression(self, ast):
        return T.Bool(False, ast.pos), self.Types(Bool)

    @infer.on(E.YieldFromExpression)
    def infer_YieldFromExpression(self, ast):
        t = self.Types()

        tc = CoroutineType(self.fresh(), self.fresh(), self.fresh())
        ef, tf = self.infer(ast.expr)
        self.unify(tc, tf.vtype)

        self.unify(t.vtype, tc.rtype)
        self.unify(t.ytype, tc.ytype)
        self.unify(t.stype, tc.stype)
        self.unify(t.rtype, tf.rtype)
        self.unify(t.ctype, tf.ctype)

        # yield from x should opaquely forward things sent to *the current function* on to *whatever we're yielding from*.
        # and forwarding things yielded by *whatever we're yielding from* to *whatever we're yielding to*.
        # stype=stype
        # ytype=ytype
        #     ?=?

    @infer.on(E.YieldExpression)
    def infer_YieldExpression(self, ast):
        e, t = self.infer(ast.expr)
        self.unify(t.vtype, t.ytype, ast.pos)
        return T.Yield(e), t.but(t.stype)

    @infer.on(E.AnnotatedAssignment)
    def infer_AnnotatedAssignment(self, ast):
        t, k = self.parse_type(ast.annotation)
        raise

    @infer.on(E.Statements)
    def infer_Statements(self, ast):
        t = self.Types()
        exprs = []
        for i, stmt in enumerate(ast.stmts):
            expr, typ = self.infer(stmt)
            if i == len(ast.stmts) - 1:
                self.unify_all(typ, t, stmt.pos)
            else:
                self.unify_all(typ, t.but(Unit), stmt.pos)
            exprs.append(expr)
        return T.Statements(exprs), t

    @infer.on(E.LogicalAndExpressions)
    def infer_LogicalAndExpressions(self, ast):
        t = self.Types(Bool)
        es = []
        for expr in ast.exprs:
            ee, te = self.infer(expr)
            self.unify_all(te, t, expr.pos)
            es.append(ee)
        return T.LogicalAnd(es), t

    @infer.on(E.LogicalOrExpressions)
    def infer_LogicalOrExpressions(self, ast):
        t = self.Types(Bool)
        es = []
        for expr in ast.exprs:
            ee, te = self.infer(expr)
            self.unify_all(te, t, expr.pos)
            es.append(ee)
        return T.LogicalOr(es), t

    # Recall that parameter default values are evaluated at definition time in
    # Python. This is also true in Beatle.
    #
    # Thus:
    #  - the v-types of the default value expressions must unify with the types
    #    of the parameters.
    #  - the other types of the default value expressions must unify with the
    #    other types of the *function call definition expression*.
    def infer_params(self, params):
        defaults = self.Types()
        exprs = []
        types = []
        for p in params:
            if isinstance(p, E.Param):
                tv = self.fresh()
                if p.annotation is not None:
                    tann, kann = self.parse_type(p.annotation)
                    self.unify(tv, tann, p.annotation.pos)
                    self.unify_kind(kann, Star, p.annotation.pos)
                if p.default is not None:
                    edef, tdef = self.infer(p.default)
                    self.unify(tv, tdef.vtype, p.default.pos)
                    self.unify_others(defaults, tdef, p.default.pos)
                    exprs.append(edef)
                else:
                    exprs.append(None)
                types.append(tv)
                self.add_name(p.name.name, tv)
            #elif isinstance(p, StarVarParams):
            #    pass
            #elif isinstance(p, StarStarKwParams):
            #    pass
            else:
                raise ApeNotImplementedError(msg=f'{p.__class__.__name__} is not supported yet', pos=p.pos)
        return TupleType(types), exprs, defaults

    def infer_function(self, ast):
        with self.subenv():
            tt, exprs, tD = self.infer_params(ast.params)

            t = self.fresh()
            eb, tb = self.infer(ast.body)
            self.unify(tb.vtype, t, ast.pos)
            self.unify(tb.rtype, t, ast.pos)

            if is_gen(ast.body):
                y, s = self.fresh(), self.fresh()
                self.unify(tb.ytype, y, ast.pos)
                self.unify(tb.stype, s, ast.pos)
                tr = CoroutineType(y, s, t)
            else:
                self.unify(tb.ytype, Void, ast.pos)
                self.unify(tb.stype, Void, ast.pos)
                tr = t

            if ast.return_annotation:
                tann, kann = self.parse_type(ast.return_annotation)
                self.unify(tr, tann, ast.return_annotation.pos)
                self.unify_kind(kann, Star, ast.return_annotation.pos)

            tf = FunctionType(tt, tr)
        return exprs, eb, tf, tD

    @infer.on(E.FunctionExpression)
    def infer_FunctionExpression(self, ast):
        eparams, ebody, tf, tD = self.infer_function(ast)
        return T.Function(eparams, ebody, ast.pos), tD.but(tf)

    @infer.on(E.FunctionDefinition)
    def infer_FunctionDefinition(self, ast):
        eparams, ebody, tf, tD = self.infer_function(ast)
        self.add_name(ast.name.name, tf)
        return T.FunctionDefinition(ast.name, eparams, ebody, ast.pos), tD.but(Unit)

    @infer.on(E.Decorated)
    def infer_Decorated(self, ast):
        raise

    @infer.on(E.NamespaceDefinition)
    def infer_NamespaceDefinition(self, ast):
        nstype = self.Types.void(Unit)
        with self.clean_subenv():
            eb, tb = self.infer(ast.expr)
            self.unify_all(tb, nstype, ast.pos)
            child_env = self.env
            child_type_env = self.type_env
        self.add_namespace(ast.name, child_env.env)
        self.add_type_namespace(ast.name, child_type_env.env)
        return T.NamespaceDefn(ast.name, ast.key, eb, ast.pos), nstype

    @infer.on(E.NamespaceReferenceDefinition)
    def infer_NamespaceReferenceDefinition(self, ast):
        return T.NamespaceDefn(ast.name, ast.key, T.NoneExpr(ast.pos), ast.pos), self.Types.void(Unit)

    @infer.on(E.ModuleDefinition)
    def infer_ModuleDefinition(self, ast):
        raise
        return T.ModuleDefn(), self.Types(Unit)

    @infer.on(E.SignatureDefinition)
    def infer_SignatureDefinition(self, ast):
        with self.subenv():
            types = {}
            names = {}
            for decl in ast.body:
                if isinstance(decl, E.TypeDeclaration):
                    name = decl.name.name
                    if len(decl.args) == 0:
                        types[name] = self.type_env[name] = Object(Nullary(self.fresh()))
                    else:
                        types[name] = self.type_env[name] = Object(TypeConstructor(name, ArrowKind([Star for _ in decl.args], Star)))
                elif isinstance(decl, E.NameDeclaration):
                    scm = self.parse_toplevel_type(decl.annotation)
                    name = decl.name.name
                    names[name] = self.env[name] = Object(scm)
                elif isinstance(decl, E.LawDeclaration):
                    with self.subenv():
                        self.env.update({name.name: Object(TypeScheme([], self.fresh())) for name in decl.names})
                        e, t = self.infer(decl.expr)
                        law_type = self.Types.void(Bool)
                        self.unify_all(t, law_type, decl.pos)
                else:
                    raise RuntimeError()
            self.type_env[ast.name.name] = Object(Nullary(SignatureType(types, names)))
        return T.SignatureDefn(ast.name.name, self.Types(Unit), ast.pos), self.Types(Unit)

    def print_env(self):
        print('self.type_env:')
        self.type_env.printout()
        print('self.env:')
        self.env.printout()

    def update_with_subst(self, subst):
        for ast, env in self.asts:
            ast.type.vtype = (ast.type.vtype.apply(subst))
            ast.type.rtype = (ast.type.rtype.apply(subst))
            ast.type.ytype = (ast.type.ytype.apply(subst))
            ast.type.stype = (ast.type.stype.apply(subst))
            ast.type.ctype = (ast.type.ctype.apply(subst))

    @contextlib.contextmanager
    def clean_subenv(self):
        old, oldT = self.env, self.type_env
        self.env = Namespace(collections.ChainMap(BASE_ENVIRONMENT).new_child())
        self.type_env = Namespace(collections.ChainMap(BASE_TYPE_ENVIRONMENT).new_child())
        yield
        self.env, self.type_env = old, oldT

    @contextlib.contextmanager
    def subenv(self):
        old, oldT = self.env.env, self.type_env.env
        self.env.env = self.env.env.new_child()
        self.type_env.env = self.type_env.env.new_child()
        yield
        self.env.env, self.type_env.env = old, oldT

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

    if t1 is Any or t2 is Any:
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
    return {**su2, **su1}

def solve_kind(su, cs):
    su = {**su}
    for c in cs:
        c = c.apply(su)
        su1 = unifies_kind(c.k1, c.k2, c.pos)
        su.update(su1)
    return su

def solve(su, cs):
    su = {**su}
    for c in (k.apply(su) for k in cs):
        su1 = unifies(c.t1, c.t2, c.pos)
        su.update(su1)
    return su

######

def infer(ast, input_text):
    i = TypeChecker(input_text)
    e, t = i.infer(ast)
    s = solve({}, i.unifiers)
    # s1 = solve_kind({}, i.kind_unifiers)
    #print(len(i.unifiers), len(i.kind_unifiers))
    i.update_with_subst(s)
    #i.print_env()
    return e
