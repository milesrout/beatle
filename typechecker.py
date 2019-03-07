from collections import ChainMap
import contextlib
from itertools import chain, count

import typednodes as T
import cstnodes as E
from astpass import DeepAstPass
from is_gen import is_gen
from utils import (Type, Ast, ApeError, ApeSyntaxError, ApeInternalError,
                   ApeNotImplementedError, ApeTypeError, overloadmethod, unzip)
from kinds import *
from apetypes import *
from environments import Object, Namespace, add_namespace

#⭑: 2b51
#★: 2605

class ApeInferenceError(ApeTypeError):
    pass

class ApeUnificationError(ApeTypeError):
    pass

def pos_only_params(*args):
    return ParamsType.pos_only(
        [AnonParamType(a) for a in args])

tY = TypeVariable('y')
tS = TypeVariable('s')
tR = TypeVariable('r')

BASE_TYPE_ENVIRONMENT = {
    'generator': Object(AbstractTypeConstructor(TernaryKind)),
    'coroutine': Object(AbstractTypeConstructor(TernaryKind)),
    #'list':  Object(AbstractTypeConstructor(UnaryKind)),
    'int':   Object(Nullary(Int)),
    'float': Object(Nullary(Float)),
    'bool':  Object(Nullary(Bool)),
    'error': Object(Nullary(Error)),
    'any':   Object(Nullary(Any)),
}

def CoroutineType(y, s, r):
    return TypeCall(BASE_TYPE_ENVIRONMENT['coroutine'].value, (y, s, r))

def GeneratorType(y, s, r):
    return TypeCall(BASE_TYPE_ENVIRONMENT['generator'].value, (y, s, r))

def IterationResultType(y, s, r):
    return DisjunctionType([TaggedType('return', r),
                            TaggedType('yield', TupleType([y, CoroutineType(y, s, r)]))])

BASE_ENVIRONMENT = {
    # Want to add 'next', 'send', 'throw', etc. for generators.
    # But we need linear types before we can do that. Then we can say
    # iter:   Iterable(y,r)         ~> r + (y, Iterator(y,r)),
    # next:   Iterator(y,r)         ~> r + (y, Iterator(y,r)),
    # start:  Generator(y,s,r)      ~> r + (y, Coroutine(y,s,r)), and
    # send:   (s, Coroutine(y,s,r)) ~> r + (y, Coroutine(y,s,r)),
    # where ~> is the LinearArrowType, Iterable(y,r) = Generator(y,(),r),
    # and Iterator(y,r) = Coroutine(y,(),r).
    'iter': Object(TypeScheme(['y', 'r'], FunctionType(
        pos_only_params(GeneratorType(tY, Unit, tR)),
        IterationResultType(tY, Unit, tR)))),
    'next': Object(TypeScheme(['y', 'r'], FunctionType(
        pos_only_params(CoroutineType(tY, Unit, tR)),
        IterationResultType(tY, Unit, tR)))),
    'start': Object(TypeScheme(['y', 's', 'r'], FunctionType(
        pos_only_params(GeneratorType(tY, tS, tR)),
        IterationResultType(tY, tS, tR)))),
    'send': Object(TypeScheme(['y', 's', 'r'], FunctionType(
        pos_only_params(tS, CoroutineType(tY, tS, tR)),
        IterationResultType(tY, tS, tR)))),
    'litstr':    Object(TypeScheme([],    FunctionType(pos_only_params(String), Expression))),
    'stringify': Object(TypeScheme([],    FunctionType(pos_only_params(Expression), Expression))),
    'gensym':    Object(TypeScheme([],    FunctionType(ParamsType.unit(), String))),
    'something': Object(TypeScheme(['a'], FunctionType(pos_only_params(TypeVariable('a')), MaybeType(TypeVariable('a'))))),
    'nothing':   Object(TypeScheme(['a'], MaybeType(TypeVariable('a')))),
    'print':     Object(TypeScheme([],    FunctionType(ParamsType.varargs(ListType(String)), Unit))),
    'str':       Object(TypeScheme(['a'], FunctionType(pos_only_params(TypeVariable('a')), String))),
    'set':       Object(TypeScheme(['a'], FunctionType(ParamsType.unit(), SetType(TypeVariable('a'))))),
    'len':       Object(TypeScheme(['a'], FunctionType(
        pos_only_params(ListType(TypeVariable('a'))),
        Int))),
    'cons':      Object(TypeScheme(['a'], FunctionType(
        pos_only_params(TypeVariable('a'), ListType(TypeVariable('a'))),
        ListType(TypeVariable('a'))))),
    'RuntimeError': Object(TypeScheme([], FunctionType(pos_only_params(String), Error))),
    'AssertionError': Object(TypeScheme([], FunctionType(pos_only_params(String), Error))),
    'eval': Object(TypeScheme(['a'], FunctionType(
        pos_only_params(Expression),
        TypeVariable('a')))),
}

generic_unary_op = TypeScheme(['a'], FunctionType(pos_only_params(TypeVariable('a')), TypeVariable('a')))
generic_binary_op = TypeScheme(['a'], FunctionType(pos_only_params(TypeVariable('a'), TypeVariable('a')), TypeVariable('a')))

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
    def __init__(self):
        self.env = Namespace(BASE_ENVIRONMENT)
        self.type_env = Namespace(BASE_TYPE_ENVIRONMENT)
        self.subst = {}
        self.unifiers = []
        self.kind_unifiers = []
        self.fresh_vars = (f'a{i}' for i in count(1))
        self.fresh_kind_vars = (f'k{i}' for i in count(1))
        self.asts = []
        self.tags = {}

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

        self.Types = Types

    ######

    def clean_vars(self, disallowed):
        return (TypeVariable(x)
                for x in chain('abcdefghijklmnopqrstuvwxyz', (f'a{i}' for i in count(1)))
                if x not in disallowed)

    def clean_fresh(self, disallowed):
        return next(self.clean_vars(disallowed))

    def free_vars_in_env(self):
        return set.union(*(ty.value.ftv() for ty in self.env.env.values() if isinstance(ty, Object)), self.type_env.env.keys())

    def fresh(self):
        return TypeVariable(next(self.fresh_vars))

    def fresh_kind(self):
        return KindVariable(next(self.fresh_kind_vars))

    def instantiate(self, scm):
        return scm.t.apply({tvar: self.fresh() for tvar in scm.tvars})

    def generalise(self, t):
        existing_types = self.free_vars_in_env()
        ftv = t.ftv() - existing_types
        subst = dict(zip(ftv, [self.fresh() for tv in range(len(ftv))]))
        tt = t.apply(subst)
        return TypeScheme([subst[tvar].tvar for tvar in ftv], tt)

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
        self.env.update({name: Object(scm.value.apply(self.subst)) for name, scm in self.env.env.items() if isinstance(scm, Object)})
        return self.subst

    def add_name_ungeneralised(self, name, t):
        subst = self.solve()
        self.env[name] = Object(TypeScheme([], t.apply(subst)))

    def add_type_name(self, name, tc):
        """Add a type constructor (possibly nullary) to the type environment"""
        self.type_env[name] = Object(tc)

    def add_name(self, name, t):
        subst = self.solve()
        self.env[name] = Object(self.generalise(t.apply(subst)))

    def add_type_namespace(self, name, env):
        add_namespace(self.type_env, name, env)

    def add_namespace(self, name, env):
        add_namespace(self.env, name, env)

    #def add_type_namespace(self, name, env):
    #    environment = self.type_env
    #    for part in name.parts[:-1]:
    #        environment[part.name] = Namespace({})
    #        environment = environment[part.name]
    #    environment[name.parts[-1].name] = Namespace(env)

    #def add_namespace(self, name, env):
    #    environment = self.env
    #    for part in name.parts[:-1]:
    #        environment[part.name] = Namespace({})
    #        environment = environment[part.name]
    #    environment[name.parts[-1].name] = Namespace(env)

    def print_env(self):
        print('self.type_env:')
        self.type_env.printout()
        print('self.tags:')
        for k, v in self.tags.items():
            print(k, v)
        print('self.env:')
        self.env.printout()

    def update_with_subst(self, subst):
        for ast, env in self.asts:
            ast.type.vtype = (ast.type.vtype.apply(subst))
            ast.type.rtype = (ast.type.rtype.apply(subst))
            ast.type.ytype = (ast.type.ytype.apply(subst))
            ast.type.stype = (ast.type.stype.apply(subst))
            ast.type.ctype = (ast.type.ctype.apply(subst))

    def update_with_subst(self, subst):
        for ast, env in self.asts:
            ast.type.vtype = (ast.type.vtype.apply(subst))
            ast.type.rtype = (ast.type.rtype.apply(subst))
            ast.type.ytype = (ast.type.ytype.apply(subst))
            ast.type.stype = (ast.type.stype.apply(subst))
            ast.type.ctype = (ast.type.ctype.apply(subst))

    @contextlib.contextmanager
    def clean_subenv(self):
        with self.type_env.clean_subenv(), self.env.clean_subenv():
            yield

    @contextlib.contextmanager
    def subenv(self):
        with self.type_env.subenv(), self.env.subenv():
            yield

#    @contextlib.contextmanager
#    def clean_subenv(self):
#        old, oldT = self.env, self.type_env
#        self.env = Namespace(ChainMap(BASE_ENVIRONMENT).new_child())
#        self.type_env = Namespace(ChainMap(BASE_TYPE_ENVIRONMENT).new_child())
#        yield
#        self.env, self.type_env = old, oldT
#
#    @contextlib.contextmanager
#    def subenv(self):
#        old, oldT = self.env.env, self.type_env.env
#        self.env.env = self.env.env.new_child()
#        self.type_env.env = self.type_env.env.new_child()
#        yield
#        self.env.env, self.type_env.env = old, oldT
#
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
        if isinstance(t1, self.Types):
            raise NotImplementedError
        if isinstance(t2, self.Types):
            raise NotImplementedError
        if isinstance(t1, TypeConstructor):
            raise NotImplementedError
        if isinstance(t2, TypeConstructor):
            raise NotImplementedError
        if t1 is not Any and t2 is not Any:
            self.unifiers.append(Constraint(t1, t2, pos))

    def unify_kind(self, k1, k2, pos):
        self.kind_unifiers.append(KindConstraint(k1, k2, pos))

    ######

    def infer_error(self, ast):
        try:
            pos = ast.pos
        except Exception:
            pos = None
        raise ApeInternalError(pos=pos, msg='no overload found for {}'.format(ast.__class__))

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

    @infer_kind.on(E.TypeMaybeExpression)
    def infer_kind_TypeMaybeExpression(self, ast):
        self.unify_kind(self.infer_kind(ast.t), Star, ast.pos)
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

    @infer_type.on(E.TypeTaggedExpression)
    def infer_type_TypeTaggedExpression(self, ast):
        if ast.t is None:
            t = Unit
        else:
            t = self.infer_type(ast.t)
        #print('TAGGED', ast.tag, 'TYPE', t)
        return TaggedType(ast.tag, t)
        #if ast.tag in self.tags:
        #    if self.tags[ast.tag] is not t:
        #        raise ApeTypeError(pos=ast.pos, msg=f'Cannot redefine tag \'{ast.tag}\': already represents {self.tags[ast.tag]}, not {t}')
        #else:
        #    g = TypeScheme([], t)
        #    self.tags[ast.tag] = g
        #if ast.tag in self.tags:
        #    if self.tags[ast.tag] is not t:
        #        raise ApeTypeError(pos=ast.pos, msg=f'Cannot redefine tag \'{ast.tag}\': already represents {self.tags[ast.tag]}, not {t}')
        #else:
        #    g = self.generalise(t)
        #    print('g', g, 't', t)
        #    print('g', repr(g), 't', repr(t))
        #    self.tags[ast.tag] = g
        #return TaggedType(ast.tag, t)

    @infer_type.on(E.TypeDisjunctionExpression)
    def infer_type_TypeDisjunctionExpression(self, ast):
        return DisjunctionType([self.infer_type(expr) for expr in ast.exprs])

    @infer_type.on(E.TypeFunctionExpression)
    def infer_type_TypeFunctionExpression(self, ast):
        return FunctionType(pos_only_params(self.infer_type(ast.t1)), self.infer_type(ast.t2))

    @infer_type.on(E.TypeMaybeExpression)
    def infer_type_TypeMaybeExpression(self, ast):
        t = self.infer_type(ast.t)
        return MaybeType(t)

    @infer_type.on(E.TypeCallExpression)
    def infer_type_TypeCallExpression(self, ast):
        tc = self.infer_type(ast.atom)
        #print('TC', tc, repr(tc))
        ts = [self.infer_type(t) for t in ast.args]
        return TypeCall(tc, ts)

    def parse_type(self, ast):
        t, k = self.infer_type(ast), self.infer_kind(ast)
        return t, k

    ######

    @overloadmethod(use_as_wrapper=True, error_function=infer_error)
    def infer(self, original, ast_and_type):
        if isinstance(ast_and_type, Ast):
            ast, t = ast_and_type.node, ast_and_type.type
        else:
            if isinstance(ast_and_type, self.Types):
                raise ApeInternalError(
                    pos=original.pos,
                    msg='You forgot to return the typed node as well as the type itself: {ast_and_type} <= {original}')
                raise
            try:
                ast, t = ast_and_type
            except TypeError as exc:
                raise ApeInternalError(
                    pos=original.pos,
                    msg=f'Only returned one of ast and type: {ast_and_type} <= {original}')
        ret = Ast(ast, t, original.pos)
        self.asts.append((ret, self.env))
        return ret, t

    @infer.on(E.EmptyListExpression)
    def infer_EmptyListExpression(self, ast):
        tv = self.fresh()
        return T.EmptyList(), self.Types(ListType(tv))

    @infer.on(E.EmptySetExpression)
    def infer_EmptySetExpression(self, ast):
        tk = self.fresh()
        return T.EmptySet(), self.Types(SetType(tk))

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
                exprs.append(T.Star(es))
            else:
                ev, tv = self.infer(expr)
                self.unify_all(tV, tv, expr.pos)
                exprs.append(ev)

        return T.SetLit(exprs), tS

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
                exprs.append(T.DictKV(ek, ev))
            elif isinstance(expr, E.StarStarExpr):
                ee, te = self.infer(expr.expr)
                self.unify_all(tD, te, expr.expr.pos)
                exprs.append(T.StarStar(ee))
            else:
                raise ApeInternalError(pos=ast.pos, msg=f'Unexpected {expr.__class__.__name__} in {ast.__class__.__name__}')

        return T.DictLit(exprs), tD

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
                exprs.append(T.Star(el))
            else:
                ev, tv = self.infer(expr)
                self.unify_all(tV, tv, [ast.pos, expr.pos])
                exprs.append(ev)

        return T.List(exprs), tL

    @infer.on(E.TupleLiteral)
    def infer_TupleLiteral(self, ast):
        ets = [self.infer(expr) for expr in ast.exprs]
        es, ts = unzip(ets) if len(ets) > 0 else ([], [])

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

        return T.Raise(e1, e2), t1.but(vtype=self.fresh())

    @infer.on(E.NoneExpression)
    def infer_NoneExpression(self, ast):
        return T.NoneExpr(), self.Types.any(Unit)

    @infer.on(E.StringExpression)
    def infer_StringExpression(self, ast):
        return T.String(''.join(a.string for a in ast.unparsed)), self.Types(String)

    @infer.on(E.FloatExpression)
    def infer_FloatExpression(self, ast):
        return T.Float(ast.format, ast.value), self.Types(Float)

    @infer.on(E.IntExpression)
    def infer_IntExpression(self, ast):
        return T.Int(ast.base, ast.value), self.Types(Int)

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

    @infer.on(E.TryStatement)
    def infer_TryStatement(self, ast):
        tt = self.Types()
        eb, tb = self.infer(ast.body)
        self.unify(tt.rtype, tb.rtype, [ast.body.pos, ast.pos])
        self.unify(tt.ytype, tb.ytype, [ast.body.pos, ast.pos])
        self.unify(tt.stype, tb.stype, [ast.body.pos, ast.pos])
        for exb in ast.excepts:
            pass
        raise

    @infer.on(E.WhileStatement)
    def infer_WhileStatement(self, ast):
        tt = self.Types()

        ec, tc = self.infer(ast.cond)
        self.unify(tc.vtype, Bool, ast.cond.pos)
        self.unify_others(tc, tt, [ast.cond.pos, ast.pos])

        eb, tb = self.infer(ast.body)
        self.unify(tb.vtype, Unit, ast.pos)
        self.unify_others(tb, tt, [ast.body.pos, ast.pos])

        if ast.alt is not None:
            ea, ta = self.infer(ast.alt)
            self.unify(ta.vtype, Unit, ast.alt.pos)
            self.unify_others(ta, tt, [ast.alt.pos, ast.pos])

        return T.While(ec, eb, ea if ast.alt is not None else None), tt

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

            elifs.append(T.ElifBranch(eeic, eeib))

        if ast.else_branch is not None:
            eeb, teb = self.infer(ast.else_branch.body)
            self.unify_all(teb, tib, ast.else_branch.pos)
            eelse = T.ElseBranch(eeb)
        else:
            eelse = None

        return T.IfElifElse(T.IfBranch(eic, eib), elifs, eelse), tib

    @infer.on(E.UnaryExpression)
    def infer_UnaryExpression(self, ast):
        e, t = self.infer(ast.expr)

        tv = self.fresh()
        u1 = FunctionType(ArgsType.create(t.vtype), tv)
        u2 = self.instantiate(UNARY_OPERATORS[ast.op])
        self.unify(u1, u2, ast.pos)

        return T.Unary(ast.op, e), t.but(tv)

    @infer.on(E.ArithExpression)
    def infer_ArithExpression(self, ast):
        e1, t1 = self.infer(ast.left)
        e2, t2 = self.infer(ast.right)
        self.unify_others(t1, t2, [ast.left.pos, ast.right.pos])

        tv = self.fresh()
        u1 = FunctionType(ArgsType.create(t1.vtype, t2.vtype), tv)
        u2 = self.instantiate(BINARY_OPERATORS[ast.op])
        self.unify(u1, u2, [ast.left.pos, ast.right.pos])

        return T.Arith(ast.op, e1, e2), t1.but(tv)

    @infer.on(E.ReturnStatement)
    def infer_ReturnStatement(self, ast):
        if ast.expr is None:
            return T.Return(None), self.Types.any(Unit, rtype=Unit)

        e, t = self.infer(ast.expr)
        self.unify(t.vtype, t.rtype, ast.pos)
        return T.Return(e), t.but(self.fresh())

    @infer.on(E.PassStatement)
    def infer_PassStatement(self, ast):
        return T.Pass(), self.Types(Unit)

    @infer.on(E.IndexExpression)
    def infer_IndexExpression(self, ast):
        # For now, only support one index and no slices
        assert len(ast.indices) == 1
        assert type(ast.indices[0]) == E.Index

        t = self.Types()

        ef, tf = self.infer(ast.atom)
        self.unify_others(t, tf, ast.atom.pos)
        self.unify(ListType(t.vtype), tf.vtype, ast.atom.pos)

        ei, ti = self.infer(ast.indices[0].idx)
        self.unify_others(t, ti, ast.indices[0].idx.pos)
        self.unify(ti.vtype, Int, ast.indices[0].idx.pos)

        return T.Index(ef, [ei]), t

        es, ts = [], []
        for index in ast.indices:
            ei, ti = self.infer(index)
            self.unify_others(tf, ti, ast.pos)
            es.append(ei)
            ts.append(ti)

        tt = ArgsType(HListType([t.vtype for t in ts]), HDictType([]))
        tv = self.fresh()
        self.unify(tf.vtype, FunctionType(tt, tv), ast.pos)
        return T.Call(ef, es), tf.but(tv)

    @infer.on(E.CallExpression)
    def infer_CallExpression(self, ast):
        ef, tf = self.infer(ast.atom)

        eps, tps = [], []
        eks, tks = [], []
        for arg in ast.args:
            if isinstance(arg, E.PlainArg):
                ea, ta = self.infer(arg.expr)
                self.unify_others(tf, ta, arg.expr.pos)
                eps.append(ea)
                tps.append(ta)
            elif isinstance(arg, E.StarArg):
                raise ApeNotImplementedError(msg='splat arguments not yet supported', pos=arg.name.pos)
            elif isinstance(arg, E.StarStarKwarg):
                raise ApeNotImplementedError(msg='keyword splat arguments not yet supported', pos=arg.name.pos)
            elif isinstance(arg, E.KeywordArg):
                if not isinstance(arg.name, E.IdExpression):
                    raise ApeSyntaxError(msg=f'Argument keywords must be simple identifiers, not {arg.name.__class__.__name__}', pos=arg.name.pos)
                ea, ta = self.infer(arg.expr)
                self.unify_others(tf, ta, arg.expr.pos)
                eks.append((arg.name.name, ea))
                tks.append((arg.name.name, ta))
                #raise ApeNotImplementedError(msg='keyword arguments not yet supported', pos=arg.name.pos)
            elif isinstance(arg, E.CompForArg):
                raise ApeNotImplementedError(msg='comprehension arguments not yet supported', pos=ast.comp.pos)
            else:
                raise ApeNotImplementedError(msg=f'argument type \'{arg.__class__.__name__}\' not yet supported', pos=ast.pos)

        tt = ArgsType(HListType([t.vtype for t in tps]),
                      HDictType([(k, t.vtype) for k, t in tks]))
        tv = self.fresh()
        self.unify(tf.vtype, FunctionType(tt, tv), ast.pos)
        return T.Call(ef, eps, eks), tf.but(tv)

    @infer.on(E.Comparison)
    def infer_Comparison(self, ast):
        e1, t1 = self.infer(ast.a)
        e2, t2 = self.infer(ast.b)
        self.unify_all(t1, t2, ast.pos)
        return T.Comparison(ast.op, e1, e2), t1.but(Bool)

    @infer.on(E.IdExpression)
    def infer_IdExpression(self, ast):
        return T.Id(ast.name), self.Types(self.lookup_name(ast.name, ast.pos))

    @infer.on(E.TrueExpression)
    def infer_TrueExpression(self, ast):
        return T.Bool(True), self.Types(Bool)

    @infer.on(E.FalseExpression)
    def infer_FalseExpression(self, ast):
        return T.Bool(False), self.Types(Bool)

    @infer.on(E.YieldFromExpression)
    def infer_YieldFromExpression(self, ast):
        t = self.Types()

        tc = CoroutineType(self.fresh(), self.fresh(), self.fresh())

        self.unify(t.ytype, tc.ts[0], [ast.pos, ast.expr.pos])
        self.unify(t.stype, tc.ts[1], [ast.pos, ast.expr.pos])
        self.unify(t.vtype, tc.ts[2], [ast.pos, ast.expr.pos])

        ef, tf = self.infer(ast.expr)

        self.unify_all(tf, t.but(tc), ast.expr.pos)

        return T.YieldFrom(ef), t

    @infer.on(E.YieldExpression)
    def infer_YieldExpression(self, ast):
        e, t = self.infer(ast.expr)
        self.unify(t.vtype, t.ytype, ast.pos)
        return T.Yield(e), t.but(t.stype)

    @infer.on(E.TaggedExpression)
    def infer_TaggedExpression(self, ast):
        # e.g. nil : ∀a.((unit/) -> list[a]), but instantiated
        tft = self.Types(self.instantiate(self.tags[ast.tag]))

        tv = self.fresh()
        tr = self.Types(self.fresh())

        # e.g. (a/) -> b
        self.unify(tft.vtype, FunctionType(ArgsType.create(tv), tr.vtype), ast.pos)

        if ast.expr is None:
            ee = None
            self.unify(tv, Unit, ast.pos)
        else:
            ee, te = self.infer(ast.expr)
            self.unify_all(tr.but(tv), te, ast.pos)
        return T.Tagged(ast.tag, ee), tr

    @infer.on(E.DoStatement)
    def infer_DoStatement(self, ast):
        e, t = self.infer(ast.body)
        return T.Do(e), t

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

    @infer.on(E.LogicalNotExpression)
    def infer_LogicalNotExpression(self, ast):
        t = self.Types(Bool)
        ee, te = self.infer(ast.expr)
        self.unify_all(te, t, ast.pos)
        return T.LogicalNot(ee), t

    @infer.on(E.Quasiquote)
    def infer_Quasiquote(self, ast):
        return QuoteTypeInference(self).visit(ast.expr)

    # Recall that parameter default values are evaluated at definition time in
    # Python. This is also true in Beatle.
    #
    # Thus:
    #  - the v-types of the default value expressions must unify with the types
    #    of the parameters.
    #  - the other types of the default value expressions must unify with the
    #    other types of the *function call definition expression*.
    def infer_params(self, params):
        tdefs = self.Types()
        edefs = []
        pots = []
        pkts = []
        kots = []
        vargs = Void
        kwargs = Void
        # always start off appending to the pkts
        types = pkts
        for p in params:
            if isinstance(p, E.Param):
                tv = self.fresh()
                edef = None
                if p.annotation is not None:
                    tann, kann = self.parse_type(p.annotation)
                    self.unify(tv, tann, p.annotation.pos)
                    self.unify_kind(kann, Star, p.annotation.pos)
                if p.default is not None:
                    edef, tdef = self.infer(p.default)
                    self.unify(tv, tdef.vtype, p.default.pos)
                    self.unify_others(tdefs, tdef, p.default.pos)
                edefs.append(edef)
                types.append(ParamType(p.name.name, tv, p.default is not None))
                self.add_name_ungeneralised(p.name.name, tv)
            elif isinstance(p, E.EndOfPosOnlyParams):
                pots, pkts = pkts, pots
                types = pkts
            elif isinstance(p, E.EndOfPosParams):
                types = kots
            elif isinstance(p, E.StarVarParams):
                vargs = self.fresh()
            elif isinstance(p, E.StarStarKwParams):
                kwargs = self.fresh()
            else:
                raise ApeNotImplementedError(msg=f'{p.__class__.__name__} is not supported yet', pos=p.pos)
        return ParamsType.make(pots, pkts, kots, vargs, kwargs), edefs, tdefs

    def infer_function(self, ast):
        with self.subenv():
            tt, defaults, tD = self.infer_params(ast.params)

            t = self.fresh()
            eb, tb = self.infer(ast.body)
            self.unify(tb.vtype, t, ast.pos)
            self.unify(tb.rtype, t, ast.pos)

            if is_gen(ast.body):
                y, s = self.fresh(), self.fresh()
                self.unify(tb.ytype, y, ast.pos)
                self.unify(tb.stype, s, ast.pos)
                tr = GeneratorType(y, s, t)
            else:
                self.unify(tb.ytype, Void, ast.pos)
                self.unify(tb.stype, Void, ast.pos)
                tr = t

            if ast.return_annotation:
                tann, kann = self.parse_type(ast.return_annotation)
                self.unify(tr, tann, ast.return_annotation.pos)
                self.unify_kind(kann, Star, ast.return_annotation.pos)

            tf = FunctionType(tt, tr)
        return defaults, eb, tf, tD

    @infer.on(E.LambdaExpression)
    def infer_LambdaExpression(self, ast):
        edefaults, ebody, tf, tD = self.infer_function(ast)
        return T.Lambda(ast.params, edefaults, ebody), tD.but(tf)

    @infer.on(E.FunctionExpression)
    def infer_FunctionExpression(self, ast):
        edefaults, ebody, tf, tD = self.infer_function(ast)
        return T.Function(ast.params, edefaults, ebody), tD.but(tf)

    @infer.on(E.FunctionDefinition)
    def infer_FunctionDefinition(self, ast):
        edefaults, ebody, tf, tD = self.infer_function(ast)
        self.add_name(ast.name.name, tf)
        return T.FunctionDefinition(ast.name.name, ast.params, edefaults, ebody), tD.but(Unit)

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
        return T.NamespaceDefn(ast.name, ast.key, eb), nstype

    @infer.on(E.NamespaceReferenceDefinition)
    def infer_NamespaceReferenceDefinition(self, ast):
        return T.NamespaceDefn(ast.name, ast.key, T.NoneExpr(ast.pos)), self.Types.void(Unit)

    @infer.on(E.ModuleDefinition)
    def infer_ModuleDefinition(self, ast):
        raise
        return T.ModuleDefn(), self.Types(Unit)

    @infer.on(E.TypeDefinition)
    def infer_TypeDefinition(self, ast):
        name = ast.name.name
        if len(ast.args) == 0:
            self.add_type_name(name, Nullary(self.infer_type(ast.expr)))
        else:
            tvs = [self.fresh() for a in ast.args]
            tvar = TypeVariable(name)
            with self.subenv():
                for n, tv in zip(ast.args, tvs):
                    self.type_env[n.name] = Object(Nullary(tv))
                self.type_env[name] = Object(Nullary(tvar))
                t = self.infer_type(ast.expr)
            self.add_type_name(name, ConcreteTypeConstructor(
                [Star] * len(ast.args), Star, tvs, MuType(tvar, t)))
            if isinstance(t, DisjunctionType) and all(isinstance(s, TaggedType) for s in t.ts):
                for s in t.ts:
                    self.tags[s.tag] = self.generalise(FunctionType(ArgsType.create(s.t), TypeCall(tvar, tvs)))
        return T.NoneExpr(), self.Types(Unit)

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
                        # HERE
                    else:
                        types[name] = self.type_env[name] = Object(AbstractTypeConstructor(ArrowKind([Star for _ in decl.args], Star)))
                        # HERE
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
            # HERE
        return T.SignatureDefn(ast.name.name, self.Types(Unit)), self.Types(Unit)

    @infer.on(E.AnnotatedAssignment)
    def infer_AnnotatedAssignment(self, ast):
        ta, ka = self.parse_type(ast.annotation)
        if not isinstance(ast.assignee, E.IdExpression):
            raise ApeTypeError(pos=ast.pos, msg='Cannot write a type-annotated assignment for assignees that are not identifiers')
        self.unify_kind(ka, Star, ast.annotation.pos)
        ee, te = self.infer(ast.expr)
        tta = self.Types(ta)
        self.unify_all(tta, te, ast.pos)
        self.add_name(ast.assignee.name, ta)
        return T.Assignment([ast.assignee.name], ee), tta

    @infer.on(E.ChainedAssignment)
    def infer_ChainedAssignment(self, ast):
        assignees = ast.assignees[:-1]
        expr = ast.assignees[-1]
        ma = MatchAssignee(self)

        ee, te = self.infer(expr)
        targets = []
        assignments = []

        for a in assignees:
            asgnmts, target, t = ma.match_assignee(a)
            assignments.extend(asgnmts)
            targets.append(target)

        for name, ty in assignments:
            self.add_name(name, ty)

        return T.Assignment(targets, ee), te.but(Unit)

class MatchAssignee:
    def __init__(self, tc):
        self.tc = tc

    def match_error(self, asgn):
        raise ApeNotImplementedError(asgn.pos, f'{asgn.__class__.__name__} assignee matching not yet implemented')

    @overloadmethod(error_function=match_error)
    def match_assignee(self, asgn):
        ...

    @match_assignee.on(E.EmptyListExpression)
    def match_assignee_EmptyListExpression(self, asgn):
        return [], T.List([]), self.tc.Types(ListType(self.tc.fresh()))

    @match_assignee.on(E.TaggedExpression)
    def match_assignee_TaggedExpression(self, asgn):
        assignments, targets, types = self.match_assignee(asgn.expr)
        return assignments, T.Tagged(asgn.tag, targets), types.but(TaggedType(asgn.tag, types.vtype))

    @match_assignee.on(E.TupleLiteral)
    def match_assignee_TupleLiteral(self, asgn):
        assignments = []
        targets = []
        types = []
        tt = self.tc.Types()
        for a in asgn.exprs:
            asgn, e, t = self.match_assignee(a)
            assignments.extend(asgn)
            targets.append(e)
            types.append(t.vtype)
            self.tc.unify_others(t, tt, a.pos)
        return assignments, T.Tuple(targets), tt.but(TupleType(types))

# Commenting this out for now. We need some way to deal with:
#    [a, b, c] = [1, 2, 3, 4]
#    [a, b, c] = [1, 2]
#    [a, b, c] = [1, 'a', 3.0]
# and
#    [a, b, *c] = [1, 2, 'h', 'e', 'l', 'l', 'o']
# while maintaining type safety and compatibility with Python. Using an
# HListType is a possibility.  Getting this right means we are compatible with
# Perl-style argument passing as well. Whether that is a good or a bad thing is
# dependent upon your point of view.
#
# Don't forget to fix EmptyListExpression matching at the same time
#
#    @match_assignee.on(E.ListLiteral)
#    def match_assignee_ListLiteral(self, asgn):
#        assignments = []
#        targets = []
#        types = []
#        t = self.tc.Types()
#        for a in asgn.exprs:
#            ea, ta = self.match_assignee(a)
#            targets.append(ea)
#            types.append(ta)
#            self.unify_all(t, ta)
#        return assignments, T.List(targets), t.but(ListType(t.vtype))

    @match_assignee.on(E.IdExpression)
    def match_assignee_IdExpression(self, asgn):
        tv = self.tc.fresh()
        return [(asgn.name, tv)], T.Id(asgn.name), self.tc.Types(tv)

class QuoteTypeInference(DeepAstPass):
    def __init__(self, tc):
        self.tc = tc

    def override_do_visit_wrapper(self, ast, new):
        if ast is new:
            return Ast(T.Quote(ast, ()), self.tc.Types(Any), new.pos)

        cls, args = new
        try:
            return Ast(T.Quote(cls, args), self.tc.Types(Any), ast.pos)
        except TypeError:
            print(cls.__name__)
            raise

    @overloadmethod(use_as_default=True)
    def visit(self, ast):
        return self.do_visit(ast)

    @visit.on(E.Unquote)
    def qtinfer_Unquote(self, ast):
        e, t = self.tc.infer(ast.expr)
        self.tc.unify_all(t, self.tc.Types(Any), ast.pos)
        return e

######

def bind_kind(name, k, pos):
    if isinstance(k, KindVariable) and k.name == name:
        return {}
    if name in k.fkv():
        raise ApeUnificationError(msg=f'infinite kind with {name}={k}!', pos=pos)
    return {name: k}

def bind(tvar, t, pos):
    if isinstance(t, TypeVariable) and t.tvar == tvar:
        return {}
    if tvar in t.ftv():
        raise ApeUnificationError(msg=f'infinite type with {tvar}={t}!', pos=pos)
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

def unify_constructor(t1, t2, pos):
    su1 = unifies_kind(t1.con.kind, ArrowKind([t.kind for t in t1.ts], Star), pos)
    s = {x.tvar: t for x, t in zip(t1.con.tvars, t1.ts)}
    su2 = unifies(t1.apply(su1).con.expr.apply(s), t2.apply(su1), pos)
    return compose_subst(su2, su1)

def unifies(t1, t2, pos):
    if isinstance(t1, AbstractTypeConstructor) and isinstance(t2, AbstractTypeConstructor):
        return unifies_kind(t1.kind, t2.kind, pos)

    if isinstance(t1, TypeCall) and isinstance(t1.con, ConcreteTypeConstructor):
        if len(t1.ts) == len(t1.con.tvars):
            return unify_constructor(t1, t2, pos)
    if isinstance(t2, TypeCall) and isinstance(t2.con, ConcreteTypeConstructor):
        if len(t2.ts) == len(t2.con.tvars):
            return unify_constructor(t2, t1, pos)

    if not isinstance(t1, Type) or not isinstance(t2, Type):
        raise ApeUnificationError(pos=pos, msg=f'Can only unify types, not {t1}: {type(t1)} and {t2}: {type(t2)}')

    if t1 == t2:
        return {}

    if t1 is Any and t2 is Any:
        raise ApeUnificationError(pos=pos, msg=f'Cannot unify Any and Any')

    if t1 is Any or t2 is Any:
        return {}

    if isinstance(t1, TypeVariable):
        return bind(t1.tvar, t2, pos)
    if isinstance(t2, TypeVariable):
        return bind(t2.tvar, t1, pos)

    if isinstance(t1, TupleType) and isinstance(t2, TupleType):
        if len(t1.ts) == len(t2.ts):
            return unify_many(t1.ts, t2.ts, pos)

    if isinstance(t1, DisjunctionType) and isinstance(t2, DisjunctionType):
        if len(t1.ts) == len(t2.ts):
            return unify_many(t1.ts, t2.ts, pos)

    if isinstance(t1, AnonParamType) and isinstance(t2, AnonParamType):
        if t1.opt == t2.opt:
            return unifies(t1.t, t2.t, pos)
    if isinstance(t1, AnonParamType):
        return unifies(t1.t, t2, pos)
    if isinstance(t2, AnonParamType):
        return unifies(t2, t1, pos)

    if isinstance(t1, ParamType) and isinstance(t2, ParamType):
        if t1.name == t2.name and t1.opt == t2.opt:
            return unifies(t1.t, t2.t, pos)
    if isinstance(t1, ParamType):
        return unifies(t1.t, t2, pos)
    if isinstance(t2, ParamType):
        return unifies(t2, t1, pos)

    if isinstance(t1, HListType) and isinstance(t2, ListType):
        return unify_many(t1.ts, [t2.t] * len(t1.ts), pos)
    if isinstance(t1, ListType) and isinstance(t2, HListType):
        return unifies(t2, t1, pos)

    if isinstance(t1, HListType) and isinstance(t2, HListType):
        if len(t1.ts) == len(t2.ts):
            return unify_many(t1.ts, t2.ts, pos)

    if isinstance(t1, HDictType) and isinstance(t2, HDictType):
        if len(t1.ts) == len(t2.ts):
            # require the same sets of names, and no duplicates
            if t1.names == t2.names == list(sorted(set(t1.names))):
                return unify_many(t1.types, t2.types, pos)

    if isinstance(t1, ParamsType) and isinstance(t2, ParamsType):
        return unify_many([t1.pots, t1.pkts, t1.kots, t1.args, t1.kwds],
                          [t2.pots, t2.pkts, t2.kots, t2.args, t2.kwds], pos)

    if isinstance(t1, ArgsType) and isinstance(t2, ParamsType):
        return unify_call(t1, t2, pos)
    if isinstance(t1, ParamsType) and isinstance(t2, ArgsType):
        return unify_call(t2, t1, pos)

    if isinstance(t1, ArgsType) and isinstance(t2, ArgsType):
        return unify_many([t1.pts, t1.kts], [t2.pts, t2.kts], pos)

    if isinstance(t1, MaybeType) and isinstance(t2, MaybeType):
        return unifies(t1.t, t2.t, pos)
    if isinstance(t1, SetType) and isinstance(t2, SetType):
        return unifies(t1.t, t2.t, pos)
    if isinstance(t1, ListType) and isinstance(t2, ListType):
        return unifies(t1.t, t2.t, pos)

    if isinstance(t1, DictType) and isinstance(t2, DictType):
        return unify_many([t1.k, t1.v], [t2.k, t2.v], pos)
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        return unify_many([t1.p, t1.r], [t2.p, t2.r], pos)
    if isinstance(t1, TypeCall) and isinstance(t2, TypeCall):
        return unify_many([t1.con, *t1.ts], [t2.con, *t2.ts], pos)

    raise ApeUnificationError(pos=pos, msg=f'Cannot unify {t1} and {t2}')

def apply(ts, subst):
    return [t.apply(subst) for t in ts]

def unify_call(a, p, pos, seen=set()):
    if len(a.pts.ts) != 0 and len(p.pots) != 0:
        if len(a.pts.ts) < len(p.pots):
            raise ApeUnificationError(msg='Not enough positional arguments', pos=pos)
        # if any(t.name in seen for t in p.pots):
        #     raise ApeUnificationError(msg='Already have an argument for {t.name}', pos=pos)
        l = len(p.pots)
        su1 = unifies(a.pts[:l], HListType(p.pots), pos)
        at = ArgsType(a.pts[l:].apply(su1), a.kts.apply(su1))
        pt = ParamsType.make([], apply(p.pkts, su1), apply(p.kots, su1),
                             p.args.apply(su1), p.kwds.apply(su1))
        su2 = unify_call(at, pt, pos, seen)
        return compose_subst(su2, su1)
    if len(a.pts.ts) != 0 and len(p.pkts) != 0:
        l = min(len(a.pts.ts), len(p.pkts))
        su1 = unifies(a.pts[:l], HListType(p.pkts[:l]), pos)
        at = ArgsType(a.pts[l:].apply(su1), a.kts.apply(su1))
        pt = ParamsType.make([], apply(p.pkts[l:], su1), apply(p.kots, su1),
                             p.args.apply(su1), p.kwds.apply(su1))
        new_seen = {t.name for t in p.pkts[:l]}
        su2 = unify_call(at, pt, pos, seen | new_seen)
        return compose_subst(su2, su1)
    if len(a.pts.ts) != 0:
        su1 = unifies(a.pts, p.args, pos)
        at = ArgsType(HListType([]), a.kts.apply(su1))
        pt = ParamsType.make([], [], apply(p.kots, su1), p.args.apply(su1), p.kwds.apply(su1))
        su2 = unify_call(at, pt, pos, seen)
        return compose_subst(su2, su1)
    if len(a.kts) != 0 and len(p.pots) != 0:
        raise ApeUnificationError(msg='Not enough positional arguments', pos=pos)
    if len(a.kts) != 0 and len(p.pkts) != 0:
        raise ApeUnificationError(msg='Not enough positional arguments', pos=pos)
    if len(a.kts) != 0 and len(p.kots) != 0:
        raise ApeUnificationError(msg='Not enough positional arguments', pos=pos)
    if len(a.kts) != 0:
        raise ApeUnificationError(msg='Not enough positional arguments', pos=pos)
    return {}
    #raise ApeNotImplementedError(pos=pos, msg='We need type lists to properly handle f(*[1, 2, 3]) where f: (int, int, int) -> int')

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

def test_infer():
    try:
        unify_call(ArgsType(HListType([Int, Int, Int, Int]), HDictType([])), ParamsType.make([ParamType('a', Int), ParamType('b',  Int)], [ParamType('c', Int)], [], TypeVariable('a'), TypeVariable('b')), 0)
    except BaseException as e:
        print(e.format_with_context(''))

def infer(ast):
    try:
        i = TypeChecker()
        e, t = i.infer(ast)
        s = solve({}, i.unifiers)
        i.update_with_subst(s)
        return e
    except ApeTypeError:
        raise
    except ApeError as exc:
        raise ApeTypeError(pos=None, msg='Unexpected error in typechecker') from exc
