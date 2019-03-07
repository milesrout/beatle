import kinds as K
from utils import Type, ApeInternalError

class PrimitiveType(Type):
    def apply(self, subst):
        return self

    def ftv(self):
        return set()

    @property
    def kind(self):
        return K.Star

    def __hash__(self):
        return hash((type(self),))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if self is other:
                return True
            raise RuntimeError
        return NotImplemented

class ExpressionType(PrimitiveType):
    def __repr__(self):
        return 'Expression'

    def __str__(self):
        return 'expression'

    def __hash__(self):
        return hash(ExpressionType)
Expression = ExpressionType()

class BoolType(PrimitiveType):
    def __repr__(self):
        return 'Bool'

    def __str__(self):
        return 'bool'

    def __hash__(self):
        return hash(BoolType)
Bool = BoolType()

class FloatType(PrimitiveType):
    def __repr__(self):
        return 'Float'

    def __str__(self):
        return 'float'

    def __hash__(self):
        return hash(FloatType)
Float = FloatType()

class IntType(PrimitiveType):
    def __repr__(self):
        return 'Int'

    def __str__(self):
        return 'int'

    def __hash__(self):
        return hash(IntType)
Int = IntType()

class ErrorType(PrimitiveType):
    def __repr__(self):
        return 'Error'

    def __str__(self):
        return 'error'

    def __hash__(self):
        return hash(ErrorType)
Error = ErrorType()

class StringType(PrimitiveType):
    def __repr__(self):
        return 'String'

    def __str__(self):
        return 'string'

    def __hash__(self):
        return hash(StringType)
String = StringType()

class AnyType(PrimitiveType):
    """This type exists to be unified with type variables that are required to
    be unconstrained."""
    def __repr__(self):
        return 'Any'

    def __str__(self):
        return 'any'

    def __hash__(self):
        return hash(AnyType)
Any = AnyType()

class HListType(Type):
    def __init__(self, ts):
        if isinstance(ts, TupleType):
            raise
        self.ts = ts

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return HListType(self.ts[idx])
        return self.ts[idx]

    def __repr__(self):
        return f'HListType({self.ts!r})'

    def __str__(self):
        return '<' + ', '.join(str(t) for t in self.ts) + '>'

    def apply(self, subst):
        return HListType([t.apply(subst) for t in self.ts])

    def ftv(self):
        if len(self.ts) == 0:
            return set()
        return set.union(*[t.ftv() for t in self.ts])

    def __eq__(self, other):
        if isinstance(other, HListType):
            return self.ts == other.ts
        return NotImplemented

    def __hash__(self):
        return hash((HListType, tuple(self.ts)))

class HDictType(Type):
    """A 'dict' type that is a list of pairs (to maintain order)"""
    def __init__(self, ts):
        self.ts = ts

    def __len__(self):
        return len(self.ts)

    @property
    def types(self):
        return [t for k, t in sorted(self.ts)]

    @property
    def names(self):
        return list(sorted(k for k, t in self.ts))

#     def __getitem__(self, idx):
#         return self.ts[idx]
#
    def __repr__(self):
        return f'HDictType({self.ts!r})'

    def __str__(self):
        return '<' + ', '.join(f'{k}={v}' for k, v in self.ts) + '>'

    def apply(self, subst):
        return HDictType([(k, v.apply(subst)) for k, v in self.ts])

    def ftv(self):
        if len(self.ts) == 0:
            return set()
        return set.union(*[v.ftv() for k, v in self.ts])

    def __eq__(self, other):
        if isinstance(other, HDictType):
            return self.ts == other.ts
        return NotImplemented

    def __hash__(self):
        return hash((HDictType, tuple(self.ts)))

class AnonParamType(Type):
    def __init__(self, t, opt=False):
        if isinstance(t, list):
            raise
        self.t = t
        self.opt = opt

    def __repr__(self):
        return f'AnonParamType({self.t!r}, {self.opt!r})'

    def __str__(self):
        if self.opt:
            return f'{self.t} (optional)'
        else:
            return f'{self.t}'

    def apply(self, subst):
        t = self.t.apply(subst)
        if self.t is t:
            return self
        return AnonParamType(t, self.opt)

    def ftv(self):
        return self.t.ftv()

    def __eq__(self, other):
        if isinstance(other, AnonParamType):
            return self.t == other.t and self.opt == other.opt
        return NotImplemented

    def __hash__(self):
        return hash((AnonParamType, self.t, self.opt))

class ParamType(Type):
    def __init__(self, name, t, opt=False):
        self.name = name
        self.t = t
        self.opt = opt

    def __repr__(self):
        if self.opt:
            return f'ParamType({self.name!r}, {self.t!r}, {self.opt!r})'
        return f'ParamType({self.name!r}, {self.t!r})'

    def __str__(self):
        if self.opt:
            return f'{self.name}?: {self.t}'
        else:
            return f'{self.name}: {self.t}'

    def apply(self, subst):
        t = self.t.apply(subst)
        if self.t is t:
            return self
        return ParamType(self.name, t, self.opt)

    def ftv(self):
        return self.t.ftv()

    def __eq__(self, other):
        if isinstance(other, ParamType):
            return self.t == other.t and self.opt == other.opt and self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash((ParamType, self.name, self.t, self.opt))

class ArgsType(Type):
    def __init__(self, pts, kts):
        self.pts = pts
        self.kts = kts

    @classmethod
    def create(cls, *args, **kwds):
        return cls(HListType(args), HDictType(kwds))

    def __repr__(self):
        return f'ArgsType({self.pts!r}, {self.kts!r})'

    def __str__(self):
        pts = [str(t) for t in self.pts.ts]
        kts = [f'{k}={v}' for k, v in self.kts.ts]
        return 'A(' + ', '.join(pts + kts) + ')'

    def apply(self, subst):
        return ArgsType(self.pts.apply(subst),
                        self.kts.apply(subst))

    def ftv(self):
        return set.union(self.pts.ftv(), self.kts.ftv())

    def __eq__(self, other):
        if isinstance(other, ArgsType):
            return self.pts == other.pts and self.kts == other.kts
        return NotImplemented

    def __hash__(self):
        return hash((ArgsType, self.pts, self.kts))

class ParamsType(Type):
    def __init__(self, pots, pkts, kots, args, kwds):
        self.pots = pots
        self.pkts = pkts
        self.kots = kots
        self.args = args
        self.kwds = kwds

    @classmethod
    def unit(cls):
        return cls.make([], [], [], Void, Void)

    @classmethod
    def varargs(cls, t):
        return cls.make([], [], [], t, Void)

    @classmethod
    def pos_only(cls, ts):
        return cls.make(ts, [], [], Void, Void)

    @classmethod
    def make(cls, pots, pkts, kots, args, kwds):
        return cls(HListType(pots), HListType(pkts), HListType(kots), args, kwds)

    def __repr__(self):
        return f'ParamsType({self.pots}, {self.pkts}, {self.kots}, {self.args}, {self.kwds})'

    def __str__(self):
        types = []
        if self.pots.ts != []:
            types.extend([str(x) for x in self.pots.ts])
            types.append('/')
        types.extend([str(x) for x in self.pkts.ts])
        if self.kots.ts != []:
            types.append('*')
            types.extend([str(x) for x in self.kots.ts])
        if self.args != Void:
            types.append('*' + str(self.args))
        if self.kwds != Void:
            types.append('**' + str(self.kwds))
        return 'P(' + ', '.join(types) + ')'

    def apply(self, subst):
        return ParamsType(
            self.pots.apply(subst),  # [t.apply(subst) for t in self.pots],
            self.pkts.apply(subst),  # [t.apply(subst) for t in self.pkts],
            self.kots.apply(subst),  # [t.apply(subst) for t in self.kots],
            self.args.apply(subst),
            self.kwds.apply(subst))

    def ftv(self):
        return set.union(
            self.pots.ftv(),  # *[t.ftv() for t in self.pots],
            self.pkts.ftv(),  # *[t.ftv() for t in self.pkts],
            self.kots.ftv(),  # *[t.ftv() for t in self.kots],
            self.args.ftv(),
            self.kwds.ftv())

    def __eq__(self, other):
        if isinstance(other, ParamsType):
            return (self.pots == other.pots and
                    self.pkts == other.pkts and
                    self.kots == other.kots and
                    self.args == other.args and
                    self.kwds == other.kwds)

    def __hash__(self):
        return hash((ParamsType, self.pots, self.pkts, self.kots, self.args, self.kwds))

def skip__new__(cls, *args, **kwds):
    #print('\t\t', cls, args, kwds)
    obj = object.__new__(cls)
    obj.__init__(*args, **kwds)
    #print(repr(obj))
    return obj

class DisjunctionType(Type):
    def __new__(self, ts):
        if len(ts) == 0:
            return Void
        if len(ts) == 1:
            return ts[0]
        return super().__new__(self)

    def __init__(self, ts):
        self.ts = ts
        self._kind = K.Star

    def __repr__(self):
        if len(self.ts) == 0:
            return 'Void'
        return f'DisjunctionType({self.ts!r})'

    def __str__(self):
        if len(self.ts) == 0:
            return 'void'
        if len(self.ts) == 1:
            return str(self.ts[0])
        return '(' + ' | '.join(map(str, self.ts)) + ')'

    def apply(self, subst):
        return DisjunctionType([t.apply(subst) for t in self.ts])

    def ftv(self):
        if len(self.ts) == 0:
            return set()
        return set.union(*[t.ftv() for t in self.ts])

    def __eq__(self, other):
        if isinstance(other, DisjunctionType):
            if len(self.ts) == len(other.ts):
                return all(t == s for t, s in zip(self.ts, other.ts))
        return NotImplemented

    def __hash__(self):
        return hash((DisjunctionType, tuple(self.ts)))

# Need to do this to avoid the __new__ above
Void = skip__new__(DisjunctionType, [])

class TupleType(Type):
    def __new__(self, ts):
        if len(ts) == 0:
            return Unit
        if len(ts) == 1:
            return ts[0]
        return super().__new__(self)

    def __init__(self, ts):
        self.ts = ts

    def __repr__(self):
        if len(self.ts) == 0:
            return f'Unit'
        return f'TupleType({self.ts!r})'

    def __str__(self):
        if len(self.ts) == 0:
            return 'unit'
        if len(self.ts) == 1:
            return str(self.ts[0])
        return 'T(' + ' ⨯ '.join(map(str, self.ts)) + ')'

    def apply(self, subst):
        return TupleType([t.apply(subst) for t in self.ts])

    def ftv(self):
        if len(self.ts) == 0:
            return set()
        return set.union(*[t.ftv() for t in self.ts])

    def __hash__(self):
        return hash((TupleType, tuple(self.ts)))

    def __eq__(self, other):
        if isinstance(other, TupleType):
            if len(self.ts) == len(other.ts):
                return all(t == s for t, s in zip(self.ts, other.ts))
        return NotImplemented

# Need to do this to avoid the __new__ above
Unit = skip__new__(TupleType, [])

class MaybeType(Type):
    def __init__(self, t):
        self.t = t

    def __repr__(self):
        return f'MaybeType({self.t!r})'

    def __str__(self):
        return f'{self.t}?'

    def apply(self, subst):
        return MaybeType(self.t.apply(subst))

    def ftv(self):
        return self.t.ftv()

    def __eq__(self, other):
        if isinstance(other, MaybeType):
            return self.t == other.t
        return NotImplemented

    def __hash__(self):
        return hash((MaybeType, self.t))

class TypeVariable(Type):
    def __init__(self, tvar):
        self.tvar = tvar

    def __repr__(self):
        return f'TypeVariable({self.tvar!r})'

    def __str__(self):
        return self.tvar

    def apply(self, subst):
        return subst.get(self.tvar, self)

    def ftv(self):
        return {self.tvar}

    def __hash__(self):
        return hash((TypeVariable, self.tvar,))

    def __eq__(self, other):
        if isinstance(other, TypeVariable):
            return self.tvar == other.tvar
        return NotImplemented

class TypeCall(Type):
    def __init__(self, con, ts):
        self.con = con
        self.ts = ts

    def __repr__(self):
        return f'TypeCall({self.con!r}, {self.ts!r})'

    def __str__(self):
        params = ', '.join(map(str, self.ts))
        return f'{self.con}[{params}]'

    def apply(self, subst):
        return TypeCall(self.con.apply(subst), [t.apply(subst) for t in self.ts])

    def ftv(self):
        if len(self.ts) == 0:
            return self.con.ftv()
        return set.union(self.con.ftv(), *[t.ftv() for t in self.ts])

    def __hash__(self):
        return hash((TypeCall, self.con, tuple(self.ts)))

    def __eq__(self, other):
        if isinstance(other, TypeCall):
            if self.con == other.con:
                if len(self.ts) == len(other.ts):
                    return all(t == s for t, s in zip(self.ts, other.ts))
        return NotImplemented

class TaggedType(Type):
    def __init__(self, tag, t):
        if not isinstance(tag, str):
            raise TypeError('TaggedType tags must be strings')
        self.tag = tag
        self.t = t

    def __repr__(self):
        return f'TaggedType({self.tag!r}, {self.t!r})'

    def __str__(self):
        return f'!{self.tag} {self.t}'

    def apply(self, subst):
        return TaggedType(self.tag, self.t.apply(subst))

    def ftv(self):
        return self.t.ftv()

    def __hash__(self):
        return hash((TaggedType, self.tag, self.t))

    def __eq__(self, other):
        if isinstance(other, TaggedType):
            return self.tag == other.tag and self.t == other.t
        return NotImplemented

    def kind(self):
        return K.Star

class ListType(Type):
    def __init__(self, t):
        self.t = t

    def __repr__(self):
        return f'ListType({self.t!r})'

    def __str__(self):
        return f'[{self.t}]'

    def apply(self, subst):
        return ListType(self.t.apply(subst))

    def ftv(self):
        return self.t.ftv()

    def __hash__(self):
        return hash((ListType, self.t))

    def __eq__(self, other):
        if isinstance(other, ListType):
            return self.t == other.t
        return NotImplemented

    def kind(self):
        return K.Star

class SetType(Type):
    def __init__(self, t):
        self.t = t

    def __repr__(self):
        return f'SetType({self.t!r})'

    def __str__(self):
        return f'{{{self.t}}}'

    def apply(self, subst):
        return SetType(self.t.apply(subst))

    def ftv(self):
        return self.t.ftv()

    def __eq__(self, other):
        if isinstance(other, SetType):
            return self.t == other.t
        return NotImplemented

    def __hash__(self):
        return hash((SetType, self.t))

    def kind(self):
        return K.Star

class DictType(Type):
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __repr__(self):
        return f'DictType({self.k!r}, {self.v!r})'

    def __str__(self):
        return f'{{{self.k}: {self.v}}}'

    def apply(self, subst):
        return DictType(self.k.apply(subst), self.v.apply(subst))

    def ftv(self):
        return set.union(self.k.ftv(), self.v.ftv())

    def __eq__(self, other):
        if isinstance(other, DictType):
            return self.k == other.k and self.v == other.v
        return NotImplemented

    def __hash__(self):
        return hash((DictType, self.t))

    def kind(self):
        return K.Star

class SignatureType(Type):
    def __init__(self, types, names):
        self.types = types
        self.names = names

    def __repr__(self):
        return f'SignatureType({self.types!r}, {self.names!r})'

    def __str__(self):
        types = '\n'.join(f'\t\t\t\t{k}:{v}' for k, v in self.types.items())
        names = '\n'.join(f'\t\t\t\t{k}:{v}' for k, v in self.names.items())
        return f'signature(\n{types}\n{names}\n\t\t\t)'

    def __eq__(self, other):
        if isinstance(other, SignatureType):
            return self.types == other.types and self.names == other.names
        return NotImplemented

    def __hash__(self):
        return hash((SignatureType, self.types, self.names))

    def ftv(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError

    def kind(self):
        return K.Star

class FunctionType(Type):
    def __init__(self, p, r):
        if not isinstance(p, (TypeVariable, AnyType, ArgsType, ParamsType)):
            print(p)
            raise NotImplementedError
        self.p = p
        self.r = r

    def __repr__(self):
        return f'FunctionType({self.p!r}, {self.r!r})'

    def __str__(self):
        return f'({self.p} → {self.r})'

    def apply(self, subst):
        # print('FunctionType', repr(self), 'SUBST', subst)
        return FunctionType(self.p.apply(subst), self.r.apply(subst))

    def ftv(self):
        return set.union(self.p.ftv(), self.r.ftv())

    def __hash__(self):
        return hash((FunctionType, self.p, self.r))

    def __eq__(self, other):
        if isinstance(other, FunctionType):
            return self.p == other.p and self.r == other.r
        return NotImplemented

    def kind(self):
        return K.Star

class TypeScheme:
    def __init__(self, tvars, t):
        self.tvars = tvars
        self.t = t
        # print('TSCHEME', type(self.t))

    def __repr__(self):
        return f'TypeScheme({self.tvars!r}, {self.t!r})'

    def __str__(self):
        v = ','.join(map(str, self.tvars))
        return f'∀{v}.{self.t}'

    def apply(self, subst):
        # print('TypeScheme', repr(self), 'SUBST', subst)
        #print(self.tvars, self.t, subst)
        #print(repr(self.tvars), repr(self.t), repr(subst))
        s = {k: v for k, v in subst.items() if k not in self.tvars}
        return TypeScheme(self.tvars, self.t.apply(s))

    def ftv(self):
        return self.t.ftv() - set(self.tvars)

    def __hash__(self):
        return hash((TypeScheme, self.tvars, self.t))

class Nullary:
    def __init__(self, t):
        self.t = t

    def __str__(self):
        return str(self.t)

    def apply(self, subst):
        # print('Nullary', repr(self), 'SUBST', subst)
        return Nullary(self.t.apply(subst))

    def __eq__(self):
        pass

    def __hash__(self):
        return hash((self.t,))

    def ftv(self):
        return self.t.ftv()

class MuType(Type):
    def __init__(self, tv, t):
        self.tv = tv
        self.t = t

    def roll(self):
        return MuType(self.tv, self.t.apply({self.tv: self.t}))

    def __repr__(self):
        return f'MuType({self.tv!r}, {self.t!r})'

    def __str__(self):
        return f'μ{self.tv}.{self.t}'

    def ftv(self):
        # print('FREE TYPE VARIABLES OF MUTYPE', repr(self.t.ftv()))
        return self.t.ftv() - {self.tv.tvar}

    def apply(self, s):
        # print('S', s)
        s1 = {k: v for k, v in s.items() if k != self.tv}
        # print('S1', s1)
        # print('SELF.T', self.t, repr(self.t))
        # print('SELF.T_APP', self.t.apply(s1), repr(self.t.apply(s1)))
        return MuType(self.tv, self.t.apply(s1))

    def __eq__(self, other):
        if isinstance(other, MuType):
            raise NotImplementedError
        return NotImplemented

    def __hash__(self):
        return hash((self.tv, self.t))

class TypeConstructor:
    pass

class AbstractTypeConstructor(TypeConstructor):
    def __init__(self, kind):
        self.kind = kind

    def __repr__(self):
        return f'AbstractTypeConstructor({self.kind!r})'

    def __str__(self):
        return f'(_ :: {self.kind})'

    def apply(self, subst):
        if self in subst:
            return subst[self]
        return self

    def __hash__(self):
        return hash((AbstractTypeConstructor, self.kind))

    def ftv(self):
        return set()

class ConcreteTypeConstructor(TypeConstructor):
    def __init__(self, param_kinds, result_kind, tvars, expr):
        if len(param_kinds) != len(tvars):
            raise ApeInternalError(pos=None, msg='Must pass the same number of parameter kinds and type variables')
        self.param_kinds = param_kinds
        self.result_kind = result_kind
        self.tvars = tvars
        self.expr = expr
        self.kind = K.ArrowKind(param_kinds, result_kind)

    def __repr__(self):
        return f'ConcreteTypeConstructor({self.param_kinds!r}, {self.result_kind!r}, {self.tvars!r}, {self.expr!r})'

    def __str__(self):
        v = ','.join(map(str, self.tvars))
        return f'(Λ{v}.{self.expr} :: {self.kind})'

    def apply(self, s):
        s1 = {k: v for k, v in s.items() if k not in self.tvars}
        return ConcreteTypeConstructor(self.param_kinds, self.result_kind,
                                       self.tvars, self.expr.apply(s1))
        #if self in subst:
        #    return subst[self]
        #return ConcreteTypeConstructor(self.kind, self.expr.apply(subst))

    def ftv(self):
        return self.expr.ftv() - set(self.tvars)
