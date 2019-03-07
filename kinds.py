from abc import ABCMeta, abstractmethod

class Kind(metaclass=ABCMeta):
    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def apply(self, subst: dict):
        ...

    @abstractmethod
    def fkv(self) -> set:
        ...

class KindConstant(Kind):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        if self is Star:
            return 'Star'
        return f'KindConstant({self.name!r})'

    def __str__(self):
        return self.name

    def apply(self, subst):
        return self

    def fkv(self):
        return set()
Star = KindConstant('★')

class KindVariable(Kind):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'KindVariable({self.name!r})'

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

    def __repr__(self):
        return f'ArrowKind({self.ks!r}, {self.k!r})'

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

UnaryKind = ArrowKind((Star,), Star)
TernaryKind = ArrowKind((Star, Star, Star), Star)
