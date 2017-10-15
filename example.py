a = 1
b = 2
c = 3
d = 'hello'
h = set()
i = {'a': 1, 'b': 2}
j = {'c': 3, **i}
k = {'a', 'b', *h}
l = {1, 2, 3}
m = [4, 5, 6]
zero_result = 0
nonzero_result = 1

def foo(x, y, z, *q):
    pass

def bar(x, *, bar=None):
    pass

def baz(trailing_comma=1, **foo):
    pass

if a != 0:
    if b != 0:
        if c != 0:
            print("hello")
            print("hello")
        print("goodbye")
    print("world")
print("okay")


if a == 0 or b == 1:
    a = (if b == 0:
             zero_result
         else:
             nonzero_result)
    b = (if b == 0:
             zero_result
         else:
             nonzero_result)
    raise RuntimeError('that is not valid!')
    print(str(a + b))

interface IQueue:
    Queue[a]: type

    QueueError: Error
    empty: Queue[a]
    is_empty: Queue[a] -> bool
    singleton: a -> Queue[a]
    insert: (a, Queue[a]) -> Queue[a]
    peek: Queue[a] -> a
    remove: Queue[a] -> (a, Queue[a])

# we don't do macroexpansion yet - need to do it before we do type-checking.
# macro definitions are still parsed, but are removed by the macroexpander.
# they just aren't actually *applied*.

# unless I use structured binding types (λ_s calculus) to have type-checked
# hygenic macros???

macro log(e: Expression):
    print(\[$e])
    print([e])
    \${1, 2, 3}
    return \print(str(e), $e)


x = (
        1,
        2,
        3
    )

#log(x)

print(str((1.0, 2.0, 3.0)))

def foo():
    print_hello = (def(a, b, c):
                       a; b; c
                       print(str(def():
                                   print('what?')
                                   print(
                                      str(1, 2, 3) +
                                    str(a, b, c))))
                       print(str(['Hello',
                             str((def():
                                     print(str((1, 2, 3))); print(str(a, b, c))
                                     pass)),
                        'World'])))
