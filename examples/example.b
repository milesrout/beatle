a = 1
b = 2
c = 3
y = x = True
d = 'hello'
h = set()
i = {'a': 1, 'b': 2}
j = {'c': 3, **i}
k = {'a', 'b', *h}
l = {1, 2, 3}
m = [4, 5, 6]
zero_result = 0
nonzero_result = 1

def foo(x, y, z):
    raise RuntimeError('what?')
    #yield x
    #yield y
    #yield z

def bar(x, bar=None):
    pass

def baz(trailing_comma=1):
    pass

if a != 0:
    if b != 0:
        if c != 0:
            print("hello")
            print("hello")
        print("goodbye")
    print("world")
print("okay")

if x:
    if y:
        if (a == b
          or
        c
      ==
   d):
          pass
if a == 0 or b == 1:
    a = (if b == 0:
             "hello"
         else:
             "world")
    a = (if b == 0:
             2
         else:
             3)
    raise RuntimeError('that is not valid!')
    print(str(a + b))

signature Queue:
    type queue[a]

    queue_error:     error
    empty:       ∀a. queue[a]
    is_empty:    ∀a. queue[a] → bool
    singleton:   ∀a. a → queue[a]
    insert:      ∀a. (a, queue[a]) → queue[a]
    peek:        ∀a. queue[a] → a
    remove:      ∀a. queue[a] → (a, queue[a])

    law ∀x. is_empty(x) == (x is empty)
    law ∀x. is_empty(x) == (peek(x) is queue_error)

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
