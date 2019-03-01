a = 0
b = 1
c = 3
zero_result = 0
nonzero_result = 1

def foo(x):
    print(x + (yield from x))
    pass

if a == 0:
    if b:
        if c:
            pass
            #(print("hello"),
            # print("hello"))
        print("goodbye")
    print("world")
print("okay")


if a == 0 or b == 1:
    a = (if a == 0:
             zero_result
         else:
             nonzero_result)
    b = (if b == 0:
             zero_result
         else:
             nonzero_result)
    print(a, b)

# we don't do macroexpansion yet - need to do it before we do
# type-checking unless I use structured binding types (λ_s calculus) to
# have type-checked hygenic macros.
# 2019: why did I ever think that quasiquoting had anything to do with macros??

x = (
        1,
        2,
        3
    )
print('Hello1')

print(\[$x])
print([x])

print('Hello2')
print(\{1, 2, 3})
print('Hello3')
print(\${1, 2, 3})
print('Hello4')


print((def(a):
          confuser = (
              def(b = (yield)):
                  yield from b
          )
          return None))

def foo():
    print_hello = (def(a, b, c):
                       a; b; c
                       print(def():
                               print()
                               print(
                                  (1, 2, 3),
                                (a, b, c)))
                       print('Hello',
                             (def():
                                 print(1, 2, 3); print(a, b, c)
                                 pass),
                        'World'))
print(foo)
