if a:
    if b:
        if c:
            (print("hello"),
             print("hello"))
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
    print(a, b)

print(\[$x])
print([x])

\${1, 2, 3}

x = (
        1,
        2,
        3
    )


def bar():
    confuser = (
        def(a = (yield)):
            yield from a
    )

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
