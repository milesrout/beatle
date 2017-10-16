def foo(x):
    y = yield x
    z = yield y
    w = yield z
    return w
    #raise RuntimeError('hi')

print(str(foo(1)))
print(str(foo(2)))
