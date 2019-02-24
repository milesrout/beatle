def bar(x, y: int, z: int? = nothing, w=True):
    return x, y, z, w

def baz(a: int):
    return (yield a)

def lorem(x: int):
    y = yield x
    z = yield y
    w = yield z
    raise RuntimeError('hi')
    return w

def ipsum(x: int):
    y = yield x
    z = yield y
    w = yield z
    raise RuntimeError('hi')
    return w

print(str(1))
print(str(bar(1, 2)))
print(str(bar(2, 3)))
print(str(bar(2, 3, something(2), False)))

result = iter(lorem(1))
while True:
    match result:
        case (0, r):
            print('done')
            return r
        case (1, (y, c)):
            print(y)
            result = next(c)
