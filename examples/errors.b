def foo():
    raise (yield 1) from (yield 2)
