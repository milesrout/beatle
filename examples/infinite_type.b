DEBUG = True

macro ASSERT(e):
    return \(if DEBUG:
                 if not $e:
                     raise AssertionError($(str(e))))


def assert_equal(x, z):
    # x and z are compared so they must have the same type
    ASSERT(x == z)
    1

def foo(x):
    # x is used in an addition to an integer, so it gets unified with integer
    y = x + 1

    z = "hello"
    
    # assert_equal takes two parameters of the same type, so these must have the same type
    print(str(assert_equal(str(x), z)))

    ASSERT(str(x) == "hello")

foo(10)
