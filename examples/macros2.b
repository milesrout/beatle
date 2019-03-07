DEBUG = True

macro ASSERT(e):
    \(if DEBUG:
        if not $e:
            raise AssertionError($(stringify(e))))

a = 1
b = 2

ASSERT(a + b == 3)

print('test')
