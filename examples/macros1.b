import macros2

macro log(e):
    \(print($(str(e)), str($e)))

a, b = 0, 1
log(5)
log(a + b)

ASSERT(True)
