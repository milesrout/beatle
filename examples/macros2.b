DEBUG = True

macro ASSERT(e):
    \(if DEBUG:
        if not $e:
            raise AssertionError($(stringify(e))))

a = 1
b = 2
ASSERT(a + b == 2)
#try:
#    ASSERT(a + b == 2)
#except AssertionError as e:
#    print('Error occurred:' + reason(e))

print('test')
