macro until(arm):
    cond, body = arm
    \(while not $(cond):
          $body)

control_structure (until <test>)

x = 0
until x == 10:
    print(x)
    x = x + 1


type list[a] = !nil | !cons (a, list[a])

x = !cons (1, !cons (2, !cons (3, !nil)))
y = (until x == !nil:
        (!cons (a, x)) = x
        print('and', x, a))

print('done', x, y)
