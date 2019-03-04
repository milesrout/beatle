macro until(arms):
    if len(arms) == 1:
        cond, body = arms[0]
        \(while not $(cond[0]):
              $@body)
    if len(arms) == 2:
        cond, body = arms[0]
        [], alt = arms[1]
        \(while not $(cond[0]):
              $@body
          else:
              $@alt)
    raise AssertionError('malformed \'unless\' statement')

control_structure (until <test>)

type maybe[a] = !nothing | !something a

x: any = !something !something !something 1
x = (until (!nothing) == x:
        (!something x) = x)
