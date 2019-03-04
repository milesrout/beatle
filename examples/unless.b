unless = 'unless is still an identifier here'

macro unless(arms):
    if len(arms) == 1:
        [cond], body = arms[0]
        \(if not $cond:
              $@body)
    if len(arms) == 2:
        [cond], body = arms[0]
        [], alt = arms[1]
        \(if not $cond:
              $@body
          else:
              $@alt)
    raise AssertionError('malformed \'unless\' statement')
#    if len(arms) == 1:
#        \(if not $(cadar(exprs)):
#              $@(cddar(exprs)))
#    elif len(arms) == 2:
#        \(if not $(cadar(exprs)):
#              $@(cddar(exprs))
#          else:
#              $@(cddadr(exprs)))
#    else:
#        raise AssertionError('unless must have exactly two arms')

control_structure (unless <test>) (else)
control_structure (with_ <expr> &as <id>)

# after the directive it becomes a keyword

# we can't call it like this anymore! Probably not a big deal, but if it is,
# some fiddling around in parser.py should sort it.
#unless(x == 1, (do:
#                    print(str(x))))

x = 1
unless x == 1:
    print(str(x))
