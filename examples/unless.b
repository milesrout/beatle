unless = 'unless is still an identifier here'

macro unless(exprs):
    if len(exprs) == 1:
        return \(do:
                     if not $(cadar(exprs)):
                         $@(cddar(exprs)))
    else:
        return \(do:
                     if not $(cadar(exprs)):
                         $@(cddar(exprs))
                     else:
                         $@(cddadr(exprs)))

control_structure (unless <expr>) (else)

# after the directive it becomes a keyword

x = 1
unless x == 1:
    print(str(x))
