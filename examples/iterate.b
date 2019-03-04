# An attempt to replicate the ITERATE macro of Common Lisp fame
# Requires empty suite syntax to be usable: ';;' or something
# 
# EXAMPLE:
#
#     iterate!
#     for i from 1 upto m!
#     for k from 1 upto p!
#     setting C(i, k) to 0!
#     for j from 1 upto n!
#     summing A(i, j) * B(j, k) into C(i, k)!

iterate = 'iterate is still an identifier here!'

macro iterate(comps):
    return \(do: pass)

control_structure (iterate) \
    (repeat <expr> !) \
    (for <id> {(from <expr>) (upfrom <expr>) (downfrom <expr>) \
               (to <expr>) (upto <expr>) (downto <expr>) \
	       (above <expr>) (below <expr>) (by <expr>)}* !) \
    (foreach <exprlist> !)
    (for <exprlist> in <expr> (by <expr>)) \
    (for <exprlist> on <expr>) \
    (for <exprlist> on <expr> by <expr>)
    (for <exprlist> next <expr>) \
    (for <exprlist> iter <expr> t>) \
    (summing <expr>) (summing <expr> into <id>) \
    (counting <expr>) (counting <expr> into <id>) \
    (multiplying <expr>) (multiplying <expr> into <id>) \
    (maximising <expr>) (maximising <expr> into <id>) \
    (minimising <expr>) (minimising <expr> into <id>) \
    (reducing <expr> by <expr>) \
    (reducing <expr> into <id>) \
    (reducing <expr> initial_value <expr>) \
    (reducing <expr> initial_value <expr> into <id>) \
    (collect <id>) (collect <id> into <id>) \
    (accumulate <expr> by <expr>) \
    (accumulate <expr> by <expr> into <id>)



iterate:
    for i upfrom 0: ()
    for i from 5: ()
    for i downfrom 5: ()
    for i from 1 to 3: ()
    for i from 1 below 3: ()
    for i from 1 to 3 by 2: ()
    for i from 1 below 3 by 2: ()
    for i from 5 downto 3: ()
