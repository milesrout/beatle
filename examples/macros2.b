macro ASSERT(e):
    g = gensym()
    \(if 1:
         $g = $e
         if not $g:
             raise AssertionError(str(e)))
     

ASSERT(1 + 2 == 3)
