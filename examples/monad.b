signature Functor:
    type f[a]
    
    map: ∀a,b. (a → b) → f[a] → f[b]

signature Monad:
    type m[a]

    pure: ∀a.   a → m[a]
    bind: ∀a,b. (a → m[b]) → m[a] → m[b]
