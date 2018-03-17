signature Stack:
    type stack[a]

    stack_error:     error
    empty:       ∀a. stack[a]
    is_empty:    ∀a. stack[a] → bool
    singleton:   ∀a. a → stack[a]
    insert:      ∀a. (a, stack[a]) → stack[a]
    peek:        ∀a. stack[a] → a
    remove:      ∀a. stack[a] → (a, stack[a])

    law ∀x. is_empty(x) == (x is empty)
