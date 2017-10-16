signature Queue:
    type queue[a]

    queue_error:     error
    empty:       ∀a. queue[a]
    is_empty:    ∀a. queue[a] → bool
    singleton:   ∀a. a → queue[a]
    insert:      ∀a. (a, queue[a]) → queue[a]
    peek:        ∀a. queue[a] → a
    remove:      ∀a. queue[a] → (a, queue[a])

    law ∀x. is_empty(x) == (x is empty)
