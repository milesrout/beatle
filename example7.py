interface IQueue:
    type Queue[a]

    QueueError:        error
    empty:       ∀a,b. Queue[a]
    is_empty:    ∀a.   Queue[a] → bool
    singleton:   ∀a.   a → Queue[a]
    insert:      ∀a.   (a, Queue[a]) → Queue[a]
    peek:        ∀a.   Queue[a] → a
    remove:      ∀a.   Queue[a] → (a, Queue[a])
