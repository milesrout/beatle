signature Coroutine:
    start: ∀y,s,r. (g: generable[y,s,r], /) → either[r, y ⨯ generator[y,s,r]]
    send:  ∀y,s,r. (s: s, g: generator[y,s,r], /) → either[r, y ⨯ generator[y,s,r]]

iter: ∀y,r. generable[y,(),r] → either[r, y ⨯ generator[y,(),r]]
next: ∀y,r. generator[y,(),r] → either[r, y ⨯ generator[y,(),r]]
