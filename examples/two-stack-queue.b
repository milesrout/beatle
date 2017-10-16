from stack import Stack
from queue import Queue

class TwoStackQueue(S: Stack) -> Queue:
    type queue[a] = (S.stack[a], S.stack[a])

    queue_error = Error("Queue error")
    empty = (S.empty, S.empty)

    def is_empty((s1, s2)): S.is_empty(s1) and S.is_empty(s2)

    def singleton(x): (S.empty, S.singleton(x))

    def insert(x, q):
        match q:
            (S.empty, S.empty): (S.empty, S.singleton(x))
            (s1, s2): (S.insert(x, s1), s2)

    def peek(q):
        match q:
            (_, S.empty): raise queue_error
            (_, s2): S.pop(s2)

    def remove(q):
        ...


