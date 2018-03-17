from stack import Stack
from queue import Queue

module TwoStackQueue(S: Stack) -> Queue:
    type queue[a] = (S.stack[a], S.stack[a])

    queue_error = Error("Queue error")
    empty = (S.empty, S.empty)

    def is_empty(s1, s2): S.is_empty(s1) and S.is_empty(s2)

    def singleton(x): (S.empty, S.singleton(x))

    def insert(x, q):
        match q:
            case (S.empty, S.empty): (S.empty, S.singleton(x))
            case (s1, s2): (S.insert(x, s1), s2)

    def peek(q):
        match q:
            case (_, S.empty): raise queue_error
            case (_, s2): S.pop(s2)

    def remove(q):
        ...


