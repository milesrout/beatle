def send(m, x):
    match x:
        case None:
            pass
        case Msgs [x, xs...]:
            m <- x
            recur(m, xs)

def send(m, [x, xs...]):
    m <- x
    recur(m, xs)

def send(m, []):
    pass
