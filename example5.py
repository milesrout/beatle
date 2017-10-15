def send_many(ch, msgs):
    match msgs:
        case []:
            pass
        case [x, *xs]:
            ch.send(x)
            recur(ch, xs)
