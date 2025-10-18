import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

def vecdot(x,y):
    r=0
    for i in range(len(x)):
        r+=x[i]*y[i]
    return r

def cosine(x,y):
    v = vecdot(x, y)
    m1 = math.sqrt(vecdot(x, x))
    m2 = math.sqrt(vecdot(y, y))
    if m1 == 0 or m2 == 0:
        return 0.0
    return v / (m1 * m2)

def minmax(x):
    norm=[]
    min_x = min(x)
    max_x = max(x)
    if max_x == min_x:
        return [0.0] * len(x)

    for i in x:
        norm.append((i - min_x) / (max_x - min_x))
    return norm


def l1(x):
    norm = []
    l1_norm = sum(abs(i) for i in x)

    if l1_norm == 0:
        return [0.0] * len(x)

    for i in x:
        norm.append(i / l1_norm)
    return norm

def l2(x):
    norm=[]
    l2_norm = math.sqrt(sum(i**2 for i in x))

    if l2_norm == 0:
        return [0.0] * len(x)

    for i in x:
        norm.append(i / l2_norm)
    return norm

##chess strat

def chesstrat(pieces):
    values = {
        'Pawn': 1,
        'Knight': 3,
        'Bishop': 3,
        'Rook': 5,
        'Queen': 9
    }
    white = 0
    black = 0

    for piece, player in pieces:
        if piece == 'King':
            continue
        value = values.get(piece, 0)

        if player == 'White':
            white += value
        elif player == 'Black':
            black += value

    return white - black

    #positive score means white is in advantage and negative means black is