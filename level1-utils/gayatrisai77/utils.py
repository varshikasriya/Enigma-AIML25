import math


def sigmoid(x):
    return 1/(1+math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return (math.exp(x) - math.exp(-x))/(math.exp(x)+math.exp(-x))

#dotproduct
def dotproduct(x,y):
    ans=0
    for i in range(len(x)):
        ans+=x[i]*y[i]
    return ans    

def cossimilarity(a, b):
    return dotproduct(a, b) / (math.sqrt(dotproduct(a,a)) * math.sqrt(dotproduct(b,b)))

#normalization ranges from [0,1]
def norm(arr):
    minval=min(arr)
    maxval=max(arr)
    ansnorm=[]
    for x in arr:
      val=(x-minval)/(maxval-minval)
      ansnorm.append(val)
    return ansnorm

#l1 norm
def l1norm(arr):
    total=0
    for x in arr:
      total+=abs(x)
    for x in arr:
      norm.append(x/total)
    return norm

#l2 norm
def l2norm(arr):
    total = 0
    for x in arr:
        total += x**2 
    norm = math.sqrt(total)
    normarr=[]
    for x in arr:
      normarr.append(x/norm)
    return normarr 

"""tictactoe greedy approch"""
""" the best immediate move : if player can win in the next move or opponent can win intheir next move or else an empty cell is choosen next.
"""

def check_winner(board, player):
    
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] == player:
            return True
        if board[0][i] == board[1][i] == board[2][i] == player:
            return True
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False

def greedymove(board, player):
    opponent = 'O' if player == 'X' else 'X'

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = player
                if check_winner(board, player):
                    board[i][j] = ' '
                    return (i, j)
                board[i][j] = ' '

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = opponent
                if check_winner(board, opponent):
                    board[i][j] = ' '
                    return (i, j)
                board[i][j] = ' '

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                return (i, j)
"""if for exmple board looks like this """
board=[
    ['X', 'O', 'X'],
    [' ', 'O', ' '],
    [' ', ' ', ' ']
]

move = greedymove(board, 'O')
print("greedy move for O iss", move)
