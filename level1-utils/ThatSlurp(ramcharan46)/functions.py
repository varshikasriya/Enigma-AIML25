import math as m

# 1)Sigmoid=1/(1+e**(-x))
def sigmoid(x):
    return 1/(1+m.exp(-x))

# 2)ReLU=max(0,x)
def relu(x):
    return max(0,x)

# 3)Tanh=sinh(x)/cosh(x)
def tanh(x):
    return m.tanh(x)

# 4)Vector dot product
def dot(x,y):
    result=0
    for i in range(len(x)):
        result+=x[i]*y[i]
    return result

# 5)Cosine similarity(taking 2 arrays)
def cs(x,y):
    dotp=dot(x,y)
    magx=m.sqrt(dot(x,x))
    magy=m.sqrt(dot(y,y))
    return dotp/(magx*magy)

# 6)Normalisation(L1,L2,minmax)
#---->6A)Min-Max Normalisation
def minmax(x):
    norm=[]
    for i in x:
        norm.append((i-min(x))/(max(x)-min(x)))
    return norm
#---->6B)L1 Normalisation
def l1(x):
    norm=[]
    for i in x:
        norm.append(i/sum(abs(i) for i in x))
    return norm
#---->6C)L2 Normalisation
def l2(x):
    norm=[]
    for i in x:
        norm.append(i/m.sqrt(sum(i**2 for i in x)))
    return norm

# 7)Heuristic function
''' This heuristic function will take 2 3*3 grid arrays filled with numbers 1-9 in any order,
    the first grid is the current grid and the second grid is the goal grid,
    this function will calculate the count of mismatch tiles between the 2 grids and return that count.'''
def heuristic(a,b):
    count=0
    for i in range(3):
        for j in range(3):
            if (a[i][j]!=b[i][j]):
                count+=1
    return count