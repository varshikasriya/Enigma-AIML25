import math as m

def sigmoid(x):
    return 1/(1+m.exp(-x))

def relu(x):
    return max(0,x)

def tanh(x):
    return m.tanh(x)

def dot(x,y):
    result=0
    for i in range(len(x)):
        result+=x[i]*y[i]
    return result

def cs(x,y):
    dotp=dot(x,y)
    magx=m.sqrt(dot(x,x))
    magy=m.sqrt(dot(y,y))
    return dotp/(magx*magy)

def minmax(x):
    norm=[]
    for i in x:
        norm.append((i-min(x))/(max(x)-min(x)))
    return norm

def l1(x):
    norm=[]
    for i in x:
        norm.append(i/sum(abs(i) for i in x))
    return norm

def l2(x):
    norm=[]
    for i in x:
        norm.append(i/m.sqrt(sum(i**2 for i in x)))
    return norm

def heuristic(a,b):
    count=0
    for i in range(3):
        for j in range(3):
            if (a[i][j]!=b[i][j]):
                count+=1
    return count
