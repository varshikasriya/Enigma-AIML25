import math

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

print(sigmoid(7)) 
print(relu(7))
print(tanh(7))



def dot(x,y):
    r=0
    for i in range(len(x)):
        r+=x[i]*y[i]
    return r





def cs(x,y):
    p=dot(x,y)
    mx=m.sqrt(dot(x,x))
    my=m.sqrt(dot(y,y))
    return p/(mx*my)







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

'''
this is a clothing heuristic function based on 
temperature it will suggest 
what to wear based on the temperature mentioned'''



def clothing_heur(temp):
    if temp < 15: 
        return "Put on The  jacket"
    elif temp < 25: 
        return "Unequip The  jaceket"
    else: 
        return "CHoice of cloths is urs"