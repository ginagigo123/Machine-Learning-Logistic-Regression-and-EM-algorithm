# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:03:33 2021

@author: ginag
content:
    Logistic Regression:
        1. Gradient Descent
        2. Newton's method
"""
import numpy as np # for normal distribution
import matplotlib.pyplot as plt
import math
from scipy.special import expit

# Generate value from normal distribution - Box Muller
def Box_Muller_method(mean, var):
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)
    X = math.sqrt(-2 * math.log(U)) * math.cos(2 * math.pi * V)
    Y = math.sqrt(-2 * math.log(U)) * math.sin(2 * math.pi * V)
    X = mean + math.sqrt(var) * X
    return X

def generate_Distribution(n, xmean, xvar, ymean, yvar):
    D = np.zeros(shape=(n, 2))
    for i in range(n):
        # x from N(mx1, vx1), y from N(my1, vy1)
        D[i, 0] = Box_Muller_method(xmean, xvar)
        D[i, 1] = Box_Muller_method(ymean, yvar)
    return D
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Gradient_descent(X, w, y, lr=0.01):
    gradient = 100
    
    print("Gradient Descent:\n")
    while np.sqrt(np.sum(gradient ** 2)) > 0.01:
        gradient = np.transpose(X) @ ( y - sigmoid( X @ w))
        w = w + lr * gradient
        
        """
        gradient = np.zeros(len(w))
        for i in range(len(X)):
            gradient += np.transpose(X) @ (y[i] - sigmoid( X[i] @ w) )
        """
        #print("Gradient:", gradient)

        #print_W(w)
        #print()
    print_W(w)
    return w

def Newton_Method(X, w, y, lr=0.01):
    # w = w + Hession.inv() @ gradient
    # original math : x[n] = x[n-1] - f'(x[n]) / f''(x[n]) => x[n] - gradeint / Hession
    delta = 100
    time = 0
    print("Newton's method':\n")
    while np.sqrt(np.sum(delta ** 2)) > 0.01:
        gradient = np.transpose(X) @ ( y - sigmoid(X @ w))
        
        D = np.zeros(shape=(2*n, 2*n))
        for i in range(2 * n):
            D[i , i] = np.exp(- X[i] @ w ) / (1 + np.exp(- X[i] @ w) ** 2 )
        
        Hession = np.transpose(X) @ D @ X
        delta = np.linalg.inv(Hession) @ gradient
        w = w + lr * delta    
        #print_W(w)
        #print()
        if time == 0:
            fdelta = delta
        time += 1
        
    print("First delta\n", fdelta)
    print("Final delta\n", delta)
    print("Total time:", time)
    print_W(w)
    return w

def getX(D1, D2):
    X = np.append(D1, D2, axis=0)
    # insert 1 to column 1
    X = np.insert(X, 0, 1, axis=1)
    return X

def getY(n):
    # data from distribution 1: class 0, distribution 2: class 1
    # first 50: class 0 // last 50: class 1
    return np.array([0 if i < n else 1 for i in range(2*n)]).reshape(2*n, 1)

def print_W(w):
    print("w:")
    for i in w:
        print(i)
    print()
        
def print_ConfusionMatrix(X, w, y):
    '''
    class 0 : postive, class 1 : negative
    TP / FN
    FP / TN
    '''
    print("Confusion Matrix:")
    predict = sigmoid (  X @ w )
    
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(2 * n):
        if predict[i] > 0.5 :
            predict[i] = 1
        else :
            predict[i] = 0
        # 0 for positive, 1 for negative
        if y[i] == 0 and predict[i] == 0:
            TP += 1
        elif y[i] == 1 and predict[i] == 1:
            TN += 1
        elif y[i] == 0 and predict[i] == 1:
            FN += 1
        elif y[i] == 1 and predict[i] == 0:
            FP += 1
    
    print(" " * 12,"Predict Cluster 1 / Predict Cluser 2")
    print("Is cluster 1 {value:^18}{value1:^18}".format(value=TP, value1=FN))
    print("Is cluster 2 {value:^18}{value1:^18}".format(value=FP, value1=TN))
    
    print("\nSensitivity (Successfully predict cluster 1): ", TP / (TP+FN) )
    print("Specificity (Successfully predict cluster 2): ", TN / (TN+FP) )
    
    return predict

def plot_figures(D1, D2, Gpredict_D1, Gpredict_D2, Npredict_D1, Npredict_D2):
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))

    axes[0].scatter(D1[:, 0], D1[:, 1], c="r")
    axes[0].scatter(D2[:, 0], D2[:, 1], c="b")
    axes[0].set_title("Ground Truth")
    
    axes[1].scatter(Gpredict_D1[:, 0], Gpredict_D1[:, 1], c="r")
    axes[1].scatter(Gpredict_D2[:, 0], Gpredict_D2[:, 1], c="b")
    axes[1].set_title("Gradient Descent")
    
    axes[2].scatter(Npredict_D1[:, 0], Npredict_D1[:, 1], c="r")
    axes[2].scatter(Npredict_D2[:, 0], Npredict_D2[:, 1], c="b")
    axes[2].set_title("Newton's method")

n = int(input("n:"))
basis = 3
mx1, my1, mx2, my2  = map(int, input("(mx1, my1, mx2, my2):").split(" "))
vx1, vy1, vx2, vy2 = map(int, input("(vx1, vy1, vx2, vy2):").split(" "))


# generate n data point
D1 = generate_Distribution(n, mx1, vx1, my1, vy1)
D2 = generate_Distribution(n, mx2, vx2, my2, vy2)

# check their distribution
#plt.scatter(D1[:, 0], D1[:, 1])
#plt.scatter(D2[:, 0], D2[:, 1])

# the decision boundary we want to find is h(x) = w0 + w1x1 + w2x2
# the x2 (vertical): -(w0 + w1x1) / (w2)
w = np.random.rand(basis, 1)
X = getX(D1, D2)
y = getY(n)

# Gradient Descent method (steepest)
w = Gradient_descent(X, w, y)
predict_y = print_ConfusionMatrix(X, w, y)

# plotting the result Graph
cluster1 = np.where(predict_y == 0)[0]
cluster2 = np.where(predict_y == 1)[0]
D =  np.append(D1, D2, axis=0)
Gpredict_D1, Gpredict_D2 = D[ cluster1 , : ], D[ cluster2 , :]

# Newton Method
w = np.random.rand(basis, 1)
w = Newton_Method(X, w, y)  
Newton_Predict_y = print_ConfusionMatrix(X, w, y)

cluster1 = np.where( Newton_Predict_y == 0)[0]
cluster2 = np.where( Newton_Predict_y == 1)[0]

Newton_D1, Newton_D2 = D[ cluster1, :], D[ cluster2, :]
plot_figures(D1, D2, Gpredict_D1, Gpredict_D2, Newton_D1, Newton_D2)





