# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:56:31 2021

@author: ginag
"""
import gzip
import numpy as np
import math
from tqdm import trange

def training_images():
    with gzip.open("data/train-images-idx3-ubyte.gz", 'r') as f:
        """
            int.from_bytes: bytes -> integer
            byteorder = 'big' -> big endian
        """
        # first 4 bytes = magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes = number of images
        num_image = int.from_bytes(f.read(4), 'big')
        # thrid 4 bytes = row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes = col count
        col_count = int.from_bytes(f.read(4), 'big')
        
        # rest = image pixel data, each pixel is stored as unsigned byte
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_image, row_count, col_count)
        return images
    
def training_label():
    with gzip.open("data/train-labels-idx1-ubyte.gz", 'r') as f:
        # first 4 bytes = magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes = number of images
        num_label = int.from_bytes(f.read(4), 'big')
        
        # rest = label data, each label is stored as unsigned byte
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8).reshape(num_label, 1)
        return labels

def print_num(x, threshold=True):
    for i in range(28):
        for j in range(28):
            if threshold == True:
                if x[i][j] > 0.85:
                    print("*", end=" ")
                else:
                    print("-", end=" ")
            else:
                if x[i][j] < 128:
                    print("-", end=" ")
                else:
                    print("*", end=" ")
        print()

def print_Imagination(x, mapping):
    print("Imagination of numbers:")
    for label in range(10):
        print(label, ":")
        real_label = mapping[label]
        for i in range(28):
            for j in range(28):
                if x[real_label, i, j] > 0.5:
                        print("*", end=" ")
                else:
                    print("-", end=" ")
            print()    
            
def turn_to_bins(data):
    X = data.copy()
    X[np.where(X_train < 128)] = 0
    X[np.where(X_train >= 128)] = 1
    return X

def match_label(y_train, W):
    mapping = np.zeros((10), dtype=np.uint32)
    counting = np.zeros((10, 10), dtype=np.uint32)
    for k in range(60000):
        counting[ y_train[k], np.argmax(W[k]) ] += 1
        
    for n in range(10):
        index = np.argmax(counting) # return a 0~99 value
        label = index // 10
        _class = index % 10
        mapping[label] = _class
        counting[:, _class] = 0
        counting[label, :] = 0
    return mapping

def update_posterior(X, P, Lambda):
    # i for numbers of image
    W = np.zeros(shape=(60000, 10))
    for i in trange(60000):
        # 出現此 image 且此 outcome 為label 0~9的機率 
        for j in range(10):
            W[i, j] = lambda_MLE[j]
            W[i, j] *= np.prod( P[j] ** X[i] )
            W[i, j] *= np.prod( (1- P[j])  ** (1 - X[i] ))
            #print(i, j, W[i, j], "前者:", np.prod( P[j] ** X[i] ), "後者:", np.prod( (1- P[j])  ** (1 - X[i] )))
    
    # marginalize
    marginal = np.sum(W, axis=1).reshape(60000, 1)
    W = W / marginal
    return W

def print_confusion(W, y_train, mapping):
    total_acc = 0
    for i in range(10):
        TP, FP, FN, TN = 0, 0, 0, 0
        
        predict = np.zeros(60000)
        
        for j in range(60000):
            fake_label = np.argmax(W[j])
            predict[j] =  np.where(mapping == fake_label)[0]
        
        for j in range(60000):
            # 0 for positive, 1 for negative
            if y_train[j] == i and predict[j] == i:
                TP += 1
            elif y_train[j] != i and predict[j] != i:
                TN += 1     
            elif y_train[j] == i and predict[j] != i:
                FN += 1
            elif y_train[j] != i and predict[j] == i:
                FP += 1
        total_acc += TP
        print("-" * 50)
        print("Confusion Matrix", i, ":")
        print(" " * 12,"Predict Cluster 1 / Predict Cluser 2")
        print("Is number {number} {value:^18}{value1:^18}".format(number = i, value=TP, value1=FN))
        print("Isn't number {number} {value:^18}{value1:^18}".format(number = i, value=FP, value1=TN))
        
        print("\nSensitivity (Successfully predict number", i," ): ", TP / (TP+FN) )
        print("Specificity (Successfully predict not number", i,"): ", TN / (TN+FP) )
        print()
        print("-" * 50)
        print()
    return total_acc / 60000
        

X_train = training_images()
y_train = training_label()

X = turn_to_bins(X_train)

print_num(X_train[0], False)

# initial value
# lambda -> probability of choosing the label 0..9
# P -> for label 0..9, 28 * 28 pixel probability of each pixel 出現1的機率
lambda_MLE = np.ones(shape=(10, 1)) / 10
P = np.random.rand(10, 28, 28) / 2 + 0.25

delta_lambda = 100
delta_Dist = 100

num_loop = 0
while delta_lambda > 0.1 or delta_Dist > 20:
    
    # E step
    W = update_posterior(X, P, lambda_MLE)

    # M step
    # update lambda
    new_P = np.zeros((10, 28, 28))
    new_lambda_MLE = np.sum(W, axis = 0) / 60000
    tmp = X.copy().reshape(60000, 784)
    
    # update P
    new_P = (np.transpose(W) @ tmp).reshape(10, 28, 28)
    
    # marginal
    for j in range(10):
        new_P[j , : , : ] /= np.sum(W, axis=0)[j]
    new_P[ new_P == 0 ] = 1e-5
    
    delta_lambda =  np.sum( np.abs(new_lambda_MLE - lambda_MLE) )
    delta_Dist = np.sum( np.abs(new_P - P ))
    print("\ndelta_lambda:" ,delta_lambda, "delta_Distr:", delta_Dist)
    
    lambda_MLE = new_lambda_MLE
    P = new_P
    num_loop +=1
    

# find the label 
mapping = match_label(y_train, W)

print_Imagination(P, mapping)
accuracy = print_confusion(W, y_train, mapping)  

print("Total iteration to converge:", num_loop)
print("Total error rate", 1 - accuracy )
    