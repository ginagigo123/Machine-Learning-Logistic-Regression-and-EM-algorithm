# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:56:31 2021

@author: ginag
"""
import gzip
import numpy as np

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

def print_num(x):
    for i in range(28):
        for j in range(28):
            if x[i][j] < 128:
                print("-", end=" ")
            else:
                print("*", end=" ")
        print()

def print_num_DIS(x):
    print("Imagination of numbers in Bayesian classifier:")
    for label in range(10):
        print(label, ":")
        for i in range(28):
            for j in range(28):
                value = np.argmax(x[label][i][j])
                if value < 15:
                    print("-", end=" ")
                else:
                    print("*", end=" ")
            print()    
            
def turn_to_bins(data):
    X = data.copy()
    X[np.where(X_train < 128)] = 0
    X[np.where(X_train > 128)] = 1
    return X
    
X_train = training_images()
y_train = training_label()

X = turn_to_bins(X_train)

print_num(X_train[0])
print(X[0])