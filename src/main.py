import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('dataset/train.csv')     #Reading the dataset

data=np.array(data)     #converting the datafarme to numpy array format
m,n=data.shape

np.random.shuffle(data) #Shuffling the dataset as I do not want the data to be in specific order

#print(m, n)

train_data=data[0:int(0.8*m), :]  #dividing 80% of data to train 
Val_data=data[int(0.8*m):m, :]      # Remaining 20% data to validate

X_train=train_data[:, 1:].T
X_train=X_train/255.0       #Scalling
Y_train=train_data[:, 0]

X_val=Val_data[:, 1:].T
X_val=X_val/255.0       #Scalling (Normalizing the image pixel values, as we want the max pixel value to be 1 (instead of 0-255))
Y_val=Val_data[:, 0]

print(X_val.shape)      # (n x m)
print(Y_val.shape)      #Labels in validation

print(X_train.shape)     #Training set has 33600 examples
print(Y_train.shape)

def initialize_parameters():

    return

def forward_propagation():

    return

def backward_propagation():

    return

def update_parameters():

    return

