import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

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

# print(X_val.shape)      # (n x m)
# print(Y_val.shape)      #Labels in validation

# print(X_train.shape)     #Training set has 33600 examples
# print(Y_train.shape)

def initialize_parameters():
    W1=np.random.rand(10,784)-0.5
    B1=np.random.rand(10,1)-0.5
    W2=np.random.rand(10,10)-0.5
    B2=np.random.rand(10,1)-0.5

    return W1, B1, W2, B2 

def ReLU(X):
    return np.maximum(X,0)

def softmax_calculator(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(W1, B1, W2, B2, X):
    Z1= W1.dot(X)+B1        # y=mx+c
    A1=ReLU(Z1)

    Z2=W2.dot(A1)+B2
    A2=softmax_calculator(Z2)

    return Z1, A1, Z2, A2

def one_hot_converter(Y):
    one_hot_Y=np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y]=1
    return one_hot_Y.T

def backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y):
    one_hot_Y= one_hot_converter(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    dB2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1/m * dZ1.dot(X.T)
    dB1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, dB1, dW2, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1

    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    
    return W1, B1, W2, B2

def get_prediction(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions==Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):      #alpha=learning_rate, iterations=Epochs
    W1, B1, W2, B2 = initialize_parameters()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)      #Forward propagation
        dW1, dB1, dW2, dB2 = backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y)
        W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

        if (i%20) == 0:
            print("Iteration number: ", i)
            print("Accuracy = ", get_accuracy(get_prediction(A2), Y))
    return W1, B1, W2, B2

W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 0.1, 1000)

#Validation
val_index=1000
Z1val, A1val, Z2val, A2val = forward_propagation(W1, B1, W2, B2, X_val[:,val_index].reshape(-1,1))
print("Predicted Label: ", get_prediction(A2val))
print("Actual Label: ", Y_val[val_index])

#Displaying Image
image_array = X_val[:, val_index].reshape(28,28)
plt.imshow(image_array, cmap='gray')
plt.show()

#Validation Accuracy 
Z1val, A1val, Z2val, A2val = forward_propagation(W1, B1, W2, B2, X_val)
val_acc = get_accuracy(get_prediction(A2val), Y_val)

print("Validation Accuracy: ", val_acc)

#Saving the model
with open("model/model.pkl", "wb") as f:
    pickle.dump((W1, B1, W2, B2), f)