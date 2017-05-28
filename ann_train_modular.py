# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:06:10 2017

@author: CStorm
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 21 10:43:01 2017

@author: CStorm
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data


def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def initialize(M):

    X, Y = get_data()
    X, Y = shuffle(X, Y)
    Y = Y.astype(np.int32)
    M = 5
    D = X.shape[1]
    K = len(set(Y))    
    # create train and test sets

    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Ytrain_ind = y2indicator(Ytrain, K)
    Xtest = X[-100:]
    Ytest = Y[-100:]
    Ytest_ind = y2indicator(Ytest, K)
    
    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)
    b2 = np.zeros(K)
    
    return(Xtrain, Ytrain, Ytrain_ind, Xtest, Ytest, Ytest_ind, W1, b1, W2, b2)

# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))
 
def train_loop(iterations, learning_rate, Xtrain, Ytrain, Ytrain_ind, Xtest, Ytest, Ytest_ind, W1, b1, W2, b2):
    # train loop
    train_costs = []
    test_costs = []
    #learning_rate = 0.001
    for i in range(iterations):
        pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
        pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)
    
        ctrain = cross_entropy(Ytrain_ind, pYtrain)
        ctest = cross_entropy(Ytest_ind, pYtest)
        train_costs.append(ctrain)
        test_costs.append(ctest)
    
        # gradient descent
        W2 -= learning_rate*Ztrain.T.dot(pYtrain - Ytrain_ind)
        b2 -= learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)
        dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain*Ztrain)
        W1 -= learning_rate*Xtrain.T.dot(dZ)
        b1 -= learning_rate*dZ.sum(axis=0)
        if i % 1000 == 0:
            print(i, ctrain, ctest)
    
    print("Final train classification_rate:", classification_rate(Ytrain, predict(pYtrain)))
    print("Final test classification_rate:", classification_rate(Ytest, predict(pYtest)))
    
    train_classification = classification_rate(Ytrain, predict(pYtrain))
    test_classification = classification_rate(Ytest, predict(pYtest))
    
    return train_costs, test_costs, train_classification, test_classification
    
    """
    legend1, = plt.plot(train_costs, label='train cost')
    legend2, = plt.plot(test_costs, label='test cost')
    plt.legend([legend1, legend2])
    plt.show()
    """
    
    

M_range = 9
Iterations = 10000

train_costs_total = np.zeros((M_range, Iterations))
test_costs_total = np.zeros((M_range, Iterations))

train_classification_total = np.zeros((M_range, 1))
test_classification_total = np.zeros((M_range, 1))

for M in range(1, M_range+1):
    Xtrain, Ytrain, Ytrain_ind, Xtest, Ytest, Ytest_ind, W1, b1, W2, b2 = initialize(M)
    train_costs, test_costs, train_classification, test_classification = train_loop(Iterations, 0.001, Xtrain, Ytrain, Ytrain_ind, Xtest, Ytest, Ytest_ind, W1, b1, W2, b2)
    
    train_costs_total[M-1] = train_costs
    test_costs_total[M-1] = test_costs
                    
    train_classification_total[M-1] = np.int(train_classification*100)
    test_classification_total[M-1] = np.int(test_classification*100)

plot_dims = np.int(np.sqrt(M_range))

fig, ax = plt.subplots(plot_dims, plot_dims, sharex = True, sharey = True)
for i in range(M_range):
    
    row = np.int(np.floor(i/plot_dims))
    col = np.int(i%plot_dims)
    legend1, = ax[row, col].plot(train_costs_total[i], label='train_cost')
    legend2, = ax[row, col].plot(test_costs_total[i], label='test_cost')
    ax[row, col].legend([legend1, legend2])
    #ax[row, col].title.set_text('M:')
    title = 'M:{},Test:{},Train:{}'.format(i+1,test_classification_total[i][0],train_classification_total[i][0])
    ax[row, col].set_title(title)

#plt.tight_layout()
plt.show()


