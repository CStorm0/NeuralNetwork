# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:13:20 2017

@author: CStorm
"""
from process import get_data
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

Nclass = 500

X, Y = get_data()

X, Y = shuffle(X, Y)
Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

model = MLPClassifier(hidden_layer_sizes = (20,20), max_iter=2000)

model.fit(Xtrain, Ytrain)

train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)
print('Train:{}, Test:{}'.format(train_accuracy, test_accuracy))

#model.predict(Xtest)

