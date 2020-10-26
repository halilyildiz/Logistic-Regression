# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:09:54 2020

@author: halil
"""

#%% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read_csv

data = pd.read_csv("data.csv")

data.drop(["Unnamed: 32","id"], axis = 1, inplace = True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis = 1)

#%% normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
# (x - min(x))/(max(x)-min(x))

#%% split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

#%% create logistic regression model

 #intialize
def initialize_weights_and_bias(x_train): 
    dimension = x_train.shape[0] # that is 30
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

#sigmoid function
def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))
    return y_head

#forward propagation
def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    return y_head, cost

#backward propagation
def backward_propagation(y_head):
    #derivative_weight
    d_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    #derivative_bias
    d_bias = np.sum(y_head - y_train)/x_train.shape[1]

    return d_weight, d_bias

#update
def update(w,b,d_weight,d_bias,learning_rate):   
    w = w - learning_rate * d_weight
    b = b - learning_rate * d_bias
    return w,b

#train
def train(x_train,y_train,x_test,y_test,learning_rate, number_of_iterations):
   
    cost_list = []
    
    w,b = initialize_weights_and_bias(x_train)
    
    #update paremeters is number_of_iterarion times
    for i in range(number_of_iterations):
        
        y_head, cost = forward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        d_weight, d_bias = backward_propagation(y_head)
        w,b = update(w,b,d_weight, d_bias,learning_rate)
        
        if i % 10 == 0:            
            print("Cost after iteration %i: %f"%(i,cost))
            
    #update parameters weigt and bias
    parameters = {"weight": w,"bias":b}
  
    return parameters, cost_list, number_of_iterations
 
#predict    
def predict(parameters,x_test,y_test):

    z = sigmoid(np.dot(parameters["weight"].T,x_test)+parameters["bias"])
    y_pred = np.zeros((1,x_test.shape[1]))
    #if z is bigger than 0.5 our prediction is sign one,
    #if z is smaller than 0.5 our prediction is sign zero,
    for i in range(z.shape[1]):
        if z[0,i]<=0.5:
            y_pred[0,i] = 0
        else:
            y_pred[0,i] = 1

    #print test errors
    print("test_accuracy: {} %".format(100-np.mean(np.abs(y_pred-y_test))*100))
    
    
#%%

parameters, cost_list, number_of_iterations = train(x_train,y_train,x_test,y_test,
                                                         learning_rate = 0.2,number_of_iterations = 10000)

predict(parameters,x_test,y_test)

#visualization
plt.plot(range(number_of_iterations),cost_list)
plt.xlabel("Number of iteration")
plt.ylabel("Cost")
plt.show()
       





    
    

