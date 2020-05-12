import numpy as np
import pandas as pd
import zipfile

#read data
zf = zipfile.ZipFile('mnist_train.zip') 
train_all = pd.read_csv(zf.open(r'mnist_train.csv'))
test = pd.read_csv('mnist_test.csv')

#regularization
train_all[train_all.columns[1:]] = train_all[train_all.columns[1:]].div(255)
test[test.columns[1:]] = test[test.columns[1:]].div(255)

#split training data into training data and validation data
proportion = 0.8
train_all = (train_all.sample(frac=1)).reset_index(drop=True)
train = train_all[:int(len(train_all)*proportion)]
validation = train_all[int(len(train_all)*proportion):]

#split test label and data
test_y = np.array(test.label)
test_x = np.array(test[test.columns[1:]])

#define sigmoid and its derivative
def sigmoid(z):
    sigz = np.zeros(z.shape)
    for (x, y), val in np.ndenumerate(z):
        if val >= 100:
            sigz[x][y] = 1.
        elif val <= -100:
            sigz[x][y] = 0.
        else:
            sigz[x][y] = 1.0 / (1.0 + np.exp(-val))
    return sigz

def sigmoid_derivative(z):
    return sigmoid(z) * (1. - sigmoid(z))
    
def NeuralwithSGD(epochs,W0,W1,b0,b1,train,batch_size,eta,validation,best_W0, best_W1, best_b0, best_b1, best_acc,best_epoch,best_batch_size,best_eta):

    #start training
    print('eta = %s batch_size = %s' % (eta, batch_size))
    for epoch in range(epochs):
        #shuffle
        train = (train.sample(frac=1)).reset_index(drop=True)

        #divide into mini batches
        mini_batches = [train[k:k+batch_size] for k in range(0,len(train),batch_size)] #list of dataframe
        
        for iteration in range(len(mini_batches)):
            #split data and label (take the first batch to train)
            y = np.array(mini_batches[iteration].label) 
            y_vector = np.zeros((y.size, y.max()+1))
            y_vector[np.arange(y.size),y] = 1
            X0 = np.array(mini_batches[iteration][train.columns[1:]]) 
    
            #feedforward
            Z0 = np.dot(X0, W0) + b0
            X1 = sigmoid(Z0)
            Z1 = np.dot(X1, W1) + b1
            X2 = sigmoid(Z1)
    
            #backpropogation
            dZ1 = (X2 - y_vector)*sigmoid_derivative(Z1)
            db1 = np.mean(dZ1, axis=0)
            dW1 = np.dot(np.transpose(X1), dZ1)
            dZ0 = np.dot(dZ1, np.transpose(W1)) * sigmoid_derivative(Z0)
            db0 = np.mean(dZ0, axis=0)
            dW0 = np.dot(np.transpose(X0), dZ0)
    
            #update weights and biases
            W0 = W0 - eta*dW0
            W1 = W1 - eta*dW1
            b0 = b0 - eta*db0
            b1 = b1 - eta*db1
            
            
        #split validation label and data
        validation_y = np.array(validation.label)
        validation_x = np.array(validation[validation.columns[1:]])
        predict = np.dot(validation_x, W0) + b0
        predict = sigmoid(predict)
        predict = np.dot(predict, W1) + b1
        predict_label = sigmoid(predict)

        accuracy = (np.argmax(predict_label,axis=1) == validation_y).sum()
        print('epoch %s: accuracy %0.2f ' %(epoch, 100*accuracy/len(validation)))
              
        if(accuracy > best_acc):
            best_W0 = W0
            best_W1 = W1
            best_b0 = b0
            best_b1 = b1
            best_epoch = epoch
            best_acc = accuracy
            best_batch_size = batch_size
            best_eta = eta
            print("Best model saved")  
    return best_W0, best_W1, best_b0, best_b1, best_epoch, best_acc, best_batch_size, best_eta
    
#choose parameters
num_neurons = [784, 100, 10]
best_acc = 0
best_epoch = 0
best_batch_size = 0
best_eta = 0
best_W0 = 0
best_W1 = 0
best_b0 = 0
best_b1 = 0

#initialize weights and biases
W0 = 2*np.random.random((num_neurons[0],num_neurons[1]))-1
W1 = 2*np.random.random((num_neurons[1],num_neurons[2]))-1
b0 = 2*np.random.random(num_neurons[1])-1
b1 = 2*np.random.random(num_neurons[2])-1
best_W0, best_W1, best_b0, best_b1, best_epoch, best_acc, best_batch_size, best_eta = NeuralwithSGD(1000,W0,W1,b0,b1,train,250,0.005,validation,best_W0, best_W1, best_b0, best_b1, best_acc,best_epoch,best_batch_size,best_eta)

predict = np.dot(test_x, best_W0) + best_b0
predict = sigmoid(predict)
predict = np.dot(predict, best_W1) + best_b1
predict_label = sigmoid(predict)

accuracy = (np.argmax(predict_label,axis=1) == test_y).sum()
print('The test data with best model has accuracy %0.2f ' %(epoch, 100*accuracy/len(test)))
