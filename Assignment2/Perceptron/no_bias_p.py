import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

def dot_product(V1, V2):
    return sum(i[0]*i[1] for i in zip(V1, V2))

def Accuracy(devel_d, devel_l, wts):
    nrm = 0
    for ex in range(len(devel_d)):
        row = devel_d[ex]
        pred = -1
        dt_p = np.matmul(np.append(1, row), np.transpose(wts))
        pred = np.argmax(dt_p)
        if(pred != devel_l[ex][0]):
            nrm += 1
    print "Accuracy of a Perceptron is",
    print float((len(devel_d)-nrm)*100)/len(devel_d)

def Perceptron(data, label):
    weights = np.zeros((10,785))

    nrm = 0
    numIter = 200
    weight_hist = []
    for i in range(numIter):
        nrm = 0
        #weight_hist.append(list(weights))
        for ex in range(len(data)):
            row = data[ex]
            pred = -1;
            row = np.append(np.ones((1,1)), row)
            '''
            row: A single training exmaple: 1x784 dimension.
            weights: 9x784 matrix.
            Multiply these two together to generate 1x9 matrix

            Result = row*weights
            '''
            dt_p = np.matmul(row, np.transpose(weights))
            pred = np.argmax(dt_p)
            '''
            Update weights
            '''
            if(pred != label[ex]):
                cc = label[ex]
                nrm += 1
                weights[cc[0]] = weights[cc[0]]+row
                weights[pred] = weights[pred]-row
        print i
        if(nrm == 0):
            break

    print "Accuracy of a Perceptron is:",
    print (float((len(data)-nrm)*100))/len(data)
    print weights[0][0]
    return weights

training_d = "processed_data.pkl"

with open(training_d, "rb") as f:
    data=pk.load(f)

training_data = data['in_train_data']
training_label = data['in_train_label']

wts = Perceptron(training_data, training_label)

print "Printing weights:"
for i in range(len(wts)):
    print wts[i][0]

devel_d = data['in_dev_data']
devel_l = data['in_dev_label']

Accuracy(devel_d, devel_l, wts)
