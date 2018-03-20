import random
import numpy as np
import pickle as pk
from NN import NeuralNet 

pNet = "../bestmodels/testNN.pkl"

with open(pNet, "rb") as f:
    NN=pk.load(f)

training_as_pickle="../Data/mnist_rowmajor.pkl"
with open(training_as_pickle, "rb") as f:
    data=pk.load(f)

test_data = data['images_test']
test_label = data['labels_test']

noise = 0

if(noise == 1):
    test_data = test_data + np.random.normal(0, 0.05, test_data.shape)

'''
training_d = "../Perceptron/processed_data.pkl"

with open(training_d, "rb") as f:
    data=pk.load(f)

devel_d = data['in_dev_data']
devel_l = data['in_dev_label']
'''
nrm = 0
for ex in range(len(test_data)):
    row = test_data[ex].reshape(1, 784)
    pred = -1
    binput, hout, hact, fout, fact = NN.forward(row)
    pred = np.argmax(fact)
    
    if(pred != test_label[ex][0]):
        nrm += 1
acc = float((len(test_data)-nrm)*100)/len(test_data)
print " Accuracy of a NN:", acc
