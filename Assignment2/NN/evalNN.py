import random
import numpy as np
import pickle as pk
from NN import NeuralNet 

pNet = "30hunits_randombias.pkl"

with open(pNet, "rb") as f:
    NN=pk.load(f)

training_d = "../Perceptron/processed_data.pkl"

with open(training_d, "rb") as f:
    data=pk.load(f)

devel_d = data['in_dev_data']
devel_l = data['in_dev_label']

print "Model magic is:",
print NN.magic
nrm = 0
for ex in range(len(devel_d)):
        row = devel_d[ex].reshape(1, 784)
        pred = -1
        binput, hout, hact, fout, fact = NN.forward(row)
        pred = np.argmax(fact)
        if(pred != devel_l[ex][0]):
            nrm += 1

print "Accuracy of a NN is",
print float((len(devel_d)-nrm)*100)/len(devel_d)

