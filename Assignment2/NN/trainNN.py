import random
import numpy as np
import pickle as pk
from NN import NeuralNet

training_d = "../Perceptron/processed_data.pkl"
with open(training_d, "rb") as f:
    data=pk.load(f)
td = data['in_train_data']
tl = data['in_train_label']

'''
dummy test data
nrex = 30
td=np.random.uniform(0,1,nrex*784)
tl=np.random.uniform(-1,1,nrex*1)

tl = np.where(tl > 0, 1, 0)
td = td.reshape(nrex, 784)
tl = tl.reshape(nrex, 1)
'''

NN = NeuralNet(nrclasses=10, nrfeatures=784, nrhunits=30, epoch=900, ll=0.001, nrbatches=100, bias=3)
NN.TrainNet(td, tl, 422229944)
NN.pickleClass('30hunits_randombias.pkl')
