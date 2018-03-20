import random
import numpy as np
import pickle as pk
from NN import NeuralNet

training_as_pickle="../Data/mnist_rowmajor.pkl"
with open(training_as_pickle, "rb") as f:
    data=pk.load(f)

training_data = data['images_train']
training_label = data['labels_train']


'''
dummy test data
nrex = 30
td=np.random.uniform(0,1,nrex*784)
tl=np.random.uniform(-1,1,nrex*1)

tl = np.where(tl > 0, 1, 0)
td = td.reshape(nrex, 784)
tl = tl.reshape(nrex, 1)
'''

print "Training data: ", training_data.shape
print "Training label: ", training_label.shape

noise = 0
sort_in = 1

if(noise == 1):
    training_data = training_data + np.random.normal(0, 0.05, training_data.shape)
elif(sort_in == 1):
    idx = np.argsort(training_label.flatten())
    training_data=np.array(training_data)[idx]
    training_label=np.array(training_label)[idx]


NN = NeuralNet(nrclasses=10, nrfeatures=784, nrhunits=60, epoch=900, ll=0.0001, nrbatches=100, bias=3, ifeature=1, tao=0.04)
NN.TrainNet(training_data, training_label, 422229944)
NN.pickleClass('noise/sort.pkl')
