import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from liblinearutil import *

training_as_pickle="../Data/mnist_rowmajor.pkl"
with open(training_as_pickle, "rb") as f:
    data=pk.load(f)

training_data = data['images_train']
training_label = data['labels_train']

tao = 0

noise_train = 0
sort_in = 0

if(noise_train == 1):
    training_data = training_data + np.random.normal(0, 0.05, training_data.shape)
elif(sort_in == 1):
    idx = np.argsort(training_label.flatten())
    training_data=np.array(training_data)[idx]
    training_label=np.array(training_label)[idx]

prob = problem(training_label.flatten(), training_data)
param = parameter('-s 6 -B 0')

model = train(prob, param)
save_model('bestmaxent', model)

noise_test = 0
test_d = data['images_test']
test_l = data['labels_test']

if(noise_test == 1):
    test_d = test_d + np.random.normal(0, 0.05, test_d.shape)
p_labs, p_acc, p_vals = predict(test_l.flatten(), test_d, model)

print "Accuracy:",
print p_acc
