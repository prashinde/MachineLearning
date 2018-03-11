import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from liblinearutil import *

training_as_pickle="../../Data/mnist_rowmajor.pkl"
with open(training_as_pickle, "rb") as f:
    data=pk.load(f)

training_data = data['images_train']
training_label = data['labels_train']

'''
training_d = "../../processed_data.pkl"

with open(training_d, "rb") as f:
    data=pk.load(f)
training_data = data['in_train_data']
training_label = data['in_train_label']
'''
tao = 0
#prob = problem(training_label.flatten(), np.where(training_data > tao, 1, 0))
prob = problem(training_label.flatten(), training_data)
param = parameter('-s 6 -B 0')

model = train(prob, param)
#save_model('train_model', model)

test_d = data['images_test']
test_l = data['labels_test']
p_labs, p_acc, p_vals = predict(test_l.flatten(), test_d, model)
#p_labs, p_acc, p_vals = predict(test_d.flatten(), np.where(devel_d > tao, 1, 0), model)

print "Accuracy:",
print p_acc
