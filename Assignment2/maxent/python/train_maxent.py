import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from liblinearutil import *

training_d = "../../processed_data.pkl"

with open(training_d, "rb") as f:
    data=pk.load(f)
training_data = data['in_train_data']
training_label = data['in_train_label']

tao = 0
prob = problem(training_label.flatten(), np.where(training_data > tao, 1, 0))
#prob = problem(training_label.flatten(), training_data)
param = parameter('-s 6 -B 1')

model = train(prob, param)
#save_model('train_model', model)

devel_d = data['in_dev_data']
devel_l = data['in_dev_label']
#p_labs, p_acc, p_vals = predict(devel_l.flatten(), devel_d, model)
p_labs, p_acc, p_vals = predict(devel_l.flatten(), np.where(devel_d > tao, 1, 0), model)

print "Accuracy:",
print p_acc
