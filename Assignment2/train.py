import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

training_d = "processed_data.pkl"

with open(training_d, "rb") as f:
    data=pk.load(f)

training_data = data['in_train_data']
training_label = data['in_train_label']
SD=training_data[0].reshape((28,28))

for row in SD:
    for i in row:
        if i > 0:
            print "1 ",
        else:
            print "0 ",
    print " "
