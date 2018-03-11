import numpy as np
import pickle as pk

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


def Accuracy_th(data, label, wts, tao):
    nrm = 0
    for ex in range(len(data)):
        row = data[ex]
        row = np.where(row > tao, 1, 0)
        pred = -1
        dt_p = np.matmul(np.append(1, row), np.transpose(wts))
        pred = np.argmax(dt_p)
        if(pred != label[ex][0]):
            nrm += 1
    acc = float((len(data)-nrm)*100)/len(data)
    print "Tao is", tao, " Accuracy of a Perceptron is", acc
    return acc


training_d = "../Data/processed_data.pkl"

training_as_pickle="../Data/mnist_rowmajor.pkl"
with open(training_as_pickle, "rb") as f:
    data=pk.load(f)

test_data = data['images_test']
test_label = data['labels_test']

with open("60kexamples.pkl", "rb") as f:
    wts=pk.load(f)


Accuracy(test_data, test_label, wts)
'''
tao = 0
maxacc = 0
btao = 0
while tao < 1:
    acc = Accuracy_th(devel_d, devel_l, wts, tao)
    if(acc > maxacc):
        maxacc = acc
        btao = tao
    tao += 0.01

print "Best accuracy:", maxacc, " Corresponding Tao:", btao
'''
