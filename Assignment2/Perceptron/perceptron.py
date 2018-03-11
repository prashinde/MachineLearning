import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

def train(data, label, tao):
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
            if(tao != 0):
                row = np.where(row > tao, 1, 0)

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

        acc = float((len(data)-nrm)*100)/len(data)
        print "Iteration:", i, " Accuracy:", acc

    return weights

training_as_pickle="../Data/mnist_rowmajor.pkl"
with open(training_as_pickle, "rb") as f:
    data=pk.load(f)

training_data = data['images_train']
training_label = data['labels_train']


'''
training_d = "../Data/processed_data.pkl"

with open(training_d, "rb") as f:
    data=pk.load(f)

training_data = data['in_train_data']
training_label = data['in_train_label']
'''

tao = 0
wts = train(training_data, training_label, tao)

output = open('60kexamples.pkl', 'wb')
wts.dump(output)
output.close()

print "Printing weights:"
for i in range(len(wts)):
    print wts[i][0]
