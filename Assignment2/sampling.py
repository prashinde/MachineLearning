import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

prcent_split = 0.75
training_as_pickle="/home/pratik/MachineLearning/TrainingAss2/mnist_rowmajor.pkl"

def pick_first_M(TD, TL):
    TDSize = len(TD)
    
    split = int(0.75*TDSize)

    in_train_d = TD[0:split]
    in_train_l = TL[0:split]

    in_dev_d = TD[split:TDSize]
    in_dev_l = TL[split:TDSize]

    return in_train_d, in_train_l, in_dev_d, in_dev_l

with open(training_as_pickle, "rb") as f:
    data=pk.load(f)

training_data = data['images_train']
training_label = data['labels_train']

[in_train_data, in_train_label, in_dev_data, in_dev_label] = pick_first_M(training_data, training_label)

lables=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
freqs=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in in_train_label:
    freqs[i[0]] += 1

'''
Plot a historgram of numbers
'''
x = np.arange(len(freqs))
plt.bar(x, freqs)
plt.xticks(x, lables)
plt.show()

'''
Create pickle file for splitted data
'''
to_pickle = {'in_train_data':in_train_data,
        'in_train_label':in_train_label,
        'in_dev_data':in_dev_data,
        'in_dev_label':in_dev_label}

output = open('processed_data.pkl', 'wb')
pk.dump(to_pickle, output)
output.close()
