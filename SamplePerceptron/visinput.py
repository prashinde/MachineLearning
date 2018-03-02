import matplotlib
matplotlib.rcParams["backend"]="TkAgg"
import numpy as np
from pylab import rand
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def generateData(n):
    '''
    Generate random data
    '''
    xb = (rand(n)*2-1)
    yb = (rand(n)*2-1)
    xr = (rand(n)*2-1)+2
    yr = (rand(n)*2-1)+2

    inputs1 = []
    inputs2 = []
    for i in range(len(xb)):
        inputs1.append([1, xb[i], yb[i], 1])
        inputs2.append([1, xr[i], yr[i], -1])
    return inputs1, inputs2

def visualizeData(data1, data2, line, line_h):
    plt.scatter([row[1] for row in data1], [row[2] for row in data1], color='red')
    plt.scatter([row[1] for row in data2], [row[2] for row in data2], color='blue')

    #for row in line_h:
    x = np.array(range(-3,3))
    if(line[1] != 0):
        y = eval('(-1*line[1]*x/line[2])-(line[0]/line[2])')
        plt.plot(x,y);

    plt.show()

def dot_product(V1, V2):
    return sum(i[0]*i[1] for i in zip(V1, V2))

def Perceptron(data):
    colsize=len(data[0])
    weights = [0]*(colsize-1)
    numIter = 50
    weight_hist = []
    for i in range(numIter):
        nr_mistakes = 0
        weight_hist.append(list(weights))
        for row in data:
            pred = 0;
            dt_p = dot_product(weights, [row[0], row[1], row[2]])
            if(dt_p > 0):
                pred = 1
            else:
                pred = -1

            if(row[3] != pred):
                nr_mistakes += 1
                weights[0] = weights[0]+row[3]*row[0]
                weights[1] = weights[1]+row[3]*row[1]
                weights[2] = weights[2]+row[3]*row[2]
        if(nr_mistakes == 0):
            break
    return weights,weight_hist

[data1,data2]=generateData(500)
[wts, wts_h] = Perceptron(data1+data2)
visualizeData(data1, data2, wts, wts_h)
