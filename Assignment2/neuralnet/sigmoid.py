import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

from preprocessing import *

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_prime(xlist):
    sgscore = sigmoid(xlist)
    return sgscore*(1-sgscore)
'''
x = np.linspace(-10., 10., num=100)
sig = sigmoid(x)
sig_prime = sigmoid_prime(x)

plt.plot(x, sig, label="sigmoid")
plt.plot(x, sig_prime, label="sigmoid prime")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(prop={'size' : 16})
plt.show()
'''
