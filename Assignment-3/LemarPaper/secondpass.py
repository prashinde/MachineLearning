import numpy as np
import operator
np.set_printoptions(threshold=np.nan)
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from kmeans import kmeans
import json

with open("Dmatrixp2", "rb") as f:
    D = pickle.load(f)
f.close()

with open("Frequency", "rb") as f:
    TypeFreq = pickle.load(f)
f.close()

k = 48
nm = kmeans(k, 10)
centroids, clusters, objective = nm.cluster(D, TypeFreq)
plt.plot(objective, '--o')
plt.show()
