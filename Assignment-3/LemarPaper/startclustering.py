import numpy as np
import operator
np.set_printoptions(threshold=np.nan)
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from kmeans import kmeans

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    if a == 0:
        return True
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

with open("Dmatrix", "rb") as f:
    D = pickle.load(f)
f.close()

'''
idx = 0
for row in D:
    nrmed = np.linalg.norm(row[0:1000], ord=2, axis=0)
    nrmed1 = np.linalg.norm(row[1000:2000], ord=2, axis=0)
    assert isclose(nrmed, 1.0), "Row is"
    assert isclose(nrmed1, 1.0), "Row2 is"
    idx = idx+1
'''
with open("Frequency", "rb") as f:
    TypeFreq = pickle.load(f)
f.close()

k = 500
nm = kmeans(k)
centroids, clusters, objective = nm.cluster(D, TypeFreq)
plt.plot(objective, '--o')
plt.show()
