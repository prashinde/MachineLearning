import numpy as np
import random
import csv
from numpy.random import choice

def generate_weighted_list(weights, nsamples):
    return choice(len(weights), nsamples, list(weights))


'''
We will generate K gaussians
'''
k=3

'''
and nr points
'''
nr=1000

'''
Each point is D dimentional
'''
D=10

'''
We set variance to 0.8
'''
sigma = 1

'''
Generate "k" pik's. All of them should sum to one
'''
pik = np.random.random(k)
pik = pik / pik.sum()

'''
We have to generate k means, each is D dimensional
'''

muks = np.random.random((k, D))
sigmak = np.identity(D)*sigma

'''
Generate index Zi based on pik weights
'''
indices = generate_weighted_list(pik, nr)

'''
Just pick up ith element from indeices[i]th gaussian
'''
points=[]
for i in range(nr):
    ix = indices[i]
    points.append(list(np.random.multivariate_normal(muks[ix], sigmak)))

fname='data_'+str(D)+'_'+str(k)+'_'+str(nr)

with open('../Data/'+fname, 'wb') as mf:
    wr = csv.writer(mf, delimiter=' ')
    wr.writerows(points)
