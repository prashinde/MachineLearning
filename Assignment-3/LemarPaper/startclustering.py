import numpy as np
import operator
np.set_printoptions(threshold=np.nan)
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from kmeans import kmeans
import json

'''
Normalizes each row of a given matrix to unit length
'''
def normalize_row(C):
    l2 = np.atleast_1d(np.linalg.norm(C, ord=2, axis=1))
    l2[l2==0] = 1
    return C / np.expand_dims(l2, 1)

'''
Reduce rank of a vector U, by replacing N
elements which are close to zero with zeros
'''
def reduce_rank(U, N):
    U = U.flatten()
    U.sort()
    U[-N:len(U)] = 0
    return U

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    if a == 0:
        return True
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def all_sentences(data, N):
    rsentences=[]
    r = 0
    for section in data:
        for sentence in data[section]:
            rsentences.append(sentence)
            r = r+1
            if (N != 0) and (r > N):
                return rsentences
    return rsentences

def cluster_index(clusters, N):
    for cluster in clusters:
        if N in clusters[cluster]:
            return cluster
    return -1

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
nm = kmeans(k, 36)
centroids, clusters, objective = nm.cluster(D, TypeFreq)
#plt.plot(objective, '--o')
#plt.show()

TypetoIndex = {}

index = 0
TFMC = TypeFreq.most_common()
for wtype in TFMC:
    stype = wtype[0].lower()
    if stype not in TypetoIndex:
        TypetoIndex[stype] = index
        index = index+1

del TFMC
'''
Start second pass
'''
fname="../Data/a3-data/train.json"
data = json.load(open(fname))

sentences = all_sentences(data, 0)

'''
We are interested in only top 10 most frequent words
'''
w1 = 1000

'''
print most common w1 words
'''
Lcontext =  np.zeros((len(TypetoIndex), k))
Rcontext =  np.zeros((len(TypetoIndex), k))

wordtocindex={}

print 'Computing cluster indices'
for i in range(len(TypetoIndex)):
    wordtocindex[i] = cluster_index(clusters, i)
print 'Done Computing cluster indices'

print "Started computing Lcontext and Rcontext"
PrevToken = '__seq__' 
for sentence in sentences:
    for word in sentence:
        stype = word['text'].lower()
        pretindex = TypetoIndex[PrevToken]
        curindex = TypetoIndex[stype]

        pclusterindex = wordtocindex[pretindex]
        cclusterindex = wordtocindex[curindex]
        Rcontext[pretindex][cclusterindex] += 1
        Lcontext[curindex][pclusterindex] += 1
        PrevToken = stype
print "Done computing Lcontext and Rcontext"

del sentences
#TypetoIndex
#TypeFreq

print "Started computing SVD"
UL, SL, VL = np.linalg.svd(Lcontext, full_matrices=False)
UR, SR, VR = np.linalg.svd(Rcontext, full_matrices=False)
print "Done Computing SVD.."
del Lcontext
del Rcontext

SL = reduce_rank(SL, k-200)
SR = reduce_rank(SR, k-200)

SLstar = SL*np.identity(len(SL))
SRstar = SR*np.identity(len(SR))

Lstar = UL.dot(SLstar)
Rstar = UR.dot(SRstar)

Ldstar = normalize_row(Lstar)
del Lstar
Rdstar = normalize_row(Rstar)
del Rstar

D = np.concatenate((Ldstar, Rdstar), axis=1)
del Ldstar
del Rdstar

with open("Dmatrixp2", 'wb') as f:
    pickle.dump(D, f, -1)
f.close()

#print D.shape
#k = 500
#nm = kmeans(k)
#centroids = nm.cluster(D, TypeFreq)


#with open("Dmatrix", 'wb') as f:
#    pickle.dump(D, f, -1)
#f.close()
#plt.imshow(D, cmap='hot', interpolation='nearest')
#plt.show()
