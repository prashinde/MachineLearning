import numpy as np
import operator
np.set_printoptions(threshold=np.nan)
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from kmeans import kmeans
from manytoone import manytoone
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
    Uprime = U.flatten()
    Uprime.sort()
    return np.where(U <= Uprime[-N], 0, U)

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

with open("Dmatrixp2", "rb") as f:
    D = pickle.load(f)
f.close()

with open("Frequency", "rb") as f:
    TypeFreq = pickle.load(f)
f.close()

k = 48
nm = kmeans(k, 80)
centroids, clusters, objective = nm.cluster(D, TypeFreq)
#plt.plot(objective, '--o')
#plt.show()

del D

GoldenTags = {}
fname = "../Data/a3-data/train.json"
data = json.load(open(fname))

print 'Started computing Golden Tags...'
for section in data:
    for sentence in data[section]:
        for word in sentence:
            stype = word['text'].lower()
            if stype not in GoldenTags:
                GoldenTags[stype] = word['tag']
print 'Golden Tags computation done'

wordtocindex={}

print 'Computing cluster indices'
for i in range(len(TypeFreq)):
    wordtocindex[i] = cluster_index(clusters, i)
print 'Done Computing cluster indices'

mtoone = manytoone(clusters, GoldenTags, TypeFreq)

clustertags = mtoone.assign()
accuracy = mtoone.evaluate(clusters, clustertags)
print "Accuracy of many to one", accuracy

del GoldenTags

wordtocluster = {}
#TODO: Build word to cluster mapping.

fname = "../Data/a3-data/dev.json"
data = json.load(open(fname))

TypetoIndex = {}
TypeFreq = Counter()

sentences = all_sentences(data, 0)

for sentence in sentences:
    for word in sentence:
        stype = word['text'].lower()
        TypeFreq[stype] += 1

index = 0
for wtype in TypeFreq.most_common():
    stype = wtype[0].lower()
    if stype not in TypetoIndex:
        TypetoIndex[stype] = index
        index = index+1

Lcontext =  np.zeros((len(TypetoIndex), 500))
Rcontext =  np.zeros((len(TypetoIndex), 500))


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

'''
Dictionary of word to cluster
'''
assignment = nm.assignCluster(centroids, D, TypeFreq)

GoldenTags = {}
print 'Started computing Golden Tags...'
for section in data:
    for sentence in data[section]:
        for word in sentence:
            stype = word['text'].lower()
            if stype not in GoldenTags:
                GoldenTags[stype] = word['tag']
print 'Golden Tags computation done'

correct = 0
total = 0
for key,value in assignment.items():
    if GoldenTags[key] == clustertags[value]:
        correct = correct+1
    print "predicted:", clustertags[value], ' correct:', GoldenTags[key]
    total = total+1

print "Accuracy on Dev data is", float(correct)/float(total)
