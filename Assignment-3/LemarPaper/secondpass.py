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

#TypeFreq['ovv'] = 0
k = 45
nm = kmeans(k, 30)
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
            GoldenTags[stype] = word['tag']

GoldenTags['ovv'] = 'ovv'
print 'Golden Tags computation done'

trwordtocindex = {}
trwordtocluster = {}

trclustertoword = {}

for i in range(k):
    trclustertoword[i] = []

TFMC = TypeFreq.most_common()

print 'Computing cluster indices'
for i in range(len(TypeFreq)):
    trwordtocindex[i] = cluster_index(clusters, i)
    trwordtocluster[TFMC[i][0]] = trwordtocindex[i]
    trclustertoword[trwordtocindex[i]].append(TFMC[i][0])
print 'Done Computing cluster indices'

mtoone = manytoone(clusters, GoldenTags, TypeFreq)

clustertags = mtoone.assign()

print "***********************************************************"
print "training Tags"
print clustertags
print "***********************************************************"
accuracy = mtoone.evaluate(clusters, clustertags)
print "Accuracy of many to one", accuracy

del GoldenTags

fname = "../Data/a3-data/dev.json"
data = json.load(open(fname))

TypetoIndex = {}
TypeFreq = Counter()

sentences = all_sentences(data, 0)

devclusters = {}

for i in range(k):
    devclusters[i] = []

correct = 0
incorrect = 0
total = 0
for sentence in sentences:
    for word in sentence:
        stype = word['text'].lower()
        if stype in trwordtocluster:
            cindex = trwordtocluster[stype]
            devclusters[cindex].append(word)

clustertotag = {}
debug_acc = 0
twords = 0
for cluster in devclusters:
    Tags=[]
    for word in devclusters[cluster]:
        Tags.append(word['tag'])
        twords += 1

    counter = Counter(Tags)
    clustertotag[cluster] = counter.most_common()[0][0]
    debug_acc = debug_acc + counter.most_common()[0][1]

print "***********************************************************"
print "development tags"
print clustertotag
print "***********************************************************"
print "Accuracy on Dev data is", float(debug_acc)/float(twords)

fname = "../Data/a3-data/test.json"
data = json.load(open(fname))

TypetoIndex = {}
TypeFreq = Counter()

sentences = all_sentences(data, 0)

testclusters = {}

for i in range(k):
    testclusters[i] = []

correct = 0
incorrect = 0
total = 0
for sentence in sentences:
    for word in sentence:
        stype = word['text'].lower()
        if stype in trwordtocluster:
            cindex = trwordtocluster[stype]
            testclusters[cindex].append(word)

clustertotag = {}
debug_acc = 0
twords = 0
for cluster in testclusters:
    Tags=[]
    for word in testclusters[cluster]:
        Tags.append(word['tag'])
        twords += 1

    counter = Counter(Tags)
    clustertotag[cluster] = counter.most_common()[0][0]
    debug_acc = debug_acc + counter.most_common()[0][1]

print "***********************************************************"
print "Test tags"
print clustertotag
print "***********************************************************"
print "Accuracy on Test data is", float(debug_acc)/float(twords)
