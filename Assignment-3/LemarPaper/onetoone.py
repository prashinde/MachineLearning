import numpy as np
import operator
np.set_printoptions(threshold=np.nan)
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from kmeans import kmeans
import json

class onetoone:
    def __init__(self, clustertoword, nclustertoword):
        self.clusters = clustertoword
        self.nclustertoword = nclustertoword
        self.debug_acc = 0

    def assign(self):
        '''
        build unique tags first
        '''
        Tags = []
        for cluster in self.clusters:
            for word in self.clusters[cluster]:
                Tags.append(word['tag'])

        tagfrequency = Counter(Tags)
        GoldTags = {}

        for ttype in tagfrequency.most_common():
            tag = ttype[0]
            accuracies = np.zeros(len(self.clusters), dtype=float)
            '''
            For each cluster compute the accruracy of assigning that 
            tag to cluster
            '''
            for cluster in self.clusters:
                correct = 0
                total = 0
                if cluster in GoldTags:
                    accuracies[cluster] = 0
                else:
                    for word in self.nclustertoword[cluster]:
                        if word['tag'] == tag:
                            correct += 1
                        total += 1
                    accuracies[cluster] = float(correct)/float(total)

            dcluster = np.argmax(accuracies)
            if accuracies[dcluster] != 0:
                GoldTags[dcluster] = tag

        return GoldTags

    def accuracy(self, gtags):
        correct = 0
        total = 0
        for cluster in self.nclustertoword:
            if cluster not in gtags:
                continue
            for word in self.nclustertoword[cluster]:
                ptag = gtags[cluster]
                if ptag == word['tag']:
                    correct += 1
                total += 1
        return float(correct)/float(total)
''''
clustertoword = {}

for i in range(2):
    clustertoword[i] =[]


A=[{'text':'Pratik', 'tag':'NN'},{'text':'SS', 'tag':'Verb'},{'text':'ML', 'tag':'Noun'},{'text':'QQ', 'tag':'ADj'},{'text':'Alg', 'tag':'NN'}]
for cluster in clustertoword:
    [clustertoword[cluster].append(x) for x in A]

o2m = onetomany(clustertoword, A)
gtags = o2m.assign()
print gtags
'''
