import numpy as np
import operator
np.set_printoptions(threshold=np.nan)
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from kmeans import kmeans
import json

class manytoone:
    def __init__(self, clusters, GoldenTags, TypeFreq):
        self.clusters = clusters
        self.tags = GoldenTags
        self.TypeFreqs = TypeFreq.most_common()
        self.debug_acc = 0

    def assign(self):
        clustertotag = {}
        for cluster in self.clusters:
            Tags=[]
            for point in self.clusters[cluster]:
                Tags.append(self.tags[(self.TypeFreqs[point][0]).lower()])

            counter = Counter(Tags)
            clustertotag[cluster] = counter.most_common()[0][0]
            self.debug_acc = self.debug_acc + counter.most_common()[0][1]
        return clustertotag

    def evaluate(self, clusters, clustertags):
        correct = 0
        twords = 0
        incorrect = 0
        for cluster in self.clusters:
            cctag = clustertags[cluster]
            for point in self.clusters[cluster]:
                wtag = self.tags[(self.TypeFreqs[point][0]).lower()]
                if wtag == cctag:
                    correct = correct+1
                else:
                    incorrect = incorrect + 1
                twords = twords+1
        print 'predicted correct', self.debug_acc
        print correct, " ", twords, " ", incorrect
        return float(correct)/float(twords)
