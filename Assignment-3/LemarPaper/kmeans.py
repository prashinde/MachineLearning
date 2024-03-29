import numpy as np
import random
import json
class kmeans:
    def __init__(self, k=5, miter=2):
        self.k = k
        self.miter = miter

    def normalize_row(self, C):
        '''
        compute on seperate left and right descriptors
        '''
        ldesc = C[0:len(C)/2]
        rdesc = C[len(C)/2:]

        l2 = np.atleast_1d(np.linalg.norm(ldesc, ord=2, axis=0))
        if l2 == 0:
            l2 = 1
        ldesc = ldesc / l2

        l2 = np.atleast_1d(np.linalg.norm(rdesc, ord=2, axis=0))
        if l2 == 0:
            l2 = 1
        rdesc = rdesc / l2
        return np.concatenate((ldesc, rdesc))

    def newCentroid(self, pointlist, TypeFreq, D):
        centroid = np.zeros(D[0].shape, dtype=np.float)
        tfreq = 0
        for point in pointlist:
            freq = TypeFreq[point][1]
            tfreq = tfreq + freq
            centroid = centroid+(freq*D[point])
        if tfreq != 0:
            centroid = centroid/tfreq
        nc = self.normalize_row(centroid)
        return nc

    def assignCluster(self, centroids, points, TypeFreq):
        wordtocluster = {}
        ddistance = points.dot(self.centroids.T)
        TypeFreqMC = TypeFreq.most_common()
        idx = 0
        for point in points:
            cluster = np.argmax(ddistance[idx])
            wordtocluster[TypeFreqMC[idx][0]] = cluster
            idx = idx + 1
        return wordtocluster

    def cluster(self, points, TypetoFreq):
        self.centroids = np.zeros(self.k*points.shape[1], dtype=float)
        self.centroids = self.centroids.reshape((self.k, points.shape[1]))
        self.objective = []
        
        self.olcentroids = np.zeros(self.k*points.shape[1], dtype=float)
        self.olcentroids = self.centroids.reshape((self.k, points.shape[1]))

        TypeFreqS = TypetoFreq.most_common()
        pavgd = 0

        #ilist = random.sample(xrange(len(points)), self.k)

        '''
        Pickup first k points from TypetoFreq as centroid
        '''
        self.centroids = points[0:self.k]

        '''
        First k data points are centroids
        for i in range(self.k):
            self.centroids[i] = points[i]
        '''
        for rounds in range(self.miter):
            self.clusters={}
            self.distances={}

            avgdistance = 0

            for i in range(self.k):
                self.clusters[i]=[]
                self.distances[i]=[]

            idx = 0
            '''
            Points is Ntx2000
            Centroids is Kx2000
            Points . Centroids.T = Nt x 2000 . 2000 x k
            result is Ntxk
            '''
            ddistance = points.dot(self.centroids.T)
            for point in points:
                cluster = np.argmax(ddistance[idx])
                self.clusters[cluster].append(idx)
                #elf.distances[cluster].append(ddistance[idx][cluster])
                avgdistance = avgdistance + ddistance[idx][cluster]
                idx = idx + 1

            for cluster in self.clusters:
                #if len(self.clusters[cluster]) > 0:
                #self.centroids[cluster]
                nc = self.newCentroid(self.clusters[cluster], TypeFreqS, points)
                self.centroids[cluster] = nc

            self.objective.append(avgdistance)
            print "Finished Round", rounds
            print avgdistance, avgdistance-pavgd
            if avgdistance - pavgd < 0.001:
                break
            else:
                pavgd = avgdistance
        return self.centroids, self.clusters, self.objective
        #return self.centroids
