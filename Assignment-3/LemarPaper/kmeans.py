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
        ldesc = C[0:500]
        rdesc = C[500:1000]

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
        return self.normalize_row(centroid)


    def cluster(self, points, TypetoFreq):
        self.centroids = np.zeros(self.k*points.shape[1], dtype=float)
        self.centroids = self.centroids.reshape((self.k, points.shape[1]))
        self.objective = []
        TypeFreqS = TypetoFreq.most_common()
        pavgd = 999

        print points.shape
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
                if not np.isnan(ddistance[idx][cluster]):
                    self.distances[cluster].append(ddistance[idx][cluster])
                    avgdistance = avgdistance + ddistance[idx][cluster]
                idx = idx + 1

            sums = np.array([])
            #print '************************************************'
            for cluster in self.clusters:
                if len(self.clusters[cluster]) > 0:
                    self.centroids[cluster] = self.newCentroid(self.clusters[cluster], TypeFreqS, points)
                    #print self.centroids[cluster]
                    #self.objective[cluster].append(np.average(self.distances[cluster], axis=0))
                    #avgdistance = avgdistance+sum(self.distances[cluster])
                    sums = np.append(sums, sum(self.distances[cluster]))
            #print '---------------------------------------------------'
            #if not rounds == 0:
            self.objective.append(avgdistance)
            print "Finished Round", rounds
            print pavgd, avgdistance, abs(pavgd-avgdistance)
            #if abs(pavgd - avgdistance) < 0.0001:
            #    break
            #else:
            #    pavgd = avgdistance
            #    avgdistance = 0
        return self.centroids, self.clusters, self.objective
        #return self.centroids
