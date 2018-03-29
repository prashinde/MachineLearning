import numpy as np
import random

class kmeans:
    def __init__(self, k=5, miter=300):
        self.k = k
        self.miter = miter

    def cluster(self, points):
        self.centroids = np.zeros(self.k*points.shape[1], dtype=float)
        self.centroids = self.centroids.reshape((self.k, points.shape[1]))
        self.objective = []
        pavgd = 999

        ilist = random.sample(xrange(len(points)), self.k)
        self.centroids = np.array([points[i] for i in ilist])
        '''
        First k data points are centroids
        for i in range(self.k):
            #self.centroids[i] = points[i]
            self.objective[i] = []
        '''

        for rounds in range(self.miter):
            '''
            Each cluster is initially empty
            '''
            self.clusters={}
            self.distances={}

            avgdistance = 0

            for i in range(self.k):
                '''
                Each cluster is initially empty
                '''
                self.clusters[i]=[]
                self.distances[i]=[]

            '''
            Start going through each of the data point
            '''
            for point in points:
                edistance = [np.linalg.norm(point-centroid) for centroid in self.centroids]
                cluster = np.argmin(edistance)
                self.clusters[cluster].append(point)
                if not np.isnan(edistance[cluster]):
                    self.distances[cluster].append(edistance[cluster])

            '''
            Rebalance the centroids
            '''
            for cluster in self.clusters:
                self.centroids[cluster]=np.average(self.clusters[cluster], axis=0)
                #self.objective[cluster.append(np.average(self.distances[cluster], axis=0))
                avgdistance += np.average(self.distances[cluster], axis = 0)

            self.objective.append(avgdistance)
            if pavgd - avgdistance < 0.0001:
                break
            else:
                pavgd = avgdistance
                avgdistance = 0
        return self.centroids, self.clusters, self.objective
