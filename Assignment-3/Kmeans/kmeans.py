import numpy as np

class kmeans:
    def __init__(self, k=5, miter=300):
        self.k = k
        self.miter = miter

    def cluster(self, points):
        self.centroids = np.zeros(self.k*points.shape[1], dtype=float)
        self.centroids = self.centroids.reshape((self.k, points.shape[1]))
        '''
        First k data points are centroids
        '''
        for i in range(self.k):
            self.centroids[i]=points[i]

        for rounds in range(self.miter):
            '''
            Each cluster is initially empty
            '''
            self.clusters={}

            for i in range(self.k):
                '''
                Each cluster is initially empty
                '''
                self.clusters[i]=[]

            '''
            Start going through each of the data point
            '''
            for point in points:
                edistance = [np.linalg.norm(point-centroid) for centroid in self.centroids]
                cluster = np.argmin(edistance)
                self.clusters[cluster].append(point)

            '''
            Rebalance the centroids
            '''
            for cluster in self.clusters:
                self.centroids[cluster]=np.average(self.clusters[cluster], axis=0)

        return self.centroids, self.clusters
