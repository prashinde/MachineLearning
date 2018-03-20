import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans

'''
https://stackoverflow.com/questions/33143815/reading-a-two-column-file-into-python
https://stackoverflow.com/questions/41468116/python-how-to-combine-two-flat-lists-into-a-2d-array/41468178
'''
def readfile(fname):
    with open(fname, 'r') as data:
        xs = []
        ys = []
        for line in data:
            p = line.split()
            xs.append(float(p[0]))
            ys.append(float(p[1]))

    return np.asarray(xs), np.asarray(ys)

fname='../Data/a3-data/2d-kmeans-input.ssv'
xs, ys = readfile(fname)
points = np.column_stack((xs, ys))
#print points

k=5
nm = kmeans(k)
centroids, clusters = nm.cluster(points)

scls=[]
for cluster in clusters:
    x=[]
    y=[]
    cpoints = clusters[cluster]
    for point in cpoints:
        x.append(point[0])
        y.append(point[1])
    scls.append(np.random.rand(3,))
    plt.scatter(x, y, c=scls[cluster])

#plt.scatter(xs, ys)
i = 0
for c in centroids:
    plt.plot(c[0], c[1], c=scls[i], marker='D', markerfacecolor='white')
    i = i+1
plt.show()
