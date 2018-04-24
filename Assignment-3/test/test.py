import numpy as np

A=np.array([[1,2,3,2],[-3,2,1,-4]], dtype=np.float)

print A
for row in A:
    fmax = row.max()
    tmin = 0
    tmax = 1
    fmin = row.min()
    print ((row-fmin)*(tmax-tmin))/((fmax-fmin)+tmin)
