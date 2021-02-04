from __future__ import absolute_import
import engineWrapper
import numpy as np
a = engineWrapper.Wrapper(5)
f = open('../../sample.txt', 'r')

l = [line.strip() for line in f]

size = int(l[0])

X = []
for line in l[1:]:
    X.append(line.split(' '))

X = np.asarray(X)
a.pySetPoints(X)
a.constructGraph()
print(a.pyFindKNearestNeighbors(np.asarray([[1000, 395]])))
