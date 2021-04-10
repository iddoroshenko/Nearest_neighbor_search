import engineWrapper
import numpy as np

e = engineWrapper.Wrapper(5)
X = [[1, 1], [3, 3], [6, 6]]
X = np.asarray(X)
e.pySetPoints(X)
e.constructGraph()
Y = [[4, 4]]
Y = np.asarray(Y)
Z = e.pyFindKNearestNeighbors(Y, 3)
print(Z)