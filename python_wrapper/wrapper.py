from __future__ import absolute_import
import engineWrapper
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN


class enginewrapper(BaseANN):
    def __init__(self, metric):
        self.metric = metric
        self.name = 'Our'

    def fit(self, X):
        self.p = engineWrapper.Wrapper(5)
        X = np.asarray(X)
        self.p.pySetPoints(X)
        self.p.constructGraph()

    def query(self, v, n):
        v = [v]
        return self.p.pyFindKNearestNeighbors(np.asarray(v), n)

    def set_query_arguments(self, ef):
        pass
