from __future__ import absolute_import
import enginewrapper
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN


class enginewrapper(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = metric
        self.method_param = method_param
        self.name = 'Our'

    def fit(self, X):
        self.p = enginewrapper.Wrapper(5)
        X = np.asarray(X)
        self.p.pySetPoints(X)
        self.p.constructGraph()

    def query(self, v, n):
        return self.a.pyFindKNearestNeighbors(np.asarray(v), n)
