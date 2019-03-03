import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from future.utils import iteritems

from datetime import datetime
import numpy as np

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): 
            sl = SortedList() 
            for j,xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add( (d, self.y[j]) )

            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v,0) + 1
            max_votes = 0
            max_votes_class = -1
            for v,count in iteritems(votes):
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)