# This is the arithmetic mean (Euclidean)
from averages.dba import dba


def mean(tseries):
    return dba(tseries, distance_algorithm='ed', max_iter=1)