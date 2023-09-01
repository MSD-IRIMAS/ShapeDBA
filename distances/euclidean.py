# this contains the euclidean distance 
# to average using the ed you can simply call dba with the argument distance_algorithm = 'ed'
from distances.dtw.dtw import dynamic_time_warping as dtw

def euclidean_distance(x,y,w=None):
    """
    This function returns the euclidean distance between two multivariate
    time series x and y
    """
    # simply returns dtw with warping window equal to zero 
    return dtw(x,y,w=0)