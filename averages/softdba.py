from tslearn.barycenters import softdtw_barycenter
from averages.dba import medoid
import numpy as np
import utils


def softdba(tseries, max_iter=100, verbose=False, init_avg_method='medoid',
            init_avg_series=None, distance_algorithm='softdtw', weights=None):
    # get the distance function
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get the distance function params
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]

    # init the average
    if init_avg_method == 'medoid':
        avg = np.copy(medoid(tseries, dist_fun, dist_fun_params)[1])

    avg = avg.reshape(-1, 1)

    avg = softdtw_barycenter(tseries, init=avg, gamma=dist_fun_params['gamma'],
                             max_iter=max_iter)

    return avg.reshape(-1, )
