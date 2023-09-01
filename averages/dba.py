import numpy as np
import utils


def calculate_dist_matrix(tseries, dist_fun, dist_fun_params):
    N = len(tseries)
    pairwise_dist_matrix = np.zeros((N, N), dtype=np.float64)
    # pre-compute the pairwise distance
    for i in range(N - 1):
        x = tseries[i]
        for j in range(i + 1, N):
            y = tseries[j]
            dist = dist_fun(x, y, **dist_fun_params)[0]
            # because dtw returns the sqrt
            dist = dist * dist
            pairwise_dist_matrix[i, j] = dist
            # dtw is symmetric 
            pairwise_dist_matrix[j, i] = dist
        pairwise_dist_matrix[i, i] = 0
    return pairwise_dist_matrix


def medoid(tseries, dist_fun, dist_fun_params, tseries_pad=None):
    """
    Calculates the medoid of the given list of MTS
    :param tseries: The list of time series 
    """
    N = len(tseries)
    if N == 1:
        return 0, tseries[0]
    if 'shape_reach' in dist_fun_params:
        tseries_to_medoid = tseries_pad
    else:
        tseries_to_medoid = tseries
    pairwise_dist_matrix = calculate_dist_matrix(tseries_to_medoid, dist_fun,
                                                 dist_fun_params)

    sum_dist = np.sum(pairwise_dist_matrix, axis=0)
    min_idx = np.argmin(sum_dist)
    med = tseries[min_idx]
    return min_idx, med


def sum_of_squares_partial(s, series, indices_for_sum, dist_fun, dist_fun_params):
    sum = 0
    for i in indices_for_sum:
        sum += dist_fun(s, series[i], **dist_fun_params)[0]
    return sum


def approx_medoid(tseries, dist_fun, dist_fun_params, tseries_pad=None):
    """
    Calculates the approximate medoid
    """
    N = len(tseries)
    if N == 1:
        return 0, tseries[0]
    if 'shape_reach' in dist_fun_params:
        tseries_to_medoid = tseries_pad
    else:
        tseries_to_medoid = tseries

    n_samples_candidates = min(10, N)
    n_samples_ss = min(20, N)
    medoid_ind = -1
    indices_candidates = np.random.choice(range(N), n_samples_candidates, replace=False)
    indices_ss = np.random.choice(range(N), n_samples_ss, replace=False)

    for i in range(indices_candidates.shape[0]):
        index_candidate = indices_candidates[i]
        candidate = tseries_to_medoid[index_candidate]
        ss = sum_of_squares_partial(candidate, tseries_to_medoid, indices_ss, dist_fun, dist_fun_params)
        if medoid_ind == -1 or ss < best_ss:
            best_ss = ss
            medoid_ind = index_candidate
    return medoid_ind, tseries[medoid_ind]


def compute_intraclass_inertia(x, y, distance_algorithm='dtw'):
    # get the distance function for the inertia calculations 
    dist_fun_inertia_dtw = utils.constants.DISTANCE_ALGORITHMS['dtw']
    # get the distance function params for the inertia calculations 
    dist_fun_params_inertia_dtw = utils.constants.DISTANCE_ALGORITHMS_PARAMS['dtw']
    # now for shapedtw 
    # get the distance function for the inertia calculations 
    dist_fun_inertia_shapedtw = utils.constants.DISTANCE_ALGORITHMS['shapedtw']
    # get the distance function params for the inertia calculations 
    dist_fun_params_inertia_shapedtw = utils.constants.DISTANCE_ALGORITHMS_PARAMS['shapedtw']
    # get unique classes list
    classes = np.unique(y)
    # the array of inertia per class 
    inertia_s_dtw = []
    inertia_s_shapedtw = []
    # loop through classes
    for c in classes:
        # get the series of this class only 
        x_c = x[np.where(y == c)]
        # average the set of MTS x_c
        avg_s = dba(x_c, distance_algorithm=distance_algorithm)
        inertia_dtw = []
        inertia_shapedtw = []
        # get the inertia 
        for mts in x_c:
            # get the distance between the average and the mts 
            dist_dtw = dist_fun_inertia_dtw(mts, avg_s, **dist_fun_params_inertia_dtw)[0]
            inertia_dtw.append(dist_dtw * dist_dtw)
            if distance_algorithm == 'shapedtw':
                dist_shapedtw = dist_fun_inertia_shapedtw(mts, avg_s, **dist_fun_params_inertia_shapedtw)[0]
                inertia_shapedtw.append(dist_shapedtw * dist_shapedtw)
        # append the mean of the inertia of this class 
        inertia_s_dtw.append(np.mean(inertia_dtw))
        if distance_algorithm == 'shapedtw':
            inertia_s_shapedtw.append(np.mean(inertia_shapedtw))
        else:
            inertia_s_shapedtw.append(np.mean(inertia_dtw))
    # return the mean of the inertia of all classes 
    return np.mean(inertia_s_dtw), np.mean(inertia_s_shapedtw)


def _dba_iteration(tseries, avg, dist_fun, dist_fun_params, weights, tseries_pad):
    """
    Perform one weighted dba iteration and return the new average 
    """
    # the number of time series in the set
    n = len(tseries)
    # length of the time series 
    ntime = avg.shape[0]
    # array containing the new weighted average sequence 
    new_avg = np.zeros((ntime,), dtype=np.float64)
    # array of sum of weights 
    sum_weights = np.zeros((ntime,), dtype=np.float64)
    avg_4_dist = avg
    if 'shape_reach' in dist_fun_params:
        reach = dist_fun_params['shape_reach']
        avg_4_dist = np.pad(avg, (reach, reach), mode='edge')
    # loop the time series 
    for s in range(n):
        series = tseries[s]
        ss = series
        if 'shape_reach' in dist_fun_params:
            ss = tseries_pad[s]
        dtw_dist, dtw = dist_fun(avg_4_dist, ss, **dist_fun_params)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # im = ax.imshow(dtw)
        # plt.savefig('out.pdf')
        # print(dtw)
        # exit()

        i = ntime
        j = series.shape[0]
        while i >= 1 and j >= 1:
            new_avg[i - 1] += series[j - 1] * weights[s]
            sum_weights[i - 1] += weights[s]

            a = dtw[i - 1, j - 1]
            b = dtw[i, j - 1]
            c = dtw[i - 1, j]
            if a < b:
                if a < c:
                    # a is the minimum
                    i -= 1
                    j -= 1
                else:
                    # c is the minimum
                    i -= 1
            else:
                if b < c:
                    # b is the minimum
                    j -= 1
                else:
                    # c is the minimum
                    i -= 1
    # update the new weighted avgerage 
    new_avg = new_avg / sum_weights

    return new_avg


def dba(tseries, max_iter=1, verbose=False, init_avg_method='approx_medoid',
        init_avg_series=None, distance_algorithm='dtw', weights=None, tseries_pad=None):
    """
    Computes the Dynamic Time Warping (DTW) Barycenter Averaging (DBA) of a 
    group of Multivariate Time Series (MTS). 
    :param tseries: A list containing the series to be averaged, where each 
        MTS has a shape (l,m) where l is the length of the time series and 
        m is the number of dimensions of the MTS - in the case of univariate 
        time series m should be equal to one
    :param max_iter: The maximum number of iterations for the DBA algorithm.
    :param verbose: If true, then provide helpful output.
    :param init_avg_method: Either: 
        'random' the average will be initialized by a random time series, 
        'medoid'(default) the average will be initialized by the medoid of tseries, 
        'manual' the value in init_avg_series will be used to initialize the average
    :param init_avg_series: this will be taken as average initialization if 
        init_avg_method is set to 'manual'
    :param distance_algorithm: Determine which distance to use when aligning 
        the time series
    :param weights: An array containing the weights to calculate a weighted dba
        (NB: for MTS each dimension should have its own set of weights)
        expected shape is (n,m) where n is the number of time series in tseries 
        and m is the number of dimensions
    """
    # get the distance function 
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get the distance function params 
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]

    # check if given dataset is empty 
    if len(tseries) == 0:
        # then return a random time series because the average cannot be computed 
        start_idx = np.random.randint(0, len(tseries))
        return np.copy(tseries[start_idx])

    if distance_algorithm == 'shapedtw':
        utils.utils.add_dist_fun_params_for_shapedtw(dist_fun_params, tseries[0].shape[0])

    # init DBA
    if init_avg_method == 'medoid':
        avg = np.copy(medoid(tseries, dist_fun, dist_fun_params, tseries_pad)[1])
    elif init_avg_method == 'approx_medoid':
        avg = np.copy(approx_medoid(tseries, dist_fun, dist_fun_params, tseries_pad)[1])
    elif init_avg_method == 'random':
        start_idx = np.random.randint(0, len(tseries))
        avg = np.copy(tseries[start_idx])
    else:  # init with the given init_avg_series
        avg = np.copy(init_avg_series)

    if len(tseries) == 1:
        return avg
    if verbose == True:
        print('Doing iteration')

    # main DBA loop 
    for i in range(max_iter):
        if verbose == True:
            print(' ', i, '...')
        if weights is None:
            # when giving all time series a weight equal to one we have the 
            # non - weighted version of DBA 
            weights = np.ones((len(tseries)), dtype=np.float64)
        # dba iteration 
        avg = _dba_iteration(tseries, avg, dist_fun, dist_fun_params, weights, tseries_pad)

    return avg
