import numpy as np
import utils
import operator


def kmeans(tseries, nbclusters, nb_iterations=10, init_clusters=None, init_avg_with_clusters=False,
           averaging_algorithm='dba', distance_algorithm='dtw', tseries_pad=None):
    """
    Performs the k-means algorithm using the averaging algorithm specified. 
    :param tseries: The list of time series on which we would like to apply k-means. 
    :param nb_iterations: Maximum number of k-means iterations. 
    :param nbclusters: The number of clusters a.k.a 'k' in the k-means  
    """
    # get the averaging method (check constants.py)
    avg_method = utils.constants.AVERAGING_ALGORITHMS[averaging_algorithm]
    # get the distance method (check constants.py)
    distance_method = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get dsitance method parameters 
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]

    if distance_algorithm == 'shapedtw':
        utils.utils.add_dist_fun_params_for_shapedtw(dist_fun_params, tseries[0].shape[0])

    current_pad = None

    # init the clusters 
    if init_clusters is None:
        # init the k-means algorithm randomly 
        clusterIdx = np.random.permutation(len(tseries))[:nbclusters]
        clusters = [tseries[i] for i in clusterIdx]
    else:
        clusters = init_clusters

    for i in range(nb_iterations):

        if distance_algorithm == 'shapedtw':
            reach = dist_fun_params['shape_reach']
            clusters_pad = np.pad(np.asarray(clusters), ((0, 0), (reach, reach)), mode='edge')

        # affect the time series to its closest clusters 
        affect = []
        # init array of distance to closest cluster for each time series 
        distances_clust = []
        # start the affectation of each time series 
        for idx_series, series in enumerate(tseries):
            # initialize the cluster index of the series 
            cidx = -1
            # initialize the minimum distance to a cluster 
            minDist = np.inf
            # loop through all clusters 
            for i, cluster in enumerate(clusters):
                # calculate the distance between the cluster and the series
                ss = series
                cc = cluster
                if distance_algorithm == 'shapedtw':
                    ss = tseries_pad[idx_series]
                    cc = clusters_pad[i]
                dist, __ = distance_method(ss, cc, **dist_fun_params)
                # check if it is better than best so far minimum distance 
                if dist < minDist:
                    # assign the new cluster and the minimum distance 
                    minDist = dist
                    cidx = i
            # add the best cluster index to the affectation array 
            affect.append(cidx)
            # this is used to solve the empty clusters problem 
            distances_clust.append((idx_series, minDist))

        # empty clusters list 
        empty_clusters = []
        # recompute the clusters 
        for i in range(nbclusters):
            # get the new group of time series affected to cluster index i 
            current = [tseries[j] for j, x in enumerate(affect) if x == i]
            if distance_algorithm == 'shapedtw':
                current_pad = [tseries_pad[j] for j, x in enumerate(affect) if x == i]
            # check if cluster is empty 
            if len(current) == 0:
                # add the cluster index to the empty clusters list
                empty_clusters.append(i)
                # skip averaging an empty cluster
                continue
                # compute the new avg or center of this cluster
            current = np.asarray(current)
            if distance_algorithm == 'shapedtw':
                current_pad = np.asarray(current_pad)
                dba_avg = avg_method(current, tseries_pad=current_pad)
            else:
                dba_avg = avg_method(current)
            # assign this new cluster center whose index is i
            clusters[i] = dba_avg

        # check if we have empty clusters  
        if len(empty_clusters) > 0:
            # sort the distances to get the ones furthest from their clusters 
            distances_clust.sort(key=operator.itemgetter(1), reverse=True)
            # loop through the empty clusters 
            for i, idx_clust in enumerate(empty_clusters):
                # replace the empty cluster with the farest time series from its old cluster 
                clusters[idx_clust] = tseries[distances_clust[i][0]]

    # re-affectation 
    # affect the time series to its closest clusters 
    affect = []
    # start the affectation of each time series 
    for idx_series, series in enumerate(tseries):
        # initialize the cluster index of the series 
        cidx = -1
        # initialize the minimum distance to a cluster 
        minDist = np.inf
        # loop through all clusters 
        for i, cluster in enumerate(clusters):
            ss = series
            cc = cluster
            if distance_algorithm == 'shapedtw':
                ss = tseries_pad[idx_series]
                cc = clusters_pad[i]
            # calculate the distance between the cluster and the series 
            dist, __ = distance_method(ss, cc, **dist_fun_params)
            # check if it is better than best so far minimum distance 
            if dist < minDist:
                # assign the new cluster and the minimum distance 
                minDist = dist
                cidx = i
        # add the best cluster index to the affectation array 
        affect.append(cidx)

    # return the clusters as well as the corre
    return clusters, affect


def kmeans_save_itr(curr_dir, tseries, nbclusters, nb_iterations=10, init_clusters=None,
                    averaging_algorithm='dba', distance_algorithm='dtw'):
    """
    This function is similar to normal kmeans
    but saves the state at each kmeans iteration
    in order to continue execution on mesocentre
    :param curr_dir:
    :param tseries:
    :param nbclusters:
    :param nb_iterations:
    :param init_clusters:
    :param averaging_algorithm:
    :param distance_algorithm:
    :return: usual kmeans result
    """
    tseries_pad = None
    # if shapedtw is used then you should pad before
    # this is used because padding inside the distance function is very expensive
    if distance_algorithm == 'shapedtw':
        reach = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]['shape_reach']
        tseries_pad = np.pad(np.asarray(tseries), ((0, 0), (reach, reach)), mode='edge')

    clusters = init_clusters
    for k_itr in range(nb_iterations):
        kmeans_file = curr_dir + '/kmeans_clusters_' + str(k_itr) + '.npy'
        kmeans_affect_file = curr_dir + '/kmeans_affect_' + str(k_itr) + '.npy'
        if utils.utils.check_if_file_exits(kmeans_file):
            # clusters already computed till this iteration
            clusters = np.load(kmeans_file)
            affect = np.load(kmeans_affect_file)
        else:
            # run the kmeans algorithm for one iteration
            # with this clusters as init
            clusters, affect = kmeans(tseries, nbclusters, nb_iterations=1, init_clusters=clusters,
                                      averaging_algorithm=averaging_algorithm, init_avg_with_clusters=True,
                                      distance_algorithm=distance_algorithm, tseries_pad=tseries_pad)
            np.save(kmeans_file, np.array(clusters))
            np.save(kmeans_affect_file, np.array(affect))

    return init_clusters, affect
