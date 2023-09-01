from tslearn.clustering import KShape
import numpy as np


def KShape_save_itr(curr_dir, tseries, nbclusters, max_iter=100, init_clusters=None):
    data = np.array(tseries)
    init_data = np.array(init_clusters)

    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
        init_data = init_data[:, :, np.newaxis]

    kshape = KShape(n_clusters=nbclusters, max_iter=max_iter, init=init_data)

    affect = kshape.fit_predict(data)
    clusters = kshape.cluster_centers_

    if len(data.shape) == 2:
        clusters = np.squeeze(clusters, axis=-1)

    np.save(curr_dir + '/clusters.npy', clusters)
    np.save(curr_dir + '/affect.npy', affect)
    return init_clusters, affect
