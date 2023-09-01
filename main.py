from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import NB_ITERATIONS
from utils.constants import MAX_PROTOTYPES_PER_CLASS
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES
from utils.constants import DATA_CLUSTERING_ALGORITHMS

from utils.utils import read_all_datasets
from utils.utils import calculate_metrics
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import get_random_initial_for_kmeans
from utils.utils import read_dataset
from utils.utils import check_if_file_exits
from utils.utils import get_random_initial_for_kmeans_not_per_class

import utils
from kmeans import kmeans_save_itr
from kshape import KShape_save_itr

import numpy as np
import sys

import matplotlib.pyplot as plt

from averages.dba import dba
from averages.shapedba import shapedba
from averages.mean import mean
from averages.softdba import softdba

import time

def read_data_from_dataset(use_init_clusters=True):
    dataset_out_dir = root_dir_output + archive_name + '/' + dataset_name + '/'

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    y_train, y_test = transform_labels(y_train, y_test)

    classes, classes_counts = np.unique(y_train, return_counts=True)

    max_prototypes = min(classes_counts.max() + 1,
                         MAX_PROTOTYPES_PER_CLASS + 1)
    init_clusters = None

    if use_init_clusters == True:
        init_clusters = get_random_initial_for_kmeans(x_train, y_train,
                                                      max_prototypes, nb_classes, dataset_out_dir)
    return x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes, init_clusters

def data_culstering_function(curr_dir, data_clustering_algorithm_name, x_train,
                             K, init_clusters):
    if data_clustering_algorithm_name == 'kmeans_shapedba_shapedtw':
        return kmeans_save_itr(curr_dir, x_train, K, init_clusters=init_clusters,
                               distance_algorithm='shapedtw', averaging_algorithm='shapedba')
    if data_clustering_algorithm_name == 'kmeans_dba_dtw':
        return kmeans_save_itr(curr_dir, x_train, K, init_clusters=init_clusters,
                               distance_algorithm='dtw', averaging_algorithm='dba')
    if data_clustering_algorithm_name == 'kmeans_softdba_softdtw':
        return kmeans_save_itr(curr_dir, x_train, K, init_clusters=init_clusters,
                               distance_algorithm='softdtw', averaging_algorithm='softdba')
    if data_clustering_algorithm_name == 'kmeans_mean_ed':
        return kmeans_save_itr(curr_dir, x_train, K, init_clusters=init_clusters,
                               distance_algorithm='ed', averaging_algorithm='mean')
    if data_clustering_algorithm_name == 'kshape':
        return KShape_save_itr(curr_dir, x_train, K, init_clusters=init_clusters)

root_dir = '/home/aismailfawaz/phd/others_code/shape_dba/'
root_dir_output = root_dir + 'results/'
root_dir_dataset_archive = '/home/aismailfawaz/datasets/TSC/'

if sys.argv[1] == 'data_clustering':
    for archive_name in ARCHIVE_NAMES:
        datasets_dict = read_all_datasets(root_dir_dataset_archive, archive_name)
        d_names = utils.constants.dataset_names_for_archive[archive_name]
        for dataset_name in d_names:
            print('dataset_name: ', dataset_name)

            x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes, \
            init_clusters = read_data_from_dataset(use_init_clusters=False)

            dataset_out_dir = root_dir_output + archive_name + '/' + dataset_name + '/'

            init_clusters = get_random_initial_for_kmeans_not_per_class(
                x_train, y_train, nb_classes, dataset_out_dir)
            for data_clustering_algorithm_name in DATA_CLUSTERING_ALGORITHMS:
                for itr in range(NB_ITERATIONS):
                    new_out_dir = dataset_out_dir + data_clustering_algorithm_name + \
                                    '/itr_' + str(itr) + '/'
                    temp = new_out_dir

                    if check_if_file_exits(new_out_dir + 'df_metrics.csv'):
                        print('Already_done:', temp)
                        continue

                    new_out_dir_test = create_directory(new_out_dir + 'Doing/')
                    if new_out_dir_test is None:
                        print('Running:', temp)
                        continue

                    print('Doing:', temp)

                    # concat train and test since it is unsupervised 
                    y_true = np.concatenate((y_train, y_test), axis=0)
                    x = np.concatenate((x_train, x_test), axis=0)

                    start_time = time.time()
                    # apply the clustering algorithm 
                    _, y_pred = data_culstering_function(new_out_dir, data_clustering_algorithm_name,
                                                            x, nb_classes, init_clusters[itr])
                    duration = time.time() - start_time
                    # calculate the metrics 
                    df_metrics = calculate_metrics(y_true, y_pred, duration, clustering=True)
                    # save 
                    df_metrics.to_csv(new_out_dir + 'df_metrics.csv')
                    print(df_metrics)
                print('DONE')

elif sys.argv[1] == 'visualize_average':

    dataset_name = sys.argv[2]
    archive_name = sys.argv[3]
    averaging_method = sys.argv[4]

    datasets_dict = read_dataset(root_dir_dataset_archive, archive_name,
                                    dataset_name)

    x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes, init_clusters = \
        read_data_from_dataset()
    
    create_directory(directory_path=root_dir_output + archive_name + '/' + dataset_name + '/average_visualization')

    create_directory(directory_path=root_dir_output + archive_name + '/' + dataset_name + '/average_visualization/' + averaging_method)

    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)

    if averaging_method == 'shapeDBA':
        reach = utils.constants.DISTANCE_ALGORITHMS_PARAMS['shapedtw']['shape_reach']
        tseries_pad = np.pad(np.asarray(X), ((0, 0), (reach, reach)), mode='edge')

    plt.figure(figsize=(20,10))

    for c in range(nb_classes):

        if averaging_method == 'shapeDBA':
            avg = shapedba(tseries=X[Y == c], tseries_pad=tseries_pad)
        
        elif averaging_method == 'DBA':
            avg = dba(tseries=X[Y == c])
        
        elif averaging_method == 'mean':
            avg = mean(tseries=X[Y == c])
        
        elif averaging_method == 'softDBA':
            avg = softdba(tseries=X[Y == c])
        
        else:
            raise ValueError("No such averaging method exists.")
        
        plt.plot(X[Y == c].T, color='blue', alpha=0.7, lw=3)
        plt.plot(avg, lw=3, color='red', label='Average Time Series')

        plt.legend(prop={'size' : 20})

        plt.savefig(root_dir_output + archive_name + '/' + dataset_name + '/average_visualization/' + averaging_method + '/avg_class_'+str(c)+'.pdf')

        plt.cla()
    
    plt.clf()
    plt.close()