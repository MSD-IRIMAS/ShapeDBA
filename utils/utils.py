import numpy as np
import pandas as pd
import os
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

import operator
from scipy.stats import wilcoxon

from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import NB_ITERATIONS
from utils.constants import MAX_PROTOTYPES_PER_CLASS
from utils.constants import DATA_CLUSTERING_ALGORITHMS


def get_random_initial_for_kmeans(x_train, y_train, max_prototypes, nb_classes,
                                  dataset_out_dir):
    """
    This function returns the initial clusters for kmeans based
    It is to make sur that all algorithms are initialised with the same elements 
    In this case al time series should be same length because we use np arrays 
    """
    dirr = create_directory(dataset_out_dir + 'init_clusters/')
    if dirr is None:
        # it has already been initialized ; load it 
        res = np.load(dataset_out_dir + 'init_clusters/init_clusters.npy')
    else:
        res = np.zeros((max_prototypes, NB_ITERATIONS, nb_classes, max_prototypes,
                        x_train.shape[1]), dtype=np.float64)
        for i in range(1, max_prototypes):
            for j in range(NB_ITERATIONS):
                for c in range(nb_classes):
                    c_x_train = x_train[np.where(y_train == c)]
                    num_clusters = min(i, len(c_x_train))
                    clusterIdx = np.random.permutation(len(c_x_train))[:num_clusters]
                    clusters = [c_x_train[cl] for cl in clusterIdx]
                    for cc in range(num_clusters):
                        res[i, j, c, cc, :] = np.array(clusters[cc])
        np.save(dirr + 'init_clusters.npy', res)
    return res


def get_random_initial_for_kmeans_not_per_class(x_train, y_train,
                                                K, dataset_out_dir):
    """
    It is mainly used for pure unsupervised clustering 
    We assume that the nb_clusters will be equal to nb_classes
    This function returns the initial clusters for kmeans based
    It is to make sur that all algorithms are initialised with the same elements 
    In this case al time series should be same length because we use np arrays 
    """
    dirr = create_directory(dataset_out_dir + 'init_clusters_not_per_class/')
    if dirr is None:
        # it has already been initialized ; load it 
        res = np.load(dataset_out_dir + 'init_clusters_not_per_class/init_clusters.npy')
    else:
        res = np.zeros((NB_ITERATIONS, K,
                        x_train.shape[1]), dtype=np.float64)
        for j in range(NB_ITERATIONS):
            clusterIdx = np.random.permutation(len(x_train))[:K]
            clusters = [x_train[cl] for cl in clusterIdx]
            for cc in range(K):
                res[j, cc, :] = np.array(clusters[cc])
        np.save(dirr + 'init_clusters.npy', res)
    return res


def zNormalize(x):
    x_mean = x.mean(axis=0)  # mean for each dimension of time series x
    x_std = x.std(axis=0)  # std for each dimension of time series x
    x = (x - x_mean) / (x_std)
    return x


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def check_if_file_exits(file_name):
    return os.path.exists(file_name)


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile
            return None
        return directory_path


def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous 
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit 
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # fit the encoder 
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels 
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test


def read_all_datasets(root_dir, archive_name, sort_dataset_name=False):
    datasets_dict = {}

    dataset_names_to_sort = []

    if archive_name == 'UCRArchive_2018':

        for dataset_name in DATASET_NAMES_2018:
            root_dir_dataset = root_dir + archive_name + '/' + dataset_name + '/'

            df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

            df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    return datasets_dict


def calculate_metrics(y_true, y_pred, duration, clustering=False):
    """
    Return a data frame that contains the precision, accuracy, recall and the duration
    For clustering it applys the adjusted rand index
    """
    if clustering == False:
        res = pd.DataFrame(data=np.zeros((1, 5), dtype=np.float64), index=[0],
                           columns=['precision', 'accuracy', 'error', 'recall', 'duration'])
        res['precision'] = precision_score(y_true, y_pred, average='macro')
        res['accuracy'] = accuracy_score(y_true, y_pred)
        res['recall'] = recall_score(y_true, y_pred, average='macro')
        res['duration'] = duration
        res['error'] = 1 - res['accuracy']
        return res
    else:
        res = pd.DataFrame(data=np.zeros((1, 2), dtype=np.float64), index=[0],
                           columns=['ari', 'duration'])
        res['duration'] = duration
        res['ari'] = adjusted_rand_score(y_pred, y_true)
        return res


def dataset_is_ready_to_plot(df_res, dataset_name, archive_name, array_algorithm_names):
    for algorithm_name in array_algorithm_names:
        # if any algorithm algorithm is not finished do not plot
        if not any(df_res.loc[(df_res['dataset_name'] == dataset_name) \
                              & (df_res['archive_name'] == archive_name)] \
                           ['algorithm_name'] == algorithm_name) \
                or (df_res.loc[(df_res['dataset_name'] == dataset_name) \
                               & (df_res['archive_name'] == archive_name) \
                               & (df_res['algorithm_name'] == algorithm_name)] \
                            ['nb_prototypes'].max() != MAX_PROTOTYPES_PER_CLASS):
            return False
    return True


def plot_pairwise(root_dir, classifier_name_1, classifier_name_2,
                  res_df=None, not_completed=None, title='', fig=None, color='green', label=None):
    if fig is None:
        plt.figure()
    else:
        plt.figure(fig)

    if res_df is None:
        res_df, not_completed = generate_results_csv_data_cluster('results-data-cluster.csv', root_dir,
                                                                  [classifier_name_1, classifier_name_2])

    sorted_df = res_df.loc[(res_df['algorithm_name'] == classifier_name_1) | \
                           (res_df['algorithm_name'] == classifier_name_2)]. \
        sort_values(['algorithm_name', 'archive_name', 'dataset_name'])
    # number of classifier we are comparing is 2 since pairwise
    m = 2

    # get max nb of ready datasets
    # count the number of tested datasets per classifier
    # df_counts = pd.DataFrame({'count': sorted_df.groupby(
    #     ['algorithm_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    # max_nb_datasets = df_counts['count'].max()
    # min_nb_datasets = df_counts['count'].min()

    # print(max_nb_datasets, min_nb_datasets)
    # # both classifiers should have finished
    # assert (max_nb_datasets == min_nb_datasets)

    # data = np.array(sorted_df['ari']).reshape(m, max_nb_datasets).transpose()

    # concat the dataset name and the archive name to put them in the columns s
    sorted_df['archive_dataset_name'] = sorted_df['archive_name'] + '__' + \
                                        sorted_df['dataset_name']

    sorted_df.reset_index(inplace=True)
    sorted_df.drop('index', axis=1, inplace=True)

    # remove non completed
    for not_completed_dname in not_completed:
        sorted_df.drop(sorted_df.loc[sorted_df['dataset_name'] == not_completed_dname].index, axis=0,
                       inplace=True)
    sorted_df.reset_index(inplace=True)
    sorted_df.drop('index', axis=1, inplace=True)

    nb_datasets = sorted_df.shape[0] // 2

    data = np.array(sorted_df['ari']).reshape(m, nb_datasets).transpose()

    # create the data frame containg the accuracies
    df_data = pd.DataFrame(data=data, columns=np.sort([classifier_name_1, classifier_name_2]),
                           index=np.unique(sorted_df['archive_dataset_name']))

    # assertion
    dnamee = 'BirdChicken'
    p1 = float(sorted_df.loc[(sorted_df['algorithm_name'] == classifier_name_1) &
                             (sorted_df['dataset_name'] == dnamee)]['ari'])
    p2 = float(df_data[classifier_name_1][sorted_df['archive_name'][0] + '__' + dnamee])
    assert (p1 == p2)
    x_line = np.arange(start=-0.1, stop=1.02, step=0.01)
    plt.xlim(xmax=1.02, xmin=-0.1)
    plt.ylim(ymax=1.02, ymin=-0.1)

    plt.scatter(x=df_data[classifier_name_1], y=df_data[classifier_name_2], color=color,
                label=label)

    # annotate with dataset name
    for label in df_data.index:
        if np.abs(df_data.loc[label][classifier_name_1] - df_data.loc[label][classifier_name_2]) >= 0.2:
            plt.annotate(label.split('__')[-1], (df_data[classifier_name_1][label]
                                                 , df_data[classifier_name_2][label]))

    # # add std lines
    # for dn in np.unique(sorted_df['dataset_name']):
    #     cur_df = sorted_df.loc[sorted_df['dataset_name'] == dn]
    #     if len(cur_df) == 2:
    #         x = cur_df.loc[cur_df['algorithm_name'] == classifier_name_1]['ari'].values[0]
    #         y = cur_df.loc[cur_df['algorithm_name'] == classifier_name_2]['ari'].values[0]
    #         xstd = cur_df.loc[cur_df['algorithm_name'] == classifier_name_1]['std'].values[0]
    #         ystd = cur_df.loc[cur_df['algorithm_name'] == classifier_name_2]['std'].values[0]
    #         plt.plot([x, x], [y - ystd, y + ystd], color='gray')
    #         plt.plot([x - xstd, x + xstd], [y, y], color='gray')

    plt.xlabel(classifier_name_1, fontsize='large')
    plt.ylabel(classifier_name_2, fontsize='large')
    plt.plot(x_line, x_line, color='black')
    # plt.legend(loc='upper left')
    plt.title(title)

    print('Wins are for', classifier_name_2)

    uniq, counts = np.unique(df_data[classifier_name_1] < df_data[classifier_name_2], return_counts=True)
    print('Wins', counts[-1])

    uniq, counts = np.unique(df_data[classifier_name_1] == df_data[classifier_name_2], return_counts=True)
    print('Draws', counts[-1])

    uniq, counts = np.unique(df_data[classifier_name_1] > df_data[classifier_name_2], return_counts=True)
    print('Losses', counts[-1])

    p_value = wilcoxon(df_data[classifier_name_1], df_data[classifier_name_2], zero_method='pratt')[1]
    print(p_value)

    plt.savefig(root_dir + '/' + classifier_name_1 + '-' + classifier_name_2 + '_' + title + '.pdf'
                , bbox_inches='tight')

def init_empty_df_metrics():
    return pd.DataFrame(data=np.zeros((0, 5), dtype=np.float64), index=[],
                        columns=['precision', 'accuracy', 'error', 'recall', 'duration'])


def get_df_metrics_from_avg(avg_df_metrics):
    res = pd.DataFrame(data=np.zeros((1, 5), dtype=np.float64), index=[0],
                       columns=['precision', 'accuracy', 'error', 'recall', 'duration'])
    res['accuracy'] = avg_df_metrics['accuracy'].mean()
    res['precision'] = avg_df_metrics['precision'].mean()
    res['error'] = avg_df_metrics['error'].mean()
    res['recall'] = avg_df_metrics['recall'].mean()
    res['duration'] = avg_df_metrics['duration'].mean()
    return res


def get_df_metrics_from_avg_data_cluster(avg_df_metrics):
    res = pd.DataFrame(data=np.zeros((1, 3), dtype=np.float64), index=[0],
                       columns=['ari', 'duration', 'std'])
    res['ari'] = avg_df_metrics['ari'].mean()
    res['std'] = avg_df_metrics['ari'].std()
    res['duration'] = avg_df_metrics['duration'].mean()
    return res


def add_dist_fun_params_for_shapedtw(dist_fun_params, max_length):
    shape_reach = dist_fun_params['shape_reach']
    length_padded = max_length + 2 * shape_reach
    cost_mat, cost_mat_dtw, delta_mat, delta_mat_padded = np.zeros((max_length, max_length), dtype=np.float64), \
                                                          np.zeros((max_length, max_length), dtype=np.float64), \
                                                          np.zeros((max_length, max_length), dtype=np.float64), \
                                                          np.zeros((length_padded, length_padded), dtype=np.float64)
    dist_fun_params['cost_mat'] = cost_mat
    dist_fun_params['cost_mat_dtw'] = cost_mat_dtw
    dist_fun_params['delta_mat'] = delta_mat
    dist_fun_params['delta_mat_padded'] = delta_mat_padded


def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}

    file_name = root_dir + '/' + archive_name + '/' + dataset_name + '/' + dataset_name

    if archive_name == 'UCRArchive_2018':
        root_dir_dataset = root_dir + archive_name + '/' + dataset_name + '/'

        df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

        df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]

        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    else:

        x_train, y_train = readucr(file_name + '_TRAIN')
        x_test, y_test = readucr(file_name + '_TEST')
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    return datasets_dict


def generate_results_csv_data_cluster(output_file_name, root_dir, array_algorithm_names):
    res = pd.DataFrame(data=np.zeros((0, 6), dtype=np.float64), index=[],
                       columns=['algorithm_name', 'archive_name', 'dataset_name',
                                'ari', 'duration', 'std'])
    import utils

    not_completed = set()

    for archive_name in ARCHIVE_NAMES:
        for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
            for algorithm_name in array_algorithm_names:
                output_dir = root_dir + '/results/' + archive_name + \
                             '/' + dataset_name + '/' + algorithm_name + '/'
                avg_df_metrics = pd.DataFrame(data=np.zeros((0, 3), dtype=np.float64), index=[],
                                              columns=['ari', 'duration', 'std'])

                for itr in range(NB_ITERATIONS):
                    temp_dir = output_dir + 'itr_' + str(itr) + '/df_metrics.csv'
                    if not os.path.exists(temp_dir):
                        print('Not_completed:', temp_dir)
                        not_completed.add(dataset_name)
                        continue
                    df_metrics_itr = pd.read_csv(temp_dir)
                    avg_df_metrics = pd.concat((avg_df_metrics, df_metrics_itr), sort=False)
                if not avg_df_metrics.empty:
                    avg_df_metrics = get_df_metrics_from_avg_data_cluster(avg_df_metrics)
                    avg_df_metrics['algorithm_name'] = algorithm_name
                    avg_df_metrics['archive_name'] = archive_name
                    avg_df_metrics['dataset_name'] = dataset_name
                    res = pd.concat((res, avg_df_metrics), axis=0, sort=False)

    res.to_csv(root_dir + output_file_name, index=False)
    return res, not_completed

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name)
    plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float64), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code 

    # plot losses 
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')


def plot_compare_curves():
    from scipy.interpolate import spline
    root_dir = '/mnt/nfs/casimir/'
    classifier_name = 'resnet'
    archive_name = 'TSC'
    dataset_name = 'Meat'
    metric = 'loss'
    # read the original history
    df_history_or = pd.read_csv(root_dir + 'results/' + classifier_name +
                                '/' + archive_name + '/' + dataset_name + '/history.csv')
    # read the history data frame for augmentation
    df_history_tf = pd.read_csv(root_dir + 'results/resnet_augment/Viz-for-aaltd-18/' +
                                dataset_name + '/history.csv')

    # max_epoch = df_history_tf.shape[0]
    max_epoch = 1400
    smoothness = 300
    plt.figure()
    plt.title('title here plz', fontsize='x-large')
    plt.ylabel('model\'s ' + metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.ylim(ymax=5)
    # plot orignal train
    y = df_history_or[metric].iloc[0:max_epoch]
    x = y.keys()
    tenth = int(len(x) / 1)
    if tenth % 2 == 1:
        tenth = tenth + 1
    w = tenth + 1
    y = y.rolling(window=w, center=False, min_periods=1).mean()
    # linear interpolate to smooth
    x_new = np.linspace(x.min(), x.max(), smoothness)
    y_new = spline(x, y, x_new)
    y = pd.Series(data=y_new, index=x_new)
    plt.plot(y, label='train_original',
             color=(255 / 255, 160 / 255, 14 / 255))  ## original train
    # plot orignal test
    y = df_history_or['val_' + metric].iloc[0:max_epoch]
    x = y.keys()
    tenth = int(len(x) / 1)
    if tenth % 2 == 1:
        tenth = tenth + 1
    w = tenth + 1
    y = y.rolling(window=w, center=False, min_periods=1).mean()
    # linear interpolate to smooth
    x_new = np.linspace(x.min(), x.max(), smoothness)
    y_new = spline(x, y, x_new)
    y = pd.Series(data=y_new, index=x_new)
    plt.plot(y, label='test_original',
             color=(210 / 255, 0 / 255, 0 / 255))  ## original test
    # plot transfer train
    y = df_history_tf[metric].iloc[0:max_epoch]
    x = y.keys()
    tenth = int(len(x) / 1)
    if tenth % 2 == 1:
        tenth = tenth + 1
    w = tenth + 1
    y = y.rolling(window=w, center=False, min_periods=1).mean()
    # linear interpolate to smooth
    x_new = np.linspace(x.min(), x.max(), smoothness)
    y_new = spline(x, y, x_new)
    y = pd.Series(data=y_new, index=x_new)
    plt.plot(y, label='train_data_augment',
             color=(181 / 255, 87 / 255, 181 / 255))  # transfer train
    # plot transfer test
    y = df_history_tf['val_' + metric].iloc[0:max_epoch]
    x = y.keys()
    tenth = int(len(x) / 1)
    if tenth % 2 == 1:
        tenth = tenth + 1
    w = tenth + 1
    y = y.rolling(window=w, center=False, min_periods=1).mean()
    # linear interpolate to smooth
    x_new = np.linspace(x.min(), x.max(), smoothness)
    y_new = spline(x, y, x_new)
    y = pd.Series(data=y_new, index=x_new)
    plt.plot(y, label='test_data_augment',
             color=(27 / 255, 32 / 255, 101 / 255))  # transfer test
    plt.legend(loc='best')

    plt.savefig(root_dir + '/' +
                dataset_name + 'data-augmentation.pdf', bbox_inches='tight')
    plt.close()


def compare_results_ucr_2018():
    root_dir = '/b/home/uha/hfawaz-datas/dba-python/'

    df_ori = pd.read_csv(root_dir + 'results-ucr-18-ori-after-correction-anh.csv', delimiter=',', index_col=0)

    df_ours = pd.read_csv(root_dir + 'results-ucr-18-running-anh-pre-processing.csv')

    df_ours = df_ours.round(4)
    df_ori = df_ori.round(4)

    df_ori.sort_values(['Name'], inplace=True)
    df_ours.sort_values(['dataset_name'], inplace=True)

    dtw_ours = df_ours.loc[df_ours['dist_name'] == 'dtw']['error']

    dtw_ori = df_ori['DTW (w=100)']

    print('DTW')

    dtw_diff = np.abs(dtw_ori.values - dtw_ours.values)

    idx_diff_ori = dtw_ori.index[dtw_diff > 0.0]

    print('########### -- DTW -- #############')
    for idx_ori in idx_diff_ori:
        dataset_name = df_ori.loc[idx_ori]['Name']
        print('dataset_name', dataset_name)
        print('Your error', df_ori.loc[idx_ori]['DTW (w=100)'])
        print('Our error', df_ours.loc[(df_ours['dataset_name'] == dataset_name) &
                                       (df_ours['dist_name'] == 'dtw')]['error'].values[0])
        print('######################################')

    print(np.sum(np.abs(dtw_ours.values - dtw_ori.values)))

    ed_ours = df_ours.loc[df_ours['dist_name'] == 'ed']['error']

    ed_ori = df_ori['ED (w=0)']

    print('ED')

    ed_diff = np.abs(ed_ori.values - ed_ours.values)

    idx_diff_ori = ed_ori.index[ed_diff > 0.0]

    print('########### -- ED -- #############')
    for idx_ori in idx_diff_ori:
        dataset_name = df_ori.loc[idx_ori]['Name']
        print('dataset_name', dataset_name)
        print('Your error', df_ori.loc[idx_ori]['ED (w=0)'])
        print('Our error', df_ours.loc[(df_ours['dataset_name'] == dataset_name) &
                                       (df_ours['dist_name'] == 'ed')]['error'].values[0])
        print('######################################')

    # print(np.sum(np.abs(ed_ours.values - ed_ori.values))


def compare_shapeDBA_with_without_warping(root_dir, reload=False):
    fname = 'results-data-cluster.csv'
    if reload:
        res_df, not_completed = generate_results_csv_data_cluster(fname, root_dir,
                                                                  DATA_CLUSTERING_ALGORITHMS)
    else:
        not_completed = {'ElectricDevices', 'FordA', 'UWaveGestureLibraryAll', 'FordB', 'StarLightCurves', 'yoga',
                         'MALLAT', 'CinC_ECG_torso', 'HandOutlines'}
        res_df = pd.read_csv(fname)
    for not_completed_dname in not_completed:
        res_df.drop(res_df.loc[res_df['dataset_name'] == not_completed_dname].index, axis=0,
                    inplace=True)

    from utils.constants import UCR_names_old_to_new_2018
    old_dataset_names = pd.unique(res_df.loc[res_df['archive_name'] == 'TSC']['dataset_name'])
    for odn in old_dataset_names:
        res_df.loc[(res_df['archive_name'] == 'TSC') & (res_df['dataset_name'] == odn), 'dataset_name'] = \
            UCR_names_old_to_new_2018[odn]

    res_pairwise = {ARCHIVE_NAMES[0]: [], ARCHIVE_NAMES[1]: []}
    dnames = np.unique(res_df['dataset_name'])

    ddnames = []
    for dn in dnames:
        cur_df = res_df.loc[res_df['dataset_name'] == dn]
        if len(cur_df) == 2:
            ddnames.append(dn)
            for idx, row in cur_df.iterrows():
                res_pairwise[row['archive_name']].append(row['ari'])

    x = np.arange(start=-0.1, stop=1.02, step=0.01)
    plt.xlim(xmax=1.02, xmin=-0.1)
    plt.ylim(ymax=1.02, ymin=-0.1)

    plt.scatter(x=res_pairwise[ARCHIVE_NAMES[0]], y=res_pairwise[ARCHIVE_NAMES[1]])

    # annotate with dataset name
    for i in range(len(ddnames)):
        label = ddnames[i]
        if np.abs(res_pairwise[ARCHIVE_NAMES[0]][i] - res_pairwise[ARCHIVE_NAMES[1]][i]) >= 0.2:
            plt.annotate(label, (res_pairwise[ARCHIVE_NAMES[0]][i]
                                 , res_pairwise[ARCHIVE_NAMES[1]][i]))

    classifier_name_1 = 'with_warping'
    classifier_name_2 = 'without_warping'

    plt.xlabel(classifier_name_1, fontsize='large')
    plt.ylabel(classifier_name_2, fontsize='large')
    plt.plot(x, x, color='black')

    print('Wins are for', classifier_name_2)
    clf1 = np.array(res_pairwise[ARCHIVE_NAMES[0]])
    clf2 = np.array(res_pairwise[ARCHIVE_NAMES[1]])

    uniq, counts = np.unique(clf1 < clf2, return_counts=True)
    print('Wins', counts[-1])

    uniq, counts = np.unique(clf1 == clf2, return_counts=True)
    print('Draws', counts[-1])

    uniq, counts = np.unique(clf1 > clf2, return_counts=True)
    print('Losses', counts[-1])

    p_value = wilcoxon(clf1, clf2, zero_method='pratt')[1]
    print(p_value)

    plt.savefig('shapedba_' + classifier_name_1 + '-' + classifier_name_2 + '.pdf'
                , bbox_inches='tight')


def get_best_iteration(cur_dir, x, x_pad, distance_algorithm):
    import utils
    best_itr = -1
    best_ss = np.inf

    # get the distance method (check constants.py)
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get dsitance method parameters
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]

    if distance_algorithm == 'shapedtw':
        utils.utils.add_dist_fun_params_for_shapedtw(dist_fun_params, x[0].shape[0])

    for itr in range(NB_ITERATIONS):
        itr_ = 'itr_' + str(itr)
        clusters_file_path = cur_dir + itr_ + '/kmeans_clusters_9.npy'
        affects_file_path = cur_dir + itr_ + '/kmeans_affect_9.npy'
        if not check_if_file_exits(clusters_file_path) or not check_if_file_exits(affects_file_path):
            return -1
        cur_ss = 0
        clusters = np.load(clusters_file_path)
        affects = np.load(affects_file_path)

        for idx, c_idx in enumerate(affects):
            avg_4_dist = clusters[c_idx]
            series_4_dist = x[idx]
            if 'shape_reach' in dist_fun_params:
                reach = dist_fun_params['shape_reach']
                avg_4_dist = np.pad(avg_4_dist, (reach, reach), mode='edge')
                series_4_dist = x_pad[idx]

            cur_ss += dist_fun(avg_4_dist, series_4_dist, **dist_fun_params)[0] ** 2

        if best_ss > cur_ss:
            best_itr = itr
            best_ss = cur_ss

    return best_itr


def get_distance_name_from_clustering_algorithm(algorithm_name):
    import utils
    for distance_algorithm in utils.constants.DISTANCE_ALGORITHMS:
        if distance_algorithm in algorithm_name:
            return distance_algorithm
    return None


def plot_pairwise_best_clustering(root_dir, classifier_name_1, classifier_name_2):
    import utils
    archive_name = 'TSC'
    array_algorithm_names = [classifier_name_1, classifier_name_2]
    root_dir_dataset_archive = root_dir + '../dl-tsc/archives/'

    datasets_dict = read_all_datasets(root_dir_dataset_archive, 'TSC')

    res_df = pd.DataFrame(data=np.zeros((0, 5), dtype=np.float64), index=[],
                          columns=['algorithm_name', 'archive_name', 'dataset_name',
                                   'ari', 'duration'])
    not_completed = set()

    output_file_name = 'results-with-best-ss.csv'

    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
        print(dataset_name)
        # old_dataset_name = utils.constants.UCR_names_old_to_new_2018[dataset_name]
        old_dataset_name = dataset_name
        x_train = datasets_dict[old_dataset_name][0]
        x_test = datasets_dict[old_dataset_name][2]

        x = np.concatenate((x_train, x_test), axis=0)

        # this padding is used for shapedtw
        reach = utils.constants.DISTANCE_ALGORITHMS_PARAMS['shapedtw']['shape_reach']
        x_pad = np.pad(np.asarray(x), ((0, 0), (reach, reach)), mode='edge')

        for algorithm_name in array_algorithm_names:
            distance_algorithm = get_distance_name_from_clustering_algorithm(algorithm_name)
            print('\t' + algorithm_name)
            cur_dir = root_dir + '/results/' + archive_name + \
                      '/' + dataset_name + '/' + algorithm_name + '/'

            # get the best iteration
            best_itr = get_best_iteration(cur_dir, x, x_pad, distance_algorithm)

            if best_itr == -1:
                not_completed.add(dataset_name)
                continue

            temp_dir = cur_dir + 'itr_' + str(best_itr) + '/df_metrics.csv'
            df_metrics_itr = pd.read_csv(temp_dir, index_col=0)
            df_metrics_itr['algorithm_name'] = algorithm_name
            df_metrics_itr['archive_name'] = archive_name
            df_metrics_itr['dataset_name'] = dataset_name
            res_df = pd.concat((res_df, df_metrics_itr), axis=0, sort=False)

    res_df.reset_index(inplace=True)
    res_df.drop('index', axis=1, inplace=True)
    res_df.to_csv(root_dir + output_file_name, index=False)

    plot_pairwise(root_dir, classifier_name_1, classifier_name_2,
                  res_df=res_df, not_completed=not_completed)
