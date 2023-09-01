from kshape import KShape_save_itr
from kmeans import kmeans_save_itr
from averages.dba import dba

from distances.dtw.dtw import dynamic_time_warping as dtw
from distances.shapeDTWefficient.shapeDTWefficient import shape_DTW

from distances.descriptors import identity
from distances.descriptors import paa
from distances.descriptors import slope
from distances.descriptors import derivative

from distances.euclidean import euclidean_distance as ed
from distances.softdtw import softdtw
from averages.softdba import softdba
from averages.mean import mean
from averages.shapedba import shapedba

UNIVARIATE_DATASET_NAMES_2018 = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',
                                 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown',
                                 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX',
                                 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction',
                                 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                                 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200',
                                 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
                                 'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
                                 'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
                                 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
                                 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
                                 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
                                 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
                                 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
                                 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                                 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                                 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                                 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
                                 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
                                 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
                                 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
                                 'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
                                 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                                 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2',
                                 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll',
                                 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1',
                                 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
                                 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                                 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll',
                                 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
                                 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

UNIVARIATE_DATASET_NAMES_2018 = ['SmoothSubspace']

UNIVARIATE_ARCHIVE_NAMES = ['UCRArchive_2018']

dataset_names_for_archive = {'UCRArchive_2018': UNIVARIATE_DATASET_NAMES_2018}

CLUSTERING_ALGORITHMS = {'kmeans': kmeans_save_itr, 'kshape': KShape_save_itr}

AVERAGING_ALGORITHMS = {'dba': dba, 'shapedba': shapedba, 'softdba': softdba, 'mean': mean}

DISTANCE_ALGORITHMS = {'dtw': dtw, 'shapedtw': shape_DTW,
                       'ed': ed, 'softdtw': softdtw}

SOFTDTW_PARAMS = {'gamma': 0.01}
DTW_PARAMS = {'w': 1.0}  # warping window should be given in percentage (negative means no warping window)
SHAPEDTW_PARAMS = {'shape_reach': 15, 'w': 1.0}
ED_PARAMS = {'w': None}  # these are the weights, ignore
DISTANCE_ALGORITHMS_PARAMS = {'dtw': DTW_PARAMS, 'oldshapedtw': SHAPEDTW_PARAMS, 'ed': ED_PARAMS,
                              'shapedtw': SHAPEDTW_PARAMS, 'softdtw': SOFTDTW_PARAMS}

DATA_CLUSTERING_ALGORITHMS = ['kmeans_shapedba_shapedtw', 'kmeans_softdba_softdtw',
                              'kmeans_dba_dtw', 'kmeans_mean_ed', 'kshape']

SHAPE_DTW_F = {'identity': identity, 'paa': paa, 'slope': slope, 'derivative': derivative}

NB_ITERATIONS = 5  # how many iterations to eliminate the random bias

MAX_PROTOTYPES_PER_CLASS = 5