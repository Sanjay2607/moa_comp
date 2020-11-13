import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import  KMeans
from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.filterwarnings('ignore')

# rankGauss
def rankgauss(train, test, col):
    transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
    train[col] = transformer.fit_transform(train[col].values)
    test [col] = transformer.transform    (test [col].values)
    return train, test

col =  GENES + CELLS

# PCA
def pca(train, test, col, n_comp, prefix):
    data = pd.concat([pd.DataFrame(train[col]), pd.DataFrame(test[col])])
    data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data))

    train2 = data2[:train.shape[0]] 
    test2 = data2[-test.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_{prefix}-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_{prefix}-{i}' for i in range(n_comp)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train = pd.concat((train, train2), axis=1)
    test = pd.concat((test, test2), axis=1)
    return train, test

train_features, test_features = pca(train_features, test_features, GENES, 600, 'G')
train_features, test_features = pca(train_features, test_features, CELLS,  50, 'C')
train_features.shape, test_features.shape

def sanity_check(): return train_features.shape, test_features.shape


# thresholdValue = 0.8
def VarianceThresholdOperation(train, test, thresholdValue):
    var_thresh = VarianceThreshold(thresholdValue)  
    data = train.append(test)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[ : train.shape[0]]
    test_features_transformed = data_transformed[-test.shape[0] : ]


    train = pd.DataFrame(train[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])

    train = pd.concat([train, pd.DataFrame(train_features_transformed)], axis=1)


    test = pd.DataFrame(test[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                columns=['sig_id','cp_type','cp_time','cp_dose'])

    test = pd.concat([test, pd.DataFrame(test_features_transformed)], axis=1)

    return train , test

def fe_cluster(train, test, n_clusters_g = 35, n_clusters_c = 5, SEED = 123):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test


    def fe_stats(train, test):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return train, test

    def process(train, test):
    GENES = [col for col in train.columns if col.startswith('g-')]
    CELLS = [col for col in train.columns if col.startswith('c-')]
    
    # normalize data using RankGauss
    train, test = rankgauss(train, test, GENES + CELLS)
    
    # get PCA components
    train, test = pca(train, test, GENES, 600, 'G') # GENES
    train, test = gpca(train, test, CELLS, 50 , 'C') # CELLS
    
    # select features using variance thresholding
    train, test = VarianceThresholdOperation(train, test)
    
    # feature engineering
    train, test = fe_cluster(train, test)
    train, test = fe_stats  (train, test)
    return train, test