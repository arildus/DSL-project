import random
import numpy as np
import pandas as pd

def get_column_names(features=[], indexes=[]):
    """
    Get the cartetian proudcts of features and indexes. For dropping columns in the dataframes.
    """
    column_names = []

    for index in indexes:
        for feature in features:
            column_names.append(f"{feature}[{index}]")
    
    return column_names

def mean_euclid_dist(y_true, y_pred):
    """
    Compute the mean euclidean distance between two series of 2D vectors.
    """
    diff = y_true - y_pred
    sqrd = diff**2
    summed = sqrd.sum(axis=1)
    euclid_dist = np.sqrt(summed)
    n = euclid_dist.shape[0]
    sigma = euclid_dist.sum(axis=0)

    mean_euclid_dist = sigma / n
    
    return mean_euclid_dist

def insert_zeros(X_train, acc_idxs, threshold):

    pd.set_option('mode.chained_assignment', None)

    df = X_train.copy()

    y_train_valid = df[['x', 'y']].copy()

    X_train_valid = df.drop(columns=['x', 'y'])

    df_pmax = df[[f'pmax[{i}]' for i in acc_idxs]]
    df_negpmax = df[[f'negpmax[{i}]' for i in acc_idxs]]
    df_area = df[[f'area[{i}]' for i in acc_idxs]]

    mask = df_pmax < threshold
    df_pmax[mask] = 0

    mask.columns = get_column_names(['negpmax'], acc_idxs)
    df_negpmax[mask] = 0

    mask.columns = get_column_names(['area'], acc_idxs)
    df_area[mask] = 0

    X_train_valid = pd.concat([df_pmax, df_negpmax, df_area], axis=1)

    return X_train_valid, y_train_valid
