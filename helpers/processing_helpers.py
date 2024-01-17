import numpy as np

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