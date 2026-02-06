import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # error handling for inputs
        if not isinstance(X, np.ndarray): # check if X is a numpy array
            raise TypeError(f"X must be a numpy array, got {type(X).__name__}")
        if not isinstance(y, np.ndarray): # check if y is a numpy array
            raise TypeError(f"y must be a numpy array, got {type(y).__name__}")
        if X.ndim != 2: # check if X is 2D
            raise ValueError(f"X must have 2 dimensions, got {X.ndim} dimensions")
        if y.ndim != 1: # check if y is 1D
            raise ValueError(f"y must have 1 dimension, got {y.ndim} dimensions")
        if X.shape[0] != y.shape[0]: # check if number of rows in X matches number of elements in y
            raise ValueError(f"X and y must have the same number of rows/elements. Got shapes {X.shape} and {y.shape}")

        n = X.shape[0] # num of samples
        k = len(np.unique(y)) # num of clusters

        silhouette_scores = np.zeros(n) # initialize array to store silhouette scores

        # compute silhouette score for each sample
        for i in range(n):
            cluster_label = y[i] # get cluster for current observation
            cluster_points = X[y == cluster_label] # find all points in the same cluster as current observation
            
            distances_within_cluster = cdist([X[i]], cluster_points)[0] # compute distances from current observation to all other points in the same cluster
            a = np.mean(distances_within_cluster) # compute average distance within cluster (a)

            # compute average distance to nearest other cluster (b)
            b_min = float('inf') # initialize b_min to infinity so that any computed b will be smaller
            for j in range(k):
                if j != cluster_label: # only compute b for other clusters, not the same cluster
                    other_cluster_points = X[y == j] # find all points in the other cluster
                    distances_to_other_cluster = cdist([X[i]], other_cluster_points)[0] # compute distances from current observation to all points in the other cluster
                    b_mean = np.mean(distances_to_other_cluster) # compute average distance to the other cluster
                    b_min = min(b_min, b_mean) # update b_min if this other cluster is closer than previous closest cluster

            # calc silhouette score for observation i
            if a == b_min and a == 0: # if a and b are both zero
                silhouette_scores[i] = 0.0 # then silhouette score is set to 0, since the observation is perfectly clustered
            else: # otherwise, calc silhouette score = (b - a) / max(a, b)
                silhouette_scores[i] = (b_min - a) / max(a, b_min) if max(a, b_min) > 0 else 0.0

        return silhouette_scores