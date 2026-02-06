import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # error handling for k inputs
        if not isinstance(k, int): # check if k is an integer
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k <= 0: # check if k is zero or negative
            raise ValueError(f"k must be >= 1, got {k}")
        
        # error handling for tol inputs
        if not isinstance(tol, (float, int)): # check if tol is a float or int
            raise TypeError(f"tol must be a float, got {type(tol).__name__}")
        if tol < 0: # check if tol is negative
            raise ValueError("tol must be >= 0")
        
        # error handling for max_iter inputs
        if not isinstance(max_iter, int): # check if max_iter is an integer
            raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}")
        if max_iter <= 0: # check if max_iter is zero or negative
            raise ValueError("max_iter must be >= 1")
        
        # initialize attributes
        self.k = k
        self.tol = float(tol)
        self.max_iter = max_iter
        self.centroids = None
        self.error = None
        self.n_features = None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
