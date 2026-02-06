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
        # error handling for matrix input
        if not isinstance(mat, np.ndarray): # check if mat is a numpy array
            raise TypeError(f"mat must be a numpy array, got {type(mat).__name__}")
        if mat.ndim != 2: # check if mat is 2D
            raise ValueError(f"mat must have 2 dimensions, got {mat.ndim} dimensions")
        if mat.shape[0] == 0: # check if mat has zero samples
            raise ValueError("mat is empty, got 0 samples")
        if mat.shape[0] < self.k: # check if mat has fewer samples than k
            raise ValueError(f"mat has fewer samples ({mat.shape[0]}), but k={self.k}. need at least {self.k} samples")
        
        self.n_features = mat.shape[1] # store num of features
        n = mat.shape[0] # num of samples
        if self.k > n: # check if k is greater than num of samples
            raise ValueError(f"k={self.k}, cannot be greater than number of samples ({n})")
        
        # initialize centroids randomly by selecting k random samples from the data
        np.random.seed(42)  # set seed for reproducibility
        indices = np.random.choice(n, size=self.k, replace=False)
        self.centroids = mat[indices].copy() # use to avoid modifying original data
        self.error = None  # initialize error
        
        # run kmeans algorithm
        for _ in range(self.max_iter):

            distances = cdist(mat, self.centroids) # calc dist from each point to each centroid
            labels = np.argmin(distances, axis=1) # assign each point to nearest centroid
            
            # current error = sum of squared distances to nearest centroid / number of samples
            current_error = np.sum(np.min(distances, axis=1) ** 2) / mat.shape[0]
            
            # convergence check: if error change is less than tol and error is not None, then break loop
            if self.error is not None and abs(current_error - self.error) < self.tol:
                break
            
            # new centroid = mean of the assigned points
            new_centroids = np.array([ # if no points get assigned to a centroid, just keep the old centroid
                mat[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
                for i in range(self.k) # loop through each centroid index and compute new centroid location
            ])
            
            # update with new centroids and error
            self.centroids = new_centroids
            self.error = current_error

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
        # error handling for matrix input
        if not isinstance(mat, np.ndarray): # check if mat is a numpy array
            raise TypeError(f"mat must be a numpy array, got {type(mat).__name__}")
        if mat.ndim != 2: # check if mat is 2D
            raise ValueError(f"mat must have 2 dimensions, got {mat.ndim} dimensions")
        if mat.shape[0] == 0: # check if mat has zero samples
            raise ValueError("mat is empty, got 0 samples")
        if mat.shape[0] < self.k: # check if mat has fewer samples than k
            raise ValueError(f"mat has fewer samples ({mat.shape[0]}), but k={self.k}. need at least {self.k} samples")
        # check if fit was called before predict
        if self.centroids is None:
            raise ValueError("must call .fit() before .predict()")
        
        distances = cdist(mat, self.centroids) # calc dist from each point to each centroid
        labels = np.argmin(distances, axis=1) # assign each point to nearest centroid
        
        return labels


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        # if error is None, then fit was not called yet, so raise an error
        if self.error is None:
            raise ValueError("must call .fit() before getting error")
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        # if centroids is None, then fit was not called yet, so raise an error
        if self.centroids is None:
            raise ValueError("must call .fit() before getting centroids")
        return self.centroids
