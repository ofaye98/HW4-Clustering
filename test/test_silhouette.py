# write your silhouette score unit tests here
import numpy as np
import pytest
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_samples

def test_silhouette_score():
    # create a simple dataset with 2 clusters
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    y = np.array([0, 0, 0, 1, 1, 1]) # cluster labels for each observation
    
    silhouette = Silhouette() # initialize silhouette object
    scores = silhouette.score(X, y) # compute silhouette scores for each observation
    
    # check that silhouette scores are between -1 and 1
    assert np.all(scores >= -1) and np.all(scores <= 1), "silhouette scores should be between -1 and 1"
    
    # check that silhouette scores for points in the same cluster are higher than points in different clusters
    assert scores[0] > scores[3], "point in cluster 0 should have higher silhouette score than point in cluster 1"
    assert scores[3] > scores[0], "point in cluster 1 should have higher silhouette score than point in cluster 0"

    # make sure score method raises errors for invalid inputs
    with pytest.raises(TypeError):
        silhouette.score(X.tolist(), y) # X is not a numpy array
    with pytest.raises(TypeError):
        silhouette.score(X, y.tolist()) # y is not a numpy array
    with pytest.raises(ValueError):
        silhouette.score(X.reshape(-1), y) # X is not 2D
    with pytest.raises(ValueError):
        silhouette.score(X, y.reshape(-1, 1)) # y is not 1D
    with pytest.raises(ValueError):
        silhouette.score(X, y[:-1]) # X and y have different number of rows/elements   

def test_silhouette_matches_sklearn():    
    # create a simple dataset with 3 clusters
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0],
                  [8, 2], [8, 4], [8, 0]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) # cluster labels for each observation
    
    silhouette = Silhouette() # initialize silhouette object
    scores = silhouette.score(X, y) # compute silhouette scores using my implementation
    
    sklearn_scores = silhouette_samples(X, y) # compute silhouette scores using sklearn's implementation
    
    # check that my silhouette scores match sklearn's silhouette scores
    assert np.allclose(scores, sklearn_scores), f"Expected silhouette scores to match sklearn's silhouette scores, got {scores} and {sklearn_scores}"