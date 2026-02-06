# Write your k-means unit tests here
import numpy as np
import pytest
from cluster.kmeans import KMeans

def test_kmeans_init():
    # test valid initialization
    kmeans = KMeans(k=3, tol=1e-5, max_iter=200)
    assert kmeans.k == 3, f"Expected k to be 3, got {kmeans.k}"
    assert kmeans.tol == 1e-5, f"Expected tol to be 1e-5, got {kmeans.tol}"
    assert kmeans.max_iter == 200, f"Expected max_iter to be 200, got {kmeans.max_iter}"
    
    # test invalid k inputs
    with pytest.raises(TypeError):
        KMeans(k='3') # k is not an integer
    with pytest.raises(ValueError):
        KMeans(k=0) # k is less than 1
    with pytest.raises(ValueError):
        KMeans(k=-1) # k is less than zero
    
    # test invalid tol inputs
    with pytest.raises(TypeError):
        KMeans(k=3, tol='1e-5') # tol is not a float
    with pytest.raises(ValueError):
        KMeans(k=3, tol=-1) # tol is less than zero
    
    # test invalid max_iter inputs
    with pytest.raises(TypeError):
        KMeans(k=3, max_iter='100') # max_iter is not an integer
    with pytest.raises(ValueError):
        KMeans(k=3, max_iter=0) # max_iter is less than 1
    with pytest.raises(ValueError):
        KMeans(k=3, max_iter=-1) # max_iter is less than zero

def test_kmeans_fit():
    # create a simple dataset with 2 clusters
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    
    # fit kmeans with k=2
    kmeans = KMeans(k=2)
    kmeans.fit(X)
    
    # get centroids and error
    centroids = kmeans.get_centroids()
    error = kmeans.get_error()
    
    # check that centroids are close to the true cluster centers
    assert np.allclose(centroids[0], [1, 2], atol=1e-6) or np.allclose(centroids[0], [4, 2], atol=1e-6), f"Expected one centroid to be close to [1, 2] and the other to be close to [4, 2], got {centroids}"
    assert np.allclose(centroids[1], [1, 2], atol=1e-6) or np.allclose(centroids[1], [4, 2], atol=1e-6), f"Expected one centroid to be close to [1, 2] and the other to be close to [4, 2], got {centroids}"
    
    # check that error is non-negative
    assert error >= 0, f"Expected error to be non-negative, got {error}"

def test_kmeans_predict():
    # create a simple dataset with 2 clusters
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    
    # fit kmeans with k=2
    kmeans = KMeans(k=2)
    kmeans.fit(X)
    
    # predict cluster labels for the same dataset
    predictions = kmeans.predict(X)
    
    # check that predictions are integers in range [0, k-1]
    assert np.all((predictions >= 0) & (predictions < 2)), f"Expected predictions to be integers in range [0, 1], got {predictions}"

def test_kmeans_get_error():
    kmeans = KMeans(k=2)
    with pytest.raises(ValueError):
        kmeans.get_error() # should raise error since fit has not been called yet

def test_kmeans_get_centroids():
    kmeans = KMeans(k=2)
    with pytest.raises(ValueError):
        kmeans.get_centroids() # should raise error since fit has not been called yet