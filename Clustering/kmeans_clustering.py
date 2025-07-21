"""
K-means clustering implementation and core clustering algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def perform_kmeans_clustering(feature_matrix: pd.DataFrame, n_clusters: int = 4) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-means clustering on driver feature matrix.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Driver feature matrix
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    Tuple[np.ndarray, KMeans]
        Cluster labels and fitted KMeans model
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    logger.info(f"K-means clustering completed with {n_clusters} clusters")
    logger.info(f"Cluster sizes: {np.bincount(cluster_labels)}")
    
    return cluster_labels, kmeans