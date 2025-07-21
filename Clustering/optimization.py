"""
Clustering optimization functions for finding optimal parameters.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_optimal_clusters(feature_matrix: pd.DataFrame, max_clusters: int = 8) -> int:
    """
    Find optimal number of clusters using elbow method.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Driver feature matrix
    max_clusters : int
        Maximum number of clusters to test
        
    Returns:
    --------
    int
        Optimal number of clusters
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    inertias = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection (could be improved)
    # Calculate second derivative to find elbow
    if len(inertias) >= 3:
        second_derivative = np.diff(inertias, 2)
        optimal_k = K_range[np.argmax(second_derivative) + 1]
    else:
        optimal_k = 3  # Default fallback
    
    logger.info(f"Optimal number of clusters: {optimal_k}")
    
    return optimal_k