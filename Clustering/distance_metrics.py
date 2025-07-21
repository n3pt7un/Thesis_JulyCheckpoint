"""
Distance calculation functions for driver similarity analysis.

This module provides various distance metrics to measure similarity
between drivers based on their behavioral feature vectors.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from scipy.spatial.distance import pdist, squareform
import logging
from typing import Union, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_distance_matrix(feature_matrix: pd.DataFrame, method: str = 'euclidean') -> pd.DataFrame:
    """
    Calculate distance matrix between drivers based on their feature vectors.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Driver feature matrix with drivers/samples as rows and features as columns
    method : str, default='euclidean'
        Distance calculation method:
        - 'euclidean': Euclidean distance
        - 'manhattan': Manhattan (L1) distance  
        - 'cosine': Cosine distance
        - 'correlation': Correlation distance
        - 'chebyshev': Chebyshev distance
        
    Returns:
    --------
    pd.DataFrame
        Distance matrix with driver names as indices and columns
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # Calculate distance matrix based on method
    if method == 'euclidean':
        distances = euclidean_distances(scaled_features)
    elif method == 'manhattan':
        distances = manhattan_distances(scaled_features)
    elif method == 'cosine':
        distances = cosine_distances(scaled_features)
    elif method in ['correlation', 'chebyshev']:
        # Use scipy for additional distance metrics
        distances = squareform(pdist(scaled_features, metric=method))
    else:
        raise ValueError(f"Unsupported distance method: {method}. "
                        f"Choose from: 'euclidean', 'manhattan', 'cosine', 'correlation', 'chebyshev'")
    
    # Create DataFrame with driver names as indices
    distance_df = pd.DataFrame(
        distances,
        index=feature_matrix.index,
        columns=feature_matrix.index
    )
    
    logger.info(f"Created {method} distance matrix for {len(distance_df)} drivers/samples")
    
    return distance_df


def find_similar_drivers(distance_matrix: pd.DataFrame, 
                        target_driver: str, 
                        n_similar: int = 5,
                        exclude_self: bool = True) -> pd.Series:
    """
    Find the most similar drivers to a target driver based on distance matrix.
    
    Parameters:
    -----------
    distance_matrix : pd.DataFrame
        Distance matrix between drivers
    target_driver : str
        Name of the target driver to find similarities for
    n_similar : int, default=5
        Number of similar drivers to return
    exclude_self : bool, default=True
        Whether to exclude the target driver from results
        
    Returns:
    --------
    pd.Series
        Series with similar drivers and their distances, sorted by similarity
    """
    if target_driver not in distance_matrix.index:
        raise ValueError(f"Target driver '{target_driver}' not found in distance matrix")
    
    similarities = distance_matrix.loc[target_driver].copy()
    
    if exclude_self:
        similarities = similarities.drop(target_driver, errors='ignore')
    
    # Sort by distance (ascending - closer means more similar)
    most_similar = similarities.sort_values().head(n_similar)
    
    logger.info(f"Found {len(most_similar)} most similar drivers to {target_driver}")
    
    return most_similar


def calculate_driver_centrality(distance_matrix: pd.DataFrame) -> pd.Series:
    """
    Calculate centrality scores for each driver based on average distance to all others.
    Lower scores indicate more central/typical drivers.
    
    Parameters:
    -----------
    distance_matrix : pd.DataFrame
        Distance matrix between drivers
        
    Returns:
    --------
    pd.Series
        Centrality scores for each driver (lower = more central)
    """
    # Calculate mean distance to all other drivers
    centrality_scores = distance_matrix.mean(axis=1)
    
    # Sort by centrality (ascending - lower means more central)
    centrality_scores = centrality_scores.sort_values()
    
    logger.info(f"Calculated centrality scores for {len(centrality_scores)} drivers")
    
    return centrality_scores


def calculate_cluster_cohesion(distance_matrix: pd.DataFrame, 
                              cluster_labels: Union[np.ndarray, List],
                              cluster_id: int = None) -> Union[float, pd.Series]:
    """
    Calculate the cohesion (internal similarity) of clusters.
    
    Parameters:
    -----------
    distance_matrix : pd.DataFrame
        Distance matrix between drivers
    cluster_labels : array-like
        Cluster assignments for each driver
    cluster_id : int, optional
        Specific cluster ID to calculate cohesion for. If None, calculates for all clusters.
        
    Returns:
    --------
    float or pd.Series
        Average intra-cluster distance for specified cluster or all clusters
    """
    cluster_labels = np.array(cluster_labels)
    unique_clusters = np.unique(cluster_labels)
    
    if cluster_id is not None:
        if cluster_id not in unique_clusters:
            raise ValueError(f"Cluster {cluster_id} not found in cluster labels")
        unique_clusters = [cluster_id]
    
    cohesion_scores = {}
    
    for cluster in unique_clusters:
        # Get indices of drivers in this cluster
        cluster_mask = cluster_labels == cluster
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) < 2:
            cohesion_scores[cluster] = 0.0
            continue
        
        # Extract sub-matrix for this cluster
        cluster_drivers = distance_matrix.index[cluster_indices]
        cluster_distances = distance_matrix.loc[cluster_drivers, cluster_drivers]
        
        # Calculate average distance within cluster (excluding diagonal)
        upper_triangle = np.triu(cluster_distances.values, k=1)
        non_zero_distances = upper_triangle[upper_triangle > 0]
        
        if len(non_zero_distances) > 0:
            cohesion_scores[cluster] = np.mean(non_zero_distances)
        else:
            cohesion_scores[cluster] = 0.0
    
    logger.info(f"Calculated cohesion scores for {len(cohesion_scores)} clusters")
    
    if cluster_id is not None:
        return cohesion_scores[cluster_id]
    else:
        return pd.Series(cohesion_scores, name='cluster_cohesion')


def calculate_silhouette_coefficient(distance_matrix: pd.DataFrame, 
                                   cluster_labels: Union[np.ndarray, List]) -> pd.Series:
    """
    Calculate silhouette coefficient for each driver based on distance matrix.
    
    Parameters:
    -----------
    distance_matrix : pd.DataFrame
        Distance matrix between drivers
    cluster_labels : array-like
        Cluster assignments for each driver
        
    Returns:
    --------
    pd.Series
        Silhouette coefficients for each driver
    """
    cluster_labels = np.array(cluster_labels)
    n_samples = len(cluster_labels)
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        driver_cluster = cluster_labels[i]
        
        # Calculate a(i): mean distance to other points in same cluster
        same_cluster_mask = (cluster_labels == driver_cluster) & (np.arange(n_samples) != i)
        if np.sum(same_cluster_mask) > 0:
            a_i = distance_matrix.iloc[i, same_cluster_mask].mean()
        else:
            a_i = 0
        
        # Calculate b(i): mean distance to nearest cluster
        other_clusters = np.unique(cluster_labels[cluster_labels != driver_cluster])
        if len(other_clusters) > 0:
            b_values = []
            for other_cluster in other_clusters:
                other_cluster_mask = cluster_labels == other_cluster
                b_cluster = distance_matrix.iloc[i, other_cluster_mask].mean()
                b_values.append(b_cluster)
            b_i = min(b_values)
        else:
            b_i = 0
        
        # Calculate silhouette coefficient
        if max(a_i, b_i) > 0:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_scores[i] = 0
    
    logger.info(f"Calculated silhouette coefficients for {n_samples} drivers")
    
    return pd.Series(silhouette_scores, index=distance_matrix.index, name='silhouette_score')


def compare_distance_methods(feature_matrix: pd.DataFrame, 
                           methods: List[str] = None) -> dict:
    """
    Compare different distance calculation methods on the same feature matrix.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Driver feature matrix
    methods : List[str], optional
        List of distance methods to compare. If None, uses common methods.
        
    Returns:
    --------
    dict
        Dictionary with method names as keys and distance matrices as values
    """
    if methods is None:
        methods = ['euclidean', 'manhattan', 'cosine', 'correlation']
    
    distance_matrices = {}
    
    for method in methods:
        try:
            distance_matrices[method] = calculate_distance_matrix(feature_matrix, method)
            logger.info(f"Successfully calculated {method} distance matrix")
        except Exception as e:
            logger.warning(f"Failed to calculate {method} distance matrix: {e}")
    
    logger.info(f"Compared {len(distance_matrices)} distance methods")
    
    return distance_matrices