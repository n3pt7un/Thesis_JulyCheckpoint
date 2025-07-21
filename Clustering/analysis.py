"""
Cluster analysis and interpretation functions.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_cluster_characteristics(feature_matrix: pd.DataFrame, cluster_labels: np.ndarray) -> Tuple[pd.DataFrame, dict]:
    """
    Analyze the characteristics of each cluster.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Driver feature matrix
    cluster_labels : np.ndarray
        Cluster assignments
        
    Returns:
    --------
    Tuple[pd.DataFrame, dict]
        Cluster characteristics summary (DataFrame) and distinguishing features (dict)
    """
    # Add cluster labels to feature matrix
    feature_matrix_with_clusters = feature_matrix.copy()
    feature_matrix_with_clusters['Cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = feature_matrix_with_clusters.groupby('Cluster').agg(['mean', 'std']).round(3)
    
    # Identify distinguishing features for each cluster
    cluster_means = feature_matrix_with_clusters.groupby('Cluster').mean()
    overall_means = feature_matrix.mean()
    
    distinguishing_features = {}
    for cluster in cluster_means.index:
        cluster_data = cluster_means.loc[cluster]
        differences = ((cluster_data - overall_means) / overall_means * 100).abs()
        top_features = differences.nlargest(5)
        distinguishing_features[f'Cluster_{cluster}'] = top_features.to_dict()
    
    logger.info("Cluster characteristics analysis completed")
    
    return cluster_stats, distinguishing_features