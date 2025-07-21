"""
Clustering module for F1 driver behavioral analysis.

This module provides clustering algorithms and analysis tools for 
identifying driving style patterns in Formula 1 telemetry data.
"""

# Clustering optimization functions
from .optimization import find_optimal_clusters

# Core clustering algorithms
from .kmeans_clustering import perform_kmeans_clustering

# Cluster analysis and interpretation
from .analysis import analyze_cluster_characteristics

# Feature engineering functions
from .feature_engineering import (
    create_driver_feature_matrix,
    get_key_features,
    validate_corner_stats,
    preprocess_feature_matrix
)

# Distance calculation functions
from .distance_metrics import (
    calculate_distance_matrix,
    find_similar_drivers,
    calculate_driver_centrality,
    calculate_cluster_cohesion,
    calculate_silhouette_coefficient,
    compare_distance_methods
)

# Complete workflow functions
from .workflow import (
    run_clustering_analysis,
    compare_granularities,
    analyze_clustering_results,
    save_analysis_results,
    load_analysis_results
)

__all__ = [
    # Optimization
    'find_optimal_clusters',
    
    # Core clustering
    'perform_kmeans_clustering',
    
    # Analysis
    'analyze_cluster_characteristics',
    
    # Feature engineering
    'create_driver_feature_matrix',
    'get_key_features',
    'validate_corner_stats',
    'preprocess_feature_matrix',
    
    # Distance metrics
    'calculate_distance_matrix',
    'find_similar_drivers',
    'calculate_driver_centrality',
    'calculate_cluster_cohesion',
    'calculate_silhouette_coefficient',
    'compare_distance_methods',
    
    # Workflow
    'run_clustering_analysis',
    'compare_granularities',
    'analyze_clustering_results',
    'save_analysis_results',
    'load_analysis_results'
]