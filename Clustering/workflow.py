"""
Complete clustering analysis workflow for F1 driver behavioral analysis.

This module provides high-level functions to run the complete clustering
analysis pipeline from data extraction to results interpretation.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from typing import List, Dict, Optional, Tuple

from .feature_engineering import create_driver_feature_matrix, validate_corner_stats
from .distance_metrics import calculate_distance_matrix
from .optimization import find_optimal_clusters
from .kmeans_clustering import perform_kmeans_clustering
from .analysis import analyze_cluster_characteristics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_clustering_analysis(corner_stats: pd.DataFrame, 
                          granularity: str = 'corner_specific',
                          n_clusters: Optional[int] = None,
                          distance_method: str = 'euclidean',
                          save_results: bool = True,
                          results_prefix: str = 'clustering_results') -> Dict:
    """
    Run complete clustering analysis pipeline.
    
    Parameters:
    -----------
    corner_stats : pd.DataFrame
        Corner statistics data with driver behavioral metrics
    granularity : str, default='corner_specific'
        Level of granularity for feature matrix:
        - 'corner_specific': Each row represents driver-corner combination
        - 'speed_class': Each row represents driver-speed_class combination  
        - 'driver_aggregated': Each row represents driver (original behavior)
    n_clusters : int, optional
        Number of clusters. If None, uses elbow method to find optimal number
    distance_method : str, default='euclidean'
        Distance calculation method for similarity analysis
    save_results : bool, default=True
        Whether to save results to files
    results_prefix : str, default='clustering_results'
        Prefix for saved result files
        
    Returns:
    --------
    Dict
        Complete analysis results including feature matrix, clusters, and statistics
    """
    logger.info(f"Starting clustering analysis with {granularity} granularity")
    
    # Step 1: Validate input data
    if not validate_corner_stats(corner_stats):
        raise ValueError("Invalid corner statistics data")
    
    # Step 2: Create feature matrix
    feature_matrix, feature_names = create_driver_feature_matrix(corner_stats, granularity=granularity)
    
    if len(feature_matrix) == 0:
        raise ValueError("No valid feature matrix created")
    
    # Step 3: Calculate distance matrix
    distance_matrix = calculate_distance_matrix(feature_matrix, method=distance_method)
    
    # Step 4: Determine optimal number of clusters
    if n_clusters is None:
        optimal_clusters = find_optimal_clusters(feature_matrix)
    else:
        optimal_clusters = n_clusters
        logger.info(f"Using specified number of clusters: {optimal_clusters}")
    
    # Step 5: Perform clustering
    cluster_labels, kmeans_model = perform_kmeans_clustering(feature_matrix, optimal_clusters)
    
    # Step 6: Analyze cluster characteristics
    cluster_stats, distinguishing_features = analyze_cluster_characteristics(feature_matrix, cluster_labels)
    
    # Step 7: Compile results
    results = {
        'feature_matrix': feature_matrix,
        'feature_names': feature_names,
        'distance_matrix': distance_matrix,
        'cluster_labels': cluster_labels,
        'cluster_stats': cluster_stats,
        'distinguishing_features': distinguishing_features,
        'n_clusters': optimal_clusters,
        'granularity': granularity,
        'distance_method': distance_method,
        'kmeans_model': kmeans_model,
        'corner_stats': corner_stats
    }
    
    # Step 8: Save results if requested
    if save_results:
        save_analysis_results(results, prefix=results_prefix)
    
    logger.info(f"Clustering analysis completed successfully with {granularity} granularity")
    logger.info(f"Found {optimal_clusters} clusters from {len(feature_matrix)} samples")
    
    return results


def compare_granularities(corner_stats: pd.DataFrame,
                         granularities: List[str] = None,
                         n_clusters: Optional[int] = None,
                         distance_method: str = 'euclidean') -> Dict[str, Dict]:
    """
    Run clustering analysis with different granularities for comparison.
    
    Parameters:
    -----------
    corner_stats : pd.DataFrame
        Corner statistics data
    granularities : List[str], optional
        List of granularities to compare. If None, uses all three.
    n_clusters : int, optional
        Number of clusters. If None, uses elbow method for each granularity
    distance_method : str, default='euclidean'
        Distance calculation method
        
    Returns:
    --------
    Dict[str, Dict]
        Results for each granularity
    """
    if granularities is None:
        granularities = ['corner_specific', 'speed_class', 'driver_aggregated']
    
    logger.info(f"Comparing clustering results across granularities: {granularities}")
    
    all_results = {}
    
    for granularity in granularities:
        logger.info(f"\n{'='*50}")
        logger.info(f"ANALYZING WITH {granularity.upper()} GRANULARITY")
        logger.info(f"{'='*50}")
        
        try:
            results = run_clustering_analysis(
                corner_stats, 
                granularity=granularity,
                n_clusters=n_clusters,
                distance_method=distance_method,
                save_results=True,
                results_prefix=f'clustering_results_{granularity}'
            )
            all_results[granularity] = results
            
            # Summary statistics
            logger.info(f"Summary for {granularity}:")
            logger.info(f"  - Samples: {len(results['feature_matrix'])}")
            logger.info(f"  - Clusters found: {results['n_clusters']}")
            logger.info(f"  - Features: {len(results['feature_names'])}")
            
        except Exception as e:
            logger.error(f"Failed to analyze {granularity}: {e}")
            continue
    
    # Compare results summary
    logger.info(f"\n{'='*50}")
    logger.info("GRANULARITY COMPARISON SUMMARY")
    logger.info(f"{'='*50}")
    
    for granularity, results in all_results.items():
        logger.info(f"{granularity:20}: {len(results['feature_matrix']):3d} samples → {results['n_clusters']:2d} clusters")
    
    return all_results


def analyze_clustering_results(results: Dict, granularity: str) -> None:
    """
    Analyze and log clustering results based on granularity type.
    
    Parameters:
    -----------
    results : Dict
        Clustering analysis results
    granularity : str
        Type of granularity used in analysis
    """
    feature_matrix = results['feature_matrix']
    cluster_labels = results['cluster_labels']
    
    if granularity == 'corner_specific':
        logger.info("=== CORNER-SPECIFIC CLUSTERING RESULTS ===")
        logger.info("Each cluster represents similar corner approach patterns")
        
        # Group results by driver to see their corner-specific behaviors
        results_by_driver = {}
        for i, sample_name in enumerate(feature_matrix.index):
            driver = sample_name.split('_')[0]
            if driver not in results_by_driver:
                results_by_driver[driver] = []
            results_by_driver[driver].append((sample_name, cluster_labels[i]))
        
        for driver, corners in results_by_driver.items():
            corner_clusters = [f"{corner.split('_')[1]}_{corner.split('_')[2]}:C{cluster}" 
                              for corner, cluster in corners]
            logger.info(f"{driver}: {', '.join(corner_clusters)}")
            
    elif granularity == 'speed_class':
        logger.info("=== SPEED-CLASS CLUSTERING RESULTS ===")
        logger.info("Each cluster represents similar behavior across corner types")
        
        for i, sample_name in enumerate(feature_matrix.index):
            driver, speed_class = sample_name.split('_')
            logger.info(f"{driver} {speed_class} corners: Cluster {cluster_labels[i]}")
            
    elif granularity == 'driver_aggregated':
        logger.info("=== DRIVER-AGGREGATED CLUSTERING RESULTS ===")
        logger.info("Each cluster represents overall driver characteristics")
        
        for i, driver in enumerate(feature_matrix.index):
            logger.info(f"{driver}: Cluster {cluster_labels[i]}")


def save_analysis_results(results: Dict, prefix: str = 'clustering_results') -> None:
    """
    Save clustering analysis results to files.
    
    Parameters:
    -----------
    results : Dict
        Complete analysis results
    prefix : str, default='clustering_results'
        Prefix for output files
    """
    granularity = results['granularity']
    
    # Save feature matrix
    feature_file = f'{prefix}_{granularity}_features.csv'
    results['feature_matrix'].to_csv(feature_file)
    logger.info(f"Saved feature matrix to {feature_file}")
    
    # Save distance matrix
    distance_file = f'{prefix}_{granularity}_distances.csv'
    results['distance_matrix'].to_csv(distance_file)
    logger.info(f"Saved distance matrix to {distance_file}")
    
    # Save cluster statistics
    stats_file = f'{prefix}_{granularity}_cluster_stats.csv'
    results['cluster_stats'].to_csv(stats_file)
    logger.info(f"Saved cluster statistics to {stats_file}")
    
    # Save complete results as pickle
    pickle_file = f'{prefix}_{granularity}_complete.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved complete results to {pickle_file}")


def load_analysis_results(file_path: str) -> Dict:
    """
    Load previously saved clustering analysis results.
    
    Parameters:
    -----------
    file_path : str
        Path to the pickle file containing results
        
    Returns:
    --------
    Dict
        Complete analysis results
    """
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded analysis results from {file_path}")
    logger.info(f"Granularity: {results['granularity']}, Clusters: {results['n_clusters']}")
    
    return results