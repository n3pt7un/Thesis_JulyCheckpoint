"""
Feature engineering functions for F1 driver clustering analysis.

This module handles the creation of feature matrices from corner statistics
with different levels of granularity to capture driver behavioral patterns.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_driver_feature_matrix(corner_stats: pd.DataFrame, granularity: str = 'corner_specific') -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a feature matrix for clustering based on corner approach statistics.
    Preserves granularity to capture driver behavior in different scenarios.
    
    Parameters:
    -----------
    corner_stats : pd.DataFrame
        Corner statistics data with columns: Driver, Corner, SpeedClass, etc.
    granularity : str, default='corner_specific'
        Level of granularity to maintain:
        - 'corner_specific': Each row represents driver-corner combination
        - 'speed_class': Each row represents driver-speed_class combination  
        - 'driver_aggregated': Each row represents driver (original behavior)
        
    Returns:
    --------
    Tuple[pd.DataFrame, List[str]]
        Feature matrix and list of feature names
    """
    # Define the key features for driving style analysis
    key_features = [
        'AvgSpeed', 
        'TopSpeed', 
        'MinSpeed', 
        'EntrySpeed', 
        'ExitSpeed', 
        'AvgGear', 
        'GearChanges', 
        'AvgThrottleIntensity', 
        'MaxThrottle', 
        'MinThrottle', 
        'MaxDeceleration', 
        'MaxAcceleration', 
        'AvgAcceleration', 
    ]
    
    if granularity == 'corner_specific':
        # Group by Driver AND Corner to preserve corner-specific behavior
        grouping_cols = ['Driver', 'Corner', 'SpeedClass']
        group_data = corner_stats.groupby(grouping_cols)[key_features].agg([
            'mean',   # Average performance in this corner
            'std',    # Consistency in this corner
            'min',    # Conservative approach in this corner
            'max'     # Aggressive approach in this corner
        ]).round(3)
        
        # Create meaningful index for clustering
        index_names = [f"{driver}_{corner}_{speed_class}" 
                      for driver, corner, speed_class in group_data.index]
        
    elif granularity == 'speed_class':
        # Group by Driver AND SpeedClass to capture behavior by corner type
        grouping_cols = ['Driver', 'SpeedClass']
        group_data = corner_stats.groupby(grouping_cols)[key_features].agg([
            'mean',   # Average performance in this corner type
            'std',    # Consistency in this corner type
            'min',    # Conservative approach in this corner type
            'max'     # Aggressive approach in this corner type
        ]).round(3)
        
        # Create meaningful index for clustering
        index_names = [f"{driver}_{speed_class}" 
                      for driver, speed_class in group_data.index]
        
    elif granularity == 'driver_aggregated':
        # Original behavior: aggregate everything at driver level
        grouping_cols = ['Driver']
        group_data = corner_stats.groupby(grouping_cols)[key_features].agg([
            'mean',   # Average performance across all corners
            'std',    # Overall consistency
            'min',    # Most conservative approach
            'max'     # Most aggressive approach
        ]).round(3)
        
        # Use driver names as index
        index_names = [driver for driver in group_data.index]
        
    else:
        raise ValueError(f"Unknown granularity: {granularity}. Choose from: 'corner_specific', 'speed_class', 'driver_aggregated'")
    
    # Flatten column names
    feature_columns = []
    for feature in key_features:
        for stat in ['mean', 'std', 'min', 'max']:
            feature_columns.append(f"{feature}_{stat}")
    
    # Flatten the MultiIndex columns
    group_data.columns = feature_columns
    
    # Reset index and create meaningful row identifiers
    feature_matrix = group_data.reset_index(drop=True)
    feature_matrix.index = index_names
    
    # Add metadata columns for analysis
    if granularity == 'corner_specific':
        # Add corner and speed class info as metadata (not for clustering)
        metadata = pd.DataFrame(list(corner_stats.groupby(grouping_cols).groups.keys()), 
                               columns=['Driver', 'Corner', 'SpeedClass'])
        metadata.index = index_names
    elif granularity == 'speed_class':
        metadata = pd.DataFrame(list(corner_stats.groupby(grouping_cols).groups.keys()), 
                               columns=['Driver', 'SpeedClass'])
        metadata.index = index_names
    else:
        metadata = pd.DataFrame(list(corner_stats.groupby(grouping_cols).groups.keys()), 
                               columns=['Driver'])
        metadata.index = index_names
    
    # Save for debugging/inspection
    feature_matrix_with_metadata = pd.concat([metadata, feature_matrix], axis=1)
    feature_matrix_with_metadata.to_csv(f'driver_features_{granularity}.csv')
    
    # Handle missing values
    feature_matrix = feature_matrix.fillna(feature_matrix.mean())
    feature_matrix = feature_matrix.fillna(0)  # Fill any remaining NaNs with 0
    
    # Remove rows with too many missing values
    feature_matrix = feature_matrix.dropna(thresh=len(feature_columns) * 0.7)
    
    logger.info(f"Created {granularity} feature matrix with {len(feature_matrix)} samples and {len(feature_columns)} features")
    logger.info(f"Sample distribution: {dict(pd.Series(index_names).str.split('_').str[0].value_counts())}")
    
    return feature_matrix, feature_columns


def get_key_features() -> List[str]:
    """
    Get the list of key features used for driving style analysis.
    These should match the features used in create_driver_feature_matrix.
    
    Returns:
    --------
    List[str]
        List of key feature names from the original corner analysis data
    """
    return [
        'AvgSpeed', 
        'TopSpeed', 
        'MinSpeed', 
        'EntrySpeed', 
        'ExitSpeed', 
        'AvgGear', 
        'GearChanges', 
        'AvgThrottleIntensity', 
        'MaxThrottle', 
        'MinThrottle', 
        'MaxDeceleration', 
        'MaxAcceleration', 
        'AvgAcceleration', 
    ]


def validate_corner_stats(corner_stats: pd.DataFrame) -> bool:
    """
    Validate that the corner statistics DataFrame has the required columns.
    
    Parameters:
    -----------
    corner_stats : pd.DataFrame
        Corner statistics data
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    required_columns = ['Driver', 'Corner', 'SpeedClass'] + get_key_features()
    missing_columns = set(required_columns) - set(corner_stats.columns)
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    if corner_stats.empty:
        logger.error("Corner statistics DataFrame is empty")
        return False
    
    logger.info(f"Corner statistics validation passed. Shape: {corner_stats.shape}")
    return True


def preprocess_feature_matrix(feature_matrix: pd.DataFrame, 
                            fill_method: str = 'mean',
                            drop_threshold: float = 0.7) -> pd.DataFrame:
    """
    Preprocess the feature matrix by handling missing values and outliers.
    
    Parameters:
    -----------
    feature_matrix : pd.DataFrame
        Raw feature matrix
    fill_method : str, default='mean'
        Method to fill missing values ('mean', 'median', 'zero')
    drop_threshold : float, default=0.7
        Threshold for dropping rows with too many missing values (as fraction)
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed feature matrix
    """
    logger.info(f"Preprocessing feature matrix. Original shape: {feature_matrix.shape}")
    
    # Handle missing values
    if fill_method == 'mean':
        feature_matrix = feature_matrix.fillna(feature_matrix.mean())
    elif fill_method == 'median':
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
    elif fill_method == 'zero':
        feature_matrix = feature_matrix.fillna(0)
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")
    
    # Fill any remaining NaNs with 0
    feature_matrix = feature_matrix.fillna(0)
    
    # Remove rows with too many missing values
    min_valid_features = int(len(feature_matrix.columns) * drop_threshold)
    feature_matrix = feature_matrix.dropna(thresh=min_valid_features)
    
    logger.info(f"Preprocessed feature matrix. Final shape: {feature_matrix.shape}")
    
    return feature_matrix