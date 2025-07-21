"""
Multi-driver analysis and comparison functions.
"""

import pandas as pd
import numpy as np
import fastf1
import logging
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback

try:
    from .corner_analysis import extract_corner_telemetry_sections, compare_corner_sections
except ImportError:
    from corner_analysis import extract_corner_telemetry_sections, compare_corner_sections

def _process_single_driver(args):
    """
    Worker function to process a single driver's corner data.
    
    Parameters:
    -----------
    args : tuple
        Tuple containing (session, driver, distance_before, distance_after, corner_selection_method)
        
    Returns:
    --------
    tuple
        (driver, driver_corners, driver_comparison, success, error_msg)
    """
    session, driver, distance_before, distance_after, corner_selection_method = args
    
    try:
        # Extract corner data for this driver
        driver_corners = extract_corner_telemetry_sections(
            session, driver, distance_before, distance_after, 
            corner_selection_method=corner_selection_method
        )
        
        # Create comparison DataFrame for this driver
        driver_comparison = compare_corner_sections(driver_corners)
        if not driver_comparison.empty:
            driver_comparison['Driver'] = driver
        
        return driver, driver_corners, driver_comparison, True, None
        
    except Exception as e:
        error_msg = f"Could not process driver {driver}: {e}"
        logging.warning(error_msg)
        return driver, None, pd.DataFrame(), False, str(e)


def extract_all_drivers_corner_data(
    session: fastf1.core.Session,
    drivers: Optional[List[str]] = None,
    distance_before: float = 100.0,
    distance_after: float = 100.0,
    corner_selection_method: str = 'default',
    max_workers: Optional[int] = 4,
    use_processes: bool = False
) -> Dict:
    """
    Extract corner telemetry data for all specified drivers in a session using parallel processing.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        FastF1 session object with loaded telemetry data
    drivers : List[str], optional
        List of driver identifiers. If None, uses all drivers in session
    distance_before : float, default=100.0
        Distance in meters before corner apex
    distance_after : float, default=100.0  
        Distance in meters after corner apex
    corner_selection_method : str, default='default'
        Corner selection method ('default', 'all')
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses min(32, number of drivers)
    use_processes : bool, default=False
        If True, uses ProcessPoolExecutor for CPU-intensive tasks. 
        If False, uses ThreadPoolExecutor (recommended for I/O bound operations)
        
    Returns:
    --------
    Dict
        Dictionary with driver data and combined comparison DataFrame
    """
    
    if drivers is None:
        drivers = session.laps['Driver'].unique().tolist()
    
    all_driver_data = {}
    all_comparison_data = []
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(32, len(drivers))  # Reasonable default
    
    executor_type = "process" if use_processes else "thread"
    logging.info(f"Extracting corner data for {len(drivers)} drivers using {max_workers} parallel {executor_type} workers...")
    
    # Prepare arguments for parallel processing
    driver_args = [
        (session, driver, distance_before, distance_after, corner_selection_method)
        for driver in drivers
    ]
    
    # Process drivers in parallel
    successful_drivers = 0
    failed_drivers = 0
    
    # Choose executor type based on use_processes parameter
    # Note: ProcessPoolExecutor may have issues with pickling FastF1 objects
    # ThreadPoolExecutor is generally recommended for this use case
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    try:
        with ExecutorClass(max_workers=max_workers) as executor:
            # Submit all driver processing tasks
            future_to_driver = {
                executor.submit(_process_single_driver, args): args[1] 
                for args in driver_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_driver):
                driver = future_to_driver[future]
                try:
                    driver_name, driver_corners, driver_comparison, success, error_msg = future.result()
                    
                    if success:
                        all_driver_data[driver_name] = driver_corners
                        if not driver_comparison.empty:
                            all_comparison_data.append(driver_comparison)
                        successful_drivers += 1
                    else:
                        failed_drivers += 1
                        
                except Exception as e:
                    logging.error(f"Unexpected error processing driver {driver}: {e}")
                    failed_drivers += 1
    
    except Exception as e:
        logging.error(f"Failed to initialize {executor_type} executor: {e}")
        if use_processes:
            logging.info("Falling back to ThreadPoolExecutor due to process executor failure...")
            # Fallback to ThreadPoolExecutor if ProcessPoolExecutor fails
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_driver = {
                    executor.submit(_process_single_driver, args): args[1] 
                    for args in driver_args
                }
                
                for future in as_completed(future_to_driver):
                    driver = future_to_driver[future]
                    try:
                        driver_name, driver_corners, driver_comparison, success, error_msg = future.result()
                        
                        if success:
                            all_driver_data[driver_name] = driver_corners
                            if not driver_comparison.empty:
                                all_comparison_data.append(driver_comparison)
                            successful_drivers += 1
                        else:
                            failed_drivers += 1
                            
                    except Exception as e:
                        logging.error(f"Unexpected error processing driver {driver}: {e}")
                        failed_drivers += 1
        else:
            raise
    
    logging.info(f"Driver processing complete: {successful_drivers} successful, {failed_drivers} failed")
    
    # Combine all driver comparison data
    if all_comparison_data:
        combined_comparison = pd.concat(all_comparison_data, ignore_index=True)
    else:
        combined_comparison = pd.DataFrame()
    
    return {
        'driver_data': all_driver_data,
        'comparison_df': combined_comparison,
        'drivers': drivers
    }


def get_corner_performance_view(comparison_df: pd.DataFrame, view_type: str = 'detailed') -> pd.DataFrame:
    """
    Get different views of the corner performance data.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        The detailed comparison DataFrame from extract_all_drivers_corner_data()
    view_type : str, default='detailed'
        Type of view to return:
        - 'detailed': Full comparison_df (no aggregation)
        - 'driver_summary': Aggregated metrics per driver (loses corner resolution)
        - 'corner_summary': Aggregated metrics per driver per corner
        - 'speed_class_summary': Aggregated metrics per driver per speed class
        
    Returns:
    --------
    pd.DataFrame
        Requested view of the data
    """
    
    if comparison_df.empty:
        return pd.DataFrame()
    
    if view_type == 'detailed':
        # Return the full detailed data - this preserves maximum resolution
        return comparison_df.copy()
    
    elif view_type == 'driver_summary':
        # Aggregate across all corners and sections for each driver (original behavior)
        return comparison_df.groupby('Driver').agg({
            'AvgSpeed': 'mean',
            'EntrySpeed': 'mean', 
            'ExitSpeed': 'mean',
            'MaxAcceleration': 'mean',
            'MaxDeceleration': 'mean',
            'AvgThrottleIntensity': 'mean',
            'DataPoints': 'sum',
            'Corner': 'nunique'  # Number of unique corners per driver
        }).round(2).reset_index()
    
    elif view_type == 'corner_summary':
        # Aggregate across sections but preserve corner identity
        return comparison_df.groupby(['Driver', 'Corner', 'SpeedClass']).agg({
            'AvgSpeed': 'mean',
            'EntrySpeed': lambda x: x[comparison_df.loc[x.index, 'Section'] == 'into_turn'].mean(),
            'ExitSpeed': lambda x: x[comparison_df.loc[x.index, 'Section'] == 'out_of_turn'].mean(),
            'MaxAcceleration': 'mean',
            'MaxDeceleration': 'mean', 
            'AvgThrottleIntensity': 'mean',
            'DataPoints': 'sum'
        }).round(2).reset_index()
    
    elif view_type == 'speed_class_summary':
        # Aggregate by speed class (slow/medium/fast corners)
        return comparison_df.groupby(['Driver', 'SpeedClass']).agg({
            'AvgSpeed': 'mean',
            'EntrySpeed': 'mean',
            'ExitSpeed': 'mean', 
            'MaxAcceleration': 'mean',
            'MaxDeceleration': 'mean',
            'AvgThrottleIntensity': 'mean',
            'Corner': 'nunique',  # Number of corners per speed class
            'DataPoints': 'sum'
        }).round(2).reset_index()
    
    else:
        raise ValueError(f"Unknown view_type: {view_type}. Choose from: 'detailed', 'driver_summary', 'corner_summary', 'speed_class_summary'")