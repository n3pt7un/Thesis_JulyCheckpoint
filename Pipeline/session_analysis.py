"""
Session-level analysis and processing functions.
"""

import fastf1
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .multi_driver_analysis import extract_all_drivers_corner_data
from .data_io import save_processed_data

# Thread-local storage for FastF1 sessions to avoid conflicts
thread_local = threading.local()

def analyze_session_corner_performance(
    session: fastf1.core.Session,
    drivers: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    max_workers: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Complete analysis workflow for session corner performance.

    Parameters:
    -----------
    session : fastf1.core.Session
        FastF1 session object
    drivers : List[str], optional
        Drivers to analyze
    save_path : str, optional
        Path to save the processed data
    max_workers : int, optional
        Maximum number of parallel workers for driver processing

    Returns:
    --------
    pd.DataFrame or None
        The detailed comparison DataFrame with all corner performance data, or None if no data is available.
    """
    
    logging.info("Starting corner performance analysis...")
    
    multi_driver_data = extract_all_drivers_corner_data(
        session, drivers, max_workers=max_workers
    )
    
    if multi_driver_data['comparison_df'].empty:
        logging.info("No data available for analysis")
        return None
    
    # Use the comparison_df directly - it already contains all the data we need
    comparison_df = multi_driver_data['comparison_df']
    
    print("\nDetailed Corner Performance Data:")
    print(f"Shape: {comparison_df.shape}")
    print(f"Columns: {list(comparison_df.columns)}")
    print(f"Drivers: {sorted(comparison_df['Driver'].unique())}")
    print(f"Corners: {sorted(comparison_df['Corner'].unique())}")
    print(f"Speed Classes: {sorted(comparison_df['SpeedClass'].unique())}")
    print("\nSample data:")
    print(comparison_df.head().round(2))

    if save_path:
        save_processed_data(comparison_df, save_path)
    
    return comparison_df


def create_clustering_compatible_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary compatible with the clustering analysis expectations.
    This function transforms the detailed comparison_df into the format expected
    by driver_clustering_analysis.py.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        The detailed comparison DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Summary table with driver-level metrics in the expected format
    """
    
    if comparison_df.empty:
        return pd.DataFrame()
    
    summary_data = []
    
    # Group by both Driver and Corner to preserve corner-specific resolution
    for driver in comparison_df['Driver'].unique():
        driver_data = comparison_df[comparison_df['Driver'] == driver]
        
        for corner in driver_data['Corner'].unique():
            corner_data = driver_data[driver_data['Corner'] == corner]
            
            into_data = corner_data[corner_data['Section'] == 'into_turn']
            out_data = corner_data[corner_data['Section'] == 'out_of_turn']
            
            # Get corner classification if available
            speed_class = corner_data['SpeedClass'].iloc[0] if not corner_data.empty and 'SpeedClass' in corner_data.columns else 'Unknown'
            
            summary = {
                'Driver': driver,
                'Corner': corner,
                'SpeedClass': speed_class,
                # Use the old column names expected by clustering analysis
                'Avg_Entry_Speed': into_data['EntrySpeed'].mean() if not into_data.empty else np.nan,
                'Avg_Exit_Speed': out_data['ExitSpeed'].mean() if not out_data.empty else np.nan,
                'Max_Accel_Into': into_data['MaxAcceleration'].mean() if not into_data.empty else np.nan,
                'Max_Accel_Out': out_data['MaxAcceleration'].mean() if not out_data.empty else np.nan,
                'Max_Decel_Into': into_data['MaxDeceleration'].mean() if not into_data.empty else np.nan,
                'Avg_Throttle_Into': into_data['AvgThrottleIntensity'].mean() if not into_data.empty else np.nan,
                'Avg_Throttle_Out': out_data['AvgThrottleIntensity'].mean() if not out_data.empty else np.nan,
                'Speed_Consistency': corner_data['AvgSpeed'].std() if not corner_data.empty else np.nan,
                # Additional metrics from comparison_df
                'Corner_Entry_Count': len(into_data),
                'Corner_Exit_Count': len(out_data),
                'Total_Corner_Samples': len(corner_data)
            }
            
            summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def process_single_event(event_data: tuple, season: int, drivers: Optional[List[str]], clustering_compatible: bool, max_workers: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Process a single race event to extract corner telemetry data.
    
    Parameters:
    -----------
    event_data : tuple
        Event data from season_calendar.itertuples()
    season : int
        Season year
    drivers : List[str], optional
        List of driver identifiers
    clustering_compatible : bool
        Whether to return clustering-compatible format
    max_workers : int, optional
        Maximum number of parallel workers for driver processing within this event
        
    Returns:
    --------
    pd.DataFrame or None
        Processed event data or None if processing failed
    """
    event = event_data
    
    try:
        # Set logging level for this thread to avoid spam
        fastf1.logger.LoggingManager().set_level(logging.CRITICAL)
        
        # Conservative driver processing to avoid cache conflicts
        if max_workers is None or max_workers > 2:
            max_workers = 2  # Limit to 2 to reduce SQLite cache contention
        
        logging.info(f"Processing {event.EventName} {season}")
        
        race_session = fastf1.get_session(season, event.EventName, 'R')
        race_session.load(telemetry=True, laps=True, weather=False)
        
        drivers_in_session = race_session.results['Abbreviation'].dropna().unique().tolist()
        current_drivers = drivers if drivers is not None else drivers_in_session
        
        # Get the detailed comparison_df
        comparison_df = analyze_session_corner_performance(
            race_session, current_drivers, max_workers=max_workers
        )

        if comparison_df is not None and not comparison_df.empty:
            # Transform to clustering-compatible format if requested
            if clustering_compatible:
                summary = create_clustering_compatible_summary(comparison_df)
            else:
                summary = comparison_df
            
            if not summary.empty:
                summary['EventName'] = event.EventName
                summary['EventDate'] = event.EventDate
                summary['Circuit'] = event.Location 
                return summary
                
        return None

    except Exception as e:
        logging.error(f"Could not process {event.EventName} for season {season}: {e}")
        return None


def extract_corner_telemetry_season(
    season: int, 
    include_testing: bool = False,
    drivers: Optional[List[str]] = None,
    clustering_compatible: bool = True,
    max_workers: int = 4,
    max_driver_workers: Optional[int] = None
) -> pd.DataFrame:
    """ 
    Extract corner telemetry data for all specified drivers in a season using parallel processing.
    
    Parameters:
    -----------
    season : int
        Season to extract data from
    include_testing : bool, default=False
        Whether to include testing sessions
    drivers : List[str], optional
        List of driver identifiers. If None, uses all drivers in session
    clustering_compatible : bool, default=True
        If True, returns data in format compatible with clustering analysis.
        If False, returns raw comparison_df format.
    max_workers : int, default=4
        Maximum number of parallel workers for processing events
    max_driver_workers : int, optional
        Maximum number of parallel workers for processing drivers within each event.
        If None, defaults to min(4, number_of_drivers_per_event)
        
    Returns:
    --------
    pd.DataFrame
        Comparison table with metrics for each corner section
    """
    fastf1.logger.LoggingManager().set_level(logging.CRITICAL)
    season_calendar = fastf1.events.get_event_schedule(season)
    if not include_testing:
        season_calendar = season_calendar.copy()
        season_calendar = season_calendar[season_calendar['EventFormat'] != 'testing']
    
    all_summaries = []
    
    # Convert calendar to list of events for parallel processing
    events_list = list(season_calendar.itertuples())
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_event = {
            executor.submit(process_single_event, event, season, drivers, clustering_compatible, max_driver_workers): event 
            for event in events_list
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(events_list), desc=f"Processing {season} season") as pbar:
            for future in as_completed(future_to_event):
                event = future_to_event[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_summaries.append(result)
                except Exception as e:
                    logging.error(f"Error processing {event.EventName}: {e}")
                finally:
                    pbar.update(1)

    fastf1.logger.LoggingManager().set_level(logging.INFO)
    
    if all_summaries:
        return pd.concat(all_summaries, ignore_index=True)
    return pd.DataFrame()