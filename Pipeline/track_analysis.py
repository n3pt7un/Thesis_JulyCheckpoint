import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    # Try relative imports first (when used as part of package)
    from .telemetry_extraction import get_all_car_data
except ImportError:
    # Fall back to absolute imports (when run directly)
    from telemetry_extraction import get_all_car_data

def extract_driver_track_analysis(session: fastf1.core.Session, driver: str) -> pd.DataFrame:
    """Extract track analysis for a driver.
    Add generic track analysis such as:
        - Average speed 
        - Speed consistency (measured as standard deviation of speed)
        - Average acceleration (positive values only)
        - Acceleration consistency (measured as standard deviation of positive acceleration)
        - Average deceleration (negative acceleration values only)
        - Deceleration consistency (measured as standard deviation of negative acceleration)
        - Throttle usage percentage (% of time throttle > 0)
        - Brake usage percentage (% of time brake > 0)
        - DRS usage percentage (% of time DRS is active)
    Args:
        session: fastf1.core.Session
        driver: str
        
    Returns:
        pd.DataFrame
    """
    tel = get_all_car_data(session, driver)
    
    # Average speed
    avg_speed = tel['Speed'].mean()
    # Speed consistency
    speed_std = tel['Speed'].std()
    
    # Separate acceleration and deceleration
    acceleration_data = tel['Acceleration(m/s^2)'][tel['Acceleration(m/s^2)'] > 0]
    deceleration_data = tel['Acceleration(m/s^2)'][tel['Acceleration(m/s^2)'] < 0]
    
    # Average acceleration (positive values only)
    avg_acceleration = acceleration_data.mean()
    # Acceleration consistency
    acceleration_std = acceleration_data.std()
    # Average deceleration (negative values only, but show as absolute value)
    avg_deceleration = abs(deceleration_data.mean())
    # Deceleration consistency
    deceleration_std = deceleration_data.std()
    
    # Calculate percentage of time each system is actively used
    total_data_points = len(tel)
    
    # Throttle usage percentage (% of time throttle > 0)
    throttle_active_time = (tel['Throttle'] > 0).sum()
    throttle_usage_percentage = (throttle_active_time / total_data_points) * 100
    
    # Brake usage percentage (% of time brake > 0)
    brake_active_time = (tel['Brake'] > 0).sum()
    brake_usage_percentage = (brake_active_time / total_data_points) * 100
    
    # DRS usage percentage (% of time DRS is active)
    # DRS is typically 1 when active, 0 when inactive
    drs_active_time = (tel['DRS'] > 0).sum()
    drs_usage_percentage = (drs_active_time / total_data_points) * 100

    # Create a DataFrame with the results
    track_analysis = pd.DataFrame({
        'Metric': [
            'Average Speed', 
            'Speed Consistency', 
            'Average Acceleration', 
            'Acceleration Consistency', 
            'Average Deceleration', 
            'Deceleration Consistency', 
            'Throttle Usage Percentage', 
            'Brake Usage Percentage', 
            'DRS Usage Percentage'
        ],
        'Value': [
            avg_speed, 
            speed_std, 
            avg_acceleration, 
            acceleration_std, 
            avg_deceleration, 
            deceleration_std, 
            throttle_usage_percentage, 
            brake_usage_percentage, 
            drs_usage_percentage
        ]
    })
    
    return track_analysis


def extract_comprehensive_driver_features(
    sessions_data: list,
    drivers: list = None,
    include_race_identifiers: bool = True
) -> pd.DataFrame:
    """
    Extract comprehensive driver features combining track analysis and corner statistics
    for multiple races into a driver×features matrix.
    
    Creates a table with:
    - d rows (number of drivers)  
    - 33×r features (33 features per race × r races)
    
    The 33 features per race consist of:
    - 9 track analysis features (speed, acceleration, usage percentages)
    - 24 corner statistics features (8 features × 3 corner speed classes)
    
    Args:
        sessions_data: List of tuples [(session, race_name), ...] 
                      where session is fastf1.core.Session and race_name is str
        drivers: List of driver abbreviations. If None, uses all drivers from first session
        include_race_identifiers: If True, includes race name columns for reference
        
    Returns:
        pd.DataFrame: Driver feature matrix with shape (d_drivers, 33*r_races + race_info)
    """
    
    if not sessions_data:
        return pd.DataFrame()
    
    # Get all drivers if not specified
    if drivers is None:
        first_session = sessions_data[0][0]
        drivers = first_session.laps['Driver'].unique().tolist()
    
    # Initialize feature collection
    all_driver_features = []
    race_names = []
    
    print(f"Processing {len(sessions_data)} races for {len(drivers)} drivers...")
    
    for session, race_name in sessions_data:
        print(f"Processing {race_name}...")
        race_names.append(race_name)
        
        # Extract track analysis for all drivers
        track_features = {}
        for driver in drivers:
            try:
                track_analysis = extract_driver_track_analysis(session, driver)
                # Convert to dictionary with feature names
                track_dict = {
                    f'TrackAnalysis_{metric}': value 
                    for metric, value in zip(track_analysis['Metric'], track_analysis['Value'])
                }
                track_features[driver] = track_dict
            except Exception as e:
                print(f"Warning: Could not extract track analysis for {driver} in {race_name}: {e}")
                # Fill with NaN for missing drivers
                track_features[driver] = {
                    'TrackAnalysis_Average Speed': np.nan,
                    'TrackAnalysis_Speed Consistency': np.nan,
                    'TrackAnalysis_Average Acceleration': np.nan,
                    'TrackAnalysis_Acceleration Consistency': np.nan,
                    'TrackAnalysis_Average Deceleration': np.nan,
                    'TrackAnalysis_Deceleration Consistency': np.nan,
                    'TrackAnalysis_Throttle Usage Percentage': np.nan,
                    'TrackAnalysis_Brake Usage Percentage': np.nan,
                    'TrackAnalysis_DRS Usage Percentage': np.nan
                }
        
        # Extract corner analysis for all drivers
        try:
            # Lazy import to avoid circular imports
            try:
                from .multi_driver_analysis import extract_all_drivers_corner_data
            except ImportError:
                from multi_driver_analysis import extract_all_drivers_corner_data
                
            multi_driver_data = extract_all_drivers_corner_data(session, drivers)
            comparison_df = multi_driver_data['comparison_df']
            
            # Create corner features by speed class for each driver
            corner_features = {}
            for driver in drivers:
                driver_corner_data = comparison_df[comparison_df['Driver'] == driver] if not comparison_df.empty else pd.DataFrame()
                corner_dict = _extract_corner_features_by_speed_class(driver_corner_data)
                corner_features[driver] = corner_dict
                
        except Exception as e:
            print(f"Warning: Could not extract corner analysis for {race_name}: {e}")
            # Fill with NaN for all drivers
            corner_features = {driver: _get_empty_corner_features() for driver in drivers}
        
        # Combine track and corner features for this race
        race_features = {}
        for driver in drivers:
            combined_features = {}
            combined_features.update(track_features.get(driver, {}))
            combined_features.update(corner_features.get(driver, {}))
            
            # Add race prefix to feature names
            race_prefixed_features = {
                f"{race_name}_{feature_name}": value 
                for feature_name, value in combined_features.items()
            }
            race_features[driver] = race_prefixed_features
        
        all_driver_features.append(race_features)
    
    # Combine all races into final driver×features matrix
    final_features = {}
    for driver in drivers:
        driver_row = {}
        driver_row['Driver'] = driver
        
        # Add race identifiers if requested
        if include_race_identifiers:
            driver_row['Races_Processed'] = ','.join(race_names)
            driver_row['Number_of_Races'] = len(race_names)
        
        # Flatten all race features for this driver
        for race_data in all_driver_features:
            if driver in race_data:
                driver_row.update(race_data[driver])
        
        final_features[driver] = driver_row
    
    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(final_features, orient='index').reset_index(drop=True)
    
    print(f"Final feature matrix shape: {result_df.shape}")
    print(f"Drivers: {len(drivers)}")
    print(f"Races: {len(race_names)}")
    print(f"Features per race: ~33")
    print(f"Total feature columns: {result_df.shape[1] - (3 if include_race_identifiers else 1)}")
    
    return result_df


def _extract_corner_features_by_speed_class(driver_corner_data: pd.DataFrame) -> dict:
    """
    Extract 8 corner features for each of the 3 speed classes (Fast, Medium, Slow).
    Returns 24 corner features total.
    """
    corner_features = {}
    
    # Define the 3 speed classes
    speed_classes = ['Fast', 'Medium', 'Slow']
    
    for speed_class in speed_classes:
        class_data = driver_corner_data[driver_corner_data['SpeedClass'] == speed_class] if not driver_corner_data.empty else pd.DataFrame()
        
        if not class_data.empty:
            # Separate into/out sections
            into_data = class_data[class_data['Section'] == 'into_turn']
            out_data = class_data[class_data['Section'] == 'out_of_turn']
            
            # Extract 8 features for this speed class
            features = {
                f'Corner_{speed_class}_Avg_Entry_Speed': into_data['EntrySpeed'].mean() if not into_data.empty else np.nan,
                f'Corner_{speed_class}_Avg_Exit_Speed': out_data['ExitSpeed'].mean() if not out_data.empty else np.nan,
                f'Corner_{speed_class}_Max_Accel_Into': into_data['MaxAcceleration'].mean() if not into_data.empty else np.nan,
                f'Corner_{speed_class}_Max_Accel_Out': out_data['MaxAcceleration'].mean() if not out_data.empty else np.nan,
                f'Corner_{speed_class}_Max_Decel_Into': into_data['MaxDeceleration'].mean() if not into_data.empty else np.nan,
                f'Corner_{speed_class}_Avg_Throttle_Into': into_data['AvgThrottleIntensity'].mean() if not into_data.empty else np.nan,
                f'Corner_{speed_class}_Avg_Throttle_Out': out_data['AvgThrottleIntensity'].mean() if not out_data.empty else np.nan,
                f'Corner_{speed_class}_Speed_Consistency': class_data['AvgSpeed'].std() if not class_data.empty else np.nan
            }
        else:
            # Fill with NaN if no data for this speed class
            features = {
                f'Corner_{speed_class}_Avg_Entry_Speed': np.nan,
                f'Corner_{speed_class}_Avg_Exit_Speed': np.nan,
                f'Corner_{speed_class}_Max_Accel_Into': np.nan,
                f'Corner_{speed_class}_Max_Accel_Out': np.nan,
                f'Corner_{speed_class}_Max_Decel_Into': np.nan,
                f'Corner_{speed_class}_Avg_Throttle_Into': np.nan,
                f'Corner_{speed_class}_Avg_Throttle_Out': np.nan,
                f'Corner_{speed_class}_Speed_Consistency': np.nan
            }
        
        corner_features.update(features)
    
    return corner_features


def _get_empty_corner_features() -> dict:
    """Get empty corner features filled with NaN for when corner analysis fails."""
    corner_features = {}
    speed_classes = ['Fast', 'Medium', 'Slow']
    
    for speed_class in speed_classes:
        features = {
            f'Corner_{speed_class}_Avg_Entry_Speed': np.nan,
            f'Corner_{speed_class}_Avg_Exit_Speed': np.nan,
            f'Corner_{speed_class}_Max_Accel_Into': np.nan,
            f'Corner_{speed_class}_Max_Accel_Out': np.nan,
            f'Corner_{speed_class}_Max_Decel_Into': np.nan,
            f'Corner_{speed_class}_Avg_Throttle_Into': np.nan,
            f'Corner_{speed_class}_Avg_Throttle_Out': np.nan,
            f'Corner_{speed_class}_Speed_Consistency': np.nan
        }
        corner_features.update(features)
    
    return corner_features


# Example usage function for multiple races
def extract_season_driver_features(
    year: int,
    race_names: list = None,
    drivers: list = None,
    max_races: int = None,
    include_testing: bool = False
) -> pd.DataFrame:
    """
    Extract comprehensive driver features for multiple races in a season.
    
    Args:
        year: Season year (e.g., 2024)
        race_names: List of race names. If None, uses first max_races from season
        drivers: List of driver abbreviations. If None, uses all drivers
        max_races: Maximum number of races to process. If None, processes all
        include_testing: If True, only processes first 3 races
    Returns:
        pd.DataFrame: Driver feature matrix
    """
    
    # Get race schedule
    import fastf1
    schedule = fastf1.events.get_event_schedule(year)
    if not include_testing:
        schedule = schedule.copy()
        schedule = schedule[schedule['EventFormat'] != 'testing']
    
    if race_names is None:
        available_races = schedule['EventName'].tolist()
        if max_races:
            available_races = available_races[:max_races]
        race_names = available_races
    
    # Load sessions
    sessions_data = []
    for race_name in race_names:
        try:
            session = fastf1.get_session(year, race_name, 'R')
            session.load()
            sessions_data.append((session, race_name))
            print(f"Loaded {race_name}")
        except Exception as e:
            print(f"Could not load {race_name}: {e}")
    
    if not sessions_data:
        print("No sessions could be loaded")
        return pd.DataFrame()
    
    return extract_comprehensive_driver_features(sessions_data, drivers)

# Example usage
if __name__ == "__main__":
    # Example 1: Process multiple races for comprehensive driver features
    print("=== Example: Comprehensive Driver Feature Extraction ===")
    
    # Load 2-3 races from 2024 season  
    try:
        # Load specific races
        sessions_data = []
        race_names = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix']
        
        for race_name in race_names:
            try:
                session = fastf1.get_session(2024, race_name, 'R')
                session.load()
                sessions_data.append((session, race_name))
                print(f"✓ Loaded {race_name}")
            except Exception as e:
                print(f"✗ Could not load {race_name}: {e}")
        
        if sessions_data:
            # Extract comprehensive features 
            print(f"\nExtracting comprehensive features for {len(sessions_data)} races...")
            
            # Test with a subset of drivers first
            test_drivers = ['VER', 'HAM', 'LEC', 'RUS', 'NOR']  # Top drivers
            
            comprehensive_df = extract_comprehensive_driver_features(
                sessions_data, 
                drivers=test_drivers,
                include_race_identifiers=True
            )
            
            print(f"\n=== RESULTS ===")
            print(f"Shape: {comprehensive_df.shape}")
            print(f"Columns: {comprehensive_df.columns.tolist()[:10]}...")  # Show first 10 columns
            print(f"\nFirst few rows:")
            print(comprehensive_df[['Driver', 'Number_of_Races']].head())
            
            # Save results
            output_file = 'comprehensive_driver_features_sample.csv'
            comprehensive_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved to {output_file}")
            
            # Show feature structure for one driver
            if not comprehensive_df.empty:
                sample_driver = comprehensive_df['Driver'].iloc[0]
                print(f"\n=== Feature Structure for {sample_driver} ===")
                
                # Count features by type
                track_features = [col for col in comprehensive_df.columns if 'TrackAnalysis' in col]
                corner_features = [col for col in comprehensive_df.columns if 'Corner_' in col]
                
                print(f"Track Analysis Features: {len(track_features)}")
                print(f"Corner Features: {len(corner_features)}")
                print(f"Total Racing Features: {len(track_features) + len(corner_features)}")
                print(f"Features per race: {(len(track_features) + len(corner_features)) // len(sessions_data)}")
                
                # Show sample feature names
                print(f"\nSample Track Features:")
                for feat in track_features[:5]:  
                    print(f"  - {feat}")
                    
                print(f"\nSample Corner Features:")  
                for feat in corner_features[:5]:
                    print(f"  - {feat}")
                    
        else:
            print("No sessions could be loaded")
            
    except Exception as e:
        print(f"Error in comprehensive feature extraction: {e}")
        
    # Example 2: Single race track analysis (original functionality)
    print(f"\n=== Example: Single Race Track Analysis (Original) ===")
    try:
        monza = fastf1.get_session(2024, 'Monza', 'R')
        monza.load()
        df = extract_driver_track_analysis(monza, 'VER')
        print("Track Analysis for VER at Monza:")
        print(df)
    except Exception as e:
        print(f"Error in single race analysis: {e}")
