#!/usr/bin/env python3
"""
F1 Telemetry Data Extraction Pipeline - Main Script

This script provides a command-line interface for extracting F1 telemetry data
using the Pipeline module. It supports various extraction modes including single
race analysis, season-wide analysis, and corner-specific telemetry extraction.

The pipeline uses spline-based distance alignment to ensure consistent driver
comparisons by mapping all telemetry data to a common reference frame.
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd
import fastf1

# Add Pipeline module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline'))

from Pipeline import (
    extract_season_driver_features,
    extract_corner_telemetry_season,
    get_all_car_data,
    save_processed_data,
    extract_driver_track_analysis
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_fastf1_cache():
    """Setup FastF1 cache directory."""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    fastf1.logger.set_log_level(logging.CRITICAL)

def get_available_races(year: int) -> List[str]:
    """Get available race names for a given year."""
    try:
        schedule = fastf1.events.get_event_schedule(year)
        return schedule['EventName'].tolist()
    except Exception as e:
        logging.error(f"Error getting race schedule for {year}: {e}")
        return []

def get_session_drivers(year: int, race: str) -> List[str]:
    """Get available drivers for a specific session."""
    try:
        session = fastf1.get_session(year, race, 'R')
        session.load()
        return list(session.drivers)
    except Exception as e:
        logging.error(f"Error getting drivers for {year} {race}: {e}")
        return []

def extract_single_race_data(year: int, race: str, drivers: List[str], output_dir: str):
    """Extract comprehensive data for a single race."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n=== Extracting Single Race Data ===")
    print(f"Race: {year} {race}")
    print(f"Drivers: {', '.join(drivers)}")
    
    try:
        session = fastf1.get_session(year, race, 'R')
        session.load()
        
        all_data = []
        
        for driver in drivers:
            print(f"Processing driver: {driver}")
            try:
                # Extract comprehensive track analysis
                driver_data = extract_driver_track_analysis(session, driver)
                if not driver_data.empty:
                    driver_data['Year'] = year
                    driver_data['Race'] = race
                    all_data.append(driver_data)
                    print(f"  ✓ Successfully extracted data for {driver}")
                else:
                    print(f"  ⚠ No data found for {driver}")
            except Exception as e:
                print(f"  ✗ Error processing {driver}: {e}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            output_file = os.path.join(output_dir, f"single_race_{year}_{race.replace(' ', '_')}_{timestamp}.csv")
            save_processed_data(combined_data, output_file)
            print(f"\n✓ Single race data saved to: {output_file}")
            print(f"  - Total records: {len(combined_data)}")
            print(f"  - Features: {len(combined_data.columns)}")
        else:
            print("\n⚠ No data extracted for any driver")
            
    except Exception as e:
        print(f"\n✗ Error in single race extraction: {e}")

def extract_season_data(year: int, races: Optional[List[str]], drivers: Optional[List[str]], 
                       max_races: Optional[int], output_dir: str):
    """Extract season-wide comprehensive driver features."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n=== Extracting Season Data ===")
    print(f"Year: {year}")
    if races:
        print(f"Specific races: {', '.join(races)}")
    if drivers:
        print(f"Specific drivers: {', '.join(drivers)}")
    if max_races:
        print(f"Max races: {max_races}")
    
    try:
        # Extract comprehensive season features
        season_data = extract_season_driver_features(
            year=year,
            race_names=races,
            drivers=drivers,
            max_races=max_races,
            include_testing=False
        )
        
        if not season_data.empty:
            output_file = os.path.join(output_dir, f"season_features_{year}_{timestamp}.csv")
            save_processed_data(season_data, output_file)
            print(f"\n✓ Season data saved to: {output_file}")
            print(f"  - Total records: {len(season_data)}")
            print(f"  - Features: {len(season_data.columns)}")
            print(f"  - Unique drivers: {season_data['Driver'].nunique()}")
        else:
            print("\n⚠ No season data extracted")
            
    except Exception as e:
        print(f"\n✗ Error in season extraction: {e}")

def extract_corner_telemetry_data(year: int, drivers: Optional[List[str]], output_dir: str):
    """Extract corner-specific telemetry data for the season."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n=== Extracting Corner Telemetry Data ===")
    print(f"Year: {year}")
    if drivers:
        print(f"Specific drivers: {', '.join(drivers)}")
    
    try:
        # Extract corner telemetry data
        corner_data = extract_corner_telemetry_season(
            season=year,
            include_testing=False,
            drivers=drivers,
            clustering_compatible=True,
            max_workers=4
        )
        
        if not corner_data.empty:
            output_file = os.path.join(output_dir, f"corner_telemetry_{year}_{timestamp}.csv")
            save_processed_data(corner_data, output_file)
            print(f"\n✓ Corner telemetry data saved to: {output_file}")
            print(f"  - Total records: {len(corner_data)}")
            print(f"  - Features: {len(corner_data.columns)}")
            print(f"  - Unique drivers: {corner_data['Driver'].nunique()}")
        else:
            print("\n⚠ No corner telemetry data extracted")
            
    except Exception as e:
        print(f"\n✗ Error in corner telemetry extraction: {e}")

def extract_raw_telemetry(year: int, race: str, drivers: List[str], output_dir: str):
    """Extract raw telemetry data for specific drivers in a race."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n=== Extracting Raw Telemetry Data ===")
    print(f"Race: {year} {race}")
    print(f"Drivers: {', '.join(drivers)}")
    
    try:
        session = fastf1.get_session(year, race, 'R')
        session.load()
        
        all_telemetry = []
        
        for driver in drivers:
            print(f"Processing driver: {driver}")
            try:
                telemetry_data = get_all_car_data(session, driver)
                if not telemetry_data.empty:
                    telemetry_data['Year'] = year
                    telemetry_data['Race'] = race
                    all_telemetry.append(telemetry_data)
                    print(f"  ✓ Successfully extracted {len(telemetry_data)} telemetry points for {driver}")
                else:
                    print(f"  ⚠ No telemetry data found for {driver}")
            except Exception as e:
                print(f"  ✗ Error processing {driver}: {e}")
        
        if all_telemetry:
            combined_telemetry = pd.concat(all_telemetry, ignore_index=True)
            output_file = os.path.join(output_dir, f"raw_telemetry_{year}_{race.replace(' ', '_')}_{timestamp}.csv")
            save_processed_data(combined_telemetry, output_file)
            print(f"\n✓ Raw telemetry data saved to: {output_file}")
            print(f"  - Total telemetry points: {len(combined_telemetry)}")
            print(f"  - Features: {len(combined_telemetry.columns)}")
        else:
            print("\n⚠ No telemetry data extracted for any driver")
            
    except Exception as e:
        print(f"\n✗ Error in raw telemetry extraction: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="F1 Telemetry Data Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract single race data for specific drivers
  python main.py --mode single --year 2024 --race "Monaco" --drivers VER,HAM,LEC

  # Extract season features for all drivers
  python main.py --mode season --year 2024

  # Extract season features for specific drivers and races
  python main.py --mode season --year 2024 --drivers VER,HAM --races "Monaco,Silverstone" --max-races 5

  # Extract corner telemetry for the season
  python main.py --mode corner --year 2024 --drivers VER,HAM,LEC

  # Extract raw telemetry data
  python main.py --mode raw --year 2024 --race "Monza" --drivers VER,HAM

  # List available races for a year
  python main.py --list-races --year 2024

  # List available drivers for a specific race
  python main.py --list-drivers --year 2024 --race "Monaco"
        """
    )
    
    # Main arguments
    parser.add_argument('--mode', choices=['single', 'season', 'corner', 'raw'], 
                       help='Extraction mode')
    parser.add_argument('--year', type=int, required=True,
                       help='Season year (e.g., 2024)')
    parser.add_argument('--race', type=str,
                       help='Race name (e.g., "Monaco", "Silverstone")')
    parser.add_argument('--races', type=str,
                       help='Comma-separated list of race names')
    parser.add_argument('--drivers', type=str,
                       help='Comma-separated list of driver codes (e.g., "VER,HAM,LEC")')
    parser.add_argument('--max-races', type=int,
                       help='Maximum number of races to process (for season mode)')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for CSV files (default: ./output)')
    
    # Utility arguments
    parser.add_argument('--list-races', action='store_true',
                       help='List available races for the specified year')
    parser.add_argument('--list-drivers', action='store_true',
                       help='List available drivers for the specified race')
    
    args = parser.parse_args()
    
    # Setup
    setup_fastf1_cache()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Utility functions
    if args.list_races:
        print(f"\nAvailable races for {args.year}:")
        races = get_available_races(args.year)
        for i, race in enumerate(races, 1):
            print(f"  {i:2d}. {race}")
        return
    
    if args.list_drivers:
        if not args.race:
            print("Error: --race is required when using --list-drivers")
            return
        print(f"\nAvailable drivers for {args.year} {args.race}:")
        drivers = get_session_drivers(args.year, args.race)
        for i, driver in enumerate(drivers, 1):
            print(f"  {i:2d}. {driver}")
        return
    
    # Validate mode argument
    if not args.mode:
        print("Error: --mode is required. Use -h for help.")
        return
    
    # Parse driver and race lists
    driver_list = args.drivers.split(',') if args.drivers else None
    race_list = args.races.split(',') if args.races else None
    
    # Execute based on mode
    print(f"F1 Telemetry Data Extraction Pipeline")
    print(f"=====================================")
    
    if args.mode == 'single':
        if not args.race:
            print("Error: --race is required for single race mode")
            return
        if not driver_list:
            print("Error: --drivers is required for single race mode")
            return
        extract_single_race_data(args.year, args.race, driver_list, args.output_dir)
    
    elif args.mode == 'season':
        extract_season_data(args.year, race_list, driver_list, args.max_races, args.output_dir)
    
    elif args.mode == 'corner':
        extract_corner_telemetry_data(args.year, driver_list, args.output_dir)
    
    elif args.mode == 'raw':
        if not args.race:
            print("Error: --race is required for raw telemetry mode")
            return
        if not driver_list:
            print("Error: --drivers is required for raw telemetry mode")
            return
        extract_raw_telemetry(args.year, args.race, driver_list, args.output_dir)
    
    print(f"\nExtraction complete!")

if __name__ == "__main__":
    main() 