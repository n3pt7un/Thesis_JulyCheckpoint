"""
Data aggregation functions for telemetry processing.
"""

import pandas as pd
import numpy as np


def calculate_lap_gear_changes(gear_series):
    """
    Calculates the number of gear changes for a given pandas Series
    (representing one lap's gear data).
    Assumes the series is ordered chronologically for the lap.
    """
    # Ensure we're working with a clean Series index within the group
    gear_series = gear_series.reset_index(drop=True)

    if gear_series.shape[0] < 2:
        # Need at least two data points to potentially have a change
        return 0

    # Identify points where gear is different from the previous point
    is_change = gear_series.ne(gear_series.shift())

    # Assign a group ID to each block of consecutive gears
    # The first block starts with ID 1
    gear_block_groups = is_change.cumsum()

    # Count the number of distinct gear blocks
    num_blocks = gear_block_groups.nunique()

    # The number of changes is the number of blocks minus 1
    # If num_blocks is 1, gear was constant (or only one data point), so 0 changes.
    num_changes = max(0, num_blocks - 1)

    return num_changes


def aggregation_function(telemetry):
    """
    Aggregates telemetry data by lap number.
    Args:
        telemetry (pd.DataFrame): Telemetry data.
    Returns:
        pd.DataFrame: Aggregated telemetry data.
        AvgSpeed: Average speed of the lap.
        TopSpeed: Maximum speed of the lap.
        AvgGear: Average gear of the lap.
        GearChanges: Number of gear changes of the lap.
        AvgThrottleIntensity: Average throttle intensity of the lap.
        MaxDeceleration: Maximum deceleration of the lap.
        MaxAcceleration: Maximum acceleration of the lap.
    """
    telemetry_sorted = telemetry.sort_values(by=['LapNumber', 'Time']) 
    return telemetry_sorted.groupby('LapNumber').agg(
        AvgSpeed = ('Speed', 'mean'),
        TopSpeed = ('Speed', 'max'),
        AvgGear = ('nGear', 'mean'),
        GearChanges = ('nGear', calculate_lap_gear_changes),
        AvgThrottleIntensity = ('Throttle', 'mean'),
        MaxDeceleration = ('Acceleration(m/s^2)', 'min'),
        MaxAcceleration = ('Acceleration(m/s^2)', 'max'), 
        AvgAcceleration = ('Acceleration(m/s^2)', 'mean'),
        Position = ('Position', 'min')
    ).reset_index()


def aggregate_section_data(section_data: pd.DataFrame) -> dict:
    """Aggregate telemetry for a corner section using 's_lap' and 'DateTime'."""
    if section_data.empty: return None
    gc = calculate_lap_gear_changes(section_data['nGear'])
    return {
        'AvgSpeed':section_data['Speed'].mean(),
        'TopSpeed':section_data['Speed'].max(),
        'MinSpeed':section_data['Speed'].min(),
        'EntrySpeed':section_data['Speed'].iloc[0],
        'ExitSpeed':section_data['Speed'].iloc[-1],
        'AvgGear':section_data['nGear'].mean(),
        'GearChanges':gc,
        'AvgThrottleIntensity':section_data['Throttle'].mean(),
        'MaxThrottle':section_data['Throttle'].max(),
        'MinThrottle':section_data['Throttle'].min(),
        'MaxDeceleration':section_data['Acceleration(m/s^2)'].min(),
        'MaxAcceleration':section_data['Acceleration(m/s^2)'].max(),
        'AvgAcceleration':section_data['Acceleration(m/s^2)'].mean(),
        'DataPoints':len(section_data),
        'DistanceCovered':section_data['s_lap'].max()-section_data['s_lap'].min(),
        'TimeDuration':(
            section_data['Time'].max()-section_data['Time'].min()
        ).total_seconds()
    }