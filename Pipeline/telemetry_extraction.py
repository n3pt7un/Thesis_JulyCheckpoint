"""
Telemetry data extraction functions for F1 data processing.
"""

import fastf1
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import logging
from typing import List, Optional, Dict

# Enable FastF1 cache
import os
cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data_processing', 'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.logger.set_log_level(logging.CRITICAL)
fastf1.Cache.enable_cache(cache_dir)

def get_all_car_data(session: fastf1.core.Session, driver: str) -> pd.DataFrame:
    """
    Extracts telemetry and position data for a given driver,
    aligns it to a racing-line spline per lap using DateTime, and returns a combined DataFrame.

    Process:
      1. Build a cubic spline of the racing line from the session's fastest lap.
      2. Sample the spline densely and build a KD-Tree for nearest-point snapping.
      3. Loop through each lap of the given driver:
         - Merge car and position channels.
         - Interpolate X/Y by DateTime.
         - Snap each (X,Y) to the spline to assign a spline-based distance.
         - Reset spline distance at the start of the lap to get lap-relative distance ('s_lap').
         - Compute raw and snapped coordinates and acceleration.
      4. Concatenate all laps into one DataFrame.

    Returns:
        DataFrame with columns including:
          - spline_dist: absolute spline distance along track
          - s_lap: spline distance reset to zero at lap start
          - raw_distance: original Distance channel
          - X_snap, Y_snap: snapped coordinates on spline
          - Acceleration(m/s^2)
          - LapNumber, Driver, Session, Position, InPit, OutPit, TyreCompound, TyreLife
    """
    # ----------------------------------------------------------------
    # 1) Build reference spline once, from the session's fastest lap
    # ----------------------------------------------------------------
    reference_lap = session.laps.pick_fastest()
    car_data = reference_lap.get_car_data()
    pos_data = reference_lap.get_pos_data()
    ref_merged = car_data.merge_channels(pos_data).add_distance()

    # Use Time column for temporal operations
    ref_merged = ref_merged.set_index('Time')
    ref_merged[['X', 'Y']] = ref_merged[['X', 'Y']].interpolate(method='time').ffill().bfill()
    ref_merged = ref_merged.reset_index()

    # Reference arrays for spline
    u_ref = ref_merged['Distance'].values
    x_ref = ref_merged['X'].values
    y_ref = ref_merged['Y'].values

    # Fit a periodic cubic spline through (X,Y)
    tck, _ = splprep([x_ref, y_ref], u=u_ref, s=0, k=3)

    # Sample the spline at 0.1 m resolution
    u_fine = np.linspace(0, u_ref.max(), int(u_ref.max() / 0.1))
    x_fine, y_fine = splev(u_fine, tck)

    # Build KD-Tree for snapping
    track_tree = cKDTree(np.column_stack((x_fine, y_fine)))

    # ----------------------------------------------------------------
    # 2) Loop over driver laps, merge data, snap to spline, reset per lap
    # ----------------------------------------------------------------
    driver_laps = (
        session.laps
               .pick_drivers(driver)
               .assign(
                   InPit=lambda df: df['PitOutTime'].notna(),
                   OutPit=lambda df: df['PitInTime'].notna()
               )
               .pick_accurate()
    )

    all_data = []
    for _, lap in driver_laps.iterrows():
        car = lap.get_car_data()
        pos = lap.get_pos_data()

        merged = car.merge_channels(pos).add_distance()
        merged['raw_distance'] = merged['Distance']

        # Interpolate X/Y by DateTime
        t = merged['Time'].dt.total_seconds()
        merged['X'] = merged['X'].interpolate(method='linear', x=t).ffill().bfill()
        merged['Y'] = merged['Y'].interpolate(method='linear', x=t).ffill().bfill()

        # Snap to spline
        pts = np.column_stack((merged['X'].values, merged['Y'].values))
        _, idx = track_tree.query(pts, k=1)
        merged['X_snap'] = x_fine[idx]
        merged['Y_snap'] = y_fine[idx]
        merged['spline_dist'] = u_fine[idx]

        # Reset per-lap spline distance
        start_s = merged['spline_dist'].min()
        merged['s_lap'] = merged['spline_dist'] - start_s

        # Annotate lap and driver info
        merged['LapNumber']    = int(lap.LapNumber)
        merged['Driver']       = driver
        merged['Session']      = session.name
        merged['Position']     = lap.Position
        merged['InPit']        = merged['LapNumber'].isin(
                                    driver_laps.loc[driver_laps['InPit'], 'LapNumber']
                                )
        merged['OutPit']       = merged['LapNumber'].isin(
                                    driver_laps.loc[driver_laps['OutPit'], 'LapNumber']
                                )
        merged['TyreCompound'] = lap.Compound
        merged['TyreLife']     = lap.TyreLife

        # Compute acceleration using DateTime
        secs = merged['Time'].dt.total_seconds().values
        merged['Acceleration(m/s^2)'] = np.gradient(merged['Speed'].values/3.6, secs)

        all_data.append(merged)

    return pd.concat(all_data, ignore_index=True)

monza = fastf1.get_session(2024, 'Monza', 'R')
monza.load()
df = get_all_car_data(monza, 'VER')
df.to_csv('monza_ver.csv')