"""
Corner analysis and classification functions.
"""

import fastf1
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import logging
from typing import List, Optional, Dict

try:
    from .telemetry_extraction import get_all_car_data
    from .data_aggregation import aggregate_section_data
except ImportError:
    from telemetry_extraction import get_all_car_data
    from data_aggregation import aggregate_section_data


def classify_corners_by_speed(
    session: fastf1.core.Session,
    speed_col: str = "Speed",
    window: float = 25.0
) -> pd.DataFrame:
    """
    Classify corners by speed using spline-based distances.

    Returns a DataFrame with:
      - ApexDistance: spline arc-length of each apex
      - MinSpeed: minimum speed within window around apex
      - Class: speed-based classification (Fast/Medium/Slow)
    """
    # 1) Get circuit corners (no angle classification needed)
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners.copy()

    # Build a reference spline from the fastest lap for consistent distance mapping
    ref_lap = session.laps.pick_fastest()
    ref_tel = get_all_car_data(session, ref_lap.Driver) # Use our get_all_car_data for consistency

    # Create KD-Tree for snapping corner coordinates to spline distance
    # Ensure ref_tel has 'spline_dist', 'X_snap', 'Y_snap' from get_all_car_data
    if 'spline_dist' not in ref_tel.columns or ref_tel.empty:
        logging.error("Reference telemetry missing spline_dist. Cannot classify corners.")
        return pd.DataFrame()
        
    track_tree = cKDTree(np.column_stack((ref_tel['X_snap'], ref_tel['Y_snap'])))
    u_fine = ref_tel['spline_dist'].values # Use the spline_dist as the u_fine for consistency

    # Project apex coordinates to spline distance
    # Use the original X, Y from FastF1 corners and snap them to our spline
    corners['ApexDistance'] = corners.apply(
        lambda r: u_fine[track_tree.query([r['X'], r['Y']])[1]], axis=1
    )

    # Get telemetry snapped to spline for the fastest driver
    fastest_driver_tel = get_all_car_data(session, ref_lap.Driver)

    # Compute MinSpeed around each ApexDistance
    records = []
    for _, row in corners.iterrows():
        s0 = row['ApexDistance']
        
        # Use s_lap for filtering as it's lap-relative spline distance
        mask = (fastest_driver_tel['s_lap'] >= s0 - window) & (fastest_driver_tel['s_lap'] <= s0 + window)
        seg = fastest_driver_tel.loc[mask]
        
        min_spd = seg[speed_col].min() if not seg.empty else np.nan
        
        records.append({
            'ApexDistance': s0,
            'MinSpeed': min_spd
        })
    df = pd.DataFrame(records)

    # Speed quantile classification
    speeds = df['MinSpeed'].dropna()
    if not speeds.empty:
        q1, q2 = np.percentile(speeds, [33, 66])
        df['Class'] = df['MinSpeed'].apply(
            lambda s: 'Slow' if s < q1 else ('Medium' if s < q2 else 'Fast')
        )
    else:
        df['Class'] = 'Unknown' # Default if no speeds are available

    return df


def extract_corner_telemetry_sections(
    session: fastf1.core.Session,
    driver: str,
    distance_before: float = 100.0,
    distance_after: float = 100.0,
    selected_corners: list = None,
    corner_selection_method: str = 'default'
) -> dict:
    """
    Extract telemetry sections before/after corners using 's_lap'.
    """
    tel = get_all_car_data(session, driver)
    corners = classify_corners_by_speed(session)
    # Select corners
    if selected_corners is None:
        if corner_selection_method == 'default':
            vc = corners.dropna(subset=['MinSpeed'])
            if len(vc) >= 3:
                sc = vc.sort_values('MinSpeed')
                sel = [sc.index[0], sc.index[len(sc)//2], sc.index[-1]]
            else:
                sel = vc.index.tolist()
        elif corner_selection_method == 'all':
            sel = corners.index.tolist()
        else:
            raise ValueError("corner_selection_method must be 'default' or 'all'")
    else:
        sel = selected_corners
    results = {}
    for ci in sel:
        info = corners.loc[ci]
        s0 = info['ApexDistance']
        into_frames, out_frames = [], []
        for lap_num in tel['LapNumber'].unique():
            df_lap = tel[tel['LapNumber'] == lap_num]
            # Into turn
            into = df_lap[(df_lap['s_lap'] >= s0 - distance_before) & (df_lap['s_lap'] < s0)]
            if not into.empty:
                into = into.copy()
                into.loc[:, 'Section'] = 'into_turn'
                into.loc[:, 'CornerIndex'] = ci
                into_frames.append(into)
            # Out of turn
            out = df_lap[(df_lap['s_lap'] >= s0) & (df_lap['s_lap'] <= s0 + distance_after)]
            if not out.empty:
                out = out.copy()
                out.loc[:, 'Section'] = 'out_of_turn'
                out.loc[:, 'CornerIndex'] = ci
                out_frames.append(out)
        into_agg = aggregate_section_data(pd.concat(into_frames, ignore_index=True)) if into_frames else None
        out_agg = aggregate_section_data(pd.concat(out_frames, ignore_index=True)) if out_frames else None
        results[f'corner_{ci}'] = {
            'corner_info': {
                'apex_distance': s0,
                'min_speed': info['MinSpeed'],
                'speed_class': info['Class']
            },
            'into_turn': into_agg,
            'out_of_turn': out_agg
        }
    return results


def compare_corner_sections(corner_results: dict, corner_name: str = None) -> pd.DataFrame:
    """Compare into vs out section metrics per corner."""
    rows=[]
    keys=[corner_name] if corner_name else corner_results.keys()
    for k in keys:
        if k not in corner_results: continue
        ci = corner_results[k]['corner_info']
        for sec in ['into_turn','out_of_turn']:
            d=corner_results[k][sec]
            if d:
                row = d.copy()
                row.update({'Corner':k,'Section':sec,
                            'SpeedClass':ci['speed_class']})
                rows.append(row)
    return pd.DataFrame(rows)