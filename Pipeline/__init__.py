"""
Pipeline module for F1 telemetry data processing.

This module provides comprehensive telemetry data extraction and processing
for Formula 1 telemetry analysis using spline-based distance calculations.
"""

# Core telemetry extraction functions
from .telemetry_extraction import get_all_car_data

# Data aggregation functions
from .data_aggregation import (
    calculate_lap_gear_changes,
    aggregation_function,
    aggregate_section_data
)

# Corner analysis functions
from .corner_analysis import (
    classify_corners_by_speed,
    extract_corner_telemetry_sections,
    compare_corner_sections
)

# Multi-driver analysis functions
from .multi_driver_analysis import (
    extract_all_drivers_corner_data,
    get_corner_performance_view
)

# Track analysis functions
from .track_analysis import (
    extract_driver_track_analysis,
    extract_comprehensive_driver_features,
    extract_season_driver_features
)

# Session-level analysis functions
from .session_analysis import (
    analyze_session_corner_performance,
    create_clustering_compatible_summary,
    extract_corner_telemetry_season
)

# Data I/O functions
from .data_io import save_processed_data

__all__ = [
    # Telemetry extraction
    'get_all_car_data',
    
    # Data aggregation
    'calculate_lap_gear_changes',
    'aggregation_function',
    'aggregate_section_data',
    
    # Corner analysis
    'classify_corners_by_speed',
    'extract_corner_telemetry_sections',
    'compare_corner_sections',
    
    # Multi-driver analysis
    'extract_all_drivers_corner_data',
    'get_corner_performance_view',
    
    # Track analysis
    'extract_driver_track_analysis',
    'extract_comprehensive_driver_features',
    'extract_season_driver_features',
    
    # Session analysis
    'analyze_session_corner_performance',
    'create_clustering_compatible_summary',
    'extract_corner_telemetry_season',
    
    # Data I/O
    'save_processed_data'
]