"""
Data input/output and file management functions.
"""

import pandas as pd
import logging


def save_processed_data(
    data: pd.DataFrame,
    file_path: str
):
    """
    Saves the processed data to a CSV file.

    Parameters:
    -----------
    data : pd.DataFrame
        The data to save.
    file_path : str
        The path to the output file.
    """
    try:
        data.to_csv(file_path, index=False)
        logging.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")