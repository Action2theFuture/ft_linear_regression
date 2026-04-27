import numpy as np
import os
import argparse
from typing import Tuple

# Color constants for consistent logging
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def csv_file_validator(filename: str) -> str:
    """Checks if the file has a .csv extension for argparse."""
    if not filename.lower().endswith('.csv'):
        raise argparse.ArgumentTypeError(f"File '{filename}' must be a .csv file.")
    return filename

def load_csv(directory: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads raw mileage and price data from a CSV file."""
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        print(f"{RED}❌ Error: File '{filepath}' not found.{RESET}")
        return None, None

    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data.ndim < 2 or data.shape[0] < 2:
            print(f"{RED}❌ Error: Insufficient data in '{filename}'.{RESET}")
            return None, None
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"{RED}❌ Error: Failed to read file ({e}){RESET}")
        return None, None

def normalize_features(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Performs Min-Max Scaling and returns normalized data with scaling info."""
    min_val = np.min(x)
    max_val = np.max(x)
    denom = max_val - min_val if max_val != min_val else 1.0
    return (x - min_val) / denom, min_val, max_val