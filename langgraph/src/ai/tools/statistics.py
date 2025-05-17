# src/ai/tools/statistics.py
import statistics
from typing import List

def summarize_statistics(data: List[float]) -> str:
    """
    Returns descriptive statistics (mean, median, mode, std dev) for a list of numbers.

    Args:
        data: List of numbers.

    Returns:
        A string summary of the main descriptive statistics.
    """
    try:
        return (
            f"Mean: {statistics.mean(data):.2f}, "
            f"Median: {statistics.median(data):.2f}, "
            f"Mode: {statistics.mode(data):.2f}, "
            f"Standard Deviation: {statistics.stdev(data):.2f}, "
            f"Variance: {statistics.variance(data):.2f}"
        )
    except Exception as e:
        return f"Error computing statistics: {e}"
