"""Functions and classes for the health study analysis (Part 2)."""

import numpy as np
import pandas as pd

def load_data(path):
    """Read the health study CSV file and return a DataFrame."""
    return pd.read_csv(path)

def summary_stats(df, columns):
    """Return mean, median, min and max for the given columns."""
    return df[columns].agg(['mean', 'median', 'min', 'max']).T

def normal_ci(values, z=1.96):
    """Confidence interval for the mean with normal approximation.
    Returns (low, high). z=1.96 gives a 95% interval."""
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean - z * se, mean + z * se

def bootstrap_ci(values, n_boot=10000, level=95):
    """Confidence interval for the mean with bootstrap.
    Returns (low, high). level=95 gives a 95% interval.
    """
    boot_means = []
    for _ in range(n_boot):
        # Sample with replacement, same size as the data
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())

    # Cut the same share at each end, 2.5% each for a 95% interval
    cut = (100 - level) / 2
    low, high = np.percentile(boot_means, [cut, 100 - cut])
    return low, high