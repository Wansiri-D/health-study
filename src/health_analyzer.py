"""Functions and classes for the health study analysis (Part 2)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    """Read the health study CSV file and return a DataFrame."""
    return pd.read_csv(path)


def summary_stats(df, columns):
    """Return mean, median, min and max for the given columns."""
    return df[columns].agg(["mean", "median", "min", "max"]).T


def normal_ci(values, z=1.96):
    """Confidence interval for the mean with normal approximation.

    Returns (low, high). z=1.96 gives a 95% interval.
    """
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean - z * se, mean + z * se


def bootstrap_ci(values, n_boot=10000, level=95):
    """Confidence interval for the mean with bootstrap.
    Returns (low, high). level=95 gives a 95% interval."""
    boot_means = []
    for _ in range(n_boot):
        # Sample with replacement, same size as the data
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())

    # Cut the same share at each end, 2.5% each for a 95% interval
    cut = (100 - level) / 2
    low, high = np.percentile(boot_means, [cut, 100 - cut])
    return low, high


def t_statistic(group_a, group_b):
    """Welch t-statistic for the difference in means (group_a - group_b)."""
    diff = np.mean(group_a) - np.mean(group_b)
    # ddof=1 divides by n-1, so pandas and numpy give the same result
    se = np.sqrt(np.var(group_a, ddof=1) / len(group_a) +
                 np.var(group_b, ddof=1) / len(group_b))
    return diff / se


def permutation_test(group_a, group_b, n_perm=10000):
    """One-sided permutation test. Returns the p-value for mean(a) > mean(b)."""
    observed = np.mean(group_a) - np.mean(group_b)
    all_values = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count = 0
    for _ in range(n_perm):
        # Shuffle all values and split into two new groups with the same sizes
        shuffled = np.random.permutation(all_values)
        diff = shuffled[:n_a].mean() - shuffled[n_a:].mean()
        if diff >= observed:
            count += 1
    return count / n_perm


def simulate_power(true_diff, n_a, n_b, sd, n_sim=1000):
    """Returns how often the t-test finds a difference when the true difference is true_diff."""
    rejections = 0
    for _ in range(n_sim):
        # Fake data: group a is true_diff higher than group b
        fake_a = np.random.normal(loc=true_diff, scale=sd, size=n_a)
        fake_b = np.random.normal(loc=0, scale=sd, size=n_b)
        # If t is above 1.645, the test found the difference (one-sided, alpha = 0.05)
        if t_statistic(fake_a, fake_b) > 1.645:
            rejections += 1
    return rejections / n_sim


def design_matrix(X):
    """Add a column of ones to X, so the model gets an intercept."""
    return np.column_stack([np.ones(len(X)), X])


def linear_regression(X, y):
    """Fit y = b0 + b1*x1 + ... with least squares.
    Returns the coefficients [b0, b1, ...], where b0 is the intercept."""
    X_design = design_matrix(X)
    # lstsq finds the coefficients that make the squared errors as small as possible
    coef = np.linalg.lstsq(X_design, y, rcond=None)[0]
    return coef


def predict(X, coef):
    """Return predicted values for X with the given coefficients."""
    return design_matrix(X) @ coef


def r_squared(y, y_pred):
    """Share of the variation in y that the model explains (0 to 1)."""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


class HealthAnalyzer:
    """Runs statistics and graphs on the health study data.
    The data is stored in the object, so I do not need to send it
    to every function again."""

    def __init__(self, df):
        """Save the DataFrame in the object."""
        self.df = df

    def describe(self, columns):
        """Return mean, median, min and max for the given columns."""
        return summary_stats(self.df, columns)

    def confidence_interval(self, column, method="normal"):
        """95% confidence interval for the mean of a column.
        method can be "normal" or "bootstrap"."""
        values = self.df[column]
        if method == "normal":
            return normal_ci(values)
        elif method == "bootstrap":
            return bootstrap_ci(values)
        else:
            raise ValueError("method must be 'normal' or 'bootstrap'")

    def plot_histogram(self, column, bins=30):
        """Draw a histogram of a column with a line for the mean."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.df[column], bins=bins, edgecolor="black")
        ax.axvline(self.df[column].mean(), color="red", linestyle="--", label="Mean")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Number of people")
        ax.legend()
        return ax

    def regression(self, target, features):
        """Linear regression of target on one or more features.
        Returns (coefficients, r2). coefficients[0] is the intercept."""
        X = self.df[features].to_numpy()
        y = self.df[target].to_numpy()
        coef = linear_regression(X, y)
        r2 = r_squared(y, predict(X, coef))
        return coef, r2

    def disease_rate_by(self, group_col):
        """Share of people with the disease (in %) for each group."""
        return self.df.groupby(group_col, observed=True)["disease"].mean() * 100

    def plot_boxplot(self, column, by):
        """Draw one box of a column for each group in another column."""
        names = []
        data = []
        for name, group in self.df.groupby(by, observed=True):
            names.append(str(name))
            data.append(group[column])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data)
        ax.set_xticklabels(names)
        ax.set_title(f"{column} by {by}")
        ax.set_xlabel(by)
        ax.set_ylabel(column)
        return ax