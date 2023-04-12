import os

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats

from src.environment import FIGURES_DIR


def test_normality(x, group=None, ax=None, figsize=(5, 5)):
    result = stats.shapiro(x)
    if group is None:
        print(f"Shapiro test for normality: W = {result[0]} and p-value {result[1]}")
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        sm.qqplot(x, line="q", ax=ax)
    else:
        print(f"Shapiro test for normality of group {group}: W = {result[0]} and p-value {result[1]}")
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        sm.qqplot(x, line="q", ax=ax)
        ax.set_title(f"Q-Q plot for group {group}")
    return result


def test_assumptions(*args, nrows=1, ncols=1, figsize=(5, 5)):
    if len(args) < 2:
        return test_normality(np.asarray(args[0]), figsize=figsize)
    args = [np.asarray(arg) for arg in args]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    for i, arg in enumerate(args):
        if arg.ndim != 1:
            raise ValueError("Input samples must be one-dimensional.")
        if arg.size <= 1:
            raise ValueError("Input sample size must be greater than one.")
        if np.isinf(arg).any():
            raise ValueError("Input samples must be finite.")
        test_normality(arg, i, axes[i])

    result = stats.levene(*args)
    print(f"Levene test for equal variances: W = {result[0]} and p-value = {result[1]}")


def eta_squared(H, k, n):
    """
    Compute the eta-squared measure for the Kruskal-Wallis H-test.
    :param H: The value obtained in the Kruskal-Wallis test.
    :param k: The number of groups.
    :param n: The total number of samples.
    :return: The eta-squared estimate.
    """
    return (H - k + 1) / (n - k)


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = (costs[:, 0] <= costs[next_point_index, 0]) | (
            costs[:, 1] >= costs[next_point_index, 1]
        )
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def boxplot(
    data,
    x: str = None,
    y: str = None,
    title=None,
    xlabel=None,
    ylabel=None,
    figname=None,
    figsize=(5, 5),
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.boxplot(data, x=x, y=y, ax=ax)
    ax.yaxis.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if figname is not None:
        plt.savefig(os.path.join(FIGURES_DIR, figname))


def barplot(
    data,
    x=None,
    y=None,
    xlabel=None,
    ylabel=None,
    title=None,
    hue=None,
    hue_order=None,
    errorbar=None,
    estimator="mean",
    figname=None,
    figsize=(5, 5),
    barlabel=False,
    ax=None,
):
    if barlabel and errorbar is not None:
        raise Warning(
            "Setting using the errorbar and barlabel parameters will produce overlapping labels. Please use one or the other."
        )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    ax = sns.barplot(
        data,
        x=x,
        y=y,
        ax=ax,
        hue=hue,
        hue_order=hue_order,
        errorbar=errorbar,
        errwidth=3,
        estimator=estimator,
    )
    ax.yaxis.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if barlabel:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

    if figname is not None:
        plt.savefig(os.path.join(FIGURES_DIR, figname))
    return fig, ax


def print_improvement(dataframe, metric):
    median_values = dataframe.groupby(["architecture", "training environment"], as_index=False)[metric].median()
    median_values["combination"] = median_values.architecture + " - " + median_values["training environment"]
    tmp = pd.DataFrame(columns=median_values.combination.unique(), index=median_values.combination.unique())

    "Iterate over all combinations of architectures and training environments and calculate the relative difference in energy consumption. Then get the maximum and minimum values."
    for i, row in median_values.iterrows():
        for j, row2 in median_values.iterrows():
            if i != j:
                tmp.loc[row.combination, row2.combination] = (row[metric] - row2[metric]) / row[metric]
            else:
                tmp.loc[row.combination, row2.combination] = 0.0

    tmp = tmp.astype(float)

    "Get the rows and columns with maximum and minimum values of the relative difference in energy consumption."
    max_row = tmp.max(axis=1).idxmax()
    max_col = tmp.max(axis=0).idxmax()
    min_row = tmp[tmp > 0].min(axis=1).idxmin()
    min_col = tmp[tmp > 0].min(axis=0).idxmin()

    print(f"Maximum improvement: {tmp.loc[max_row, max_col]} ({max_row}, {max_col})")
    print(f"Minimum improvement: {tmp.loc[min_row, min_col]} ({min_row}, {min_col})")
