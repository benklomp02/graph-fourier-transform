import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List

from statistics import mean
from tests.IO.graph import read_graph_input

compute_basis_t = Callable[[int, np.ndarray], np.ndarray]


def _input_filename(N: int, num_tests: int, directed: bool = False) -> str:
    filename = f"public/input/{"directed" if directed else "undirected"}/N{N}_t{num_tests}.txt".lower()
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
    return filename


def _saved_filename(title: str, num_tests: int, directed: bool = False):
    return f"public/output/{"directed" if directed else "undirected"}/{title.replace(" ", "_")}_t{num_tests}.png".lower()


def _saved_filename_log(title: str, num_tests: int, directed: bool = False):
    return f"public/output/{"directed" if directed else "undirected"}/{title.replace(" ", "_")}_t{num_tests}_log.png".lower()


def _plt_config(title: str, xlabel: str, ylabel: str, log: bool = False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if log:
        plt.yscale("log")


def compute_basis_placeholder(n: int, weights: np.ndarray) -> np.ndarray:
    """
    Placeholder function for computing the basis.
    This function should never be called.
    """
    raise NotImplementedError("This function should only be used as a placeholder.")


def run_experiment(
    n_range: range,
    num_tests: int,
    compute_basis: compute_basis_t,
    metric_fn: Callable[[int, np.ndarray, compute_basis_t], float],
    is_directed: bool,
) -> List[float]:
    """
    For each graph size n in n_range, open the corresponding file,
    read the number of tests and compute the average metric given by metric_fn.
    The metric_fn should have signature (n, weights, basis) -> value.
    """
    results = []
    for n in n_range:
        filename = _input_filename(n, num_tests, is_directed)
        if not os.path.exists(filename):
            print(f"Skipping non-existent file: {filename}")
            continue
        with open(filename, "r") as f:
            tests = int(f.readline())
            values = [
                metric_fn(*read_graph_input(f), compute_basis) for _ in range(tests)
            ]
            results.append(mean(values))
    return results


def run_experiment_pair(
    n_range,
    num_tests: int,
    compute_basis_A: compute_basis_t,
    compute_basis_B: compute_basis_t,
    metric_fn: Callable[[int, np.ndarray, compute_basis_t, compute_basis_t], float],
    is_directed: bool,
):
    results = []
    for n in n_range:
        filename = _input_filename(n, num_tests, is_directed)
        if not os.path.exists(filename):
            print(f"Skipping non-existent file: {filename}")
            continue
        with open(filename, "r") as f:
            tests = int(f.readline())
            values = []
            for _ in range(tests):
                n, weights = read_graph_input(f)
                values.append(metric_fn(n, weights, compute_basis_A, compute_basis_B))
            results.append(mean(values))
    return results


def run_experiment_file(
    n_range: range,
    num_tests: int,
    compute_basis: compute_basis_t,
    metric_fn: Callable[[int, np.ndarray], float],
    is_directed: bool,
) -> List[float]:
    """
    Similar to run_experiment but where the metric_fn operates on the entire file.
    Here metric_fn should have signature (file_handle, basis) -> value.
    """
    results = []
    for n in n_range:
        filename = _input_filename(n, num_tests, is_directed)
        if not os.path.exists(filename):
            print(f"Skipping non-existent file: {filename}")
            continue
        with open(filename, "r") as f:
            results.append(metric_fn(f, compute_basis))
    return results


def plot_and_save(
    title,
    x_axis,
    series,
    series_labels,
    xlabel,
    ylabel,
    num_tests,
    save_fig,
    show_plot,
    is_directed,
):
    for data, label in zip(series, series_labels):
        plt.plot(x_axis, data, label=label, marker="o")
    if save_fig:
        _plt_config(title, xlabel, ylabel, log=False)
        plt.savefig(_saved_filename(title, num_tests, is_directed))
        _plt_config(title, xlabel, ylabel, log=True)
        plt.savefig(_saved_filename_log(title, num_tests, is_directed))
    _plt_config(title, xlabel, ylabel, log=False)
    if show_plot:
        plt.show()
    plt.clf()
