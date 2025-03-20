from typing import Callable, List
import matplotlib.pyplot as plt
from statistics import mean
import time
import os

from src.algorithms.greedy import compute_greedy_basis_undirected
from src.algorithms.l1_norm import compute_l1_norm_basis_undirected
from src.algorithms.laplacian import compute_laplacian_basis
from src.utils.measurements import (
    relative_error,
    relative_error_single_vector,
    average_time,
)
from src.utils.graphs import next_graph_input

MAX_SIZE = 7
SMALL_SET_SIZE = 20
LARGE_SET_SIZE = 100


def input_filename(N, num_tests, directed: bool = False):
    filename = f"public/input/{"directed" if directed else "undirected"}/input_N{N}_t{num_tests}.txt"
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
    return filename


def saved_filename(title: str, num_tests: int, directed: bool = False):
    return f"public/plots/{"directed" if directed else "undirected"}/{title.replace(" ", "_")}_t{num_tests}.png"


def saved_filename_log(title: str, num_tests: int, directed: bool = False):
    return f"public/plots/{"directed" if directed else "undirected"}/{title.replace(" ", "_")}_t{num_tests}_log.png"


def plt_config(title: str, xlabel: str, ylabel: str, log: bool = False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if log:
        plt.yscale("log")


def run_experiment(n_range, num_tests, compute_basis, metric_fn):
    """
    For each graph size n in n_range, open the corresponding file,
    read the number of tests and compute the average metric given by metric_fn.
    The metric_fn should have signature (n, weights, basis) -> value.
    """
    results = []
    for n in n_range:
        filename = input_filename(n, num_tests)
        if not os.path.exists(filename):
            print(f"Skipping non-existent file: {filename}")
            continue
        with open(filename, "r") as f:
            tests = int(f.readline())
            values = [
                metric_fn(*next_graph_input(f), compute_basis) for _ in range(tests)
            ]
            results.append(mean(values))
    return results


def run_experiment_file(n_range, num_tests, compute_basis, metric_fn):
    """
    Similar to run_experiment but where the metric_fn operates on the entire file.
    Here metric_fn should have signature (file_handle, basis) -> value.
    """
    results = []
    for n in n_range:
        filename = input_filename(n, num_tests)
        if not os.path.exists(filename):
            print(f"Skipping non-existent file: {filename}")
            continue
        with open(filename, "r") as f:
            results.append(metric_fn(f, compute_basis))
    return results


def plot_and_save(
    title, x_axis, series, series_labels, xlabel, ylabel, num_tests, save_fig
):
    for data, label in zip(series, series_labels):
        plt.plot(x_axis, data, label=label, marker="o")
    plt_config(title, xlabel, ylabel, log=False)
    if save_fig:
        plt.savefig(saved_filename(title, num_tests))
        plt_config(title, xlabel, ylabel, log=True)
        plt.savefig(saved_filename_log(title, num_tests))
    plt.clf()


def comparison_of_variation_single_vector(
    min_N=3,
    max_N=MAX_SIZE,
    k=1,
    save_fig=True,
    num_tests=LARGE_SET_SIZE,
    show_plot=False,
):
    print(f"Starting comparison of variation on single vector k={k+1}.")
    start_time = time.time()
    assert 3 <= min_N < max_N <= MAX_SIZE and 0 <= k < min_N
    x_axis = range(min_N, max_N + 1)

    metric_fn = lambda n, weights, basis: relative_error_single_vector(
        basis(n, weights)[:, k],
        compute_l1_norm_basis_undirected(n, weights)[:, k],
        weights,
    )

    greedy_errors = run_experiment(
        x_axis, num_tests, compute_greedy_basis_undirected, metric_fn
    )
    laplacian_errors = run_experiment(
        x_axis, num_tests, compute_laplacian_basis, metric_fn
    )

    plot_title = f"Comparison of variation on single vector k={k+1}"
    plot_and_save(
        plot_title,
        x_axis,
        [greedy_errors, laplacian_errors],
        ["Greedy", "Laplacian"],
        "N",
        "Average relative error",
        num_tests,
        save_fig,
    )

    print(
        f"Finished comparison of variation on single vector k={k+1} in {time.time()-start_time:.2f} seconds."
    )
    if show_plot:
        plt.show()


def comparison_of_variation(
    min_N=3, max_N=MAX_SIZE, save_fig=True, num_tests=LARGE_SET_SIZE, show_plot=False
):
    print("Starting comparison of variation.")
    start_time = time.time()
    assert 3 <= min_N < max_N <= MAX_SIZE
    x_axis = range(min_N, max_N + 1)

    metric_fn = lambda n, weights, basis: relative_error(
        basis(n, weights), compute_l1_norm_basis_undirected(n, weights), weights
    )

    greedy_errors = run_experiment(
        x_axis, num_tests, compute_greedy_basis_undirected, metric_fn
    )
    laplacian_errors = run_experiment(
        x_axis, num_tests, compute_laplacian_basis, metric_fn
    )

    plot_title = "Comparison of variation"
    plot_and_save(
        plot_title,
        x_axis,
        [greedy_errors, laplacian_errors],
        ["Greedy", "Laplacian"],
        "N",
        "Average relative error",
        num_tests,
        save_fig,
    )

    print(f"Finished comparison of variation in {time.time()-start_time:.2f} seconds.")
    if show_plot:
        plt.show()


def comparison_of_time(
    xfunc: List[Callable] = [
        compute_l1_norm_basis_undirected,
        compute_greedy_basis_undirected,
        compute_laplacian_basis,
    ],
    series_labels: List[str] = ["L1 Norm", "Greedy", "Laplacian"],
    min_N=3,
    max_N=MAX_SIZE,
    save_fig=True,
    num_tests=LARGE_SET_SIZE,
    show_plot=False,
):
    print("Starting comparison of time.")
    start_time = time.time()
    assert 3 <= min_N < max_N <= MAX_SIZE
    x_axis = range(min_N, max_N + 1)

    time_metric = lambda f, basis: average_time(f, basis)
    series = [
        run_experiment_file(x_axis, num_tests, basis, time_metric) for basis in xfunc
    ]

    plot_title = "Comparison of time"
    plot_and_save(
        plot_title,
        x_axis,
        series,
        series_labels,
        "N",
        "Average time",
        num_tests,
        save_fig,
    )

    print(f"Finished comparison of time in {time.time()-start_time:.2f} seconds.")
    if show_plot:
        plt.show()


if __name__ == "__main__":
    comparison_of_variation_single_vector(num_tests=LARGE_SET_SIZE)
    comparison_of_variation(num_tests=LARGE_SET_SIZE)
    comparison_of_time(num_tests=LARGE_SET_SIZE)
