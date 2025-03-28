import os
import matplotlib.pyplot as plt

from statistics import mean
from tests.IO.graph_generator import next_graph_input


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


def run_experiment(n_range, num_tests, compute_basis, metric_fn, is_directed):
    """
    For each graph size n in n_range, open the corresponding file,
    read the number of tests and compute the average metric given by metric_fn.
    The metric_fn should have signature (n, weights, basis) -> value.
    """
    results = []
    for n in n_range:
        filename = input_filename(n, num_tests, is_directed)
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


def run_experiment_file(n_range, num_tests, compute_basis, metric_fn, is_directed):
    """
    Similar to run_experiment but where the metric_fn operates on the entire file.
    Here metric_fn should have signature (file_handle, basis) -> value.
    """
    results = []
    for n in n_range:
        filename = input_filename(n, num_tests, is_directed)
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
        plt_config(title, xlabel, ylabel, log=False)
        plt.savefig(saved_filename(title, num_tests, is_directed))
        plt_config(title, xlabel, ylabel, log=True)
        plt.savefig(saved_filename_log(title, num_tests, is_directed))
    plt_config(title, xlabel, ylabel, log=False)
    if show_plot:
        plt.show()
    plt.clf()
