from typing import Callable, List
import matplotlib.pyplot as plt
from statistics import mean
import time


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

input_filename = (
    lambda N, num_tests: f"public/input/fixed_sized/input_N{N}_t{num_tests}.txt"
)
saved_filename = lambda title, num_tests: f"public/plots/{title}_t{num_tests}.png"


def plt_config(title, xlabel, ylabel):
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")


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
    # Let k be zero-based index of the vector we are interested in
    assert 3 <= min_N < max_N <= MAX_SIZE and 0 <= k < min_N

    def average_relative_error_single_vector(f, compute_basis):
        num_tests = int(f.readline())
        avg_err = lambda n, weights: relative_error_single_vector(
            compute_basis(n, weights)[:, k],
            compute_l1_norm_basis_undirected(n, weights)[:, k],
            weights,
        )
        return mean(avg_err(*next_graph_input(f)) for _ in range(num_tests))

    x_axis = range(min_N, max_N + 1)
    avg_errors_greedy = []
    avg_errors_laplacian = []
    for n in x_axis:
        with open(
            input_filename(n, num_tests),
            "r",
        ) as f:
            avg_errors_greedy.append(
                average_relative_error_single_vector(f, compute_greedy_basis_undirected)
            )
        with open(
            input_filename(n, num_tests),
            "r",
        ) as f:
            avg_errors_laplacian.append(
                average_relative_error_single_vector(f, compute_laplacian_basis)
            )
    plt.plot(x_axis, avg_errors_greedy, label="Greedy", marker="o")
    plt.plot(x_axis, avg_errors_laplacian, label="Laplacian", marker="o")
    plt_config(
        f"Comparison of variation on single vector k={k+1}",
        "N",
        "Average relative error",
    )
    if save_fig:
        plt.savefig(
            saved_filename(f"comparison_of_variation_single_vector_k{k+1}", num_tests)
        )
    print(
        f"Finished comparison of variation on single vector k={k+1} in {time.time()-start_time:.2f} seconds."
    )
    if show_plot:
        plt.show()


def comparison_of_variation(
    min_N=3, max_N=MAX_SIZE, save_fig=True, num_tests=LARGE_SET_SIZE, show_plot=False
):
    print(f"Starting comparison of variation.")
    start_time = time.time()
    assert 3 <= min_N < max_N <= MAX_SIZE

    def average_relative_error(f, compute_basis):
        num_tests = int(f.readline())
        avg_err = lambda n, weights: relative_error(
            compute_basis(n, weights),
            compute_l1_norm_basis_undirected(n, weights),
            weights,
        )
        return mean(avg_err(*next_graph_input(f)) for _ in range(num_tests))

    x_axis = range(min_N, max_N + 1)
    avg_errors_greedy = []
    avg_errors_laplacian = []
    for n in x_axis:
        with open(
            input_filename(n, num_tests),
            "r",
        ) as f:
            avg_errors_greedy.append(
                average_relative_error(f, compute_greedy_basis_undirected)
            )
        with open(input_filename(n, num_tests), "r") as f:
            avg_errors_laplacian.append(
                average_relative_error(f, compute_laplacian_basis)
            )
    plt.plot(x_axis, avg_errors_greedy, label="Greedy", marker="o")
    plt.plot(x_axis, avg_errors_laplacian, label="Laplacian", marker="o")
    plt_config("Comparison of variation", "N", "Average relative error")
    if save_fig:
        plt.savefig(saved_filename("comparison_of_variation", num_tests))
    print(f"Finished comparison of variation in {time.time()-start_time:.2f} seconds.")
    if show_plot:
        plt.show()


std_functions = [
    compute_l1_norm_basis_undirected,
    compute_greedy_basis_undirected,
    compute_laplacian_basis,
]
std_labels = ["L1 Norm", "Greedy", "Laplacian"]


def comparison_of_time(
    xfunc: List[Callable] = std_functions,
    xlabel: List[str] = std_labels,
    min_N=3,
    max_N=MAX_SIZE,
    save_fig=True,
    num_tests=LARGE_SET_SIZE,
    show_plot=False,
):
    print(f"Starting comparison of time.")
    start_time = time.time()
    assert 3 <= min_N < max_N <= MAX_SIZE
    x_axis = range(min_N, max_N + 1)
    s_avg_times = []
    for compute_basis in xfunc:
        x_avg_times = []
        for n in x_axis:
            with open(
                input_filename(n, num_tests),
                "r",
            ) as f:
                x_avg_times.append(average_time(f, compute_basis))
        s_avg_times.append(x_avg_times)
    for xtime, label in zip(s_avg_times, xlabel):
        plt.plot(x_axis, xtime, label=label, marker="o")
    plt_config("Comparison of time", "N", "Average time")
    if save_fig:
        plt.savefig(saved_filename("comparison_of_time", num_tests))
    print(f"Finished comparison of time in {time.time()-start_time:.2f} seconds.")
    if show_plot:
        plt.show()


if __name__ == "__main__":
    comparison_of_variation_single_vector(num_tests=LARGE_SET_SIZE)
    comparison_of_variation(num_tests=LARGE_SET_SIZE)
    comparison_of_time(num_tests=LARGE_SET_SIZE)
