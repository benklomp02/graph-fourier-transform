from typing import Callable, List
import matplotlib.pyplot as plt
from statistics import mean


from src.algorithms.greedy import compute_greedy_basis_undirected
from src.algorithms.l1_norm import compute_l1_norm_basis_undirected
from src.algorithms.laplacian import compute_laplacian_basis
from src.utils.measurements import (
    relative_error,
    relative_error_single_vector,
    average_time,
)
from src.utils.graphs import next_graph_input


def plt_config(title, xlabel, ylabel):
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")


def comparison_of_variation_single_vector(min_N=3, max_N=6, k=1, save_fig=True):
    # Let k be zero-based index of the vector we are interested in
    assert 3 <= min_N < max_N <= 7 and 0 <= k < min_N

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
        with open(f"public/input/fixed_sized/fixed_size_{n}.txt", "r") as f:
            avg_errors_greedy.append(
                average_relative_error_single_vector(f, compute_greedy_basis_undirected)
            )
        with open(f"public/input/fixed_sized/fixed_size_{n}.txt", "r") as f:
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
        plt.savefig(f"public/plots/comparison_of_variation_single_vector_k{k+1}.png")
    plt.show()


def comparison_of_variation(min_N=3, max_N=6, save_fig=True):
    assert 3 <= min_N < max_N <= 7

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
        with open(f"public/input/fixed_sized/fixed_size_{n}.txt", "r") as f:
            avg_errors_greedy.append(
                average_relative_error(f, compute_greedy_basis_undirected)
            )
        with open(f"public/input/fixed_sized/fixed_size_{n}.txt", "r") as f:
            avg_errors_laplacian.append(
                average_relative_error(f, compute_laplacian_basis)
            )
    plt.plot(x_axis, avg_errors_greedy, label="Greedy", marker="o")
    plt.plot(x_axis, avg_errors_laplacian, label="Laplacian", marker="o")
    plt_config("Comparison of variation", "N", "Average relative error")
    if save_fig:
        plt.savefig("public/plots/comparison_of_variation.png")
    plt.show()


def comparison_of_time(
    xfunc: List[Callable], xlabel: list[str], min_N=3, max_N=6, save_fig=True
):
    assert 3 <= min_N < max_N <= 7
    x_axis = range(min_N, max_N + 1)
    s_avg_times = []
    for compute_basis in xfunc:
        x_avg_times = []
        for n in x_axis:
            with open(f"public/input/fixed_sized/fixed_size_{n}.txt", "r") as f:
                x_avg_times.append(average_time(f, compute_basis))
        s_avg_times.append(x_avg_times)
    for xtime, label in zip(s_avg_times, xlabel):
        plt.plot(x_axis, xtime, label=label, marker="o")
    plt_config("Comparison of time", "N", "Average time")
    if save_fig:
        plt.savefig("public/plots/comparison_of_time.png")
    plt.show()


if __name__ == "__main__":
    # comparison_of_variation_single_vector()
    # comparison_of_variation()
    xfunc = [
        compute_l1_norm_basis_undirected,
        compute_greedy_basis_undirected,
        compute_laplacian_basis,
    ]
    xlabel = ["L1 Norm", "Greedy", "Laplacian"]
    comparison_of_time(xfunc, xlabel)
