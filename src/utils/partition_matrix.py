import concurrent
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Generator
from line_profiler import profile


def get_all_partition_matrices(
    n: int, m: int, parallel: bool = False
) -> Generator[np.ndarray, None, None]:
    """
    Generates all possible partition matrices of size n x m.

    Each matrix represents a way to assign one of `m` values (or labels)
    to each of the `n` indices, encoded as a binary or categorical matrix.

    Args:
        n (int): Number of indices (rows in the matrix).
        m (int): Number of distinct values or categories (columns in the matrix).
        parallel (bool, optional): Whether to execute in parallel for faster computation. Defaults to False.

    Yields:
        np.ndarray: A partition matrix of shape (n, m) for each possible partitioning.
    """
    if parallel:
        return _compute_parallel(n, m, _compute)
    return _compute(n, m)


def _compute(n, m):
    assert n >= m >= 2
    M = np.zeros((n, m))

    def f(i, free, toBeUsed):
        if i == n:
            yield M
        else:
            for j in range(m):
                if (toBeUsed >> j) & 0x1:
                    M[i][j] = 1
                    yield from f(i + 1, free ^ (1 << j), toBeUsed ^ (1 << j))
                    M[i][j] = 0
            if n - i > toBeUsed.bit_count():
                for j in range(m):
                    if (free >> j) & 0x1:
                        M[i][j] = 1
                        yield from f(i + 1, free, toBeUsed)
                        M[i][j] = 0

    yield from f(0, 0, (1 << m) - 1)


def _compute_parallel(n, m, solve_fn):
    assert n >= m >= 2
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _compute_parallel_worker, 0, 0, (1 << m) - 1, n, m, solve_fn
            )
        ]
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
        return results


def _compute_parallel_worker(i, free, toBeUsed, n, m, solve_fn):
    M = np.zeros((n, m))

    def f(i, free, toBeUsed):
        if i == n:
            return [solve_fn(M)]
        else:
            results = []
            for j in range(m):
                if (toBeUsed >> j) & 0x1:
                    M[i][j] = 1
                    results.extend(f(i + 1, free ^ (1 << j), toBeUsed ^ (1 << j)))
                    M[i][j] = 0
            if n - i > toBeUsed.bit_count():
                for j in range(m):
                    if (free >> j) & 0x1:
                        M[i][j] = 1
                        results.extend(f(i + 1, free, toBeUsed))
                        M[i][j] = 0
            return results

    return f(i, free, toBeUsed)
