import concurrent
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from line_profiler import profile


# --- Sequential version ---
@profile
def get_all_partition_matrices(n: int, m: int):
    """Generating all partition matrices for a signal of size n with m different values."""
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


# --- Parallel version ---


def get_all_solution_vectors_par(n: int, m: int, solve_fn):
    """Parallel version of generating all partition matrices for a signal of size n with m different values."""
    assert n >= m >= 2
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _get_all_solution_vectors_worker, 0, 0, (1 << m) - 1, n, m, solve_fn
            )
        ]
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
        return results


def _get_all_solution_vectors_worker(
    i: int, free: int, toBeUsed: int, n: int, m: int, solve_fn
):
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
