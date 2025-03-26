import numpy as np

from src.algorithms.laplacian import laplacian_matrix


def approx_error(x: np.ndarray, y_n: np.ndarray):
    return np.linalg.norm(x - y_n) / np.linalg.norm(x)


def compute_nterm_error(n_approx: int, signal: np.ndarray, weights, compute_basis):
    signal = signal.ravel()
    n = signal.size
    basis = compute_basis(n, weights)
    x_ft = basis.T @ signal
    kx = sorted(range(n), key=lambda k: abs(x_ft[k]), reverse=True)
    y_n = sum(x_ft[kx[i]] * basis[:, kx[i]] for i in range(n_approx))
    return approx_error(signal, y_n)


def compute_random_laplacian_signal(n: int, weights, mu=1):
    xlambda = np.linalg.eigvals(laplacian_matrix(n, weights=weights))
    return np.array(
        [1 / (1 + mu * xlambda[k]) * np.random.uniform(-1, 1) for k in range(n)]
    )
