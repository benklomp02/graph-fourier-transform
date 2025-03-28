import numpy as np
from scipy.linalg import null_space
from scipy.sparse.linalg import svds, ArpackNoConvergence
from line_profiler import profile


def solve_minimisation_problem(
    M: np.ndarray, U: np.ndarray = None, is_constant: bool = False
) -> np.ndarray:
    if is_constant:
        """Solves the minimisation problem for constant signal."""
        c1, c2 = np.sum(M, axis=0)
        a = np.array([1.0, -c1 / c2])
        _x = M @ a
        return _x / np.linalg.norm(_x)
    else:
        """Solves the general minimisation problem."""
        X = U.T @ M
        try:
            if min(X.shape) < 10:
                raise ValueError("Matrix is too small for svds.")
            _, _, V = svds(X, k=1)
            _x = M @ V[:, 0]
        except (ArpackNoConvergence, ValueError, np.linalg.LinAlgError):
            ns = null_space(X)
            _x = M @ ns[:, 0]
        return _x / np.linalg.norm(_x)
