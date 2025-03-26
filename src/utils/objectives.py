import ctypes
import numpy as np
import os

LIB_PATH = os.path.join(os.path.dirname(__file__), "../c_ext/libobjectives.so")
lib = ctypes.CDLL(LIB_PATH)


def W(A: int, B: int, weights: np.ndarray) -> float:
    n = weights.shape[0]
    weights = np.ascontiguousarray(weights, dtype=np.float64).flatten()
    return lib.W(A, B, weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)


def F(A: int, B: int, weights: np.ndarray) -> float:
    n = weights.shape[0]
    weights = np.ascontiguousarray(weights, dtype=np.float64).flatten()
    return lib.F(A, B, weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)


def S_undirected(x: np.ndarray, weights: np.ndarray) -> float:
    weights = np.ascontiguousarray(weights, dtype=np.float64).flatten()
    return lib.S_undirected(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x.size,
    )


def S_directed(x: np.ndarray, weights: np.ndarray) -> float:
    weights = np.ascontiguousarray(weights, dtype=np.float64).flatten()
    return lib.S_directed(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x.size,
    )
