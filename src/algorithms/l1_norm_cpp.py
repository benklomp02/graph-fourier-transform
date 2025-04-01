import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os


LIB_PATH = os.path.join(os.path.dirname(__file__), "../cpp_ext/libl1.so")
lib = ctypes.CDLL(LIB_PATH)


lib.compute_l1_norm_basis_c.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
lib.compute_l1_norm_basis_c.restype = ctypes.POINTER(ctypes.c_double)

lib.free_allocated_array.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.free_allocated_array.restype = None


def compute_l1_norm_basis_cpp(n: int, weights: np.ndarray):
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    out_rows = ctypes.c_int()
    out_cols = ctypes.c_int()

    ptr = lib.compute_l1_norm_basis_c(
        n, weights, ctypes.byref(out_rows), ctypes.byref(out_cols)
    )
    rows, cols = out_rows.value, out_cols.value

    buffer = np.ctypeslib.as_array(ptr, shape=(rows * cols,))
    matrix = buffer.reshape((rows, cols), order='F')

    return matrix.copy()
