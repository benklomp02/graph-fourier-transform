import ctypes
import numpy as np
import os


LIB_PATH = os.path.join(os.path.dirname(__file__), "../c_ext/liblinalg.so")
lib = ctypes.CDLL(LIB_PATH)

lib.norm.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.norm.restype = ctypes.c_double


# A slight improvement in performance by 23% compared to the numpy version
def c_norm(arr: np.ndarray) -> float:
    arr = arr.astype(np.float64)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return lib.norm(ptr, arr.size)


lib.matvec.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
]
lib.matvec.restype = None


# This function is by 700% worse than the @ numpy version
def c_matvec(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    M = M.astype(np.float64)
    v = v.astype(np.float64)
    rows, cols = M.shape
    result = np.zeros(rows, dtype=np.float64)
    lib.matvec(
        M.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rows,
        cols,
    )
    return result
