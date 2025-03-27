import ctypes
import numpy as np
import os
from typing import Tuple


LIB_PATH = os.path.join(os.path.dirname(__file__), "../c_ext/liblinalg.so")
lib = ctypes.CDLL(LIB_PATH)

lib.build_masked_array.restype = ctypes.POINTER(ctypes.c_int)
lib.build_masked_array.argtypes = [ctypes.c_int, ctypes.c_int]


def build_masked_array(mask: int, n: int) -> np.ndarray:
    c_array_ptr = lib.build_masked_array(mask, n)
    return np.ctypeslib.as_array(c_array_ptr, shape=(n,))


lib.arg_max_greedy.restype = ctypes.POINTER(ctypes.c_int)
lib.arg_max_greedy.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32),
    np.ctypeslib.ndpointer(dtype=np.float64),
]


def arg_max_greedy(n: int, tau: np.ndarray, memo: np.ndarray) -> Tuple[int, int]:
    tau = np.ascontiguousarray(tau, dtype=np.int32)
    memo = np.ascontiguousarray(memo, dtype=np.float64)
    c_array_ptr = lib.arg_max_greedy(n, tau.flatten(), memo.flatten())
    result = ctypes.cast(c_array_ptr, ctypes.POINTER(ctypes.c_int * 2)).contents
    return result[0], result[1]
