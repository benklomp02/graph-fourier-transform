import ctypes
import numpy as np
import os
from typing import Tuple
import weakref


LIB_PATH = os.path.join(os.path.dirname(__file__), "../c_ext/liblinalg.so")
lib = ctypes.CDLL(LIB_PATH)

lib.build_masked_array.restype = ctypes.POINTER(ctypes.c_int)
lib.build_masked_array.argtypes = [ctypes.c_int, ctypes.c_int]

lib.free_masked_array.restype = None
lib.free_masked_array.argtypes = [ctypes.POINTER(ctypes.c_int)]


def build_masked_array(mask: int, n: int) -> np.ndarray:
    """Returns a masked array of size n, where the indices specified by the mask are set to 1 and others to 0.

    Args:
        mask (int): A bitmask representing the indices to be included in the set.
        n (int): The total number of indices.

    Returns:
        np.ndarray: An array of size n.
    """
    c_array_ptr = lib.build_masked_array(mask, n)
    x = np.ctypeslib.as_array(c_array_ptr, shape=(n,))
    weakref.finalize(x, lib.free_masked_array, c_array_ptr)
    return x


lib.arg_max_greedy.restype = ctypes.POINTER(ctypes.c_int)
lib.arg_max_greedy.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32),
    np.ctypeslib.ndpointer(dtype=np.float64),
]


def arg_max_greedy(n: int, tau: np.ndarray, memo: np.ndarray) -> Tuple[int, int]:
    """
    Finds the two arguments yielding the highest value of the objective function.

    Args:
        n (int): _description_
        tau (np.ndarray): Holds all sets as integer bitmasks.
        memo (np.ndarray): Holds W(A, B) for all sets A, B.

    Returns:
        Tuple[int, int]: A tuple containing the two indices that yield the maximum value.
    """
    tau = np.ascontiguousarray(tau, dtype=np.int32)
    memo = np.ascontiguousarray(memo, dtype=np.float64)
    c_array_ptr = lib.arg_max_greedy(n, tau.flatten(), memo.flatten())
    result = ctypes.cast(c_array_ptr, ctypes.POINTER(ctypes.c_int * 2)).contents
    return result[0], result[1]
