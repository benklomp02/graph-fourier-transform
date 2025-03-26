import numpy as np


# --- some example graphs ---
def path(n=7):
    return n, np.array(
        [[1 if i == j + 1 or i == j - 1 else 0 for j in range(n)] for i in range(n)]
    )


def comet(n=7):
    m = n // 2
    return n, np.array(
        [
            [
                (
                    1
                    if i < m
                    and j == m
                    or i == m
                    and j < m
                    or (i >= m and (i == j - 1 or i == j + 1))
                    else 0
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
    )


def sensor(n=8):
    return n, np.array([[1 if i != j else 0 for j in range(n)] for i in range(n)])
