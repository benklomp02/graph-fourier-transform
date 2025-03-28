import numpy as np

from plot.utils.visualisation import visualize_graph_from_weights


# ---- Example Graphs ----
def path(n, visualize=False):
    weights = np.array(
        [[1 if i == j + 1 or i == j - 1 else 0 for j in range(n)] for i in range(n)]
    )
    if visualize:
        visualize_graph_from_weights(weights)
    return weights


def comet(n, visualize=False):
    m = n // 2
    weights = np.array(
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
    if visualize:
        visualize_graph_from_weights(weights)
    return weights


def sensor(n, visualize=False):
    weights = np.array([[1 if i != j else 0 for j in range(n)] for i in range(n)])
    if visualize:
        visualize_graph_from_weights(weights)
    return weights
