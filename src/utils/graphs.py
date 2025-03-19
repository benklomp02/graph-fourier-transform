import numpy as np

from itertools import combinations
from collections import deque


# ---- some constants ---
NUM_TESTS = 20
MAX_SIZE_LARGE = 20
MAX_SIZE_SMALL = 7
MAX_WEIGHT = 10


# --- some example graphs ---
def path(n=7):
    return n, [
        [1 if i == j + 1 or i == j - 1 else 0 for j in range(n)] for i in range(n)
    ]


def comet(n=7):
    m = n // 2
    return n, [
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


def sensor(n=8):
    return n, [[1 if i != j else 0 for j in range(n)] for i in range(n)]


# ---- some graph generators ---
def simple_graph(n: int, p) -> list[list[int]]:
    adj = [[0] * n for _ in range(n)]
    for x, y in combinations(range(n), 2):
        adj[x][y] = adj[y][x] = np.random.choice([0, 1], p=[p, 1 - p])
        wt = np.random.randint(1, MAX_WEIGHT)
        adj[x][y] *= wt
        adj[y][x] *= wt
    groups = [None for i in range(n)]
    for x in range(n):
        if groups[x] is None:
            groups[x] = x
            s = deque([x])
            while s:
                u = s.popleft()
                for v in range(n):
                    if adj[u][v] and groups[v] is None:
                        groups[v] = x
                        s.append(v)
    for x in range(n):
        for y in range(x + 1, n):
            gx = groups[x]
            while gx != groups[gx]:
                gx = groups[gx]
            gy = groups[y]
            while gy != groups[gy]:
                gy = groups[gy]
            if gx != gy:
                adj[x][y] = adj[y][x] = np.random.randint(1, MAX_WEIGHT)
    return adj


# --- read test cases ---
def next_graph_input(f):
    line = f.readline()
    if line == "":
        raise EOFError
    n = int(line)
    weights = [list(map(int, f.readline().split())) for _ in range(n)]
    return n, weights


# --- write test cases ---
def write_test_case(file_name, create_graph, max_size):
    with open(f"public/input/{file_name}.txt", "w") as f:
        print(NUM_TESTS, file=f)
        for _ in range(NUM_TESTS):
            n = np.random.randint(2, max_size)
            print(n, file=f)
            weights = create_graph(n)
            for row in weights:
                print(*row, file=f)


def write_test_case_fixed_size(create_graph, fixed_size):
    with open(
        f"public/input/fixed_sized/fixed_size_{fixed_size}_{NUM_TESTS}.txt", "w"
    ) as f:
        print(NUM_TESTS, file=f)
        for _ in range(NUM_TESTS):
            print(fixed_size, file=f)
            weights = create_graph(fixed_size)
            for row in weights:
                print(*row, file=f)


if __name__ == "__main__":
    # write_test_case("small", lambda n: simple_graph(n, 0.5), MAX_SIZE_SMALL)
    # write_test_case("large", lambda n: simple_graph(n, 0.5), MAX_SIZE_LARGE)
    for i in range(3, 8):
        write_test_case_fixed_size(lambda n: simple_graph(n, 0.5), i)
