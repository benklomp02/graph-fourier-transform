#include "partition_matrix.hpp"

generator<Eigen::MatrixXd> compute(
    int n, int m, int i,
    uint32_t free, uint32_t toBeUsed,
    Eigen::MatrixXd &M)
{
    if (i == n)
    {
        co_yield M;
        co_return;
    }

    for (int j = 0; j < m; ++j)
    {
        if ((toBeUsed >> j) & 1)
        {
            M(i, j) = 1;
            for (auto &matrix : compute(n, m, i + 1, free ^ (1 << j), toBeUsed ^ (1 << j), M))
                co_yield matrix;
            M(i, j) = 0;
        }
    }

    int deg_freedom = __builtin_popcount(toBeUsed);
    if (n - i > deg_freedom)
    {
        for (int j = 0; j < m; ++j)
        {
            if ((free >> j) & 1)
            {
                M(i, j) = 1;
                for (auto &matrix : compute(n, m, i + 1, free, toBeUsed, M))
                    co_yield matrix;
                M(i, j) = 0;
            }
        }
    }
}

generator<Eigen::MatrixXd> get_all_partition_matrices(int n, int m)
{
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, m);
    for (auto &matrix : compute(n, m, 0, 0, (1 << m) - 1, M))
        co_yield matrix;
}
