#include <stdlib.h>

int *build_masked_array(int mask, int n)
{
    int *arr = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        arr[i] = mask >> i & 0x1;
    }
    return arr;
}

static int bit_count(int n)
{
    int count = 0;
    while (n)
    {
        count += n & 0x1;
        n >>= 1;
    }
    return count;
}

int *arg_max_greedy(int n, int *tau, double *memo)
{
    int *result = (int *)malloc(2 * sizeof(int));
    double max_val = -1.0;
    int best_a = 0, best_b = 0;

    for (int i = 1; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            double denom = bit_count(tau[i]) * bit_count(tau[j]);
            double val = memo[i * n + j] / denom;
            if (val > max_val)
            {
                max_val = val;
                best_a = i;
                best_b = j;
            }
            val = memo[j * n + i] / denom;
            if (val > max_val)
            {
                max_val = val;
                best_a = j;
                best_b = i;
            }
        }
    }
    result[0] = best_a;
    result[1] = best_b;
    return result;
}