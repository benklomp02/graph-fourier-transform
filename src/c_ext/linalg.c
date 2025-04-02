#include <stdlib.h>

// The function creates a masked array of size n, where each element is either 0 or 1.
int *build_masked_array(long mask, int n)
{
    int *arr = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        arr[i] = mask >> i & 0x1;
    }
    return arr;
}

void free_masked_array(long *masked_array)
{
    free(masked_array);
}

static int bit_count(long n)
{
    int count = 0;
    while (n)
    {
        count += n & 0x1;
        n >>= 1;
    }
    return count;
}

// Finds the best indices i, j s.t that the ratio of
// memo[i][j] / (bit_count(tau[i]) * bit_count(tau[j])),
// i.e. the function F in the script, is maximized.
int *arg_max_greedy(int n, long *tau, double *memo)
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