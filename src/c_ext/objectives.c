
double W(long A, long B, double *weights, int n)
{
    double sum = 0.;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            if (A >> i & 1 && B >> j & 1)
                sum += weights[i * n + j];
        }
    return sum;
}

static int bit_count(long n)
{
    int count = 0;
    while (n)
    {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

double F(long A, long B, double *weights, int n)
{
    return W(A, B, weights, n) / (bit_count(A) * bit_count(B));
}

// Objective function working for both directed and undirected graphs.
double S(double *x, double *weights, int n)
{
    double sum = 0.;
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
        {
            if (x[i] > x[j])
                sum += (x[i] - x[j]) * weights[i * n + j];
            else
                sum += (x[j] - x[i]) * weights[j * n + i];
        }
    return sum;
}