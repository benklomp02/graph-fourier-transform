#include <math.h>

double norm(double *a, int len)
{
    double sum = 0.0;
    for (int i = 0; i < len; i++)
        sum += a[i] * a[i];
    return sqrt(sum);
}

void matvec(double *M, double *v, double *result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++)
            result[i] += M[i * cols + j] * v[j];
    }
}