#ifndef LINALG_H
#define LINALG_H

double norm(double *a, int len);
void matvec(double *M, double *v, double *result, int rows, int cols);

#endif