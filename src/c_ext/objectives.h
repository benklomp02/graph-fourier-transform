#ifndef OBJECTIVES_H
#define OBJECTIVES_H

double W(int A, int B, double *weights, int n);

int count_set_bits(int n);

double F(int A, int B, double *weights, int n);

int S_undirected(double *x, double **weights, int n);

int S_directed(double *x, double **weights, int n);

#endif