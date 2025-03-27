#ifndef LINALG_H
#define LINALG_H

int *build_masked_array(int mask, int n);

int *arg_max_greedy(int n, int *tau, int *memo);

#endif