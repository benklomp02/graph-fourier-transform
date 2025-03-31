#ifndef LINALG_H
#define LINALG_H

int *build_masked_array(long mask, int n);

void free_masked_array(int *masked_array);

int *arg_max_greedy(int n, long *tau, int *memo);

#endif