#ifndef _RANDOM_H
#define _RANDOM_H

#include <vector>
#include <log.h>

void kmeans_srand(unsigned int cmd_seed); // cmd_seed is a cmdline arg

int kmeans_rand();

std::vector<int> select_random_subset(int k, int max_value);

std::vector<std::vector<double>> random_centroids
    (int k, const std::vector<std::vector<double>>& data);

void random_centroids(int k, int dims, int N, double* data, double* centroids);

#endif