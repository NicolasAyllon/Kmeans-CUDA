#ifndef _KMEANS_KERNELS
#define _KMEANS_KERNELS

#include <random.h>
#include <argparse.h>
#include <io.h>

struct clustering_data_gpu {

  int iterations;
  float elapsed_milliseconds;
  double* centroids;
  int* labels;

  ~clustering_data_gpu() {
    if(centroids != nullptr)  delete[] centroids;
    if(labels != nullptr)     delete[] labels;
  }
};

struct clustering_data_gpu kmeans_gpu(double* data, int N, struct options_t* opts);

#endif