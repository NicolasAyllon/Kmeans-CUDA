#include <iostream>
#include <argparse.h>
#include <io.h>
#include <random.h>
#include <kmeans.h>

#if (defined(CUDA) || defined(CUDA_SHARED) || defined(THRUST))
#include <kmeans_kernels.h>
#endif

// By default, define CPU
#if !(defined(CPU) || defined(CUDA) || defined(CUDA_SHARED) || defined(THRUST))
  #define CPU
#endif

int main(int argc, char** argv) {

  struct options_t opts;
  get_opts(argc, argv, &opts);

  int num_points = 0;

#ifdef CPU
  // std::cout << "running CPU version" << std::endl;

  // Read file into vector<vector<double>>
  auto data = read_file(&opts, num_points);
  
  // Initialize random number generator with seed.
  kmeans_srand(opts.seed);

  // Run kmeans
  struct clustering_data results = kmeans(data, &opts);
  print_output(results, &opts);

#elif CUDA
  // std::cout << "running CUDA version" << std::endl;
  double* data = nullptr;
  // Read file and store points in array pointed to by data.
  read_file(&opts, &data, &num_points);

  // Initialize random number generator with seed.
  kmeans_srand(opts.seed);

  // Run kmeans
  struct clustering_data_gpu results = kmeans_gpu(data, num_points, &opts);
  print_output(results, &opts, num_points);
    
#elif CUDA_SHARED
  // std::cout << "running CUDA SHARED version" << std::endl;
  double* data = nullptr;
  // Read file and store points in array pointed to by data.
  read_file(&opts, &data, &num_points);

  // Initialize random number generator with seed.
  kmeans_srand(opts.seed);

  // Run kmeans
  struct clustering_data_gpu results = kmeans_gpu(data, num_points, &opts);
  print_output(results, &opts, num_points);

#elif THRUST
  std::cout << "running THRUST version" << std::endl;
#endif

  return 0;
}