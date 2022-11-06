#include <random.h>

// These static variables need to be in the .cpp file (not the header)
static const unsigned long int special_seed = 8675309; // used by autograder
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

// Seed initialization
void kmeans_srand(unsigned int seed) {
    next = seed;
}

// Use this provided function to match the autograder.
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

// Helping method used in get_random_initial_centroids
// Returns a vector containing k numbers in the range [0, max_value)
// Note: Assume the kmeans_rand() generator will never produce repeats.
std::vector<int> select_random_subset(int k, int max_value) {
  std::vector<int> result;
  result.reserve(k);
  while(k--) {
    result.emplace_back(kmeans_rand() % max_value);
  }
  return result;
}

std::vector<std::vector<double>> random_centroids
    (int k, const std::vector<std::vector<double>>& data) {
  
  std::vector<std::vector<double>> centroids;
  centroids.reserve(k);

  auto random_indices = select_random_subset(k, data.size());
  // print_container_items_by_line(random_indices);

  for(int index : random_indices) {
    centroids.emplace_back(data[index]);
  }
  return centroids;
}

// Note: centroids is an out parameter
void random_centroids(int k, int dims, int N, double* data, double* centroids) {
  for(int i = 0; i < k; ++i) {
    int point_idx = kmeans_rand() % N;
    // std::cout << point_idx << '\n'; // [?] check indices match first
    for(int d = 0; d < dims; ++d) {
      centroids[i*dims + d] = data[point_idx*dims + d];
    }
  }
}

// for (int i=0; i<k; i++){
//     int index = kmeans_rand() % _numpoints;
//     // you should use the proper implementation of the following
//     // code according to your data structure
//     centers[i] = points[index];
