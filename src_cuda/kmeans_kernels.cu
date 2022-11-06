#include <stdio.h>          //printf
#include <helper_cuda.h>    // NVIDIA_SDK_SAMPLES/common/inc
#include <kmeans_kernels.h>
#include <stopwatch.h>


// Additional declaration needed for built-in atomicAdd(double*, double)
// https://stackoverflow.com/questions/37566987/
__device__ double atomicAdd(double* address, double val);

__device__ float* data_transfer_ms = 0;

// Global Stopwatches
struct stopwatch mem_stopwatch;
struct stopwatch algo_stopwatch;
struct stopwatch total_stopwatch;

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////

// Returns squared distance between d-dimensional vectors pointed by a and b
__device__ double sq_distance(double* a, double* b, int dims) {
  double sq_dist = 0;
  for(int i = 0; i < dims; ++i) {
    sq_dist += (b[i] - a[i]) * (b[i] - a[i]);
  }
  return sq_dist;
}

// Calculates the index of the centroid closest to each point in data, and
// writes that index into the labels array.
// - 1 block per point, 1 thread per centroid
// - Uses global memory
__global__ void nearest_centroids_k (double* d_data, int N, int dims, 
    double* d_centroids, int k, int* d_labels, double* d_sq_dists, int* d_centroid_idxs) {
  
  int p_idx = blockIdx.x;   // 1 block for 1 point in data
  int c_idx = threadIdx.x;  // 1 thread for each centroid
  int tid = threadIdx.x;    // convenience alias

  // Get starting address of data point and centroid
  double* point = &d_data[p_idx * dims];
  double* centroid = &d_centroids[c_idx * dims];
  // Get starting address of square distances and indices array
  double* sq_dists = &d_sq_dists[p_idx * k];
  int* centroid_idxs = &d_centroid_idxs[p_idx * k];

  if(tid < k) {
    sq_dists[tid] = sq_distance(point, centroid, dims);
    centroid_idxs[tid] = tid;
  }

  // Parallel reduce
  for(int s = blockDim.x/2; s > 0; s /= 2) {
    if(tid < s && tid + s < k) {
      if(sq_dists[centroid_idxs[tid]] < sq_dists[centroid_idxs[tid + s]]) {
        centroid_idxs[tid] = centroid_idxs[tid];
      } else {
        centroid_idxs[tid] = centroid_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  // At this point, centroid_idxs[0] contains the index of the closest centroid
  // Thread 0 writes it into the labels array.
  if(tid == 0) {
    d_labels[p_idx] = centroid_idxs[tid];
  }
}

// Version of nearest_centroids_k using shared memory for the sq_dists and
// centroid_idxs arrays.
__global__ void nearest_centroids_shared_k(double* data, int N, int dims, 
    double* centroids, int k, int* labels) {

  // The shared memory was allocated in one chunk, so first get indices to the
  // start of both arrays. Then, offset them by the block index (1 per point).
  extern __shared__ double chunk[];
  double* sq_dists = chunk;
  int* centroid_idxs = (int*)&sq_dists[k]; 

  // Use block index for point and thread index into centroids.
  int p_idx = blockIdx.x;
  int c_idx = threadIdx.x;
  int tid = threadIdx.x;

  if (tid < k) {
    sq_dists[tid] = sq_distance(&data[p_idx*dims], &centroids[c_idx*dims], dims);
    centroid_idxs[tid] = tid;
  } 

  // Parallel reduce to find minimum sq_dist
  for (int s = blockDim.x/2; s > 0; s /= 2) {
    // tid < s:     Thread is in the first half of the block
    // tid + s < k: the upper index to compare with (tid + s)
    //              is not out of bounds of sq_dists and centroid_idxs (size k)
    if(tid < s && tid + s < k) {
      if(sq_dists[centroid_idxs[tid]] < sq_dists[centroid_idxs[tid + s]]) {
        centroid_idxs[tid] = centroid_idxs[tid];
      } else {
        centroid_idxs[tid] = centroid_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  // At this point, centroid_idxs[0] contains the index of the closest centroid
  // Thread 0 writes it into the labels array.
  if(tid == 0) {
    labels[p_idx] = centroid_idxs[tid];
  }
}


// Step 1 of Average
// Add a data point to the closest centroid to compute the sum.
// - 1 block per point, 1 thread per coordinate
// - Uses global memory
// Note: This kernel does not require shared memory.
__global__ void sum_labeled_centroids_k(double* d_data, int N, int dims, 
    double* d_centroids, int k, int* d_labels, int* d_labels_counts) {
  
  int p_idx = blockIdx.x;   // Block index is the point index
  int i = threadIdx.x;      // Thread index is the dimension
  int tid = threadIdx.x;    // Convenience alias
  
  // Get start address of point and handled by this block
  double* point = &d_data[p_idx * dims];
  // Get index of closest centroid and starting address of centroid
  int centroid_idx = d_labels[p_idx];
  double* centroid = &d_centroids[centroid_idx * dims];
  
  // Add this coordinate to new centroids array at index in labels, atomically
  if(i < dims) {
    atomicAdd(&centroid[i], point[i]);
  }

  // Thread 0 is responsible for incrementing label counts for this point
  if(tid == 0) {
    atomicAdd(&d_labels_counts[centroid_idx], 1);
  }
}


// Step 2 of Average (Global)
// Divide each centroid sum by the count to compute the average
// - 1 block per centroid, 1 thread per coordinate
// - Uses global memory
__global__ void divide_labeled_centroids_k(double* d_centroids, int k, 
    int dims, int* d_labels_counts) {
  
  int c_idx = blockIdx.x;   // Block index is the centroid index
  int i = threadIdx.x;      // Thread index is the dimension

  // Get starting address of centroid
  double* centroid = &d_centroids[c_idx * dims];

  // Divide coordinates of this centroid by the count
  if(i < dims) {
    centroid[i] /= d_labels_counts[c_idx];
  }
}

// Step 2 of Average (Shared)
// Divide each centroid sum by the count to compute the average
// - 1 block per centroid, 1 thread per coordinate
// - Uses global memory
__global__ void divide_labeled_centroids_shared_k(double* d_centroids, int k, 
    int dims, int* d_labels_counts) {

  extern __shared__ double centroid[];
  
  int c_idx = blockIdx.x;   // Block index is the centroid index
  int i = threadIdx.x;      // Thread index is the dimension

  // Get starting address of centroid
  double* d_centroid = &d_centroids[c_idx * dims];
  // Copy global memory into shared
  if(i < dims) {
    centroid[i] = d_centroid[i];
  }
  // Divide coordinates of this centroid (in shared memory) by the count
  if(i < dims) {
    centroid[i] /= d_labels_counts[c_idx];
  }
  // Copy shared memory back to global
  if(i < dims) {
    d_centroid[i] = centroid[i];
  }
}

// Note: Unused Function
// This function determined convergence based on changes in individual
// coordinates. This method was dropped in favor of Euclidean distance.
__global__ void converged_abs_k (double* d_centroids, double* d_old_centroids, 
    int k, int dims, double threshold, unsigned int* d_over_threshold) {
  
  int c_idx = blockIdx.x;   // Block index is the centroid index
  int i = threadIdx.x;      // Thread index is the dimension

  // Get starting address of centroid
  double* centroid = &d_centroids[c_idx * dims];
  double* old_centroid = &d_old_centroids[c_idx * dims];
  double delta = 0; // thread-local variable

  if(i < dims) {
    delta = abs(centroid[i] - old_centroid[i]);
  }

  // If this coordinate delta is over threshold and indicator d_over_threshold
  // (unsigned int acting as bool) isn't set yet, toggle to 1.
  if(delta > threshold && *d_over_threshold == 0) {
    atomicOr(d_over_threshold, 1U);
  }
}

// Tests convergence using Euclidean distance between old centroid 
// and new centroid. Launch 1 thread block and use 1 thread for each centroid.
__global__ void converged_k(double* d_centroids, double* d_old_centroids, 
    int k, int dims, double threshold, unsigned int* d_over_threshold) {
    
  int c_idx = threadIdx.x; // Thread index is the centroid index

  // Get starting address of current and old centroids
  double* centroid = &d_centroids[c_idx*dims];
  double* old_centroid = &d_old_centroids[c_idx*dims];

  double sq_dist = 0;

  if(c_idx < k) {
    sq_dist = sq_distance(centroid, old_centroid, dims);
  }

  // Note: For positive numbers, d^2 > t^2 => d > t.
  // If this coordinate delta is over threshold and indicator d_over_threshold
  // (unsigned int acting as bool) isn't set yet, toggle to 1.
  if(sq_dist > threshold * threshold && *d_over_threshold == 0) {
    atomicOr(d_over_threshold, 1U);
  }
}


__global__ void converged_shared_k(double* d_centroids, 
    double* d_old_centroids, int k, int dims, double threshold, 
    unsigned int* d_over_threshold) {

  // Divide shared memory chunk into 2 arrays: centroid and old_centroid
  extern __shared__ double chunk[];
  double* centroid = chunk;
  double* old_centroid = &chunk[dims];

  int c_idx = threadIdx.x; // Thread index is the centroid index

  // Copy global memory to shared
  memcpy(centroid, &d_centroids[c_idx*dims], sizeof(double) * dims);
  memcpy(old_centroid, &d_old_centroids[c_idx*dims], sizeof(double) * dims);

  double sq_dist = 0;

  if(c_idx < k) {
    sq_dist = sq_distance(centroid, old_centroid, dims);
  }

  if(sq_dist > threshold * threshold && *d_over_threshold == 0) {
    atomicOr(d_over_threshold, 1U);
  }

  // Copy shared memory back to global
  memcpy(&d_centroids[c_idx*dims], centroid, sizeof(double) * dims);
  memcpy(&d_old_centroids[c_idx*dims], old_centroid, sizeof(double) * dims);
}



////////////////////////////////////////////////////////////////////////////////
// Wrappers
////////////////////////////////////////////////////////////////////////////////

// The pointers passed in are device pointers
void nearest_centroids_gpu(double* d_data, int N, int dims, 
    double* d_centroids, int k, int* d_labels) {

#ifdef CUDA
  // printf("Calculating nearest centroids with global memory...\n");
  // Without shared memory
  // Create two arrays: one for square distances from points (N) centroids (k)
  // and another array (same size) for parallel reduce on centroid indices
  double* d_sq_dists;
  checkCudaErrors(cudaMalloc(&d_sq_dists, sizeof(double) * N * k));
  int* d_centroid_idxs;
  checkCudaErrors(cudaMalloc(&d_centroid_idxs, sizeof(int) * N * k));

  unsigned int num_blocks = N;
  unsigned int threads_per_block = 32;
  nearest_centroids_k <<<num_blocks, 
                         threads_per_block>>> 
                         (d_data, N, dims, d_centroids, k, d_labels, d_sq_dists, d_centroid_idxs);

  getLastCudaError("nearest_centroids_k execution failed");
  checkCudaErrors(cudaFree(d_sq_dists));
  checkCudaErrors(cudaFree(d_centroid_idxs));

#elif CUDA_SHARED
  // printf("Calculating nearest centroids with shared memory...\n");
  // With shared memory
  // Give each block shared memory for distances to centroids, and indices.
  unsigned int num_blocks = N;
  unsigned int threads_per_block = 32;
  unsigned int shared_sz1 = sizeof(double) * k; // bytes per block, distances
  unsigned int shared_sz2 = sizeof(int) * k;    // bytes per block, indices

  nearest_centroids_shared_k <<<num_blocks,
                                threads_per_block, 
                                shared_sz1 + shared_sz2>>>
                                (d_data, N, dims, d_centroids, k, d_labels);

  getLastCudaError("nearest_centroids_shared_k execution failed");
#endif
}

void average_labeled_centroids_gpu(double* d_data, int N, int dims, 
    double* d_centroids, int k, int* d_labels) {

  // Allocate memory for counts and initialize to 0
  int* d_labels_counts;
  checkCudaErrors(cudaMalloc(&d_labels_counts, sizeof(int) * k));
  checkCudaErrors(cudaMemset(d_labels_counts, 0, sizeof(int) * k));
  // Clear memory for centroids, so we can reuse it to calculate new centroids.
  // The centroids were already copied to old_centroids for convergence check.
  checkCudaErrors(cudaMemset(d_centroids, 0, sizeof(double) * k * dims));

  // 1. Sum points mapped to each centroid
  // Only 1 version
  unsigned int num_blocks = N;          // one block for each point
  unsigned int threads_per_block = 32;  // one thread for each dimension
  sum_labeled_centroids_k <<<num_blocks, 
                             threads_per_block>>> 
                             (d_data, N, dims, d_centroids, k, d_labels,
                             d_labels_counts);
  
  getLastCudaError("sum_labeled_centroids_k execution failed");

  // 2. Divide point sum by count to get the average (a new centroid)
  // Two versions: global and shared
#ifdef CUDA
  num_blocks = k;                       // one block per centroid
  threads_per_block = 32;               // one thread for each dimension
  divide_labeled_centroids_k <<<num_blocks,
                               threads_per_block>>>
                               (d_centroids, k, dims, d_labels_counts);

  getLastCudaError("divide_labeled_centroids_k execution failed");

#elif CUDA_SHARED
  num_blocks = k;                       // one block per centroid
  threads_per_block = 32;               // one thread for each dimension
  int shared_sz = sizeof(double) * dims;    // shared memory holds 1 centroid
  divide_labeled_centroids_shared_k <<<num_blocks,
                                       threads_per_block,
                                       shared_sz>>>
                                       (d_centroids, k, dims, d_labels_counts);

  getLastCudaError("divide_labeled_centroids_shared_k execution failed");
#endif
  // Free memory for counts
  checkCudaErrors(cudaFree(d_labels_counts));
}


bool converged_gpu(double* d_centroids, double* d_old_centroids, int k, int   
    dims, double threshold) {

  // Declare unsigned int over_threshold to act as a boolean 
  // (0 = false, 1 = true) to indicate whether any coordinate deltas 
  // are above threshold.
  unsigned int over_threshold = 0;
  unsigned int* d_over_threshold;
  // Allocate memory and copy to device.
  checkCudaErrors(cudaMalloc(&d_over_threshold, sizeof(unsigned int)));
  mem_stopwatch.start();
  checkCudaErrors(cudaMemcpy(d_over_threshold, &over_threshold, sizeof(unsigned int), cudaMemcpyHostToDevice));
  mem_stopwatch.stop();

#ifdef CUDA
  unsigned int num_blocks = 1;
  unsigned int threads_per_block = k;
  converged_k <<<num_blocks, 
                 threads_per_block>>> 
                 (d_centroids, d_old_centroids, k, dims, threshold,
                  d_over_threshold);
  
  getLastCudaError("converged_k execution failed");

#elif CUDA_SHARED
  unsigned int num_blocks = 1;
  unsigned int threads_per_block = k;
  unsigned int shared_sz1 = sizeof(double) * dims;
  unsigned int shared_sz2 = sizeof(double) * dims;
  converged_k <<<num_blocks, 
                 threads_per_block,
                 shared_sz1 + shared_sz2>>> 
                 (d_centroids, d_old_centroids, k, dims, threshold,
                  d_over_threshold);
  
  getLastCudaError("converged_shared_k execution failed");
#endif
  // Copy result back to host and free memory
  mem_stopwatch.start();
  checkCudaErrors(cudaMemcpy(&over_threshold, d_over_threshold, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  mem_stopwatch.stop();
  checkCudaErrors(cudaFree(d_over_threshold));
  
  return over_threshold == 0 ? true : false;
}


////////////////////////////////////////////////////////////////////////////////
// K Means 
////////////////////////////////////////////////////////////////////////////////

// Future improvement: refactor this function into a kmeans kernel, 
// which launches the other kernels as intermediate steps (dynamic parallelism).
struct clustering_data_gpu kmeans_gpu(double* data, int N, 
    struct options_t* opts) {

  // Start timer for total time
  total_stopwatch.start();

  // Make shorter variables
  int k = opts->num_clusters;
  int m = opts->max_num_iter;
  int dims = opts->dims;
  float threshold = opts->threshold;

  // Initialize centroids & labels
  int* labels = new int[N];
  double* centroids = new double[k*dims]; // freed by result struct's destructor
  random_centroids(k, dims, N, data, centroids);

  // Declare device pointers
  double* d_data;
  int*    d_labels;
  double* d_centroids;
  double* d_old_centroids;

  // Allocate memory on device
  checkCudaErrors(cudaMalloc(&d_data, sizeof(double) * N * dims));
  checkCudaErrors(cudaMalloc(&d_labels, sizeof(int) * N));
  checkCudaErrors(cudaMalloc(&d_centroids, sizeof(double) * k * dims));
  checkCudaErrors(cudaMalloc(&d_old_centroids, sizeof(double) * k * dims));

  // Copy host memory to device memory
  mem_stopwatch.start();
  checkCudaErrors(cudaMemcpy(d_data, data, sizeof(double) * N * dims, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_labels, labels, sizeof(int) * N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_centroids, centroids, sizeof(double) * k * dims, cudaMemcpyHostToDevice));
  mem_stopwatch.stop();
  // d_old_centroids is initialized in 1st iteration as copy of d_centroids

  // Book-keeping
  int iterations = 0;
  double* old_centroids = new double[k*dims];

  // Timer 2/4: Start algo timer
  // checkCudaErrors(cudaEventRecord(start_algo));
  algo_stopwatch.start();

  // Core algorithm
  bool done = false;
  while(!done) {

    checkCudaErrors(cudaMemcpy(d_old_centroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToDevice));
    ++iterations;

    // Write label of nearest centroid into output array labels
    nearest_centroids_gpu(d_data, N, dims, d_centroids, k, d_labels);

    mem_stopwatch.start();
    checkCudaErrors(cudaMemcpy(labels, d_labels, N*sizeof(int), cudaMemcpyDeviceToHost));
    mem_stopwatch.stop();

    // Calculate new centroids as average of labeled points 
    // that map to each centroid.   
    average_labeled_centroids_gpu(d_data, N, dims, d_centroids, k, d_labels);
    mem_stopwatch.start();
    checkCudaErrors(cudaMemcpy(centroids, d_centroids, sizeof(double) * k * dims, cudaMemcpyDeviceToHost));
    mem_stopwatch.stop();

    // Test for max iterations or convergence
    done = (iterations == m || 
            converged_gpu(d_centroids, d_old_centroids, k, dims, threshold));
  }

  // Timer 3/4: Stop algo timer
  // checkCudaErrors(cudaEventRecord(stop_algo));
  algo_stopwatch.stop();

  // Copy memory back to host
  mem_stopwatch.start();
  checkCudaErrors(cudaMemcpy(labels, d_labels, sizeof(int) * N, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(centroids, d_centroids, sizeof(double) * k * dims, cudaMemcpyDeviceToHost));
  mem_stopwatch.stop();

  // Free memory on device
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_labels));
  checkCudaErrors(cudaFree(d_centroids));
  checkCudaErrors(cudaFree(d_old_centroids));

  // Stop timer for total time
  total_stopwatch.stop();

  // Return a struct clustering_data_gpu
  return { 
    iterations,
    algo_stopwatch.elapsed(),
    centroids,
    labels
  };
}
