#include <kmeans.h>

using std::vector;

// TODO: Possible revision is to check whether the absolute value difference is less than the threshold for all coordinates of all points.
// Returns whether all centroids have converged.
// A centroid is defined to have converged when the Euclidean distance
// between its new and previous position is less than the threshold.
bool converged(const vector<vector<double>>& centroids, 
    const vector<vector<double>>& old_centroids, float threshold) {
  for(size_t i = 0; i < centroids.size(); ++i) {
    // Note: For positive numbers, d^2 > t^2 => d > t.
    // Here, the squared distance is compared to the squared threshold.
    if(sq_distance(centroids[i], old_centroids[i]) > threshold * threshold) {
      return false;
    }
  }
  return true;
}

// Returns the squared distance between vectors a and b.
// Note: vectors a and b shall have the same length.
double sq_distance(const vector<double>& a, const vector<double>& b) {
  double result = 0;
  for(size_t i = 0; i < a.size(); ++i) {
    result += (b[i] - a[i]) * (b[i] - a[i]);
  }
  return result;
}

// Returns the index (in centroids) of the centroid nearest to point.
int nearest_centroid_index(const vector<double>& point, 
    const vector<vector<double>>& centroids) {
  int index = 0;
  double min_sq_distance = std::numeric_limits<double>::infinity();
  for(size_t i = 0; i < centroids.size(); ++i) {
    if(sq_distance(point, centroids[i]) < min_sq_distance) {
      min_sq_distance = sq_distance(point, centroids[i]);
      index = i;
    }
  }
  return index;
}

// Returns a vector of the same size as data where each entry is the index 
// (in centroids) of the centroid closest to that point.
vector<int> nearest_centroids(const vector<vector<double>>& data, 
    const vector<vector<double>>& centroids) {
  vector<int> result(data.size());
  for(size_t i = 0; i < data.size(); ++i) {
    // get index of closest centroid
    result[i] = nearest_centroid_index(data[i], centroids);
  }
  return result;
}


// Returns a new set of centroids calculated as the average position of all
// points mapped to each centroid.
// Strategy:
// Create vectors holding:
//   S: the sum of position vectors (points) mapping to that centroid. 
//   N: the number of points mapping to that centroid
// Then, average points by dividing each sum of vectors (S) by the count (N).
vector<vector<double>> average_labeled_centroids
    (const vector<vector<double>>& data, const vector<int>& labels, int k) {
  // Initialize k vectors with dim elements, initially at 0.
  vector<vector<double>> centroids(k);
  size_t dims = data.front().size();
  for(auto& centroid : centroids) { centroid.assign(dims, 0); }
  vector<int> counts(k, 0); // Initialize counts: {0, 0, 0, ... 0} of length k

  // Iterate through data and sum positions for respective centroids.
  for(size_t i = 0; i < data.size(); ++i) {
    // Increment count at the index specified by the label.
    counts[labels[i]]++;
    // Add coordinates to centroid at that index.
    for(size_t d = 0; d < dims; ++d) { 
      // std::cout << "i=" << i << ", d=" << d << std::endl;
      centroids[labels[i]][d] += data[i][d];
    }
  }
  // At this point, centroids holds the *sum* of points mapping to them.
  // Divide sum of positions by count to compute the average position.
  for(size_t i = 0; i < centroids.size(); ++i) {
    for(size_t d = 0; d < dims; ++d) {
      centroids[i][d] /= counts[i];
    }
  }
  // Now, centroids holds the *average* of points mapping to them, so return.
  return centroids;
}


// Performs k-means clustering on data with specified options (containing k).
// Returns iterations, time (ms), centroids, and labels in struct.
struct clustering_data kmeans(const vector<vector<double>>& data, 
    struct options_t* opts) {
  // Start timer
  auto start = std::chrono::high_resolution_clock::now();

  // Make short readable names
  int k = opts->num_clusters;
  int m = opts->max_num_iter;
  float threshold = opts->threshold;

  // Initialize centroids & labels
  auto centroids = random_centroids(k, data);
  auto labels = vector<int>(data.size());
  // write_file(opts, centroids, "test/cpu/n-" + std::to_string(data.size()) 
  //                           + "/centroids/initial.txt");

  // Book-keeping
  int iterations = 0;
  vector<vector<double>> old_centroids(k);

  // Core algorithm
  bool done = false;
  while(!done) {

    old_centroids = centroids;
    ++iterations;

    // Labels identifies the index of the centroid closest to each point.
    labels = nearest_centroids(data, centroids);
    // write_file(opts, labels, "test/cpu/n-" + std::to_string(data.size())
    //                        + "/labels/iter_" + std::to_string(iterations)
    //                        + ".txt");

    // print_container_items_by_line(labels);

    // Calculate new centroids as the average (centroid) of labeled points
    // that map to each centroid.
    centroids = average_labeled_centroids(data, labels, k);
    // write_file(opts, centroids, "test/cpu/n-" + std::to_string(data.size()) 
    //                           + "/centroids/itr_" + std::to_string(iterations)
    //                           + ".txt");
    
    // Test for convergence
    done = (iterations == m || converged(centroids, old_centroids, threshold));
  }

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();
  int elapsed_milliseconds = 
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // write_file(opts, centroids, "test/cpu/n-" + std::to_string(data.size()) 
  //                           + "/centroids/final.txt");
  // write_file(opts, labels, "test/cpu/n-" + std::to_string(data.size()) 
  //                        + "/labels/final.txt");

  // Create struct clustering_data using braced-init-list and return.
  return {
    iterations,
    elapsed_milliseconds,
    centroids,
    labels
  };
}