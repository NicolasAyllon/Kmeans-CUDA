#include <io.h>

using std::vector;

// Create filename by concatenating prefix, iteration number n, and suffix.
// Note: suffix must include a dot "." (Example: ".txt")
std::string make_filename(const std::string& prefix, int n, 
    const std::string& suffix) {
  return prefix + std::to_string(n) + suffix;
}

// Note: num_points and input_data are output parameters.
vector<vector<double>> read_file(struct options_t* opts, int& num_points) {
  
  // Read number of points, given at the first line of the file.
  std::ifstream infile(opts->inputfilename);
  if (!infile.is_open()) {
    std::cout << "Error: can't find file \"" 
              << std::string(opts->inputfilename) << "\"\nExiting...\n";
    exit(EXIT_FAILURE);
  }
  infile >> num_points;
  infile.ignore(); // Ignore 1 character: the following newline '\n'

  // Create vector for data and reserve space
  vector<vector<double>> data(num_points);

  // Read the data and fill in the vector
  for(int p = 0; p < num_points; ++p) {
    std::string line;
    std::getline(infile, line);
    std::stringstream ss(line);

    int point_number; // This value is intentionally unused.
    ss >> point_number;
    // infile.ignore(std::numeric_limits<std::streamsize>::max(), ' '); 
    // Ignore point number at start of line. Parallel version may need this.
    
    data[p].assign(opts->dims, 0);
    // Fill with 0 for every dimension,
    // so operator[] can be used below.
    
    for(int d = 0; d < opts->dims; ++d) {
      ss >> data[p][d];
    }
  }

  return data;
}

void read_file(struct options_t* opts, double** p_data, int* num_points) {
  // Open file
  std::ifstream infile(opts->inputfilename);
    if (!infile.is_open()) {
    std::cout << "Error: can't find file \"" 
              << std::string(opts->inputfilename) << "\"\nExiting...\n";
    exit(EXIT_FAILURE);
  }
  // Get number of points (metadata on first line), store output parameter
  infile >> *num_points;
  infile.ignore(); // Ignore 1 character: the following newline '\n'

  // Allocate 1-dimensional input array, store output parameter 
  *p_data = (double*) malloc(*num_points * opts->dims * sizeof(double));

  // Read the data and fill in the array
  for(int n = 0; n < *num_points; ++n) {
    std::string line;
    std::getline(infile, line);
    std::stringstream ss(line);

    int point_number; // This value is intentionally unused.
    ss >> point_number;

    for(int d = 0; d < opts->dims; ++d) {
      // 1D array index calculated using point index n and dim index d
      ss >> (*p_data)[n*opts->dims + d];
    }
  }
}

// Write file for intermediate data (centroids) for vectors (sequential)
void write_file(struct options_t* opts, const vector<vector<double>>& data, const std::string& filename) {
  // trunc discards old file contents before writing
  std::ofstream outfile(filename, std::ofstream::trunc);
  if(!outfile.is_open()) {
    // std::cout << "Error opening file " << filename << "\n";
    printf ("Error opening file %s: %s\n", filename.c_str(), strerror(errno));
    return;
  }
  const int decimal_places = 12;
  outfile.precision(decimal_places);

  for(size_t n = 0; n < data.size(); ++n) {
    outfile << n << " "; // 0-based index
    // print coordinates followed by space (all but the last one)
    for(size_t d = 0; d < opts->dims-1; ++d) {
      outfile << data[n][d] << " ";
    }
    // print last coordinate without space
    outfile << data[n][opts->dims-1] << "\n";
  }
}

// Write file for intermediate data (labels) for vectors (sequential)
void write_file(struct options_t* opts, const vector<int>& labels, 
    const std::string& filename) {
  std::ofstream outfile(filename, std::ofstream::trunc);
  if(!outfile.is_open()) {
    printf ("Error opening file %s: %s\n", filename.c_str(), strerror(errno));
    return;
  }
  for(size_t i = 0; i < labels.size(); ++i) {
    outfile << i << " " << labels[i] << '\n';
  }
}

void write_file(struct options_t* opts, double* data, int num_points, const std::string& filename) {
  // trunc discards old file contents before writing
  std::ofstream outfile(filename, std::ofstream::trunc);
  if(!outfile.is_open()) {
    printf ("Error opening file %s: %s\n", filename.c_str(), strerror(errno));
    return;
  }
  const int decimal_places = 12;
  outfile.precision(decimal_places);

  // outfile << num_points << '\n'; // temporary format to match input file

  for(int n = 0; n < num_points; ++n) {
    // outfile << (n+1) << " "; // temporary format to match input file
    outfile << n << " ";
    // print coordinates followed by space (all but the last one)
    for(int d = 0; d < opts->dims-1; ++d) {
      outfile << data[n*opts->dims + d] << " ";
    }
    // print last coordinate without space
    outfile << data[n*opts->dims + opts->dims-1] << "\n";
  }
}

void write_file(struct options_t* opts, int* labels, int num_points, 
    const std::string& filename) {
  std::ofstream outfile(filename, std::ofstream::trunc);
  if(!outfile.is_open()) {
    printf ("Error opening file %s: %s\n", filename.c_str(), strerror(errno));
    return;
  }
  for(int i = 0; i < num_points; ++i) {
    outfile << i << " " << labels[i] << '\n';
  }
}

// Print output for CPU kmeans
void print_output(const struct clustering_data& results, 
    struct options_t* opts) {

  // Number of iterations and average iteration time
  float time_per_iter_in_ms = 
    static_cast<float>(results.elapsed_milliseconds) / results.iterations;
    printf("%d, %lf\n", results.iterations, time_per_iter_in_ms);

  // Depending on options, print either centroids or labels:
  // Centroids
  if (opts->output_centroids) {
    for (size_t i = 0; i < results.centroids.size(); ++i) { 
      printf("%lu ", i);
      for (int d = 0; d < opts->dims-1; d++) {
        printf("%.5f ", results.centroids[i][d]);
      }
      printf("%.5f", results.centroids[i][opts->dims-1]);
      printf("\n");
    }
  } 
  // Labels
  else {
    printf("clusters:");
    for (size_t i = 0; i < results.labels.size(); ++i) {
      printf(" %d", results.labels[i]);
    }
    printf("\n");
  }
}

// Print output for GPU kmeans
void print_output(const struct clustering_data_gpu& results, 
    struct options_t* opts, int num_points) {

  // Number of iterations and average iteration time
  float time_per_iter_in_ms = results.elapsed_milliseconds / results.iterations;
  printf("%d, %lf\n", results.iterations, time_per_iter_in_ms);
  // Depending on options, print either centroids or labels:
  // Centroids
  if (opts->output_centroids) {
    for (size_t i = 0; i < opts->num_clusters; ++i) { 
      printf("%lu ", i);
      // Get starting address of centroid i
      double* centroid = &results.centroids[i * opts->dims];
      for (int d = 0; d < opts->dims-1; d++) {
        printf("%.5f ", centroid[d]);
      }
      printf("%.5f", centroid[opts->dims-1]);
      printf("\n");
    }
  } 
  // Labels
  else {
    printf("clusters:");
    for (int i = 0; i < num_points; ++i) {
      printf(" %d", results.labels[i]);
    }
    printf("\n");
  }
}