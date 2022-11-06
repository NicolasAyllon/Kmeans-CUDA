#ifndef _KMEANS_H
#define _KMEANS_H

#include <vector>
#include <chrono>
#include <random.h>
#include <argparse.h>
#include <log.h>
#include <io.h>

using std::vector;

struct clustering_data {
  int iterations;
  int elapsed_milliseconds;
  vector<vector<double>> centroids;
  vector<int> labels;
};

bool converged(const vector<vector<double>>& centroids, 
    const vector<vector<double>>& old_centroids, float threshold);

double sq_distance(const vector<double>& a, const vector<double>& b);

int nearest_centroid_index(const vector<double>& point, 
    const vector<vector<double>>& centroids);

vector<int> nearest_centroids(const vector<vector<double>>& data, 
    const vector<vector<double>>& centroids);

vector<vector<double>> average_labeled_centroids
    (const vector<vector<double>>& data, const vector<int>& labels, int k);

struct clustering_data kmeans(const vector<vector<double>>&, 
    struct options_t* opts);

#endif