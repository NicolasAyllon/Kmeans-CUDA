#ifndef _IO_H
#define _IO_H

#include <argparse.h>
#include <kmeans.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <string>
#include <cstring>

#include <kmeans_kernels.h>

using std::vector;

std::string make_filename(const std::string& prefix, int n, const std::string& suffix);

vector<vector<double>> read_file(struct options_t* opts, int& num_points);

void read_file(struct options_t* opts, double** p_data, int* num_points);

// Points vector
void write_file(struct options_t* opts, const vector<vector<double>>& data, const std::string& filename);

// Labels vector
void write_file(struct options_t* opts, const vector<int>& labels, const std::string& filename);

// Points array
void write_file(struct options_t* opts, double* data, int num_points, const std::string& filename);

// Labels array
void write_file(struct options_t* opts, int* labels, int num_points, const std::string& filename);

void print_output(const struct clustering_data& results, struct options_t* opts);

void print_output(const struct clustering_data_gpu& results, struct options_t* opts, int num_points);

#endif