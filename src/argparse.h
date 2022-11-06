#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
// #include <stdlib.h>
#include <iostream>
#include <sstream>
#include <vector>

struct options_t {
  int num_clusters;       // -k: number of clusters
  int dims;               // -d: dimension of the points
  char* inputfilename;          // -i: input filename
  int max_num_iter;       // -m: maximium number of iterations
  double threshold;       // -t: threshold for the convergence test
  bool output_centroids;  // -c: true: outputs centroid of all clusters
                          //     false: output labels of all points
  int seed;               // -s: seed for rand()
};

void print_opts(struct options_t* opts);

void set_default_opts(struct options_t* opts);

bool contains_undefined_opts(struct options_t* opts);

void get_opts(int argc, char** argv, struct options_t* opts);

#endif