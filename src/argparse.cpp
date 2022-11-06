#include <argparse.h>

// For testing and debugging
void print_opts(struct options_t* opts) {
  std::cout << "Options Data:" << std::endl;
  std::cout << "\t-i: " << 
    (opts->inputfilename ? std::string(opts->inputfilename) : "nullptr")
    << std::endl; 
  std::cout << "\t-k: " << opts->num_clusters << std::endl;
  std::cout << "\t-d: " << opts->dims         << std::endl;
  std::cout << "\t-m: " << opts->max_num_iter << std::endl;
  std::cout << "\t-t: " << opts->threshold    << std::endl;
  std::cout << "\t-c: " << opts->output_centroids << std::endl;
  std::cout << "\t-s: " << opts->seed         << std::endl;
}

void set_default_opts(struct options_t* opts) {
    //Set Default Values
  opts->num_clusters = -1;
  opts->dims = 0;
  opts->inputfilename = nullptr;
  opts->max_num_iter = 0;
  opts->threshold = 0;
  // Set option (-c) output_centroids to false by default.
  // By default the program will output labels of all points.
  opts->output_centroids = false;
  opts->seed = 0;
}

bool contains_undefined_opts(struct options_t* opts) {
  return (opts->num_clusters == -1 ||
          opts->dims == 0 ||
          opts->inputfilename == nullptr ||
          opts->max_num_iter == 0 ||
          opts->threshold == 0 ||
          opts->seed == 0);
}

std::string get_undefined_opts_string(struct options_t* opts) {
  std::vector<char> v;
  if (opts->num_clusters == -1)        v.push_back('k');
  if (opts->dims == 0)                 v.push_back('d');
  if (opts->inputfilename == nullptr)  v.push_back('i');
  if (opts->max_num_iter == 0)         v.push_back('m');
  if (opts->threshold == 0)            v.push_back('t');
  if (opts->seed == 0)                 v.push_back('s');

  if (v.empty()) return "";
  // Write option characters separated by commas ","
  std::stringstream result;
  for (auto it = v.begin(); it != v.end()-1; ++it) {
    result << *it << ", ";
  }
  result << v.back();

  // Get data from stringstream as string and return.
  return result.str();
}

void get_opts(int argc, char** argv, struct options_t* opts) {
  // print_opts(opts);

  if(argc == 1) {
    std::cout << "Usage:" << std::endl;
    std::cout << "\t-i <inputfilename>" << std::endl;
    std::cout << "\t-k <num_clusters>" << std::endl;
    std::cout << "\t-d <dimensions>" << std::endl;
    std::cout << "\t-m <max_num_iterations>" << std::endl;
    std::cout << "\t-t <threshold>" << std::endl;
    std::cout << "\t-c <output centroids, labels if absent>" << std::endl;
    std::cout << "\t-s <seed>" << std::endl;
    exit(EXIT_SUCCESS);
  }

  //Set Default Values
  set_default_opts(opts);
  //print_opts(opts);

  int c = 0;
  // char* optarg;  // stores string following option character
  // int optopt;    // stores unrecognized option character
  while((c = getopt(argc, argv, "i:k:d:m:t:cs:")) != -1) {
    // Debugging
    // print_opts(opts);
    // std::cout << "c: " << (char)c << std::endl;
    // std::cout << "ind: " << ind << std::endl;
    // std::cout << "optind: " << optind << std::endl;
    // std::cout << "optarg: " << (optarg ? optarg : "nullptr") << std::endl;
    // std::cin.get();

    switch (c) {

      case 'i':
        opts->inputfilename = optarg;
        break;
      case 'k':
        opts->num_clusters = atoi(optarg);
        break;
      case 'd':
        opts->dims = atoi(optarg);
        break;
      case 'm':
        opts->max_num_iter = atoi(optarg);
        break;
      case 't':
        opts->threshold = strtod(optarg, NULL);
        if(opts->threshold == 0) {
          std::cout << "Error: can't parse " << optarg << " to double." 
                    << std::endl;
          exit(EXIT_FAILURE);
        }
        break;
      case 'c':
        opts->output_centroids = true;
        break;
      case 's':
        opts->seed = atoi(optarg);
        break;
      default:
        std::cout << "Error: unknown option or missing argument." << std::endl;
        exit(EXIT_FAILURE);
        break;
    }
  }
  // std::cout << "We made it out of the while loop." << std::endl;
  //print_opts(opts);

  if(contains_undefined_opts(opts)) {
    std::cout << "Error: these options have missing or invalid values: " 
              <<  get_undefined_opts_string(opts) << std::endl;
    exit(EXIT_FAILURE);
  }
}