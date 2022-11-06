#ifndef _LOG_H
#define _LOG_H

#include <iostream>
#include <sstream>

// Template must be defined in header.
template<typename Container>
void print_container_items_by_line(Container& container) {
  int i = 0;
  for(typename Container::const_iterator it = container.begin(); it != container.end(); ++it, ++i) {
    std::cout << "item " << i << ": " << *it << std::endl;
  }
}

template<typename T>
std::string toString(std::vector<T> data) {
  std::stringstream ss;
  if(data.empty()) return "";
  for(auto it = data.begin(); it != data.end() - 1; ++it) {
    ss << *it << " ";
  }
  ss << data.back();
  return ss.str();
}

template<typename T>
void print_data(std::vector<std::vector<T>> data) {
  for(auto v : data) {
    std::cout << toString(v) << std::endl;
  }
}

// Previous version
template <typename T>
void print_2D_vector (const std::vector<std::vector<T>>& data, int decimal_places) {
  // Display up to 12 decimal places
  std::cout.unsetf(std::ios::floatfield);
  std::cout.setf(std::ios::fixed);
  std::cout.precision(decimal_places);

  for(size_t i = 0; i < data.size(); ++i) {
    std::cout << i << " ";
    for(size_t j = 0; j < data[i].size() - 1; ++j) {
      std::cout << data[i][j] << " ";
    }
    if(!data[i].empty())
      std::cout << data[i].back();
    std::cout << '\n';
   }

  //Unset format flag
  std::cout.unsetf(std::ios::floatfield); // unsets std::ios::fixed
}

template <typename T>
void print_points_array(T* arr, size_t num_points, size_t dim) {
  for(size_t n = 0; n < num_points; ++n) {
    for(size_t d = 0; d < dim; ++d) {
      std::cout << arr[n*dim + d] << " ";
    }
    std::cout << '\n';
  }
}

#endif
