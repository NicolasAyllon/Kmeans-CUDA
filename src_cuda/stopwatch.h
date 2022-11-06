#ifndef _STOPWATCH_H
#define _STOPWATCH_H

// Idea for timer struct from:
// https://stackoverflow.com/questions/7876624
struct stopwatch {
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  float total_ms;

  stopwatch() {
    total_ms = 0;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
  }

  void start() {
    cudaEventRecord(start_event);
  }

  void stop() {
    float interval_ms;
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&interval_ms, start_event, stop_event);
    total_ms += interval_ms;
  }

  float elapsed() {
    return total_ms;
  }

  void reset() {
    total_ms = 0;
  }

  ~stopwatch() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }
};

#endif