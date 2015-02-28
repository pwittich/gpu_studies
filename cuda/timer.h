#ifndef TIMER_H
#define TIMER_H

#include <cstdio>

#include "cuda.h"

class timer
{
 private:
  cudaEvent_t c_start, c_stop;
  const char *msg;
 public:
 timer(const char *m ):
  msg(m)
  {}
  void start_time() {
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start, 0);
  }
  void stop_time(const char *msg) {
    cudaEventRecord(c_stop, 0);
    cudaEventSynchronize(c_stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
    printf("Time to %s: %.3f ms\n", msg, elapsedTime);
  }
};

#endif // TIMER_H
