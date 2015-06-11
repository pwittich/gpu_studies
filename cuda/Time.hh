#ifndef PWTIME
#define PWTIME
#include <chrono>
// stolen from mtorture
typedef std::chrono::time_point<std::chrono::high_resolution_clock> timepoint;
typedef std::chrono::duration<double> tick;

static timepoint  now()
{
  return std::chrono::system_clock::now();
}

static tick  delta(timepoint& t0)
{
  timepoint t1(now());
  tick d = t1 - t0;
  t0 = t1;
  return d;
}

#endif // PWTIME
