#include "timer.h"

void start_timer(QOCOTimer* timer)
{
  clock_gettime(CLOCK_MONOTONIC, &timer->tic);
}

void stop_timer(QOCOTimer* timer)
{
  clock_gettime(CLOCK_MONOTONIC, &timer->toc);
}

QOCOFloat get_elapsed_time_sec(QOCOTimer* timer)
{
  struct timespec temp;

  if ((timer->toc.tv_nsec - timer->tic.tv_nsec) < 0) {
    temp.tv_sec = timer->toc.tv_sec - timer->tic.tv_sec - 1;
    temp.tv_nsec = 1e9 + timer->toc.tv_nsec - timer->tic.tv_nsec;
  }
  else {
    temp.tv_sec = timer->toc.tv_sec - timer->tic.tv_sec;
    temp.tv_nsec = timer->toc.tv_nsec - timer->tic.tv_nsec;
  }
  return (QOCOFloat)temp.tv_sec + (QOCOFloat)temp.tv_nsec / 1e9;
}