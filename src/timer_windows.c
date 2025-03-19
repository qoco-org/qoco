#include "timer.h"

void start_timer(QOCOTimer* timer)
{
  QueryPerformanceFrequency(&timer->freq);
  QueryPerformanceCounter(&timer->tic);
}

void stop_timer(QOCOTimer* timer) { QueryPerformanceCounter(&timer->toc); }

QOCOFloat get_elapsed_time_sec(QOCOTimer* timer)
{
  return (timer->toc.QuadPart - timer->tic.QuadPart) /
         (QOCOFloat)timer->freq.QuadPart;
}
