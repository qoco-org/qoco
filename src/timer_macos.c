#include "timer.h"

void start_timer(QOCOTimer* timer) { timer->tic = mach_absolute_time(); }

void stop_timer(QOCOTimer* timer) { timer->toc = mach_absolute_time(); }

QOCOFloat get_elapsed_time_sec(QOCOTimer* timer)
{
  uint64_t duration; /* Elapsed time in clock cycles*/

  duration = timer->toc - timer->tic;

  /* Conversion from clock cycles to nanoseconds*/
  mach_timebase_info(&(timer->tinfo));
  duration *= timer->tinfo.numer;
  duration /= timer->tinfo.denom;

  return (QOCOFloat)duration / 1e9;
}