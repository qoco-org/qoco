#ifndef TIMER_H
#define TIMER_H

#include "definitions.h"
#include <time.h>

typedef struct {
  struct timespec tic;
  struct timespec toc;
} QCOSTimer;

/**
 * @brief Starts timer and sets tic field of struct to the current time.
 *
 * @param timer Pointer to timer struct.
 */
void start_timer(QCOSTimer* timer);

/**
 * @brief Stops timer and sets toc field of struct to the current time.
 *
 * @param timer Pointer to timer struct.
 */
void stop_timer(QCOSTimer* timer);

/**
 * @brief Gets time in seconds recorded by timer. Must be called after
 * start_timer() and stop_timer().
 *
 * @param timer Pointer to timer struct.
 */
QCOSFloat get_elapsed_time_sec(QCOSTimer* timer);

#endif /* #ifndef TIMER_H */