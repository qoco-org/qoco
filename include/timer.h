#ifndef TIMER_H
#define TIMER_H

#include "definitions.h"
#include <time.h>

typedef struct {
  struct timespec tic;
  struct timespec toc;
} QOCOTimer;

/**
 * @brief Starts timer and sets tic field of struct to the current time.
 *
 * @param timer Pointer to timer struct.
 */
void start_timer(QOCOTimer* timer);

/**
 * @brief Stops timer and sets toc field of struct to the current time.
 *
 * @param timer Pointer to timer struct.
 */
void stop_timer(QOCOTimer* timer);

/**
 * @brief Gets time in seconds recorded by timer. Must be called after
 * start_timer() and stop_timer().
 *
 * @param timer Pointer to timer struct.
 */
QOCOFloat get_elapsed_time_sec(QOCOTimer* timer);

#endif /* #ifndef TIMER_H */