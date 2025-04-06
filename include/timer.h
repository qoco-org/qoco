/**
 * @file timer.h
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 *
 * @section DESCRIPTION
 *
 * Provides timing functions.
 */

#ifndef QOCO_TIMER_H
#define QOCO_TIMER_H

#include "definitions.h"

#ifdef IS_LINUX
#include <time.h>
typedef struct {
  struct timespec tic;
  struct timespec toc;
} QOCOTimer;
#endif

#ifdef IS_MACOS
#include <mach/mach_time.h>
typedef struct {
  uint64_t tic;
  uint64_t toc;
  mach_timebase_info_data_t tinfo;
} QOCOTimer;
#endif

#ifdef IS_WINDOWS
#define NOGDI
#include <windows.h>
typedef struct {
  LARGE_INTEGER tic;
  LARGE_INTEGER toc;
  LARGE_INTEGER freq;
} QOCOTimer;
#endif

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

#endif /* #ifndef QOCO_TIMER_H */