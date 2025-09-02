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

/* Indicate whether we have a real timer impl */
#if defined(QOCO_TIMER_CUSTOM) || defined(IS_LINUX) || defined(IS_MACOS) || defined(IS_WINDOWS)
#  define QOCO_TIMER_AVAILABLE 1
#else
#  define QOCO_TIMER_AVAILABLE 0
#endif

#ifdef QOCO_TIMER_CUSTOM
/* Your header must define:
 *   typedef ... QOCOTimer;
 *   void      start_timer(QOCOTimer*);
 *   void      stop_timer(QOCOTimer*);
 *   QOCOFloat get_elapsed_time_sec(const QOCOTimer*);
 */
#include "qoco_timer_custom.h"

#elif defined(IS_LINUX)
#  include <time.h>
typedef struct {
  struct timespec tic;
  struct timespec toc;
} QOCOTimer;

#elif defined(IS_MACOS)
#  include <stdint.h>
#  include <mach/mach_time.h>
typedef struct {
  uint64_t tic;
  uint64_t toc;
  mach_timebase_info_data_t tinfo;
} QOCOTimer;

#elif defined(IS_WINDOWS)
#  ifndef NOGDI
#    define NOGDI
#  endif
#  include <windows.h>
typedef struct {
  LARGE_INTEGER tic;
  LARGE_INTEGER toc;
  LARGE_INTEGER freq;
} QOCOTimer;

#else
/* Fallback: no real timer available */
typedef struct {
  int unused_;
} QOCOTimer;
#endif /* platform selection */

#if QOCO_TIMER_AVAILABLE
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
#else
/* Provide no-op inline stubs when timing is unavailable */
static inline void start_timer(QOCOTimer* timer) { (void)timer; }
static inline void stop_timer(QOCOTimer* timer) { (void)timer; }
static inline QOCOFloat get_elapsed_time_sec(QOCOTimer* timer) {
  (void)timer;
  return (QOCOFloat)0.0;
}
#endif

#endif /* #ifndef QOCO_TIMER_H */