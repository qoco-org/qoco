/**
 * @file timer_fallback.c
 * @author Govind M. Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind M. Chari
 * This source code is licensed under the BSD 3-Clause License
 */
#include "timer.h"

#if !defined(IS_LINUX) && !defined(IS_MACOS) && !defined(IS_WINDOWS) && !defined(QOCO_TIMER_CUSTOM)

void start_timer(QOCOTimer* timer)
{
  // No-op for fallback timer
  (void)timer;
}

void stop_timer(QOCOTimer* timer)
{
  // No-op for fallback timer
  (void)timer;
}

QOCOFloat get_elapsed_time_sec(QOCOTimer* timer)
{
  // Always return 0 for fallback timer
  (void)timer;
  return 0.0;
}

#endif
