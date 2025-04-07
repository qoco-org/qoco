#ifndef QOCO_STATUS_H
#define QOCO_STATUS_H

#include "definitions.h"
#include "enums.h"
#include <stdio.h>

/**
 * @brief Function to print error messages.
 *
 * @param error_code
 * @return Error code as an QOCOInt.
 */
QOCOInt qoco_error(enum qoco_error_code error_code);

extern const char* QOCO_ERROR_MESSAGE[];
extern const char* QOCO_SOLVE_STATUS_MESSAGE[];

#endif /* #ifndef QOCO_STATUS_H*/