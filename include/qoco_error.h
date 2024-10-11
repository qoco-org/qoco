#ifndef QOCO_ERROR_H
#define QOCO_ERROR_H

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

#endif /* #ifndef ERROR_H*/