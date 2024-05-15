#ifndef QCOS_ERROR_H
#define QCOS_ERROR_H

#include "definitions.h"
#include "enums.h"

/**
 * @brief Function to print error messages.
 *
 * @param error_code
 * @return Error code as an QCOSInt.
 */
QCOSInt qcos_error(enum qcos_error_code error_code);

#endif /* #ifndef ERROR_H*/