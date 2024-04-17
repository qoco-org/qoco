#ifndef UTILS_H
#define UTILS_H

#include "structs.h"
#include <stdio.h>

/**
 * @brief Prints QCOSCscMatrix.
 *
 * @param M Pointer to QCOSCscMatrix that will be printed.
 */
void print_qcos_csc_matrix(QCOSCscMatrix* M);

/**
 * @brief Prints QCOSVector.
 *
 * @param V Pointer to QCOSVector that will be printed.
 */
void print_qcos_vector(QCOSVector* V);

#endif /* #ifndef UTILS_H */