#ifndef LINALG_H
#define LINALG_H
#include "definitions.h"
#include "structs.h"

/**
 * @brief Allocates a QCOSVector of length n and zeros out the data.
 *
 * @param n Length of vector to allocate.
 * @return Pointer to allocated QCOSVector.
 */
QCOSVector* qcos_vector_calloc(QCOSInt n);

/**
 * @brief Allocates a new csc matrix and copies A to it.
 *
 * @param A Matrix to copy.
 * @return Pointer to new constructed matrix.
 */
QCOSCscMatrix* new_qcos_csc_matrix(QCOSCscMatrix* A);

/**
 * @brief Constructs a new QCOSVector from raw array.

 *
 * @param x Raw array of data
 * @param n Length of raw array
 * @return Pointer to new vector
 */
QCOSVector* new_qcos_vector_from_array(QCOSFloat* x, QCOSInt n);

/**
 * @brief Copies array of QCOSFloats from x to array y
 *
 * @param x Source array
 * @param y Destination array
 * @param n Length of arrays
 */
void copy_arrayf(const QCOSFloat* x, QCOSFloat* y, QCOSInt n);

/**
 * @brief Copies array of QCOSInts from x to array y
 *
 * @param x Source array
 * @param y Destination array
 * @param n Length of arrays
 */
void copy_arrayi(const QCOSInt* x, QCOSInt* y, QCOSInt n);

#endif /* #ifndef LINALG_H*/
