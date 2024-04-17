#ifndef QCOS_KKT_H
#define QCOS_KKT_H

#include "structs.h"

/**
 * @brief Allocate memory for KKT matrix and fill in blocks with respective
 * matrices.
 *
 * @param data Pointer to problem data.
 * @return Pointer to initialized KKT matrix.
 */
QCOSCscMatrix* initialize_kkt(QCOSProblemData* data);
#endif /* #ifndef QCOS_KKT_H */