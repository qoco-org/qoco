#ifndef CONE_H
#define CONE_H

#include "linalg.h"

/**
 * @brief Computes cone product u * v = p
 *
 * @param u
 * @param v
 * @param p
 * @param data
 */
void cone_product(QCOSFloat* u, QCOSFloat* v, QCOSFloat* p,
                  QCOSProblemData* data);

#endif