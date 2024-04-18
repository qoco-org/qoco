/**
 * @file cone.h
 * @author Govind Chari <govindchari1@gmail.com>
 *
 * @section LICENSE
 *
 * Copyright (c) 2024, Govind Chari
 * This source code is licensed under the BSD 2-Clause License
 *
 * @section DESCRIPTION
 *
 * Includes various functions necessary for cone operations.
 */

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