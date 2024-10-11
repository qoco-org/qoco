#include "qoco_error.h"

QOCOInt qoco_error(enum qoco_error_code error_code)
{
  printf("ERROR: %s\n", QOCO_ERROR_MESSAGE[error_code]);
  return (QOCOInt)error_code;
}