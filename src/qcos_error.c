#include "qcos_error.h"

QCOSInt qcos_error(enum qcos_error_code error_code)
{
  printf("ERROR: %s\n", QCOS_ERROR_MESSAGE[error_code]);
  return (QCOSInt)error_code;
}