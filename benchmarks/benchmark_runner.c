#include "qoco.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: %s problem.bin\n", argv[0]);
    return 1;
  }

  FILE* f = fopen(argv[1], "rb");
  const char* filename = argv[1];
  if (!f) {
    perror("fopen");
    return 1;
  }

  // Header
  int n, m, p, l, nsoc, Pnnz, Annz, Gnnz;
  fread(&n, sizeof(int), 1, f);
  fread(&m, sizeof(int), 1, f);
  fread(&p, sizeof(int), 1, f);
  fread(&l, sizeof(int), 1, f);
  fread(&nsoc, sizeof(int), 1, f);
  fread(&Pnnz, sizeof(int), 1, f);
  fread(&Annz, sizeof(int), 1, f);
  fread(&Gnnz, sizeof(int), 1, f);

  // Dense vectors
  double* c = malloc(n * sizeof(double));
  double* b = malloc(p * sizeof(double));
  double* h = malloc(m * sizeof(double));
  int* q = malloc(nsoc * sizeof(int));

  fread(c, sizeof(double), n, f);
  fread(b, sizeof(double), p, f);
  fread(h, sizeof(double), m, f);
  fread(q, sizeof(int), nsoc, f);

  // P
  double* Px = malloc(Pnnz * sizeof(double));
  int* Pi = malloc(Pnnz * sizeof(int));
  int* Pp = malloc((n + 1) * sizeof(int));

  fread(Px, sizeof(double), Pnnz, f);
  fread(Pi, sizeof(int), Pnnz, f);
  fread(Pp, sizeof(int), n + 1, f);

  // A
  double* Ax = malloc(Annz * sizeof(double));
  int* Ai = malloc(Annz * sizeof(int));
  int* Ap = malloc((n + 1) * sizeof(int));

  fread(Ax, sizeof(double), Annz, f);
  fread(Ai, sizeof(int), Annz, f);
  fread(Ap, sizeof(int), n + 1, f);

  // G
  double* Gx = malloc(Gnnz * sizeof(double));
  int* Gi = malloc(Gnnz * sizeof(int));
  int* Gp = malloc((n + 1) * sizeof(int));

  fread(Gx, sizeof(double), Gnnz, f);
  fread(Gi, sizeof(int), Gnnz, f);
  fread(Gp, sizeof(int), n + 1, f);
  fclose(f);

  QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));

  // Check for missing data
  if (Pnnz == 0) {
    P = NULL;
  }
  else {
    qoco_set_csc(P, n, n, Pnnz, Px, Pp, Pi);
  }
  if (Annz == 0) {
    A = NULL;
    b = NULL;
  }
  else {
    qoco_set_csc(A, p, n, Annz, Ax, Ap, Ai);
  }
  if (Gnnz == 0) {
    G = NULL;
    h = NULL;
  }
  else {
    qoco_set_csc(G, m, n, Gnnz, Gx, Gp, Gi);
  }
  if (nsoc == 0) {
    q = NULL;
  }

  QOCOSettings* settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));
  set_default_settings(settings);
  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit =
      qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }

  // Print summary: filename, exit_code, iters, setup time, solve time
  printf("%s %d %d %f %f\n", filename, exit, solver->sol->iters,
         solver->sol->setup_time_sec, solver->sol->solve_time_sec);

  // Free memory
  free(c);
  free(b);
  free(h);
  free(q);
  free(Px);
  free(Pi);
  free(Pp);
  free(Ax);
  free(Ai);
  free(Ap);
  free(Gx);
  free(Gi);
  free(Gp);
  qoco_cleanup(solver);

  return 0;
}
