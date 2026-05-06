#include "qoco.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static QOCOFloat* read_double_vector(FILE* f, size_t n)
{
  double* tmp = (double*)malloc(n * sizeof(double));
  QOCOFloat* out = (QOCOFloat*)malloc(n * sizeof(QOCOFloat));

  if (!tmp || !out) {
    free(tmp);
    free(out);
    return NULL;
  }

  if (fread(tmp, sizeof(double), n, f) != n) {
    free(tmp);
    free(out);
    return NULL;
  }

  for (size_t i = 0; i < n; ++i) {
    out[i] = (QOCOFloat)tmp[i];
  }

  free(tmp);
  return out;
}

static int* read_int_vector(FILE* f, size_t n)
{
  int* out = (int*)malloc(n * sizeof(int));

  if (!out) {
    return NULL;
  }

  if (fread(out, sizeof(int), n, f) != n) {
    free(out);
    return NULL;
  }

  return out;
}

static void apply_setting(QOCOSettings* settings, const char* arg)
{
  char key[64];
  char val[64];
  if (sscanf(arg, "%63[^=]=%63s", key, val) != 2) {
    fprintf(stderr, "Warning: could not parse setting '%s'\n", arg);
    return;
  }

  if (strcmp(key, "max_iters") == 0) {
    settings->max_iters = atoi(val);
  }
  else if (strcmp(key, "ruiz_iters") == 0) {
    settings->ruiz_iters = atoi(val);
  }
  else if (strcmp(key, "ruiz_scaling_min") == 0) {
    settings->ruiz_scaling_min = atof(val);
  }
  else if (strcmp(key, "ruiz_scaling_max") == 0) {
    settings->ruiz_scaling_max = atof(val);
  }
  else if (strcmp(key, "max_ir_iters") == 0) {
    settings->max_ir_iters = atoi(val);
  }
  else if (strcmp(key, "ir_tol") == 0) {
    settings->ir_tol = atof(val);
  }
  else if (strcmp(key, "kkt_static_reg_P") == 0) {
    settings->kkt_static_reg_P = atof(val);
  }
  else if (strcmp(key, "kkt_static_reg_A") == 0) {
    settings->kkt_static_reg_A = atof(val);
  }
  else if (strcmp(key, "kkt_static_reg_G") == 0) {
    settings->kkt_static_reg_G = atof(val);
  }
  else if (strcmp(key, "kkt_dynamic_reg") == 0) {
    settings->kkt_dynamic_reg = atof(val);
  }
  else if (strcmp(key, "kkt_static_reg_proportional") == 0) {
    settings->kkt_static_reg_proportional = atof(val);
  }
  else if (strcmp(key, "abstol") == 0) {
    settings->abstol = atof(val);
  }
  else if (strcmp(key, "reltol") == 0) {
    settings->reltol = atof(val);
  }
  else if (strcmp(key, "abstol_inacc") == 0) {
    settings->abstol_inacc = atof(val);
  }
  else if (strcmp(key, "reltol_inacc") == 0) {
    settings->reltol_inacc = atof(val);
  }
  else if (strcmp(key, "verbose") == 0) {
    settings->verbose = atoi(val);
  }
  else {
    fprintf(stderr, "Warning: unknown setting '%s'\n", key);
  }
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: %s problem.bin [setting=value ...]\n", argv[0]);
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
  QOCOFloat* c = read_double_vector(f, (size_t)n);
  QOCOFloat* b = read_double_vector(f, (size_t)p);
  QOCOFloat* h = read_double_vector(f, (size_t)m);
  int* q = read_int_vector(f, (size_t)nsoc);

  // P
  QOCOFloat* Px = read_double_vector(f, (size_t)Pnnz);
  int* Pi = read_int_vector(f, (size_t)Pnnz);
  int* Pp = read_int_vector(f, (size_t)(n + 1));

  // A
  QOCOFloat* Ax = read_double_vector(f, (size_t)Annz);
  int* Ai = read_int_vector(f, (size_t)Annz);
  int* Ap = read_int_vector(f, (size_t)(n + 1));

  // G
  QOCOFloat* Gx = read_double_vector(f, (size_t)Gnnz);
  int* Gi = read_int_vector(f, (size_t)Gnnz);
  int* Gp = read_int_vector(f, (size_t)(n + 1));
  fclose(f);

  if (!c || !b || !h || !Px || !Ax || !Gx || !q || !Pi || !Pp || !Ai || !Ap ||
      !Gi || !Gp) {
    fprintf(stderr, "out of memory or truncated benchmark file\n");
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
    return 1;
  }

  QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
  QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));

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

  // Apply overrides from argv[2..argc-1]
  for (int i = 2; i < argc; i++) {
    apply_setting(settings, argv[i]);
  }

  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  QOCOInt exit =
      qoco_setup(solver, n, m, p, P, c, A, b, G, h, l, nsoc, q, settings);
  if (exit == QOCO_NO_ERROR) {
    exit = qoco_solve(solver);
  }
  else {
    printf("%s %d 0 0 0 0\n", filename, exit);
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
    free(P);
    free(A);
    free(G);
    free(settings);
    free(solver);
    return 0;
  }

  // Print summary: filename, exit_code, iters, ir_iters, setup time, solve time
  printf("%s %d %d %d %" QOCOFloat_PRINT_FORMAT " %" QOCOFloat_PRINT_FORMAT "\n",
         filename, exit, solver->sol->iters, solver->sol->ir_iters,
         QOCOFloat_PRINT_ARG(solver->sol->setup_time_sec),
         QOCOFloat_PRINT_ARG(solver->sol->solve_time_sec));

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
