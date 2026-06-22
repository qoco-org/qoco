#include "pdg_data.h"
#include "qoco.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr QOCOInt kDefaultNumInstances = 1000;
constexpr QOCOInt kDefaultBatchWidth = 1000;
constexpr QOCOInt kPdgInitialConditionStart = 6 * (300 - 1);
constexpr QOCOFloat kRelativePerturbation = 1e-3;

typedef struct {
  double serial_factor_sec;
  double serial_solve_sec;
  long long serial_factor_calls;
  long long serial_solve_calls;
  double batch_factor_sec;
  double batch_solve_sec;
  long long batch_factor_calls;
  long long batch_solve_calls;
} QOCOCudaLinsysTiming;

extern "C" void qoco_cuda_linsys_timing_reset(void);
extern "C" void qoco_cuda_linsys_timing_set_enabled(int enabled);
extern "C" void qoco_cuda_linsys_timing_get(QOCOCudaLinsysTiming* timing);

double now_seconds()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void set_pdg_matrices(QOCOCscMatrix* P, QOCOCscMatrix* A, QOCOCscMatrix* G)
{
  qoco_set_csc(P, pdg_n, pdg_n, pdg_P_nnz, pdg_P_x, pdg_P_p, pdg_P_i);
  qoco_set_csc(A, pdg_p, pdg_n, pdg_A_nnz, pdg_A_x, pdg_A_p, pdg_A_i);
  qoco_set_csc(G, pdg_m, pdg_n, pdg_G_nnz, pdg_G_x, pdg_G_p, pdg_G_i);
}

void make_b_values(std::vector<QOCOFloat>& b_values, QOCOInt num_instances)
{
  b_values.resize((size_t)num_instances * (size_t)pdg_p);

  for (QOCOInt item = 0; item < num_instances; ++item) {
    QOCOFloat* b = b_values.data() + (size_t)item * (size_t)pdg_p;
    for (QOCOInt i = 0; i < pdg_p; ++i) {
      b[i] = pdg_b[i];
    }

    for (QOCOInt i = 0; i < 6; ++i) {
      const QOCOFloat base = pdg_b[kPdgInitialConditionStart + i];
      const QOCOFloat perturb =
          kRelativePerturbation * base *
          std::sin(0.017 * (double)(item + 1) * (double)(i + 1));
      b[kPdgInitialConditionStart + i] = base + perturb;
    }
  }
}

const QOCOFloat* b_item(const std::vector<QOCOFloat>& b_values, QOCOInt item)
{
  return b_values.data() + (size_t)item * (size_t)pdg_p;
}

bool status_ok(QOCOInt status)
{
  return status == QOCO_SOLVED || status == QOCO_SOLVED_INACCURATE;
}

void print_cudss_timing(const char* label, double factor_sec,
                        double solve_sec, long long factor_calls,
                        long long solve_calls)
{
  std::printf(
      "%s cuDSS factor seconds: %.6f (%lld calls)\n"
      "%s cuDSS solve seconds: %.6f (%lld calls)\n"
      "%s cuDSS factor+solve seconds: %.6f\n",
      label, factor_sec, factor_calls, label, solve_sec, solve_calls, label,
      factor_sec + solve_sec);
}

void warmup(QOCOCscMatrix* P, QOCOCscMatrix* A, QOCOCscMatrix* G,
            QOCOSettings* settings, const std::vector<QOCOFloat>& b_values)
{
  QOCOSolver* solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
  if (!solver) {
    std::fprintf(stderr, "warmup solver allocation failed\n");
    std::exit(1);
  }

  QOCOInt exit = qoco_setup(solver, pdg_n, pdg_m, pdg_p, P, pdg_c, A, pdg_b, G,
                            pdg_h, pdg_l, pdg_nsoc, pdg_q, settings);
  if (exit != QOCO_NO_ERROR) {
    std::fprintf(stderr, "warmup setup failed: %d\n", exit);
    std::exit(1);
  }

  qoco_update_vector_data(solver, nullptr, (QOCOFloat*)b_item(b_values, 0),
                          nullptr);
  exit = qoco_solve(solver);
  if (!status_ok(exit)) {
    std::fprintf(stderr, "warmup solve failed: %d\n", exit);
    std::exit(1);
  }
  qoco_cleanup(solver);
}

} // namespace

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  QOCOInt num_instances = kDefaultNumInstances;
  QOCOInt batch_width = kDefaultBatchWidth;
  bool run_serial = true;
  bool run_batch = true;

  if (argc > 1) {
    num_instances = (QOCOInt)std::atoi(argv[1]);
  }
  if (argc > 2) {
    batch_width = (QOCOInt)std::atoi(argv[2]);
  }
  if (argc > 3) {
    run_serial = std::atoi(argv[3]) != 0;
  }
  if (argc > 4) {
    run_batch = std::atoi(argv[4]) != 0;
  }
  if (num_instances <= 0 || batch_width <= 0) {
    std::fprintf(stderr,
                 "usage: %s [num_instances=1000] [batch_width=1000] "
                 "[run_serial=1] [run_batch=1]\n",
                 argv[0]);
    return 1;
  }

  QOCOCscMatrix P;
  QOCOCscMatrix A;
  QOCOCscMatrix G;
  set_pdg_matrices(&P, &A, &G);

  QOCOSettings settings;
  set_default_settings(&settings);
  settings.verbose = 0;

  std::vector<QOCOFloat> b_values;
  make_b_values(b_values, num_instances);

  std::printf("PDG batch comparison\n");
  std::printf("instances: %d\n", (int)num_instances);
  std::printf("batch width: %d\n", (int)batch_width);
  std::printf("initial condition b offset: %d\n",
              (int)kPdgInitialConditionStart);
  std::printf("relative perturbation: %.3e\n",
              (double)kRelativePerturbation);
  std::printf("base initial condition:");
  for (QOCOInt i = 0; i < 6; ++i) {
    std::printf(" %.9g", (double)pdg_b[kPdgInitialConditionStart + i]);
  }
  std::printf("\n");

  warmup(&P, &A, &G, &settings, b_values);

  double serial_setup_sec = 0.0;
  double serial_update_sec = 0.0;
  double serial_solve_sec = 0.0;
  QOCOInt serial_solved = 0;
  QOCOInt serial_failed = 0;
  double serial_obj_accum = 0.0;
  QOCOCudaLinsysTiming serial_cudss = {};

  if (run_serial) {
    QOCOSolver* serial_solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));
    if (!serial_solver) {
      std::fprintf(stderr, "serial solver allocation failed\n");
      return 1;
    }

    double t0 = now_seconds();
    QOCOInt exit = qoco_setup(serial_solver, pdg_n, pdg_m, pdg_p, &P, pdg_c,
                              &A, pdg_b, &G, pdg_h, pdg_l, pdg_nsoc, pdg_q,
                              &settings);
    serial_setup_sec = now_seconds() - t0;
    if (exit != QOCO_NO_ERROR) {
      std::fprintf(stderr, "serial setup failed: %d\n", exit);
      return 1;
    }

    qoco_cuda_linsys_timing_reset();
    for (QOCOInt item = 0; item < num_instances; ++item) {
      t0 = now_seconds();
      qoco_update_vector_data(serial_solver, nullptr,
                              (QOCOFloat*)b_item(b_values, item), nullptr);
      serial_update_sec += now_seconds() - t0;

      t0 = now_seconds();
      exit = qoco_solve(serial_solver);
      serial_solve_sec += now_seconds() - t0;

      if (status_ok(exit)) {
        ++serial_solved;
      }
      else {
        ++serial_failed;
      }
      serial_obj_accum += serial_solver->sol->obj;
    }
    qoco_cuda_linsys_timing_get(&serial_cudss);
    qoco_cleanup(serial_solver);
  }

  const double serial_total_sec = serial_update_sec + serial_solve_sec;

  if (run_serial) {
    std::printf("\nserial setup seconds: %.6f\n", serial_setup_sec);
    std::printf("serial update seconds: %.6f\n", serial_update_sec);
    std::printf("serial solve seconds: %.6f\n", serial_solve_sec);
    std::printf("serial update+solve seconds: %.6f\n", serial_total_sec);
    std::printf("serial solved: %d failed: %d obj checksum: %.12e\n",
                (int)serial_solved, (int)serial_failed, serial_obj_accum);
    print_cudss_timing("serial", serial_cudss.serial_factor_sec,
                       serial_cudss.serial_solve_sec,
                       serial_cudss.serial_factor_calls,
                       serial_cudss.serial_solve_calls);
  }
  else {
    std::printf("\nserial skipped\n");
  }

  if (!run_batch) {
    return serial_failed;
  }

  double batch_setup_sec = 0.0;
  double batch_update_sec = 0.0;
  double batch_solve_sec = 0.0;
  QOCOInt batch_solved = 0;
  QOCOInt batch_failed = 0;
  double batch_obj_accum = 0.0;
  QOCOCudaLinsysTiming batch_cudss = {};

  qoco_cuda_linsys_timing_reset();
  for (QOCOInt offset = 0; offset < num_instances; offset += batch_width) {
    const QOCOInt this_batch =
        (offset + batch_width <= num_instances) ? batch_width
                                                : (num_instances - offset);
    std::printf("batch chunk offset %d size %d\n", (int)offset,
                (int)this_batch);

    QOCOBatchSolver batch;
    double t0 = now_seconds();
    QOCOInt exit = qoco_batch_setup(&batch, this_batch, pdg_n, pdg_m, pdg_p,
                                    &P, pdg_c, &A, pdg_b, &G, pdg_h, pdg_l,
                                    pdg_nsoc, pdg_q, &settings);
    batch_setup_sec += now_seconds() - t0;
    if (exit != QOCO_NO_ERROR) {
      std::fprintf(stderr, "batch setup failed at offset %d: %d\n",
                   (int)offset, exit);
      return 1;
    }

    t0 = now_seconds();
    for (QOCOInt item = 0; item < this_batch; ++item) {
      exit = qoco_batch_update_vector_data(
          &batch, item, nullptr,
          (QOCOFloat*)b_item(b_values, offset + item), nullptr);
      if (exit != QOCO_NO_ERROR) {
        std::fprintf(stderr, "batch update failed for item %d: %d\n",
                     (int)(offset + item), exit);
        qoco_batch_cleanup(&batch);
        return 1;
      }
    }
    batch_update_sec += now_seconds() - t0;

    t0 = now_seconds();
    exit = qoco_batch_solve(&batch);
    batch_solve_sec += now_seconds() - t0;
    if (exit != QOCO_NO_ERROR) {
      std::fprintf(stderr, "batch solve dispatch failed at offset %d: %d\n",
                   (int)offset, exit);
      qoco_batch_cleanup(&batch);
      return 1;
    }

    for (QOCOInt item = 0; item < this_batch; ++item) {
      if (status_ok(batch.statuses[item])) {
        ++batch_solved;
      }
      else {
        ++batch_failed;
      }
      QOCOSolution* sol = qoco_batch_get_solution(&batch, item);
      if (sol) {
        batch_obj_accum += sol->obj;
      }
    }
    qoco_batch_cleanup(&batch);
  }
  qoco_cuda_linsys_timing_get(&batch_cudss);
  qoco_cuda_linsys_timing_set_enabled(0);

  const double batch_total_sec = batch_update_sec + batch_solve_sec;

  std::printf("\nbatch setup seconds: %.6f\n", batch_setup_sec);
  std::printf("batch update seconds: %.6f\n", batch_update_sec);
  std::printf("batch solve seconds: %.6f\n", batch_solve_sec);
  std::printf("batch update+solve seconds: %.6f\n", batch_total_sec);
  std::printf("batch solved: %d failed: %d obj checksum: %.12e\n",
              (int)batch_solved, (int)batch_failed, batch_obj_accum);
  print_cudss_timing("batch", batch_cudss.batch_factor_sec,
                     batch_cudss.batch_solve_sec,
                     batch_cudss.batch_factor_calls,
                     batch_cudss.batch_solve_calls);

  if (run_serial) {
    std::printf("\nsolve speedup serial/batch: %.6f\n",
                serial_solve_sec / batch_solve_sec);
    std::printf("update+solve speedup serial/batch: %.6f\n",
                serial_total_sec / batch_total_sec);
    std::printf("including setup speedup serial/batch: %.6f\n",
                (serial_setup_sec + serial_total_sec) /
                    (batch_setup_sec + batch_total_sec));
    std::printf("cuDSS solve speedup serial/batch: %.6f\n",
                serial_cudss.serial_solve_sec / batch_cudss.batch_solve_sec);
    std::printf("cuDSS factor+solve speedup serial/batch: %.6f\n",
                (serial_cudss.serial_factor_sec +
                 serial_cudss.serial_solve_sec) /
                    (batch_cudss.batch_factor_sec +
                     batch_cudss.batch_solve_sec));
  }

  return batch_failed || serial_failed;
}
