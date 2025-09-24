#include "qoco.h"
#include <benchmark/benchmark.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Simple RAII deleter for solver
struct SolverDeleter {
  void operator()(QOCOSolver* solver) const
  {
    if (solver)
      qoco_cleanup(solver);
  }
};

using SolverPtr = std::unique_ptr<QOCOSolver, SolverDeleter>;

struct ProblemData {
  int n, m, p, l, nsoc;
  int Pnnz, Annz, Gnnz;
  std::vector<double> c, b, h;
  std::vector<int> q;
  std::vector<double> Px, Ax, Gx;
  std::vector<int> Pi, Ai, Gi;
  std::vector<int> Pp, Ap, Gp;
  QOCOCscMatrix *P, *A, *G;

  ProblemData(const char* filename)
  {
    FILE* f = fopen(filename, "rb");
    if (!f) {
      perror("fopen");
    }

    fread(&n, sizeof(int), 1, f);
    fread(&m, sizeof(int), 1, f);
    fread(&p, sizeof(int), 1, f);
    fread(&l, sizeof(int), 1, f);
    fread(&nsoc, sizeof(int), 1, f);
    fread(&Pnnz, sizeof(int), 1, f);
    fread(&Annz, sizeof(int), 1, f);
    fread(&Gnnz, sizeof(int), 1, f);

    c.resize(n);
    b.resize(p);
    h.resize(m);
    q.resize(nsoc);
    fread(c.data(), sizeof(double), n, f);
    fread(b.data(), sizeof(double), p, f);
    fread(h.data(), sizeof(double), m, f);
    fread(q.data(), sizeof(int), nsoc, f);

    Px.resize(Pnnz);
    Pi.resize(Pnnz);
    Pp.resize(n + 1);
    fread(Px.data(), sizeof(double), Pnnz, f);
    fread(Pi.data(), sizeof(int), Pnnz, f);
    fread(Pp.data(), sizeof(int), n + 1, f);

    Ax.resize(Annz);
    Ai.resize(Annz);
    Ap.resize(n + 1);
    fread(Ax.data(), sizeof(double), Annz, f);
    fread(Ai.data(), sizeof(int), Annz, f);
    fread(Ap.data(), sizeof(int), n + 1, f);

    Gx.resize(Gnnz);
    Gi.resize(Gnnz);
    Gp.resize(n + 1);
    fread(Gx.data(), sizeof(double), Gnnz, f);
    fread(Gi.data(), sizeof(int), Gnnz, f);
    fread(Gp.data(), sizeof(int), n + 1, f);

    fclose(f);

    if (Pnnz > 0) {
      P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
      qoco_set_csc(P, n, n, Pnnz, Px.data(), Pp.data(), Pi.data());
    }
    else {
      P = nullptr;
    }
    if (Annz > 0) {
      A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
      qoco_set_csc(A, p, n, Annz, Ax.data(), Ap.data(), Ai.data());
    }
    else {
      A = nullptr;
      b.clear();
    }
    if (Gnnz > 0) {
      G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));
      qoco_set_csc(G, m, n, Gnnz, Gx.data(), Gp.data(), Gi.data());
    }
    else {
      G = nullptr;
      h.clear();
    }
    if (nsoc == 0)
      q.clear();
  }
};

static std::unique_ptr<ProblemData> g_problem;
static std::unique_ptr<QOCOSettings> g_settings;

// Benchmark: setup only
static void BM_QOCO_Setup(benchmark::State& state)
{
  for (auto _ : state) {
    SolverPtr solver((QOCOSolver*)malloc(sizeof(QOCOSolver)));
    benchmark::DoNotOptimize(solver);
    qoco_setup(
        solver.get(), g_problem->n, g_problem->m, g_problem->p, g_problem->P,
        g_problem->c.data(), g_problem->A,
        g_problem->b.empty() ? nullptr : g_problem->b.data(), g_problem->G,
        g_problem->h.empty() ? nullptr : g_problem->h.data(), g_problem->l,
        g_problem->nsoc, g_problem->q.empty() ? nullptr : g_problem->q.data(),
        g_settings.get());
  }
}
BENCHMARK(BM_QOCO_Setup);

// Benchmark: setup + solve
static void BM_QOCO_Solve(benchmark::State& state)
{
  for (auto _ : state) {
    SolverPtr solver((QOCOSolver*)malloc(sizeof(QOCOSolver)));
    if (qoco_setup(solver.get(), g_problem->n, g_problem->m, g_problem->p,
                   g_problem->P, g_problem->c.data(), g_problem->A,
                   g_problem->b.empty() ? nullptr : g_problem->b.data(),
                   g_problem->G,
                   g_problem->h.empty() ? nullptr : g_problem->h.data(),
                   g_problem->l, g_problem->nsoc,
                   g_problem->q.empty() ? nullptr : g_problem->q.data(),
                   g_settings.get()) == QOCO_NO_ERROR) {
      qoco_solve(solver.get());
    }
  }
}
BENCHMARK(BM_QOCO_Solve);

int main(int argc, char** argv)
{
  g_problem = std::make_unique<ProblemData>("./benchmarks/data/BOYD1.bin");
  g_settings = std::make_unique<QOCOSettings>();
  set_default_settings(g_settings.get());

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
