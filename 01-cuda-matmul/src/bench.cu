// Benchmark: naive vs tiled vs cuBLAS SGEMM. Checks results against cuBLAS.
// Usage: ./bench [M N K] [iters]   (default 1024 1024 1024, 20 iters)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <cublas_v2.h>
#include "kernels.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

// Time fn() with CUDA events. One warmup run, then the mean of `iters` runs (ms).
static float time_ms(const std::function<void()>& fn, int iters) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    fn(); CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    CHECK(cudaGetLastError());
    return ms / iters;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024, iters = 20;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc == 5) iters = atoi(argv[4]);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    srand(0);
    for (auto& v : hA) v = rand() / (float)RAND_MAX - 0.5f;
    for (auto& v : hB) v = rand() / (float)RAND_MAX - 0.5f;

    float *A, *B, *C, *Cref;
    CHECK(cudaMalloc(&A, hA.size() * 4)); CHECK(cudaMalloc(&B, hB.size() * 4));
    CHECK(cudaMalloc(&C, (size_t)M * N * 4)); CHECK(cudaMalloc(&Cref, (size_t)M * N * 4));
    CHECK(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

    cublasHandle_t h; cublasCreate(&h);
    const float one = 1.f, zero = 0.f;
    // cuBLAS is column-major. Row-major C = A*B equals column-major C^T = B^T * A^T.
    auto run_cublas = [&] {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, Cref, N);
    };
    dim3 blk(TILE, TILE), grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    auto run_naive = [&] { matmul_naive<<<grd, blk>>>(A, B, C, M, N, K); };
    auto run_tiled = [&] { matmul_tiled<<<grd, blk>>>(A, B, C, M, N, K); };

    double flops = 2.0 * M * N * K;
    float t_cublas = time_ms(run_cublas, iters);  // also fills Cref
    std::vector<float> ref((size_t)M * N), out((size_t)M * N);
    CHECK(cudaMemcpy(ref.data(), Cref, ref.size() * 4, cudaMemcpyDeviceToHost));

    printf("M=%d N=%d K=%d iters=%d TILE=%d\n", M, N, K, iters, TILE);
    printf("%-8s %10s %10s %12s\n", "kernel", "ms", "GFLOP/s", "max_abs_err");
    printf("%-8s %10.4f %10.1f %12s\n", "cublas", t_cublas, flops / t_cublas / 1e6, "-");
    struct { const char* n; std::function<void()> f; } ks[] = {{"naive", run_naive}, {"tiled", run_tiled}};
    for (auto& k : ks) {
        float t = time_ms(k.f, iters);
        CHECK(cudaMemcpy(out.data(), C, out.size() * 4, cudaMemcpyDeviceToHost));
        float err = 0.f;
        for (size_t i = 0; i < out.size(); ++i) err = fmaxf(err, fabsf(out[i] - ref[i]));
        printf("%-8s %10.4f %10.1f %12.3e\n", k.n, t, flops / t / 1e6, err);
    }
    cublasDestroy(h);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Cref);
    return 0;
}
