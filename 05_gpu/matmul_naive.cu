// Naive matrix multiplication: C[M,N] = A[M,K] @ B[K,N].
//
// One thread produces one element of C. That thread reads K elements of A
// (a row) and K elements of B (a column) from HBM. With 1024 threads
// computing C[0, 0..1023], every single one of them reads the same row of A,
// so A's row gets pulled from HBM 1024 times. This is what `matmul_tiled.cu`
// fixes.
//
// Build: nvcc -O3 -arch=sm_80 matmul_naive.cu -o matmul_naive

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void matmul_naive(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

int main() {
    const int M = 2048, N = 2048, K = 2048;
    size_t sA = (size_t)M * K * sizeof(float);
    size_t sB = (size_t)K * N * sizeof(float);
    size_t sC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(sA), *hB = (float*)malloc(sB);
    for (int i = 0; i < M * K; ++i) hA[i] = (float)(rand() % 1000) / 1000.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand() % 1000) / 1000.0f;

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA)); CHECK(cudaMalloc(&dB, sB)); CHECK(cudaMalloc(&dC, sC));
    CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // Warmup + timed runs.
    matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    const int REPS = 5;
    for (int i = 0; i < REPS; ++i) matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= REPS;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms * 1e-3) / 1e12;
    printf("naive matmul %dx%dx%d: %.2f ms, %.2f TFLOPS\n", M, N, K, ms, tflops);

    free(hA); free(hB);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
