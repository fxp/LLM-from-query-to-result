// Tiled matrix multiplication. C[M,N] = A[M,K] @ B[K,N].
//
// Trick: one CUDA block of TILE×TILE threads cooperatively computes a
// TILE×TILE tile of C. For each "chunk" of K (of size TILE), all threads
// first collectively load an A tile and a B tile into SHARED memory, then
// each thread does its TILE multiply-accumulates out of shared memory —
// which is ~15× faster than HBM.
//
// Net effect: each element of A is read from HBM M/TILE times instead of N
// times. For TILE=32 that's a 32× reduction in HBM traffic.
//
// Build: nvcc -O3 -arch=sm_80 matmul_tiled.cu -o matmul_tiled

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int TILE = 32;

__global__ void matmul_tiled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        // Each thread loads one element of the A tile and one of the B tile.
        // Together the TILE*TILE threads cover the whole tile.
        As[threadIdx.y][threadIdx.x] =
            (row < M && t + threadIdx.x < K) ? A[row * K + t + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (t + threadIdx.y < K && col < N) ? B[(t + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        // Now multiply the tiles: this is the hot loop, and every load is
        // from shared memory (not HBM).
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
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

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    const int REPS = 5;
    for (int i = 0; i < REPS; ++i) matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= REPS;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms * 1e-3) / 1e12;
    printf("tiled matmul %dx%dx%d (TILE=%d): %.2f ms, %.2f TFLOPS\n",
           M, N, K, TILE, ms, tflops);

    free(hA); free(hB);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
