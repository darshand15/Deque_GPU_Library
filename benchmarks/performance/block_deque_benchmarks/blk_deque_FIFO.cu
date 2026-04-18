#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include "gpu_deque.cuh"

// --- Error Check ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[BENCH] CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// --- Benchmark Parameters ---
const int THREADS_PER_BLOCK = 512;
const int NUM_BLOCKS = 1024;
const int BLOCK_PRODUCERS = THREADS_PER_BLOCK / 2;
const int ITEMS_PER_PRODUCER = 5;
const int BLOCK_CAPACITY = BLOCK_PRODUCERS * ITEMS_PER_PRODUCER + 1;
const int TOTAL_ITEMS = NUM_BLOCKS * BLOCK_PRODUCERS * ITEMS_PER_PRODUCER;

// Kernel to perform FIFO test on block level shared memory deque
__global__ void fifo_test_kernel(int* d_results) 
{
    // Create the deque in __shared__ memory
    __shared__ BlockDeque<int, BLOCK_CAPACITY> my_deque;

    // Initialize deque
    if (threadIdx.x == 0) 
    {
        my_deque.init();
    }

    __syncthreads();

    // Producers
    if (threadIdx.x < BLOCK_PRODUCERS) 
    {
        for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) 
        {
            int item = (blockIdx.x * BLOCK_PRODUCERS + threadIdx.x) * ITEMS_PER_PRODUCER + i;
            my_deque.pushBack(item);
        }
    }

    // Synchronize block
    __syncthreads();

    // Consumers
    if (threadIdx.x >= BLOCK_PRODUCERS) 
    {
        int item;
        while (my_deque.popFront(&item))
        {
            d_results[item] = 1;
        }
    }
}

int main() 
{
    printf("--- Block-Level Shared Deque - FIFO ---\n");
    printf("Blocks: %d | Threads/Block: %d | Items/Block: %d\n", NUM_BLOCKS, THREADS_PER_BLOCK, BLOCK_PRODUCERS * ITEMS_PER_PRODUCER);

    int *d_results, *h_results;
    CHECK_CUDA(cudaMalloc(&d_results, TOTAL_ITEMS * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_results, 0, TOTAL_ITEMS * sizeof(int)));
    h_results = (int*)malloc(TOTAL_ITEMS * sizeof(int));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    fifo_test_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_results);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time Elapsed: %.4f ms\n", elapsed_time);

    CHECK_CUDA(cudaMemcpy(h_results, d_results, TOTAL_ITEMS * sizeof(int), cudaMemcpyDeviceToHost));
    long long success_count = 0;
    for(int i = 0; i < TOTAL_ITEMS; ++i) 
    {
        if(h_results[i] == 1) success_count++;
    }

    printf("Verification: %s (Got %lld / %d items)\n", (success_count == TOTAL_ITEMS ? "PASS" : "FAIL"), success_count, TOTAL_ITEMS);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_results));
    free(h_results);

    return 0;
}