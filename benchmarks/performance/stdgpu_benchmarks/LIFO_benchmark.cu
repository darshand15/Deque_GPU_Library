#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>

#include "stdgpu/deque.cuh"

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[BENCH] CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)


// --- Benchmark Parameters ---
const int CAPACITY = 1'000'000;
const int NUM_ITEMS = 1'000'000;
const int WORK_PER_THREAD = 10;
const int NUM_THREADS = (NUM_ITEMS + WORK_PER_THREAD - 1) / WORK_PER_THREAD;
const int THREADS_PER_BLOCK = 256;
const int NUM_BLOCKS = (NUM_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

// Kernel to push to the front of the stdgpu deque
__global__ void push_front_kernel(stdgpu::deque<int> d_deque, int* d_push_failures, int total_items, int work_per_thread) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = tid * work_per_thread;

    for (int i = 0; i < work_per_thread; ++i) 
    {
        int item_id = start_idx + i;
        if (item_id >= total_items) return;
        
        if (!d_deque.push_front(item_id)) 
        {
            atomicAdd(d_push_failures, 1);
        }
    }
}

// Kernel to pop from the back of the stdgpu deque
__global__ void pop_back_kernel(stdgpu::deque<int> d_deque, int* d_results, int* d_pop_failures, int total_items, int work_per_thread) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = tid * work_per_thread;

    for (int i = 0; i < work_per_thread; ++i) 
    {
        int item_idx = start_idx + i;
        if (item_idx >= total_items) return;
                
        auto ret = d_deque.pop_back();

        if (ret.second) 
        {
            d_results[item_idx] = ret.first;
        } 
        else 
        {
            d_results[item_idx] = -1;
            atomicAdd(d_pop_failures, 1);
        }
    }
}

// Function to verify the results
void verify_results(int* h_results, int h_push_failures, int h_pop_failures) 
{
    printf("--- Verifying stdgpu::deque (LIFO: push_front / pop_back) ---\n");
    
    long long expected_sum = (long long)NUM_ITEMS * (NUM_ITEMS - 1) / 2;
    long long actual_sum = 0;
    int success_pops = 0;
    
    std::vector<bool> seen(NUM_ITEMS, false);
    bool duplicate_found = false;
    
    for (int i = 0; i < NUM_ITEMS; ++i) 
    {
        int val = h_results[i];
        if (val != -1) 
        {
            actual_sum += val;
            success_pops++;
            
            if (val >= 0 && val < NUM_ITEMS) 
            {
                if (seen[val]) 
                {
                    duplicate_found = true;
                }
                seen[val] = true;
            }
        }
    }

    printf("Push Failures: %d (Expected 0)\n", h_push_failures);
    printf("Pop Failures:  %d (Expected 0)\n", h_pop_failures);
    printf("Successful Pops: %d (Expected %d)\n", success_pops, NUM_ITEMS);
    
    bool sum_correct = (actual_sum == expected_sum);
    printf("Sum Check:     %s (Expected: %lld, Got: %lld)\n", 
           sum_correct ? "PASS" : "FAIL", expected_sum, actual_sum);
           
    bool all_items_seen = (success_pops == NUM_ITEMS);
    printf("Uniqueness/Completeness Check: %s\n", 
           (all_items_seen && !duplicate_found) ? "PASS" : "FAIL");
           
    if (h_push_failures == 0 && h_pop_failures == 0 && sum_correct && all_items_seen && !duplicate_found) 
    {
        printf("Result: PASS\n");
    } 
    else 
    {
        printf("Result: FAIL\n");
    }
    printf("------------------------------------\n");
}


int main() 
{
    printf("--- GPU stdgpu::deque Benchmark (LIFO) ---\n");
    printf("Capacity:      %d\n", CAPACITY);
    printf("Items:         %d\n", NUM_ITEMS);
    printf("Total Threads: %d\n", NUM_THREADS);
    printf("Work/Thread:   %d\n", WORK_PER_THREAD);
    printf("Grid Size:     %d blocks, %d threads/block\n", NUM_BLOCKS, THREADS_PER_BLOCK);
    printf("------------------------------------\n");

    int *d_results, *h_results;
    int *d_push_failures, *d_pop_failures;
    int h_push_failures, h_pop_failures;

    CHECK_CUDA(cudaMalloc(&d_results, NUM_ITEMS * sizeof(int)));
    h_results = (int*)malloc(NUM_ITEMS * sizeof(int));
    
    CHECK_CUDA(cudaMalloc(&d_push_failures, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_pop_failures, sizeof(int)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float elapsed_time;

    // --- Run LIFO Test ---
    {
        printf("Running (push_front / pop_back)...\n");
        
        stdgpu::deque<int> d_deque = stdgpu::deque<int>::createDeviceObject(CAPACITY);
        
        CHECK_CUDA(cudaMemset(d_push_failures, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_pop_failures, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_results, 0, NUM_ITEMS * sizeof(int)));

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));

        push_front_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_deque, d_push_failures, NUM_ITEMS, WORK_PER_THREAD);
        pop_back_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_deque, d_results, d_pop_failures, NUM_ITEMS, WORK_PER_THREAD);

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

        printf("Time Elapsed: %.4f ms\n", elapsed_time);

        CHECK_CUDA(cudaMemcpy(&h_push_failures, d_push_failures, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_pop_failures, d_pop_failures, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_results, d_results, NUM_ITEMS * sizeof(int), cudaMemcpyDeviceToHost));
        
        verify_results(h_results, h_push_failures, h_pop_failures);
    }

    printf("Benchmark complete.\n");
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_results));
    CHECK_CUDA(cudaFree(d_push_failures));
    CHECK_CUDA(cudaFree(d_pop_failures));
    free(h_results);

    return 0;
}