#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>

#include "gpu_deque.cuh"

// Macro for error checking
#define CHECK_CUDA(call) do \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)


// Struct representing a Complex data type
struct DeviceComplex 
{
    float r;
    float i;

    __host__ __device__ DeviceComplex() : r(0), i(0) {}
    __host__ __device__ DeviceComplex(float _r, float _i) : r(_r), i(_i) {}

    // Operator overload for equality check
    __host__ __device__ bool operator==(const DeviceComplex& other) const 
    {
        // Float comparison with epsilon as a threshold for equality
        float epsilon = 1e-5;
        return (abs(r - other.r) < epsilon) && (abs(i - other.i) < epsilon);
    }

    __host__ __device__ bool operator!=(const DeviceComplex& other) const 
    {
        return !(*this == other);
    }
    
    __device__ void print() const 
    {
        printf("(%.1f, %.1fi)", r, i);
    }
};

// Fixed char buffer to represent a string
template <int MAX_LEN>
struct FixedString 
{
    char data[MAX_LEN];

    __host__ __device__ FixedString() 
    {
        for(int i=0; i<MAX_LEN; ++i)
        {
            data[i] = 0;
        }
    }

    __host__ __device__ FixedString(const char* input) 
    {
        for(int i=0; i<MAX_LEN; ++i) 
        {
            data[i] = input[i];
            if (input[i] == '\0') break;
        }
        data[MAX_LEN-1] = '\0';
    }

    __host__ __device__ bool operator==(const FixedString& other) const 
    {
        for(int i=0; i<MAX_LEN; ++i) 
        {
            if (data[i] != other.data[i]) return false;
            if (data[i] == '\0') return true;
        }
        return true;
    }

    __host__ __device__ bool operator!=(const FixedString& other) const 
    {
        return !(*this == other);
    }

    __device__ void print() const 
    {
        printf("%s", data);
    }
};


// Helper print functions
__device__ void print_val(int v) { printf("%d", v); }
__device__ void print_val(float v) { printf("%.2f", v); }
__device__ void print_val(DeviceComplex v) { v.print(); }
template <int N> __device__ void print_val(FixedString<N> v) { v.print(); }


template <typename T>
__global__ void verify_logic_kernel(GpuDeque<T>* deque, T val_a, T val_b, T val_c, int* errors) 
{
    if (threadIdx.x > 0) return;
    
    T out_val;
    
    // Logic: [PushB(A), PushF(B), PushB(C)] -> [B, A, C]
    deque->pushBack(val_a); 
    deque->pushFront(val_b);
    deque->pushBack(val_c);

    // Pop Front -> B
    if (!deque->popFront(&out_val) || out_val != val_b) 
    {
        printf("Err 1: Expected "); print_val(val_b); printf(" Got "); print_val(out_val); printf("\n");
        atomicAdd(errors, 1);
    }

    // Pop Back -> C
    if (!deque->popBack(&out_val) || out_val != val_c) 
    {
        printf("Err 2: Expected "); print_val(val_c); printf(" Got "); print_val(out_val); printf("\n");
        atomicAdd(errors, 1);
    }

    // Pop Front -> A
    if (!deque->popFront(&out_val) || out_val != val_a) 
    {
        printf("Err 3: Expected "); print_val(val_a); printf(" Got "); print_val(out_val); printf("\n");
        atomicAdd(errors, 1);
    }
}

template <typename T>
void run_test(const char* type_name, T a, T b, T c) 
{
    printf("Testing Type: %-15s => ", type_name);
    int* d_errors;
    int h_errors = 0;
    CHECK_CUDA(cudaMalloc(&d_errors, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_errors, 0, sizeof(int)));
    
    GpuDequeHandle<T> handle(10);
    GpuDeque<T>* d_ptr = handle.get_device_ptr();

    verify_logic_kernel<<<1, 1>>>(d_ptr, a, b, c, d_errors);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_errors == 0) printf("PASS\n");
    else printf("FAIL\n");
    CHECK_CUDA(cudaFree(d_errors));
}


int main() 
{
    printf("==========================================\n");
    printf("Functionality Test: Deque\n");
    printf("==========================================\n");

    // 1. Integer Test
    run_test<int>("int", 10, 20, 30);

    // 2. Float Test
    run_test<float>("float", 1.1f, 2.2f, 3.3f);

    // 3. Complex Number Test
    DeviceComplex c1(1, 1), c2(2, 2), c3(3, 3);
    run_test<DeviceComplex>("DeviceComplex", c1, c2, c3);

    // 4. String Test (Fixed size of 8 chars)
    FixedString<8> s1("Hello"), s2("World"), s3("CUDA");
    run_test<FixedString<8>>("FixedString<8>", s1, s2, s3);

    printf("==========================================\n");
    return 0;
}