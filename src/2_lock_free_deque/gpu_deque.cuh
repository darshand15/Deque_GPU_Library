#ifndef DEQUE_LOCKFREE_H
#define DEQUE_LOCKFREE_H

#include <cuda_runtime.h>
#include <stdio.h>

// To get the right mod value between 0 and n-1 for any given i (including negative i)
__device__ __forceinline__ int positive_mod(int i, int n) 
{
    return (i % n + n) % n;
}

// Forward declaration of the device-side struct
template <typename T>
class GpuDeque;

// Host-side (CPU) handle for the lock-free GpuDeque
template <typename T>
class GpuDequeHandle 
{
public:
    
    GpuDequeHandle(int capacity) : d_deque(nullptr), _capacity(capacity) 
    {
        if (capacity <= 0) 
        {
            printf("Error: Deque capacity must be > 0\n");
            return;
        }

        // Allocating device memory for the data buffer
        T* d_buffer = nullptr;
        cudaError_t err = cudaMalloc(&d_buffer, capacity * sizeof(T));
        if (err != cudaSuccess) 
        {
            printf("Error: cudaMalloc for buffer failed: %s\n", cudaGetErrorString(err));
            return;
        }

        // Creating a host-side template of the GpuDeque struct to set its initial state
        GpuDeque<T> h_deque_template;
        h_deque_template.buffer = d_buffer;
        h_deque_template.capacity = capacity;
        h_deque_template.head = 0;
        h_deque_template.tail = 0;

        // Allocating device memory for the GpuDeque struct itself
        err = cudaMalloc(&d_deque, sizeof(GpuDeque<T>));
        if (err != cudaSuccess) 
        {
            printf("Error: cudaMalloc for struct failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_buffer);
            d_buffer = nullptr;
            return;
        }

        // Copying the initialized struct from host to device
        err = cudaMemcpy(d_deque, &h_deque_template, sizeof(GpuDeque<T>), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) 
        {
            printf("Error: cudaMemcpy for struct init failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_buffer);
            cudaFree(d_deque);
            d_deque = nullptr;
            d_buffer = nullptr;
        }
    }

    ~GpuDequeHandle() 
    {
        if (d_deque) 
        {
            // Copying the struct back to host to get the buffer pointer
            GpuDeque<T> h_deque_template;
            cudaMemcpy(&h_deque_template, d_deque, sizeof(GpuDeque<T>), cudaMemcpyDeviceToHost);

            if(h_deque_template.buffer) 
            {
                cudaFree(h_deque_template.buffer);
            }

            cudaFree(d_deque);
        }
    }

    // Disable copy constructor and assignment
    GpuDequeHandle(const GpuDequeHandle&) = delete;
    GpuDequeHandle& operator=(const GpuDequeHandle&) = delete;

    // Get an (approximate) size from the host
    int get_size() 
    {
        if (!d_deque) return 0;

        GpuDeque<T> h_deque_template;
        cudaMemcpy(&h_deque_template, d_deque, sizeof(GpuDeque<T>), cudaMemcpyDeviceToHost);
        
        volatile int tail = h_deque_template.tail;
        volatile int head = h_deque_template.head;
        return tail - head;
    }

    // Get the maximum capacity
    int get_capacity() 
    { 
        return _capacity; 
    }

    // Get the device-side pointer to pass to a kernel
    GpuDeque<T>* get_device_ptr() 
    {
        return d_deque;
    }

private:
    GpuDeque<T>* d_deque; // Pointer to the struct on the device
    int _capacity;
};


/*
 * A concurrent, fixed-size Deque for use within CUDA kernels
 * (This is the device-side struct)
 * It uses atomic 'head' and 'tail' counters to manage concurrent access
 * 'head' counts total items popped from front
 * 'tail' counts total items pushed to back
 * The current size is always (tail - head)
 * The index is (counter % capacity)
 */
template <typename T>
class GpuDeque 
{
public:
    T* buffer; // Pointer to the data buffer (device memory)
    int capacity; // Total allocated size of the array
    int head; // Counts total items popped from front
    int tail; // Counts total items pushed to back

    // Adds an element to the rear of the deque
    __device__ bool pushBack(const T& element) 
    {
        // Atomically claim a slot at the tail
        int my_tail = atomicAdd(&tail, 1);

        // Check if the deque is full
        if (my_tail - (*(volatile int*)&head) >= capacity) 
        {
            // Deque is full
            // Roll back the atomic add and fail
            atomicSub(&tail, 1);
            return false;
        }

        buffer[positive_mod(my_tail, capacity)] = element;
        return true;
    }

    // Removes and returns (through the passed argument) an element from the front of the deque
    __device__ bool popFront(T* out_element) 
    {
        // Atomically claim an item from the head
        int my_head = atomicAdd(&head, 1);

        // Check if the deque was empty
        if (my_head >= (*(volatile int*)&tail)) 
        {
            // Empty, Roll back and fail
            atomicSub(&head, 1);
            return false;
        }

        *out_element = buffer[positive_mod(my_head, capacity)];
        return true;
    }

    // Adds an element to the front of the deque
    __device__ bool pushFront(const T& element) 
    {
        // Atomically claim a slot by decrementing the head
        int my_head = atomicSub(&head, 1);

        // Check if the deque is full 
        if ((*(volatile int*)&tail) - my_head >= capacity) 
        {
            // Full, Roll back and fail
            atomicAdd(&head, 1);
            return false;
        }

        buffer[positive_mod(my_head - 1, capacity)] = element;
        return true;
    }

    // Removes and returns (through the passed argument) an element from the rear of the deque
    __device__ bool popBack(T* out_element) 
    {
        // Atomically claim an item from the tail
        int my_tail = atomicSub(&tail, 1);

        // Check if deque was empty
        if ((*(volatile int*)&head) >= my_tail) 
        {
            // Empty, Roll back
            atomicAdd(&tail, 1);
            return false;
        }

        // element to be retrieved is 1 behind my_tail as my_tail is the current count of items pushed to back
        *out_element = buffer[positive_mod(my_tail - 1, capacity)];
        return true;
    }

    // Checks if the deque is empty
    __device__ bool isEmpty() const
    {
        int current_tail = (*(volatile int*)&tail);
        int current_head = (*(volatile int*)&head);

        // If head has caught up to or passed tail, it's empty
        return current_head >= current_tail;
    }

    // Get the approximate current size
    __device__ int getSize() const 
    {
        int current_tail = (*(volatile int*)&tail);
        int current_head = (*(volatile int*)&head);
        return current_tail - current_head;
    }

    // Gets the capacity of the deque
    __device__ int getCapacity() const 
    {
        return capacity;
    }
};


#endif // DEQUE_LOCKFREE_H