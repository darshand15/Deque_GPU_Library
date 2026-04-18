#ifndef DEQUE_LOCK_H
#define DEQUE_LOCK_H

#include <cuda_runtime.h>
#include <stdio.h>

// Forward declaration of the device-side struct
template <typename T>
struct GpuDeque;

// Host-side (CPU) handle for the lock-based GpuDeque
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
        h_deque_template.front = 0;
        h_deque_template.rear = 0;
        h_deque_template.count = 0;
        h_deque_template.mutex = 0;

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
        // Copying the struct back to host to get the buffer pointer
        GpuDeque<T> h_deque_template;
        cudaMemcpy(&h_deque_template, d_deque, sizeof(GpuDeque<T>), cudaMemcpyDeviceToHost);

        if(h_deque_template.buffer) 
        {
            cudaFree(h_deque_template.buffer);
        }

        if (d_deque) 
        {
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
        return h_deque_template.count;
    }

    // Get the maximum capacity
    int get_capacity() const 
    { 
        return _capacity; 
    }

    // Get the device-side pointer to pass to a kernel
    GpuDeque<T>* get_device_ptr() 
    {
        return d_deque;
    }

private:
    GpuDeque<T>* d_deque;  // Pointer to the struct on the device
    int _capacity;
};


// A thread-safe, fixed-size Deque for use within CUDA kernels
// (This is the device-side struct)
// It uses a single atomic spinlock to protect all operations
template <typename T>
struct GpuDeque 
{
    T* buffer;      // Pointer to the data buffer (device memory)
    int front;      // Index of the front element
    int rear;       // Index of the slot after the rear element
    int capacity;   // Total allocated size of the array
    int count;      // Current number of elements in the deque
    int mutex;      // Spinlock (0 = unlocked, 1 = locked)

    // Acquires the spinlock using CAS
    __device__ void lock() 
    {
        // Spin in a loop until we successfully swap 0 (unlocked) to 1 (locked)
        while (atomicCAS(&mutex, 0, 1) != 0) 
        {
            // Spin/Busy waiting ...
        }
    }

    // Releases the spinlock
    __device__ void unlock() 
    {
        // Atomically set the mutex back to 0
        atomicExch(&mutex, 0);
    }

    // Adds an element to the front of the deque
    __device__ bool pushFront(const T& element) 
    {
        lock();
        if (count == capacity) 
        {
            unlock();
            return false; // Deque is full
        }
        
        front = (front - 1 + capacity) % capacity;
        buffer[front] = element;
        count++;
        
        unlock();
        return true;
    }

    // Adds an element to the rear of the deque
    __device__ bool pushBack(const T& element) 
    {
        lock();
        if (count == capacity) 
        {
            unlock();
            return false; // Deque is full
        }

        buffer[rear] = element;
        rear = (rear + 1) % capacity;
        count++;

        unlock();
        return true;
    }

    // Removes and returns (through the passed argument) an element from the front of the deque
    __device__ bool popFront(T* out_element) 
    {
        lock();
        if (count == 0) 
        {
            unlock();
            return false; // Deque is empty
        }

        *out_element = buffer[front];
        front = (front + 1) % capacity;
        count--;

        unlock();
        return true;
    }

    // Removes and returns (through the passed argument) an element from the rear of the deque
    __device__ bool popBack(T* out_element) 
    {
        lock();
        if (count == 0) 
        {
            unlock();
            return false; // Deque is empty
        }

        // element to be retrieved is 1 behind the rear as rear points to the slot after the rear element
        rear = (rear - 1 + capacity) % capacity;
        *out_element = buffer[rear];
        count--;

        unlock();
        return true;
    }

    // Checks if the deque is empty
    __device__ bool isEmpty() const
    {
        lock();
        bool empty = (count == 0);
        unlock();
        return empty;
    }

    // Gets the current size
    __device__ int getSize() const
    {
        lock();
        int s = count;
        unlock();
        return s;
    }

    // Gets the capacity of the deque
    __device__ int getCapacity() const 
    {
        return capacity;
    }
};


#endif // DEQUE_LOCK_H