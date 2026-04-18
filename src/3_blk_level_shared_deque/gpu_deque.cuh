#ifndef BLOCK_LEVEL_DEQUE_H
#define BLOCK_LEVEL_DEQUE_H

#include <cuda_runtime.h>
#include <stdio.h>

// To get the right mod value between 0 and n-1 for any given i (including negative i)
__device__ __forceinline__ int positive_mod(int i, int n) 
{
    return (i % n + n) % n;
}

/*
 * A concurrent, fixed-size Deque for intra-block communication,
 * which lives entirely in __shared__ memory.
 * It uses atomic 'head' and 'tail' counters to manage concurrent access
 * 'head' counts total items popped from front
 * 'tail' counts total items pushed to back
 * The current size is always (tail - head)
 * The index is (counter % capacity)
 */
template <typename T, int CAPACITY>
class BlockDeque 
{
private:
    
    T buffer[CAPACITY]; // The shared data buffer
    int head;           // Counts total items popped from front
    int tail;           // Counts total items pushed to back

public:

    // Initializes the deque.
    // Only one thread (e.g., thread 0) in the block should call this.
    __device__ void init() 
    {
        head = 0;
        tail = 0;
    }

    // Adds an element to the rear of the deque
    __device__ bool pushBack(const T& element) 
    {
        // Atomically claim a slot at the tail
        // Using _block scope for atomics on __shared__ memory
        int my_tail = atomicAdd_block(&tail, 1);

        // Check if the deque is full
        if (my_tail - (*(volatile int*)&head) >= CAPACITY) 
        {
            // Deque is full
            // Roll back the atomic add and fail
            atomicSub_block(&tail, 1);
            return false;
        }

        buffer[positive_mod(my_tail, CAPACITY)] = element;
        return true;
    }

    // Removes and returns (through the passed argument) an element from the front of the deque
    __device__ bool popFront(T* out_element) 
    {
        // Atomically claim an item from the head
        int my_head = atomicAdd_block(&head, 1);

        // Check if the deque was empty
        if (my_head >= (*(volatile int*)&tail)) 
        {
            // Empty, Roll back and fail
            atomicSub_block(&head, 1);
            return false;
        }

        *out_element = buffer[positive_mod(my_head, CAPACITY)];
        return true;
    }

    // Adds an element to the front of the deque
    __device__ bool pushFront(const T& element) 
    {
        // Atomically claim a slot by decrementing the head
        int my_head = atomicSub_block(&head, 1);

        // Check if the deque is full
        if ((*(volatile int*)&tail) - my_head >= CAPACITY) 
        {
            // Full, Roll back and fail
            atomicAdd_block(&head, 1);
            return false;
        }

        buffer[positive_mod(my_head - 1, CAPACITY)] = element;
        return true;
    }

    // Removes and returns (through the passed argument) an element from the rear of the deque
    __device__ bool popBack(T* out_element) 
    {
        // Atomically claim an item from the tail
        int my_tail = atomicSub_block(&tail, 1);

        // Check if deque was empty
        if ((*(volatile int*)&head) >= my_tail) 
        {
            // Empty, Roll back
            atomicAdd_block(&tail, 1);
            return false;
        }

        // element to be retrieved is 1 behind my_tail as my_tail is the current count of items pushed to back
        *out_element = buffer[positive_mod(my_tail - 1, CAPACITY)];
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
        return CAPACITY;
    }
};

#endif // BLOCK_LEVEL_DEQUE_H