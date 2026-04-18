all : FIFO_1_deq FIFO_2_deq LIFO_1_deq LIFO_2_deq

FIFO_1_deq : FIFO_benchmark.cu
	nvcc FIFO_benchmark.cu -std=c++17 -O3 -arch=native -I../../../src/1_lock_based_deque/ -o FIFO_1_deq

FIFO_2_deq : FIFO_benchmark.cu
	nvcc FIFO_benchmark.cu -std=c++17 -O3 -arch=native -I../../../src/2_lock_free_deque/ -o FIFO_2_deq

LIFO_1_deq : LIFO_benchmark.cu
	nvcc LIFO_benchmark.cu -std=c++17 -O3 -arch=native -I../../../src/1_lock_based_deque/ -o LIFO_1_deq

LIFO_2_deq : LIFO_benchmark.cu
	nvcc LIFO_benchmark.cu -std=c++17 -O3 -arch=native -I../../../src/2_lock_free_deque/ -o LIFO_2_deq

clean:
	rm -rf FIFO_1_deq FIFO_2_deq LIFO_1_deq LIFO_2_deq