all : FIFO_1_deq FIFO_2_deq FIFO_3_deq LIFO_1_deq LIFO_2_deq LIFO_3_deq

FIFO_1_deq : glb_deque_FIFO.cu
	nvcc glb_deque_FIFO.cu -std=c++17 -O3 -arch=native -I../../../src/1_lock_based_deque/ -o FIFO_1_deq

FIFO_2_deq : glb_deque_FIFO.cu
	nvcc glb_deque_FIFO.cu -std=c++17 -O3 -arch=native -I../../../src/2_lock_free_deque/ -o FIFO_2_deq

FIFO_3_deq : blk_deque_FIFO.cu
	nvcc blk_deque_FIFO.cu -std=c++17 -O3 -arch=native -I../../../src/3_blk_level_shared_deque/ -o FIFO_3_deq

LIFO_1_deq : glb_deque_LIFO.cu
	nvcc glb_deque_LIFO.cu -std=c++17 -O3 -arch=native -I../../../src/1_lock_based_deque/ -o LIFO_1_deq

LIFO_2_deq : glb_deque_LIFO.cu
	nvcc glb_deque_LIFO.cu -std=c++17 -O3 -arch=native -I../../../src/2_lock_free_deque/ -o LIFO_2_deq

LIFO_3_deq : blk_deque_LIFO.cu
	nvcc blk_deque_LIFO.cu -std=c++17 -O3 -arch=native -I../../../src/3_blk_level_shared_deque/ -o LIFO_3_deq


clean:
	rm -rf FIFO_1_deq FIFO_2_deq FIFO_3_deq LIFO_1_deq LIFO_2_deq LIFO_3_deq