all : 1_deq 2_deq

1_deq : func_tests.cu
	nvcc func_tests.cu -I../../src/1_lock_based_deque/ -o 1_deq

2_deq : func_tests.cu
	nvcc func_tests.cu -I../../src/2_lock_free_deque/ -o 2_deq

clean:
	rm -rf 1_deq 2_deq