all : FIFO_stdgpu_deq LIFO_stdgpu_deq

FIFO_stdgpu_deq : FIFO_benchmark.cu
	nvcc FIFO_benchmark.cu -std=c++17 -O3 -arch=native -I../../../stdgpu_test/stdgpu/bin/include/ -L../../../stdgpu_test/stdgpu/bin/lib64 -lstdgpu -o FIFO_stdgpu_deq

LIFO_stdgpu_deq : LIFO_benchmark.cu
	nvcc LIFO_benchmark.cu -std=c++17 -O3 -arch=native -I../../../stdgpu_test/stdgpu/bin/include/ -L../../../stdgpu_test/stdgpu/bin/lib64 -lstdgpu -o LIFO_stdgpu_deq

clean:
	rm -rf FIFO_stdgpu_deq LIFO_stdgpu_deq