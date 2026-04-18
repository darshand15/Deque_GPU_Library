# Graphics Processing Units (GPUs): Architecture & Programming Project

## Deque Data Structure Library for GPUs

This repository contains three different implementations of a GPU-based double-ended queue (deque) along with a comprehensive suite of functional and performance benchmarks.

## Team Members
Darshan Dinesh Kumar

## Abstract

The end of Moore’s Law necessitates the development of innovative solutions to augment the performance of applications rather than attempting to pack more transistors on a chip and/or increasing the CPU frequency. In this direction, Graphics Processing Units (GPUs) are now ubiquitous for high-performance computing and enable a wide variety of applications. However, programming for GPUs is not necessarily straightforward nor intuitive. It requires a fundamental shift in thinking by the Software Developers. Consequently, this new paradigm and programming model can hinder the adoption of GPUs despite their immense capabilities. Further, even when adopted, they may be severely under-utilized due to inefficiencies in the developed Software. Building and supporting a data structure library for GPUs can significantly bridge this gap. This project is precisely such an effort to design and implement a data structure library for GPUs. The project specifically implements a generic (supporting all data-types) deque (double-ended queue) data structure which is a fundamental building block for a variety of applications and is highly versatile given that it can be reduced to represent other structures like stacks and queues. As part of this project, various implementations of the deque were experimented with and analyzed, including a naïve global lock-based implementation, a lock-free atomic counter-based implementation, both for global use across different blocks of a grid, and a shared memory-based implementation for intra-block communication. The generated results for the various benchmarks highlight the pros and cons of the different implementations and denote their suitability for different kinds of applications. The results for the lock-free version are quite promising as compared to the production grade, stdgpu library’s deque, with an approximate 36% improvement in execution time for a benchmark. Further, the block-level shared memory deque demonstrates a significant improvement, of up to 67% as compared to the lock free version, highlighting its suitability for applications requiring a deque only for Intra-Block coordination.

## Directory Structure

### `src/` (Source Code)
All the source code for the deque implementations is placed within the `src` directory, organized into three nested subdirectories:

1. **`1_lock_based_deque/`**: Lock-based global deque implementation.
2. **`2_lock_free_deque/`**: Lock-free global deque implementation.
3. **`3_blk_level_shared_deque/`**: Block-level shared memory deque implementation.

*Note: Each of the above directories contains a `gpu_deque.cuh` header file. To use a specific deque in your client CUDA programs, simply include the header file from the appropriate directory.*

### `benchmarks/` (Benchmark Programs)
All benchmark programs are located within the `benchmarks` directory, organized as follows:

* **`functional/`**: Contains functionality verification tests.
  * `func_tests.cu`: Verifies the functionality of the implemented deques.
  * `Makefile`: Builds the executables for testing both the lock-based and lock-free global deques.

* **`performance/`**: Contains performance benchmarks separated by category.
  * **`block_deque_benchmarks/`**: Intra-Block benchmarks for performance comparison.
    * `blk_deque_FIFO.cu`: FIFO benchmark for the block-level shared deque.
    * `blk_deque_LIFO.cu`: LIFO benchmark for the block-level shared deque.
    * `glb_deque_FIFO.cu`: FIFO benchmark for the global lock-based and lock-free deques.
    * `glb_deque_LIFO.cu`: LIFO benchmark for the global lock-based and lock-free deques.
    * `Makefile`: Builds all executables for the block-level, lock-based, and lock-free deque comparisons.
  * **`global_deque_benchmarks/`**: Global (Inter-Block) benchmarks for performance comparison of this project's deques.
    * `FIFO_benchmark.cu`: FIFO benchmark for the lock-based and lock-free deques.
    * `LIFO_benchmark.cu`: LIFO benchmark for the lock-based and lock-free deques.
    * `Makefile`: Builds the executables for the lock-based and lock-free benchmarking.
  * **`stdgpu_benchmarks/`**: Global (Inter-Block) benchmarks for comparing against the `stdgpu` deque.
    * `FIFO_benchmark.cu`: FIFO benchmark for the `stdgpu` deque.
    * `LIFO_benchmark.cu`: LIFO benchmark for the `stdgpu` deque.
    * `Makefile`: Builds the executables for `stdgpu` benchmarking.

---

## How to Build and Execute Benchmarks

To run the standard benchmarks (functional, block deque, and global deque), follow these steps:

1. **Load the CUDA Module** (ensure you use the appropriate version number for your system):
   ```bash
   module load cuda-13.0
   ```
2. **Navigate to the target benchmark directory**, for example:
   ```bash
   cd benchmarks/performance/global_deque_benchmarks
   ```
3. **Clean previous builds and compile the code**:
   ```bash
   make clean
   make
   ```
4. **Execute the generated binary** (replace `<executable_name>` with the name generated by the Makefile):
   ```bash
   ./<executable_name>
   ```

---

## Testing the `stdgpu` Benchmarks

To run the benchmarks specifically comparing against the `stdgpu` library, you must first clone and build `stdgpu`. Follow these exact steps:

1. **Create and navigate to a setup directory** (at the same level as the `src` and `benchmarks` directories):
   ```bash
   mkdir stdgpu_test
   cd stdgpu_test
   ```
2. **Clone the `stdgpu` repository**:
   ```bash
   git clone [https://github.com/stotko/stdgpu.git](https://github.com/stotko/stdgpu.git)
   cd stdgpu
   ```
3. **Configure the build using CMake**:
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=bin
   ```
4. **Build and install the artifacts** (this installs everything under the `stdgpu/bin` directory):
   ```bash
   cmake --build build --config Release --parallel 8
   cmake --install build
   ```
5. **Build and run the benchmarks**:
   Navigate to the `stdgpu_benchmarks` directory, build the executables, and run them:
   ```bash
   cd ../../benchmarks/performance/stdgpu_benchmarks
   make
   ./<executable_name>
   ```