# CUDA Keywords Cheat Sheet

CUDA programming introduces specific keywords to control where and how code is executed and where data is stored. This guide breaks these keywords into three classes: **Execution Context Keywords**, **Computation Division Keywords**, and **Memory Management Keywords**.

---

### **1. Execution Context Keywords**
These keywords define where the code is executed and from where it can be called.

| **Keyword**  | **Meaning**                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `__host__`   | Indicates the function executes on the **CPU** and can only be called from the CPU. |
| `__global__` | Marks the function as a **kernel function** that executes on the GPU and is called from the CPU. |
| `__device__` | Specifies the function executes on the GPU and is called **only by other GPU functions**. |

#### Example:
```cpp
__global__ void kernelFunction() {
    // Code to run on GPU
}

__device__ int gpuFunction() {
    // Code callable only by other GPU functions
}

__host__ void cpuFunction() {
    // Code runs on CPU
}
```

---

### **2. Computation Division Keywords**
These keywords allow the distribution of computation across CUDA hardware, such as threads and blocks.

| **Concept**           | **Description**                                                                                   |
|-----------------------|---------------------------------------------------------------------------------------------------|
| **Threads**           | Smallest unit of computation.                                                                     |
| **Blocks**            | Groups of threads that share **shared memory** and communicate within the block.                  |
| **Grids**             | Groups of blocks, allowing for large-scale parallel computation.                                   |
| **Thread Dimensions** | Threads can be arranged in **1D, 2D, or 3D** layouts. Useful for multi-dimensional data like images. |
| **Block Dimensions**  | Blocks can also be arranged in **1D, 2D, or 3D** layouts.                                          |

#### Example: Kernel Invocation
```cpp
dim3 threadsPerBlock(32, 32); // 32x32 threads in each block
dim3 blocksPerGrid(16, 16);   // 16x16 blocks in the grid
kernelFunction<<<blocksPerGrid, threadsPerBlock>>>();
```

- **Threads per block**: Should be a multiple of 32 for efficiency.
- **Blocks per grid**: Determines how many blocks are launched.

---

### **3. Memory Management Keywords**
CUDA provides keywords to control data placement in different memory types.

| **Keyword**         | **Memory Type**                                                                                 |
|---------------------|------------------------------------------------------------------------------------------------|
| **Global Memory**   | Large memory accessible by all threads. High latency.                                          |
| `__constant__`      | Constant memory on the GPU. Fast for read-only data.                                           |
| `__shared__`        | Shared memory within a block. Fast and used for inter-thread communication.                    |
| **Registers**       | Fast, private memory for each thread. Limited in size.                                         |
| **Device Memory**   | Memory specifically allocated on the GPU.                                                      |

#### Example: Memory Declaration
```cpp
__constant__ int constantData[256]; // Shared across all threads, read-only

__shared__ int sharedData[128];    // Shared within a block

__device__ int deviceMemory;       // GPU memory, callable only on GPU
```

#### Tips for Memory Usage:
- Use **register memory** for frequently accessed variables.
- Use **shared memory** for communication between threads in the same block.
- Use **constant memory** for read-only data shared across all threads.

---

### **Key Takeaways**
1. **Execution Context**:  
   - Use `__global__` for kernels, `__host__` for CPU-only code, and `__device__` for GPU-only sub-functions.

2. **Computation Division**:  
   - Design kernels with threads and blocks to parallelize computation. Use 1D, 2D, or 3D layouts based on the data structure.

3. **Memory Management**:  
   - Use `__constant__` for static data, `__shared__` for thread collaboration, and registers for high-speed access.

By understanding these keywords, you can write efficient CUDA programs that leverage the full power of GPUs! 🚀
