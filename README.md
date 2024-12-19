# C++ to CUDA Conversion 

## Overview
This guide demonstrates how to convert a simple C++ program to CUDA, enabling you to leverage the GPU for parallel computation. We'll go through the steps required to modify a C++ program to run on the GPU using CUDA.

## Prerequisites
Before you begin, ensure you have the following installed:
- **CUDA Toolkit**: Install the CUDA Toolkit from [NVIDIA's official website](https://developer.nvidia.com/cuda-toolkit).
- **NVIDIA Driver**: Ensure that you have an NVIDIA GPU and its corresponding drivers installed.
- **G++ or Clang**: A C++ compiler (e.g., GCC for Linux or Clang for macOS).
- **nvcc Compiler**: The CUDA compiler (`nvcc`) should be installed with the CUDA Toolkit.

## Steps to Convert a C++ Program to CUDA

### Step 1: CUDA Kernel (`__global__`)

- **What is a Kernel?**: In CUDA, a kernel is a function that runs on the GPU. The `__global__` keyword indicates that the function will run on the GPU and can be executed by multiple threads in parallel.
- **Thread Index**: Each thread in CUDA has a unique index, and you can use `threadIdx.x` to get this index for each thread. In our example, each thread will add one element from two arrays.

#### Example:
```cpp
__global__ void add_arrays(int *a, int *b, int *c, int N) {
    int index = threadIdx.x;  // Get the thread index
    if (index < N) {  // Ensure the thread index is within bounds
        c[index] = a[index] + b[index];  // Add corresponding elements
    }
}
```

### Step 2: Allocate Memory on the GPU

- **Why Allocate Memory?**: Before using data on the GPU, we need to allocate memory using `cudaMalloc`.
- **How to Allocate?**: `cudaMalloc` allocates memory for variables on the GPU.

#### Example:
```cpp
int *d_a, *d_b, *d_c;
cudaMalloc(&d_a, N * sizeof(int));  // Allocate memory for array 'a' on the GPU
cudaMalloc(&d_b, N * sizeof(int));  // Allocate memory for array 'b' on the GPU
cudaMalloc(&d_c, N * sizeof(int));  // Allocate memory for array 'c' on the GPU
```

### Step 3: Copy Data from CPU to GPU

- **Why Copy Data?**: The GPU operates on its own memory, so we need to copy data from the CPU to the GPU using `cudaMemcpy`.
- **How to Copy?**: Use `cudaMemcpy` to transfer data from CPU memory to GPU memory.

#### Example:
```cpp
cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);  // Copy data from 'a' (CPU) to 'd_a' (GPU)
cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);  // Copy data from 'b' (CPU) to 'd_b' (GPU)
```

### Step 4: Launch the Kernel

- **Kernel Launch**: To run the kernel on the GPU, use the `<<<blocks, threads>>>` syntax. Each thread in the kernel will perform one element of the addition.
- **Launch Configuration**: We use `1 block` and `N threads` (one per array element) for this simple example.

#### Example:
```cpp
add_arrays<<<1, N>>>(d_a, d_b, d_c, N);  // Launch kernel with 1 block and N threads
```

### Step 5: Copy the Result from GPU to CPU

- **Why Copy the Result?**: After the GPU kernel finishes, you need to copy the result from GPU memory back to CPU memory using `cudaMemcpy`.
- **How to Copy Back?**: Use `cudaMemcpy` to transfer the result back to the CPU.

#### Example:
```cpp
cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);  // Copy result from 'd_c' (GPU) to 'c' (CPU)
```

### Step 6: Free Memory

- **Why Free Memory?**: It is important to free GPU memory when you're done with it to avoid memory leaks.
- **How to Free?**: Use `cudaFree` to free allocated memory on the GPU.

#### Example:
```cpp
cudaFree(d_a);  // Free memory for 'a' on the GPU
cudaFree(d_b);  // Free memory for 'b' on the GPU
cudaFree(d_c);  // Free memory for 'c' on the GPU
```

## Complete Example Code

```cpp
#include <iostream>
using namespace std;

// CUDA kernel function to add two arrays
__global__ void add_arrays(int *a, int *b, int *c, int N) {
    int index = threadIdx.x;  // Get the thread index
    if (index < N) {  // Ensure the thread index is within bounds
        c[index] = a[index] + b[index];  // Add corresponding elements
    }
}

int main() {
    int N = 5;  // Size of the arrays
    int a[N] = {1, 2, 3, 4, 5};    // First array
    int b[N] = {10, 20, 30, 40, 50};  // Second array
    int c[N];  // Resultant array to store the sum

    // Step 1: Allocate memory on the GPU (device)
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));  // Allocate memory for 'a' on the GPU
    cudaMalloc(&d_b, N * sizeof(int));  // Allocate memory for 'b' on the GPU
    cudaMalloc(&d_c, N * sizeof(int));  // Allocate memory for 'c' on the GPU

    // Step 2: Copy data from CPU to GPU
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Step 3: Launch the CUDA kernel on the GPU with N threads
    add_arrays<<<1, N>>>(d_a, d_b, d_c, N);

    // Step 4: Copy the result from GPU to CPU
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 5: Print the result
    for (int i = 0; i < N; i++) {
        cout << c[i] << " ";  // Output the result of array 'c'
    }
    cout << endl;

    // Step 6: Free the GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

