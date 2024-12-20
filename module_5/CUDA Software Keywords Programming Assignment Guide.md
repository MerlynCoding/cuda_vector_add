# **CUDA Software Keywords Programming Assignment Guide**

This guide outlines the structure, steps, and tips to successfully complete the **CUDA Software Keywords Programming Assignment**.

---

### **Goals of the Assignment**
1. **Understand and apply CUDA keywords** (`__global__`, `__device__`, and `__host__`) to a CUDA program.
2. Correctly configure thread and block dimensions for kernel execution.
3. Debug and compile CUDA code using the `nvcc` compiler via a `Makefile`.
4. Generate accurate output for vector operations (e.g., vector addition or multiplication).
5. Submit the results and achieve 100% correctness.

---

### **Steps to Complete the Assignment**
#### **Step 1: Open and Explore the Project Files**
1. **Open the Terminal**: Navigate to the project folder in the VS Code terminal.
2. **Review the Project Structure**:
   - **`Makefile`**: Handles cleaning, building, and executing the program. Contains placeholders (e.g., compiler name) that need fixing.
   - **`simple.cu`**: Contains the CUDA code for vector operations. This is where you will:
     - Apply CUDA keywords.
     - Modify kernel execution configuration (threads, blocks, and grids).
   - **`output.txt`**: The file where program output will be redirected.

#### **Step 2: Modify the CUDA Code in `simple.cu`**
1. **Add CUDA Keywords**:
   - Apply `__global__` to the kernel function (executed on the GPU).
   - Use `__host__` or `__device__` for helper functions as needed.
   - Example:
     ```cpp
     __global__ void vectorAddKernel(float *A, float *B, float *C, int N) {
         int idx = blockIdx.x * blockDim.x + threadIdx.x;
         if (idx < N) {
             C[idx] = A[idx] + B[idx];
         }
     }
     ```
2. **Configure Kernel Launch Parameters**:
   - Replace placeholders (`X`, `Y`, `Z`) for **blocks** and **threads** with actual variables.
   - Define:
     - Number of threads per block (e.g., `threadsPerBlock = 256`).
     - Number of blocks (e.g., `blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock`).
   - Example:
     ```cpp
     int threadsPerBlock = 256;
     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
     vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
     ```
3. **Pass Appropriate Pointers**:
   - Ensure correct host and device memory pointers (`float*`) are passed to the kernel.

#### **Step 3: Modify and Fix the `Makefile`**
1. Replace the placeholder compiler with `nvcc`:
   ```makefile
   NVCC = nvcc
   ```
2. Ensure the `Makefile` builds the `simple.cu` file into an executable (`simple.exe`).
   Example:
   ```makefile
   build:
       $(NVCC) -o simple.exe simple.cu
   ```

#### **Step 4: Compile and Run the Program**
1. **Clean**: Remove old executable files:
   ```bash
   make clean
   ```
2. **Build**: Compile the program:
   ```bash
   make build
   ```
3. **Run**: Execute the program and redirect output to `output.txt`:
   ```bash
   ./simple.exe > output.txt
   ```

#### **Step 5: Submit the Assignment**
1. Click the **Submit** button.
2. Verify the grade under **Submissions**.

---

### **Tips for Success**
1. **Understand CUDA Keywords**:
   - `__global__`: For kernel functions callable from the host and executed on the device.
   - `__device__`: For functions callable and executed only on the device.
   - `__host__`: For functions callable and executed only on the host (CPU).

2. **Check Kernel Configuration**:
   - Always ensure the number of threads is a multiple of 32 for GPU efficiency.
   - Use calculations like `(N + threadsPerBlock - 1) / threadsPerBlock` to avoid out-of-bound errors.

3. **Debugging**:
   - Use `cudaMemcpy` to verify memory transfer between host and device.
   - Add `cudaDeviceSynchronize()` after kernel launches to catch runtime errors.

4. **Testing**:
   - Verify the output in `output.txt`. Ensure all operations (e.g., vector addition) are computed correctly.

5. **Refer to Resources**:
   - Use the CUDA toolkit documentation for clarification on APIs and functions.
   - The `nvcc -h` command can help you understand compilation options.

---

### **Common Errors and Solutions**
1. **"No kernel image is available for execution on the device"**:
   - Ensure the correct GPU architecture is targeted in the `Makefile`.
   - Example:
     ```makefile
     NVCC_FLAGS = -arch=sm_60
     ```

2. **"Segmentation fault"**:
   - Check for out-of-bound memory access in kernel code.

3. **"Undefined reference to kernel function"**:
   - Ensure kernel function is declared with `__global__`.

By following these steps and tips, you'll successfully complete the CUDA Software Keywords assignment and understand the foundational concepts of CUDA programming! 🚀
