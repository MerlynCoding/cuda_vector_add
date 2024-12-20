# **C++ Parallel Programming Projec**

Welcome to the C++ Parallel Programming Project! This guide will walk you through the project structure and help you understand the key concepts of **threads**, **mutexes**, **futures**, and **atomics** in C++. Whether you're new to parallel programming or want to solidify your basics, this guide is here to make things simple and clear.

---

## **Project Structure**

Inside the project directory, you’ll find the following components:
1. **Header Files (`.h`)**:
   - Contain **function signatures** (declarations) and constants.
   - These files show **what the functions do**, not how they are implemented.

2. **C++ Source Files (`.cpp`)**:
   - Contain the **actual implementation** of the functions declared in the headers.

3. **Makefile**:
   - A file used to **compile**, **run**, and **clean up** the project.
   - Contains commands for building and running individual examples or the entire project.

---

## **Key Concepts**

This project focuses on four major parallel programming concepts in C++:
1. **Threads**: Running tasks simultaneously.
2. **Mutexes**: Ensuring safe access to shared resources.
3. **Futures**: Handling results of tasks that complete in the future.
4. **Atomics**: Managing shared data safely across threads.

---

### **1. Threads**
Threads allow you to run multiple tasks simultaneously. This section includes:
- **`thread_example.h`**: Header file for thread-related functions.
- **`thread_example.cpp`**: Implements three key functions:
  1. **`do_work()`**:
     - A thread runs this function to perform its task.
     - Logs the thread ID as it works.
  2. **`execute_threads()`**:
     - Creates and runs multiple threads in parallel.
     - Uses `join()` to wait for all threads to finish.
  3. **`execute_and_detach_thread()`**:
     - Runs a thread in detached mode.
     - A detached thread operates independently, even if the main program exits.

---

### **2. Mutexes**
A **mutex** (mutual exclusion) is used to prevent multiple threads from accessing the same resource simultaneously. This section includes:
- **`mutex_example.h`**: Header file for mutex-related functions.
- **`mutex_example.cpp`**: Implements four functions:
  1. **`do_work_with_mutex_lock()`**:
     - A thread locks the mutex, performs its work, and then unlocks it.
  2. **`execute_threads_with_mutex_lock()`**:
     - Runs multiple threads, each using the `do_work_with_mutex_lock()` function.
  3. **`do_work_with_mutex_try_lock()`**:
     - A thread tries to lock the mutex. If it can’t, it skips the critical section.
  4. **`execute_and_detach_threads_with_mutex_try_lock()`**:
     - Runs detached threads, each using the `do_work_with_mutex_try_lock()` function.

---

### **3. Futures**
**Futures** allow threads to execute tasks asynchronously and provide their results later. This section includes:
- **`future_example.h`**: Header file for future-related functions.
- **`future_example.cpp`**: Implements four functions:
  1. **`do_work_with_futures()`**:
     - A thread performs its task and returns the result using a `std::future`.
  2. **`execute_threads_with_futures()`**:
     - Runs multiple threads, each using the `do_work_with_futures()` function.
  3. **`do_work_with_async()`**:
     - Runs a task asynchronously using `std::async`.
  4. **`execute_with_async()`**:
     - Runs multiple tasks asynchronously, each using the `do_work_with_async()` function.

---

### **4. Atomics**
**Atomic variables** ensure safe access to shared data without using locks. This section includes:
- **`atomic_example.h`**: Header file for atomic-related functions.
- **`atomic_example.cpp`**: Implements four functions:
  1. **`do_work_with_atomic_bool()`**:
     - A thread works with an `std::atomic<bool>` to manage shared data safely.
  2. **`execute_threads_with_atomic_bool()`**:
     - Runs multiple threads using `do_work_with_atomic_bool()`.
  3. **`do_work_with_atomic_thread_fence()`**:
     - Ensures threads coordinate through an `atomic_thread_fence`.
  4. **`execute_with_atomic_thread_fence()`**:
     - Runs multiple threads, each using the `do_work_with_atomic_thread_fence()` function.

---

## **Makefile**
The **Makefile** automates the building, running, and cleaning of the project. It includes:
- **Build Commands**:
  - Compile individual examples (e.g., `threads`, `mutexes`, etc.).
  - Compile all examples at once using the `all` target.
- **Run Commands**:
  - Run individual examples or all examples at once.
- **Clean Commands**:
  - Remove temporary files (e.g., object files, executables).

---

## **How to Use the Project**

### **1. Build the Project**
To compile the project, use:
```bash
make all  # Build all examples
make threads  # Build only the thread example
```

### **2. Run the Project**
To run the compiled examples, use:
```bash
make run-all  # Run all examples
make run-threads  # Run only the thread example
```

### **3. Clean Up**
To remove generated files, use:
```bash
make clean
```

---

## **Summary**
This project helps you learn and practice four major concepts in C++ parallel programming:
1. **Threads**: Run tasks in parallel.
2. **Mutexes**: Safely manage shared resources.
3. **Futures**: Handle asynchronous tasks and their results.
4. **Atomics**: Safely modify shared data without locks.

