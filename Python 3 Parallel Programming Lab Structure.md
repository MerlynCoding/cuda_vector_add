# **Python 3 Parallel Programming Lab Structure**

In this guide, we'll explore the **Python 3 parallel programming lab**, which focuses on **multithreading** and **thread synchronization** using libraries such as **threading**, **locks**, and **semaphores**. This lab includes various Python files, each demonstrating different concepts related to parallelism.

### **Directory Structure Overview**

The Python 3 parallel programming lab activity contains several important files:
- **`__init__.py`**: Initializes the package.
- **`_pycache_`**: Stores compiled Python files.
- **`requirements.txt`**: Lists the libraries needed for the project, which can be installed using `pip`.
- **Python Example Files**: These include `core.py`, `start_new_thread_example.py`, `threading_lock_acquire_release_example.py`, and `threading_semaphore_example.py`.

### **Purpose of `core.py`**

The `core.py` file contains core functionality shared across multiple example Python files. Here's a breakdown of what `core.py` does:
1. **Thread Functions**:
   - `thread_function`: This function represents what each thread will do. In this case, it prints a message indicating that the thread is starting, then sleeps for 1 second before printing that it has finished.
   - `critical_section_acquire_release`: This function handles thread synchronization. It acquires a lock (or semaphore) to ensure that only one thread accesses the critical section at a time, performs its work (in this case, calls `thread_function`), and then releases the lock.

2. **Class with Parsing Capabilities**:
   - The **`core.py` class** also contains methods for handling **command-line arguments** using the `argparse` library.
   - **`parse_args()`** is used to parse the arguments passed from the command line.
   - The **`add_argument()`** method defines the command-line arguments that can be passed to the script (like `-n` for the number of threads to run).

### **Example Files for Parallel Programming Concepts**

1. **`StartNewThreadExample.py`**:
   - This file demonstrates creating new threads in Python.
   - The constructor parses the `-n` argument, which specifies how many threads should be executed in parallel.
   - The `run()` function starts multiple threads, each running the `thread_function()` from **`core.py`**.
   - Each thread prints that it has started, performs its work (sleeps for 1 second), and then prints that it has finished.

2. **`ThreadingLockAcquireReleaseExample.py`**:
   - This example shows how to use **locks** to manage thread synchronization.
   - Each thread in the `run()` method creates a thread object that executes the same critical section of code.
   - A **lock object** is used to ensure that no two threads can enter the critical section at the same time, preventing race conditions.

3. **`ThreadingSemaphoreExample.py`**:
   - This example demonstrates the use of **semaphores** to control access to a critical section by limiting how many threads can enter at once.
   - A **semaphore** allows a predefined number of threads to execute the critical section of code simultaneously. Each thread acquires the semaphore before entering and releases it afterward.

### **Summary of Key Concepts**

- **Threading**: Used for concurrent execution, allowing multiple tasks to run simultaneously. Threads are great for tasks like I/O operations that can be performed while waiting for resources.
  
- **Locks**: Ensures that only one thread can access a critical section of code at a time. This prevents **race conditions** where two threads might modify shared data simultaneously.

- **Semaphores**: Similar to locks but can allow a limited number of threads to access a critical section concurrently, which can be more efficient when you need to control access by multiple threads.

- **Command-Line Argument Parsing**: The **`core.py`** file allows for the setup of arguments, such as the number of threads, which can be passed when running the Python script from the command line.

---

### **Conclusion**

The **Python 3 parallel programming lab** introduces fundamental concepts for working with **threads**, **locks**, and **semaphores** in Python. By running the provided example scripts, you can learn how to:
- Create and manage threads.
- Use synchronization techniques (like locks and semaphores) to manage concurrent access to shared resources.
- Parse command-line arguments to control the number of threads or processes to run.

This hands-on experience will give you a solid foundation in parallel programming, which is crucial for building efficient applications that can perform multiple tasks concurrently.
