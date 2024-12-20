# Parallel Programming in C++

This guide is a beginner-friendly introduction to parallel programming in C++. Parallel programming allows multiple tasks to be executed simultaneously, improving the performance of your programs. This is especially useful for multi-core processors, where tasks can be distributed across different cores for faster execution.

### **What Will You Learn?**
In this guide, we will explore the following core concepts in C++ parallel programming:
1. **Threading** (How to run tasks in parallel)
2. **Mutexes** (How to ensure that only one thread accesses a specific part of code at a time)
3. **Atomic Variables** (How to safely modify shared data)
4. **Futures** (How to handle tasks that complete in the future)

Let's break down these concepts step-by-step, keeping things simple and easy to follow.

---

### **1. Threading in C++**
A **thread** in C++ is like a mini-program inside your main program. You can run several threads at the same time to do different tasks.

- **Creating a thread**: You create a thread by specifying the function you want it to run. Each thread runs independently of the others.
  
  Example:
  ```cpp
  std::thread myThread(myFunction);  // Create a thread
  myThread.join();  // Wait for the thread to finish
  ```

- **Joining a thread**: You can use the `join()` method to make sure the main program waits for the thread to finish before continuing.
  
- **Detaching a thread**: If you want the thread to run independently without waiting for it to finish, you can use `detach()`. This will let the thread run by itself and not block the main program.

---

### **2. Mutexes (Mutual Exclusion)**
A **mutex** is a tool that makes sure that only one thread can access a specific part of the code at a time. It’s like locking a door so only one person can enter.

- **Locking a mutex**: Before entering a critical section (a part of the code that needs exclusive access), a thread locks the mutex. This prevents other threads from entering that section.
  
- **Unlocking the mutex**: After the thread is done with the critical section, it unlocks the mutex, allowing other threads to enter.

Example:
```cpp
std::mutex mtx;  // Mutex declaration

void threadFunction() {
    mtx.lock();  // Lock the mutex
    // Critical section
    mtx.unlock();  // Unlock the mutex
}
```

- **Trylock**: If a thread tries to lock a mutex but can’t because it’s already locked, it can either wait or move on. Using `try_lock()` allows a thread to attempt to get the lock without blocking.

---

### **3. Atomic Variables**
An **atomic variable** is a type of variable that can be safely modified by multiple threads at the same time. Without atomic variables, if two threads try to change the same variable at the same time, they could end up messing things up.

- **Atomic operations**: An atomic variable ensures that when a thread modifies it, the operation is completed without any interference from other threads. This is useful when you need to increment a variable or perform similar operations on shared data.
  
  Example:
  ```cpp
  std::atomic<int> counter(0);  // Declare an atomic variable
  counter++;  // Safely increment the counter
  ```

- **Atomic Thread Fence**: This ensures that all threads reach a certain point before continuing, helping to manage the order of operations across threads.

---

### **4. Futures**
A **future** in C++ is a way to get the result of a task that is running in parallel. It allows your program to continue working on other tasks while waiting for the result of a parallel task.

- **Async Execution**: You can start a function asynchronously, meaning it will run in the background while your main program continues to run. The future object will hold the result of that operation once it’s done.

- **Waiting for Results**: You can use `get()` to block the main program until the future task is completed and the result is available.

Example:
```cpp
std::future<int> result = std::async(std::launch::async, computeFunction);
int value = result.get();  // Wait for the result and get it
```

---

### **Summary of Key Concepts**

1. **Threads**: Run tasks independently of the main program, allowing multiple tasks to run in parallel.
2. **Mutexes**: Protect critical sections of code by allowing only one thread to access them at a time.
3. **Atomic Variables**: Allow multiple threads to safely modify a variable without interference.
4. **Futures**: Handle tasks that will complete in the future, allowing other tasks to continue in the meantime.

### **Conclusion**
Parallel programming in C++ involves using **threads** to run tasks simultaneously, ensuring safe access to shared resources with **mutexes** and **atomic variables**, and managing future tasks with **futures**. These concepts are essential for improving the performance of your programs, especially on multi-core processors.

