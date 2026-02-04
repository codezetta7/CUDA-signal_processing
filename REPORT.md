```markdown
# Project Report: CUDA Signal Processing at Scale

## Project Overview
The goal of this project was to develop a GPU-accelerated application capable of processing large batches of signal data. I implemented a CUDA kernel that performs parallel signal amplification on raw binary data streams.

## Implementation Details
* **Algorithm:** The core algorithm maps each data point of the signal to a thread on the GPU. I used a 1D grid and block configuration to ensure scalability across varying signal lengths.
* **Memory Management:** Unified memory or explicit `cudaMemcpy` was used to handle data transfer between Host (CPU) and Device (GPU).
* **Batch Processing:** The host code utilizes `dirent.h` to iterate through input directories, allowing the software to handle hundreds of files in a single execution loop.

## Performance & Lessons Learned
* **Throughput:** Moving memory between CPU and GPU is the bottleneck for small files. Future optimizations could involve using CUDA Streams to overlap transfer and execution.
* **Scalability:** The kernel approach is highly scalable; increasing the file size (signal length) automatically utilizes more GPU threads without code changes.