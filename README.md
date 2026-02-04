# CUDA Signal Processor

## Description
This project implements a parallelized signal processing pipeline using CUDA. It reads a batch of binary signal files, transfers them to the GPU, applies an amplification kernel, and writes the results back to disk. It is designed to demonstrate high-throughput processing of hundreds of small data files.

## Requirements
* CUDA Toolkit (NVCC)
* Linux Environment
* GNU Make

## Directory Structure
* `src/`: Contains source code.
* `data/`: Contains input and output directories.
* `run.sh`: Automated build and test script.

## Usage
### Automated Run
The easiest way to test the project is using the helper script, which generates data, compiles, and runs:
```bash
./run.sh