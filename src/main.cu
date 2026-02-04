/*
 * CUDA Signal Amplifier
 * * Description: Reads binary data files, amplifies signal using GPU, writes output.
 * Adheres to Google C++ Style Guide basics.
 */

#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

// --- CUDA KERNEL ---
// Simple kernel to scale signal values (amplify)
__global__ void AmplifySignalKernel(float* d_signal, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_signal[idx] = d_signal[idx] * factor;
    }
}

// --- HELPER FUNCTIONS ---

// Checks CUDA error status
void CheckCudaError(cudaError_t err, const std::string& msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Reads a file into a vector of floats
std::vector<float> ReadFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return {};
    
    // Get file size
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) return buffer;
    return {};
}

// Writes a vector of floats to a file
void WriteFile(const std::string& filepath, const std::vector<float>& data) {
    std::ofstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    }
}

// --- MAIN EXECUTION ---
int main(int argc, char** argv) {
    std::string input_dir = "./data/input";
    std::string output_dir = "./data/output";
    float scale_factor = 2.0f;

    // 1. Parse CLI Arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--scale" && i + 1 < argc) scale_factor = std::stof(argv[++i]);
    }

    std::cout << "Starting Batch Processing..." << std::endl;
    std::cout << "Input: " << input_dir << " | Output: " << output_dir << std::endl;

    // 2. Iterate through files
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(input_dir.c_str())) != nullptr) {
        int file_count = 0;
        
        while ((ent = readdir(dir)) != nullptr) {
            std::string filename = ent->d_name;
            if (filename == "." || filename == "..") continue;

            std::string input_path = input_dir + "/" + filename;
            std::string output_path = output_dir + "/" + filename;

            // Load Data
            std::vector<float> h_data = ReadFile(input_path);
            if (h_data.empty()) continue;

            int num_elements = h_data.size();
            size_t bytes = num_elements * sizeof(float);

            // Allocate GPU Memory
            float* d_data = nullptr;
            CheckCudaError(cudaMalloc(&d_data, bytes), "Alloc");

            // Copy to Device
            CheckCudaError(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice), "HtoD");

            // Launch Kernel
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            AmplifySignalKernel<<<blocks, threads>>>(d_data, num_elements, scale_factor);
            CheckCudaError(cudaGetLastError(), "Kernel Launch");
            
            // Sync to ensure completion
            CheckCudaError(cudaDeviceSynchronize(), "Sync");

            // Copy back to Host
            CheckCudaError(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost), "DtoH");

            // Save Result
            WriteFile(output_path, h_data);
            
            // Cleanup
            cudaFree(d_data);
            file_count++;
            
            if (file_count % 10 == 0) std::cout << "Processed " << file_count << " files..." << std::endl;
        }
        closedir(dir);
        std::cout << "Total files processed: " << file_count << std::endl;
    } else {
        std::cerr << "Could not open input directory." << std::endl;
        return 1;
    }

    return 0;
}