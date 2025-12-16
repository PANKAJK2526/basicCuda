#include <iostream>
#include <cuda_runtime.h>

// A simple empty kernel
__global__ void messyKernel() {
    // This code never actually runs if the configuration is invalid
}

int main() {
    // ---------------------------------------------------------
    // EXPERIMENT 1: The Valid Launch
    // ---------------------------------------------------------
    std::cout << "Attempting valid launch (1 block, 1024 threads)...\n";
    messyKernel<<<1, 1024>>>();
    
    // Check for launch errors immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Valid launch FAILED: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Valid launch LOOKS good (so far)...\n";
    }

    // Force Host to wait for Device to finish
    cudaDeviceSynchronize();
    std::cout << "--------------------------------------------------\n";

    // ---------------------------------------------------------
    // EXPERIMENT 2: The Invalid Launch
    // ---------------------------------------------------------
    // Your RTX 3090 has a limit of 1024 threads per block.
    // We will try 1025.
    
    std::cout << "Attempting INVALID launch (1 block, 1025 threads)...\n";
    messyKernel<<<1, 1025>>>(); // <--- THE ERROR IS HERE

    // 1. Check for immediate configuration errors (e.g., invalid args)
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << ">>> CAUGHT ERROR: " << cudaGetErrorString(err) << "\n";
    } else {
        // If we see this, the runtime didn't catch it immediately
        std::cout << "Launch technically accepted commands... waiting for GPU...\n";
    }

    // 2. Synchronize to catch execution errors
    // Even if the launch syntax was okay, the GPU might crash instantly.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << ">>> CAUGHT SYNC ERROR: " << cudaGetErrorString(err) << "\n";
    }

    return 0;
}