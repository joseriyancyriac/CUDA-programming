#include <stdio.h>

// This is our GPU function (a kernel)
__global__ void helloFromGPU() {
    // Each thread prints its unique ID
    printf("Hello from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch the kernel with 2 blocks, each having 14 threads
    helloFromGPU<<<2, 14>>>();

    // Wait for GPU to finish before CPU exits
    cudaDeviceSynchronize();

    return 0;
}
