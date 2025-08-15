#include <stdio.h>

__global__ void hello2D() {
    printf("Block(%d, %d), Thread(%d, %d)\n", 
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main() {
    dim3 grid(2, 3); // 2 blocks in x, 3 blocks in y
    dim3 block(4, 2); // 4 threads in x, 2 threads in y

    // Launch the kernel
    hello2D<<<grid, block>>>();
    
    cudaDeviceSynchronize();
    
    return 0;
}