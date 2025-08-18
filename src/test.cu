#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    kernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}