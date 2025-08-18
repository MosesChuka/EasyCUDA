// CUDA provides "gridDim.x" which contains the number of blocks in the grid, and "blockIdx.x",
// which contains the index of the current thread block in the grid. The idea is that each thread gets its index by computing the offset to the 
// beginning of its block (the blcok index times the block size, and adding the thread's index within the block. 
// The code "blockIdx.x * blockDim.x + threadIdx.x" is idiomatic CUDA.
#include <iostream>
#include <math.h>
#include <bitset>


// We need to modify the function so that it distributes the computation accross threads
// CUDA C++ provides keywords that let kernels get the indices of the running threads. 
// "ThreadIdx.x" contains the index of the current thread within its block, and 
// "blockDim.x" contains the number of threads in the block.
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // block index * number of threads in a block * specific thread index
    int stride = blockDim.x * gridDim.x;               // block dimension or number of threads in a block * number of blocks in

    for(int i = index; i < n; i += stride){  // Each thead begins with a different index value. When the block is full, the stride is increased.
        y[i] = x[i] + y[i];
    }
}


int main(void){
    int N = 1 <<20; // 1 million elements
    float *x, *y;

    // Need to first initialize the memory before Prefetching, else a seg error occurs
    cudaMallocManaged(&x,N*sizeof(float));
    cudaMallocManaged(&y,N*sizeof(float));

    // Memory Allocation in device instead of host(contrast to program 4)
    cudaMemPrefetchAsync(x, N*sizeof(float),0,0); //the 0,0 represents the destination device and stream respectively. 
    cudaMemPrefetchAsync(y, N*sizeof(float),0,0);
   
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1)/blockSize;

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

  
    add<<<numBlocks, blockSize>>>(N,x,y);

    // CPU should wait for GPU
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << std::float_t(maxError) << std::endl;


    // Freeing memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}