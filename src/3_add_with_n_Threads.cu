// The previous kernel runs on one thread and one block. To change this, we edit the CUDA's <<<1, 1>>> syntax, called the execution configuration.
// The execution configuration tells the cuda runtime how many parallel threads to use for the launch on the GPU.
// We will start by changing the second one which represents the number of threads in a thread block.
// CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size;
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
    int index = threadIdx.x;
    int stride = blockDim.x;

    for(int i = index; i < n; i += stride)  //Does index increase automatically?
        y[i] = x[i] + y[i];

}


int main(void){
    int N = 1 <<20; // 1 million elements
    float *x, *y;

    // Memory Allocation in CUDA
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
   

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // CUDA kernel launches are specified using the TRIPLE ANGLE bracket syntax <<<...>>>.
    // <<<>>> is the execution configuration which tells CUDA runtime how many parallel threads to use for the launch on the GPU.
    // There are two parameters in the execution configuration, but only one will be changed(the second parameter).
    // Run kernel with 1M elements on the GPU
    // 256 threads is a reasonable size to choose
    add<<<1, 256>>>(N,x,y);

    cudaError_t err = cudaGetLastError();
    if (err !=cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // We need CPU to wait until the kernel is done before accessing the results. 
    // CUDA kernel launches don't block the calling CPU thread.
    // We do this by calling the synchronize function:
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