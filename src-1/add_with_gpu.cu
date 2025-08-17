// Here the "add" fuction in the "add_with_cpu.cpp" file was turned to a function that GPU can run,
// which is commonly refered to as a kernel in CUDA.
// The kernal is defined by adding the "__global__" specifier to above the function, which tells the CUDA C++ compiler
// That this is a function that runs on the GPU and can be called from CPU code.
// Finally, cuda files have the file extension ".cu" and are compiled using "nvcc"
#include <iostream>
#include <math.h>
#include <bitset>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
 for(int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}


int main(void){
    int N = 1 <<20; // 1 million elements
    float *x, *y;

    // Memory Allocation in CUDA
    // Unified memory in CUDA makes it easy to provide single memory space accessible by all GPUs and CPUs in the system.
    // To allocate data in the unified memory, the "cudaMallocManaged()", returns a pointer that the CPU(host) or GPU(device) code can access.
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
   

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // CUDA kernel launches are specified using the TRIPLE ANGLE bracket syntax <<<...>>>.
    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N,x,y);


    // We need CPU to wait until the kernel is done before accessing the results. CUDA kernel launches don't block the calling CPU thread.
    // we do this by calling the synchronize function:
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
    // Note that a data race will occure in the "add" kernel because it tries to access N element sin a single thread at once which is inefficient.
}