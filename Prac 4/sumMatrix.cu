
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <helper_cuda.h>

/*
 * This example implements matrix element-wise addition on the host and GPU.
 * sumMatrixOnHost iterates over the rows and columns of each matrix, adding
 * elements from A and B together and storing the results in C. The current
 * offset in each matrix is stored using pointer arithmetic. sumMatrixOnGPU2D
 * implements the same logic, but using CUDA threads to process each matrix.
 */

void initialData(float *ip, const int size)
{
    int i;
    for(i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
    unsigned int ix = 16 * (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x);
    
    //unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    //unsigned int idx = iy * NX + ix;
    for(int start = 0;start<16;start++){
        if (ix < NX*NY )
        {
            C[ix] = A[ix] + B[ix];
        }
        ix++;
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    checkCudaErrors(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 12;  // 14
    int ny = 1 << 12;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    sumMatrixOnHost (h_A, h_B, hostRef, nx, ny);

	// malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    checkCudaErrors(cudaMalloc((void **)&d_MatA, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_MatB, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_MatC, nBytes));

   // setup kernel launch parameters
    int dimx = 64; // default block size if no runtime parameters given
    int dimy = 2;

    if(argc > 2)
    {
        int storeInputx = atoi(argv[1]);
        int storeInputy = atoi(argv[2]);
        if(storeInputy<4){
            dimy = 1;
        }
        if(storeInputx<4){
            dimx = 1;
        }
        dimx = storeInputx/4;
        dimy = storeInputy/4;
    }
	
	dim3 block(dimx, dimy);
    dim3 grid((((nx + (block.x*16) - 1) / (block.x*16))*((ny + (block.y) - 1) / (block.y))), 1 );

    // transfer data from host to device
    checkCudaErrors(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

	// initialise CUDA timing
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);  // start timing

	// execute the kernel
    checkCudaErrors(cudaDeviceSynchronize());
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaEventRecord(stop);
	checkCudaErrors(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&milli, start, stop);  // time random generation

	printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> (ms): %f \n", grid.x, grid.y,
           block.x, block.y, milli);

    checkCudaErrors(cudaGetLastError());

    // copy kernel result back to host side
    checkCudaErrors(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // checkCudaErrors device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    checkCudaErrors(cudaFree(d_MatA));
    checkCudaErrors(cudaFree(d_MatB));
    checkCudaErrors(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    checkCudaErrors(cudaDeviceReset());

    return EXIT_SUCCESS;
}
