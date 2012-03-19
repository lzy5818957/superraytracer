#include "camera_kernel.cuh"
#include <cstdio>


__global__ void genViewRayKernel(RayTracing::Ray_t *rays)
{
    int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;

	printf("%d, %d\n", c ,r);
}


extern "C" cudaError_t genViewRayWithCuda(RayTracing::Ray_t *host_rays, const int w, const int h)
{

    RayTracing::Ray_t *dev_rays = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_rays, w * h * sizeof(RayTracing::Ray_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	dim3 threadsPerBlock(8, 8);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
                h/threadsPerBlock.y);  


    // Launch a kernel on the GPU with one thread for each element.
    genViewRayKernel<<<numBlocks, threadsPerBlock>>>(dev_rays);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_rays, dev_rays, w * h * sizeof(RayTracing::Ray_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_rays);
    
    return cudaStatus;

}
