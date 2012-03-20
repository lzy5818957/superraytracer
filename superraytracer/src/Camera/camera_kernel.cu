#include "camera_kernel.cuh"
#include "curand_kernel.h"
#include <cstdio>

__global__ void setup_rand_kernel ( curandState * state , int w, int h)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	/* Each thread gets same seed , a different sequence
	number , no offset */
	curand_init (1234 , 3 * c + w * r, 0, & state [  3 * c + w * r ]);
	curand_init (1234 , 1 + 3 * c + w * r, 0, & state [ 1 + 3 * c + w * r ]);
}
__global__ void generate_rand_kernel ( curandState *state ,	float *result, int w, int h )
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;

	
	curandState localState = state [3 * c + w * r ];
	/* Store results */
	result [3 * c + w * r ] = curand_uniform (& localState );

	/* Copy state to local memory for efficiency */
	localState = state [1 + 3 * c + w * r ];
	result [ 1 + 3 * c + w * r ] = curand_uniform (& localState );


}

__global__ void genViewRayKernel(float *rayDirs, float* rand_result, int w, int h)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;


	float screenPositionInWorld4;


	const float x = c - 0.5 + rand_result[ 3 * c + w * r], y = r - 0.5 + rand_result[1 + 3 * c + r * w];

	float screenPos[4] = {x, y, 1, 1};

	/*
	screenPositionInWorld4 = gml::mul(m_windowToWorld, gml::vec4_t(x, y, 1, 1));
	gml::vec3_t screenPositionInWorld3 = gml::vec3_t(screenPositionInWorld4.x/screenPositionInWorld4.w,
		screenPositionInWorld4.y/screenPositionInWorld4.w,
		screenPositionInWorld4.z/screenPositionInWorld4.w);
*/
	//ray.d = gml::normalize(gml::sub(screenPositionInWorld3,ray.o));

	rayDirs[ 3 * c + w * r] = x;
	rayDirs[ 1 + 3 * c + r * w ] = y;
	rayDirs[ 2 + 3 * c + r * w ] = 0.0f;

}


extern "C" cudaError_t genViewRayWithCuda(float *hostRayDirs, const int w, const int h)
{

	float *devRayDirs = 0;
	curandState * devStates;
	float *dev_rand_result;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&devRayDirs, w * h * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	/*
	cudaMemset ( devRayDirs , 0, w * h * 3 * sizeof ( float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	*/
	cudaStatus = cudaMalloc (( void **)& dev_rand_result , 2 * w * h * sizeof ( float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaMemset ( dev_rand_result , 0, 2 * w * h * sizeof ( int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaMalloc (( void **)& devStates , 2 * w * h * sizeof ( curandState ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	dim3 threadsPerBlock(8, 8);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);  
	
	setup_rand_kernel <<<numBlocks, threadsPerBlock>>>( devStates, w, h );
	

	generate_rand_kernel <<<numBlocks, threadsPerBlock>>>( devStates , dev_rand_result, w, h );
	

	cudaFree(devStates);
	// Launch a kernel on the GPU with one thread for each element.
	genViewRayKernel<<<numBlocks, threadsPerBlock>>>(devRayDirs, dev_rand_result, w, h);
	
	cudaFree(dev_rand_result);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostRayDirs, devRayDirs, 3 * w * h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(devRayDirs);
	return cudaStatus;

}
