#include "camera_kernel.cuh"
#include "curand_kernel.h"
#include "cublas_v2.h"

#include <cstdio>

__global__ void setup_rand_kernel ( curandState * state , int w, int h)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;

	/* Each thread gets same seed , a different sequence
	number , no offset */
	curand_init (1234 , arrayPos1, 0, & state [ arrayPos1 ]);
}
__global__ void generate_rand_kernel ( curandState *state ,	float *result, int w, int h )
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos2 = 2 * c + w * r;
	
	curandState localState = state [ c + w * r ];
	/* Store results */
	result [ arrayPos2 ] = curand_uniform (& localState );

	/* Copy state to local memory for efficiency */
	result [ 1 + arrayPos2 ] = curand_uniform (& localState );


}

__global__ void genScreenPosKernel(float *screenPos, float* rand_result, int w, int h)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;

	int arrayPos2 = 2 * c + w * r;
	int arrayPos4 = 4 * c + w * r;

	float screenPositionInWorld4[16];

	const float x = c - 0.5 + rand_result[ arrayPos2 ], y = r - 0.5 + rand_result[1 + arrayPos2];

	/*
	screenPositionInWorld4 = gml::mul(m_windowToWorld, gml::vec4_t(x, y, 1, 1));
	gml::vec3_t screenPositionInWorld3 = gml::vec3_t(screenPositionInWorld4.x/screenPositionInWorld4.w,
		screenPositionInWorld4.y/screenPositionInWorld4.w,
		screenPositionInWorld4.z/screenPositionInWorld4.w);
		*/
	//ray.d = gml::normalize(gml::sub(screenPositionInWorld3,ray.o));

	screenPos[ arrayPos4] = x;
	screenPos[ 1 + arrayPos4 ] = y;
	screenPos[ 2 + arrayPos4 ] = 1.0f;
	screenPos[ 3 + arrayPos4 ] = 1.0f;

}


extern "C" cudaError_t genViewRayWithCuda(float *hostRayDirs, const int w, const int h,  float *camPos, float *hostWindowToWorld)
{


	float *devRayDirs = 0;
	float *devScreenPos = 0;
	float *devScreenPosInWorld = 0;
	float *devWindowToWorld = 0;
	curandState * devStates;
	float *devRandResult;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&devRayDirs, 3 * w * h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&devScreenPos, 4 * w * h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&devScreenPosInWorld, 4 * w * h * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc (( void **)& devRandResult , 2 * w * h * sizeof ( float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc (( void **)& devStates , w * h * sizeof ( curandState ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devWindowToWorld , 16 * sizeof ( float ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	dim3 threadsPerBlock(8, 8);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);  
	
	setup_rand_kernel <<<numBlocks, threadsPerBlock>>>( devStates, w, h );

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	generate_rand_kernel <<<numBlocks, threadsPerBlock>>>( devStates , devRandResult, w, h );
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching generate_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree(devStates);


	// Launch a kernel on the GPU with one thread for each element.
	genScreenPosKernel<<<numBlocks, threadsPerBlock>>>(devScreenPos, devRandResult, w, h);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching genViewRayKernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree(devRandResult);

	//init cublas
	cublasStatus_t cublasStatus; 
	cublasHandle_t cublasHandle; 
	cublasStatus = cublasCreate(&cublasHandle);

	float alpha = 1.0f;
	float beta = 0.0f;

	cublasStatus = cublasSetVector(16, sizeof(hostWindowToWorld[0]), hostWindowToWorld, 1, devWindowToWorld, 1);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSetVector returned error code %d after launching cublasSetVector!\n", cublasStatus);
		goto Error;
    }

	cublasStatus = cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 1, 4, &alpha, devWindowToWorld, 4, devScreenPos, 4, &beta, devScreenPosInWorld, 4);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "cublasSgemm returned error code %d after launching cublasSgemm!\n", cublasStatus);
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching cublasSgemm!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostRayDirs, devScreenPosInWorld, 4 * w * h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(devRayDirs);
	return cudaStatus;

}
