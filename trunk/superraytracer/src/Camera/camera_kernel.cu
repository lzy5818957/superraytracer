#include "camera_kernel.cuh"
#include "curand_kernel.h"
#include <cutil_math.h>

#include <cstdio>

#include "../Util/cudaVectUtil.cu"


#define BLOCK_SIZE 8

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

__global__ void genRaysKernel(float *rays, float *camPos, float* rand_result, int w, int h, float *m_windowToWorld)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;

	int arrayPos2 = 2 * c + w * r;
	int arrayPos6 = 6 * c + w * r;

	const float x = c - 0.5 + rand_result[ arrayPos2 ], y = r - 0.5 + rand_result[1 + arrayPos2];
	
	float screenPosition4[4] = {x, y, 1, 1};
	float4x4 screenPositionInWorld4 = *(float4*)(m_windowToWorld) * 

	Mat4x4_Mul_Vec4(,screenPosition4,screenPositionInWorld4);

	float screenPositionInWorld3Nrm[3];

	Vec3_Nrm(screenPositionInWorld4,screenPositionInWorld3Nrm);

	
	/*
	screenPositionInWorld4 = gml::mul(m_windowToWorld, gml::vec4_t(x, y, 1, 1));
	gml::vec3_t screenPositionInWorld3 = gml::vec3_t(screenPositionInWorld4.x/screenPositionInWorld4.w,
		screenPositionInWorld4.y/screenPositionInWorld4.w,
		screenPositionInWorld4.z/screenPositionInWorld4.w);
		*/
	//ray.d = gml::normalize(gml::sub(screenPositionInWorld3,ray.o));
	
	rays[ arrayPos6] = camPos[0];
	rays[ 1 + arrayPos6 ] = camPos[1];
	rays[ 2 + arrayPos6 ] = camPos[2];
	//rays[ 3 + arrayPos6 ] = 1.0f;

}


extern "C" cudaError_t genViewRayWithCuda(float *hostRays, const int w, const int h,  float *hostCamPos, float *hostWindowToWorld)
{


	float *devRays = 0;
	float *devWindowToWorld = 0;
	float *devCamPos = 0;
	float *devRandResult = 0;

	curandState * devStates;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&devRays, 6 * w * h * sizeof(float));
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

	cudaStatus = cudaMemcpy(devWindowToWorld, hostWindowToWorld, 16 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devCamPos , 3 * sizeof ( float ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devCamPos, hostCamPos, 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "hostCamPos->devCamPos cudaMemcpy failed!");
		goto Error;
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 

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
	genRaysKernel<<<numBlocks, threadsPerBlock>>>(devRays, devCamPos, devRandResult, w, h, devWindowToWorld);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching genViewRayKernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree(devRandResult);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching cublasSgemm!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostRays, devRays, 6 * w * h * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(devRays);
	cudaFree(devWindowToWorld);
	cudaFree(devRandResult);
	cudaFree(devCamPos);
	return cudaStatus;

}