#include "scene_kernel.cuh"
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 8

__global__ void findClosestHits(RayTracing::HitInfo_t** hitInfos_array, const int w, const int h, RayTracing::HitInfo_t* closestHits, const int m_nObjects)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;
	
	RayTracing::HitInfo_t closestHit = hitInfos_array[0][arrayPos1];

	for(int i = 1 ; i < m_nObjects; i++ )
	{
		if(hitInfos_array[i][arrayPos1].hitDist < closestHit.hitDist)
		{
			closestHit = hitInfos_array[i][arrayPos1];
		}
		
	}
	closestHits[arrayPos1] = closestHit;
}


extern "C" RayTracing::HitInfo_t* hitInfoDTH(const RayTracing::HitInfo_t *devHitInfos, const int w, const int h)
{
	cudaError_t cudaStatus;

	RayTracing::HitInfo_t* hostHitInfos;


	hostHitInfos = (RayTracing::HitInfo_t*)malloc(w * h * sizeof(RayTracing::HitInfo_t));

	cudaStatus = cudaMemcpy(hostHitInfos, devHitInfos, w * h * sizeof(RayTracing::HitInfo_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	return hostHitInfos;

Error:

	printf("CUDA ERROR OCCURED\n");

	return NULL;
}

extern "C" RayTracing::Ray_t* rayHTD(const RayTracing::Ray_t *hostRays, const int w, const int h)
{
	cudaError_t cudaStatus;

	RayTracing::Ray_t* devRays = 0;


	cudaStatus = cudaMalloc (( void **)& devRays , 6 * w * h * sizeof ( float ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devRays, hostRays, 6 * w * h * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	return devRays;

Error:

	printf("CUDA ERROR OCCURED\n");

	return NULL;
}

extern "C" RayTracing::HitInfo_t* findClosestHits(const RayTracing::HitInfo_t** hitInfos_array, const int w, const int h, const int m_nObjects)
{	

	RayTracing::HitInfo_t* closestHits = 0;
	RayTracing::HitInfo_t** devHitInfos_array = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc (( void **)& devHitInfos_array , m_nObjects * sizeof ( RayTracing::HitInfo_t* ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devHitInfos_array, hitInfos_array, m_nObjects * sizeof(RayTracing::HitInfo_t*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& closestHits , w * h * sizeof ( RayTracing::HitInfo_t ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching findClosestHits!\n", cudaStatus);
		goto Error;
	}

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);  
	
	findClosestHits <<<numBlocks, threadsPerBlock>>>( devHitInfos_array, w, h, closestHits, m_nObjects );


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findClosestHits!\n", cudaStatus);
		goto Error;
	}

	for(int i = 0 ; i < m_nObjects; i++)
	{
		cudaFree((void*)(hitInfos_array[i]));
		hitInfos_array[i] = 0;
	}
	cudaFree(devHitInfos_array);
	return closestHits;
	

Error:

	printf("CUDA ERROR OCCURED\n");

	return NULL;
}