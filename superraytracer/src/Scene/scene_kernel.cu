#include "scene_kernel.cuh"
#include <cstdio>
#include <cstdlib>

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