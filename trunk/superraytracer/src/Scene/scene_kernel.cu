#include "scene_kernel.cuh"
#include <cstdio>

extern "C" RayTracing::HitInfo_t* raysIntersectsWithCudaScene(float *devRays, const float t0, const float t1, const int w, const int h)
{
	cudaError_t cudaStatus;

	
Error:
	printf("CUDA ERROR OCCURED\n");

	return NULL;
}