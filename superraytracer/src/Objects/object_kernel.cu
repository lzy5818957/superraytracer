#include "object_kernel.cuh"
#include "curand_kernel.h"
#include "cublas_v2.h"

#include <cstdio>

extern "C" cudaError_t transformRayToObjSpaceWithCuda(float *hostRayDirs,float *hostRayOrig, const int w, const int h, float *m_worldToObject)
{
	cudaError_t cudaStatus;
Error:
	//cudaFree(devRayDirs);
	return cudaStatus;
}