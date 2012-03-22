#include "object_kernel.cuh"
#include "curand_kernel.h"
#include "cublas_v2.h"
#include "../Util/cudaVectUtil.cu"
#include <cutil_math.h>

#include <cstdio>


__global__ void tfRayWdToObj(float3 *rays, float *m_worldToObject, int w)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;

	int arraypos2 = 2*(c + w*r);

	float4 ray_wd = make_float4(rays[arraypos2], 1.0f);
	float4 ray_obj;
	Mat4x4_Mul_Vec4(m_worldToObject, (float *)(&ray_wd), (float *)(&ray_obj) );
	rays[arraypos2] = make_float3(ray_obj);

	ray_wd = make_float4(rays[arraypos2+1], 1.0f);
	Mat4x4_Mul_Vec4(m_worldToObject, (float *)(&ray_wd), (float *)(&ray_obj) );
	rays[arraypos2+1] = make_float3(ray_obj);
}

extern "C" cudaError_t transformRayToObjSpaceWithCuda(float *hostRays, const int w, const int h, float *m_worldToObject)
{
	float3 *devRays = 0;
	float4 *dev_wdToObj = 0;
	
	curandState * devStates;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devRays , w*h*2* sizeof ( float3 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devRays, hostRays, w*h*2* sizeof(float3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& dev_wdToObj , 4 * sizeof ( float4 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_wdToObj, m_worldToObject, 4 * sizeof(float4), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	tfRayWdToObj(devRays, m_worldToObject, w);

Error:
	//cudaFree(devRayDirs);
	free(devRays);
	free(dev_wdToObj);
	return cudaStatus;
}