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

extern "C" cudaError_t transformRayToObjSpaceWithCuda(float *devRays, const int w, const int h, float *m_worldToObject)
{

	cudaError_t cudaStatus;
Error:
	//cudaFree(devRayDirs);
	return cudaStatus;
}