#include "mesh_kernel.cuh"

#include <cstdio>
#include <cfloat>
#include <cutil_math.h>

#define BLOCK_SIZE 8


__global__ void raysIntersectsMeshKernel(float3 *m_vertPositions,GLuint i0, GLuint i1, GLuint i2,float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *hostHitInfos)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos6 = 6 * (c + w * r);
	
	float3 ray_o;
	ray_o.x = arrayPos6;
	ray_o.y = arrayPos6 +1;
	ray_o.z = arrayPos6 +2;

	float3 ray_d;
	ray_d.x = arrayPos6 +3;
	ray_d.y = arrayPos6 +4;
	ray_d.z = arrayPos6 +5;

	float3 E1 = m_vertPositions[i1] - m_vertPositions[i0];
	float3 E2 = m_vertPositions[i2] - m_vertPositions[i0];

	float3 P = cross(ray_d,E2);

	float detM = dot(P,E1);
	if(fabs(detM) < 1e-4)
	{
		hostHitInfos[arrayPos6].hitDist = FLT_MAX;
	}

	float3 T = ray_o - m_vertPositions[i0];
	
	float u = dot(P,T)/detM;
	if( u < 0.0f || 1.0f < u)
	{
		hostHitInfos[arrayPos6].hitDist = FLT_MAX;
	}

	float3 TxE1 = cross(T,E1);
	float v = dot(TxE1,ray_d) / detM;
	if( v < 0.0f || 1.0f < (v+u) )
	{
		hostHitInfos[arrayPos6].hitDist = FLT_MAX;
	}

	float t = dot (TxE1,E2) / detM;
	if( t < t0 || t1 < t1)
	{
		hostHitInfos[arrayPos6].hitDist = FLT_MAX;
	}

	hostHitInfos[arrayPos6].hitDist = t;
	hostHitInfos[arrayPos6].mesh.i0 = i0;
	hostHitInfos[arrayPos6].mesh.i1 = i1;
	hostHitInfos[arrayPos6].mesh.i2 = i2;

}

extern "C" cudaError_t raysIntersectsWithCudaMesh(float3 *m_vertPositions,GLuint i0, GLuint i1, GLuint i2,float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *hostHitInfos)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
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

	raysIntersectsMeshKernel <<<numBlocks, threadsPerBlock>>> (m_vertPositions,i0,i1,i2,devRays,t0,t1,w,h,hostHitInfos);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

devRays = 0;

Error:
	return cudaStatus;


}