#include "plane_kernel.cuh"

#include <cstdio>
#include <cutil_math.h>
#include <cfloat>

#define BLOCK_SIZE 8

__global__ void raysIntersectsPlaneKernel(float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *hitInfos, float3 vert0)
{

	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;
	int arrayPos6 = 6 * (c + w * r);

	float3 E1 = {0.0f, 0.0f, 2.0f};
	float3 E2 = {2.0f, 0.0f, 0.0f};
	float3 rayDir = {devRays[arrayPos6 + 3], devRays[arrayPos6 + 4], devRays[arrayPos6 + 5]};
	float3 rayOri = {devRays[arrayPos6], devRays[arrayPos6 + 1], devRays[arrayPos6 + 2]};
	float3 P = cross(rayDir, E2);

	float detM = dot(P, E1);

	if (fabs(detM) < 1e-4)
	{
		hitInfos[arrayPos1].hitDist = FLT_MAX;
		return;
	}

	float3 T = rayOri - vert0;

	float u = dot( P, T ) / detM;
	
	if ( u < 0.0f || 1.0f < u )
	{
		hitInfos[arrayPos1].hitDist = FLT_MAX;
		return;
	}

	float3 TxE1 = cross(T, E1);
	float v = dot( TxE1, rayDir ) / detM;
	if ( v < 0.0f || 1.0f < v)
	{
		hitInfos[arrayPos1].hitDist = FLT_MAX;
		return;
	}

	float t = dot( TxE1, E2 ) / detM;
	if (t < t0 || t1 < t)
	{
		hitInfos[arrayPos1].hitDist = FLT_MAX;
		return;
	}

	hitInfos[arrayPos1].hitDist =  t;
  
	hitInfos[arrayPos1].plane.u = u;
	hitInfos[arrayPos1].plane.v = v;
	

}

extern "C" cudaError_t raysIntersectsWithCudaPlane(float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *hitInfos)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
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

	raysIntersectsPlaneKernel <<<numBlocks, threadsPerBlock>>>(devRays, t0, t1, w, h, hitInfos, make_float3(1,1,1));


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