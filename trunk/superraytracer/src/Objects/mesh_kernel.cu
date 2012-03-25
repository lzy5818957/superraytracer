#include "mesh_kernel.cuh"

#include <cstdio>
#include <cfloat>
#include <cutil_math.h>

#define BLOCK_SIZE 8


__global__ void raysIntersectsMeshKernel(GLuint i0, GLuint i1, GLuint i2,float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *devHitInfos, float3 * dev_vertPositions,int objHitIndex)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;
	int arrayPos6 = 6 * (c + w * r);
	
	float3 ray_o;
	ray_o.x = devRays[arrayPos6];
	ray_o.y = devRays[arrayPos6 +1];
	ray_o.z = devRays[arrayPos6 +2];

	float3 ray_d;
	ray_d.x = devRays[arrayPos6 +3];
	ray_d.y = devRays[arrayPos6 +4];
	ray_d.z = devRays[arrayPos6 +5];

	float3 E1 = dev_vertPositions[i1] - dev_vertPositions[i0];
	float3 E2 = dev_vertPositions[i2] - dev_vertPositions[i0];

	float3 P = cross(ray_d,E2);

	float detM = dot(P,E1);
	if(fabs(detM) < 1e-4)
	{
		devHitInfos[arrayPos6].hitDist = FLT_MAX;
	}

	float3 T = ray_o - dev_vertPositions[i0];
	
	float u = dot(P,T)/detM;
	if( u < 0.0f || 1.0f < u)
	{
		devHitInfos[arrayPos1].hitDist = FLT_MAX;
	}

	float3 TxE1 = cross(T,E1);
	float v = dot(TxE1,ray_d) / detM;
	if( v < 0.0f || 1.0f < (v+u) )
	{
		devHitInfos[arrayPos1].hitDist = FLT_MAX;
	}

	float t = dot (TxE1,E2) / detM;
	if( t < t0 || t1 < t1)
	{
		devHitInfos[arrayPos1].hitDist = FLT_MAX;
	}

	devHitInfos[arrayPos1].hitDist = t;
	devHitInfos[arrayPos1].mesh.i0 = i0;
	devHitInfos[arrayPos1].mesh.i1 = i1;
	devHitInfos[arrayPos1].mesh.i2 = i2;
	devHitInfos[arrayPos1].objHitIndex = objHitIndex;

}

extern "C" RayTracing::HitInfo_t* raysIntersectsWithCudaMesh(GLuint i0, GLuint i1, GLuint i2,float *devRays, const float t0, const float t1, const int w, const int h,float3 *m_vertPositions,GLuint numVerts, GLuint numIndices,int objHitIndex)
{
	float* dev_vertPositions;
	RayTracing::HitInfo_t *devHitInfos = 0;
	cudaError_t cudaStatus;


	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void **)& dev_vertPositions, (2*sizeof(float3)+sizeof(float2))*numVerts+sizeof(GLuint)*numIndices);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_vertPositions, m_vertPositions, (2*sizeof(float3)+sizeof(float2))*numVerts+sizeof(GLuint)*numIndices,cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devHitInfos , w * h * sizeof ( RayTracing::HitInfo_t ));
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

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 
	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);  

	raysIntersectsMeshKernel <<<numBlocks, threadsPerBlock>>> (i0 , i1, i2, devRays, t0, t1, w, h, devHitInfos, (float3 *) dev_vertPositions,objHitIndex);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	return devHitInfos;

	


Error:
	cudaFree(dev_vertPositions);
	cudaFree(devHitInfos);
	dev_vertPositions = 0;
	devHitInfos=0;
	return NULL;
	


}