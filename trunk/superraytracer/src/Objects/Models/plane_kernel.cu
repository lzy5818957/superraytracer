#include "plane_kernel.cuh"
#include <cstdio>

#define BLOCK_SIZE 8

__global__ void raysIntersectsPlaneKernel(float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *hitInfos)
{
	gml::vec3_t E1(0.0f, 0.0f, 2.0f);
	gml::vec3_t E2(2.0f, 0.0f, 0.0f);

	gml::vec3_t P = gml::cross( ray.d, E2 );

	float detM = gml::dot(P, E1);

	if (fabs(detM) < 1e-4)
	{
		return false;
	}

	gml::vec3_t T = gml::sub( ray.o, _verts[0] );

	float u = gml::dot( P, T ) / detM;
	
	if ( u < 0.0f || 1.0f < u )
	{
		return false;
	}

	gml::vec3_t TxE1 = gml::cross(T, E1);
	float v = gml::dot( TxE1, ray.d ) / detM;
	if ( v < 0.0f || 1.0f < v)
	{
		return false;
	}

	float t = gml::dot( TxE1, E2 ) / detM;
	if (t < t0 || t1 < t)
	{
		return false;
	}

	hitinfo.hitDist =  t;
  
	hitinfo.plane.u = u;
	hitinfo.plane.v = v;
	
	return true;


}

extern "C" cudaError_t raysIntersectsWithCudaPlane(float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *hostHitInfos)
{
	RayTracing::HitInfo_t *devHitInfos = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
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

	raysIntersectsPlaneKernel <<<numBlocks, threadsPerBlock>>>(devRays, t0, t1, w, h, devHitInfos);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostHitInfos, devHitInfos, w * h * sizeof( RayTracing::HitInfo_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaFree(devHitInfos);
	devRays = 0;

Error:
	cudaFree(devHitInfos);
	return cudaStatus;
}