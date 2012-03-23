#include "sphere_kernel.cuh"
#include <cstdio>
#include <cfloat>
#include <cutil_math.h>

#define BLOCKSIZE 8

__global__ void raysIntersectsSphereKernel(float *devRays, const float t0, const float t1,const int w, const int h,RayTracing::HitInfo_t *hitInfos)
{
	float A,B,C;
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

	A = dot(ray_d,ray_d);
	B = dot(ray_d,ray_o);
	C = dot(ray_o,ray_o) - 1.0;

	float det = B*B - A*C;

	if(det < 0.0)
	{
		hitInfos[arrayPos6].hitDist = FLT_MAX;
	}else
	{
		
		float t =  (-B - sqrt(B * B - A * C)) / A;
	    if(t > t1 || t < t0 )
	    {

	       hitInfos[arrayPos6].hitDist = FLT_MAX;
	    }
	    else
	    {
 	       hitInfos[arrayPos6].hitDist = t;
		   float3 shadePoint = ray_o + (t * ray_d);
		   hitInfos[arrayPos6].sphere.shadePoint_x = shadePoint.x;
		   hitInfos[arrayPos6].sphere.shadePoint_y = shadePoint.y;
		   hitInfos[arrayPos6].sphere.shadePoint_z = shadePoint.z;

	    }
		


	}
}


extern "C" RayTracing::HitInfo_t* raysIntersectsWithCudaSphere(float *devRays, const float t0, const float t1,const int w, const int h)
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

	dim3 threadsPerBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 numBlocks(w/threadsPerBlock.x,h/threadsPerBlock.y);

	raysIntersectsSphereKernel <<< numBlocks, threadsPerBlock>>> (devRays,t0,t1,w,h,devHitInfos);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree(devHitInfos);
	devHitInfos = 0;

Error:
	cudaFree(devHitInfos);
	return devHitInfos;

}