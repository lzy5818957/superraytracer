#include <cutil_math.h>

#include "plane_kernel.cuh"
#include <cstdio>
#include <cfloat>

#define BLOCK_SIZE 8

__global__ void hitPropertiesPlaneKernel(const RayTracing::HitInfo_t *hitinfos,  const int w, const int h , float *normTex)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = (c + w * r);
	int arrayPos5 = 5 * (c + w * r);

	normTex[arrayPos5] = 0.0f;
	normTex[arrayPos5 + 1] = 1.0f;
	normTex[arrayPos5 + 2] = 0.0f;
	normTex[arrayPos5 + 3] = hitinfos[arrayPos1].plane.u;
	normTex[arrayPos5 + 4] = hitinfos[arrayPos1].plane.v;

}

__global__ void raysIntersectsPlaneKernel(float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *hitInfos, float3 *vert0, int objHitIndex)
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

	float3 T = rayOri - (*vert0);

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
	hitInfos[arrayPos1].objHit = (Object::Object*)objHitIndex;
	hitInfos[arrayPos1].plane.u = u;
	hitInfos[arrayPos1].plane.v = v;
	

}


__global__ void shadowRaysPlaneKernel(const float *devRays, const RayTracing::HitInfo_t *hitinfos, const float* lightProp, float* vert0, const int w, const int h, bool* isInShadow)
{

	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;
	int arrayPos6 = 6 * (c + w * r);

	isInShadow[arrayPos1] = false;
	return; 

	float3 E1 = {0.0f, 0.0f, 2.0f};
	float3 E2 = {2.0f, 0.0f, 0.0f};
	float3 lightPos = make_float3(lightProp[0], lightProp[1], lightProp[2]);
	float3 rayDir = {devRays[arrayPos6 + 3], devRays[arrayPos6 + 4], devRays[arrayPos6 + 5]};
	float3 rayOri = {devRays[arrayPos6], devRays[arrayPos6 + 1], devRays[arrayPos6 + 2]};
	float3 P = cross(rayDir, E2);

	float t0 = 0.0001;
	float t1 = length(lightPos - rayOri);

	float detM = dot(P, E1);

	if (fabs(detM) < 1e-4)
	{
		isInShadow[arrayPos1] = false;
		return;
	}

	float3 T = rayOri - (*vert0);

	float u = dot( P, T ) / detM;
	
	if ( u < 0.0f || 1.0f < u )
	{
		isInShadow[arrayPos1] = false;
		return;
	}

	float3 TxE1 = cross(T, E1);
	float v = dot( TxE1, rayDir ) / detM;
	if ( v < 0.0f || 1.0f < v)
	{
		isInShadow[arrayPos1] = false;
		return;
	}

	float t = dot( TxE1, E2 ) / detM;
	if (t < t0 || t1 < t)
	{
		isInShadow[arrayPos1] = false;
		return;
	}

	isInShadow[arrayPos1] = true;
	

}

extern "C" RayTracing::HitInfo_t* raysIntersectsWithCudaPlane(float *devRays, const float t0, const float t1, const int w, const int h, float* hostVerts,int objHitIndex)
{
	float *devVert0 = 0;
	RayTracing::HitInfo_t *devHitInfos = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	cudaStatus = cudaMalloc (( void **)& devVert0 , 3 * sizeof ( float ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devVert0, hostVerts, 3 * sizeof(float), cudaMemcpyHostToDevice);
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

	raysIntersectsPlaneKernel <<<numBlocks, threadsPerBlock>>>(devRays, t0, t1, w, h, devHitInfos, (float3*)devVert0, objHitIndex);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree(devRays);
	devRays = 0;

	cudaFree(devVert0);
	devVert0 = 0;

	return devHitInfos;

Error:
	cudaFree(devHitInfos);
	devHitInfos = 0;
	cudaFree(devVert0);
	devVert0 = 0;
	return NULL;
}


extern "C" float* hitPropertiesWithCudaPlane(const RayTracing::HitInfo_t *hitinfos,  const int w, const int h)
{
	float *devNormTex = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc (( void **)& devNormTex , 5 * w * h * sizeof ( float ));
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

	hitPropertiesPlaneKernel <<<numBlocks, threadsPerBlock>>>(hitinfos, w, h,devNormTex );


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}


	return devNormTex;

Error:

	cudaFree(devNormTex);
	return NULL;
}

extern "C" bool* shadowRaysWithCudaPlane(const RayTracing::Ray_t *rays, const RayTracing::HitInfo_t *hitinfos, const float* lightProp, float* hostVerts, const int w, const int h)
{

	float *devVert0 = 0;
	bool *devIsInShadow = 0;
	cudaError_t cudaStatus;
	
	cudaStatus = cudaMalloc (( void **)& devVert0 , 3 * sizeof ( float ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devVert0, hostVerts, 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc (( void **)& devIsInShadow , w * h * sizeof ( bool ));
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

	shadowRaysPlaneKernel <<<numBlocks, threadsPerBlock>>>((float*)rays, hitinfos, lightProp, devVert0, w, h, devIsInShadow);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree((void*)rays);
	rays = 0;

	cudaFree(devVert0);
	devVert0 = 0;

	return devIsInShadow;

Error:
	cudaFree((void*)rays);
	rays = 0;
	cudaFree(devIsInShadow);
	devIsInShadow = 0;
	cudaFree(devVert0);
	devVert0 = 0;
	return NULL;


}