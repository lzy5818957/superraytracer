#include "scene_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#define BLOCK_SIZE 8

__global__ void findClosestHitsKernel(RayTracing::HitInfo_t** hitInfos_array, const int w, const int h, RayTracing::HitInfo_t* closestHits, const int m_nObjects)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;
	
	RayTracing::HitInfo_t closestHit = hitInfos_array[0][arrayPos1];

	for(int i = 1 ; i < m_nObjects; i++ )
	{
		if(hitInfos_array[i][arrayPos1].hitDist < closestHit.hitDist)
		{
			closestHit = hitInfos_array[i][arrayPos1];
		}
		
	}
	closestHits[arrayPos1] = closestHit;

}

__global__ void shadeRaysKernel(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const RayTracing::Object_Kernel_t* objects,
	const float4* lightPos,
	const int remainingRecursionDepth,
	const int w, const int h,
	float3* shades)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;

	int hitIndex = (int)hitinfos[arrayPos1].objHit;

	shades[arrayPos1] = make_float3(hitIndex/10.0f,0.0f,0.0f);

	/*
	if(hitinfos[arrayPos1].hitDist > 1.0f)
	{
		shades[arrayPos1] = make_float3(1.0f,1.0f,0.0f);
	}
	else
	{
		shades[arrayPos1] = make_float3(1.0f,0.0f,0.0f);
	}
	*/

}


extern "C" float* rgbDTH(const float *devImg, const int w, const int h)
{
	cudaError_t cudaStatus;

	float* hostImg;

	hostImg = (float*)malloc(w * h * sizeof(gml::vec3_t));
	
	cudaStatus = cudaMemcpy(hostImg, devImg, w * h * sizeof(float3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	cudaFree((void*)devImg);
	return hostImg;

Error:

	printf("CUDA ERROR OCCURED\n");

	return NULL;
}


extern "C" RayTracing::HitInfo_t* findClosestHitsWithCuda(const RayTracing::HitInfo_t** hitInfos_array, const int w, const int h, const int m_nObjects)
{	

	RayTracing::HitInfo_t* closestHits = 0;
	RayTracing::HitInfo_t** devHitInfos_array = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc (( void **)& devHitInfos_array , m_nObjects * sizeof ( RayTracing::HitInfo_t* ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devHitInfos_array, hitInfos_array, m_nObjects * sizeof(RayTracing::HitInfo_t*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& closestHits , w * h * sizeof ( RayTracing::HitInfo_t ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching findClosestHits!\n", cudaStatus);
		goto Error;
	}

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);  
	
	findClosestHitsKernel <<<numBlocks, threadsPerBlock>>>( devHitInfos_array, w, h, closestHits, m_nObjects );


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findClosestHits!\n", cudaStatus);
		goto Error;
	}

	for(int i = 0 ; i < m_nObjects; i++)
	{
		cudaFree((void*)(hitInfos_array[i]));
		hitInfos_array[i] = 0;
	}

	cudaFree(devHitInfos_array);
	return closestHits;
	

Error:

	cudaFree(devHitInfos_array);
	printf("CUDA ERROR OCCURED\n");

	return NULL;
}

extern "C" float* shadeRaysWithCuda(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const RayTracing::Object_Kernel_t* objects,
	const float* lightPos,
	const int remainingRecursionDepth,
	const int w, const int h)
{
	cudaError_t cudaStatus;
	float3* devShades = 0;
	float4* devLightPos = 0;

	cudaStatus = cudaMalloc (( void **)& devLightPos , sizeof ( float4 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devLightPos, lightPos , sizeof(float4), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devShades , w * h * sizeof ( float3 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);
	
	shadeRaysKernel <<<numBlocks, threadsPerBlock>>>( rays,hitinfos, objects,devLightPos, remainingRecursionDepth,w,h,devShades);

	cudaFree((void*)rays);
	rays = 0;

	cudaFree((void*)hitinfos);
	hitinfos = 0;

	cudaFree((void*)objects);
	objects = 0;

	cudaFree((void*)devLightPos);
	objects = 0;	

	return (float*)devShades;
	
Error:

	printf("CUDA ERROR OCCURED\n");
	return NULL;
}

extern "C" RayTracing::Object_Kernel_t* objHTD(const RayTracing::Object_Kernel_t *hostObj, const int m_nObjects)
{
	cudaError_t cudaStatus;

	RayTracing::Object_Kernel_t* devObjs = 0;
	cudaStatus = cudaMalloc (( void **)& devObjs , m_nObjects * sizeof ( RayTracing::Object_Kernel_t ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devObjs, hostObj, m_nObjects * sizeof(RayTracing::Object_Kernel_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	return devObjs;

Error:
	printf("CUDA ERROR OCCURED\n");
	return NULL;
}