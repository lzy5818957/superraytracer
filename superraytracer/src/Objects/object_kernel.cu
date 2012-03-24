#include "object_kernel.cuh"
#include "curand_kernel.h"
#include "cublas_v2.h"
#include <cutil_math.h>

#include <cstdio>

#define BLOCK_SIZE 8

__device__ void Mat4x4_Mul_Vec4_obj(float *A, float *B, float *C)
{
	C[0] = A[0]*B[0]+A[4]*B[1]+A[8]*B[2]+A[12]*B[3]; 
	C[1] = A[1]*B[0]+A[5]*B[1]+A[9]*B[2]+A[13]*B[3];
	C[2] = A[2]*B[0]+A[6]*B[1]+A[10]*B[2]+A[14]*B[3];
	C[3] = A[3]*B[0]+A[7]*B[1]+A[11]*B[2]+A[15]*B[3];
}

__global__ void hitPropertiesObjectKernel(float* normTexObjSpc, float* devM_objectToWorld_Normals,const RayTracing::HitInfo_t *hitinfos,  const int w, const int h , float *normTex)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos5 = 5 * (c + w * r);

	float4 normalObjectSpc = make_float4(normTex[arrayPos5],normTex[arrayPos5+1],normTex[arrayPos5+2],0.0f);

	float4 normalWorldSpc;

	Mat4x4_Mul_Vec4_obj(devM_objectToWorld_Normals,(float*)&normalObjectSpc,(float*)&normalWorldSpc);

	normTex[arrayPos5] = normalWorldSpc.x;
	normTex[arrayPos5 + 1] = normalWorldSpc.y;
	normTex[arrayPos5 + 2] = normalWorldSpc.z;

}


__global__ void tfRayWdToObj(float3 *rays, float4 *m_worldToObject, int w, float3 *raysInObj)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;

	int arraypos2 = 2*(c + w*r);

	float4 ray_wd = make_float4(rays[arraypos2], 1.0f);
	float4 ray_obj;
	Mat4x4_Mul_Vec4_obj((float*)m_worldToObject, (float *)(&ray_wd), (float *)(&ray_obj) );
	raysInObj[arraypos2] = make_float3(ray_obj);

	ray_wd = make_float4(rays[arraypos2+1], 0.0f);
	Mat4x4_Mul_Vec4_obj((float*)m_worldToObject, (float *)(&ray_wd), (float *)(&ray_obj) );
	raysInObj[arraypos2+1] = make_float3(ray_obj);

}

extern "C"  float* transformRayToObjSpaceWithCuda(float *rays, const int w, const int h, float *m_worldToObject)
{
	float3 *devRaysObj = 0;
	float4 *dev_wdToObj = 0;
	
	curandState * devStates;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devRaysObj , 2 * w * h * sizeof ( float3 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
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

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y); 

	tfRayWdToObj <<<numBlocks,threadsPerBlock>>> ((float3*)rays, dev_wdToObj, w,devRaysObj );
	cudaFree(dev_wdToObj);
	dev_wdToObj = 0;

	return (float*)devRaysObj;
Error:
	cudaFree(dev_wdToObj);
	cudaFree(devRaysObj);
	return NULL;
}

extern "C" float* hitPropertiesWithCudaObject(float* normTexObjSpc, float *m_objectToWorld_Normals, const RayTracing::HitInfo_t *hitinfos,  const int w, const int h)
{
	float *devNormTex = 0;
	float *devM_objectToWorld_Normals;

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc (( void **)& devNormTex , 5 * w * h * sizeof ( float ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devM_objectToWorld_Normals , 16 * sizeof ( float ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devM_objectToWorld_Normals, m_objectToWorld_Normals, 16 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
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

	hitPropertiesObjectKernel <<<numBlocks, threadsPerBlock>>>(normTexObjSpc,devM_objectToWorld_Normals, hitinfos, w, h,devNormTex );


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree(normTexObjSpc);
	normTexObjSpc = 0;

	return devNormTex;

Error:

	cudaFree(devNormTex);
	return NULL;
}