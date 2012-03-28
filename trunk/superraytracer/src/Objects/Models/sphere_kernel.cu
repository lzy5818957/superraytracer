#include <cutil_math.h>
#include "sphere_kernel.cuh"
#include <cstdio>
#include <cfloat>


#define BLOCK_SIZE 8

__device__ void Mat4x4_Mul_Vec4_Sphere(const float *A, float *B, float *C)
{
	C[0] = A[0]*B[0]+A[4]*B[1]+A[8]*B[2]+A[12]*B[3]; 
	C[1] = A[1]*B[0]+A[5]*B[1]+A[9]*B[2]+A[13]*B[3];
	C[2] = A[2]*B[0]+A[6]*B[1]+A[10]*B[2]+A[14]*B[3];
	C[3] = A[3]*B[0]+A[7]*B[1]+A[11]*B[2]+A[15]*B[3];
}

__global__ void raysIntersectsSphereKernel(float *devRays, const float t0, const float t1,const int w, const int h,RayTracing::HitInfo_t *hitInfos, int objHitIndex)
{
	float A,B,C;
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

	A = dot(ray_d,ray_d);
	B = dot(ray_d,ray_o);
	C = dot(ray_o,ray_o) - 1.0f;

	float det = B*B - A*C;

	if(det < 0.0)
	{
		hitInfos[arrayPos1].hitDist = FLT_MAX;
		return;
	}else
	{
		
		float t =  (-B - sqrt(B * B - A * C)) / A;
	    if(t > t1 || t < t0 )
	    {

	       hitInfos[arrayPos1].hitDist = FLT_MAX;
		   return;
	    }
	    else
	    {
 	       hitInfos[arrayPos1].hitDist = t;
		   float3 shadePoint = ray_o + (t * ray_d);
		   hitInfos[arrayPos1].sphere.shadePoint_x = shadePoint.x;
		   hitInfos[arrayPos1].sphere.shadePoint_y = shadePoint.y;
		   hitInfos[arrayPos1].sphere.shadePoint_z = shadePoint.z;
		   hitInfos[arrayPos1].objHit = (Object::Object*)objHitIndex;
		   return;
	    }
		


	}
}

__global__ void shadowRaysSphereKernel(const float *devRays, const RayTracing::HitInfo_t *hitinfos,const RayTracing::Object_Kernel_t *objects, const float* lightProp, const int w, const int h, bool* isInShadow)
{

	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;
	int arrayPos6 = 6 * (c + w * r);


	float lightPos[3] = {lightProp[0], lightProp[1], lightProp[2]};

	float3 ray_o;
	ray_o.x = devRays[arrayPos6];
	ray_o.y = devRays[arrayPos6 +1];
	ray_o.z = devRays[arrayPos6 +2];

	float3 ray_d;
	ray_d.x = devRays[arrayPos6 +3];
	ray_d.y = devRays[arrayPos6 +4];
	ray_d.z = devRays[arrayPos6 +5];

	float A,B,C;
	A = dot(ray_d,ray_d);
	B = dot(ray_d,ray_o);
	C = dot(ray_o,ray_o) - 1.0f;

	float det = B*B - A*C;

	RayTracing::HitInfo_t hitInfo = hitinfos[arrayPos1];
	int index = (int)hitInfo.objHit;

	float lightPosObj[3];
	RayTracing::Object_Kernel_t object = objects[index];
	Mat4x4_Mul_Vec4_Sphere(object.m_worldToObject,lightPos,lightPosObj);
	float3 lightPosObjFloat3 = make_float3(lightPosObj[0],lightPosObj[1],lightPosObj[2]);

	float t0 = 0.0001f;
	float t1 = length(lightPosObjFloat3 - make_float3(hitInfo.sphere.shadePoint_x, hitInfo.sphere.shadePoint_y, hitInfo.sphere.shadePoint_z));

	if(det < 0.0)
	{
		isInShadow[arrayPos1] = false;
		return;
	}else
	{
		
		float t =  (-B - sqrt(B * B - A * C)) / A;
	    if(t > t1 || t < t0 )
	    {

			isInShadow[arrayPos1] = false;
			return;
	    }
	    else
	    {
 			isInShadow[arrayPos1] = true;
			return;

	    }
		


	}
	

}

__global__ void hitPropertiesSphereKernel(const RayTracing::HitInfo_t *hitinfos,  const int w, const int h,float *normTex)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = (c + w * r);
	int arrayPos5 = 5 * (c + w * r);

	float3 shadePoint;
	float3 normal;
	float2 texCoords;

	shadePoint.x = hitinfos[arrayPos1].sphere.shadePoint_x;
	shadePoint.y = hitinfos[arrayPos1].sphere.shadePoint_y;
	shadePoint.z = hitinfos[arrayPos1].sphere.shadePoint_z;

	normal = hitinfos[arrayPos1].hitDist * shadePoint;
	texCoords.x = (atan2 ( shadePoint.z, - shadePoint.x) / 3.14159265358979323846 +1 )/ 2.0f;
	texCoords.y = ( asin ( -shadePoint.y ) / 3.14159265358979323846 +1)/ 2;

	normTex [arrayPos5] = normal.x;
	normTex [arrayPos5 + 1] = normal.y;
	normTex [arrayPos5 + 2] = normal.z;
	normTex [arrayPos5 + 3] = texCoords.x;
	normTex [arrayPos5 + 4] = texCoords.y;
}

extern "C" RayTracing::HitInfo_t* raysIntersectsWithCudaSphere(float *devRays, const float t0, const float t1,const int w, const int h,  int objHitIndex)
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

	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 numBlocks(w/threadsPerBlock.x,h/threadsPerBlock.y);

	raysIntersectsSphereKernel <<< numBlocks, threadsPerBlock>>> (devRays,t0,t1,w,h,devHitInfos, objHitIndex);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree(devRays);
	devRays = 0;

	return devHitInfos;
	

Error:
	cudaFree(devHitInfos);
	devHitInfos = 0;
	return NULL;

}

extern "C" float* hitPropertiesWithCudaSphere(const RayTracing::HitInfo_t*hitinfos, const int w, const int h)
{
	float *devNormTex = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void **)& devNormTex , 5 * w * h * sizeof (float));
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"cudaMalloc failed! ");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 numBlocks(w/threadsPerBlock.x,h/threadsPerBlock.y);

	hitPropertiesSphereKernel<<<numBlocks,threadsPerBlock>>>(hitinfos,w,h,devNormTex);

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

extern "C" bool* shadowRaysWithCudaSphere(const RayTracing::Ray_t *rays, const RayTracing::HitInfo_t *hitinfos,const RayTracing::Object_Kernel_t *objects, const float* lightProp, const int w, const int h)
{

	float *devVert0 = 0;
	bool *devIsInShadow = 0;
	cudaError_t cudaStatus;
	
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

	shadowRaysSphereKernel <<<numBlocks, threadsPerBlock>>>((float*)rays, hitinfos, objects, lightProp, w, h, devIsInShadow);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching setup_rand_kernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree((void*)rays);
	rays = 0;


	return devIsInShadow;

Error:
	cudaFree((void*)rays);
	rays = 0;

	cudaFree(devIsInShadow);
	devIsInShadow = 0;

	return NULL;


}