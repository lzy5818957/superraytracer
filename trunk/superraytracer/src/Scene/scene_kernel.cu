#include <cutil_math.h>

#include <cstdio>
#include <cstdlib>
#include <cfloat>

#include "scene_kernel.cuh"

#define BLOCK_SIZE 8

__device__ void Mat4x4_Mul_Vec4_Scene(float *A, float *B, float *C)
{
	C[0] = A[0]*B[0]+A[4]*B[1]+A[8]*B[2]+A[12]*B[3]; 
	C[1] = A[1]*B[0]+A[5]*B[1]+A[9]*B[2]+A[13]*B[3];
	C[2] = A[2]*B[0]+A[6]*B[1]+A[10]*B[2]+A[14]*B[3];
	C[3] = A[3]*B[0]+A[7]*B[1]+A[11]*B[2]+A[15]*B[3];
}


__global__ void lightPropHTDKernel(const float* lightPos, const float* lightRad, float* lightProp)
{

	lightProp[0] = lightPos[0];
	lightProp[1] = lightPos[1];
	lightProp[2] = lightPos[2];

	lightProp[3] = lightRad[0];
	lightProp[4] = lightRad[1];
	lightProp[5] = lightRad[2];
}

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

__global__ void mergeShadowKernel(bool** isInShadow_array, const int w, const int h, bool* shadow, const int m_nObjects)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;

	for(int i = 1 ; i < m_nObjects; i++ )
	{
		if(isInShadow_array[i][arrayPos1])
		{
			shadow[arrayPos1] = true;
			return;
		}

	}

}

__global__ void shadeRaysShadowLightKernel(
	const bool *isInShadow,
	const int w, const int h,
	float3* shades)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;

	if(isInShadow[arrayPos1])
	{

		shades[arrayPos1] = make_float3(0.0f,0.0f,0.0f);
	}

	
}

__global__ void genShadowRaysKernel(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const float* lightProp,
	const int w, const int h,
	float *shadowRays)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;
	int arrayPos6 = 6*(c + w * r);

	
	RayTracing::HitInfo_t hitInfo = hitinfos[arrayPos1];
	float3 rayOri = make_float3(rays[arrayPos1].o.x, rays[arrayPos1].o.y, rays[arrayPos1].o.z);
	float3 rayDir = make_float3(rays[arrayPos1].d.x, rays[arrayPos1].d.y, rays[arrayPos1].d.z);
	float3 shadePoint = rayOri + (hitInfo.hitDist * rayDir);
	float3 m_lightPos3 = make_float3(lightProp[0],lightProp[1],lightProp[2] );
	float3 shadowRayDir = normalize(m_lightPos3 - shadePoint);
	
	
	shadowRays[arrayPos6] = shadePoint.x;
	shadowRays[arrayPos6 + 1] = shadePoint.y;
	shadowRays[arrayPos6 + 2] = shadePoint.z;

	shadowRays[arrayPos6 + 3] = shadowRayDir.x;
	shadowRays[arrayPos6 + 4] = shadowRayDir.y;
	shadowRays[arrayPos6 + 5] = shadowRayDir.z;
	
}

__device__ float3 shadeLambPhone(	
	float3 lightRad, // Light radiance
	float3 lightDir, // Direction from point to light
	float3 e, // View direction
	float3 p, // Point being shaded (world-space)
	float3 n,
	float3 surfRefl) // Normal of p (world-space)
{
	float diff = dot(lightDir, n);
	if (diff <= 0.0)
	{
		return make_float3(0.0f,0.0f,0.0f);
	}

	float3 lamb = diff * (lightRad * surfRefl);

	return lamb;

}

__global__ void shadeRaysDirectLightKernel(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const RayTracing::Object_Kernel_t* objects,
	const float* lightProp,
	const int remainingRecursionDepth,
	const int w, const int h,
	float3* shades)
{
	int c = (blockIdx.x * blockDim.x) + threadIdx.x;
	int r = (blockIdx.y * blockDim.y) + threadIdx.y;
	int arrayPos1 = c + w * r;

	if(c == 350 && r == 340)
	{
		int arrayPos1 = c + w * r;
	}

	RayTracing::HitInfo_t hitInfo = hitinfos[arrayPos1];

	float3 color;
	if(hitInfo.hitDist == FLT_MAX)
	{
		color = make_float3(0.0f,0.0f,0.0f);
	}
	else
	{
		RayTracing::Object_Kernel_t object = objects[(int)hitInfo.objHit];
		RayTracing::GeometryType_Kernel geoType = object.m_geometry_type;

		RayTracing::Material_Kernel_t mat = object.m_material;

		RayTracing::ShaderType_Kernel shadeType = mat.m_shadeType;
		bool hasSpecular = mat.m_hasSpecular;

		//shared shader data
		float3 shadePoint = (*(float3*)&rays[arrayPos1].o) + hitInfo.hitDist * (*(float3*)&rays[arrayPos1].d);
		float3 m_lightPos3 = make_float3(lightProp[0],lightProp[1],lightProp[2] );
		float3 lightRad = make_float3(lightProp[3],lightProp[4],lightProp[5] );
		float3 lightDir = normalize(m_lightPos3-shadePoint); 
		float3 viewDir = (*(float3*)&rays[arrayPos1].d);
		//shader data;
		float3 normal;

		//ShaderV
		switch(object.m_geometry_type)
		{
		case RayTracing::GeometryType_Kernel::PLANE:
			normal = make_float3(0.0f,1.0f,0.0f);
			break;
		case RayTracing::GeometryType_Kernel::SPHERE:
			float3 shadePointObj = make_float3(hitInfo.sphere.shadePoint_x, hitInfo.sphere.shadePoint_y, hitInfo.sphere.shadePoint_z);
			normal = (1/hitInfo.hitDist) * shadePointObj;
			break;
		case RayTracing::GeometryType_Kernel::OCTAHEDRON:
			break;
		default:
			break;
		}

		float4 normalWorld;
		Mat4x4_Mul_Vec4_Scene(object.m_objectToWorld_Normals, (float*)&make_float4(normal,1.0f), (float*)&normalWorld);
		normal = normalize( make_float3( normalWorld ) );

		if(hasSpecular)
		{
			//get surface normal 
			color = make_float3(1.0f,1.0f,1.0f);
		}
		else
		{
			//shade use lamb phone
			
			color = shadeLambPhone(lightRad,lightDir,viewDir,shadePoint,normal, *((float3*)&mat.m_surfRefl));
			
		}


	}

	shades[arrayPos1] = color;
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

extern "C" bool* mergeShadowWithCuda(RayTracing::Ray_t *shadowRays, const bool** isInShadow_array, const int w, const int h, const int m_nObjects)
{	

	bool* shadow = 0;
	bool** devIsInShadow_array = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc (( void **)& devIsInShadow_array , m_nObjects * sizeof (bool* ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devIsInShadow_array, isInShadow_array, m_nObjects * sizeof(bool*), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& shadow , w * h * sizeof ( bool ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemset (shadow ,0x00, w * h * sizeof ( bool ));
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

	mergeShadowKernel <<<numBlocks, threadsPerBlock>>>( devIsInShadow_array, w, h, shadow, m_nObjects );


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findClosestHits!\n", cudaStatus);
		goto Error;
	}

	for(int i = 0 ; i < m_nObjects; i++)
	{
		cudaFree((void*)(isInShadow_array[i]));
		isInShadow_array[i] = 0;
	}

	cudaFree(devIsInShadow_array);
	cudaFree((void*)shadowRays);

	return shadow;
	

Error:

	cudaFree(devIsInShadow_array);
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



extern "C" float* shadeRaysDirectLightWithCuda(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const RayTracing::Object_Kernel_t* objects,
	const float* lightProp,
	const int remainingRecursionDepth,
	const int w, const int h)
{
	cudaError_t cudaStatus;
	float3* devShades = 0;

	cudaStatus = cudaMalloc (( void **)& devShades , w * h * sizeof ( float3 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching shadeRaysKernel!\n", cudaStatus);
		goto Error;
	}


	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);

	shadeRaysDirectLightKernel <<<numBlocks, threadsPerBlock>>>( rays,hitinfos, objects, lightProp, remainingRecursionDepth,w,h,devShades);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching shadeRaysKernel!\n", cudaStatus);
		goto Error;
	}

	return (float*)devShades;

Error:

	printf("CUDA ERROR OCCURED\n");
	return NULL;
}

extern "C" float* shadeRaysShadowLightWithCuda
	(
	const bool *isInShadow,
	const int w, const int h,
	float* colors
	)
{
	cudaError_t cudaStatus;

	
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);

	shadeRaysShadowLightKernel <<<numBlocks, threadsPerBlock>>>( isInShadow,w,h,(float3*)colors);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching shadeRaysKernel!\n", cudaStatus);
		goto Error;
	}
	cudaFree((void*)isInShadow);
	isInShadow = 0;

	return colors;
Error:

	printf("CUDA ERROR OCCURED\n");
	return NULL;

}

extern "C" RayTracing::Ray_t* genShadowRaysWithCuda
	(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const float *lightProp,
	const int w, const int h
	)
{
	cudaError_t cudaStatus;

	float* shadowRays = 0;

	cudaStatus = cudaMalloc (( void **)& shadowRays , 2 * w * h * sizeof (float3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 64 threads 

	dim3 numBlocks(w/threadsPerBlock.x,  /* for instance 512/8 = 64*/ 
		h/threadsPerBlock.y);

	genShadowRaysKernel <<<numBlocks, threadsPerBlock>>>(rays, hitinfos, lightProp, w, h, shadowRays);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching genShadowRaysKernel!\n", cudaStatus);
		goto Error;
	}
	return ( RayTracing::Ray_t*)shadowRays;

Error:

	printf("CUDA ERROR OCCURED\n");
	return NULL;

}


extern "C" RayTracing::Ray_t* raysDTH(const RayTracing::Ray_t *rays, const int w, const int h)
{
	cudaError_t cudaStatus;

	RayTracing::Ray_t* hostRays = 0;
	hostRays = (RayTracing::Ray_t*)malloc( w * h * sizeof( RayTracing::Ray_t));


	cudaStatus = cudaMemcpy(hostRays, rays, w * h * sizeof( RayTracing::Ray_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	/*
	for(int i = 0; i < w * h; i++)
	{
		printf("Num = %d  %f &f %f %f %f %f\n",i, hostRays[i].o.x, hostRays[i].o.y, hostRays[i].o.z, hostRays[i].d.x, hostRays[i].d.y, hostRays[i].d.z);

	}
	*/
	return hostRays;

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

extern "C" float* lightPropHTD(	const float* lightPos,const float* lightRad, const int w, const int h)
{
	cudaError_t cudaStatus;
	float* devLightPos = 0;
	float* devLightRad = 0;
	float* devLightProp = 0;

	cudaStatus = cudaMalloc (( void **)& devLightProp ,2 * sizeof ( float3 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devLightPos ,sizeof ( float3 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(devLightPos, lightPos, sizeof(float3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc (( void **)& devLightRad ,sizeof ( float3 ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devLightRad, lightRad, sizeof(float3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	lightPropHTDKernel<<<1,1>>>(devLightPos,devLightRad,devLightProp);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching shadeRaysKernel!\n", cudaStatus);
		goto Error;
	}

	cudaFree(devLightPos);
	devLightPos = 0;

	cudaFree(devLightRad);
	devLightRad = 0;

	return (float*)devLightProp;

Error:
	printf("CUDA ERROR OCCURED\n");
	return NULL;
}

extern "C" void cleanUp(void** handles, int nHandles)
{
	for(int i = 0 ; i < nHandles; i++)
	{
		cudaFree(handles[i]);
		handles[i] = 0;
	}

}