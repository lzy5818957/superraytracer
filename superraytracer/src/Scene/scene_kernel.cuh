#pragma once
#ifndef __INC_OBJECT_KERNEL_H_
#define __INC_OBJECT_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../RayTracing/types.h"

extern "C" float* lightPropHTD(	const float* lightPos,const float* lightRad, const int w, const int h);
extern "C" float* rgbDTH(const float *devImg, const int w, const int h);
extern "C" RayTracing::Object_Kernel_t* objHTD(const RayTracing::Object_Kernel_t *hostObj, const int m_nObjects);
extern "C" RayTracing::HitInfo_t* findClosestHitsWithCuda(const RayTracing::HitInfo_t** hitInfos_array, const int w, const int h, const int m_nObjects);
extern "C" bool* mergeShadowWithCuda(RayTracing::Ray_t *shadowRays, const bool** hitInfos_array, const int w, const int h, const int m_nObjects);
extern "C" RayTracing::Ray_t* raysDTH(const RayTracing::Ray_t *rays, const int w, const int h);
extern "C" float* shadeRaysDirectLightWithCuda
	(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const RayTracing::Object_Kernel_t* objects,
	const float* lightProp,
	const int remainingRecursionDepth,
	const int w, const int h
	);

extern "C" float* shadeRaysShadowLightWithCuda
	(
	const bool *isInShadow,
	const int w, const int h,
	float* colors
	);

extern "C" RayTracing::Ray_t* genShadowRaysWithCuda
	(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const float *lightProp,
	const int w, const int h
	);

extern "C" void cleanUp(void** handles, int nHandles);

#endif