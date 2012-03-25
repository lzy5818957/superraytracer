#pragma once
#ifndef __INC_OBJECT_KERNEL_H_
#define __INC_OBJECT_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../RayTracing/types.h"

extern "C" float* rgbDTH(const float *devImg, const int w, const int h);
extern "C" RayTracing::Object_Kernel_t* objHTD(const RayTracing::Object_Kernel_t *hostObj, const int m_nObjects);
extern "C" RayTracing::HitInfo_t* findClosestHitsWithCuda(const RayTracing::HitInfo_t** hitInfos_array, const int w, const int h, const int m_nObjects);
extern "C" float* shadeRaysWithCuda
	(
	const RayTracing::Ray_t *rays,
	const RayTracing::HitInfo_t *hitinfos,
	const RayTracing::Object_Kernel_t* objects,
	const float* lightPos,
	const float* lightRad,
	const int remainingRecursionDepth,
	const int w, const int h
	);

#endif