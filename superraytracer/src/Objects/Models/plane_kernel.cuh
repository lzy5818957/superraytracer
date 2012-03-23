#pragma once
#ifndef __INC_PLANE_KERNEL_H_
#define __INC_PLANE_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../RayTracing/types.h"

extern "C" RayTracing::HitInfo_t* raysIntersectsWithCudaPlane(float *devRays, const float t0, const float t1, const int w, const int h, float* hostVerts, void *objHit);
extern "C" float* hitPropertiesWithCudaPlane(const RayTracing::HitInfo_t*hitinfos, const int w, const int h);
#endif