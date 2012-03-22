#pragma once
#ifndef __INC_PLANE_KERNEL_H_
#define __INC_PLANE_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../RayTracing/types.h"

extern "C" cudaError_t raysIntersectsWithCudaSphere(float *devRays, const float t0, const float t1, RayTracing::HitInfo_t *hitinfos)

#endif
