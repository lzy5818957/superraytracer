
#pragma once
#ifndef __INC_CAMERA_KERNEL_H_
#define __INC_CAMERA_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../RayTracing/types.h"

extern "C" cudaError_t genViewRayWithCuda(RayTracing::Ray_t *host_rays, const int w, const int h);

#endif