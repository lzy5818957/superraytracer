#pragma once
#ifndef __INC_OBJECT_KERNEL_H_
#define __INC_OBJECT_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" cudaError_t transformRayToObjSpaceWithCuda(float *hostRayDirs,float *hostRayOrig, const int w, const int h, float *m_worldToObject);

#endif