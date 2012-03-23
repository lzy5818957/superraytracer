#pragma once
#ifndef __INC_OBJECT_KERNEL_H_
#define __INC_OBJECT_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../RayTracing/types.h"

extern "C"  float* transformRayToObjSpaceWithCuda(float *devRays, const int w, const int h, float *m_worldToObject);
extern "C" float* hitPropertiesWithCudaObject(float* normTexObjSpc, float *m_objectToWorld_Normals, const RayTracing::HitInfo_t *hitinfos,  const int w, const int h);
#endif