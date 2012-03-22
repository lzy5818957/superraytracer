#pragma once
#ifndef __INC_MESH_KERNEL_H_
#define __INC_MESH_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../GL3/gl3.h"
#include "../RayTracing/types.h"

extern "C" cudaError_t raysIntersectsWithCudaMesh(float3 *m_vertPositions,GLuint i0, GLuint i1, GLuint i2,float *devRays, const float t0, const float t1, const int w, const int h, RayTracing::HitInfo_t *hostHitInfos);

#endif