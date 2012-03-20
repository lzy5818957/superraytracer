#pragma once
#ifndef __INC_CUDA_VECTOR_UTIL_H__
#define __INC_CUDA_VECTOR_UTIL_H__

#include "../GML/gml.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float* vector3ToArray(gml::vec3_t vec);

__device__ gml::vec3_t arrayToVector3(float* array);

__device__ float* matrix4x4ToArray(gml::mat4x4_t mat);

__device__ gml::mat4x4_t arrayToMatrix4x4(float* array);

#endif