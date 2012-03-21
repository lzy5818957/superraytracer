#pragma once
#ifndef __INC_VECTOR_UTILITY_H_
#define __INC_VECTOR_UTILITY_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void vectorDot(float* a, float* b, float* c);
__device__ void vectorMul(float* a, float* b, float* c);
__device__ void mat4x4Mul(float* a, float* b, float* c);

#endif