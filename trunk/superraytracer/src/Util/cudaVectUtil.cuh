#pragma once
#ifndef __INC_CUDA_VECTOR_UTIL_H__
#define __INC_CUDA_VECTOR_UTIL_H__

#include "../GML/gml.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float* Mat4x4_Mul_Mat4x4(float *A, float *B, float *c);
__device__ float* Mat4x4_Mul_Vec4(float *A, float *B, float *c);


#endif