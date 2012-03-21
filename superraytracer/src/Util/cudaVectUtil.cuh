#pragma once
#ifndef __INC_CUDA_VECTOR_UTIL_H__
#define __INC_CUDA_VECTOR_UTIL_H__

#include "../GML/gml.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void Mat4x4_Mul(float *A, float *B, float *C);
__device__ void Mat4x4_Mul_Vec4(float *A, float *B, float *C);
__device__ void Vec3_Nrm(float *A, float *B);
__device__ void Vec3_Dot(float *A, float *B, float *C);
__device__ void Vec3_Cross(float *A, float *B, float *C);

#endif