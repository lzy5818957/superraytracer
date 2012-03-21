#ifndef __INC_CUDA_VECTOR_UTIL_H__
#define __INC_CUDA_VECTOR_UTIL_H__

#include "cuda_runtime.h"


__device__ void Mat4x4_Mul(float *A, float *B, float *C)
{
	C[0] = A[0]*B[0]+A[4]*B[1]+A[8]*B[2]+A[12]*B[3];
	C[1] = A[0]*B[4]+A[4]*B[5]+A[8]*B[6]+A[12]*B[7];
	C[2] = A[0]*B[8]+A[4]*B[9]+A[8]*B[10]+A[12]*B[11];
	C[3] = A[0]*B[12]+A[4]*B[13]+A[8]*B[14]+A[12]*B[15];

	C[4] = A[1]*B[0]+A[5]*B[1]+A[9]*B[2]+A[13]*B[3];
	C[5] = A[1]*B[4]+A[5]*B[5]+A[9]*B[6]+A[13]*B[7];
	C[6] = A[1]*B[8]+A[5]*B[9]+A[9]*B[10]+A[13]*B[11];
	C[7] = A[1]*B[12]+A[5]*B[13]+A[9]*B[14]+A[13]*B[15];

	C[8] = A[2]*B[0]+A[6]*B[1]+A[10]*B[2]+A[14]*B[3];
	C[9] = A[2]*B[4]+A[6]*B[5]+A[10]*B[6]+A[14]*B[7];
	C[10] = A[2]*B[8]+A[6]*B[9]+A[10]*B[10]+A[14]*B[11];
	C[11] = A[2]*B[12]+A[6]*B[13]+A[10]*B[14]+A[14]*B[15];

	C[12] = A[3]*B[0]+A[7]*B[1]+A[11]*B[2]+A[15]*B[3];
	C[13] = A[3]*B[4]+A[7]*B[5]+A[11]*B[6]+A[15]*B[7];
	C[14] = A[3]*B[8]+A[7]*B[9]+A[11]*B[10]+A[15]*B[11];
	C[15] = A[3]*B[12]+A[7]*B[13]+A[11]*B[14]+A[15]*B[15];
}

__device__ void Mat4x4_Mul_Vec4(float *A, float *B, float *C)
{
	C[0] = A[0]*B[0]+A[4]*B[1]+A[8]*B[2]+A[12]*B[3]; 
	C[1] = A[1]*B[0]+A[5]*B[1]+A[9]*B[2]+A[13]*B[3];
	C[2] = A[2]*B[0]+A[6]*B[1]+A[10]*B[2]+A[14]*B[3];
	C[3] = A[3]*B[0]+A[7]*B[1]+A[11]*B[2]+A[15]*B[3];
}
__device__ void Vec3_Nrm(float *A, float *B)
{
	B[0] = A[0]/sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]);
	B[1] = A[1]/sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]);
	B[2] = A[2]/sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]);
}

__device__ void Vec3_Dot(float *A, float *B, float *C)
{
	C[0] = A[0]*B[0]+A[1]*B[1]+A[2]*B[2];
}

__device__ void Vec3_Cross(float *A, float *B, float *C)
{
	C[0] = A[1]*B[2] - A[2]*B[1];
	C[1] = A[2]*B[0] - A[0]*B[2];
	C[2] = A[0]*B[1] - A[1]*B[0];
}


#endif