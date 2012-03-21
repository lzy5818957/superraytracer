#include "vectorUtil_kernel.cuh"

__device__ void vectorDot(float* a, float* b, float* c)
{
	c[0] = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

__device__ void vectorMul(float* a, float* b, float* c)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ void mat4x4Mul(float* a, float* b, float* c)
{
	a[0] = 
}