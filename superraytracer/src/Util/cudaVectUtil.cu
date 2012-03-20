
#include "cudaVectUtil.cuh"
/*
__device__ float* vector3ToArray(gml::vec3_t vec)
{
	float array[3];

	array[0] = vec.x;  
	array[1] = vec.y;
	array[2] = vec.z;
	
	return array;
}

__device__ gml::vec3_t arrayToVector3(float* array)
{
	gml::vec3_t vec;
	
	vec.x = array[0];
	vec.y = array[1];
	vec.z = array[2];

	return vec;
}

__device__ float* matrix4x4ToArray(gml::mat4x4_t mat)
{
	float array[16];
	int counter = 0;

	for(int i=0; i<4; i++)
	{
		for(int j=0; j<4 ;j++)
		{
			mat[i][j] = array[counter];
			counter++;
		}
	}

	return array;
}

__device__ gml::mat4x4_t arrayToMatrix4x4(float* array)
{
	gml::mat4x4_t mat;
	int counter = 0;

	for(int i=0; i<4; i++)
	{
		for(int j=0; j<4; j++)
		{
			mat[i][j] = array[counter];
			counter++;
		}
	}

	return mat;
}
*/