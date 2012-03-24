#pragma once
#ifndef __INC_OBJECT_KERNEL_H_
#define __INC_OBJECT_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../RayTracing/types.h"

typedef enum
{
	SIMPLE,
	GOURAUD,
	PHONG,
	MIRROR
} ShaderType_Kernel;

typedef struct {

	gml::vec3_t m_surfRefl; // default: pure white

	bool m_hasSpecular; // default: false
	float m_specExp; // specular exponent
	// specular reflectance
	gml::vec3_t m_specRefl; // default: pure white

	ShaderType_Kernel m_shadeType; // default: GOURAUD

} Material_Kernel_t;

typedef struct {
	// TODO!
	// Replace placeholder with whatever information you believe to
	// be necessary to cache.
	int m_geometry_type;

	// Surface material
	Material_Kernel_t m_material;

	// object <-> world space transformations
	gml::mat4x4_t m_objectToWorld;
	gml::mat4x4_t m_objectToWorld_Normals; // Transforming normals
	gml::mat4x4_t m_worldToObject;

} Object_Kernel_t;

extern "C"  float* transformRayToObjSpaceWithCuda(float *devRays, const int w, const int h, float *m_worldToObject);
extern "C" float* hitPropertiesWithCudaObject(float* normTexObjSpc, float *m_objectToWorld_Normals, const RayTracing::HitInfo_t *hitinfos,  const int w, const int h);
#endif

