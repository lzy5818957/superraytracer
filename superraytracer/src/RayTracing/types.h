
/*
* Copyright:
* Daniel D. Neilson (ddneilson@ieee.org)
* University of Saskatchewan
* All rights reserved
*
* Permission granted to use for use in assignments and
* projects for CMPT 485 & CMPT 829 at the University
* of Saskatchewan.
*/

#pragma once
#ifndef _INC_RAYTRACING_TYPES_H_
#define _INC_RAYTRACING_TYPES_H_

#include "../GML/gml.h"
#include "../GL3/gl3.h"

// Forward declaration
namespace Object
{
	class Object;
}

namespace RayTracing
{
	typedef enum
	{
		PLANE,
		OCTAHEDRON,
		SPHERE
	} GeometryType_Kernel;

	typedef enum
	{
		SIMPLE,
		GOURAUD,
		PHONG,
		MIRROR
	} ShaderType_Kernel;

	typedef struct {

		float m_surfRefl[3]; // default: pure white

		bool m_hasSpecular; // default: false
		float m_specExp; // specular exponent
		// specular reflectance
		float m_specRefl[3]; // default: pure white

		ShaderType_Kernel m_shadeType; // default: GOURAUD

	} Material_Kernel_t;

	typedef struct {
		// TODO!
		// Replace placeholder with whatever information you believe to
		// be necessary to cache.
		GeometryType_Kernel m_geometry_type;

		// Surface material
		Material_Kernel_t m_material;

		// object <-> world space transformations
		float m_objectToWorld[16];
		float m_objectToWorld_Normals[16]; // Transforming normals
		float m_worldToObject[16];

	} Object_Kernel_t;

	// Basic ray type: r(t) = o + td
	typedef struct _Ray_t{
		gml::vec3_t o; // Ray origin.
		gml::vec3_t d; // Ray direction.
		_Ray_t() { o = gml::vec3_t(0.0,0.0,0.0); d = gml::vec3_t(0.0, 0.0, 0.0); }
		// Generate a random ray uniformly sampled from hemisphere
		// centered on n
		void randomDirection(const gml::vec3_t &n);
	} Ray_t;


	// Hit/intersection information caching types
	typedef struct {
		// TODO!
		// Replace placeholder with whatever information you believe to
		// be necessary to cache.
		float shadePoint_x;
		float shadePoint_y;
		float shadePoint_z;
	} SphereHitInfo_t;

	typedef struct {
		float u, v;
	} PlaneHitInfo_t;

	typedef struct {
		// Vertex indices for the triangle hit
		GLuint i0, i1, i2;
		// Barycentric coordinates of hitpoint
		float u, v;
	} MeshHitInfo_t;

	typedef struct _HitInfo_t {
		// Object intersected
		const Object::Object *objHit;
		// Distance along ray to the intersection point
		float hitDist;

		// Cached data from rayintersection calculation that can be
		// used for figuring out surface properties (ex: normal, texcoords, etc)
		union {
			SphereHitInfo_t sphere;
			PlaneHitInfo_t plane;
			MeshHitInfo_t mesh;
		};

		_HitInfo_t() {}
	} HitInfo_t;

}

#endif
