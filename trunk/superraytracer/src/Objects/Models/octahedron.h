
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

/*
* Geometry for an octahedron
*  - An 8-sided regular solid.
*
* This octahedron has flat faces.
* It is also centered at (0,0,0).
* The faces face 45 degree angles relative to the xz plane
*/

#pragma once
#ifndef __INC_OCTAHEDRON_H_
#define __INC_OCTAHEDRON_H_

#include "../geometry.h"
#include "../mesh.h"

namespace Object
{
	namespace Models
	{

		class Octahedron : public Geometry
		{
		protected:
			Mesh m_mesh;
		public:
			Octahedron();
			~Octahedron();

			bool init();

			virtual void rasterize() const;

			virtual bool rayIntersects(const RayTracing::Ray_t &ray, const float t0, const float t1, RayTracing::HitInfo_t &hitinfo) const;
			virtual bool shadowsRay(const RayTracing::Ray_t &ray, const float t0, const float t1) const;
			virtual void hitProperties(const RayTracing::HitInfo_t &hitinfo, gml::vec3_t &normal, gml::vec2_t &texCoords) const;

			virtual RayTracing::HitInfo_t* rayIntersectsInParallel(const RayTracing::Ray_t *rays, const float t0, const float t1, const int w, const int h, int objHitindex) const;
			virtual float* hitPropertiesInParallel(const RayTracing::HitInfo_t *hitinfos,  const int w, const int h) const;
			virtual RayTracing::GeometryType_Kernel getGeometryType() const;
			virtual bool* shadowRaysInParallel(const RayTracing::Ray_t *rays, const RayTracing::HitInfo_t *hitinfos, const RayTracing::Object_Kernel_t *objects, const float* lightProp, const int w, const int h) const;
		};

	}
}

#endif
