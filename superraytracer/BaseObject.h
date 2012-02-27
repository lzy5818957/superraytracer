#pragma once
#ifndef BASE_OBJECT_H
#define BASE_OBJECT_H

#include "Ray.h"
#include "Color.h"
#include "Surface.h"

class Color;
class Ray;

class BaseObject
{
public:
	BaseObject(void);
	~BaseObject(void);
	/*
	virtual double DistanceToIntersection(const Ray& _ray) const = 0;
	virtual Color ShadePoint(Ray& _eye, const gml::vec3_t& point) const = 0;
	virtual gml::vec3_t FindNormal(const gml::vec3_t& _pt) const = 0;
	inline void AttachSurface(Surface* _surf)
	{	
		m_surface = _surf;
	}
	inline void SetOwnerWorld(World* _world)
	{
		m_ownerWorld = _world;
	}
	*/
};

#endif