
#pragma once
#ifndef RAYTRACER_RAY_H
#define RAYTRACER_RAY_H

#include "Utils/GML/gml.h"

class Ray
{
public:
	Ray(void);
	Ray(gml::vec3_t _pos, gml::vec3_t _dir);
	Ray(const Ray& ray);
	~Ray(void);

	Ray Reflect(gml::vec3_t _normal, gml::vec3_t _pt) const;
	gml::vec3_t Extrapolate(double _t) const;
	double Interpolate(gml::vec3_t _point) const;
	void Normalize();
	const gml::vec3_t& Pos() const;
	const gml::vec3_t& Dir() const;
	inline void Age() { m_depth++; }
	inline void SetAge(int _age) { m_depth = _age; }
	inline int CurrentAge() { return m_depth; }

	private:
	gml::vec3_t m_pos, m_dir;
	int m_depth;
};

#endif
