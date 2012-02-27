#include "Ray.h"


Ray::Ray(void)
{
	m_pos = gml::vec3_t(0.0, 0.0, 0.0);
	m_dir = gml::vec3_t(0.0, 0.0, 0.0);
	m_depth = 0;
}

Ray::Ray(gml::vec3_t _pos, gml::vec3_t _dir)
{
	m_pos = _pos;
	m_dir = _dir;
	m_depth = 0;
}


Ray::~Ray(void)
{
	
}

Ray::Ray(const Ray& _ray)
{
	m_pos = _ray.m_pos;
	m_dir = _ray.m_dir;
	m_depth = _ray.m_depth;
}

Ray Ray::Reflect(gml::vec3_t _normal, gml::vec3_t _pt) const
{
	gml::vec3_t opposite_m_dir = gml::vec3_t(-m_dir.x, -m_dir.y, -m_dir.z);
	gml::vec3_t newDir =  gml::add(gml::scale( 2 * gml::dot( _normal, opposite_m_dir ),_normal),  m_dir);
	Ray result(_pt, newDir);
	result.SetAge( m_depth + 1 );
	return result;
}
gml::vec3_t Ray::Extrapolate(double _t) const
{
	return gml::add(m_pos, gml::scale(_t ,m_dir));
}
double Ray::Interpolate(gml::vec3_t _point) const
{
	for (int i = 0; i < 3; i ++)
	{
	// must check for divide by zero
		if (fabs(m_dir[i]) > 0.000001 )
		{
			return (_point[i] - m_pos[i]) / m_dir[i];
		}
	}
	return 0;
}
void Ray::Normalize()
{
	gml::normalize(m_dir);
}
const gml::vec3_t& Ray::Pos() const
{
	return m_pos;
}
const gml::vec3_t& Ray::Dir() const
{
	return m_dir;
}