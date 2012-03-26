
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

#include "scene.h"
#include "../glUtils.h"

#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cstdlib>

#include "scene_kernel.cuh"

namespace Scene
{

	// Number of Object::Object* 's to allocate at a time
	const GLuint N_PTRS = 20;

	Scene::Scene()
	{
		m_scene = 0;
		m_nObjects = 0;
		m_nObjPtrsAlloced = 0;
		// Position of a point light: (0,0,0)
		m_lightPos = gml::vec4_t(0.0f,0.0f,0.0f,1.0f);
		// Radiance of the point light
		m_lightRad = gml::vec3_t(0.6f,0.6f,0.6f);
		// Ambient radiance
		m_ambientRad = gml::vec3_t(0.025f, 0.025f, 0.025f);
	}

	Scene::~Scene()
	{
		if (m_nObjPtrsAlloced > 0)
		{
			for (GLuint i=0; i<m_nObjects; i++)
				delete m_scene[i];
			delete[] m_scene;
		}
	}

	bool Scene::init()
	{
		// Initialize the shader manager
		if ( !m_shaderManager.init() )
		{
			fprintf(stderr, "ERROR! Could not initialize Shader Manager.\n");
			return false;
		}
		return true;
	}

	bool Scene::addObject(Object::Object *obj)
	{
		if (m_nObjPtrsAlloced == 0)
		{
			m_nObjPtrsAlloced = N_PTRS;
			m_scene = new Object::Object*[m_nObjPtrsAlloced];
			if (m_scene == 0) return false;
		}
		else if (m_nObjPtrsAlloced == m_nObjects)
		{
			m_nObjPtrsAlloced += N_PTRS;
			Object::Object **temp = new Object::Object*[m_nObjPtrsAlloced];
			if (temp == 0) return false;
			memcpy(temp, m_scene, sizeof(Object::Object*)*m_nObjects);
			delete[] m_scene;
			m_scene = temp;
		}

		m_scene[m_nObjects++] = obj;
		return true;
	}

	void Scene::rasterizeDepth(const gml::mat4x4_t &worldView, const gml::mat4x4_t &projection)
	{
		const Shader::Shader *depthShader = m_shaderManager.getDepthShader();

		Shader::GLProgUniforms shaderUniforms;
		shaderUniforms.m_projection = projection;

		depthShader->bindGL(false);
		for (GLuint i=0; i<m_nObjects; i++)
		{
			shaderUniforms.m_modelView = gml::mul(worldView, m_scene[i]->getObjectToWorld());

			if ( !depthShader->setUniforms(shaderUniforms, false) ) return;

			m_scene[i]->rasterize();
			if ( isGLError() ) return;
		}
	}

	void Scene::rasterize(const gml::mat4x4_t &worldView, const gml::mat4x4_t &projection, const bool useShadows)
	{
		// Struct used to pass data values for GLSL uniform variables to
		// the shader program
		Shader::GLProgUniforms shaderUniforms;


		// Set up uniforms constant to the world
		shaderUniforms.m_lightPos = gml::extract3( gml::mul( worldView, m_lightPos ) );
		shaderUniforms.m_lightRad = m_lightRad;
		shaderUniforms.m_ambientRad = m_ambientRad;
		shaderUniforms.m_projection = projection;

		for (GLuint i=0; i<m_nObjects; i++)
		{
			// Fetch the Shader object from the ShaderManager that will perform the
			// shading calculations for this object
			const Shader::Shader *shader = m_shaderManager.getShader(m_scene[i]->getMaterial());

			if (shader->getIsReady(useShadows))
			{
				shader->bindGL(useShadows); // Bind the shader to the OpenGL context
				if (isGLError()) return;

				// Object-specific uniforms
				shaderUniforms.m_modelView = gml::mul(worldView, m_scene[i]->getObjectToWorld());
				shaderUniforms.m_normalTrans = gml::transpose( gml::inverse(shaderUniforms.m_modelView) );
				// If the surface material is not using a texture for Lambertian surface reflectance
				if (m_scene[i]->getMaterial().getLambSource() == Material::CONSTANT)
				{
					shaderUniforms.m_surfRefl = m_scene[i]->getMaterial().getSurfRefl();
				}
				else
				{
					m_scene[i]->getMaterial().getTexture()->bindGL(GL_TEXTURE0); // Set up texture
				}
				// Set up the specular components of the uniforms struct if the material
				// is specular
				if (m_scene[i]->getMaterial().hasSpecular())
				{
					shaderUniforms.m_specExp = m_scene[i]->getMaterial().getSpecExp();
					shaderUniforms.m_specRefl = m_scene[i]->getMaterial().getSpecRefl();
				}

				// Set the shader uniform variables
				if ( !shader->setUniforms(shaderUniforms, useShadows) || isGLError() ) return;

				// Rasterize the object
				m_scene[i]->rasterize();
				if (isGLError()) return;

				// Unbind the shader from the OpenGL context
				shader->unbindGL();
			}
		}
	}

	bool Scene::rayIntersects(const RayTracing::Ray_t &ray, const float t0, const float t1, RayTracing::HitInfo_t &hitinfo) const
	{
		// TODO
		//   Find the closest intersection of the ray in the distance range [t0,t1].
		// Return true if an intersection was found, false otherwise

		for(GLuint i = 0; i < m_nObjects; i++)
		{
			RayTracing::HitInfo_t hitInfo_indi;
			if(m_scene[i]->rayIntersects(ray,t0,t1,hitInfo_indi))
			{
				if(hitInfo_indi.hitDist < hitinfo.hitDist)
				{

					hitinfo = hitInfo_indi;
				}
			}

		}

		if(hitinfo.hitDist != FLT_MAX)
		{

			return true;

		}

		return false;
	}

	bool Scene::shadowsRay(const RayTracing::Ray_t &ray, const float t0, const float t1) const
	{
		// TODO
		//  Determine whether or not the ray intersects an object in the distance range [t0,t1].
		//  Note: Just need to know whether it intersects _an_ object, not the nearest.

		// Return true if the ray intersects an object, false otherwise
		for(GLuint i = 0; i < m_nObjects; i++)
		{
			if(m_scene[i]->shadowsRay(ray,t0,t1))
			{
				return true; 
			}
		}
		// Note: Having this return false will effectively disable/ignore shadows
		return false;
	}


	void Scene::hitProperties(const RayTracing::HitInfo_t &hitinfo, gml::vec3_t &normal, gml::vec2_t &texCoords) const
	{
		// You may use this function if you wish, but it is not necessary.
		//printf("u = %f\n",hitinfo.plane.u);

		hitinfo.objHit -> hitProperties(hitinfo, normal, texCoords);
	}

	gml::vec3_t Scene::shadeRay(const RayTracing::Ray_t &ray, RayTracing::HitInfo_t &hitinfo, const int remainingRecursionDepth) const
	{
		// TODO!

		// Calculate the shade/radiance/color of the given ray. Return the calculated color
		//  - Information about the ray's point of nearest intersection is located
		// in 'hitinfo'
		//  - If remainingRecursionDepth is 0, then _no_ recursive rays (mirror or indirect lighting) should be cast

		// Note: You will have to set up the values for a RayTracing::ShaderValues object, and then
		// pass the object to a shader object to do the appropriate shading.
		//   Use m_shaderManager.getShader() to get an appropriate Shader object for shading
		//  the point based on material properties of the object intersected.

		// When implementing shadows, then the direct lighting component of the
		// calculated ray color wicstdlibll be black if the point is in shadow.


		gml::vec3_t shadePoint = gml::add(ray.o, gml::scale(hitinfo.hitDist, ray.d));
		gml::vec3_t shade(0.0, 0.0, 0.0);

		RayTracing::ShaderValues sv(hitinfo.objHit->getMaterial());
		sv.p = shadePoint;				// Point being shaded (world-space)	
		sv.e = ray.d; 						// View direction
		hitProperties(hitinfo, sv.n, sv.tex);	

		if((hitinfo.objHit->getMaterial()).getShaderType() == Material::MIRROR)
		{

			const Shader::Shader *shader = m_shaderManager.getShader(hitinfo.objHit->getMaterial());

			RayTracing::Ray_t mirrorRay;
			mirrorRay.o = shadePoint;

			mirrorRay.d = gml::sub(ray.d, gml::scale(2 * gml::dot(ray.d, sv.n), sv.n));

			RayTracing::HitInfo_t mirrorRayHitinfo;
			mirrorRayHitinfo.hitDist = FLT_MAX;

			if(rayIntersects(mirrorRay,0.001,FLT_MAX,mirrorRayHitinfo))
			{
				shade = shadeRay(mirrorRay,mirrorRayHitinfo,remainingRecursionDepth-1);
			}

		}
		else
		{

			const Shader::Shader *shader = m_shaderManager.getShader(hitinfo.objHit->getMaterial());

			gml::vec3_t m_lightPos3 = gml::vec3_t(m_lightPos.x/m_lightPos.w, m_lightPos.y/m_lightPos.w, m_lightPos.z/m_lightPos.w);
			sv.lightDir = gml::normalize(gml::sub(m_lightPos3,sv.p)); 

			RayTracing::Ray_t shadowRay;
			shadowRay.o = sv.p;
			shadowRay.d = sv.lightDir;

			sv.lightRad = m_lightRad; 

			// Direction from point to light (cache shade point) 

			//direct light
			float lightDistence = gml::length(gml::sub(m_lightPos3, sv.p ));
			if(shadowsRay(shadowRay, 0.0001, lightDistence))
			{
				shade = m_ambientRad;
			}
			else
			{

				shade = shader->shade(sv);
			}


			if(remainingRecursionDepth > 0)
			{

				RayTracing::Ray_t randomRay;

				randomRay.o = shadePoint;
				randomRay.randomDirection(sv.n);

				RayTracing::HitInfo_t randomRayHitinfo;
				randomRayHitinfo.hitDist = FLT_MAX;

				if(rayIntersects(randomRay,0.001,FLT_MAX,randomRayHitinfo))
				{
					sv.lightRad = shadeRay(randomRay,randomRayHitinfo,remainingRecursionDepth-1); 				// Light radiance
					sv.lightDir = randomRay.d; 										// Direction from point to light (cache shade point) 
					shade = gml::add(shade, shader->shade(sv));	
				}

			}
		}

		// Note: For debugging your rayIntersection() function, this function
		// returns some non-black constant color at first. When you actually implement
		// this function, then initialize shade to black (0,0,0).

		return shade;
	}

	RayTracing::HitInfo_t* Scene::rayIntersectsInParallel(const RayTracing::Ray_t *rays, const float t0, const float t1,const int w, const int h, int objHitIndex) const
	{
		// TODO
		//   Find the closest intersection of the ray in the distance range [t0,t1].
		// Return true if an intersection was found, false otherwise

		const RayTracing::HitInfo_t **hitInfos_array = (const RayTracing::HitInfo_t**)malloc(m_nObjects * sizeof(RayTracing::HitInfo_t*));


		for(GLuint i = 0; i < m_nObjects; i++)
		{

			hitInfos_array[i] = m_scene[i]->rayIntersectsInParallel(rays,t0,t1, w, h, i);

		}

		RayTracing::HitInfo_t *closestHits = findClosestHitsWithCuda(hitInfos_array, w, h, m_nObjects);

		delete[] hitInfos_array;
		return closestHits;

	}

	gml::vec3_t* Scene::shadeRaysInParallel(const RayTracing::Ray_t *rays,const RayTracing::HitInfo_t *hitinfos, const int remainingRecursionDepth, const int w, const int h)
	{

		RayTracing::Object_Kernel_t* hostObjKernel = this -> createObjForKernel();
		const RayTracing::Object_Kernel_t* devObjKernel = objHTD(hostObjKernel,m_nObjects);
		delete[] hostObjKernel;
		float* lightProp = lightPropHTD((float*)&m_lightPos, (float*)&m_lightRad, w, h);

		float* devImage = shadeRaysDirectLightWithCuda(rays ,hitinfos,devObjKernel, lightProp, remainingRecursionDepth,w, h);

		bool* isInShaodow = shadowRaysInParallel(rays, hitinfos, lightProp, w, h);

		float* devImageShadow = shadeRaysShadowLightWithCuda(isInShaodow,w,h,devImage);

		void* toBeCleaned[4] = {(void*)rays, (void*)hitinfos, (void*)devObjKernel, (void*)lightProp};

		cleanUp(toBeCleaned,4);

		return (gml::vec3_t*)rgbDTH(devImageShadow,w,h);

	}

	bool* Scene::shadowRaysInParallel(const RayTracing::Ray_t *rays, const RayTracing::HitInfo_t *hitinfos, const float* lightProp, const int w, const int h) const
	{
		
		RayTracing::Ray_t *shadowRays = genShadowRaysWithCuda(rays,hitinfos,lightProp,w,h);
		
		const bool **isInshadow_array = (const bool**)malloc(m_nObjects * sizeof(bool*));

		for(GLuint i = 0; i < m_nObjects; i++)
		{

			isInshadow_array[i] = m_scene[i]->shadowRaysInParallel(shadowRays,hitinfos,lightProp,w,h);

		}
		
		bool *shadow = mergeShadowWithCuda(shadowRays, isInshadow_array, w, h, m_nObjects);

		delete[] isInshadow_array;
		return shadow;
	}

	float* Scene::hitPropertiesInParallel(const RayTracing::HitInfo_t *hitinfos,  const int w, const int h) const
	{
		return 0;
	}

	RayTracing::Object_Kernel_t* Scene::createObjForKernel() const
	{
		RayTracing::Object_Kernel_t* container = (RayTracing::Object_Kernel_t*)malloc( m_nObjects * sizeof(RayTracing::Object_Kernel_t));

		for(GLuint i = 0; i < m_nObjects; i++)
		{

			container[i].m_geometry_type = m_scene[i] -> getGeometryType();

			// copy material vaules to kernel 
			container[i].m_material.m_surfRefl[0] = m_scene[i]->getMaterial().getSurfRefl().x;
			container[i].m_material.m_surfRefl[1] = m_scene[i]->getMaterial().getSurfRefl().y;
			container[i].m_material.m_surfRefl[2] = m_scene[i]->getMaterial().getSurfRefl().z;

			container[i].m_material.m_hasSpecular = m_scene[i]->getMaterial().hasSpecular();

			container[i].m_material.m_specExp = m_scene[i]->getMaterial().getSpecExp();

			container[i].m_material.m_specRefl[0] = m_scene[i]->getMaterial().getSpecRefl().x;
			container[i].m_material.m_specRefl[1] = m_scene[i]->getMaterial().getSpecRefl().y;
			container[i].m_material.m_specRefl[2] = m_scene[i]->getMaterial().getSpecRefl().z;

			if(m_scene[i]->getMaterial().getShaderType() == Material::ShaderType::SIMPLE)
			{
				container[i].m_material.m_shadeType = RayTracing::SIMPLE;
			}
			if(m_scene[i]->getMaterial().getShaderType() == Material::ShaderType::GOURAUD)
			{
				container[i].m_material.m_shadeType = RayTracing::GOURAUD;
			}
			if(m_scene[i]->getMaterial().getShaderType() == Material::ShaderType::PHONG)
			{
				container[i].m_material.m_shadeType = RayTracing::PHONG;
			}
			if(m_scene[i]->getMaterial().getShaderType() == Material::ShaderType::MIRROR)
			{
				container[i].m_material.m_shadeType = RayTracing::MIRROR;
			}

			// copy oject <-> world space transformation to kernel 

			float* m_objectToWorldFloat = (float*)&(m_scene[i]->getObjectToWorld());
			float* m_objectToWorld_NormalsFloat = (float*)&(m_scene[i]->getObjectToWorld_Normals());
			float* m_worldToObjectFloat = (float*)&(m_scene[i]->getWorldToObject());

			for(int j = 0; j < 16; j++)
			{
				container[i].m_objectToWorld[j] = m_objectToWorldFloat[j];
				container[i].m_objectToWorld_Normals[j] = m_objectToWorld_NormalsFloat[j];
				container[i].m_worldToObject[j] =m_worldToObjectFloat[j];
			}


		}

		return container;
	}

}

