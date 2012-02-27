//
//  surface.h
//  RayTracer
//
//  Created by Jiachen Zhang on 12-02-26.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef RayTracer_surface_h
#define RayTracer_surface_h

#include "World.h"
#include "Color.h"
#include "Ray.h"
#include "GML/gml.h"

class Surface
{
public:
    virtual Color ShadePoint(Ray& eye, const vec3_t& normal, const vec3_t& point) = 0;
    
    inline void SetOwnerWorld(World* world)
    {
        ownerWorld = world;
    }
    
protected:
    World* ownerWorld;
};

#endif
