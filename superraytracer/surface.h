//
//  surface.h
//  RayTracer
//
//  Created by Jiachen Zhang on 12-02-26.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//
#pragma once

#ifndef RAYTRACER_SURFACE_H
#define RAYTRACER_SURFACE_H

#include "World.h"
#include "Color.h"
#include "Ray.h"
#include "Utils/GML/gml.h"

class Surface
{
public:
    virtual Color ShadePoint(Ray& eye, const gml::vec3_t& normal, const gml::vec3_t& point) = 0;
    
    inline void SetOwnerWorld(World* world)
    {
        ownerWorld = world;
    }
    
protected:
    World* ownerWorld;
};

#endif
