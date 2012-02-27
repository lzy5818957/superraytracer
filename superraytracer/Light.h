//
//  Light.h
//  RayTracer
//
//  Created by Jiachen Zhang on 12-02-26.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef RayTracer_Light_h
#define RayTracer_Light_h

#include "World.h"
#include "GML/gml.h"

namespace Light {
    
    class Light 
    {
    public:
        virtual Color LightPoint(const Ray& eye, const gml::vec3_t& normal, const gml::vec3_t& point) = 0;
        
        inline void SetOwnerWorld(World* world)
        {
            ownerWorld = world;
        }
        inline vec3_t Pos() { return pos; }
    protected:
        vec3 pos;
        World* ownerWorld;
        
    }
}

#endif
