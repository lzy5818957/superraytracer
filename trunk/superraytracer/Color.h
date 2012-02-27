//
//  Color.h
//  RayTracer
//
//  Created by Jiachen Zhang on 12-02-26.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//
#pragma once
#ifndef RAYTRACER_COLOR_H
#define RAYTRACER_COLOR_H

typedef float colorType;
    

class Color
{
public:
    Color();
    Color(colorType r, colorType g, colorType b);
    Color(const Color& source);
    ~Color() {}
    
    inline const colorType& R() const { return red; }
    inline const colorType& G() const { return green; }
    inline const colorType& B() const { return blue; }
    
    inline void R(colorType r) { red = r; }
    inline void G(colorType g) { green = g; }
    inline void B(colorType b) { blue = b; }
private:
    colorType red, green, blue;
};



#endif
