//
//  Color.h
//  RayTracer
//
//  Created by Jiachen Zhang on 12-02-26.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef RayTracer_Color_h
#define RayTracer_Color_h

typedef float colorType;

namespace Color {
    

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

} //namespace

#endif
