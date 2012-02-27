//
//  Color.cpp
//  RayTracer
//
//  Created by Jiachen Zhang on 12-02-26.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Color.h"

namespace Color{

    Color::Color()
    {
        // Color is black in defult
        red = 0.0;
        green = 0.0;
        blue = 0.0;
    }
    
    
    Color::Color(colorType r, colorType g, colorType b)
    {
        red = r;
        green = g;
        blue = b;
    }
    
    Color::Color(const Color& source)
    {
        red = source.R();
        green = source.G();
        blue = source.B();
    }

}






    
    
    
    
    
    
