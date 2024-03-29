
#pragma once
#ifndef __INC_CAMERA_KERNEL_H_
#define __INC_CAMERA_KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" float* genViewRayWithCuda(const int w, const int h, float *camPos, float *m_windowToWorld);

#endif