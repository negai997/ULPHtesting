#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include"fluid.h"
#include<math.h>
#include <stdio.h>

__device__ static const double SoundSpeed_d(sph::FluidType _f);


__device__ double min_d(double a, double b);

__global__ void getdt_dev(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay);

__global__ void adjustC0_dev(double* c0, double c, unsigned int particleNum);

void getdt_dev0(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay);

void adjustC0_dev0(double* c0, double c, unsigned int particleNum);