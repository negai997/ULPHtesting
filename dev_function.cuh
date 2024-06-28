#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "classes_type.h"
//#include"particle.h"
#include<math.h>
#include <stdio.h>
#include<iostream>


__device__ static const double SoundSpeed_d(sph::FluidType _f);

__device__ double min_d(double a, double b);

__global__ void getdt_dev(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay);

__global__ void adjustC0_dev(double* c0, double c, unsigned int particleNum);

__global__ void inlet_dev(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx);

__global__ void outlet_dev(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx, double outletBcxEdge, double lengthofx);

__global__ void buildNeighb_dev1(unsigned int particleNum, double* ux, double* uy, double* X, double* Y, double* X_max, double* X_min, double* Y_max, double* Y_min);

__global__ void buildNeighb_dev2(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum\
	, const int ngridx, const int ngridy, const double dxrange, const double dyrange, double x_min, double y_min\
	, int* xgcell, int* ygcell, int* celldata, double* grid_d);

__global__ void buildNeighb_dev3(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum, const double* Hsml\
	, const int ngridx, const int ngridy, const double dxrange, const double dyrange, unsigned int* idx, sph::InoutType* iotype\
	, int* xgcell, int* ygcell, int* celldata, double* grid_d, double lengthofx);

void getdt_dev0(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay);

void adjustC0_dev0(double* c0, double c, unsigned int particleNum);

void inlet_dev0(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx);

void outlet_dev0(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx, double outletBcxEdge, double lengthofx);

void buildNeighb_dev01(unsigned int particleNum, double* ux, double* uy, double* X, double* Y, double* X_max, double* X_min, double* Y_max, double* Y_min);

void buildNeighb_dev02(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum\
						, const int ngridx, const int ngridy, const double dxrange, const double dyrange, double x_min, double y_min\
						, int* xgcell, int* ygcell, int* celldata, double* grid_d, const double* Hsml, unsigned int* idx, sph::InoutType* iotype, double lengthofx);







