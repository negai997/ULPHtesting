#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dev_function.cuh"

//#include "fluid.h"
#include<math.h>
#include <stdio.h>

/*

*/

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		printf("Error: %s:%d, ", __FILE__, __LINE__);\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(1);\
	}\
}\

__device__ double atomicMinDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(min(val, __longlong_as_double(assumed))));
	} while (assumed != old);

	return __longlong_as_double(old);
}

__device__ static const double SoundSpeed_d(sph::FluidType _f) {
	switch (_f)
	{
	case sph::FluidType::Air:
	case sph::FluidType::Water:
	case sph::FluidType::Moisture:
	default:
		return 10;
		break;
	}
};

__device__ double min_d(double a, double b) {
	if (a < b)
		return a;
	else
		return b;
}


__global__ void getdt_dev(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay) {
	double dt00= 1e300;
	double dt22= 1e300;
	for (int i = blockDim.x*blockIdx.x+threadIdx.x; i < particleNum; i +=gridDim.x*blockDim.x)
	{
		double divv = divvel[i];
		const double dt0 = 0.1 * hsml[i] / (SoundSpeed_d(fltype[i]) + vmax);
		dt00 = min_d(dt00, dt0);
		//const double dt1 = 0.25 * particlesa.hsml[i] / (ii)->c;
		//dt11 = std::min(dt11, dt1);
		const double ax = Ax[i];
		const double ay = Ay[i];
		const double fa = sqrt(ax * ax + ay * ay);
		const double dt2 = 0.1 * sqrt(hsml[i] / fa);
		dt22 = min_d(dt22, dt2);

		//const double dt3 = 0.125 * particlesa.hsml[i] * particlesa.hsml[i]*particlesa.rho[i]/ sph::Fluid::Viscosity(particlesa.fltype[i]);//最大sph::Fluid::Viscosity(particlesa.fltype[i])
		//dt33 = std::min(dt33, dt3);

	}
	atomicMinDouble(dtmin, min_d(dt00, dt22));
}

__global__ void adjustC0_dev(double* c0, double c, unsigned int particleNum) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		//particle* ii = particles[i];
		c0[i] = c;
	}
}

void getdt_dev0(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay) {

	getdt_dev<<<32,126>>>(particleNum, dtmin, divvel, hsml, fltype, vmax, Ax, Ay);
	CHECK(cudaDeviceSynchronize());
}

void adjustC0_dev0(double* c0,double c,unsigned int particleNum) {
	adjustC0_dev<<<32,32>>>(c0, c, particleNum);
	CHECK(cudaDeviceSynchronize());
}