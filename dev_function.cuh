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

__device__ static inline const double func_factor(double _hsml, int _dim);

__device__ static inline const double func_bweight(double q, double _hsml, int _dim);

__device__ static const double FluidDensity(sph::FluidType _f);

__device__ static const double Gamma(sph::FluidType _f);

__device__ static const double Viscosity(sph::FluidType _f);

__device__ static const double specific_heat(sph::FluidType _f);

__device__ static const double coefficient_heat(sph::FluidType _f);

__global__ void getdt_dev(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay);

__global__ void adjustC0_dev(double* c0, double c, unsigned int particleNum);

__global__ void inlet_dev(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx);

__global__ void outlet_dev(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx, double outletBcxEdge, double lengthofx);

__global__ void buildNeighb_dev1(unsigned int particleNum, double* ux, double* uy, double* X, double* Y, double* X_max, double* X_min, double* Y_max, double* Y_min);

__global__ void buildNeighb_dev2(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum\
	, const int ngridx, const int ngridy, const double dxrange, const double dyrange, double x_min, double y_min\
	, int* xgcell, int* ygcell, int* celldata, int* grid_d, int* lock);

__global__ void buildNeighb_dev3(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum, const double* Hsml\
	, const int ngridx, const int ngridy, const double dxrange, const double dyrange, unsigned int* idx, sph::InoutType* iotype\
	, int* xgcell, int* ygcell, int* celldata, int* grid_d, double lengthofx);

__global__ void run_half1_dev1(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
	, double* x, double* y, double* vx, double* vy, double* rho, double* temperature);

__global__ void run_half2_dev1(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
	, double* x, double* y, double* vx, double* vy, double* rho, double* temperature\
	, double* drho, double* ax, double* ay, double* vol, double* mass\
	, sph::BoundaryType* btype, sph::FixType* ftype, double* temperature_t, const double dt2, double* vmax);

__global__ void singlestep_rhofilter_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* rho_min);

__global__ void singlestep_eos_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* c0, double* rho0, double* rho, double* gamma, double* back_p, double* press);

__global__ void singlestep_updateWeight_dev1(unsigned int particleNum, unsigned int* neibNum, double* hsml, unsigned int** neiblist, double* x, double* y\
											, sph::InoutType* iotype, double lengthofx, double** bweight, double** dbweightx, double** dbweighty);

__global__ void singlestep_updateWeight_dev1_grid(unsigned int particleNum, unsigned int* neibNum2 , double* hsml, double* x, double* y\
	, sph::InoutType* iotype, double lengthofx, double** bweight, double** dbweightx, double** dbweighty, int* grid_d\
	, int* xgcell, int* ygcell, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

__global__ void singlestep_boundryPNV_dev1(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass\
	, double* rho, double* press, double** bweight, double* vx, double* vy, double* Vcc);

__global__ void singlestep_shapeMatrix_dev1(unsigned int particleNum, double* rho, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist, double** bweight\
	, sph::InoutType* iotype, double lengthofx, double* mass, double* M_11, double* M_12, double* M_21, double* M_22);

__global__ void singlestep_boundaryVisc_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_21, double* m_12, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22, sph::FluidType* fltype\
	, double dp, const double C_s, double* turb11, double* turb12, double* turb21, double* turb22);

__global__ void singlestep_fluidVisc_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, double* press, unsigned int* neibNum\
	, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy\
	, double* m_11, double* m_12, double* m_21, double* m_22, double* vx, double* vy, double* mass, double* divvel\
	, sph::FluidType* fltype, double* tau11, double* tau12, double* tau21, double* tau22, double* Vort, double dp, const double C_s\
	, double* turb11, double* turb12, double* turb21, double* turb22, sph::FixType* ftype, double* drho);

__global__ void singlestep_eom_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* press, double* x, double* y, double* C0, double* C\
	, double* mass, unsigned int* neibNum, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx\
	, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* tau11, double* tau12, double* tau21, double* tau22, double* turb11, double* turb12, double* turb21, double* turb22\
	, double* vx, double* vy, double* Avx, double* Avy, double* fintx, double* finty, sph::FixType* ftype, double* ax, double* ay);

__global__ void run_half3Nshiftc_dev1(unsigned int particleNum, sph::FixType* ftype, double* rho, double* half_rho, double* drho, double dt, double* vx, double* half_vx, double* ax\
	, double* vy, double* half_vy, double* ay, double* vol, double* mass, double* x, double* half_x, double* half_y, double* y, double* ux, double* uy\
	, double* temperature, double* half_temperature, double* temperature_t, sph::ShiftingType stype, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, double* Shift_c);


__global__ void run_shifttype_divc_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, double* shift_c, unsigned int* neibNum, unsigned int** neiblist, double* mass\
	, double* rho, double** dbweightx, double** dbweighty, double* Vx, double* Vy, double shiftingCoe, double dt, double dp, double* Shift_x, double* Shift_y\
	, double* x, double* y, double* ux, double* uy, double* drmax, double* drmax2, int* lock);

__global__ void run_shifttype_velc_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, double* rho, double* C0, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, const double bweightdx, double* mass, double** dbweightx, double** dbweighty, double* Vx, double* Vy, double dp, double shiftingCoe\
	, double* Shift_x, double* Shift_y, double* x, double* y, double* ux, double* uy, double* drmax, double* drmax2, int* lock);

__global__ void density_filter_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, unsigned int* neibNum, unsigned int** neiblist, double* press\
	, double* back_p, double* c0, double* rho0, double* mass, double* rho, double** bweight);

void getdt_dev0(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay);

void adjustC0_dev0(double* c0, double c, unsigned int particleNum);

void inlet_dev0(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx);

void outlet_dev0(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx, double outletBcxEdge, double lengthofx);

void buildNeighb_dev01(unsigned int particleNum, double* ux, double* uy, double* X, double* Y, double* X_max, double* X_min, double* Y_max, double* Y_min);

void buildNeighb_dev02(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum\
						, const int ngridx, const int ngridy, const double dxrange, const double dyrange, double x_min, double y_min\
						, int* xgcell, int* ygcell, int* celldata, int* grid_d, const double* Hsml, unsigned int* idx, sph::InoutType* iotype, double lengthofx);

void run_half1_dev0(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
					, double* x, double* y, double* vx, double* vy, double* rho, double* temperature);

void run_half2_dev0(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
	, double* x, double* y, double* vx, double* vy, double* rho, double* temperature\
	, double* drho, double* ax, double* ay, double* vol, double* mass\
	, sph::BoundaryType* btype, sph::FixType* ftype, double* temperature_t, const double dt2, double* vmax);

void singlestep_rhoeos_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* rho_min, double* c0, double* rho0, double* gamma, double* back_p, double* press);

void singlestep_updateWeight_dev0(unsigned int particleNum, unsigned int* neibNum, double* hsml, unsigned int** neiblist, double* x, double* y\
	, sph::InoutType* iotype, double lengthofx, double** bweight, double** dbweightx, double** dbweighty, int* grid_d, int* xgcell, int* ygcell\
	, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

void singlestep_boundryPNV_dev0(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass\
	, double* rho, double* press, double** bweight, double* vx, double* vy, double* Vcc);

void singlestep_shapeMatrix_dev0(unsigned int particleNum, double* rho, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist, double** bweight\
	, sph::InoutType* iotype, double lengthofx, double* mass, double* M_11, double* M_12, double* M_21, double* M_22);

void singlestep_boundaryVisc_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_21, double* m_12, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22, sph::FluidType* fltype\
	, double dp, const double C_s, double* turb11, double* turb12, double* turb21, double* turb22);

void singlestep_fluidVisc_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, double* press, unsigned int* neibNum\
	, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy\
	, double* m_11, double* m_12, double* m_21, double* m_22, double* vx, double* vy, double* mass, double* divvel\
	, sph::FluidType* fltype, double* tau11, double* tau12, double* tau21, double* tau22, double* Vort, double dp, const double C_s\
	, double* turb11, double* turb12, double* turb21, double* turb22, sph::FixType* ftype, double* drho);

void singlestep_eom_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* press, double* x, double* y, double* C0, double* C\
	, double* mass, unsigned int* neibNum, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx\
	, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* tau11, double* tau12, double* tau21, double* tau22, double* turb11, double* turb12, double* turb21, double* turb22\
	, double* vx, double* vy, double* Avx, double* Avy, double* fintx, double* finty, sph::FixType* ftype, double* ax, double* ay);

void run_half3Nshiftc_dev0(unsigned int particleNum, sph::FixType* ftype, double* rho, double* half_rho, double* drho, double dt, double* vx, double* half_vx, double* ax\
	, double* vy, double* half_vy, double* ay, double* vol, double* mass, double* x, double* half_x, double* half_y, double* y, double* ux, double* uy\
	, double* temperature, double* half_temperature, double* temperature_t, sph::ShiftingType stype, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, double* Shift_c);

void run_shifttype_divc_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, double* shift_c, unsigned int* neibNum, unsigned int** neiblist, double* mass\
	, double* rho, double** dbweightx, double** dbweighty, double* Vx, double* Vy, double shiftingCoe, double dt, double dp, double* Shift_x, double* Shift_y\
	, double* x, double* y, double* ux, double* uy, double* drmax, double* drmax2, int* lock);

void run_shifttype_velc_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, double* rho, double* C0, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, const double bweightdx, double* mass, double** dbweightx, double** dbweighty, double* Vx, double* Vy, double dp, double shiftingCoe\
	, double* Shift_x, double* Shift_y, double* x, double* y, double* ux, double* uy, double* drmax, double* drmax2, int* lock);

void density_filter_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, unsigned int* neibNum, unsigned int** neiblist, double* press\
	, double* back_p, double* c0, double* rho0, double* mass, double* rho, double** bweight);


//修改single_step_temperature_gaojie()

__global__ void single_temp_eos_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* C0, double* Rho0, double* rho, double* Gamma, double* back_p, double* press);

__global__ void single_temp_boundary_dev1(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass, double* rho\
	, double* press, double** bweight, double* vx, double* vy, double* Vcc);

__global__ void single_temp_boundary_dev1_grid(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass, double* rho\
	, double* press, double** bweight, double* vx, double* vy, double* Vcc, int* xgcell, int* ygcell, int* grid_d, int* celldata\
	, int ngridx, int ngridy, double dxrange, double dyrange, double* hsml, double* x, double* y, sph::InoutType* iotype, double lengthofx);

__global__ void single_temp_shapematrix_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double* mass, double* m_11, double* m_12, double* m_21, double* m_22, double* M_11\
	, double* M_12, double* M_13, double* M_14, double* M_15, double* M_21, double* M_22, double* M_23, double* M_24, double* M_25\
	, double* M_31, double* M_32, double* M_33, double* M_34, double* M_35, double* M_51, double* M_52, double* M_53, double* M_54, double* M_55);

__global__ void single_temp_shapematrix_dev1_grid(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double* mass, double* m_11, double* m_12, double* m_21, double* m_22, double* M_11\
	, double* M_12, double* M_13, double* M_14, double* M_15, double* M_21, double* M_22, double* M_23, double* M_24, double* M_25\
	, double* M_31, double* M_32, double* M_33, double* M_34, double* M_35, double* M_51, double* M_52, double* M_53, double* M_54, double* M_55\
	, int* xgcell, int* ygcell, int* grid_d, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

__global__ void single_temp_bdvisco_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* Tau11, double* Tau12, double* Tau21, double* Tau22, sph::FluidType* fltype, double dp\
	, double C_s, double* turb11, double* turb12, double* turb21, double* turb22);

__global__ void single_temp_bdvisco_dev1_grid(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* Tau11, double* Tau12, double* Tau21, double* Tau22, sph::FluidType* fltype, double dp\
	, double C_s, double* turb11, double* turb12, double* turb21, double* turb22\
	, int* xgcell, int* ygcell, int* grid_d, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

__global__ void single_step_fluid_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22, sph::FluidType* fltype, double dp\
	, double C_s, double* turb11, double* turb12, double* turb21, double* turb22, double* temperature, double* M_31, double* M_32, double* M_33, double* M_34, double* M_35\
	, double* M_51, double* M_52, double* M_53, double* M_54, double* M_55, double* divvel, double* vort, sph::FixType* ftype, double* drho, double* temperature_x, double* temperature_t);

__global__ void single_step_fluid_dev1_grid(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22, sph::FluidType* fltype, double dp\
	, double C_s, double* turb11, double* turb12, double* turb21, double* turb22, double* temperature, double* M_31, double* M_32, double* M_33, double* M_34, double* M_35\
	, double* M_51, double* M_52, double* M_53, double* M_54, double* M_55, double* divvel, double* vort, sph::FixType* ftype, double* drho, double* temperature_x, double* temperature_t\
	, int* xgcell, int* ygcell, int* grid_d, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

__global__ void single_step_fluid_eom_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, double* C0, double* C\
	, unsigned int* neibNum, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* ax, double* ay\
	, double* m_11, double* m_12, double* m_21, double* m_22, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22\
	, double* turb11, double* turb12, double* turb21, double* turb22, double* fintx, double* finty, sph::FixType* ftype, double* Avx, double* Avy, double* turbx, double* turby);

__global__ void single_step_fluid_eom_dev1_grid(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, double* C0, double* C\
	, unsigned int* neibNum, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* ax, double* ay\
	, double* m_11, double* m_12, double* m_21, double* m_22, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22\
	, double* turb11, double* turb12, double* turb21, double* turb22, double* fintx, double* finty, sph::FixType* ftype, double* Avx, double* Avy, double* turbx, double* turby\
	, int* xgcell, int* ygcell, int* grid_d, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

void single_temp_eos_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* C0, double* Rho0, double* rho, double* Gamma, double* back_p, double* press);

void single_temp_boundary_dev0(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass, double* rho\
	, double* press, double** bweight, double* vx, double* vy, double* Vcc, int* xgcell, int* ygcell, int* grid_d, int* celldata\
	, int ngridx, int ngridy, double dxrange, double dyrange, double* hsml, double* x, double* y, sph::InoutType* iotype, double lengthofx);

void single_temp_shapematrix_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double* mass, double* m_11, double* m_12, double* m_21, double* m_22, double* M_11\
	, double* M_12, double* M_13, double* M_14, double* M_15, double* M_21, double* M_22, double* M_23, double* M_24, double* M_25\
	, double* M_31, double* M_32, double* M_33, double* M_34, double* M_35, double* M_51, double* M_52, double* M_53, double* M_54, double* M_55\
	, int* xgcell, int* ygcell, int* grid_d, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

void single_temp_bdvisco_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* Tau11, double* Tau12, double* Tau21, double* Tau22, sph::FluidType* fltype, double dp\
	, double C_s, double* turb11, double* turb12, double* turb21, double* turb22\
	, int* xgcell, int* ygcell, int* grid_d, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

void single_step_fluid_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22, sph::FluidType* fltype, double dp\
	, double C_s, double* turb11, double* turb12, double* turb21, double* turb22, double* temperature, double* M_31, double* M_32, double* M_33, double* M_34, double* M_35\
	, double* M_51, double* M_52, double* M_53, double* M_54, double* M_55, double* divvel, double* vort, sph::FixType* ftype, double* drho, double* temperature_x, double* temperature_t\
	, int* xgcell, int* ygcell, int* grid_d, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);

void single_step_fluid_eom_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, double* C0, double* C\
	, unsigned int* neibNum, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* ax, double* ay\
	, double* m_11, double* m_12, double* m_21, double* m_22, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22\
	, double* turb11, double* turb12, double* turb21, double* turb22, double* fintx, double* finty, sph::FixType* ftype, double* Avx, double* Avy, double* turbx, double* turby\
	, int* xgcell, int* ygcell, int* grid_d, int* celldata, int ngridx, int ngridy, double dxrange, double dyrange);