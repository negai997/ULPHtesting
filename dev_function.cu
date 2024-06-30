#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dev_function.cuh"

//#include "fluid.h"
//#include<math.h>
//#include <stdio.h>

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

__device__ double atomicMaxDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(max(val, __longlong_as_double(assumed))));
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
	__syncthreads();
	atomicMinDouble(dtmin, min_d(dt00, dt22));
}

__global__ void adjustC0_dev(double* c0, double c, unsigned int particleNum) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		//particle* ii = particles[i];
		c0[i] = c;
	}
}

__global__ void inlet_dev(unsigned int particleNum,double*x, sph::InoutType* iotype,double outletBcx) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x) {
		double xi = x[i];

		if (xi < 0) {
			iotype[i]= sph::InoutType::Inlet;
			//ii->temperature = temperature_min;
		}
		else if (xi > 0 && xi < outletBcx)
		{
			iotype[i] = sph::InoutType::Fluid;
		}

	}

}

__global__ void outlet_dev(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx,double outletBcxEdge,double lengthofx) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		//particle* ii = particles[i];
		//if (ii->btype == sph::BoundaryType::Boundary) continue;
		double xi = x[i];
		if (xi > outletBcx) {
			//ii->btype = BoundaryType::Boundary;
			iotype[i] = sph::InoutType::Outlet;
			/*std::cerr << std::endl << "outletBcx  " << outletBcx  << std::endl;
			std::cerr << std::endl << "outletBcx + 3.001 * dp " << outletBcx + 3.001 * dp << std::endl;*/
			if (xi > outletBcxEdge) {
				x[i] = xi - lengthofx;
				iotype[i] = sph::InoutType::Inlet;
				//ii->temperature = temperature_min;
			}
		}
	}
}

__global__ void buildNeighb_dev1(unsigned int particleNum,double*ux,double*uy,double*X,double*Y, double* X_max,double* X_min,double* Y_max,double* Y_min) {
	double x_max = 1e-300;
	double x_min = 1e300;
	double y_max = 1e-300;
	double y_min = 1e300;
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x) {
		//particle* ii = particles[i];
		ux[i] = 0;
		uy[i] = 0;
		const double x = X[i];
		const double y = Y[i];
		x_max = x_max < x ? x : x_max;
		x_min = x_min > x ? x : x_min;
		y_max = y_max < y ? y : y_max;
		y_min = y_min > y ? y : y_min;
	}
	__syncthreads();
	atomicMaxDouble(X_max, x_max);
	atomicMaxDouble(Y_max, y_max);
	atomicMinDouble(X_min, x_min);
	atomicMinDouble(Y_min, y_min);
}

__global__ void buildNeighb_dev2(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum\
								, const int ngridx, const int ngridy, const double dxrange, const double dyrange, double x_min, double y_min\
								, int* xgcell, int* ygcell, int* celldata, int* grid_d) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x + 1; i < particleNum+1; i += gridDim.x * blockDim.x)
	{
		double x, y;
		x = X[i - 1];
		//printf("%lf", x);
		y = Y[i - 1];
		//particlesa.clearNeiblist(i - 1);
		if (x != x) {
			printf("Nan_as%lf\n", X[i - 1]);
		}

		for (int j = 0; j < MAX_NEIB; j++) {
			neiblist[i - 1][j] = 0;
		}
		neibNum[i - 1] = 0;



		const int xxcell = int(static_cast<double>(ngridx) / dxrange * (x - x_min) + 1.0);//粒子x所对应的网格的坐标>=1
		const int yycell = int(static_cast<double>(ngridy) / dyrange * (y - y_min) + 1.0);
		//数组赋值，记录每个粒子的网格坐标,用两个一维数值分别记录
		xgcell[i - 1] = xxcell;
		ygcell[i - 1] = yycell;
		if (xxcell > ngridx || yycell > ngridy || i > particleNum) {
			printf("\n\nError: Neighbor indexing out of range, i%d\n",i);
			//std::cerr << std::endl << "\nError: Neighbor indexing out of range, i " << i << std::endl;
			printf("xxcell %d yycell %d", xxcell, yycell);
			//std::cerr << "xxcell " << xxcell << " yycell " << yycell;
			//exit(-1);
			//可以用全局变量将判断情况传回主机后在主机判断是否exit，目前暂且搁置。
		}
		//try {
			celldata[i - 1] = static_cast<int>(grid_d[(xxcell-1) * ngridy + yycell-1]);//记录粒子所在的网格编号；
			//但是后面有一句grid(xxcell, yycell) = i;这样就不再是记录网格编号，而是记录i，即粒子id
			//std::cerr << std::endl << "static_cast<int>(grid(xxcell, yycell)): " << celldata[i - 1] << std::endl;
		//}
		//catch (int e) {
		//	std::cout << "\nException: " << e << std::endl;
		//	std::cout << xxcell << '\t' << yycell;
		//}
		//std::cerr << std::endl << "grid(xxcell, yycell): " << grid(xxcell, yycell) << std::endl;
		grid_d[(xxcell - 1) * ngridy + yycell - 1] = i;// i starts from 0 //没理解这句什么意思？
		//std::cerr << std::endl << "grid(xxcell, yycell): " << grid(xxcell, yycell) << std::endl;
	}
}

__global__ void buildNeighb_dev3(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum, const double* Hsml\
					, const int ngridx, const int ngridy, const double dxrange, const double dyrange, unsigned int* idx, sph::InoutType* iotype\
					, int* xgcell, int* ygcell, int* celldata, int* grid_d,double lengthofx) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x + 1; i <= particleNum; i += gridDim.x * blockDim.x)// i=0 ~ ntotal-1
	{
		const double hsml =Hsml[i - 1];
		const int dnxgcell = xgcell[i - 1] - int(static_cast<double>(ngridx) * 3 * hsml / dxrange) - 1;
		const int dnygcell = ygcell[i - 1] - int(static_cast<double>(ngridy) * 3 * hsml / dyrange) - 1;
		const int dpxgcell = xgcell[i - 1] + int(static_cast<double>(ngridx) * 3 * hsml / dxrange) + 1;
		const int dpygcell = ygcell[i - 1] + int(static_cast<double>(ngridy) * 3 * hsml / dyrange) + 1;

		//const int minxcell = 1;
		const int minxcell = dnxgcell > 1 ? dnxgcell : 1;
		const int maxxcell = dpxgcell < ngridx ? dpxgcell : ngridx;
		const int minycell = dnygcell > 1 ? dnygcell : 1;
		const int maxycell = dpygcell < ngridy ? dpygcell : ngridy;
		//goto语句
		//...
		//goto语句转为if语句
		for (auto ycell = minycell; ycell <= maxycell; ycell++)
		{
			for (auto xcell = minxcell; xcell <= maxxcell; xcell++)
			{
				int j = static_cast<int>(grid_d[(xcell - 1) * ngridy + ycell - 1]);//网格编号
				for (j; j > 0; j = celldata[j - 1])//防止重复比较（并行后取消）
				{
					if (i == j)
						continue;
					//math::vector xi(particles[i-1]->getX() - particles[j-1]->getX(), particles[i-1]->getY() - particles[j-1]->getY());
					const double xi = X[i - 1];
					const double yi = Y[i - 1];
					const double xj = X[j - 1];
					const double yj = Y[j - 1];
					const double dx = xi - xj;
					const double dy = yi - yj;
					const double r = sqrt(dx * dx + dy * dy);//除了一般的r，还有周期边界的r2
					if (r == 0 && i <= particleNum && j <= particleNum &&i!=j) {
						printf("\nError: two particles occupy the same position\n");
						//std::cerr << "\nError: two particles occupy the same position\n";
						printf("i=%d j=%d ......(i,j start form 1)\n",i,j);
						//std::cerr << "i=" << i << " j=" << j << std::endl;
						printf("%d and %d\n", idx[i - 1], idx[j - 1]);
						//std::cerr << (particlesa.idx[i - 1]) << " and " << (particlesa.idx[j - 1]) << std::endl;
						printf("at %lf , %lf\n", xi, yi);
						//std::cerr << "at " << xi << '\t' << yi;
						printf("particleNum=%d\n", particleNum);
						//std::cerr << "ntotal=" << ntotal << std::endl;
						//this->output(istep);
						//exit(-1);
						//同上，暂不修改
					}
					const double mhsml = Hsml[i - 1];
					const double horizon = 3.3 * mhsml;//					
					if (r < horizon) {
						neiblist[i - 1][neibNum[i - 1]] = j - 1;
						neibNum[i - 1]++;
						//particlesa.add2Neiblist(i - 1, j - 1);
						//particlesa.add2Neiblist(j - 1, i - 1);出于并行需求，我们允许他重复比较
					}
				}
			}
		}
		//当out粒子转到inlet粒子，转过来的inlet粒子搜不到旧的inlet粒子了
		//当进出口使用周期边界时，加上下面这段:
		if (xgcell[i - 1] < 3)//当粒子数较小时，背景网格x=1的网格中有时候只有一竖条粒子，明显不足。需要把x=2的背景网格也拉进来寻找
		{
			//std::cerr << std::endl << "find xmimcell: " << xgcell[i - 1] << std::endl;
			//std::cerr << std::endl << "粒子i = " << i << std::endl;				
			for (auto ycell = minycell; ycell <= maxycell; ycell++)
			{
				for (auto xcell = ngridx - 1; xcell <= ngridx; xcell++) //最右边的两层网格 ngridx 与 ngridx-1
				{
					int j = static_cast<int>(grid_d[(xcell - 1) * ngridy + ycell - 1]);
					for (j; j > 0; j = celldata[j - 1])
					{
						//除了一般的r，还有周期边界的r2
						if (iotype[i - 1] == sph::InoutType::Inlet)
						{
							const double xi = X[i - 1];
							const double yi = Y[i - 1];
							const double xj = X[j - 1];
							const double yj = Y[j - 1];
							const double dy = yi - yj;
							const double dx2 = xi + lengthofx - xj;
							const double r2 = sqrt(dx2 * dx2 + dy * dy);
							const double mhsml = Hsml[i - 1];
							const double horizon = 3.3 * mhsml;//
							if (r2 < horizon) {
								//particlesa.add2Neiblist(i - 1, j - 1);
								neiblist[i - 1][neibNum[i - 1]] = j - 1;
								neibNum[i - 1]++;
								//particlesa.add2Neiblist(i - 1, j - 1);
								//particlesa.add2Neiblist(j - 1, i - 1);
								//std::cerr << std::endl << "find onutlet of inlet: " << j << std::endl;
							}
						}
						//std::cerr << std::endl << "j: " << j << std::endl;						
					}
				}
			}
		}
	}
}

__global__ void run_half1_dev1(unsigned int particleNum,double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
														,double* x, double* y, double* vx, double* vy, double* rho, double* temperature) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		//if (particles[i]->btype == sph::BoundaryType::Boundary) continue;
		//particle* ii = particles[i];
		/* code */
		//(*i)->storeHalf();
		half_x[i] = x[i];//i时刻的位置
		half_y[i] = y[i];
		half_vx[i] = vx[i];//i时刻的速度
		half_vy[i] = vy[i];
		half_rho[i] = rho[i];
		half_temperature[i] = temperature[i];
	}
}

__global__ void run_half2_dev1(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
														, double* x, double* y, double* vx, double* vy, double* rho, double* temperature\
														, double* drho, double* ax, double *ay, double* vol, double* mass\
														, sph::BoundaryType* btype, sph::FixType* ftype, double* temperature_t, const double dt2, double* vmax) {
	double vel = 0;
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		/* code */
		//particle* ii = particles[i];
		//(*i)->integration1sthalf(dt2);
		if (btype[i] == sph::BoundaryType::Boundary) continue;

		if (ftype[i] != sph::FixType::Fixed) {
			rho[i] = half_rho[i] + drho[i] * dt2;
			vx[i] = half_vx[i] + ax[i] * dt2;//(i+0.5*dt)时刻的速度
			vy[i] = half_vy[i] + ay[i] * dt2;
			vol[i] = mass[i] / rho[i];
			temperature[i] = half_temperature[i] + temperature_t[i] * dt2;
		}
		x[i] = half_x[i] + vx[i] * dt2;//-----------------------
		double Y = half_y[i] + vy[i] * dt2;//加个判断，防止冲进边界
		/*if (y > indiameter * 0.5 - dp|| y < -indiameter * 0.5 + dp) {
			y = half_y[i];
		}	*/
		y[i] = Y;
		const double Vx = vx[i];
		const double Vy = vy[i];
		vel = vel<sqrt(Vx * Vx + Vy * Vy)? sqrt(Vx * Vx + Vy * Vy):vel;
		
	}
	atomicMaxDouble(vmax, vel);
}

void getdt_dev0(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay) {

	getdt_dev<<<32,32>>>(particleNum, dtmin, divvel, hsml, fltype, vmax, Ax, Ay);
	CHECK(cudaDeviceSynchronize());
}

void adjustC0_dev0(double* c0,double c,unsigned int particleNum) {
	adjustC0_dev<<<32,32>>>(c0, c, particleNum);
	CHECK(cudaDeviceSynchronize());
}

void inlet_dev0(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx) {
	inlet_dev << <32, 32 >> > (particleNum, x,  iotype, outletBcx);
	CHECK(cudaDeviceSynchronize());
}

void outlet_dev0(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx, double outletBcxEdge, double lengthofx) {
	outlet_dev << <32, 32 >> > (particleNum, x, iotype, outletBcx, outletBcxEdge, lengthofx);
	CHECK(cudaDeviceSynchronize());
}

void buildNeighb_dev01(unsigned int particleNum, double* ux, double* uy, double* X, double* Y, double* X_max, double* X_min, double* Y_max, double* Y_min) {
	buildNeighb_dev1 << <32, 32 >> > (particleNum, ux, uy, X, Y, X_max, X_min, Y_max, Y_min);
	CHECK(cudaDeviceSynchronize());
}

void buildNeighb_dev02(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum\
	, const int ngridx, const int ngridy, const double dxrange, const double dyrange, double x_min, double y_min\
	, int* xgcell, int* ygcell, int* celldata, int* grid_d, const double* Hsml, unsigned int* idx, sph::InoutType* iotype, double lengthofx) {
	//目前网格法无法并行，暂不考虑其并行方法，但并行化建议使用全搜索法
	buildNeighb_dev2 << <1, 1 >> > ( particleNum, X, Y, neiblist, neibNum, ngridx, ngridy, dxrange, dyrange, x_min, y_min, xgcell, ygcell, celldata, grid_d);
	CHECK(cudaDeviceSynchronize());
	buildNeighb_dev3 << <32, 32 >> > (particleNum, X, Y, neiblist, neibNum, Hsml, ngridx, ngridy, dxrange, dyrange, idx, iotype, xgcell, ygcell, celldata, grid_d, lengthofx);
	CHECK(cudaDeviceSynchronize());
}

void run_half1_dev0(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
	, double* x, double* y, double* vx, double* vy, double* rho, double* temperature) {
	run_half1_dev1<<<32,32>>>(particleNum, half_x, half_y, half_vx, half_vy, half_rho, half_temperature, x, y, vx, vy, rho, temperature);
	CHECK(cudaDeviceSynchronize());
}

void run_half2_dev0(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
					, double* x, double* y, double* vx, double* vy, double* rho, double* temperature\
					, double* drho, double* ax, double* ay, double* vol, double* mass\
					, sph::BoundaryType* btype, sph::FixType* ftype, double* temperature_t, const double dt2, double* vmax) {

	run_half2_dev1 << <32, 32 >> > (particleNum, half_x, half_y, half_vx, half_vy, half_rho, half_temperature\
												, x, y, vx, vy, rho, temperature\
												, drho, ax, ay, vol, mass\
												, btype, ftype, temperature_t, dt2, vmax);
	CHECK(cudaDeviceSynchronize());
}








