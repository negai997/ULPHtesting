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

__device__ static inline const double func_factor(double _hsml, int _dim) {
	return 1.0 / (3.14159265358979 * pow(_hsml, _dim)) * (1.0 - expf(-9.0)) / (1.0 - 10.0 * expf(-9.0));
}
//bweight(q,hsml,2)
__device__ static inline const double func_bweight(double q, double _hsml, int _dim) {
	return q <= 3.0 ? func_factor(_hsml, _dim) * (expf(-q * q) - expf(-9.0)) : 0;
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

__device__ static const double FluidDensity(sph::FluidType _f) {
	switch (_f)
	{
	case sph::FluidType::Air:
		//return 1.225;
		return 1.225;
		break;
	case sph::FluidType::Water:
		return 1.0;//密度的大小严重影响粒子的ay
		break;
	case sph::FluidType::Moisture:
		//return 1.23;
		return 20.0;
		break;
	default:
		return 0;
		break;
	}
};
__device__ static const double Gamma(sph::FluidType _f) {
	switch (_f)
	{
	case sph::FluidType::Air:
		return 1.4;
		//return 7.0;
		break;
	case sph::FluidType::Water:
		//return 7.0;
		return 1.4;
		break;
	case sph::FluidType::Moisture:
		return 1.41;
		//return 7.0;
		break;
	default:
		return 0;
		break;
	}
};
__device__ static const double Viscosity(sph::FluidType _f) {
	switch (_f)
	{
	case sph::FluidType::Air:
		return 0.000008928;
		break;
	case sph::FluidType::Water:
		//动力粘度
		return 0.001;
		break;
	case sph::FluidType::Moisture:
		return 0.000008928;
		break;
	default:
		return 0;
		break;
	}
};
__device__ static const double specific_heat(sph::FluidType _f) {        //比热容ci
	switch (_f)
	{
	case sph::FluidType::Air:
		return 1.003;
		break;
	case sph::FluidType::Water:
		return 1000;
		break;
	case sph::FluidType::Moisture:
		return 100;
		break;
	default:
		return 0;
		break;
	}
};
__device__ static const double coefficient_heat(sph::FluidType _f) {     //传热系数ki
	switch (_f)
	{
	case sph::FluidType::Air:
		return 0.0267;
		break;
	case sph::FluidType::Water:
		//return 0.6;
		return 0.1;//
		break;
	case sph::FluidType::Moisture:
		return 0.1;
		break;
	default:
		return 0;
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
								, int* xgcell, int* ygcell, int* celldata, int* grid_d, int* lock) {

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
		//printf("CAS is %d\n", atomicCAS(lock, 0, 1));
		for (int s = 0; s < 32; s++) {
			if ((blockDim.x * blockIdx.x + threadIdx.x) % 32 != s)
				continue;
			while (atomicCAS(lock, 0, 1) != 0);

			celldata[i - 1] = static_cast<int>(grid_d[(xxcell - 1) * ngridy + yycell - 1]);//记录粒子所在的网格编号；
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
			atomicExch(lock, 0);
			__threadfence();
		}

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
						//particlesa.add2Neiblist(j - 1, i - 1);出于并行需求，我们允许他重复比较/					
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
		//printf("particle_1's neighberNum is %d\n", neibNum[0]);
	}
	//printf("particle_1's neighberNum is %d\n", neibNum[0]);
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

__global__ void singlestep_rhofilter_dev1(unsigned int particleNum,sph::BoundaryType* btype, double* rho, double* rho_min) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		//particle* ii = particles[i];  

		if (btype[i] != sph::BoundaryType::Boundary)
		{
			if (rho[i] < rho_min[i])
			{
				rho[i] = rho_min[i];
			}
		}
	}
}

__global__ void singlestep_eos_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* c0, double* rho0, double* rho, double* gamma, double* back_p, double* press) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		if (btype[i] == sph::BoundaryType::Boundary) continue;
		const double dc0 = c0[i];
		const double drho0 = rho0[i];
		const double drhoi = rho[i];
		const double dgamma = gamma[i];
		const double b = dc0 * dc0 * drho0 / dgamma;     //b--B,
		const double p_back = back_p[i];

		press[i] = b * (pow(drhoi / drho0, dgamma) - 1.0) + p_back;      // 流体区

		if (press[i] < p_back)
			press[i] = p_back;
	}
}

__global__ void singlestep_updateWeight_dev1(unsigned int particleNum, unsigned int* neibNum, double* hsml, unsigned int** neiblist, double* x, double* y\
												,sph::InoutType* iotype, double lengthofx, double** bweight, double** dbweightx, double** dbweighty) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		const unsigned int dev_neibNum = neibNum[i];
		const double dev_hsml = hsml[i];
		double sum = 0;

		for (int j = 0; j < dev_neibNum; j++)//�����κ����Աȣ�neiblist.size()=neighborNum(i)
		{
			const int jj = neiblist[i][j];//jj=k
			double dx = x[i] - x[jj];//xi(1)
			if (iotype[i] == sph::InoutType::Inlet && iotype[jj] == sph::InoutType::Outlet)
			{
				dx = x[i] + lengthofx - x[jj];//xi(1)
			}
			if (iotype[i] == sph::InoutType::Outlet && iotype[jj] == sph::InoutType::Inlet)
			{
				dx = x[i] - lengthofx - x[jj];//xi(1)
			}
			//const double dx = particlesa.x[i]-particlesa.x[jj];//xi(1)
			const double dy = y[i] - y[jj];//xi(2)
			const double r = sqrt(dx * dx + dy * dy);
			//const double q = r / hsml;   //q�Ĵ�����ͬ�������һ��mhsml(662-664)
			const double hsmlj = hsml[jj];//hsmlΪhsml(i)��hsmljΪhsml(k)
			const double mhsml = (dev_hsml + hsmlj) * 0.5;
			const double q = r / mhsml;
			if (q > 3.0) {
				bweight[i][j] = 0;
				dbweightx[i][j] = 0;
				dbweighty[i][j] = 0;
				continue;
			}

			const double fac = 1.0 / (3.14159265358979323846 * mhsml * mhsml) / (1.0 - 10.0 * expf(-9));
			const double dev_bweight = fac * (expf(-q * q) - expf(-9));

			sum += dev_bweight;
			bweight[i][j] = dev_bweight;
			const double factor = fac * expf(-q * q) * (-2.0 / mhsml / mhsml);

			dbweightx[i][j] = factor * dx;
			dbweighty[i][j] = factor * dy;
		}

		for (int j = 0; j < neibNum[i]; j++)
		{
			bweight[i][j] /= sum;
		}
	}
}

__global__ void singlestep_boundryPNV_dev1(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass\
											, double* rho, double* press, double** bweight, double* vx, double* vy, double* Vcc) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		//particle* ii = particles[i];
		if (btype[i] != sph::BoundaryType::Boundary) continue;
		//if (particlesa.iotype[i] == sph::InoutType::Buffer) continue;			
		double p = 0, vcc = 0;
		double v_x = 0, v_y = 0;

		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			//if (particlesa.btype[i] == particlesa.btype[jj]) continue;//Buffer粒子也在这里，导致流出边界固壁的压力不正常
			if (ftype[jj] == sph::FixType::Fixed) continue;//其他固壁粒子不参与，ghost不参与，buffer参与
			const double mass_j = mass[jj];
			const double rho_j = rho[jj];
			const double p_k = press[jj];
			p += p_k * bweight[i][j];//
			vcc += bweight[i][j];
			v_x += vx[jj] * bweight[i][j];//需要将速度沿法线分解，还没分
			v_y += vy[jj] * bweight[i][j];
		}
		p = vcc > 0.00000001 ? p / vcc : 0;//p有值
		v_x = vcc > 0.00000001 ? v_x / vcc : 0;//v_x一直为0！待解决
		v_y = vcc > 0.00000001 ? v_y / vcc : 0;
		double vx0 = 0;
		double vy0 = 0;
		Vcc[i] = vcc;
		press[i] = p;
		vx[i] = 2.0 * vx0 - v_x;//无滑移，要改成径向，切向
		vy[i] = 2.0 * vy0 - v_y;

	}
}

__global__ void singlestep_shapeMatrix_dev1(unsigned int particleNum, double* rho, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist, double** bweight\
											, sph::InoutType* iotype, double lengthofx, double* mass, double* M_11, double* M_12, double* M_21, double* M_22) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{

		//if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
		//const double rho_i = rho[i];
		//const double hsml = Hsml[i];
		const double xi = x[i];
		const double yi = y[i];

		double k_11 = 0;//K
		double k_12 = 0;
		double k_21 = 0;
		double k_22 = 0;
		double m_11 = 0;
		double m_12 = 0;
		double m_21 = 0;
		double m_22 = 0;

		//k

		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			if (bweight[i][j] < 0.000000001) continue;
			const double xj = x[jj];
			const double yj = y[jj];
			double dx = xj - xi;
			double dy = yj - yi;
			if (iotype[i] == sph::InoutType::Inlet && iotype[jj] == sph::InoutType::Outlet)
			{
				dx = x[jj] - x[i] - lengthofx;//xi(1)
			}
			if (iotype[i] == sph::InoutType::Outlet && iotype[jj] == sph::InoutType::Inlet)
			{
				dx = x[jj] - x[i] + lengthofx;//xi(1)
			}
			const double rho_j = rho[jj];
			const double massj = mass[jj];
			k_11 += dx * dx * massj / rho_j * bweight[i][j];	//k11
			k_12 += dx * dy * massj / rho_j * bweight[i][j];//k_12=k_21
			k_21 += dy * dx * massj / rho_j * bweight[i][j];
			k_22 += dy * dy * massj / rho_j * bweight[i][j];
		}
		const double det = k_11 * k_22 - k_12 * k_21;
		m_11 = k_22 / det;
		m_12 = -k_12 / det;
		m_21 = -k_21 / det;
		m_22 = k_11 / det;
		M_11[i] = m_11;//k矩阵求逆
		M_12[i] = m_12;
		M_21[i] = m_21;
		M_22[i] = m_22;
	}

}

__global__ void singlestep_boundaryVisc_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
											, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_21, double* m_12, double* m_22\
											, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22, sph::FluidType* fltype\
											, double dp, const double C_s, double* turb11, double* turb12, double* turb21, double* turb22) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		if (btype[i] != sph::BoundaryType::Boundary) continue;
		const double rho_i = rho[i];
		const double hsml = Hsml[i];
		const double xi = x[i];
		const double yi = y[i];

		//--------wMxij-------
		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			if (bweight[i][j] < 0.000000001) continue;
			const double xj = x[jj];
			const double yj = y[jj];
			double dx = xj - xi;
			double dy = yj - yi;
			if (iotype[i] == sph::InoutType::Inlet && iotype[jj] == sph::InoutType::Outlet)
			{
				dx = x[jj] - x[i] - lengthofx;//xi(1)
			}
			if (iotype[i] == sph::InoutType::Outlet && iotype[jj] == sph::InoutType::Inlet)
			{
				dx = x[jj] - x[i] + lengthofx;//xi(1)
			}
			wMxijx[i][j] = (m_11[i] * dx + m_12[i] * dy) * bweight[i][j];
			wMxijy[i][j] = (m_21[i] * dx + m_22[i] * dy) * bweight[i][j];
		}
		//if (particlesa.btype[i] != sph::BoundaryType::Boundary) continue;

		double epsilon_2_11 = 0;
		double epsilon_2_12 = 0;
		double epsilon_2_21 = 0;
		double epsilon_2_22 = 0;     //=dudx11
		//double epsilon_3 = 0;
		double epsilon_dot11 = 0;
		double epsilon_dot12 = 0;
		double epsilon_dot21 = 0;
		double epsilon_dot22 = 0;
		const double p_i = press[i];

		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			if (btype[jj] == sph::BoundaryType::Boundary) continue;
			if (bweight[i][j] < 0.000000001) continue;
			const double xj = x[jj];
			const double yj = y[jj];
			double dx = xj - xi;
			double dy = yj - yi;
			if (iotype[i] == sph::InoutType::Inlet && iotype[jj] == sph::InoutType::Outlet)
			{
				dx = x[jj] - x[i] - lengthofx;//xi(1)
			}
			if (iotype[i] == sph::InoutType::Outlet && iotype[jj] == sph::InoutType::Inlet)
			{
				dx = x[jj] - x[i] + lengthofx;//xi(1)
			}
			const double r2 = dx * dx + dy * dy;
			const double rho_j = rho[jj];
			const double rho_ij = rho_j - rho_i;
			const double diffx = rho_ij * dx / r2;
			const double diffy = rho_ij * dx / r2;
			//const double dvx = particlesa.vx[jj] - ((ii)->bctype == BoundaryConditionType::NoSlip ? particlesa.vx[i] : 0);
			//const double dvy = particlesa.vy[jj] - ((ii)->bctype == BoundaryConditionType::NoSlip ? particlesa.vy[i] : 0);
			const double dvx = vx[jj] - vx[i];
			const double dvy = vy[jj] - vy[i];
			const double massj = mass[jj];
			epsilon_2_11 += dvx * wMxijx[i][j] * massj / rho_j;//du/dx
			epsilon_2_12 += dvx * wMxijy[i][j] * massj / rho_j;//du/dy
			epsilon_2_21 += dvy * wMxijx[i][j] * massj / rho_j;//dv/dx
			epsilon_2_22 += dvy * wMxijy[i][j] * massj / rho_j;//dv/dy			
		}//end circle j

		 //epsilon_second= -1./3.* epsilon_3;//*��λ����

			//epsilon_temp(11)= epsilon_2(11) * particlesa.m_11[i] + epsilon_2(12) * particlesa.m_21[i];
			//epsilon_temp(12)= epsilon_2(11) * particlesa.m_12[i] + epsilon_2(12) * particlesa.m_22[i];
			//epsilon_temp(21)= epsilon_2(21) * particlesa.m_11[i] + epsilon_2(22) * particlesa.m_21[i];
			//epsilon_temp(22)= epsilon_2(21) * particlesa.m_12[i] + epsilon_2(22) * particlesa.m_22[i];

			//epsilon_dot=0.5*(epsilon_temp+transpose(epsilon_temp))+epsilon_second  !�ܵ�epsilon,����������ȣ�û��epsilon_second
		epsilon_dot11 = epsilon_2_11;
		epsilon_dot12 = 0.5 * (epsilon_2_12 + epsilon_2_21);
		epsilon_dot21 = epsilon_dot12;
		epsilon_dot22 = epsilon_2_22;
		//边界粒子的物理黏性项tau： 比较重要！
		tau11[i] = 2. * Viscosity(fltype[i]) * epsilon_dot11;//边界粒子的黏性力
		tau12[i] = 2. * Viscosity(fltype[i]) * epsilon_dot12;
		tau21[i] = 2. * Viscosity(fltype[i]) * epsilon_dot21;
		tau22[i] = 2. * Viscosity(fltype[i]) * epsilon_dot22;

		//double dist = 0;
		//const double y1 = indiameter;//height
		//const double R = sqrt((xi - 0.2 * y1) * (xi - 0.2 * y1) + yi * yi) - 3 * 0.004;//粒子距圆柱的距离
		//const double L1 = 0.5 * y1 - abs(yi);
		////const double L1 = abs(0.5 * y1 - yi);
		//const double L2 = std::min(xi + 0.08, inlen - xi);
		////const double L2 = abs(yi - 0.5 * y1);
		//double temp = std::min(R, L1);//dwall
		//dist = std::min(temp, L2);
		//if (dist < 0) {
		//	dist = 0;
		//}

		const double dvdx11 = epsilon_2_11;          // dudx11=epsilon_dot11
		const double dvdx12 = 0.5 * (epsilon_2_12 + epsilon_2_21);
		const double dvdx21 = dvdx12;
		const double dvdx22 = epsilon_2_22;
		const double s1ij = sqrt(2.0 * (dvdx11 * dvdx11 + dvdx12 * dvdx12 + dvdx21 * dvdx21 + dvdx22 * dvdx22));

		const double mut = dp * dp * C_s * C_s * s1ij;
		//const double kenergy = C_v / C_e * dp * dp * s1ij * s1ij;    //Ksps=K_turb
		const double kenergy = (vx[i] * vx[i] + vy[i] * vy[i]) * 0.5;
		turb11[i] = 2.0 * mut * dvdx11 * rho_i - 2.0 / 3.0 * kenergy * rho_i;//����tao= ( 2*Vt*Sij-2/3Ksps*�����˺��� )*rho_i// �൱��turbulence(:,:,i)
		turb12[i] = 2.0 * mut * dvdx12 * rho_i;
		turb21[i] = 2.0 * mut * dvdx21 * rho_i;
		turb22[i] = 2.0 * mut * dvdx22 * rho_i - 2.0 / 3.0 * kenergy * rho_i;
	}//end circle i
}

__global__ void singlestep_fluidVisc_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, double*press, unsigned int* neibNum\
											, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy\
											, double* m_11, double* m_12, double* m_21, double* m_22, double* vx, double* vy, double* mass, double* divvel\
											, sph::FluidType* fltype, double* tau11, double* tau12, double* tau21, double* tau22, double* Vort, double dp, const double C_s\
											, double* turb11, double* turb12, double* turb21, double* turb22, sph::FixType* ftype, double* drho) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{

		if (btype[i] == sph::BoundaryType::Boundary) continue;
		const double rho_i = rho[i];
		const double hsml = Hsml[i];
		const double xi = x[i];
		const double yi = y[i];

		//---------速度算子----			
		double epsilon_2_11 = 0;
		double epsilon_2_12 = 0;
		double epsilon_2_21 = 0;
		double epsilon_2_22 = 0;     //=dudx11			
		const double p_i = press[i];

		//--------wMxij-------
		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			if (bweight[i][j] < 0.000000001) continue;
			const double xj = x[jj];
			const double yj = y[jj];
			double dx = xj - xi;
			double dy = yj - yi;
			//周期边界
			if (iotype[i] == sph::InoutType::Inlet && iotype[jj] == sph::InoutType::Outlet)
			{
				dx = x[jj] - x[i] - lengthofx;//xi(1)
			}
			if (iotype[i] == sph::InoutType::Outlet && iotype[jj] == sph::InoutType::Inlet)
			{
				dx = x[jj] - x[i] + lengthofx;;//xi(1)
			}

			wMxijx[i][j] = (m_11[i] * dx + m_12[i] * dy) * bweight[i][j];
			wMxijy[i][j] = (m_21[i] * dx + m_22[i] * dy) * bweight[i][j];
		}

		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			if (bweight[i][j] < 0.000000001) continue;
			const double xj = x[jj];
			const double yj = y[jj];
			double dx = xj - xi;
			double dy = yj - yi;
			if (iotype[i] == sph::InoutType::Inlet && iotype[jj] == sph::InoutType::Outlet)
			{
				dx = x[jj] - x[i] - lengthofx;//xi(1)
			}
			if (iotype[i] == sph::InoutType::Outlet && iotype[jj] == sph::InoutType::Inlet)
			{
				dx = x[jj] - x[i] + lengthofx;//xi(1)
			}
			const double r2 = dx * dx + dy * dy;
			const double rho_j = rho[jj];
			const double rho_ij = rho_j - rho_i;
			const double diffx = rho_ij * dx / r2;
			const double diffy = rho_ij * dy / r2;
			const double dvx = vx[jj] - vx[i];
			const double dvy = vy[jj] - vy[i];
			const double massj = mass[jj];
			//---------density---------- 				
			//---------repulsive--------F_ab
			//if (particlesa.fltype[i] != particlesa.fltype[jj]) {
			//	const double r = sqrt(r2);
			//	const double ch = 1.0 - r / hsml;
			//	const double eta = 2.0 - r / hsml;
			//	double f_eta;
			//	if (eta > 2.0) {
			//		f_eta = 0;
			//	}
			//	else if (eta < 2.0 / 3.0) {
			//		f_eta = 2.0 / 3.0;
			//	}
			//	else if (eta <= 1.0) {
			//		f_eta = 2.0 * eta - 1.5 * eta * eta;
			//	}
			//	else {
			//		f_eta = 0.5 * (2.0 - eta) * (2.0 - eta);
			//	}
			//	const double c0 = particlesa.c0[i];
			//	//�໥��������ʽ����F_ab��
			//	replx += -0.01 * c0 * c0 * ch * f_eta * dx / r2 * particlesa.mass[i];
			//	reply += -0.01 * c0 * c0 * ch * f_eta * dy / r2 * particlesa.mass[i];
			//}
			//----------internal force--------								
			epsilon_2_11 += dvx * wMxijx[i][j] * massj / rho_j;//du/dx
			epsilon_2_12 += dvx * wMxijy[i][j] * massj / rho_j;//du/dy
			epsilon_2_21 += dvy * wMxijx[i][j] * massj / rho_j;//dv/dx
			epsilon_2_22 += dvy * wMxijy[i][j] * massj / rho_j;//dv/dy				
		}//end circle j
		//-----------------------------速度散度			
		divvel[i] = epsilon_2_11 + epsilon_2_22;//弱可压缩，散度应该不会很大		
		//epsilon_dot11 = epsilon_2_11 * particlesa.m_11[i] + epsilon_2_12 * particlesa.m_21[i] - 1. / 3. * epsilon_3;
		//epsilon_dot12 = ((epsilon_2_11 * particlesa.m_12[i] + epsilon_2_12 * particlesa.m_22[i]) + (epsilon_2_21 * particlesa.m_11[i] + epsilon_2_22 * particlesa.m_21[i])) * 0.5 + 0;
		//epsilon_dot21 = epsilon_dot12;
		//epsilon_dot22 = epsilon_2_21 * particlesa.m_12[i] + epsilon_2_22 * particlesa.m_22[i] - 1. / 3. * epsilon_3;
		////黏性力tau = 2.0* 动力黏度* 剪切应变率张量
		//particlesa.tau11[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot11;
		//particlesa.tau12[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot12;
		//particlesa.tau21[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot21;
		//particlesa.tau22[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot22;
		//-----------------------------黏性应力
		//对于不可压缩：黏性切应力 = 2. * sph::Fluid::Viscosity(particlesa.fltype[i])* 应变
		//应变为:[ du/dx 0.5*(du/dy+dv/dx) 0.5*(du/dy+dv/dx) dv/dy ]
		tau11[i] = 2. * Viscosity(fltype[i]) * epsilon_2_11;//这里的tau11代表速度的偏导
		tau12[i] = Viscosity(fltype[i]) * (epsilon_2_12 + epsilon_2_21);//
		tau21[i] = tau12[i];//
		tau22[i] = 2. * Viscosity(fltype[i]) * epsilon_2_22;//
		//-----------------------------涡量 W=0.5(dudy-dvdx)    应变率张量（正应力张量）: S=0.5(dudx+dvdy) divvel
		double vort = 0.5 * (epsilon_2_12 - epsilon_2_21);
		//particlesa.vort[i] = (epsilon_2_21 - epsilon_2_12);
		//运用Q准则 Q=0.5*(sqrt(W)-sqrt(S))
		//particlesa.vort[i] =0.5* (vort* vort - (0.5* particlesa.divvel[i])* (0.5 * particlesa.divvel[i]));
		Vort[i] = vort;
		//-----------------------------湍流应力
		//double dist = 0;
		//const double y1 = indiameter;//height
		//const double R = sqrt((xi - 0.2 * y1) * (xi - 0.2 * y1) + yi * yi) - 3 * 0.004;//粒子距圆柱的距离
		//const double L1 = 0.5 * y1 - abs(yi);
		////const double L1 = abs(0.5 * y1 - yi);
		//const double L2 = std::min(xi + 0.08, inlen - xi);
		////const double L2 = abs(yi - 0.5 * y1);
		//double temp = std::min(R, L1);//dwall
		//dist = std::min(temp, L2);
		//if (dist < 0) {
		//	dist = 0;
		//}			
		const double dvdx11 = epsilon_2_11;
		const double dvdx12 = (epsilon_2_21 + epsilon_2_12) * 0.5;
		const double dvdx21 = dvdx12;
		const double dvdx22 = epsilon_2_22;
		const double s1ij = sqrt(2.0 * (dvdx11 * dvdx11 + dvdx12 * dvdx12 + dvdx21 * dvdx21 + dvdx22 * dvdx22));
		//const double mut = std::min(dist * dist * karman * karman, dp * dp * C_s * C_s) * s1ij;//mut=Vt:the turbulence eddy viscosity
		const double mut = dp * dp * C_s * C_s * s1ij;
		//const double kenergy = C_v / C_e * dp * dp * s1ij * s1ij; //Ksps=K_turb
		//对于k的求法，不同的文献有不同的求法。
		const double kenergy = (vx[i] * vx[i] + vy[i] * vy[i]) * 0.5;
		turb11[i] = 2.0 * mut * dvdx11 * rho_i - 2.0 / 3.0 * kenergy * rho_i;//����tao= ( 2*Vt*Sij-2/3Ksps*�����˺��� )*rho_i// �൱��turbulence(:,:,i)
		turb12[i] = 2.0 * mut * dvdx12 * rho_i;
		turb21[i] = 2.0 * mut * dvdx21 * rho_i;
		turb22[i] = 2.0 * mut * dvdx22 * rho_i - 2.0 / 3.0 * kenergy * rho_i;
		//连续性方程，温度方程
		if (ftype[i] != sph::FixType::Fixed) {
			//particlesa.drho[i] = drhodt + drhodiff;
			drho[i] = -rho[i] * divvel[i];	//没有密度耗散项			
		}
		else {
			drho[i] = 0;
		}
		//(ii)->replx = replx;                
		//(ii)->reply = reply;
	}//end circle i

}

__global__ void singlestep_eom_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* press, double* x, double* y, double* C0, double* C\
									, double* mass, unsigned int* neibNum, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx\
									, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
									, double* tau11, double* tau12, double* tau21, double* tau22, double* turb11, double* turb12, double* turb21, double* turb22\
									, double* vx, double* vy, double* Avx, double* Avy, double* fintx, double* finty, sph::FixType* ftype, double* ax, double* ay) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{

		if (btype[i] == sph::BoundaryType::Boundary) continue;
		const double rho_i = rho[i];
		const double hsml = Hsml[i];
		const double p_i = press[i];
		const double xi = x[i];
		const double yi = y[i];
		const double c0 = C0[i];
		const double c = C[i];
		const double massi = mass[i];
		//------kexi_force-------			
		double temp_sigemax = 0;//动量方程中的T 
		double temp_sigemay = 0;//Tj-Ti			
		//----------artificial viscosity---
		double avx = 0;
		double avy = 0;

		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			if (bweight[i][j] < 0.000000001) continue;

			const double xj = x[jj];
			const double yj = y[jj];
			double dx = xj - xi;
			double dy = yj - yi;
			if (iotype[i] == sph::InoutType::Inlet && iotype[jj] == sph::InoutType::Outlet)
			{
				dx = x[jj] - x[i] - lengthofx;//xi(1)
			}
			if (iotype[i] == sph::InoutType::Outlet && iotype[jj] == sph::InoutType::Inlet)
			{
				dx = x[jj] - x[i] + lengthofx;//xi(1)
			}
			const double hsmlj = Hsml[jj];
			const double mhsml = (hsml + hsmlj) / 2;
			const double r2 = dx * dx + dy * dy;
			const double rho_j = rho[jj];
			const double massj = mass[jj];
			const double wMxijxV_iix = wMxijx[i][j] * massj / rho_j;
			const double wMxijyV_iiy = wMxijy[i][j] * massj / rho_j;
			//const double wMxijxV_jjx = (jj)->wMxijx[j] * massj / rho_j;//(jj)->wMxijx[j] 可能错了
			//const double wMxijyV_jjy = (jj)->wMxijy[j] * massj / rho_j;
			const double wMxijxV_jjx = (m_11[jj] * dx + m_12[jj] * dy) * bweight[i][j] * massj / rho_j;
			const double wMxijyV_jjy = (m_21[jj] * dx + m_22[jj] * dy) * bweight[i][j] * massj / rho_j;
			//-----internal_force-----fintx: 压强项（- 压强梯度/rho） + 物理黏性项（黏度 * 速度的拉普拉斯算子/rho）	
			//在计算力的时候，与对速度的运算略有不同；计算力的时候采用的是PD中的态的运算思路，				
			//double temp_ik11 = 0;
			//double temp_ik12 = 0;
			//double temp_ik21 = 0;
			//double temp_ik22 = 0;
			////对于柯西应力F：求和：bweight * (F_I* M_I + F_J* M_J)xij* V_J
			////其中，柯西应力包括有：压强项、物理黏性项、湍流黏性项
			//temp_ik11 = (-p_i + particlesa.tau11[i]) * particlesa.m_11[i] + particlesa.tau12[i] * particlesa.m_21[i] + (-particlesa.press[jj] + particlesa.tau11[jj]) * particlesa.m_11[jj] + particlesa.tau12[jj] * particlesa.m_21[jj];
			//temp_ik12 = (-p_i + particlesa.tau11[i]) * particlesa.m_12[i] + particlesa.tau12[i] * particlesa.m_22[i] + (-particlesa.press[jj] + particlesa.tau11[jj]) * particlesa.m_12[jj] + particlesa.tau12[jj] * particlesa.m_22[jj];
			//temp_ik21 = particlesa.tau21[i] * particlesa.m_11[i] + (-p_i + particlesa.tau22[i]) * particlesa.m_21[i] + particlesa.tau21[jj] * particlesa.m_11[jj] + (-particlesa.press[jj] + particlesa.tau22[jj]) * particlesa.m_21[jj];
			//temp_ik22 = particlesa.tau21[i] * particlesa.m_12[i] + (-p_i + particlesa.tau22[i]) * particlesa.m_22[i] + particlesa.tau21[jj] * particlesa.m_12[jj] + (-particlesa.press[jj] + particlesa.tau22[jj]) * particlesa.m_22[jj];
			//fintx += particlesa.bweight[i][j] * (temp_ik11 * dx + temp_ik12 * dy) * particlesa.mass[jj] / rho_j / rho_i;
			//finty += particlesa.bweight[i][j] * (temp_ik21 * dx + temp_ik22 * dy) * particlesa.mass[jj] / rho_j / rho_i;

			//---------------现用柯西应力，态来表示动量方程---------------
			//sigema_j = [ - p+txx txy tyx p+tyy ]
			//j粒子----------------有边界粒子
			//const double sigema_j11 = -particlesa.press[jj] + particlesa.tau11[jj];//湍流力一样在这里加
			//const double sigema_j12 = particlesa.tau12[jj];
			//const double sigema_j21 = particlesa.tau21[jj];
			//const double sigema_j22 = -particlesa.press[jj] + particlesa.tau22[jj];
			const double sigema_j11 = -press[jj] + tau11[jj] + turb11[jj];//加上湍流应力
			const double sigema_j12 = tau12[jj] + turb12[jj];//particlesa.tau12[jj]=particlesa.tau21[jj];particlesa.turb12[jj]=particlesa.turb21[jj]
			const double sigema_j21 = tau21[jj] + turb21[jj];
			const double sigema_j22 = -press[jj] + tau22[jj] + turb22[jj];
			//i粒子
			/*const double sigema_i11 = -p_i + tau11[i];
			const double sigema_i12 = tau12[i];
			const double sigema_i21 = tau21[i];
			const double sigema_i22 = -p_i + tau22[i];*/
			const double sigema_i11 = -p_i + tau11[i] + turb11[i];
			const double sigema_i12 = tau12[i] + turb12[i];
			const double sigema_i21 = tau21[i] + turb21[i];
			const double sigema_i22 = -p_i + tau22[i] + turb22[i];
			temp_sigemax += (sigema_j11 * wMxijxV_jjx + sigema_j12 * wMxijyV_jjy) + (sigema_i11 * wMxijxV_iix + sigema_j12 * wMxijyV_iiy);
			temp_sigemay += (sigema_j21 * wMxijxV_jjx + sigema_j22 * wMxijyV_jjy) + (sigema_i21 * wMxijxV_iix + sigema_j22 * wMxijyV_iiy);
			//-----turbulence-------湍流力加到柯西应力中去了

			//--------artificial viscosity-------
			const double dvx = vx[jj] - vx[i];    //(Vj-Vi)
			const double dvy = vy[jj] - vy[i];
			double muv = 0;
			double piv = 0;
			const double cj = C[jj];
			const double vr = dvx * dx + dvy * dy;     //(Vj-Vi)(Rj-Ri)
			const double mc = 0.5 * (cj + C[i]);
			const double mrho = 0.5 * (rho_j + rho[i]);
			if (vr < 0) {
				muv = mhsml * vr / (r2 + mhsml * mhsml * 0.01);//FAI_ij
				//piv = (0.5 * muv - 0.5 * mc) * muv / mrho;//beta项-alpha项
				piv = (0.5 * muv - 1.0 * mc) * muv / mrho;
				//piv = (0.5 * muv) * muv / mrho;//只有beta项，加速度会一直很大，停不下来，穿透。
			}
			avx += -massj * piv * wMxijx[i][j];
			avy += -massj * piv * wMxijy[i][j];
		}

		fintx[i] = temp_sigemax / rho_i;
		finty[i] = temp_sigemay / rho_i;
		//(ii)->turbx = turbx;// / rho[i];
		//(ii)->turby = turby;// / rho[i];
		Avx[i] = avx;
		Avy[i] = avy;
		//(ii)->avy = 0;							
		if (ftype[i] != sph::FixType::Fixed)//inlet粒子的加速度场为0，只有初始速度。outlet粒子呢？
		{
			ax[i] = fintx[i] + avx;
			ay[i] = finty[i] + avy;
			//ay[i] = 0;
			//ax[i] = (ii)->fintx + avx + turbx + (ii)->replx;
			//ay[i] = (ii)->finty + avy + turby + (ii)->reply + gravity;				
		}
	}

}

__global__ void run_half3Nshiftc_dev1(unsigned int particleNum, sph::FixType* ftype, double* rho, double* half_rho, double* drho, double dt, double* vx, double* half_vx, double*ax\
									, double* vy, double* half_vy, double* ay, double* vol, double* mass, double* x, double* half_x, double* half_y, double* y, double* ux, double* uy\
									, double* temperature, double* half_temperature, double* temperature_t, sph::ShiftingType stype, unsigned int* neibNum, unsigned int** neiblist\
									, double** bweight, double* Shift_c) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		/* code */
		//particle* ii = particles[i];
		//(*i)->integrationfull(dt);
		//if (particlesa.btype[i] == sph::BoundaryType::Boundary) { particlesa.shift_c[i] = 0; continue; }

		if (ftype[i] != sph::FixType::Fixed) {
			rho[i] = half_rho[i] + drho[i] * dt;//i时刻的密度+(i+0.5*dt)时刻的密度变化率×dt			 
			vx[i] = half_vx[i] + ax[i] * dt;//i时刻的速度+(i+0.5*dt)时刻的加速度×dt
			vy[i] = half_vy[i] + ay[i] * dt;
			vol[i] = mass[i] / rho[i];
			x[i] = half_x[i] + vx[i] * dt;//i时刻的位置+(i+*dt)时刻的速度×dt---------------------
			//y[i] = half_y[i] + vy[i] * dt;
			double Y = half_y[i] + vy[i] * dt;//加个判断，防止冲进边界
			/*if (y > indiameter * 0.5 - dp || y < -indiameter * 0.5 + dp) {
				y = half_y[i];
			}*/
			y[i] = Y;
			ux[i] += vx[i] * dt;
			uy[i] += vy[i] * dt;
			temperature[i] = half_temperature[i] + temperature_t[i] * dt;
		}
		if (stype != sph::ShiftingType::DivC) continue;

		double shift_c = 0;

		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			const double rho_j = rho[jj];
			const double massj = mass[jj];
			shift_c += bweight[i][j] * massj / rho_j;
		}
		Shift_c[i] = shift_c;
	}

}

__global__ void run_shifttype_divc_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, double* shift_c, unsigned int* neibNum, unsigned int** neiblist, double* mass\
										, double* rho, double** dbweightx, double** dbweighty, double* Vx, double* Vy, double shiftingCoe, double dt, double dp, double* Shift_x, double* Shift_y\
										, double* x, double* y, double* ux, double* uy, double* drmax, double* drmax2, int* lock) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		/* code */
		//particle* ii = particles[i];
		//(*i)->shifting(dt);
		if (btype[i] == sph::BoundaryType::Boundary) continue;
		//if (ftype[i] != sph::FixType::Free) continue;
		const double hsml = Hsml[i];
		const double conc = shift_c[i];

		double shift_x = 0;
		double shift_y = 0;



		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			const double mhsml = (hsml + Hsml[jj]) * 0.5;
			const double conc_ij = shift_c[jj] - conc;

			shift_x += conc_ij * mass[jj] / rho[jj] * dbweightx[i][j];
			shift_y += conc_ij * mass[jj] / rho[jj] * dbweighty[i][j];
		}
		const double vx = Vx[i];
		const double vy = Vy[i];
		const double vel = sqrt(vx * vx + vy * vy);
		shift_x *= -2.0 * dp * vel * dt * shiftingCoe;
		shift_y *= -2.0 * dp * vel * dt * shiftingCoe;

		Shift_x[i] = shift_x;
		Shift_y[i] = shift_y;

		x[i] += shift_x;
		y[i] += shift_y;
		ux[i] += shift_x;
		uy[i] += shift_y;

		const double Ux = ux[i];
		const double Uy = uy[i];
		const double disp = sqrt(Ux * Ux + Uy * Uy);

		for (int s = 0; s < 32; s++) {
			if ((blockDim.x * blockIdx.x + threadIdx.x) % 32 != s)
				continue;
			while (atomicCAS(lock, 0, 1) == 0); // 尝试获取锁

			if (disp > *drmax) {

				*drmax2 = *drmax;
				*drmax = disp;
			}
			else if (disp > *drmax2) {
				*drmax2 = disp;
			}

			atomicExch(lock, 0);

		}
	}
}

__global__ void run_shifttype_velc_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, double* rho, double* C0, unsigned int* neibNum, unsigned int** neiblist\
										, double** bweight, const double bweightdx, double* mass, double** dbweightx, double** dbweighty, double* Vx, double* Vy, double dp, double shiftingCoe\
										, double* Shift_x, double* Shift_y, double* x, double* y, double* ux, double* uy, double* drmax, double* drmax2, int* lock) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		/* code */
		//particle* ii = particles[i];
		//(*i)->shifting(dt);
		if (btype[i] == sph::BoundaryType::Boundary) continue;
		//if (ftype[i] != sph::FixType::Free) continue;
		const double hsml = Hsml[i];
		const double rho_i = rho[i];
		const double c0 = C0[i];

		double shift_x = 0;
		double shift_y = 0;

		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			const double frac = bweight[i][j] / bweightdx;
			const double head = 1.0 + 0.2 * pow(frac, 4);
			const double rho_j = rho[jj];
			const double rho_ij = rho_i + rho_j;
			const double mass_j = mass[jj];

			shift_x += head * dbweightx[i][j] * mass_j / rho_ij;
			shift_y += head * dbweighty[i][j] * mass_j / rho_ij;
		}
		const double vx = Vx[i];
		const double vy = Vy[i];
		const double vel = sqrt(vx * vx + vy * vy);
		shift_x *= -8.0 * dp * dp * vel / c0 * shiftingCoe;
		shift_y *= -8.0 * dp * dp * vel / c0 * shiftingCoe;

		Shift_x[i] = shift_x;
		Shift_y[i] = shift_y;

		x[i] += shift_x;
		y[i] += shift_y;
		ux[i] += shift_x;
		uy[i] += shift_y;

		const double Ux = ux[i];
		const double Uy = uy[i];
		const double disp = sqrt(Ux * Ux + Uy * Uy);

		for (int s = 0; s < 32; s++) {
			if ((blockDim.x * blockIdx.x + threadIdx.x) % 32 != s)
				continue;
			while (atomicCAS(lock, 0, 1) == 0);
			{
				if (disp > *drmax) {
					*drmax2 = *drmax;
					*drmax = disp;
				}
				else if (disp > *drmax2) {
					*drmax2 = disp;
				}
			}
			atomicExch(lock, 0);

		}
	}
}

__global__ void density_filter_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, unsigned int* neibNum, unsigned int** neiblist , double* press\
									, double* back_p, double* c0, double* rho0, double* mass, double*rho, double** bweight) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		//(*i)->density_filter();
		//particle* ii = particles[i];
		if (btype[i] == sph::BoundaryType::Boundary) continue;
		double beta0_mls = 0;
		double rhop_sum_mls = 0;
		const double hsml = Hsml[i];


		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			const double rho_j = (press[jj] - back_p[jj]) / c0[jj] / c0[jj] + rho0[jj];
			const double v_j = mass[jj] / rho[jj];
			const double mass_j = rho_j * v_j;
			beta0_mls += bweight[i][j] * v_j;
			rhop_sum_mls += bweight[i][j] * mass_j;
		}

		beta0_mls += func_factor(hsml, 2) * mass[i] / rho[i];
		rhop_sum_mls += func_factor(hsml, 2) * mass[i];
		rho[i] = rhop_sum_mls / beta0_mls;
	}
}








void getdt_dev0(unsigned int particleNum, double* dtmin, double* divvel, double* hsml, sph::FluidType* fltype, double vmax, double* Ax, double* Ay) {

	getdt_dev<<<32,512>>>(particleNum, dtmin, divvel, hsml, fltype, vmax, Ax, Ay);
	CHECK(cudaDeviceSynchronize());
}

void adjustC0_dev0(double* c0,double c,unsigned int particleNum) {
	adjustC0_dev<<<32,512>>>(c0, c, particleNum);
	CHECK(cudaDeviceSynchronize());
}

void inlet_dev0(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx) {
	inlet_dev << <32, 512 >> > (particleNum, x,  iotype, outletBcx);
	CHECK(cudaDeviceSynchronize());
}

void outlet_dev0(unsigned int particleNum, double* x, sph::InoutType* iotype, double outletBcx, double outletBcxEdge, double lengthofx) {
	outlet_dev << <32, 512 >> > (particleNum, x, iotype, outletBcx, outletBcxEdge, lengthofx);
	CHECK(cudaDeviceSynchronize());
}

void buildNeighb_dev01(unsigned int particleNum, double* ux, double* uy, double* X, double* Y, double* X_max, double* X_min, double* Y_max, double* Y_min) {
	buildNeighb_dev1 << <32, 512 >> > (particleNum, ux, uy, X, Y, X_max, X_min, Y_max, Y_min);
	CHECK(cudaDeviceSynchronize());
}

void buildNeighb_dev02(unsigned int particleNum, double* X, double* Y, unsigned int** neiblist, unsigned int* neibNum\
	, const int ngridx, const int ngridy, const double dxrange, const double dyrange, double x_min, double y_min\
	, int* xgcell, int* ygcell, int* celldata, int* grid_d, const double* Hsml, unsigned int* idx, sph::InoutType* iotype, double lengthofx) {
	//目前网格法无法并行，暂不考虑其并行方法，但并行化建议使用全搜索法
	int* lock;
	cudaMallocManaged(&lock,sizeof(int));
	*lock = 0;
	buildNeighb_dev2 << <32, 512 >> > ( particleNum, X, Y, neiblist, neibNum, ngridx, ngridy, dxrange, dyrange, x_min, y_min, xgcell, ygcell, celldata, grid_d, lock);
	//CHECK(cudaDeviceSynchronize());
	buildNeighb_dev3 << <32, 512 >> > (particleNum, X, Y, neiblist, neibNum, Hsml, ngridx, ngridy, dxrange, dyrange, idx, iotype, xgcell, ygcell, celldata, grid_d, lengthofx);
	CHECK(cudaDeviceSynchronize());
	//printf("particle_1's neighberNum is %d\n", neibNum[0]);
	cudaFree(lock);

}

void run_half1_dev0(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
	, double* x, double* y, double* vx, double* vy, double* rho, double* temperature) {
	run_half1_dev1<<<32,512>>>(particleNum, half_x, half_y, half_vx, half_vy, half_rho, half_temperature, x, y, vx, vy, rho, temperature);
	CHECK(cudaDeviceSynchronize());
}

void run_half2_dev0(unsigned int particleNum, double* half_x, double* half_y, double* half_vx, double* half_vy, double* half_rho, double* half_temperature\
					, double* x, double* y, double* vx, double* vy, double* rho, double* temperature\
					, double* drho, double* ax, double* ay, double* vol, double* mass\
					, sph::BoundaryType* btype, sph::FixType* ftype, double* temperature_t, const double dt2, double* vmax) {

	run_half2_dev1 << <32, 512 >> > (particleNum, half_x, half_y, half_vx, half_vy, half_rho, half_temperature\
												, x, y, vx, vy, rho, temperature\
												, drho, ax, ay, vol, mass\
												, btype, ftype, temperature_t, dt2, vmax);
	CHECK(cudaDeviceSynchronize());
}

void singlestep_rhoeos_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* rho_min, double* c0, double* rho0, double* gamma, double* back_p, double* press) {

	singlestep_rhofilter_dev1<<<32,512>>>(particleNum, btype, rho, rho_min);
	singlestep_eos_dev1<<<32,512>>>(particleNum, btype, c0, rho0, rho, gamma, back_p, press);
	//CHECK(cudaDeviceSynchronize());
}

void singlestep_updateWeight_dev0(unsigned int particleNum, unsigned int* neibNum, double* hsml, unsigned int** neiblist, double* x, double* y\
									, sph::InoutType* iotype, double lengthofx, double** bweight, double** dbweightx, double** dbweighty) {

	singlestep_updateWeight_dev1<<<32,512>>>(particleNum, neibNum, hsml, neiblist, x, y, iotype, lengthofx, bweight, dbweightx, dbweighty);
	CHECK(cudaDeviceSynchronize());
}

void singlestep_boundryPNV_dev0(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass\
	, double* rho, double* press, double** bweight, double* vx, double* vy, double* Vcc) {

	singlestep_boundryPNV_dev1<<<32,512>>>(particleNum, btype, neibNum, neiblist, ftype, mass, rho, press, bweight, vx, vy, Vcc);
	//CHECK(cudaDeviceSynchronize());
}

void singlestep_shapeMatrix_dev0(unsigned int particleNum, double* rho, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist, double** bweight\
	, sph::InoutType* iotype, double lengthofx, double* mass, double* M_11, double* M_12, double* M_21, double* M_22) {

	singlestep_shapeMatrix_dev1<<<32,512>>>(particleNum, rho, x, y, neibNum, neiblist, bweight, iotype, lengthofx, mass, M_11, M_12, M_21, M_22);
	//CHECK(cudaDeviceSynchronize());
}

void singlestep_boundaryVisc_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy, double* m_11, double* m_21, double* m_12, double* m_22\
	, double* press, double* vx, double* vy, double* mass, double* tau11, double* tau12, double* tau21, double* tau22, sph::FluidType* fltype\
	, double dp, const double C_s, double* turb11, double* turb12, double* turb21, double* turb22) {

	singlestep_boundaryVisc_dev1<<<32,512>>>(particleNum,btype, rho, Hsml, x, y, neibNum, neiblist, bweight, iotype, lengthofx, wMxijx, wMxijy, m_11, m_21, m_12, m_22\
		, press, vx, vy, mass, tau11, tau12, tau21, tau22, fltype, dp, C_s, turb11, turb12, turb21, turb22);
	//CHECK(cudaDeviceSynchronize());
}

void singlestep_fluidVisc_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, double* press, unsigned int* neibNum\
	, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx, double** wMxijx, double** wMxijy\
	, double* m_11, double* m_12, double* m_21, double* m_22, double* vx, double* vy, double* mass, double* divvel\
	, sph::FluidType* fltype, double* tau11, double* tau12, double* tau21, double* tau22, double* Vort, double dp, const double C_s\
	, double* turb11, double* turb12, double* turb21, double* turb22, sph::FixType* ftype, double* drho) {

	singlestep_fluidVisc_dev1<<<32,512>>>(particleNum,btype, rho, Hsml, x, y, press, neibNum, neiblist, bweight, iotype, lengthofx, wMxijx, wMxijy, m_11, m_12, m_21, m_22\
						, vx, vy, mass, divvel, fltype, tau11, tau12, tau21, tau22, Vort, dp, C_s, turb11, turb12, turb21, turb22, ftype, drho);
	//CHECK(cudaDeviceSynchronize());
}

void singlestep_eom_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* press, double* x, double* y, double* C0, double* C\
	, double* mass, unsigned int* neibNum, unsigned int** neiblist, double** bweight, sph::InoutType* iotype, double lengthofx\
	, double** wMxijx, double** wMxijy, double* m_11, double* m_12, double* m_21, double* m_22\
	, double* tau11, double* tau12, double* tau21, double* tau22, double* turb11, double* turb12, double* turb21, double* turb22\
	, double* vx, double* vy, double* Avx, double* Avy, double* fintx, double* finty, sph::FixType* ftype, double* ax, double* ay) {

	singlestep_eom_dev1<<<32,512>>>(particleNum, btype,  rho,  Hsml,  press,  x,  y,  C0,  C,  mass, neibNum, neiblist,  bweight, iotype, lengthofx, wMxijx, wMxijy,  m_11,  m_12,  m_21,  m_22\
							,  tau11,  tau12,  tau21,  tau22,  turb11,  turb12,  turb21,  turb22,  vx,  vy,  Avx,  Avy,  fintx,  finty, ftype,  ax,  ay);
	CHECK(cudaDeviceSynchronize());
}

void run_half3Nshiftc_dev0(unsigned int particleNum, sph::FixType* ftype, double* rho, double* half_rho, double* drho, double dt, double* vx, double* half_vx, double* ax\
	, double* vy, double* half_vy, double* ay, double* vol, double* mass, double* x, double* half_x, double* half_y, double* y, double* ux, double* uy\
	, double* temperature, double* half_temperature, double* temperature_t, sph::ShiftingType stype, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, double* Shift_c) {

	run_half3Nshiftc_dev1 << <32, 512 >> > (particleNum, ftype, rho, half_rho, drho, dt, vx, half_vx, ax\
		, vy, half_vy, ay, vol, mass, x, half_x, half_y, y, ux, uy\
		, temperature, half_temperature, temperature_t, stype, neibNum, neiblist, bweight, Shift_c);

	CHECK(cudaDeviceSynchronize());
}

void run_shifttype_divc_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, double* shift_c, unsigned int* neibNum, unsigned int** neiblist, double* mass\
	, double* rho, double** dbweightx, double** dbweighty, double* Vx, double* Vy, double shiftingCoe, double dt, double dp, double* Shift_x, double* Shift_y\
	, double* x, double* y, double* ux, double* uy, double* drmax, double* drmax2, int* lock) {

	run_shifttype_divc_dev1<<<32,512>>>(particleNum,btype, Hsml, shift_c, neibNum, neiblist, mass, rho, dbweightx, dbweighty, Vx, Vy\
										, shiftingCoe, dt, dp, Shift_x, Shift_y, x, y, ux, uy, drmax, drmax2, lock);
	CHECK(cudaDeviceSynchronize());
}

void run_shifttype_velc_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, double* rho, double* C0, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, const double bweightdx, double* mass, double** dbweightx, double** dbweighty, double* Vx, double* Vy, double dp, double shiftingCoe\
	, double* Shift_x, double* Shift_y, double* x, double* y, double* ux, double* uy, double* drmax, double* drmax2, int* lock) {

	run_shifttype_velc_dev1<<<32,512>>>(particleNum, btype, Hsml, rho, C0, neibNum, neiblist\
		, bweight, bweightdx, mass, dbweightx, dbweighty, Vx, Vy, dp, shiftingCoe\
		, Shift_x, Shift_y, x, y, ux, uy, drmax, drmax2, lock);
	CHECK(cudaDeviceSynchronize());
}

void density_filter_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* Hsml, unsigned int* neibNum, unsigned int** neiblist, double* press\
	, double* back_p, double* c0, double* rho0, double* mass, double* rho, double** bweight) {
	density_filter_dev1<<<32,512>>>(particleNum, btype, Hsml, neibNum, neiblist, press, back_p, c0, rho0, mass, rho, bweight);
	CHECK(cudaDeviceSynchronize());
}

//修改single_step_temperature_gaojie()
__device__ void inverseMatrix_d(double matrix[][5], double inverse[][5], int size) {
	// 创建增广矩阵
	double augmented[5][2 * 5];
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			augmented[i][j] = matrix[i][j];
			augmented[i][j + size] = (i == j) ? 1 : 0;
		}
	}

	// 对矩阵进行初等行变换操作，将左侧变成单位矩阵
	for (int i = 0; i < size; i++) {
		// 将第i行的第i列元素缩放到1
		double pivot = augmented[i][i];
		for (int j = 0; j < size * 2; j++) {
			augmented[i][j] /= pivot;
		}

		// 将其他行第i列元素变为0
		for (int j = 0; j < size; j++) {
			if (j != i) {
				double factor = augmented[j][i];
				for (int k = 0; k < size * 2; k++) {
					augmented[j][k] -= factor * augmented[i][k];
				}
			}
		}
	}

	// 从增广矩阵中提取逆矩阵
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			inverse[i][j] = augmented[i][j + size];
		}
	}
}


__global__ void single_temp_eos_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* C0, double* Rho0, double* rho, double* Gamma, double* back_p, double* press) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		if (btype[i] == sph::BoundaryType::Boundary) continue;
		const double c0 = C0[i];
		const double rho0 = Rho0[i];
		const double rhoi = rho[i];
		const double gamma = Gamma[i];
		const double b = c0 * c0 * rho0 / gamma;     //b--B,
		const double p_back = back_p[i];
		//			if (particlesa.iotype[i] == InoutType::Inlet) {
		//				// interpolation
		//				double p = 0;
		//				double pmax = DBL_MIN;
		//				double vcc = 0;
		//#ifdef OMP_USE
		//#pragma omp parallel for schedule (guided) reduction(+:p,vcc)
		//#endif
		//				for (int j = 0; j < ii->neiblist.size(); j++)
		//				{
		//					const int jj = particlesa.neiblist[i][j];
		//					if (particlesa.iotype[i] == particlesa.iotype[jj]) continue;
		//					const double mass_j = particlesa.mass[jj];
		//					const double rho_j = particlesa.rho[jj];
		//					const double p_k = particlesa.press[jj];
		//					p += p_k * particlesa.bweight[i][j];
		//					vcc += particlesa.bweight[i][j];					
		//				}
		//				p = vcc > 0.00000001 ? p / vcc : 0;				
		//				particlesa.vcc[i] = vcc;
		//				particlesa.press[i] = p; // inlet部分的压力是做的插值求得，按道理应该略大于右边粒子的压力，才会产生推的作用。
		//				//particlesa.press[i] = 0;			
		//			}
		//			else
					//inlet和outlet的压力应该单独设置
		press[i] = b * (pow(rhoi / rho0, gamma) - 1.0) + p_back;      // 流体区

		if (press[i] < p_back)
			press[i] = p_back;
	}

}

__global__ void single_temp_boundary_dev1(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass, double* rho\
											, double* press, double** bweight, double* vx, double* vy, double* Vcc) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		//particle* ii = particles[i];
		if (btype[i] != sph::BoundaryType::Boundary) continue;
		//if (particlesa.iotype[i] == sph::InoutType::Buffer) continue;			
		double p = 0, vcc = 0;
		double v_x = 0, v_y = 0;

		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			//if (btype[i] == btype[jj]) continue;//Buffer粒子也在这里，导致流出边界固壁的压力不正常
			if (ftype[jj] == sph::FixType::Fixed) continue;//其他固壁粒子不参与，ghost不参与，buffer参与
			const double mass_j = mass[jj];
			const double rho_j = rho[jj];
			const double p_k = press[jj];
			p += p_k * bweight[i][j];//
			vcc += bweight[i][j];
			v_x += vx[jj] * bweight[i][j];//需要将速度沿法线分解，还没分
			v_y += vy[jj] * bweight[i][j];
		}
		p = vcc > 0.00000001 ? p / vcc : 0;//p有值
		v_x = vcc > 0.00000001 ? v_x / vcc : 0;//v_x一直为0！待解决
		v_y = vcc > 0.00000001 ? v_y / vcc : 0;
		double vx0 = 0;
		double vy0 = 0;
		Vcc[i] = vcc;
		press[i] = p;
		vx[i] = 2.0 * vx0 - v_x;//无滑移，要改成径向，切向
		vy[i] = 2.0 * vy0 - v_y;
	}

}

__global__ void single_temp_shapematrix_dev1(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
											, double** bweight, sph::InoutType* iotype, double lengthofx, double* mass, double* m_11, double* m_12, double* m_21, double* m_22, double* M_11\
											, double* M_12, double* M_13, double* M_14, double* M_15, double* M_21, double* M_22, double* M_23, double* M_24, double* M_25\
											, double* M_31, double* M_32, double* M_33, double* M_34, double* M_35, double* M_51, double* M_52, double* M_53, double* M_54, double* M_55) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < particleNum; i += gridDim.x * blockDim.x)
	{
		if (btype[i] == sph::BoundaryType::Boundary) continue;//边界粒子没有计算M矩阵
		const double rho_i = rho[i];
		const double hsml = Hsml[i];
		const double xi = x[i];
		const double yi = y[i];
		double k_11 = 0;//K
		double k_12 = 0;
		double k_21 = 0;
		double k_22 = 0;
		double m_11t = 0;
		double m_12t = 0;
		double m_21t = 0;
		double m_22t = 0;
		double matrix[5][5];//M矩阵
		double inverse[5][5];//逆矩阵			
		int size = 5;
		matrix[0][0] = matrix[0][1] = matrix[0][2] = matrix[0][3] = matrix[0][4] = 0;
		matrix[1][0] = matrix[1][1] = matrix[1][2] = matrix[1][3] = matrix[1][4] = 0;
		matrix[2][0] = matrix[2][1] = matrix[2][2] = matrix[2][3] = matrix[2][4] = 0;
		matrix[3][0] = matrix[3][1] = matrix[3][2] = matrix[3][3] = matrix[3][4] = 0;
		matrix[4][0] = matrix[4][1] = matrix[4][2] = matrix[4][3] = matrix[4][4] = 0;
		//将每个元素的值代入矩阵，创建M矩阵
		double a00 = 0; double a01 = 0; double a02 = 0; double a03 = 0; double a04 = 0;
		double a11 = 0; double a14 = 0;
		double a22 = 0; double a23 = 0; double a24 = 0;
		double a34 = 0; double a44 = 0;


		for (int j = 0; j < neibNum[i]; j++)
		{
			const int jj = neiblist[i][j];
			if (bweight[i][j] < 0.000000001) continue;
			const double xj = x[jj];
			const double yj = y[jj];
			double dx = xj - xi;
			double dy = yj - yi;
			if (iotype[i] == sph::InoutType::Inlet && iotype[jj] == sph::InoutType::Outlet)
			{
				dx = x[jj] - x[i] - lengthofx;//xi(1)
			}
			if (iotype[i] == sph::InoutType::Outlet && iotype[jj] == sph::InoutType::Inlet)
			{
				dx = x[jj] - x[i] + lengthofx;//xi(1)
			}
			const double rho_j = rho[jj];
			const double massj = mass[jj];
			//一阶
			//k_11 += dx * dx * massj / rho_j * bweight[i][j];	//k11
			//k_12 += dx * dy * massj / rho_j * bweight[i][j];//k_12=k_21
			//k_21 += dy * dx * massj / rho_j * bweight[i][j];
			//k_22 += dy * dy * massj / rho_j * bweight[i][j];
			//二阶 matrix[0][0]=k_11
			/*matrix[0][0] += dx * dx * massj / rho_j * bweight[i][j];
			matrix[0][1] += dx * dy * massj / rho_j * bweight[i][j];
			matrix[0][2] += dx * dx * dx * massj / rho_j * bweight[i][j];
			matrix[0][3] += dx * dx * dy * massj / rho_j * bweight[i][j];
			matrix[0][4] += dx * dy * dy * massj / rho_j * bweight[i][j];*/
			a00 += dx * dx * massj / rho_j * bweight[i][j];// = k11
			a01 += dx * dy * massj / rho_j * bweight[i][j];// = k12 = k_21
			a02 += dx * dx * dx * massj / rho_j * bweight[i][j];
			a03 += dx * dx * dy * massj / rho_j * bweight[i][j];
			a04 += dx * dy * dy * massj / rho_j * bweight[i][j];
			/*matrix[1][0] = matrix[0][1];
			matrix[1][1] += dy * dy * massj / rho_j * bweight[i][j];
			matrix[1][2] = matrix[0][3];
			matrix[1][3] = matrix[0][4];
			matrix[1][4] += dy * dy * dy * massj / rho_j * bweight[i][j];*/
			a11 += dy * dy * massj / rho_j * bweight[i][j];// = k22
			a14 += dy * dy * dy * massj / rho_j * bweight[i][j];
			/*matrix[2][0] = matrix[0][2];
			matrix[2][1] = matrix[1][2];
			matrix[2][2] += dx * dx * dx * dx * massj / rho_j * bweight[i][j];
			matrix[2][3] += dx * dx * dx * dy * massj / rho_j * bweight[i][j];
			matrix[2][4] += dx * dx * dy * dy * massj / rho_j * bweight[i][j];*/
			a22 += dx * dx * dx * dx * massj / rho_j * bweight[i][j];
			a23 += dx * dx * dx * dy * massj / rho_j * bweight[i][j];
			a24 += dx * dx * dy * dy * massj / rho_j * bweight[i][j];
			/*matrix[3][0] = matrix[0][3];
			matrix[3][1] = matrix[1][3];
			matrix[3][2] = matrix[2][3];
			matrix[3][3] = matrix[2][4];
			matrix[3][4] += dx * dy * dy * dy * massj / rho_j * bweight[i][j];*/
			a34 += dx * dy * dy * dy * massj / rho_j * bweight[i][j];
			/*matrix[4][0] = matrix[0][4];
			matrix[4][1] = matrix[1][4];
			matrix[4][2] = matrix[2][4];
			matrix[4][3] = matrix[3][4];
			matrix[4][4] += dy * dy * dy * dy * massj / rho_j * bweight[i][j];*/
			a44 += dy * dy * dy * dy * massj / rho_j * bweight[i][j];
			//M矩阵为12个孤立的变量
		}
		matrix[0][0] = a00;
		matrix[0][1] = a01;
		matrix[0][2] = a02;
		matrix[0][3] = a03;
		matrix[0][4] = a04;
		matrix[1][0] = matrix[0][1]; matrix[1][1] = a11; matrix[1][2] = matrix[0][3]; matrix[1][3] = matrix[0][4]; matrix[1][4] = a14;
		matrix[2][0] = matrix[0][2]; matrix[2][1] = matrix[1][2]; matrix[2][2] = a22; matrix[2][3] = a23; matrix[2][4] = a24;
		matrix[3][0] = matrix[0][3]; matrix[3][1] = matrix[1][3]; matrix[3][2] = matrix[2][3]; matrix[3][3] = matrix[2][4]; matrix[3][4] = a34;
		matrix[4][0] = matrix[0][4]; matrix[4][1] = matrix[1][4]; matrix[4][2] = matrix[2][4]; matrix[4][3] = matrix[3][4]; matrix[4][4] = a44;
		//一阶
		//const double det = k_11 * k_22 - k_12 * k_21;
		//const double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[0][1];
		const double det = a00 * a11 - a01 * a01;
		/*m_11 = k_22 / det;
		m_12 = -k_12 / det;
		m_21 = -k_21 / det;
		m_22 = k_11 / det;*/
		m_11t = matrix[1][1] / det;
		m_12t = -matrix[0][1] / det;
		m_21t = m_12t;
		m_22t = matrix[0][0] / det;
		m_11[i] = m_11t;//k矩阵求逆
		m_12[i] = m_12t;
		m_21[i] = m_21t;
		m_22[i] = m_22t;
		// 二阶
		//求逆
		inverseMatrix_d(matrix, inverse, size);
		/*std::cout << "Inverse matrix of particle" <<i<< std::endl;
		displayMatrix(inverse, size);*/
		M_11[i] = inverse[0][0];
		M_12[i] = inverse[0][1];
		M_13[i] = inverse[0][2];
		M_14[i] = inverse[0][3];
		M_15[i] = inverse[0][4];
		M_21[i] = inverse[1][0];
		M_22[i] = inverse[1][1];
		M_23[i] = inverse[1][2];
		M_24[i] = inverse[1][3];
		M_25[i] = inverse[1][4];
		M_31[i] = inverse[2][0];//M的二阶逆矩阵的部分元素
		M_32[i] = inverse[2][1];
		M_33[i] = inverse[2][2];
		M_34[i] = inverse[2][3];
		M_35[i] = inverse[2][4];
		M_51[i] = inverse[4][0];
		M_52[i] = inverse[4][1];
		M_53[i] = inverse[4][2];
		M_54[i] = inverse[4][3];
		M_55[i] = inverse[4][4];
	}//end circle i
}

void single_temp_eos_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* C0, double* Rho0, double* rho, double* Gamma, double* back_p, double* press) {
	single_temp_eos_dev1<<<32,512>>>(particleNum, btype, C0, Rho0, rho, Gamma, back_p, press);
	CHECK(cudaDeviceSynchronize());
}

void single_temp_boundary_dev0(unsigned int particleNum, sph::BoundaryType* btype, unsigned int* neibNum, unsigned int** neiblist, sph::FixType* ftype, double* mass, double* rho\
	, double* press, double** bweight, double* vx, double* vy, double* Vcc) {
	single_temp_boundary_dev1<<<32,512>>>(particleNum, btype, neibNum, neiblist, ftype, mass, rho\
		, press, bweight, vx, vy, Vcc);
	CHECK(cudaDeviceSynchronize());
}

void single_temp_shapematrix_dev0(unsigned int particleNum, sph::BoundaryType* btype, double* rho, double* Hsml, double* x, double* y, unsigned int* neibNum, unsigned int** neiblist\
	, double** bweight, sph::InoutType* iotype, double lengthofx, double* mass, double* m_11, double* m_12, double* m_21, double* m_22, double* M_11\
	, double* M_12, double* M_13, double* M_14, double* M_15, double* M_21, double* M_22, double* M_23, double* M_24, double* M_25\
	, double* M_31, double* M_32, double* M_33, double* M_34, double* M_35, double* M_51, double* M_52, double* M_53, double* M_54, double* M_55) {
	single_temp_shapematrix_dev1 << <32, 512 >> > (particleNum, btype, rho, Hsml, x, y, neibNum, neiblist\
		, bweight, iotype, lengthofx, mass, m_11, m_12, m_21, m_22, M_11\
		, M_12, M_13, M_14, M_15, M_21, M_22, M_23, M_24, M_25\
		, M_31, M_32, M_33, M_34, M_35, M_51, M_52, M_53, M_54, M_55);
	CHECK(cudaDeviceSynchronize());
}