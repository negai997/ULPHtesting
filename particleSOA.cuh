#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include "vector.h"
#include "matrix.h"
#include "wfunc.h"
#include "fluid.h"
#include <iostream>
#include "particle.h"


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


//constexpr auto MAX_NEIB = 50;

namespace sph {
	//const double C_s = 0.15;
	//const double C_v = 0.08;
	//const double C_e = 1.0;

	//const double karman = 0.4;

	//enum class BoundaryType    //0为流体域，1为边界域
	//{
	//	Bulk = 0,
	//	Boundary = 1
	//};

	//enum class BoundaryConditionType
	//{
	//	FreeSlip = 1,
	//	NoSlip = 2
	//};

	//enum class FixType
	//{
	//	Free = 0,
	//	Fixed = 1,
	//	Moving = 2
	//};

	//enum class InoutType
	//{
	//	Fluid = 0,
	//	Inlet = 1,
	//	Outlet = 2,
	//	Buffer = 3,
	//	Ghost = 4
	//};

	class particleSOA
	{
		friend class domain;
	public:
		void initialize(std::vector<class particle*> particles, unsigned int idp);
		//particle(double, double);  //构造函数
		//particle(double, double, double, double, double, double, double, double, double, double, double, double, double, double, double);
		void setvolume(double, unsigned int pid);   //赋值函数（体积）
		void setdensity(double, unsigned int pid);
		void setInitPressure(double, unsigned int pid);
		void setInitSoundSpd(double, unsigned int pid);
		void setIdx(unsigned int _id, unsigned int pid) { idx[pid] = _id; };
		void setVisco(double, unsigned int pid);    //粘度
		void sethsml(double, unsigned int pid);
		void setBtype(BoundaryType, unsigned int pid);
		void setFtype(FixType, unsigned int pid);
		void setFltype(FluidType, unsigned int pid);
		void setIotype(InoutType, unsigned int pid);
		void setP_back(double _p, unsigned int pid) { back_p[pid] = _p; };
		void setDensityMin(unsigned int pid);
		const BoundaryType getBtype(unsigned int pid) { return btype[pid]; };
		const FixType getFtype(unsigned int pid) { return ftype[pid]; };
		const FluidType getFltype(unsigned int pid) { return fltype[pid]; };
		const double getX(unsigned int pid) { return x[pid]; };
		const double getY(unsigned int pid) { return y[pid]; };
		const double getVx(unsigned int pid) { return vx[pid]; };
		const double getVy(unsigned int pid) { return vy[pid]; };
		const double getFx(unsigned int pid) { return fintx[pid]; };
		const double getFy(unsigned int pid) { return finty[pid]; };
		const double getAx(unsigned int pid) { return ax[pid]; };
		const double getAy(unsigned int pid) { return ay[pid]; };
		const double getAvx(unsigned int pid) { return avx[pid]; };
		const double getAvy(unsigned int pid) { return avy[pid]; };
		const double getTx(unsigned int pid) { return turbx[pid]; };
		const double getTy(unsigned int pid) { return turby[pid]; };
		const double getPress(unsigned int pid) { return press[pid]; };
		const double getP_back(unsigned int pid) { return back_p[pid]; };
		const double getInitPressure(unsigned int pid) { return press0[pid]; };
		const double getInitSoundSpd(unsigned int pid) { return c0[pid]; };
		const double getSoundSpd(unsigned int pid) { return c[pid]; };
		const double getInitDensity(unsigned int pid) { return rho0[pid]; };
		const double getMass(unsigned int pid) { return mass[pid]; };
		const double getDensity(unsigned int pid) { return rho[pid]; };
		const double gethsml(unsigned int pid) { return hsml[pid]; };
		const double getShiftc(unsigned int pid) { return shift_c[pid]; };
		const double getDisp(unsigned int pid) { return sqrt(ux[pid] * ux[pid] + uy[pid] * uy[pid]); };
		const double getdt(unsigned int pid);
		const double getvort(unsigned int pid) { return vort[pid]; };
		const double gettemperature(unsigned int pid) { return temperature[pid]; };
		const double gettempx(unsigned int pid) { return temperature_x[pid]; };
		const double gettempy(unsigned int pid) { return temperature_y[pid]; };
		const double gettempt(unsigned int pid) { return temperature_t[pid]; };
		const unsigned int getIdx(unsigned int pid) { return idx[pid]; };
		const math::matrix getTurbM(unsigned int pid) const { return turbmat[pid]; };
		void updateVolume(unsigned int pid) { this->vol[pid] = this->mass[pid] / this->rho[pid]; };
		void storeHalf(unsigned int pid);   //
		void density_filter(unsigned int pid);
		void densityRegulation(unsigned int pid);
		void updatePressure(unsigned int pid);
		void updateDDensity(unsigned int pid);
		void updateRepulsion(unsigned int pid);
		void updateFint(unsigned int pid);
		void getTurbulence(unsigned int pid);
		void getTurbForce(unsigned int pid);
		void getArticial(unsigned int pid);
		void updateAcc(unsigned int pid);
		void integration1sthalf(const double, unsigned int pid);
		void integrationfull(const double, unsigned int pid);
		void shifting_c(unsigned int pid);
		void shifting(const double, unsigned int pid);
		//void clearNeiblist() { neiblist.clear(); };
		//void add2Neiblist(particle* _p) { neiblist.push_back(_p); neibNum = neiblist.size(); };
		void setZeroDisp(unsigned int pid) { ux[pid] = uy[pid] = 0; };
		//~particle();

	protected:
		unsigned int* idx;
		double* x;
		double* y;
		double* ux;
		double* uy;
		double* vx;
		double* vy;
		double* ax;
		double* ay;
		double* drho;
		double* replx;//repulsive x
		double* reply;//repulsive y
		double* fintx;
		double* finty;
		double* turbx;
		double* turby;
		math::matrix* turbmat;
		double* turb11;
		double* turb12;
		double* turb21;
		double* turb22;
		double* avx;
		double* avy;
		double* press0;//initial pressure
		double* press;
		double* back_p;//back pressure
		double* rho;//density
		double* rho0;//initial density
		double* rho_min;//minimal density
		double* vol;//volume
		double* c;//sound speed
		double* c0;//initial sound speed
		double* visco;//viscosity
		double* mass;
		double* hsml;//smooth length
		double* gamma;
		double* specific_heat;//比热容
		double* coefficient_heat;//传热系数
		double* temperature;
		double* temperature_t;   //温度对时间的导数
		double* temperature_x;
		double* temperature_y;
		double* vcc;
		double* shift_c;
		double* shift_x;
		double* shift_y;
		unsigned int* neibNum;
		//std::vector<class particle*> neiblist;
		unsigned int** neiblist;
		double** bweight;
		double** dbweightx;
		double** dbweighty;
		double** wMxijx;
		double** wMxijy;
		double* m_11;//M的一阶逆矩阵的元素
		double* m_12;
		double* m_21;
		double* m_22;
		double* M_11;//M的二阶逆矩阵的部分元素
		double* M_12;
		double* M_13;
		double* M_14;
		double* M_15;
		double* M_21;
		double* M_22;
		double* M_23;
		double* M_24;
		double* M_25;
		double* M_31;
		double* M_32;
		double* M_33;
		double* M_34;
		double* M_35;
		double* M_51;
		double* M_52;
		double* M_53;
		double* M_54;
		double* M_55;
		double* tau11;
		double* tau12;
		double* tau21;
		double* tau22;
		double* vort;//vorticity涡量
		double* divvel;//速度散度
		//std::vector<double*> bweight;
		//std::vector<double*> dbweightx;
		//std::vector<double*> dbweighty;
		BoundaryType* btype;
		BoundaryConditionType* bctype;
		FixType* ftype;
		FluidType* fltype;
		InoutType* iotype;
		// predictor - corrector
		double* half_rho;
		double* half_vx;
		double* half_vy;
		double* half_x;
		double* half_y;
		double* half_temperature;
	};

	inline void particleSOA::initialize(std::vector<class particle*> particles, unsigned int idp)
	{
		//initializing
		cudaMallocManaged(&idx,idp*sizeof(unsigned int));
		cudaMallocManaged(&x, idp * sizeof(double));
		cudaMallocManaged(&y, idp * sizeof(double));
		cudaMallocManaged(&ux, idp * sizeof(double));
		cudaMallocManaged(&uy, idp * sizeof(double));
		cudaMallocManaged(&vx, idp * sizeof(double));
		cudaMallocManaged(&vy, idp * sizeof(double));
		cudaMallocManaged(&ax, idp * sizeof(double));
		cudaMallocManaged(&ay, idp * sizeof(double));
		cudaMallocManaged(&drho, idp * sizeof(double));
		cudaMallocManaged(&replx, idp * sizeof(double));
		cudaMallocManaged(&reply, idp * sizeof(double));
		cudaMallocManaged(&fintx, idp * sizeof(double));
		cudaMallocManaged(&finty, idp * sizeof(double));
		cudaMallocManaged(&turbx, idp * sizeof(double));
		cudaMallocManaged(&turby, idp * sizeof(double));
	

		cudaMallocManaged(&turbmat, idp * sizeof(math::matrix));


		cudaMallocManaged(&turb11, idp * sizeof(double));
		cudaMallocManaged(&turb12, idp * sizeof(double));
		cudaMallocManaged(&turb21, idp * sizeof(double));
		cudaMallocManaged(&turb22, idp * sizeof(double));

		cudaMallocManaged(&avx, idp * sizeof(double));
		cudaMallocManaged(&avy, idp * sizeof(double));
		cudaMallocManaged(&press0, idp * sizeof(double));
		cudaMallocManaged(&press, idp * sizeof(double));
		cudaMallocManaged(&back_p, idp * sizeof(double));
		cudaMallocManaged(&rho, idp * sizeof(double));
		cudaMallocManaged(&rho0, idp * sizeof(double));
		cudaMallocManaged(&rho_min, idp * sizeof(double));
		cudaMallocManaged(&vol, idp * sizeof(double));
		cudaMallocManaged(&c, idp * sizeof(double));
		cudaMallocManaged(&c0, idp * sizeof(double));
		cudaMallocManaged(&visco, idp * sizeof(double));
		cudaMallocManaged(&mass, idp * sizeof(double));
		cudaMallocManaged(&hsml, idp * sizeof(double));
		cudaMallocManaged(&gamma, idp * sizeof(double));
		cudaMallocManaged(&specific_heat, idp * sizeof(double));
		cudaMallocManaged(&coefficient_heat, idp * sizeof(double));
		cudaMallocManaged(&temperature, idp * sizeof(double));
		cudaMallocManaged(&temperature_t, idp * sizeof(double));
		cudaMallocManaged(&temperature_x, idp * sizeof(double));
		cudaMallocManaged(&temperature_y, idp * sizeof(double));
		cudaMallocManaged(&vcc, idp * sizeof(double));
		cudaMallocManaged(&shift_c, idp * sizeof(double));
		cudaMallocManaged(&shift_x, idp * sizeof(double));
		cudaMallocManaged(&shift_y, idp * sizeof(double));

		////std::vector<class particle*> neiblist;
		cudaMallocManaged(&neibNum, idp * sizeof(unsigned int));
		cudaMallocManaged(&neiblist, idp * sizeof(unsigned int*));
		for (int i = 0; i < idp; i++) {
			cudaMallocManaged(&neiblist[i], MAX_NEIB * sizeof(unsigned int));
		}


		cudaMallocManaged(&bweight, idp * sizeof(double*));
		for (int i = 0; i < idp; i++) {
			cudaMallocManaged(&bweight[i], MAX_NEIB * sizeof(double));
		}
		cudaMallocManaged(&dbweightx, idp * sizeof(double*));
		for (int i = 0; i < idp; i++) {
			cudaMallocManaged(&dbweightx[i], MAX_NEIB * sizeof(double));
		}
		cudaMallocManaged(&dbweighty, idp * sizeof(double*));
		for (int i = 0; i < idp; i++) {
			cudaMallocManaged(&dbweighty[i], MAX_NEIB * sizeof(double));
		}
		cudaMallocManaged(&wMxijx, idp * sizeof(double*));
		for (int i = 0; i < idp; i++) {
			cudaMallocManaged(&wMxijx[i], MAX_NEIB * sizeof(double));
		}
		cudaMallocManaged(&wMxijy, idp * sizeof(double*));
		for (int i = 0; i < idp; i++) {
			cudaMallocManaged(&wMxijy[i], MAX_NEIB * sizeof(double));
		}


		cudaMallocManaged(&m_11, idp * sizeof(double));
		cudaMallocManaged(&m_12, idp * sizeof(double));
		cudaMallocManaged(&m_21, idp * sizeof(double));
		cudaMallocManaged(&m_22, idp * sizeof(double));
		cudaMallocManaged(&M_11, idp * sizeof(double));
		cudaMallocManaged(&M_12, idp * sizeof(double));
		cudaMallocManaged(&M_13, idp * sizeof(double));
		cudaMallocManaged(&M_14, idp * sizeof(double));
		cudaMallocManaged(&M_15, idp * sizeof(double));
		cudaMallocManaged(&M_21, idp * sizeof(double));
		cudaMallocManaged(&M_22, idp * sizeof(double));
		cudaMallocManaged(&M_23, idp * sizeof(double));
		cudaMallocManaged(&M_24, idp * sizeof(double));
		cudaMallocManaged(&M_25, idp * sizeof(double));
		cudaMallocManaged(&M_31, idp * sizeof(double));
		cudaMallocManaged(&M_32, idp * sizeof(double));
		cudaMallocManaged(&M_33, idp * sizeof(double));
		cudaMallocManaged(&M_34, idp * sizeof(double));
		cudaMallocManaged(&M_35, idp * sizeof(double));
		cudaMallocManaged(&M_51, idp * sizeof(double));
		cudaMallocManaged(&M_52, idp * sizeof(double));
		cudaMallocManaged(&M_53, idp * sizeof(double));
		cudaMallocManaged(&M_54, idp * sizeof(double));
		cudaMallocManaged(&M_55, idp * sizeof(double));
		cudaMallocManaged(&tau11, idp * sizeof(double));
		cudaMallocManaged(&tau12, idp * sizeof(double));
		cudaMallocManaged(&tau21, idp * sizeof(double));
		cudaMallocManaged(&tau22, idp * sizeof(double));
		cudaMallocManaged(&vort, idp * sizeof(double));
		cudaMallocManaged(&divvel, idp * sizeof(double));


		cudaMallocManaged(&btype, idp * sizeof(BoundaryType));
		cudaMallocManaged(&bctype, idp * sizeof(BoundaryConditionType));
		cudaMallocManaged(&ftype, idp * sizeof(FixType));
		cudaMallocManaged(&fltype, idp * sizeof(FluidType));
		cudaMallocManaged(&iotype, idp * sizeof(InoutType));


		cudaMallocManaged(&half_rho, idp * sizeof(double));
		cudaMallocManaged(&half_vx, idp * sizeof(double));
		cudaMallocManaged(&half_vy, idp * sizeof(double));
		cudaMallocManaged(&half_x, idp * sizeof(double));
		cudaMallocManaged(&half_y, idp * sizeof(double));
		cudaMallocManaged(&half_temperature, idp * sizeof(double));

		//shifting
		for (int i = 0; i < idp; i++) {
			idx[i] = particles[i]->idx;
			x[i] = particles[i]->x;
			y[i] = particles[i]->y;
			ux[i] = particles[i]->ux;
			uy[i] = particles[i]->uy;
			vx[i] = particles[i]->vx;
			vy[i] = particles[i]->vy;
			ax[i] = particles[i]->ax;
			ay[i] = particles[i]->ay;
			drho[i] = particles[i]->drho;
			replx[i] = particles[i]->replx;
			reply[i] = particles[i]->reply;

			fintx[i] = particles[i]->fintx;
			finty[i] = particles[i]->finty;
			turbx[i] = particles[i]->turbx;
			turby[i] = particles[i]->turby;


			turbmat[i] = particles[i]->turbmat;


			turb11[i] = particles[i]->turb11;
			turb12[i] = particles[i]->turb12;
			turb21[i] = particles[i]->turb21;
			turb22[i] = particles[i]->turb22;

			avx[i] = particles[i]->avx;
			avy[i] = particles[i]->avy;
			press0[i] = particles[i]->press0;
			press[i] = particles[i]->press;
			back_p[i] = particles[i]->back_p;
			rho[i] = particles[i]->rho;
			rho0[i] = particles[i]->rho0;
			rho_min[i] = particles[i]->rho_min;
			vol[i] = particles[i]->vol;
			c[i] = particles[i]->c;
			c0[i] = particles[i]->c0;
			visco[i] = particles[i]->visco;
			mass[i] = particles[i]->mass;
			hsml[i] = particles[i]->hsml;
			gamma[i] = particles[i]->gamma;
			specific_heat[i] = particles[i]->specific_heat;
			coefficient_heat[i] = particles[i]->coefficient_heat;
			temperature[i] = particles[i]->temperature;
			temperature_t[i] = particles[i]->temperature_t;
			temperature_x[i] = particles[i]->temperature_x;
			temperature_y[i] = particles[i]->temperature_y;
			vcc[i] = particles[i]->vcc;
			shift_c[i] = particles[i]->shift_c;
			shift_x[i] = particles[i]->shift_x;
			shift_y[i] = particles[i]->shift_y;
			
			////std::vector<class particle*> neiblist;
			neibNum[i] = particles[i]->neibNum;
			for (int i = 0; i < idp; i++) {
				for (int j = 0; j < neibNum[i]; j++)
				neiblist[i][j] = particles[i]->neiblist[j]->idx;
			}


			for (int i = 0; i < idp; i++) {
				for (int j = 0; j < neibNum[i]; j++)
				bweight[i][j] = particles[i]->bweight[j];
			}
			for (int i = 0; i < idp; i++) {
				for (int j = 0; j < neibNum[i]; j++)
					dbweightx[i][j] = particles[i]->dbweightx[j];
			}
			for (int i = 0; i < idp; i++) {
				for (int j = 0; j < neibNum[i]; j++)
					dbweighty[i][j] = particles[i]->dbweighty[j];
			}
			for (int i = 0; i < idp; i++) {
				for (int j = 0; j < neibNum[i]; j++)
					wMxijx[i][j] = particles[i]->wMxijx[j];
			}
			for (int i = 0; i < idp; i++) {
				for (int j = 0; j < neibNum[i]; j++)
					wMxijy[i][j] = particles[i]->wMxijy[j];
			}



			m_11[i] = particles[i]->m_11;
			m_12[i] = particles[i]->m_12;
			m_21[i] = particles[i]->m_21;
			m_22[i] = particles[i]->m_22;
			M_11[i] = particles[i]->M_11;
			M_12[i] = particles[i]->M_12;
			M_13[i] = particles[i]->M_13;
			M_14[i] = particles[i]->M_14;
			M_15[i] = particles[i]->M_15;
			M_21[i] = particles[i]->M_21;
			M_22[i] = particles[i]->M_22;
			M_23[i] = particles[i]->M_23;
			M_24[i] = particles[i]->M_24;
			M_25[i] = particles[i]->M_25;
			M_31[i] = particles[i]->M_31;
			M_32[i] = particles[i]->M_32;
			M_33[i] = particles[i]->M_33;
			M_34[i] = particles[i]->M_34;
			M_35[i] = particles[i]->M_35;
			M_51[i] = particles[i]->M_51;
			M_52[i] = particles[i]->M_52;
			M_53[i] = particles[i]->M_53;
			M_54[i] = particles[i]->M_54;
			M_55[i] = particles[i]->M_55;
			tau11[i] = particles[i]->tau11;
			tau12[i] = particles[i]->tau12;
			tau21[i] = particles[i]->tau21;
			tau22[i] = particles[i]->tau22;
			vort[i] = particles[i]->vort;
			divvel[i] = particles[i]->divvel;
			

			btype[i] = particles[i]->btype;
			bctype[i] = particles[i]->bctype;
			ftype[i] = particles[i]->ftype;
			fltype[i] = particles[i]->fltype;
			iotype[i] = particles[i]->iotype;


			half_rho[i] = particles[i]->half_rho;
			half_vx[i] = particles[i]->half_vx;
			half_vy[i] = particles[i]->half_vy;
			half_x[i] = particles[i]->half_x;
			half_y[i] = particles[i]->half_y;
			half_temperature[i] = particles[i]->half_temperature;
		}
		printf("initialize have been accomplished.\n");
		printf("particlesa.neibList[0]=%d\n", neiblist[0]);
	}

	/*
		particle::particle(double _x, double _y) :x{ _x }, y{ _y }
	{
		vcc = 0;
		press = press0 = 0;
		c = c0 = 0;
		neibNum = 0;
		rho = 1;
		vol = 1;
		rho0 = rho;
		mass = rho * vol;
		gamma = 0;
		ux = uy = 0;
		vx = vy = 0;
		ax = ay = 0;
		drho = 0;
		replx = reply = 0;
		fintx = finty = 0;
		turbx = turby = 0;
		avx = avy = 0;
		shift_c = 0;
		shift_x = shift_y = 0;
		half_rho = rho;
		half_vx = vx;
		half_vy = vy;
		half_x = x;
		half_y = y;
		turb11 = turb12 = turb21 = turb22 = 0;
		m_11 = m_12 = m_21 = m_22 = 0;
		M_31 = M_32 = M_33 = M_34 = M_35 = M_51 = M_52 = M_53 = M_54 = M_55 = 0;
		tau11 = tau12 = tau21 = tau22 = 0;
		temperature_x = temperature_y = 0;
		temperature_t = 0;
		vort = 0;
		divvel = 0;
		bweight = new double[MAX_NEIB];
		dbweightx = new double[MAX_NEIB];
		dbweighty = new double[MAX_NEIB];
		wMxijx = new double[MAX_NEIB];
		wMxijy = new double[MAX_NEIB];
		this->setDensityMin();
	}

	particle::particle(double _x, double _y, double _vx, double _vy, double _p0, double _pb, double _gamma, double _rho, double _vol, double _c0, double _visco, double _hsml, double _specific_heat, double _coefficient_heat, double _temperature) :
		x{ _x }, y{ _y }, vx{ _vx }, vy{ _vy }, press0{ _p0 }, back_p{ _pb }, gamma{ _gamma }, rho{ _rho }, vol{ _vol }, c0{ _c0 }, visco{ _visco }, hsml{ _hsml }, specific_heat{ _specific_heat }, coefficient_heat{ _coefficient_heat }, temperature{ _temperature }
	{
		//temperature = 0;
		vcc = 0;
		mass = rho * vol;
		rho0 = rho;
		c = c0;
		press = press0;
		neibNum = 0;
		replx = reply = 0;
		ux = uy = 0;
		vx = vy = 0;
		ax = ay = 0;
		drho = 0;
		replx = reply = 0;
		fintx = finty = 0;
		turbx = turby = 0;
		avx = avy = 0;
		vort = 0;
		divvel = 0;
		shift_c = 0;
		shift_x = shift_y = 0;
		half_rho = rho;
		half_vx = vx;
		half_vy = vy;
		half_x = x;
		half_y = y;
		turb11 = turb12 = turb21 = turb22 = 0;
		m_11 = m_12 = m_21 = m_22 = 0;
		M_31 = M_32 = M_33 = M_34 = M_35 = M_51 = M_52 = M_53 = M_54 = M_55 = 0;
		tau11 = tau12 = tau21 = tau22 = 0;
		temperature_x = temperature_y = 0;
		temperature_t = 0;
		bweight = new double[MAX_NEIB];
		dbweightx = new double[MAX_NEIB];
		dbweighty = new double[MAX_NEIB];
		wMxijx = new double[MAX_NEIB];
		wMxijy = new double[MAX_NEIB];
		//bctype = BoundaryConditionType::NoSlip;
		this->setDensityMin();
	}
	*/


	inline void particleSOA::setvolume(double _v, unsigned int pid)
	{
		vol[pid] = _v;
	}

	inline void particleSOA::setdensity(double _d, unsigned int pid)
	{
		rho[pid] = _d;
	}

	inline void particleSOA::setInitPressure(double _p, unsigned int pid)
	{
		press0[pid] = _p;
	}

	inline void particleSOA::setInitSoundSpd(double _c0, unsigned int pid)
	{
		c0[pid] = _c0;
	}

	inline void particleSOA::setVisco(double _v, unsigned int pid)
	{
		visco[pid] = _v;
	}

	inline void particleSOA::sethsml(double _hsml, unsigned int pid)
	{
		hsml[pid] = _hsml;
	}

	inline void particleSOA::setBtype(BoundaryType _b, unsigned int pid)
	{
		btype[pid] = _b;
	}

	inline void particleSOA::setFtype(FixType _f, unsigned int pid)
	{
		ftype[pid] = _f;
	}

	inline void particleSOA::setFltype(FluidType _f, unsigned int pid)
	{
		fltype[pid] = _f;
	}

	inline void particleSOA::setIotype(InoutType _i, unsigned int pid)
	{
		iotype[pid] = _i;
	}

	inline void particleSOA::setDensityMin(unsigned int pid)
	{
		rho_min[pid] = rho0[pid] - back_p[pid] / c0[pid] / c0[pid];
	}

	inline const double particleSOA::getdt(unsigned int pid)
	{
		const double alpha_pi = 1.0;
		const double hsml = this->gethsml(pid);
		double divv = 0;

		for (int i = 0; i < neibNum[pid]; i++)
		{
			const math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double r = xi.length();
			const double q = r / hsml;

			math::vector dv(this->getVx(pid) - this->getVx(neiblist[pid][i]), this->getVy(pid) - this->getVy(neiblist[pid][i]));
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			divv += dv * dbweight * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
		}
		const double beta_pi = 10.0 * hsml;
		return 0.3 * hsml / (hsml * divv + c[pid] + 1.2 * (alpha_pi * c[pid] + beta_pi * abs(divv)));
	}

	inline void particleSOA::storeHalf(unsigned int pid)
	{
		half_x[pid] = x[pid];
		half_y[pid] = y[pid];
		half_vx[pid] = vx[pid];
		half_vy[pid] = vy[pid];
		half_rho[pid] = rho[pid];
		half_temperature[pid] = temperature[pid];
	}

	inline void particleSOA::density_filter(unsigned int pid)
	{
		double beta0_mls = 0;
		double rhop_sum_mls = 0;
		const double hsml = this->gethsml(pid);
		for (int i=0; i < neibNum[pid]; i++)
		{
			const math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double r = xi.length();
			const double hsml = this->gethsml(pid);
			const double q = r / hsml;
			const double rho_b = (this->getPress(neiblist[pid][i]) - this->getP_back(pid)) / this->getInitSoundSpd(pid) / this->getInitSoundSpd(pid) + this->getInitDensity(pid);
			const double v_j = this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
			const double mass_b = rho_b * v_j;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);

			if (q >= 0 && q <= 3.0)
			{
				beta0_mls += bweight * v_j;
				rhop_sum_mls += bweight * mass_b;
			}
		}

		beta0_mls += sph::wfunc::factor(hsml, 2) * this->getMass(pid) / this->getDensity(pid);
		rhop_sum_mls += sph::wfunc::factor(hsml, 2) * this->getMass(pid);
		this->setdensity(rhop_sum_mls / beta0_mls, pid);
	}

	inline void particleSOA::densityRegulation(unsigned int pid)
	{
		if (rho[pid] < rho_min[pid])
		{
			rho[pid] = rho_min[pid];
		}
	}

	inline void particleSOA::updatePressure(unsigned int pid)
	{
		if (this->getBtype(pid) == BoundaryType::Boundary)
		{
			press[pid] = 0;
			double vcc = 0;
			for (int i = 0; i < neibNum[pid]; i++)
			{
				const math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
				const double r = xi.length();
				const double hsml = this->gethsml(neiblist[pid][i]);
				const double q = r / hsml;
				const double rho_b = (this->getPress(neiblist[pid][i]) - this->getP_back(pid)) / this->getInitSoundSpd(pid) / this->getInitSoundSpd(pid) + this->getInitDensity(pid);
				const double v_j = this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
				const double mass_b = rho_b * v_j;
				const double bweight = sph::wfunc::bweight(q, hsml, 2);

				press[pid] += this->getPress(neiblist[pid][i]) * bweight * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
				vcc += bweight * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
			}
			press[pid] = vcc > 0.00000001 ? press[pid] / vcc : 0;
			press[pid] = press[pid] > 0.00000001 ? press[pid] : 0;
			rho[pid] = (press[pid] - back_p[pid]) / c0[pid] / c0[pid] + rho0[pid];
		}
		else {
			const double p = c0[pid] * c0[pid] * (rho[pid] - rho0[pid]) + back_p[pid];
			const double b = c0[pid] * c0[pid] * rho0[pid] / gamma[pid];
			press[pid] = b * (pow((rho[pid] / rho0[pid]), gamma[pid]) - 1.0) + back_p[pid];
			c[pid] = sqrt(b * gamma[pid] / rho0[pid]);
			press[pid] = press[pid] < back_p[pid] ? back_p[pid] : press[pid];
		}
	}

	inline void particleSOA::updateDDensity(unsigned int pid)
	{
		// diffusion term
		const double chi = 0.2;

		double drhodt = 0;
		double drhodiff = 0;
		const double rho_i = this->getDensity(pid);
		const double hsml = this->gethsml(pid);

		for (int i = 0; i < neibNum[pid]; i++)
		{
			math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double r = xi.length();
			if (r == 0) {
				std::cerr << "distance between particle " << this->getIdx(pid) << " and " << this->getIdx(neiblist[pid][i]) << " is zero\n";
				std::cerr << "this particle " << this->getIdx(pid) << " : " << this->getX(pid) << '\t' << this->getY(pid) << std::endl;
				std::cerr << "that particle " << this->getIdx(neiblist[pid][i]) << " : " << this->getX(neiblist[pid][i]) << '\t' << this->getY(neiblist[pid][i]) << std::endl;
			}
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;
			const double rho_j = this->getDensity(neiblist[pid][i]);

			const math::vector diff = (rho_j - rho_i) * xi / r2;
			const math::vector diff2((rho_j - rho_i) * xi / r2);
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			math::vector dv(this->getVx(pid) - this->getVx(neiblist[pid][i]), this->getVy(pid) - this->getVy(neiblist[pid][i]));
			const double vcc_rho = dv * dbweight;
			const double vcc_diff = diff * dbweight;
			drhodiff += chi * this->getInitSoundSpd(pid) * this->gethsml(pid) * vcc_diff * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
			if (drhodiff != drhodiff) {
				std::cerr << "nan\n";
			}
			drhodt -= this->getDensity(pid) * vcc_rho * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
		}
		drho[pid] = drhodt + drhodiff;
	}

	inline void particleSOA::updateRepulsion(unsigned int pid)
	{
		// diffusion term
		const double chi = 0.2;

		replx[pid] = reply[pid] = 0;
		if (this->getBtype(pid) == BoundaryType::Boundary) return;

		const double hsml = this->gethsml(pid);

		for (int i = 0; i < neibNum[pid]; i++)
		{
			if (this->getFltype(pid) == this->getFltype(neiblist[pid][i])) continue;

			const math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double r = xi.length();
			if (r >= hsml) continue;
			const double chi = 1.0 - r / hsml;
			const double eta = 2.0 * r / hsml;
			if (eta >= 2.0) continue;
			double f_eta;
			if (eta <= 2.0 / 3.0) {
				f_eta = 2.0 / 3.0;
			}
			else if (eta <= 1.0) {
				f_eta = 2.0 * eta - 1.5 * eta * eta;
			}
			else if (eta < 2.0) {
				f_eta = 0.5 * (2.0 - eta) * (2.0 - eta);
			}
			replx[pid] += 0.01 * c0[pid] * c0[pid] * chi * f_eta * xi.getX() / (r * r) * this->getMass(pid);
			reply[pid] += 0.01 * c0[pid] * c0[pid] * chi * f_eta * xi.getY() / (r * r) * this->getMass(pid);
		}
	}

	inline void particleSOA::updateFint(unsigned int pid)
	{
		fintx[pid] = finty[pid] = 0;
		const double hsml = this->gethsml(pid);

		for (int i = 0; i < neibNum[pid]; i++)
		{
			math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			const double p_j = this->getBtype(neiblist[pid][i]) == BoundaryType::Boundary ? this->getP_back(neiblist[pid][i]) : this->getPress(neiblist[pid][i]);

			fintx[pid] -= (p_j + this->getPress(pid)) / this->getDensity(pid) * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]) * dbweight.getX();//这个地方也要改,xi.getX() = dx;
			//fintx += (ii)->bweight[j] * (temp_ik11 * dx + temp_ik12 * dy) * (jj)->mass / rho_j / rho_i;//是不是要在上面加上四个temp_ik的函数呢
			finty[pid] -= (p_j + this->getPress(pid)) / this->getDensity(pid) * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]) * dbweight.getY();
		}
	}

	inline void particleSOA::getTurbulence(unsigned int pid)
	{
		math::matrix dudx(2);
		math::matrix unit(2);
		unit(1, 1) = unit(2, 2) = 1.0;
		const double hsml = this->gethsml(pid);

		for (int i = 0; i < neibNum[pid]; i++)
		{
			math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;
			math::vector dv(this->getVx(pid) - this->getVx(neiblist[pid][i]), this->getVy(pid) - this->getVy(neiblist[pid][i]));
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			dudx += dyadic(dv, dbweight) * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
		}

		double dist;
		if (this->getX(pid) >= 0.0075 && this->getX(pid) <= 0.19) {
			math::vector xi(0.0, abs(this->getY(pid)) - 0.0125);
			dist = xi.length();
		}
		else if (this->getX(pid) < 0.0075) {
			math::vector xi(0.0075 - this->getX(pid), abs(this->getY(pid)) - 0.0125);
			dist = xi.length();
		}
		else if (this->getX(pid) > 0.19) {
			math::vector xi(0.19 - this->getX(pid), abs(this->getY(pid)) - 0.0125);
			dist = xi.length();
		}

		const math::matrix ell = (dudx + transpose(dudx)) * 0.5;
		const double sij = sqrt(2.0) * ell.norm();
		const double xjian = this->gethsml(pid) / 1.01;
		const double mut = std::min(dist * dist * karman * karman, xjian * xjian * C_s * C_s) * sij;
		const double kenergy = C_v / C_e * xjian * xjian * sij * sij;

		const math::matrix kmatrix = -2.0 / 3.0 * kenergy * unit;
		this->turbmat[pid] = 2.0 * mut * ell + kmatrix * this->getDensity(pid);
	}

	inline void particleSOA::getTurbForce(unsigned int pid)
	{
		turbx[pid] = turby[pid] = 0;

		const double rho_i = this->getDensity(pid);
		const double hsml = this->gethsml(pid);

		for (int i = 0; i < neibNum[pid]; i++)
		{
			math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;

			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			const double rho_j = this->getDensity(neiblist[pid][i]);
			const math::matrix turb_i = this->getTurbM(pid) / rho_i / rho_i;
			const math::matrix turb_j = this->getTurbM(neiblist[pid][i]) / rho_j / rho_j;
			const math::matrix turb_ij = turb_i + turb_j;

			const math::vector t_ij = turb_ij * dbweight;

			turbx[pid] += t_ij(1) * this->getMass(neiblist[pid][i]);
			turby[pid] += t_ij(2) * this->getMass(neiblist[pid][i]);
		}
	}

	inline void particleSOA::getArticial(unsigned int pid)
	{
		avx[pid] = avy[pid] = 0;

		if (this->getFltype(pid) == FluidType::Moisture) return;

		const double rho_i = this->getDensity(pid);
		const double hsml = this->gethsml(pid);
		const double zeta = 2.0 * sph::Fluid::Viscosity(this->getFltype(pid)) * (3.0 + 2.0) / hsml / c0[pid];

		for (int i = 0; i < neibNum[pid]; i++)
		{
			math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double mhsml = (hsml + this->gethsml(neiblist[pid][i])) * 0.5;
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;

			const double rho_j = this->getDensity(neiblist[pid][i]);
			math::vector dv(this->getVx(pid) - this->getVx(neiblist[pid][i]), this->getVy(pid) - this->getVy(neiblist[pid][i]));
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			const double vr = dv * xi;
			const double xjian = this->gethsml(pid) / 1.01;

			avx[pid] += zeta * hsml * this->getMass(neiblist[pid][i]) * (this->getSoundSpd(pid) + this->getSoundSpd(neiblist[pid][i])) * vr / ((rho_i + rho_j) * (r2 + 0.01 * xjian * xjian)) * dbweight(1);
			avy[pid] += zeta * hsml * this->getMass(neiblist[pid][i]) * (this->getSoundSpd(pid) + this->getSoundSpd(neiblist[pid][i])) * vr / ((rho_i + rho_j) * (r2 + 0.01 * xjian * xjian)) * dbweight(2);
		}
	}

	inline void particleSOA::updateAcc(unsigned int pid)
	{
		ax[pid] = fintx[pid] + avx[pid] + turbx[pid] + replx[pid];
		ay[pid] = finty[pid] + avy[pid] + turby[pid] + reply[pid];
	}

	inline void particleSOA::integration1sthalf(const double _dt2, unsigned int pid)
	{
		if (this->getBtype(pid) == BoundaryType::Boundary) return;

		const double dt2 = _dt2;
		if (this->getFtype(pid) != FixType::Fixed) {
			rho[pid] = half_rho[pid] + drho[pid] * dt2;
			vx[pid] = half_vx[pid] + ax[pid] * dt2;
			vy[pid] = half_vy[pid] + ay[pid] * dt2;
		}
		x[pid] = half_x[pid] + vx[pid] * dt2;
		y[pid] = half_y[pid] + vy[pid] * dt2;

		this->updateVolume(pid);
	}

	inline void particleSOA::integrationfull(const double _dt, unsigned int pid)
	{
		if (this->getBtype(pid) == BoundaryType::Boundary) return;

		const double dt = _dt;
		if (this->getFtype(pid) == FixType::Free)
		{
			rho[pid] = half_rho[pid] + drho[pid] * dt;
			vx[pid] = half_vx[pid] + ax[pid] * dt;
			vy[pid] = half_vy[pid] + ay[pid] * dt;
			ux[pid] += vx[pid] * dt;
			uy[pid] += vy[pid] * dt;
			this->updateVolume(pid);
		}
	}

	inline void particleSOA::shifting_c(unsigned int pid)
	{
		shift_c[pid] = 0;
		const double hsml = this->gethsml(pid);
		if (this->getBtype(pid) == BoundaryType::Boundary) return;

		for (int i = 0; i < neibNum[pid]; i++)
		{
			math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double mhsml = (hsml + this->gethsml(neiblist[pid][i])) * 0.5;
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;

			shift_c[pid] += bweight * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]);
		}
	}

	inline void particleSOA::shifting(const double dt, unsigned int pid)
	{
		shift_x[pid] = shift_y[pid] = 0;
		const double hsml = this->gethsml(pid);

		if (this->getBtype(pid) == BoundaryType::Boundary) return;
		if (this->getFtype(pid) != FixType::Free) return;

		math::vector dr(0.0, 0.0);

		for (int i = 0; i < neibNum[pid]; i++)
		{
			math::vector xi(this->getX(pid) - this->getX(neiblist[pid][i]), this->getY(pid) - this->getY(neiblist[pid][i]));
			const double mhsml = (hsml + this->gethsml(neiblist[pid][i])) * 0.5;
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;

			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);
			const double conc_ij = this->getShiftc(neiblist[pid][i]) - this->getShiftc(pid);

			dr += conc_ij * this->getMass(neiblist[pid][i]) / this->getDensity(neiblist[pid][i]) * dbweight;
		}

		const double xjian = this->gethsml(pid) / 1.01;
		const double vel = sqrt(vx[pid] * vx[pid] + vy[pid] * vy[pid]);
		dr = dr * 2.0 * xjian * vel * dt;

		shift_x[pid] = -dr(1);
		shift_y[pid] = -dr(2);

		x[pid] += shift_x[pid];
		y[pid] += shift_y[pid];
		ux[pid] += shift_x[pid];
		uy[pid] += shift_y[pid];
	}

	/*
	particle::~particle()
	{
		if (bweight) free(bweight);
		if (dbweightx) free(dbweightx);
		if (dbweighty) free(dbweighty);
		neiblist.clear();
		//bweight.clear();
	}
	*/
	

}

