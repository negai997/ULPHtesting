#pragma once

#include "particle.h"         
#include "fluid.h"            
#include "vector.h"          
#include "ProgressBar.hpp"
#include "inverseMatrix.h"
#include "dev_function.cuh"
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <ctime>
#include <map>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include "particleSOA.cuh"


#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define _WIN_
#define NOMINMAX
#define TICKS_PER_SEC CLOCKS_PER_SEC
#include "windows.h"
#else
#define _LINUX_
#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>
#define TICKS_PER_SEC CLOCKS_PER_SEC
//sysconf(_SC_CLK_TCK)
#endif

#define abs(N) ((N<0)?(-N):(N))

const double exp9 = exp(-9.0);

//����aph�ռ�
namespace sph {
	const int xgridmax = 600;
	const int ygridmax = 600;
	const double gravity = 0.0;
	const bool time_measure = false;
	const double angle = 0;
	const int inbcThick = 3;
	const double temperature_min = 10;
	const double temperature_max = 100;

	//�涨������0����1��
	enum class Direction
	{
		Left = 0,
		Right = 1
	};

	//None = 0,DivC = 1,Velc = 2
	enum class ShiftingType
	{
		None = 0,
		DivC = 1,
		Velc = 2
	};

	//���� ��domain
	class domain
	{
	public:
		domain();
		void createDomain0(double, double, double, double, double);
		void createDomain(double, double, double, double, double);
		void createDomain1(double, double, double, double, double);
		void createDomain2(double, double, double, double, double);
		void writeInitial();
		void solve(unsigned int);
		const double getdt() const;
		//const double getdp() const;
		bool buildNeighb(bool Forced = false);
		bool buildNeighb0(bool Forced = false);
		void updateWeight();
		void density_filter();
		void temperature_filter();
		void iterate(double);
		void predictor(double);
		void corrector(double);
		void output(unsigned int);
		void outputfluid(unsigned int);
		void output_one(unsigned int);
		void output_middle(unsigned int);
		void screen(unsigned int);
		void setOutIntval(unsigned int);
		void setOutScreen(unsigned int);
		void setOutlet(double, Direction, int);
		void run(double);
		void single_step_temperature_gaojie();
		void single_step_temperature();
		void single_step();		
		void usingProcessbar(bool);
		int getConsoleWidth();
		void applyVelBc();
		void applyVelBc(const double);
		void setInitVel(const double);
		void debugNeib(const int);
		void setParallel(bool);
		void setnbskin(double);
		void adjustC0();
		//void updateMovingPts();
		void setp_back(double _p) { p_back = _p; };
		bool inlet();
		bool outlet();
		bool buffoutlet(double);
		void updateGhostPos();
		void updateGhostInfo();
		bool updateBufferPos(double);
		bool checkFluid2Buffer();
		void Ghost2bufferInfo();
		void readinVel(std::string);
		void setShifting(ShiftingType);
		void setShiftingCoe(double);
		void setRatio(double _r) { randRatio = _r; };
		unsigned int particleNum() { return static_cast<unsigned int>(particles.size()); };
		const double getPressmax();
		~domain();

	private:
		double dp=0;//particle distance
		double p0=0;
		unsigned int istep=0;
		unsigned int total=0;
		unsigned int idp=0;
		double inlen=0;
		double indiameter=0;
		double drmax=0;
		double drmax2=0;
		double time=0;
		double vel=0;
		bool nb_updated=0;
		double vmax=0;
		unsigned int nbBuilt=0;
		unsigned int outIntval=0;
		unsigned int outScreen=0;
		double outletBcx=0;
		double outletBcxEdge=0;
		double lengthofx=0;
		Direction outletd;
		int outletlayer=0;
		double p_back=0;
		double pmax = 0;
		bool usingProgressbar=false;
		double neibskin=0;
		bool parallel=false;
		ShiftingType stype = ShiftingType::None;
		double randRatio=0;
		double shiftingCoe=0;
		particle* beacon;
		sph::particleSOA particlesa;
		std::vector<class particle*> particles;
		std::vector<class particle*> outletline;
		std::vector<class particle*> buffer;
		std::vector<class particle*> ghosts;
		std::map<class particle*, class particle*> ghost2buffer;
		std::map<class particle*, class particle*> buffer2ghost;
		std::map<double, double> velprofile;
	};

	domain::domain() :dp(0), p_back(0), istep(1), total(0), parallel(false), neibskin(0.1), randRatio(0), stype(ShiftingType::None)
	{
#ifdef _WIN_
		std::cout << "running in windows system\n";
		std::cout << "ticks per sec " << TICKS_PER_SEC << std::endl;
#endif
#ifdef _LINUX_
		std::cout << "running in linux system\n";
		std::cout << "ticks per sec " << TICKS_PER_SEC << std::endl;
#endif
	}
	//单圆柱绕流
	void domain::createDomain0(double _dp, double _xlen, double _ylen, double _inlen, double _indiameter)
	{
		std::cout << "Creating domain\n";
		const std::clock_t begin = std::clock();
		dp = _dp;
		//std::cout << std::endl << "dp=" << dp << std::endl;
		p0 = p_back;
		time = 0;
		const double radius = _indiameter * 0.5;
		const double x1 = _xlen;
		const double y1 = _ylen;
		const double xjian = _dp;
		const double inlen = _inlen;		
		this->inlen = _inlen;//将inlen改为流入区
		this->indiameter = _indiameter;		
		// inlet boundary   
		for (int i = 0; i < int(inlen / xjian); i++)
		{
			for (int j = int(radius / xjian); j < int(radius / xjian) + inbcThick; j++)
			{
				// upper half
				particle* p = new sph::particle(-(i + 1) * xjian * cos(angle) - (j + 1) * xjian * sin(angle), -i * xjian * sin(angle) + (j + 1) * xjian * cos(angle),
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01,
					Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);//在创建计算域的时候创建粒子的id
				p->setDensityMin();
				particles.push_back(p);

				// lower half   -3\-2\-1
				p = new sph::particle(-(i + 1) * xjian * cos(angle) + (j + 1) * xjian * sin(angle), -i * xjian * sin(angle) - (j + 1) * xjian * cos(angle),//�Գ�
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}
		//inlet boundary end

		const std::vector<class particle*>::iterator inlet_end = particles.end();

		//inlet flow,不包含x=0的点，
		for (int i = 0; i < int(inlen / xjian); i++)
		{
			for (int j = -int(radius / xjian); j < int(radius / xjian) + 1; j++)
			{
				particle* p = new sph::particle(-(i + 1) * xjian * cos(angle) - j * xjian * sin(angle), -i * xjian * sin(angle) + j * xjian * cos(angle),
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Bulk);
				p->setFtype(FixType::Free);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);//在给流入区赋初速度时，确定流入层粒子
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}

		const class particle* inflow_end = particles.back();

		// top boundary 1\2\3
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = 0; j < 3; j++)
			{
				particle* p = new sph::particle((i)*xjian, (j + 1) * xjian + y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		// top end

		// bottom boundary//\-3\-1\-1
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = -3; j < 0; j++)
			{
				particle* p = new sph::particle((i)*xjian, (j)*xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		// bottom end

		//cylinder  记录每个粒子的圆心角
		const double r0 = 50.0 * xjian;//圆柱半径
		const double cylinder_x = 3.0 * r0;//圆心的坐标x
		//const double cylinder_x2 = 0.03 - 4 * xjian;
		const double cylinder_y1 = y1 * 0.5;//圆心的坐标y: cylinder_y1 - y1 * 0.5
		for (int i = 0; i < 4; i++) {
			const double r = -i * xjian + r0;
			const double l = 2.0 * math::PI * r;
			const unsigned n = l / xjian;
			const double dj = 2.0 * math::PI / n;
			//const double cylinder_x = 0.005;
			//std::cerr << std::endl<<"圆心位置(outletBcx * 0.5)" << (outletBcx * 0.5) << std::endl;
			for (double j = 0; j <= 2.0 * math::PI - 0.5 * dj; j = j + dj)//j就是圆心角
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x, r * sin(j) + cylinder_y1 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				//p->sita = j;
				particles.push_back(p);
			}
		}
		//cylinder end
		// air
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = 0; j < int(y1 / xjian) + 1; j++)//y1范围：0-0.06
			{
				double dx = (i)*xjian - cylinder_x;//减去圆心位置
				//double dx2 = (i)*xjian - cylinder_x2;
				double dy1 = (j)*xjian - cylinder_y1;
				double r2_1 = dx * dx + dy1 * dy1;
				const double rmax = (r0 + 0.5 * xjian) * (r0 + 0.5 * xjian);
				//if (r2_1 > rmax && r2_2 > rmax && r2_3 > rmax && r2_4 > rmax && r2_5 > rmax && r2_6 > rmax && r2_7 > rmax && r2_8 > rmax && r2_9 > rmax)
				if (r2_1 > rmax)
				{
					particle* p = new sph::particle((i)*xjian, (j)*xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
						Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
					p->setBtype(BoundaryType::Bulk);
					p->setFtype(FixType::Free);
					p->setFltype(FluidType::Water);
					p->setIotype(InoutType::Fluid);
					p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
					particles.push_back(p);
				}
			}
		// air end

		idp = particles.size();
		const std::clock_t end = std::clock();
		std::cout << particles.size() << " particles created in " << double(end - begin) / TICKS_PER_SEC << "s\n";
	}
	//Couette flow
	void domain::createDomain1(double _dp, double _xlen, double _ylen, double _inlen, double _indiameter)
	{
		std::cout << "Creating domain\n";
		const std::clock_t begin = std::clock();
		dp = _dp;
		//std::cout << std::endl << "dp=" << dp << std::endl;
		p0 = p_back;
		time = 0;
		const double radius = _indiameter * 0.5;
		const double x1 = _xlen;
		const double y1 = _ylen;
		const double xjian = _dp;
		const double inlen = _inlen;
		this->inlen = _inlen;//将inlen改为流入区
		this->indiameter = _indiameter;
		const double v_boundary = 0.05;
		// inlet boundary   
		for (int i = 0; i < int(inlen / xjian); i++)
		{
			for (int j = int(radius / xjian); j < int(radius / xjian) + inbcThick; j++)
			{
				// upper half
				particle* p = new sph::particle(-(i + 1) * xjian * cos(angle) - (j + 1) * xjian * sin(angle), -i * xjian * sin(angle) + (j + 1) * xjian * cos(angle),
					v_boundary, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01,
					Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);//在创建计算域的时候创建粒子的id
				p->setDensityMin();
				p->vx = v_boundary;
				particles.push_back(p);

				// lower half   -3\-2\-1
				p = new sph::particle(-(i + 1) * xjian * cos(angle) + (j + 1) * xjian * sin(angle), -i * xjian * sin(angle) - (j + 1) * xjian * cos(angle),//�Գ�
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}
		//inlet boundary end

		const std::vector<class particle*>::iterator inlet_end = particles.end();

		//inlet flow,不包含x=0的点，
		for (int i = 0; i < int(inlen / xjian); i++)
		{
			for (int j = -int(radius / xjian); j < int(radius / xjian) + 1; j++)
			{
				particle* p = new sph::particle(-(i + 1) * xjian * cos(angle) - j * xjian * sin(angle), -i * xjian * sin(angle) + j * xjian * cos(angle),
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Bulk);
				p->setFtype(FixType::Free);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);//在给流入区赋初速度时，确定流入层粒子
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}

		const class particle* inflow_end = particles.back();

		// top boundary 1\2\3
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = 0; j < 3; j++)
			{
				particle* p = new sph::particle((i)*xjian, (j + 1) * xjian + y1 * 0.5, v_boundary, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				p->vx = v_boundary;
				particles.push_back(p);
			}
		// top end

		// bottom boundary//\-3\-1\-1
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = -3; j < 0; j++)
			{
				particle* p = new sph::particle((i)*xjian, (j)*xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		// bottom end
		
		// air
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = 0; j < int(y1 / xjian) + 1; j++)//y1范围：0-0.06
			{				
				{
					particle* p = new sph::particle((i)*xjian, (j)*xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
						Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
					p->setBtype(BoundaryType::Bulk);
					p->setFtype(FixType::Free);
					p->setFltype(FluidType::Water);
					p->setIotype(InoutType::Fluid);
					p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
					particles.push_back(p);
				}
			}
		// air end

		idp = particles.size();
		//particleSOA_transfer
		particlesa.initialize(particles, idp);

		const std::clock_t end = std::clock();
		std::cout << particles.size() << " particles created in " << double(end - begin) / TICKS_PER_SEC << "s\n";
	}
	//散热片，散热
	void domain::createDomain2(double _dp, double _xlen, double _ylen, double _inlen, double _indiameter)
	{
		std::cout << "Creating domain\n";
		const std::clock_t begin = std::clock();
		dp = _dp;
		//std::cout << std::endl << "dp=" << dp << std::endl;
		p0 = p_back;
		time = 0;
		const double radius = _indiameter * 0.5;
		const double x1 = _xlen;
		const double y1 = _ylen;
		const double xjian = _dp;
		const double inlen = _inlen;
		this->inlen = _inlen;//将inlen改为流入区
		this->indiameter = _indiameter;
		// inlet boundary   
		for (int i = 0; i < int(inlen / xjian); i++)
		{
			for (int j = int(radius / xjian); j < int(radius / xjian) + inbcThick; j++)
			{
				// upper half
				particle* p = new sph::particle(-(i + 1) * xjian * cos(angle) - (j + 1) * xjian * sin(angle), -i * xjian * sin(angle) + (j + 1) * xjian * cos(angle),
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01,
					Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);//在创建计算域的时候创建粒子的id
				p->setDensityMin();
				particles.push_back(p);

				// lower half   -3\-2\-1
				p = new sph::particle(-(i + 1) * xjian * cos(angle) + (j + 1) * xjian * sin(angle), -i * xjian * sin(angle) - (j + 1) * xjian * cos(angle),//�Գ�
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}
		//inlet boundary end

		const std::vector<class particle*>::iterator inlet_end = particles.end();

		//inlet flow,不包含x=0的点，
		for (int i = 0; i < int(inlen / xjian); i++)
		{
			for (int j = -int(radius / xjian); j < int(radius / xjian) + 1; j++)
			{
				particle* p = new sph::particle(-(i + 1) * xjian * cos(angle) - j * xjian * sin(angle), -i * xjian * sin(angle) + j * xjian * cos(angle),
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Bulk);
				p->setFtype(FixType::Free);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);//在给流入区赋初速度时，确定流入层粒子
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}

		const class particle* inflow_end = particles.back();

		// top boundary 1\2\3
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = 0; j < 3; j++)
			{
				particle* p = new sph::particle((i)*xjian, (j + 1) * xjian + y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		// top end

		// bottom boundary//\-3\-1\-1
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = -3; j < 0; j++)
			{
				particle* p = new sph::particle((i)*xjian, (j)*xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		// bottom end
		//加热板
		const double L1 = 0.08;//加热板的尺寸和位置
		const int L2 = int(0.08 / xjian / 2);
		const double px = 0.2;
		for (int i = 0; i < int(L1 / xjian) + 1; i++)
			for (int j = - L2; j < L2 + 1; j++)
			{
				particle* p = new sph::particle((i)*xjian + px, (j)*xjian , 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		// air
		for (int i = 0; i < int(x1 / xjian) + 1; i++)
			for (int j = 0; j < int(y1 / xjian) + 1; j++)//y1范围：0-0.06
			{
				{
					//除去散热片
					if (i> int(px / xjian)-1 &&i<int((L1+px) / xjian )+1&& j > -L2-1+0.5* int(y1 / xjian) && j< L2+1 + 0.5 * int(y1 / xjian)) continue;
					particle* p = new sph::particle((i)*xjian, (j)*xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
						Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
					p->setBtype(BoundaryType::Bulk);
					p->setFtype(FixType::Free);
					p->setFltype(FluidType::Water);
					p->setIotype(InoutType::Fluid);
					p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
					particles.push_back(p);
				}
			}
		// air end

		idp = particles.size();
		const std::clock_t end = std::clock();
		std::cout << particles.size() << " particles created in " << double(end - begin) / TICKS_PER_SEC << "s\n";
	}
	//多圆柱，圆管模型
	void domain::createDomain(double _dp, double _xlen, double _ylen, double _inlen, double _indiameter)
	{
		std::cout << "Creating domain\n";
		const std::clock_t begin = std::clock();
		dp = _dp;
		//std::cout << std::endl << "dp=" << dp << std::endl;
		p0 = p_back;
		time = 0;
		const double radius = _indiameter * 0.5;
		const double x1 = _xlen;
		const double y1 = _ylen;
		const double xjian = _dp;
		const double inlen = _inlen;
		//const double t00 = 30; //
		//const double t00 = 100;//边界温度
		//double t01 = temperature_min;  //
		this->inlen = _inlen;//将inlen改为流入区
		this->indiameter = _indiameter;
		//std::cout << dp\n;
		// inlet boundary   
		for (int i = 0; i < int(inlen / xjian); i++)
		{
			for (int j = int(radius / xjian); j < int(radius / xjian) + inbcThick; j++)
			{
				// upper half
				particle* p = new sph::particle(-(i+1) * xjian * cos(angle) - (j + 1) * xjian * sin(angle), -i * xjian * sin(angle) + (j + 1) * xjian * cos(angle),
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01,
					Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);//在创建计算域的时候创建粒子的id
				p->setDensityMin();
				particles.push_back(p);

				// lower half   -3\-2\-1
				p = new sph::particle(-(i + 1) * xjian * cos(angle) + (j + 1) * xjian * sin(angle), -i * xjian * sin(angle) - (j + 1) * xjian * cos(angle),//�Գ�
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}
		//inlet boundary end

		const std::vector<class particle*>::iterator inlet_end = particles.end();

		//inlet flow,不包含x=0的点，
		for (int i = 0; i < int(inlen / xjian); i++)
		{
			for (int j = -int(radius / xjian); j < int(radius / xjian) + 1; j++)
			{
				particle* p = new sph::particle(-(i + 1) * xjian * cos(angle) - j * xjian * sin(angle), -i * xjian * sin(angle) + j * xjian * cos(angle),
					0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Bulk);
				p->setFtype(FixType::Free);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Inlet);//在给流入区赋初速度时，确定流入层粒子
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}

		const class particle* inflow_end = particles.back();

		// left boundary
		//for (int i = 1; i < 4; i++)
		//	for (int j = -4; j < int(y1 / xjian) + 3; j++)//-3��3
		//	{
		//		bool found = false;
		//		particle* p = new sph::particle((i - 3) * xjian, (j + 1) * xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
		//			Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
		//			, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
		//		p->setBtype(BoundaryType::Boundary);
		//		p->setFtype(FixType::Fixed);
		//		p->setFltype(FluidType::Water);
		//		p->setIotype(InoutType::Fluid);
		//		p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
		//		//check position with existing particles
		//		for (std::vector<class particle*>::iterator k = particles.begin(); (*k) != inflow_end; k++)//ɾ���ظ��ı߽�
		//		{
		//			math::vector xi(p->getX() - (*k)->getX(), p->getY() - (*k)->getY());
		//			if (xi.length() < xjian * 0.3)
		//			{
		//				found = true;
		//				break;
		//			}
		//		}
		//		if (found)
		//		{
		//			free(p);
		//		}
		//		else {
		//			particles.push_back(p);
		//		}
		//	}
		// left end

		// top boundary 1\2\3
		for (int i = 0; i < int(x1 / xjian)+1; i++)
			for (int j = 0; j < 3; j++)
			{
				particle* p = new sph::particle((i) * xjian, (j + 1) * xjian + y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		// top end

		// bottom boundary//\-3\-1\-1
		for (int i = 0; i < int(x1 / xjian)+1 ; i++)
			for (int j = -3; j < 0; j++)
			{
				particle* p = new sph::particle((i) * xjian, (j)*xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		// bottom end

		//cylinder  圆心位置（(outletBcx * 0.5)，0）圆柱绕流（一个圆柱）
		const double cylinder_x = 0.01 - 4 * xjian;//0.3 * x1;//圆心的坐标x
		const double cylinder_x2 = 0.03 - 4 * xjian;
		const double cylinder_y1 = 0.01;//圆心的坐标y: cylinder_y1 - y1 * 0.5
		const double cylinder_y2 = 0.03;
		const double cylinder_y3 = 0.05;
		const double cylinder_y4 = 0.07;

		const double cylinder_y5 = 0.02;
		const double cylinder_y6 = 0.04;
		const double cylinder_y7 = 0.06;
		const double cylinder_y8 = 0.0 ;
		const double cylinder_y9 = 0.08;
		const double r0 = 0.005;//圆柱半径
		for (int i = 0; i < 4; i++) {			
			const double r = -i * xjian + r0;
			const double l = 2.0 * math::PI * r;
			const unsigned n = l / xjian;
			const double dj = 2.0 * math::PI / n;
			//const double cylinder_x = 0.005;
			//std::cerr << std::endl<<"圆心位置(outletBcx * 0.5)" << (outletBcx * 0.5) << std::endl;
			for (double j = 0; j < 2.0 * math::PI; j = j + dj)//圆心位置（(outletBcx * 0.5)，0）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x, r * sin(j)+ cylinder_y1 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
			for (double j = 0; j < 2.0 * math::PI; j = j + dj)//圆心位置（(outletBcx * 0.5)，0）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x, r * sin(j) + cylinder_y2 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
			for (double j = 0; j < 2.0 * math::PI; j = j + dj)//圆心位置（(outletBcx * 0.5)，0）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x, r * sin(j) + cylinder_y3 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
			for (double j = 0; j < 2.0 * math::PI; j = j + dj)//圆心位置（(outletBcx * 0.5)，0）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x, r * sin(j) + cylinder_y4 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
			for (double j = 0; j < 2.0 * math::PI; j = j + dj)//圆心位置（(outletBcx * 0.5)，0）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x2, r * sin(j) + cylinder_y5 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
			for (double j = 0; j < 2.0 * math::PI; j = j + dj)//圆心位置（(outletBcx * 0.5)，0）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x2, r * sin(j) + cylinder_y6 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
			for (double j = 0; j < 2.0 * math::PI; j = j + dj)//圆心位置（(outletBcx * 0.5)，0）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x2, r * sin(j) + cylinder_y7 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
			//半圆
			for (double j = 0; j < 1.0 * math::PI+ 0.2*dj; j = j + dj)//圆心位置（(cylinder_x2)， cylinder_y8 - y1 * 0.5）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x2, r * sin(j) + cylinder_y8 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
			for (double j = 1.0 * math::PI; j < 2.0 * math::PI+ 0.2 * dj; j = j + dj)//圆心位置（(cylinder_x2)， cylinder_y8 - y1 * 0.5）因为inletWidth = D.
			{
				particle* p = new sph::particle(r * cos(j) + cylinder_x2, r * sin(j) + cylinder_y9 - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
					Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
					, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_max);
				p->setBtype(BoundaryType::Boundary);
				p->setFtype(FixType::Fixed);
				p->setFltype(FluidType::Water);
				p->setIotype(InoutType::Fluid);
				p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
				particles.push_back(p);
			}
		}
		//cylinder end
		// air
		for (int i = 0; i < int(x1 / xjian)+1; i++)
			for (int j = 0; j < int(y1 / xjian) + 1; j++)//y1范围：0-0.06
			{
				double dx = (i) * xjian - cylinder_x;//减去圆心位置
				double dx2 = (i) * xjian - cylinder_x2;
				double dy1 = (j)*xjian - cylinder_y1;
				double dy2 = (j)*xjian - cylinder_y2;
				double dy3 = (j)*xjian - cylinder_y3;
				double dy4 = (j)*xjian - cylinder_y4;
				double dy5 = (j)*xjian - cylinder_y5;
				double dy6 = (j)*xjian - cylinder_y6;
				double dy7 = (j)*xjian - cylinder_y7;
				double dy8 = (j)*xjian - cylinder_y8;
				double dy9 = (j)*xjian - cylinder_y9;
				double r2_1 = dx * dx + dy1 * dy1;
				double r2_2 = dx * dx + dy2 * dy2;
				double r2_3 = dx * dx + dy3 * dy3;
				double r2_4 = dx * dx + dy4 * dy4;
				double r2_5 = dx2 * dx2 + dy5 * dy5;
				double r2_6 = dx2 * dx2 + dy6 * dy6;
				double r2_7 = dx2 * dx2 + dy7 * dy7;
				double r2_8 = dx2 * dx2 + dy8 * dy8;
				double r2_9 = dx2 * dx2 + dy9 * dy9;
				const double rmax = (r0 + 0.5 * xjian)* (r0 + 0.5 * xjian);
				if (r2_1 > rmax && r2_2 > rmax && r2_3 > rmax && r2_4 > rmax && r2_5 > rmax && r2_6 > rmax && r2_7 > rmax && r2_8 > rmax && r2_9 > rmax)
				//if (r2_2 > rmax )
				{
					particle* p = new sph::particle((i) * xjian , (j)*xjian - y1 * 0.5, 0, 0, p0, p_back, Fluid::Gamma(FluidType::Water),
						Fluid::FluidDensity(FluidType::Water), xjian * xjian, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), xjian * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature_min);
					p->setBtype(BoundaryType::Bulk);
					p->setFtype(FixType::Free);
					p->setFltype(FluidType::Water);
					p->setIotype(InoutType::Fluid);
					p->setIdx(static_cast<unsigned int>(particles.size()) + 1);
					particles.push_back(p);
				}
			}
		// air end

		idp = particles.size();
		const std::clock_t end = std::clock();
		std::cout << particles.size() << " particles created in " << double(end - begin) / TICKS_PER_SEC << "s\n";

	}


	inline void domain::writeInitial()
	{
		std::cout << "writing out initial file\n";
		if (particles.empty())
		{
			std::cout << "domain not initialized or domain is empty\n";
			return;
		}
		std::ofstream ofile("initial.dat");
		ofile << "Variables = \"X\",\"Y\",\"Temperature\",\"Boundary\",\"Fix\",\"Io\",\n";//
		ofile << "Zone I=" << particles.size() << std::endl;
		//ѭ��������vectorͷ�ļ�
		for (int i=0; i < particleNum(); i++)
		{
			int btype = particlesa.getBtype(i) == BoundaryType::Bulk ? 0 : 1;
			int ftype = particlesa.getFtype(i) == FixType::Free ? 0 : 1;
			int io = static_cast<int>(particlesa.iotype[i]);
			ofile << particlesa.getX(i) << "\t" << particlesa.getY(i) << "\t" << particlesa.gettemperature(i) << "\t" << btype << "\t" << ftype << "\t" << io << std::endl;//
		}
		ofile.close();
	}

	//��������solve (���˼·��
	inline void domain::solve(unsigned int _total)
	{
		//int deviceId;
		//cudaGetDevice(&deviceId);
		//printf("current device is %d\n", deviceId);
		const unsigned int t_old = total;
		nbBuilt = 0;
		total = total + _total;
		this->buildNeighb(true);
		//this->buildNeighb0(true);
		this->updateWeight();
		const unsigned int cols = this->getConsoleWidth() - 20 - 85;
		progresscpp::ProgressBar pbar(_total, cols);
		const std::string banner = "\033[7mParticleNo | Timestep |      Time     |  Status | BN No | time/sec |    dt   |   vel   |  Process Indicator";
		const unsigned int len = this->getConsoleWidth() - banner.length();
		const std::string mid(len, ' ');
		const std::string end = "\33[m\n";
		double tdtime = 0;
		double pertime, dt=0;
		double vel = 0,vel0 = 0.005 ;
		//this->applyVelBc(0.005);
		if (usingProgressbar)
			std::cout << banner << mid << end;
		
		particlesa.shift2dev(particleNum());

		for (; istep < total; istep++)
		{			
			const std::clock_t begin = std::clock();			
			dt = this->getdt();
			/*if (time < 0.1)
				vel = vel0 * (time / 0.1);
			else vel = vel0;*/
			//this->applyVelBc(vel);//这里判断inlet粒子
			//f = this->checkFluid2Buffer() || f;//生成ghost粒子			
			if ((istep - t_old) % 20 == 0)
			{
				this->density_filter();//温度也校正一下，将温度的校正一并放到这个函数中(暂且不GPU并行)
				//this->temperature_filter();
			}
			//this->iterate(dt);
			this->adjustC0();
			this->inlet();//给入口定压力，速度继承
			this->outlet();//这里判断outlet粒子,等out粒子变到intlet粒子，就重新建近邻，温度也得初始化
			particlesa.shift2dev(particleNum());
			this->buildNeighb(true);
			//this->buildNeighb0(true);
			this->run(dt);
			//this->buffoutlet(dt);
			const std::clock_t end = std::clock();
			const auto dtime = double(end - begin) / TICKS_PER_SEC;
			tdtime += dtime;
			++pbar;
			pertime = dt / dtime;
			if (istep == t_old + 1 || (outScreen != 0 && (istep - t_old) % outScreen == 0))
			{
				if (usingProgressbar) {
					//const std::string neibbuild = f ? "Yes" : " No";
					if (outIntval != 0 && istep == t_old + 1 || (istep - t_old) % outIntval == 0)
						printf("%11lu|%10d| %12.7e | writing | %5d | %8.3e|%8.3e|%8.4f | ",
							particles.size(), istep, time, nbBuilt, pertime, dt, vel);
					else
						printf("%11lu|%10d| %12.7e | running | %5d | %8.3e|%8.3e|%8.4f | ",
							particles.size(), istep, time, nbBuilt, pertime, dt, vel);
					pbar.display();
				}
				else
					this->screen(istep);
			}
			if (istep == t_old + 1 || (outIntval != 0 && (istep - t_old) % outIntval == 0)||istep == t_old + 10)
			{
				this->output(istep);
				//this->output_one(istep);
				//this->output_middle(istep);
				//this->outputfluid(istep);				
			}
			time += dt;
		}
		++pbar;
		this->output(istep);
		if (usingProgressbar) {
			printf("%11lu|%10d| %12.7e | running | %5d | %8.3e|%8.3e| ", particles.size(), istep, time, nbBuilt, pertime, dt);
			pbar.done();
		}
		else
			std::cout << "simulation complete.\n";
	}

	//��������getdt
	inline const double domain::getdt() const
	{
		//静态热传导，由于速度为0，求dt时涉及速度加速度的统统不要

		const double alpha_pi = 1.0;
		double* dtmin;
		cudaMallocManaged(&dtmin, sizeof(double));
		*dtmin = DBL_MAX;

		//改gpu
		getdt_dev0(particles.size(), dtmin, particlesa.divvel, particlesa.hsml, particlesa.fltype, vmax, particlesa.ax, particlesa.ay);
		//printf("dtmin=%e\n", *dtmin);
		//dtmin = std::min(dt00, dt22);		
		//dtmin = std::min(dtmin, 0.000001);
		if (*dtmin < 0) {
			std::cerr << std::endl << "dt 小于0:  " << dtmin << std::endl;
			//std::cerr << "dt00:  " << dt00 << std::endl;
			//std::cerr << "dt11:  " << dt11 << std::endl;
			//std::cerr << "dt22:  " << dt22 << std::endl;
			//std::cerr << "dt33:  " << dt33 << std::endl;
			exit(-1);
		}
		
		return *dtmin;
		//return 0.0000001;
	}

	//建近邻：需要考虑流入流出边界的近邻搜索
	inline bool domain::buildNeighb(bool Force)
	{
		int deviceId;
		cudaGetDevice(&deviceId);

		if (drmax + drmax2 <= neibskin * dp * 1.01 && !Force) return false;
		drmax = drmax2 = 0;
		nb_updated = true;
		nbBuilt++;
		//std::cout << "Building neiblist\n";
		const std::clock_t begin = std::clock();
		drmax = drmax2 = 0;

		/*double x_max = DBL_MIN;
		double x_min = DBL_MAX;
		double y_max = DBL_MIN;
		double y_min = DBL_MAX;*/
		double* x_max;
		double* x_min;
		double* y_max;
		double* y_min;

		cudaMallocManaged(&x_max, sizeof(double));
		cudaMallocManaged(&x_min, sizeof(double));
		cudaMallocManaged(&y_max, sizeof(double));
		cudaMallocManaged(&y_min, sizeof(double));

		*x_max = DBL_MIN;
		*x_min = DBL_MAX;
		*y_max = DBL_MIN;
		*y_min = DBL_MAX;

		//GPU
		buildNeighb_dev01(particles.size(), particlesa.ux, particlesa.uy, particlesa.x, particlesa.y, x_max, x_min, y_max, y_min);

		*x_max += 5.0 * dp;
		*x_min -= 5.0 * dp;
		*y_max += 5.0 * dp;
		*y_min -= 5.0 * dp;
		//确定计算域
		const double dxrange = *x_max - *x_min;
		const double dyrange = *y_max - *y_min;
		//printf("\nallow us to test the range of which:x:%lf and y:%lf\n", dxrange, dyrange);

		const int ntotal = static_cast<int>(particles.size());
		//const int gtotal = static_cast<int>(ghosts.size());
		//const int ngtotal = ntotal + gtotal;
		//给计算域确定网格的数量，比如整个计算域需要划分成10（ngridx）*8（ngridy）个网格
		const int ngridx = std::min(int(pow(ntotal * dxrange / (dyrange * 3), 1.0 / 3.0)) + 1, xgridmax);
		const int ngridy = std::min(int(ngridx * dyrange / dxrange) + 1, ygridmax);

		//math::matrix grid(ngridx, ngridy);//网格编号，使用矩阵来对网格进行编号。（10，8）
		int* grid_d;
		cudaMallocManaged(&grid_d, sizeof(int) * ngridx * ngridy);
		for (int gridi = 0; gridi < ngridx * ngridy; gridi++) {
			grid_d[gridi] = 0;
		}
		cudaMemPrefetchAsync(grid_d, sizeof(int) * ngridx * ngridy, deviceId, NULL);

		//int* xgcell = new int[ntotal];//三个数组，分别来存储每个粒子的编号信息
		//int* ygcell = new int[ntotal];
		//int* celldata = new int[ntotal];
		int* xgcell ;//三个数组，分别来存储每个粒子的编号信息
		int* ygcell ;
		int* celldata ;
		cudaMallocManaged(&xgcell, sizeof(int) * ntotal);
		cudaMallocManaged(&ygcell, sizeof(int) * ntotal);
		cudaMallocManaged(&celldata, sizeof(int) * ntotal);
		cudaMemPrefetchAsync(xgcell, sizeof(int) * ntotal, deviceId, NULL);
		cudaMemPrefetchAsync(ygcell, sizeof(int) * ntotal, deviceId, NULL);
		cudaMemPrefetchAsync(celldata, sizeof(int) * ntotal, deviceId, NULL);


		//GPU
		/*for (int j; j < particleNum(); j++) {
			printf("")
		}*/
		buildNeighb_dev02(particles.size(), particlesa.x, particlesa.y, particlesa.neiblist, particlesa.neibNum\
			, ngridx, ngridy, dxrange, dyrange, *x_min, *y_min\
			, xgcell, ygcell, celldata, grid_d, particlesa.hsml, particlesa.idx, particlesa.iotype, lengthofx);

		cudaFree(xgcell);
		cudaFree(ygcell);
		cudaFree(celldata);
		const std::clock_t end = std::clock();
		//this->debugNeib(1240);
		//this->debugNeib(1239);
		//std::cout << "Building neighbor list costs " << double(end - begin) / CLOCKS_PER_SEC << "s\n";
		return true;
	}
	//------------------------全配对搜索------------------------------目前开不了并行
	inline bool domain::buildNeighb0(bool Force)
	{
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++) {
			particle* ii = particles[i];
			(ii)->setZeroDisp();
			//const double x = particlesa.getX(i);
			//const double y = particlesa.getY(i);			
		}
		const int ntotal = static_cast<int>(particles.size());
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 1; i <= ntotal; i++)
		{
			double x, y;
			x = particlesa.getX(i-1);
			y = particlesa.getY(i-1);
			particlesa.clearNeiblist(i-1);
			if (x != x) {
				std::cerr << std::endl << "Nan in buildNeighb0\n" << std::endl;
				//exit(-1);
			}
		}
		//遍历一遍
//#ifdef OMP_USE
//#pragma omp parallel for schedule (guided)
//#endif
		for (int i = 1; i < ntotal; i++)
		{
//#ifdef OMP_USE
//#pragma omp parallel for schedule (guided)
//#endif
			for (int j = i+1; j <= ntotal; j++)
			{
				const double xi = particlesa.getX(i-1);
				const double yi = particlesa.getY(i-1);
				const double xj = particlesa.getX(j-1);
				const double yj = particlesa.getY(j-1);
				const double dx = xi - xj;
				const double dy = yi - yj;
				const double r = sqrt(dx * dx + dy * dy);//除了一般的r，还有周期边界的r2
				if (r == 0 && i <= ntotal && j <= ntotal) {
					std::cerr << "\nError: two particles occupy the same position\n";
					std::cerr << "i=" << i << " j=" << j << std::endl;
					std::cerr << (particlesa.idx[i-1]) << " and " << (particlesa.idx[j-1]) << std::endl;
					std::cerr << "at " << xi << '\t' << yi;
					std::cerr << "ntotal=" << ntotal << std::endl;
					this->output(istep);
					exit(-1);
				}
				const double mhsml = particlesa.gethsml(i-1);
				const double horizon = 3.3 * mhsml;					
				if (r < horizon) {
					particlesa.add2Neiblist(i - 1, j - 1);
					particlesa.add2Neiblist(j - 1, i - 1);
				}
				//找周期边界的近邻
				if (particles[i - 1]->iotype == InoutType::Inlet&& particles[j - 1]->iotype == InoutType::Outlet)
				{					
					const double dx2 = xi + lengthofx - xj;
					//std::cout <<std::endl<< "find!  xi= " << xi << "  yi= " << yi << std::endl;
					const double r2 = sqrt(dx2 * dx2 + dy * dy);					
					if (r2 < horizon) {
						/*std::cout << std::endl << "find!  xi= " << xi << "  xj= " << xj << std::endl;
						std::cout <<  "find r2!  r2= " << r2 << " horizon="<< horizon<<std::endl;*/
						particlesa.add2Neiblist(i - 1, j - 1);
						particlesa.add2Neiblist(j - 1, i - 1);
					}
				}
				if (particles[i - 1]->iotype == InoutType::Outlet && particles[j - 1]->iotype == InoutType::Inlet)
				{
					const double dx3 = xi - lengthofx - xj;
					const double r3 = sqrt(dx3 * dx3 + dy * dy);
					if (r3 < horizon) {
						particlesa.add2Neiblist(i - 1, j - 1);
						particlesa.add2Neiblist(j - 1, i - 1);
					}
				}
			}
		}
	}

	inline void domain::updateWeight()
	{
		const std::clock_t begin = std::clock();

		singlestep_updateWeight_dev0(particleNum(), particlesa.neibNum, particlesa.hsml, particlesa.neiblist, particlesa.x, particlesa.y\
			, particlesa.iotype, lengthofx, particlesa.bweight, particlesa.dbweightx, particlesa.dbweighty);
//#ifdef OMP_USE
//#pragma omp parallel for schedule (guided)
//#endif
//		for (int i = 0; i < particles.size(); i++)
//		{
//
//			//particle* ii = particles[i];
//
//			const unsigned int neibNum = particlesa.neibNum[i];
//			const double hsml = particlesa.hsml[i];
//			double sum = 0;
//
//			//
//			/*#ifdef OMP_USE
//				#pragma omp parallel for schedule (guided) reduction(+:sum)
//			#endif*/
//			for (int j = 0; j < neibNum; j++)//�����κ����Աȣ�neiblist.size()=neighborNum(i)
//			{
//				const int jj = particlesa.neiblist[i][j];//jj=k
//				double dx = particlesa.x[i]-particlesa.x[jj];//xi(1)
//				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
//				{
//					dx = particlesa.x[i] + lengthofx - particlesa.x[jj];//xi(1)
//				}
//				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
//				{
//					dx = particlesa.x[i] - lengthofx - particlesa.x[jj];//xi(1)
//				}
//				//const double dx = particlesa.x[i]-particlesa.x[jj];//xi(1)
//				const double dy = particlesa.y[i] - particlesa.y[jj];//xi(2)
//				const math::vector xi(dx, dy);
//				const double r = xi.length();
//				//const double q = r / hsml;   //q�Ĵ�����ͬ�������һ��mhsml(662-664)
//				const double hsmlj = particlesa.hsml[jj];//hsmlΪhsml(i)��hsmljΪhsml(k)
//				const double mhsml = (hsml + hsmlj) * 0.5;
//				const double q = r / mhsml;
//				if (q > 3.0) {
//					particlesa.bweight[i][j] = 0;
//					particlesa.dbweightx[i][j] = 0;
//					particlesa.dbweighty[i][j] = 0;
//					continue;
//				}
//
//				const double fac = 1.0 / (math::PI * mhsml * mhsml) / (1.0 - 10.0 * exp9);
//				const double bweight = fac * (exp(-q * q) - exp9);
//
//				sum += bweight;
//				particlesa.bweight[i][j] = bweight;
//				const double factor = fac * exp(-q * q) * (-2.0 / mhsml / mhsml);
//
//				particlesa.dbweightx[i][j] = factor * dx;
//				particlesa.dbweighty[i][j] = factor * dy;
//			}
//
//#ifdef OMP_USE
//#pragma omp parallel for schedule (guided)
//#endif
//			for (int j = 0; j < particlesa.neibNum[i]; j++)
//			{
//				particlesa.bweight[i][j] /= sum;
//			}
//		}
 

		// ghosts
		
//#ifdef OMP_USE
//#pragma omp parallel for schedule (guided)
//#endif
//		for (int i = 0; i < ghosts.size(); i++)
//		{
//			//(*i)->bweight.clear();
//			//(*i)->dbweightx.clear();
//			//(*i)->dbweighty.clear();
//			particle* ii = ghosts[i];
//			/*if (nb_updated)
//			{
//				if ((ii)->bweight) free((ii)->bweight);
//				if ((ii)->dbweightx) free((ii)->dbweightx);
//				if ((ii)->dbweighty) free((ii)->dbweighty);
//				particlesa.neibNum[i] = particlesa.neibNum[i];
//				const unsigned int neibNum = particlesa.neibNum[i];
//				(ii)->bweight = new double[neibNum];
//				(ii)->dbweightx = new double[neibNum];
//				(ii)->dbweighty = new double[neibNum];
//			}*/
//			const unsigned int neibNum = particlesa.neibNum[i];
//			/*if (neibNum > MAX_NEIB) {
//				std::cerr << "Error: neighbor list length exceeds " << neibNum;
//				exit(-2);
//			}*/
//			const double hsml = particlesa.hsml[i];
//			double sum = 0;
//
//#ifdef OMP_USE
//#pragma omp parallel for schedule (guided)
//#endif
//			for (int j = 0; j < particlesa.neibNum[i]; j++)
//			{
//				const int jj = particlesa.neiblist[i][j];
//				const double dx = particlesa.x[i]-particlesa.x[jj];
//				const double dy = particlesa.y[i] - particlesa.y[jj];
//				const math::vector xi(dx, dy);
//				const double r = xi.length();
//				const double hsmlj = particlesa.hsml[jj];
//				const double mhsml = (hsml + hsmlj) / 2;
//				const double q = r / mhsml;
//				if (q > 3.0) {
//					particlesa.bweight[i][j] = 0;
//					particlesa.dbweightx[i][j] = 0;
//					particlesa.dbweighty[i][j] = 0;
//					continue;
//				}
//				const double fac = 1.0 / (math::PI * mhsml * mhsml) / (1.0 - 10.0 * exp9);
//				const double bweight = fac * (exp(-q * q) - exp9);
//				sum += bweight;
//				particlesa.bweight[i][j] = bweight;
//				const double factor = fac * exp(-q * q) * (-2.0 / mhsml / mhsml);
//				//const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);
//				particlesa.dbweightx[i][j] = factor * dx;
//				particlesa.dbweighty[i][j] = factor * dy;
//			}
//
//#ifdef OMP_USE
//#pragma omp parallel for schedule (guided)
//#endif
//			for (int j = 0; j < particlesa.neibNum[i]; j++)
//			{
//				particlesa.bweight[i][j] /= sum;
//			}
//		}
		nb_updated = false;
		const std::clock_t end = std::clock();
		if (time_measure) std::cout << "update weight costs " << double(end - begin) / TICKS_PER_SEC << "s\n";
	}

	//density_filter   
	inline void domain::density_filter()
	{
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			//(*i)->density_filter();
			//particle* ii = particles[i];
			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			double beta0_mls = 0;
			double rhop_sum_mls = 0;
			const double hsml = particlesa.hsml[i];

#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:beta0_mls,rhop_sum_mls)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				const double rho_j = (particlesa.press[jj] - particlesa.back_p[jj]) / particlesa.c0[jj] / particlesa.c0[jj] + particlesa.rho0[jj];
				const double v_j = particlesa.mass[jj] / particlesa.rho[jj];
				const double mass_j = rho_j * v_j;
				beta0_mls += particlesa.bweight[i][j] * v_j;
				rhop_sum_mls += particlesa.bweight[i][j] * mass_j;
			}

			beta0_mls += sph::wfunc::factor(hsml, 2) * particlesa.mass[i] / particlesa.rho[i];
			rhop_sum_mls += sph::wfunc::factor(hsml, 2) * particlesa.mass[i];
			particlesa.rho[i] = rhop_sum_mls / beta0_mls;
		}
	}
	//temperature_filter
	inline void domain::temperature_filter()
	{
	}
	//pressmax需要每次计算都更更新
	inline const double domain::getPressmax()
	{
		pmax = 0;
		for (int i = 0; i < particles.size(); i++) {
			particle* ii = particles[i];
			//if (particlesa.iotype[i] != InoutType::Inlet) continue;
			//if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			double p = particlesa.press[i];			
			pmax = pmax > p ? pmax : p;
			return pmax;
		}
	}
	//未调用
	inline void domain::iterate(double _dt)
	{
		const double dt = _dt;
		const double dt2 = _dt * 0.5;
		this->predictor(dt2);
		this->corrector(dt);
	}

	//��������predictor
	inline void domain::predictor(double dt2)
	{
		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->storeHalf();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->densityRegulation();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->updatePressure();
		}

		this->buildNeighb();
		//this->buildNeighb0();

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->updateDDensity();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->updateFint();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->getTurbulence();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->getArticial();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->updateAcc();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->integration1sthalf(dt2);
		}
	}

	//��������corrector
	inline void domain::corrector(double _dt)
	{
		const double dt = _dt;
		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->densityRegulation();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->updatePressure();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->updateDDensity();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->updateFint();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->getTurbulence();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->getArticial();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->updateAcc();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->integrationfull(dt);
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->shifting_c();
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			(*i)->shifting(dt);
		}

		for (std::vector<class particle*>::iterator i = particles.begin(); i != particles.end(); i++)
		{
			const double dist = (*i)->getDisp();
			if (dist > drmax)
			{
				drmax2 = drmax;
				drmax = dist;
			}
			else if (dist > drmax2)
			{
				drmax2 = dist;
			}
		}
	}

	//输出所有粒子
	inline void domain::output(unsigned int istep)
	{
		//std::string filename;
		char buff[100];		
		snprintf(buff, sizeof(buff), "E:\\code1\\ULPHbase\\data\\coutte\\data2\\output%10.10d.dat", istep);//在指定路径下写文件
		//snprintf(buff, sizeof(buff), "output%10.10d.dat", istep);
		//filename = "output" + std::to_string(istep) + ".dat";
		std::string filename = buff;//得到文件名ULPH_temperature_haosan
		std::ofstream ofile(filename);		
		ofile << "Title = \"Particle Information\"" << std::endl;
		ofile << "Variables = \"X\",\"Y\",\"Vx\",\"Vy\",\"Ax\",\"Ay\",\"Fintx\",\"Finty\",\"Avx\",\"Avy\",\"Tx\",\"Ty\",\"Vel\",\"rho\",\"P\",\"drho\",\"bond\",\"Fluid\",\"Fix\",\"Bc\",\"Inout\",\"Idx\",\"vort\",\"Temperature\",\"Tempx\",\"Tempy\",\"Temp_t\",\"vcc\",\"Time\"" << std::endl;
		ofile << "Zone T=\"Fluid\" I=" << particles.size() << std::endl;
		for (int i = 0; i <particleNum(); i++)
		{
			ofile << particlesa.getX(i) << '\t' << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.getVy(i) << '\t' << particlesa.getAx(i) << '\t' << particlesa.getAy(i)
				<< '\t' << particlesa.getFx(i) << '\t' << particlesa.getFy(i) << '\t' << particlesa.getAvx(i) << '\t' << particlesa.getAvy(i) << '\t' << particlesa.getTx(i) << '\t' <<
				particlesa.getTy(i) << '\t' << sqrt(particlesa.getVx(i) * particlesa.getVx(i) + particlesa.getVy(i) * particlesa.getVy(i)) << '\t' << particlesa.getDensity(i) << '\t' << particlesa.getPress(i)
				<< '\t' << particlesa.drho[i] << '\t' << particlesa.neibNum[i] << '\t';
			ofile << '\t' << static_cast<int>(particlesa.fltype[i]) << '\t' << static_cast<int>(particlesa.ftype[i]) << '\t' << static_cast<int>(particlesa.btype[i]) << '\t' << static_cast<int>(particlesa.iotype[i]) << '\t' << particlesa.idx[i]
				<< '\t' << particlesa.getvort(i) << '\t' << particlesa.gettemperature(i) << '\t' << particlesa.gettempx(i) << '\t' << particlesa.gettempy(i) << '\t' << particlesa.gettempt(i) << '\t' << particlesa.vcc[i] << '\t' << time << std::endl;
		}
		/*ofile << "Zone T=\"Ghosts\" I=" << ghosts.size() << std::endl;
		for (auto i = ghosts.begin(); i != ghosts.end(); i++)
		{
			ofile << particlesa.getX(i) << '\t' << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.getVy(i) << '\t' << particlesa.getAx(i) << '\t' << particlesa.getAy(i)
				<< '\t' << particlesa.getFx(i) << '\t' << particlesa.getFy(i) << '\t' << particlesa.getAvx(i) << '\t' << particlesa.getAvy(i) << '\t' << particlesa.getTx(i) << '\t' <<
				particlesa.getTy(i) << '\t' << sqrt(particlesa.getVx(i) * particlesa.getVx(i) + particlesa.getVy(i) * particlesa.getVy(i)) << '\t' << particlesa.getDensity(i) << '\t' << particlesa.getPress(i)
				<< '\t' << particlesa.drho[i] << '\t' << particlesa.neibNum[i];
			ofile << '\t' << static_cast<int>(particlesa.fltype[i]) << '\t' << static_cast<int>(particlesa.ftype[i]) << '\t' << static_cast<int>(particlesa.btype[i]) << '\t' << static_cast<int>(particlesa.iotype[i]) << '\t' << particlesa.idx[i]
				<< '\t' << particlesa.getvort(i) << '\t' << particlesa.gettemperature(i) << '\t' << particlesa.gettempx(i) << '\t' << particlesa.gettempy(i) << '\t' << particlesa.gettempt(i) << '\t' << particlesa.vcc[i] << std::endl;
		}*/
		ofile.close();
	}
	//输出流体粒子
	inline void domain::outputfluid(unsigned int istep)
	{
		//std::string filename;
		char buff[100];
		snprintf(buff, sizeof(buff), "E:\\code\\ulph_example3\\data\\couette\\data10\\output%10.10d.dat", istep);//在指定路径下写文件		
		std::string filename = buff;//得到文件名
		std::ofstream ofile(filename);
		ofile << "Title = \"Particle Information of Fluid\"" << std::endl;
		ofile << "Variables = \"X\",\"Y\",\"Vx\",\"Vy\",\"Ax\",\"Ay\",\"Fintx\",\"Finty\",\"Avx\",\"Avy\",\"Tx\",\"Ty\",\"Vel\",\"rho\",\"P\",\"drho\",\"bond\",\"Fluid\",\"Fix\",\"Bc\",\"Inout\",\"Idx\",\"vort\",\"Temperature\",\"Tempx\",\"Tempy\",\"Temp_t\",\"vcc\",\"Time\"" << std::endl;
		ofile << "Zone T=\"Fluid\"" << std::endl;
		for (int i = 0; i <particleNum(); i++)
		{
			if (static_cast<int>(particlesa.btype[i]) == 1) continue;
			if (particlesa.getX(i) <= 0.2 && particlesa.getX(i) >= 0 - 0.3 * dp) {
				ofile << particlesa.getX(i) << '\t' << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.getVy(i) << '\t' << particlesa.getAx(i) << '\t' << particlesa.getAy(i)
					<< '\t' << particlesa.getFx(i) << '\t' << particlesa.getFy(i) << '\t' << particlesa.getAvx(i) << '\t' << particlesa.getAvy(i) << '\t' << particlesa.getTx(i) << '\t' <<
					particlesa.getTy(i) << '\t' << sqrt(particlesa.getVx(i) * particlesa.getVx(i) + particlesa.getVy(i) * particlesa.getVy(i)) << '\t' << particlesa.getDensity(i) << '\t' << particlesa.getPress(i)
					<< '\t' << particlesa.drho[i] << '\t' << particlesa.neibNum[i] << '\t';
				ofile << '\t' << static_cast<int>(particlesa.fltype[i]) << '\t' << static_cast<int>(particlesa.ftype[i]) << '\t' << static_cast<int>(particlesa.btype[i]) << '\t' << static_cast<int>(particlesa.iotype[i]) << '\t' << particlesa.idx[i]
					<< '\t' << particlesa.getvort(i) << '\t' << particlesa.gettemperature(i) << '\t' << particlesa.gettempx(i) << '\t' << particlesa.gettempy(i) << '\t' << particlesa.gettempt(i) << '\t' << particlesa.vcc[i] << '\t' << time << std::endl;
			}
		}		
		ofile.close();
	}
	//稳态之后，在不同x处的温度分布：x=0.1;x=0.3;x=0.5;x=0.7;x=0.9;
	inline void domain::output_one(unsigned int istep)
	{
		//std::string filename;
		char buff[100];
		snprintf(buff, sizeof(buff), "E:\\code\\ulph_example3\\data\\cylinder\\data3\\output%10.10d.dat", istep);//在指定路径下写文件
		//snprintf(buff, sizeof(buff), "output%10.10d.dat", istep);
		//filename = "output" + std::to_string(istep) + ".dat";
		std::string filename = buff;//得到文件名
		std::ofstream ofile(filename);
		ofile << "Title = \"Particle Information of different X\"" << std::endl;
		ofile << "Variables = \"Y\",\"Vx\",\"temperature\"" << std::endl;
		//ofile << "Zone T=\"Fluid\" I=" << particles.size() << std::endl;
		//ofile << "Zone T=\"Fluid\" I=" << 67 << std::endl;
		ofile << "Zone T=\"X_0.1\"" << std::endl;
		for (int i = 0; i < particleNum(); i++)
		{
			double i_x = particlesa.getX(i) - (0.1);
			double absi_x = abs(i_x);
			if (static_cast<int>(particlesa.btype[i]) == 1) continue;
			if (absi_x < 0.5 * dp)
			{
				ofile << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.gettemperature(i) << '\t' << std::endl;
			}
		}
		ofile << "Zone T=\"X_0.3\"" << std::endl;
		for (int i = 0; i < particleNum(); i++)
		{
			double i_x = particlesa.getX(i) - (0.3);
			double absi_x = abs(i_x);
			if (static_cast<int>(particlesa.btype[i]) == 1) continue;
			if (absi_x < 0.5 * dp)
			{
				ofile << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.gettemperature(i) << '\t' << std::endl;
			}
		}
		ofile << "Zone T=\"X_0.5\"" << std::endl;
		for (int i = 0; i < particleNum(); i++)
		{
			double i_x = particlesa.getX(i) - (0.5);
			double absi_x = abs(i_x);
			if (static_cast<int>(particlesa.btype[i]) == 1) continue;
			if (absi_x < 0.5 * dp)
			{
				ofile << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.gettemperature(i) << '\t' << std::endl;
			}
		}
		ofile << "Zone T=\"X_0.7\"" << std::endl;
		for (int i = 0; i < particleNum(); i++)
		{
			double i_x = particlesa.getX(i) - (0.7);
			double absi_x = abs(i_x);
			if (static_cast<int>(particlesa.btype[i]) == 1) continue;
			if (absi_x < 0.5 * dp)
			{
				ofile << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.gettemperature(i) << '\t' << std::endl;
			}
		}
		ofile << "Zone T=\"X_0.9\"" << std::endl;
		for (int i = 0; i < particleNum(); i++)
		{
			double i_x = particlesa.getX(i) - (0.9);
			double absi_x = abs(i_x);
			if (static_cast<int>(particlesa.btype[i]) == 1) continue;
			if (absi_x < 0.5 * dp)
			{
				ofile << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.gettemperature(i) << '\t' << std::endl;
			}
		}		
		ofile.close();
	}
	//输出x=0.1m处的速度，用于画速度分布图
	inline void domain::output_middle(unsigned int istep)
	{
		//std::string filename;
		char buff[100];
		snprintf(buff, sizeof(buff), "E:\\code\\ulph_example3\\data\\couette\\data11\\output%10.10d.dat", istep);//在指定路径下写文件
		//snprintf(buff, sizeof(buff), "output%10.10d.dat", istep);
		//filename = "output" + std::to_string(istep) + ".dat";
		std::string filename = buff;//得到文件名
		std::ofstream ofile(filename);
		ofile << "Title = \"Particle Information of x=0.1\"" << std::endl;
		ofile << "Variables = \"Y\",\"Vx\",\"Temperature\",\"Time\"" << std::endl;
		//ofile << "Zone T=\"Fluid\" I=" << particles.size() << std::endl;
		ofile << "Zone T=\"Fluid\"" << std::endl;
		for (int i = 0; i < particleNum(); i++)
		{
			double i_x = particlesa.getX(i) - (0.1);
			double absi_x = abs(i_x);
			double lengthx = dp * 0.5;
			if (static_cast<int>(particlesa.btype[i]) == 1) continue;
			if (absi_x < lengthx)
			{
				ofile << particlesa.getY(i) << '\t' << particlesa.getVx(i) << '\t' << particlesa.gettemperature(i) << '\t' << time << std::endl;
			}
		}
		ofile.close();
	}
	//��������screen
	inline void domain::screen(unsigned int istep)
	{
		std::cout << " running " << istep << " steps\n";
	}

	inline void domain::setOutIntval(unsigned int _i)
	{
		outIntval = _i;
	}

	inline void domain::setOutScreen(unsigned int _t)
	{
		outScreen = _t;
	}

	inline void domain::setOutlet(double _x, Direction _d, int _layer)//setOutlet(width-spacing*3.5,sph::Direction::Right,4);
	{
		//std::cerr << outletBcx << std::endl;
		outletBcx = _x;//流出边界的左边界线
		//std::cerr << outletBcx << std::endl;
		outletd = _d;
		outletlayer = _layer > 0 ? _layer : 4;
		outletBcxEdge = _d == Direction::Left ? _x - _layer * dp : _x + _layer * dp;//流出边界的右边界线 = width+0.5dx
		lengthofx = outletBcxEdge + 4.5*dp;//这个不对，短了
		std::cout << std::endl << lengthofx << std::endl;
		std::cout << std::endl << outletBcxEdge << std::endl;
		std::cout << std::endl << outletBcx << std::endl;
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)	
#endif
		for (int i = 0; i < particles.size(); i++) {
			particle* ii = particles[i];		
			 double xi = particlesa.getX(i);
			
			 if (xi > outletBcx) {
				 (ii)->setIotype(InoutType::Outlet);
			 }
			 if (xi > outletBcxEdge) {
				 particlesa.x[i] = xi - lengthofx;//减去流体域长度和inlet长度
				 particlesa.setIotype(InoutType::Inlet,i);
			 }
		}
						
	}
	//存在问题：粒子冲进边界
	inline void domain::run(double _dt)
	{
		const double dt = _dt;
		const double dt2 = _dt * 0.5;

		run_half1_dev0(particleNum(), particlesa.half_x, particlesa.half_y, particlesa.half_vx, particlesa.half_vy, particlesa.half_rho, particlesa.half_temperature\
										, particlesa.x, particlesa.y, particlesa.vx, particlesa.vy, particlesa.rho, particlesa.temperature);

		//-------------predictor------------
		this->single_step();
		//this->single_step0();
		//this->single_step_temperature();
		//this->single_step_temperature_gaojie();
		vmax = 0;

		double* d_vmax;
		cudaMallocManaged(&d_vmax, sizeof(double));
		*d_vmax = 0;

		run_half2_dev0(particleNum(), particlesa.half_x, particlesa.half_y, particlesa.half_vx, particlesa.half_vy, particlesa.half_rho, particlesa.half_temperature\
			, particlesa.x, particlesa.y, particlesa.vx, particlesa.vy, particlesa.rho, particlesa.temperature\
			, particlesa.drho, particlesa.ax, particlesa.ay, particlesa.vol, particlesa.mass\
			, particlesa.btype, particlesa.ftype, particlesa.temperature_t, dt2, d_vmax);

		vmax = *d_vmax;


		//-----------corrector--------------
		this->single_step();
		//this->single_step0();
		//this->single_step_temperature();
		//this->single_step_temperature_gaojie();//已经运动了半步了		
#ifdef OMP_USE
#pragma omp parallel for schedule (dynamic)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			/* code */
			//particle* ii = particles[i];
			//(*i)->integrationfull(dt);
			//if (particlesa.btype[i] == sph::BoundaryType::Boundary) { particlesa.shift_c[i] = 0; continue; }

			if (particlesa.ftype[i] != sph::FixType::Fixed) {
				particlesa.rho[i] = particlesa.half_rho[i] + particlesa.drho[i] * dt;//i时刻的密度+(i+0.5*dt)时刻的密度变化率×dt			 
				particlesa.vx[i] = particlesa.half_vx[i] + particlesa.ax[i] * dt;//i时刻的速度+(i+0.5*dt)时刻的加速度×dt
				particlesa.vy[i] = particlesa.half_vy[i] + particlesa.ay[i] * dt;
				particlesa.vol[i] = particlesa.mass[i] / particlesa.rho[i];				
				particlesa.x[i] = particlesa.half_x[i] + particlesa.vx[i] * dt;//i时刻的位置+(i+*dt)时刻的速度×dt---------------------
				//particlesa.y[i] = particlesa.half_y[i] + particlesa.vy[i] * dt;
				double y = particlesa.half_y[i] + particlesa.vy[i] * dt;//加个判断，防止冲进边界
				/*if (y > indiameter * 0.5 - dp || y < -indiameter * 0.5 + dp) {
					y = particlesa.half_y[i];
				}*/				
				particlesa.y[i] = y;
				particlesa.ux[i] += particlesa.vx[i] * dt;
				particlesa.uy[i] += particlesa.vy[i] * dt;
				particlesa.temperature[i] = particlesa.half_temperature[i] + particlesa.temperature_t[i] * dt;
			}			
			if (stype != ShiftingType::DivC) continue;

			double shift_c = 0;

#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:shift_c)
#endif

			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				const double rho_j = particlesa.rho[jj];
				const double massj = particlesa.mass[jj];
				shift_c += particlesa.bweight[i][j] * massj / rho_j;
			}
			particlesa.shift_c[i] = shift_c;
		}

		//this->updateBufferPos(dt);
		//this->updateGhostPos();

		//for (int i = 0; i < particleNum(); i++)
		//{
			/* code */
			//(*i)->shifting_c();
		//}

		if (stype == ShiftingType::None) return;

		if (stype == ShiftingType::DivC) {
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
			for (int i = 0; i < particles.size(); i++)
			{
				/* code */
				//particle* ii = particles[i];
				//(*i)->shifting(dt);
				if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
				//if (particlesa.ftype[i] != sph::FixType::Free) continue;
				const double hsml = particlesa.hsml[i];
				const double conc = particlesa.shift_c[i];

				double shift_x = 0;
				double shift_y = 0;

#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:shift_x,shift_y)
#endif
				for (int j = 0; j < particlesa.neibNum[i]; j++)
				{
					const int jj = particlesa.neiblist[i][j];
					const double mhsml = (hsml + particlesa.hsml[jj]) * 0.5;
					const double conc_ij = particlesa.shift_c[jj] - conc;

					shift_x += conc_ij * particlesa.mass[jj] / particlesa.rho[jj] * particlesa.dbweightx[i][j];
					shift_y += conc_ij * particlesa.mass[jj] / particlesa.rho[jj] * particlesa.dbweighty[i][j];
				}
				const double vx = particlesa.vx[i];
				const double vy = particlesa.vy[i];
				const double vel = sqrt(vx * vx + vy * vy);
				shift_x *= -2.0 * dp * vel * dt * shiftingCoe;
				shift_y *= -2.0 * dp * vel * dt * shiftingCoe;

				particlesa.shift_x[i] = shift_x;
				particlesa.shift_y[i] = shift_y;

				particlesa.x[i] += shift_x;
				particlesa.y[i] += shift_y;
				particlesa.ux[i] += shift_x;
				particlesa.uy[i] += shift_y;

				const double ux = particlesa.ux[i];
				const double uy = particlesa.uy[i];
				const double disp = sqrt(ux * ux + uy * uy);
#pragma omp critical
				{
					if (disp > drmax) {
						drmax = disp;
						drmax2 = drmax;
					}
					else if (disp > drmax2) {
						drmax2 = disp;
					}
				}
			}
		}
		else if (stype == ShiftingType::Velc) {
			const double bweightdx = sph::wfunc::bweight(1.0, dp, 2);
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
			for (int i = 0; i < particles.size(); i++)
			{
				/* code */
				//particle* ii = particles[i];
				//(*i)->shifting(dt);
				if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
				//if (particlesa.ftype[i] != sph::FixType::Free) continue;
				const double hsml = particlesa.hsml[i];
				const double rho_i = particlesa.rho[i];
				const double c0 = particlesa.c0[i];

				double shift_x = 0;
				double shift_y = 0;

#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:shift_x,shift_y)
#endif
				for (int j = 0; j < particlesa.neibNum[i]; j++)
				{
					const int jj = particlesa.neiblist[i][j];
					const double frac = particlesa.bweight[i][j] / bweightdx;
					const double head = 1.0 + 0.2 * pow(frac, 4);
					const double rho_j = particlesa.rho[jj];
					const double rho_ij = rho_i + rho_j;
					const double mass_j = particlesa.mass[jj];

					shift_x += head * particlesa.dbweightx[i][j] * mass_j / rho_ij;
					shift_y += head * particlesa.dbweighty[i][j] * mass_j / rho_ij;
				}
				const double vx = particlesa.vx[i];
				const double vy = particlesa.vy[i];
				const double vel = sqrt(vx * vx + vy * vy);
				shift_x *= -8.0 * dp * dp * vel / c0 * shiftingCoe;
				shift_y *= -8.0 * dp * dp * vel / c0 * shiftingCoe;

				particlesa.shift_x[i] = shift_x;
				particlesa.shift_y[i] = shift_y;

				particlesa.x[i] += shift_x;
				particlesa.y[i] += shift_y;
				particlesa.ux[i] += shift_x;
				particlesa.uy[i] += shift_y;

				const double ux = particlesa.ux[i];
				const double uy = particlesa.uy[i];
				const double disp = sqrt(ux * ux + uy * uy);
#pragma omp critical
				{
					if (disp > drmax) {
						drmax = disp;
						drmax2 = drmax;
					}
					else if (disp > drmax2) {
						drmax2 = disp;
					}
				}
			}
		}
	}
	//求解温度、运动
	inline void domain::single_step_temperature_gaojie() {
		const std::clock_t begin = std::clock();
		//pressure 状态方程
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)	
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			const double c0 = particlesa.c0[i];
			const double rho0 = particlesa.rho0[i];
			const double rhoi = particlesa.rho[i];			
			const double gamma = (ii)->gamma;		
			const double b = c0 * c0 * rho0 / gamma;     //b--B,
			const double p_back = (ii)->back_p;
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
				particlesa.press[i] = b * (pow(rhoi / rho0, gamma) - 1.0) + p_back;      // 流体区
				
			if (particlesa.press[i] < p_back) 
				particlesa.press[i] = p_back;
		}
		this->updateWeight();
		// boundary pressure  and 无滑移速度边界条件	上下固壁为自由滑移，圆柱边界为无滑移边界
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			//particle* ii = particles[i];
			if (particlesa.btype[i] != sph::BoundaryType::Boundary) continue;
			//if (particlesa.iotype[i] == sph::InoutType::Buffer) continue;			
			double p = 0, vcc = 0;
			double v_x = 0, v_y = 0;			
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:p,vcc,v_x,v_y)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				//if (particlesa.btype[i] == particlesa.btype[jj]) continue;//Buffer粒子也在这里，导致流出边界固壁的压力不正常
				if (particlesa.ftype[jj] == sph::FixType::Fixed) continue;//其他固壁粒子不参与，ghost不参与，buffer参与
				const double mass_j = particlesa.mass[jj];
				const double rho_j = particlesa.rho[jj];
				const double p_k = particlesa.press[jj];
				p += p_k * particlesa.bweight[i][j];//
				vcc += particlesa.bweight[i][j];
				v_x += particlesa.vx[jj] * particlesa.bweight[i][j];//需要将速度沿法线分解，还没分
				v_y += particlesa.vy[jj] * particlesa.bweight[i][j];				
			}
			p = vcc > 0.00000001 ? p / vcc : 0;//p有值
			v_x = vcc > 0.00000001 ? v_x / vcc : 0;//v_x一直为0！待解决
			v_y = vcc > 0.00000001 ? v_y / vcc : 0;				
			double vx0 = 0;
			double vy0 = 0;
			particlesa.vcc[i] = vcc;
			particlesa.press[i] = p;
			particlesa.vx[i] = 2.0*vx0 - v_x;//无滑移，要改成径向，切向
			particlesa.vy[i] = 2.0*vy0 - v_y;							
		}
		//this->updateGhostInfo();
		//this->Ghost2bufferInfo();
		
		//shap matrix   高阶M+一阶m矩阵
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			//particle* ii = particles[i];

			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;//边界粒子没有计算M矩阵
			const double rho_i = particlesa.rho[i];
			const double hsml = particlesa.hsml[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];
			double k_11 = 0;//K
			double k_12 = 0;
			double k_21 = 0;
			double k_22 = 0;
			double m_11 = 0;
			double m_12 = 0;
			double m_21 = 0;
			double m_22 = 0;
			double matrix[MAX_SIZE][MAX_SIZE];//M矩阵
			double inverse[MAX_SIZE][MAX_SIZE];//逆矩阵			
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
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:a00,a01,a02,a03,a04,a11,a14,a22,a23,a24,a34,a44)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx ;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx ;//xi(1)
				}
				const double rho_j = particlesa.rho[jj];
				const double massj = particlesa.mass[jj];
				//一阶
				//k_11 += dx * dx * massj / rho_j * particlesa.bweight[i][j];	//k11
				//k_12 += dx * dy * massj / rho_j * particlesa.bweight[i][j];//k_12=k_21
				//k_21 += dy * dx * massj / rho_j * particlesa.bweight[i][j];
				//k_22 += dy * dy * massj / rho_j * particlesa.bweight[i][j];
				//二阶 matrix[0][0]=k_11
				/*matrix[0][0] += dx * dx * massj / rho_j * particlesa.bweight[i][j];
				matrix[0][1] += dx * dy * massj / rho_j * particlesa.bweight[i][j];
				matrix[0][2] += dx * dx * dx * massj / rho_j * particlesa.bweight[i][j];
				matrix[0][3] += dx * dx * dy * massj / rho_j * particlesa.bweight[i][j];
				matrix[0][4] += dx * dy * dy * massj / rho_j * particlesa.bweight[i][j];*/
				a00 += dx * dx * massj / rho_j * particlesa.bweight[i][j];// = k11
				a01 += dx * dy * massj / rho_j * particlesa.bweight[i][j];// = k12 = k_21
				a02 += dx * dx * dx * massj / rho_j * particlesa.bweight[i][j];
				a03 += dx * dx * dy * massj / rho_j * particlesa.bweight[i][j];
				a04 += dx * dy * dy * massj / rho_j * particlesa.bweight[i][j];
				/*matrix[1][0] = matrix[0][1];
				matrix[1][1] += dy * dy * massj / rho_j * particlesa.bweight[i][j];
				matrix[1][2] = matrix[0][3];
				matrix[1][3] = matrix[0][4];
				matrix[1][4] += dy * dy * dy * massj / rho_j * particlesa.bweight[i][j];*/
				a11 += dy * dy * massj / rho_j * particlesa.bweight[i][j];// = k22
				a14 += dy * dy * dy * massj / rho_j * particlesa.bweight[i][j];
				/*matrix[2][0] = matrix[0][2];
				matrix[2][1] = matrix[1][2];
				matrix[2][2] += dx * dx * dx * dx * massj / rho_j * particlesa.bweight[i][j];
				matrix[2][3] += dx * dx * dx * dy * massj / rho_j * particlesa.bweight[i][j];
				matrix[2][4] += dx * dx * dy * dy * massj / rho_j * particlesa.bweight[i][j];*/
				a22 += dx * dx * dx * dx * massj / rho_j * particlesa.bweight[i][j];
				a23 += dx * dx * dx * dy * massj / rho_j * particlesa.bweight[i][j];
				a24 += dx * dx * dy * dy * massj / rho_j * particlesa.bweight[i][j];
				/*matrix[3][0] = matrix[0][3];
				matrix[3][1] = matrix[1][3];
				matrix[3][2] = matrix[2][3];
				matrix[3][3] = matrix[2][4];
				matrix[3][4] += dx * dy * dy * dy * massj / rho_j * particlesa.bweight[i][j];*/
				a34 += dx * dy * dy * dy * massj / rho_j * particlesa.bweight[i][j];
				/*matrix[4][0] = matrix[0][4];
				matrix[4][1] = matrix[1][4];
				matrix[4][2] = matrix[2][4];
				matrix[4][3] = matrix[3][4];
				matrix[4][4] += dy * dy * dy * dy * massj / rho_j * particlesa.bweight[i][j];*/
				a44 += dy * dy * dy * dy * massj / rho_j * particlesa.bweight[i][j];
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
			m_11 = matrix[1][1] / det;
			m_12 = -matrix[0][1] / det;
			m_21 = m_12;
			m_22 = matrix[0][0] / det;
			particlesa.m_11[i] = m_11;//k矩阵求逆
			particlesa.m_12[i] = m_12;
			particlesa.m_21[i] = m_21;
			particlesa.m_22[i] = m_22;    
			// 二阶
			//求逆
			inverseMatrix(matrix, inverse, size); 
			/*std::cout << "Inverse matrix of particle" <<i<< std::endl;
			displayMatrix(inverse, size);*/						
			particlesa.M_11[i] = inverse[0][0];
			particlesa.M_12[i] = inverse[0][1];
			particlesa.M_13[i] = inverse[0][2];
			particlesa.M_14[i] = inverse[0][3];
			particlesa.M_15[i] = inverse[0][4];
			particlesa.M_21[i] = inverse[1][0];
			particlesa.M_22[i] = inverse[1][1];
			particlesa.M_23[i] = inverse[1][2];
			particlesa.M_24[i] = inverse[1][3];
			particlesa.M_25[i] = inverse[1][4];
			particlesa.M_31[i] = inverse[2][0];//M的二阶逆矩阵的部分元素
			particlesa.M_32[i] = inverse[2][1];
			particlesa.M_33[i] = inverse[2][2];
			particlesa.M_34[i] = inverse[2][3];
			particlesa.M_35[i] = inverse[2][4];
			particlesa.M_51[i] = inverse[4][0];
			particlesa.M_52[i] = inverse[4][1];
			particlesa.M_53[i] = inverse[4][2];
			particlesa.M_54[i] = inverse[4][3];
			particlesa.M_55[i] = inverse[4][4];
		}//end circle i

		// boundary viscosity due to no-slip condition -----------------------------对边界的计算，算了边界的黏性力
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			//particle* ii = particles[i];
			if (particlesa.btype[i] != sph::BoundaryType::Boundary) continue;//存在一个边界缺失的问题：边界的邻域是不全的，所以算的边界的黏性力其实也不准确
			const double rho_i = particlesa.rho[i];
			const double hsml = particlesa.hsml[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];

#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
			//--------wMxij-------
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}

				particlesa.wMxijx[i][j] = (particlesa.m_11[i] * dx + particlesa.m_12[i] * dy) * particlesa.bweight[i][j]; //particlesa.m_12[i]可能等于0
				particlesa.wMxijy[i][j] = (particlesa.m_21[i] * dx + particlesa.m_22[i] * dy) * particlesa.bweight[i][j];
			}
			//if (particlesa.btype[i] != sph::BoundaryType::Boundary) continue;

			double epsilon_2_11 = 0;
			double epsilon_2_12 = 0;
			double epsilon_2_21 = 0;
			double epsilon_2_22 = 0;     //=dudx11
			double epsilon_3 = 0;
			double epsilon_dot11 = 0;
			double epsilon_dot12 = 0;
			double epsilon_dot21 = 0;
			double epsilon_dot22 = 0;
			double tau11 = 0;
			double tau12 = 0;
			double tau21 = 0;
			double tau22 = 0;

			const double p_i = particlesa.press[i];

#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:epsilon_2_11,epsilon_2_12,epsilon_2_21,epsilon_2_22)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.btype[jj] == sph::BoundaryType::Boundary) continue;
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}
				const double r2 = dx * dx + dy * dy;
				const double rho_j = particlesa.rho[jj];
				const double rho_ij = rho_j - rho_i;
				const double diffx = rho_ij * dx / r2;
				const double diffy = rho_ij * dx / r2;
				//const double dvx = particlesa.vx[jj] - ((ii)->bctype == BoundaryConditionType::NoSlip ? particlesa.vx[i] : 0);
				//const double dvy = particlesa.vy[jj] - ((ii)->bctype == BoundaryConditionType::NoSlip ? particlesa.vy[i] : 0);
				const double dvx = particlesa.vx[jj] - particlesa.vx[i];
				const double dvy = particlesa.vy[jj] - particlesa.vy[i];
				const double massj = particlesa.mass[jj];

				double epsilon_dot11 = 0;
				double epsilon_dot12 = 0;
				double epsilon_dot21 = 0;
				double epsilon_dot22 = 0;
				double tau11 = 0;
				double tau12 = 0;
				double tau21 = 0;
				double tau22 = 0;
				double temp_ik11 = 0;
				double temp_ik12 = 0;
				double temp_ik21 = 0;
				double temp_ik22 = 0;

				const double A_1 = dvx * particlesa.wMxijx[i][j] + dvy * particlesa.wMxijy[i][j];

				epsilon_2_11 += dvx * particlesa.wMxijx[i][j] * massj / rho_j;//du/dx
				epsilon_2_12 += dvx * particlesa.wMxijy[i][j] * massj / rho_j;//du/dy
				epsilon_2_21 += dvy * particlesa.wMxijx[i][j] * massj / rho_j;//dv/dx
				epsilon_2_22 += dvy * particlesa.wMxijy[i][j] * massj / rho_j;//dv/dy			
			}//end circle j

			 //epsilon_second= -1./3.* epsilon_3;//*��λ����

				//epsilon_temp(11)= epsilon_2(11) * particlesa.m_11[i] + epsilon_2(12) * particlesa.m_21[i];
				//epsilon_temp(12)= epsilon_2(11) * particlesa.m_12[i] + epsilon_2(12) * particlesa.m_22[i];
				//epsilon_temp(21)= epsilon_2(21) * particlesa.m_11[i] + epsilon_2(22) * particlesa.m_21[i];
				//epsilon_temp(22)= epsilon_2(21) * particlesa.m_12[i] + epsilon_2(22) * particlesa.m_22[i];

				//epsilon_dot=0.5*(epsilon_temp+transpose(epsilon_temp))+epsilon_second  !�ܵ�epsilon,����������ȣ�û��epsilon_second
			/*epsilon_dot11 = epsilon_2_11 * particlesa.m_11[i] + epsilon_2_12 * particlesa.m_21[i] - 1. / 3. * epsilon_3;
			epsilon_dot12 = ((epsilon_2_11 * particlesa.m_12[i] + epsilon_2_12 * particlesa.m_22[i]) + (epsilon_2_21 * particlesa.m_11[i] + epsilon_2_22 * particlesa.m_21[i])) * 0.5;
			epsilon_dot21 = epsilon_dot12;
			epsilon_dot22 = epsilon_2_21 * particlesa.m_12[i] + epsilon_2_22 * particlesa.m_22[i] - 1. / 3. * epsilon_3;*/
			epsilon_dot11 = epsilon_2_11 ;
			epsilon_dot12 = 0.5*(epsilon_2_12 + epsilon_2_21);
			epsilon_dot21 = epsilon_dot12;
			epsilon_dot22 = epsilon_2_22 ;
			//边界粒子的物理黏性项tau： 比较重要！
			particlesa.tau11[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot11;//边界粒子的黏性力
			particlesa.tau12[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot12;
			particlesa.tau21[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot21;
			particlesa.tau22[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot22;

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
			const double kenergy = (particlesa.vx[i] * particlesa.vx[i] + particlesa.vy[i] * particlesa.vy[i]) * 0.5;
			particlesa.turb11[i] = 2.0 * mut * dvdx11 * rho_i - 2.0 / 3.0 * kenergy * rho_i;//����tao= ( 2*Vt*Sij-2/3Ksps*�����˺��� )*rho_i// �൱��turbulence(:,:,i)
			particlesa.turb12[i] = 2.0 * mut * dvdx12 * rho_i;
			particlesa.turb21[i] = 2.0 * mut * dvdx21 * rho_i;
			particlesa.turb22[i] = 2.0 * mut * dvdx22 * rho_i - 2.0 / 3.0 * kenergy * rho_i;
		}//end circle i

		// density 连续性方程
		//const double chi = 0.2;


		//--------------------------- for fluid particles（包括inlet）-----传热方程、连续性方程、动量方程中的黏性切应力、湍流应力------------------------------------------------
		//---------注：传热方程由于需要求温度的拉普拉斯算子，因此用到了二阶的ULPH理论，用到五阶形状张量矩阵M；
		//---------注：而连续性方程和动量方程只需要用到一阶ULPH理论，因此只用到二阶形状张量矩阵。
		//
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			//particle* ii = particles[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];
			const double rho_i = particlesa.rho[i];
			const double hsml = particlesa.hsml[i];
			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			//---------density----------
			
			double drhodt = 0;
			double drhodiff = 0;
			//---------internal force-------柯西应力：压力+物理黏性力			
			double epsilon_2_11 = 0;
			double epsilon_2_12 = 0;
			double epsilon_2_21 = 0;
			double epsilon_2_22 = 0;     //=dudx11
			//double epsilon_3 = 0;
			double epsilon_dot11 = 0;
			double taoxx = 0;
			double taoyy = 0;
			double epsilon_dot12 = 0;
			double epsilon_dot21 = 0;
			double epsilon_dot22 = 0;
			double tau11 = 0;
			double tau12 = 0;
			double tau21 = 0;
			double tau22 = 0;
			const double p_i = particlesa.press[i];
			//--------turbulence---------
			double dudx11 = 0;//k
			double dudx12 = 0;
			double dudx21 = 0;
			double dudx22 = 0;
			double sij = 0;
			
			//----temperature-----目前简化版的，ki和ci为常熟
			const double ki = sph::Fluid::coefficient_heat(particlesa.fltype[i]);
			const double ci = sph::Fluid::specific_heat(particlesa.fltype[i]);
			double vcc_temperature_t = 0;
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
			//--------wMxij-------
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				//周期边界
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					 dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					 dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;;//xi(1)
				}

				particlesa.wMxijx[i][j] = (particlesa.m_11[i] * dx + particlesa.m_12[i] * dy) * particlesa.bweight[i][j];
				particlesa.wMxijy[i][j] = (particlesa.m_21[i] * dx + particlesa.m_22[i] * dy) * particlesa.bweight[i][j];
			}

#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:vcc_temperature_t,epsilon_2_11,epsilon_2_12,epsilon_dot21,epsilon_2_22)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double rho_j = particlesa.rho[jj];
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}
				const double r2 = dx * dx + dy * dy;
				const double rho_ij = rho_j - rho_i;				
				const double dvx = particlesa.vx[jj] - particlesa.vx[i];
				const double dvy = particlesa.vy[jj] - particlesa.vy[i];
				const double massj = particlesa.mass[jj];
				//---------density---------- 
				//const double vcc_diff = diffx * particlesa.wMxijx[i][j] + diffy * particlesa.wMxijy[i][j];//有问题
				//const double vcc_rho = dvx * particlesa.wMxijx[i][j] + dvy * particlesa.wMxijy[i][j];//速度的散度
				/*const double b1wMq = particlesa.m_11[i] * dx + particlesa.m_12[i] * dy + (ii)->m_13 * dx * dx + (ii)->m_14 * dx * dy + (ii)->m_15 * dy * dy;
				const double b2wMq = particlesa.m_21[i] * dx + particlesa.m_22[i] * dy + (ii)->m_23 * dx * dx + (ii)->m_24 * dx * dy + (ii)->m_25 * dy * dy;
				const double vcc_diff = diffx * b1wMq + diffy * b2wMq;
				const double vcc_rho = dvx * b1wMq + dvy * b2wMq;*/
				//const double chi = 0.2;
				//drhodiff += particlesa.bweight[i][j] * chi * particlesa.c0[i] * hsml * vcc_diff * massj / rho_j;//密度耗散项the density diffusion term
				//divv += vcc_rho * massj / rho_j;//速度散度
				//drhodt += - rho_i * vcc_rho * massj / rho_j;//连续性方程已经结束！
				//--------------------------------temperature---
				const double kj = sph::Fluid::coefficient_heat(particlesa.fltype[jj]);
				const double kij = 0.5 * (ki + kj);
				const double tij = particlesa.temperature[jj] - particlesa.temperature[i];				
				const double vcc_temperature_xx = particlesa.M_31[i] * dx + particlesa.M_32[i] * dy + particlesa.M_33[i] * dx * dx + particlesa.M_34[i] * dx * dy + particlesa.M_35[i] * dy * dy;//高阶
				const double vcc_temperature_yy = particlesa.M_51[i] * dx + particlesa.M_52[i] * dy + particlesa.M_53[i] * dx * dx + particlesa.M_54[i] * dx * dy + particlesa.M_55[i] * dy * dy;
				
				vcc_temperature_t += 2.0*(vcc_temperature_xx + vcc_temperature_yy) * tij * particlesa.bweight[i][j] * kij * particlesa.mass[jj] / rho_j;//温度的拉普拉斯运算，传热方程已经结束！
				//----------柯西应力--------   
				double epsilon_dot11 = 0;
				double epsilon_dot12 = 0;
				double epsilon_dot21 = 0;
				double epsilon_dot22 = 0;
				double tau11 = 0;
				double tau12 = 0;
				double tau21 = 0;
				double tau22 = 0;
				double temp_ik11 = 0;
				double temp_ik12 = 0;
				double temp_ik21 = 0;
				double temp_ik22 = 0;
				//const double A_1 = dvx * particlesa.wMxijx[i][j] + dvy * particlesa.wMxijy[i][j];

				//速度算子：2*2的张量
				epsilon_2_11 += dvx * particlesa.wMxijx[i][j] * massj / rho_j;//du/dx
				epsilon_2_12 += dvx * particlesa.wMxijy[i][j] * massj / rho_j;//du/dy
				epsilon_2_21 += dvy * particlesa.wMxijx[i][j] * massj / rho_j;//dv/dx
				epsilon_2_22 += dvy * particlesa.wMxijy[i][j] * massj / rho_j;//dv/dy  				
			}	//end circle j
			//---速度散度			
			particlesa.divvel[i] = epsilon_2_11 + epsilon_2_22;//弱可压缩，散度应该不会很大
			//-------------------------------------黏性力----是一个2*2的张量  
			/*taoxx = epsilon_2_11 - 1. / 3. * particlesa.divvel[i];
			taoyy = epsilon_2_22 - 1. / 3. * particlesa.divvel[i];
			epsilon_dot12 = (epsilon_2_12 + epsilon_2_21);
			epsilon_dot21 = epsilon_dot12;	*/			
			//particlesa.tau11[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * taoxx;//taoxx
			//particlesa.tau12[i] = sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot12;//taoxy: Viscosity(dv/dx+du/dy)
			//particlesa.tau21[i] = particlesa.tau12[i];//taoyx=taoxy
			//particlesa.tau22[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * taoyy;//taoyy
			//---------对于不可压缩，不再需要求粘性力，再求NS方程了。动量方程中只需要速度的拉普拉斯算子，暂时用一阶算法去求
			//particlesa.tau11[i] = epsilon_2_11;//这里的tau11代表速度的偏导
			//particlesa.tau12[i] = epsilon_2_12;//
			//particlesa.tau21[i] = epsilon_2_21;//
			//particlesa.tau22[i] = epsilon_2_22;//
			// 
			//-----现求柯西应力：sigema = p + tao
			//对于不可压缩：黏性切应力 = 2. * sph::Fluid::Viscosity(particlesa.fltype[i])* 应变
			//应变为:[ du/dx 0.5*(du/dy+dv/dx) 0.5*(du/dy+dv/dx) dv/dy ]
			particlesa.tau11[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_2_11;//这里的tau11代表速度的偏导
			particlesa.tau12[i] = sph::Fluid::Viscosity(particlesa.fltype[i]) * (epsilon_2_12+ epsilon_2_21);//
			particlesa.tau21[i] = particlesa.tau12[i];//
			particlesa.tau22[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_2_22;//
			
			//-----------------------------涡量 W=0.5(dudx-dvdy)    应变率张量（正应力张量）: S=0.5(dudx+dvdy) divvel
			double vort = 0.5 * (epsilon_2_21 - epsilon_2_12);
			//particlesa.vort[i] = (epsilon_2_21 - epsilon_2_12);
			//运用Q准则 Q=0.5*(sqrt(W)-sqrt(S))
			//particlesa.vort[i] =0.5* (vort* vort - (0.5* particlesa.divvel[i])* (0.5 * particlesa.divvel[i]));
			particlesa.vort[i] = vort;
			//----------------------------Turbulence----先不管
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
			const double kenergy = (particlesa.vx[i] * particlesa.vx[i] + particlesa.vy[i] * particlesa.vy[i]) * 0.5;
			particlesa.turb11[i] = 2.0 * mut * dvdx11 * rho_i - 2.0 / 3.0 * kenergy * rho_i;//����tao= ( 2*Vt*Sij-2/3Ksps*�����˺��� )*rho_i// �൱��turbulence(:,:,i)
			particlesa.turb12[i] = 2.0 * mut * dvdx12 * rho_i;
			particlesa.turb21[i] = 2.0 * mut * dvdx21 * rho_i;
			particlesa.turb22[i] = 2.0 * mut * dvdx22 * rho_i - 2.0 / 3.0 * kenergy * rho_i;
			//加上耗散函数：
			const double fai = sph::Fluid::Viscosity(particlesa.fltype[i]) * (2 * epsilon_2_11 * epsilon_2_11 +
				2 * epsilon_2_22 * epsilon_2_22 + (epsilon_2_12 + epsilon_2_21) * (epsilon_2_12 + epsilon_2_21));
			//连续性方程，温度方程
			if (particlesa.ftype[i] != sph::FixType::Fixed) {
				//particlesa.drho[i] = drhodt + drhodiff;
				particlesa.drho[i] = -particlesa.rho[i] * particlesa.divvel[i];
				//double temp_t = vcc_temperature_t / particlesa.rho[i] / ci;
				//加耗散函数：
				particlesa.temperature_x[i] = fai;//用来查看值
				double temp_t = (vcc_temperature_t + fai) / particlesa.rho[i] / ci;
				//if (temp_t == temp_t)
				particlesa.temperature_t[i] = temp_t;				
			}
			else {
				particlesa.temperature_t[i] = 0;
				particlesa.drho[i] = 0;
			}
		}//end circle i
		// for fluid particles（包括inlet）       -----动量方程-----
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];

			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			const double rho_i = particlesa.rho[i];
			const double hsml = particlesa.hsml[i];
			const double p_i = particlesa.press[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];
			const double c0 = particlesa.c0[i];
			const double c = (ii)->c;
			const double massi = particlesa.mass[i];
			//------对柯西应力求导-------			
			double temp_sigemax = 0;//-px，求对x的偏导
			double temp_sigemay = 0;//-py，求对y的偏导
			double temp_taoxx = 0;
			double temp_taoyy = 0;
			double temp_taoxy = 0;//黏性力分量:taoxy对y的偏导
			double temp_taoyx = 0;//黏性力分量:taoyx对x的偏导
			//--------------------
			double turbx = 0;
			double turby = 0;
			double turbxx = 0;
			double turbyy = 0;
			double turbxy = 0;//黏性力分量:taoxy对y的偏导
			double turbyx = 0;//黏性力分量:taoyx对x的偏导
			//----------artificial viscosity---
			double avx = 0;
			double avy = 0;
			
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:temp_sigemax,temp_sigemay,turbxx,turbxy,turbyx,turbyy,avx,avy)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;

				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}
				const double hsmlj = particlesa.hsml[jj];
				const double mhsml = (hsml + hsmlj) / 2;
				const double r2 = dx * dx + dy * dy;
				const double rho_j = particlesa.rho[jj];
				const double massj = particlesa.mass[jj];

				//particlesa.wMxijx[i][j] = (particlesa.m_11[i] * dx + particlesa.m_12[i] * dy) * particlesa.bweight[i][j];
				//particlesa.wMxijy[i][j] = (particlesa.m_21[i] * dx + particlesa.m_22[i] * dy) * particlesa.bweight[i][j];
				
				const double wMxijxV_iix = particlesa.wMxijx[i][j] * massj / rho_j;
				const double wMxijyV_iiy = particlesa.wMxijy[i][j] * massj / rho_j;
				//const double wMxijxV_jjx = (jj)->wMxijx[j] * massj / rho_j;//(jj)->wMxijx[j] 可能错了
				//const double wMxijyV_jjy = (jj)->wMxijy[j] * massj / rho_j;
				const double wMxijxV_jjx = (particlesa.m_11[jj] * dx + particlesa.m_12[jj] * dy) * particlesa.bweight[i][j] * massj / rho_j;
				const double wMxijyV_jjy = (particlesa.m_21[jj] * dx + particlesa.m_22[jj] * dy) * particlesa.bweight[i][j] * massj / rho_j;
				//-p的梯度
				/*temp_px += ( p_i - particlesa.press[jj] ) * wMxijxV_iix;
				temp_py += ( p_i - particlesa.press[jj] ) * wMxijyV_iiy;*/
				//黏性切应力分量求导
				//temp_taoxx += (particlesa.tau11[jj] - particlesa.tau11[i]) * wMxijxV_iix;//du/dx：uxx
				//temp_taoyy += (particlesa.tau22[jj] - particlesa.tau22[i]) * wMxijyV_iiy;//dv/dy：vyy
				//temp_taoxy += (particlesa.tau12[jj] - particlesa.tau12[i]) * wMxijxV_iix;//du/dy: uyy
				//temp_taoyx += (particlesa.tau21[jj] - particlesa.tau21[i]) * wMxijyV_iiy;//dv/dx: vxx  
				// 
				//---------------现用柯西应力，态来表示动量方程---------------
				//sigema_j = [ - p+txx txy tyx p+tyy ]
				//j粒子----------------有边界粒子
				const double sigema_j11 = -particlesa.press[jj] + particlesa.tau11[jj];//湍流力一样在这里加
				const double sigema_j12 = particlesa.tau12[jj];
				const double sigema_j21 = particlesa.tau21[jj];
				const double sigema_j22 = -particlesa.press[jj] + particlesa.tau22[jj];
				//i粒子
				const double sigema_i11 = -p_i + particlesa.tau11[i];
				const double sigema_i12 = particlesa.tau12[i];
				const double sigema_i21 = particlesa.tau21[i];
				const double sigema_i22 = -p_i + particlesa.tau22[i];
				temp_sigemax += (sigema_j11 * wMxijxV_jjx + sigema_j12 * wMxijyV_jjy)+ (sigema_i11 * wMxijxV_iix + sigema_j12 * wMxijyV_iiy);
				temp_sigemay += (sigema_j21 * wMxijxV_jjx + sigema_j22 * wMxijyV_jjy)+ (sigema_i21 * wMxijxV_iix + sigema_j22 * wMxijyV_iiy);

				//-----turbulence-------其实也可以将湍流力放到上面的内力一起去				
				//const double turb_ij11 = particlesa.turb11[i] * particlesa.m_11[i] + particlesa.turb12[i] * particlesa.m_21[i] + particlesa.turb11[jj] * particlesa.m_11[jj] + particlesa.turb12[jj] * particlesa.m_21[jj];
				//const double turb_ij12 = particlesa.turb11[i] * particlesa.m_12[i] + particlesa.turb12[i] * particlesa.m_22[i] + particlesa.turb11[jj] * particlesa.m_12[jj] + particlesa.turb12[jj] * particlesa.m_22[jj];
				//const double turb_ij21 = particlesa.turb21[i] * particlesa.m_11[i] + particlesa.turb22[i] * particlesa.m_21[i] + particlesa.turb21[jj] * particlesa.m_11[jj] + particlesa.turb22[jj] * particlesa.m_21[jj];
				//const double turb_ij22 = particlesa.turb21[i] * particlesa.m_12[i] + particlesa.turb22[i] * particlesa.m_22[i] + particlesa.turb21[jj] * particlesa.m_12[jj] + particlesa.turb22[jj] * particlesa.m_22[jj];

				//const double t1 = (turb_ij11 * dx + turb_ij12 * dy) * particlesa.bweight[i][j];
				//const double t2 = (turb_ij21 * dx + turb_ij22 * dy) * particlesa.bweight[i][j];

				//turbx += t1 * massj / rho_j;
				//turby += t2 * massj / rho_j;//����over
				turbxx += (particlesa.turb11[jj] - particlesa.turb11[i]) * wMxijxV_iix;
				turbyy += (particlesa.turb22[jj] - particlesa.turb22[i]) * wMxijyV_iiy;
				turbxy += (particlesa.turb12[jj] - particlesa.turb12[i]) * wMxijyV_iiy;
				turbyx += (particlesa.turb21[jj] - particlesa.turb21[i]) * wMxijxV_iix;
				//--------artificial viscosity-------
				const double dvx = particlesa.vx[jj] - particlesa.vx[i];    //(Vj-Vi)
				const double dvy = particlesa.vy[jj] - particlesa.vy[i];
				double muv = 0;
				double piv = 0;
				const double cj = particlesa.c[jj];
				const double vr = dvx * dx + dvy * dy;     //(Vj-Vi)(Rj-Ri)
				const double mc = 0.5 * (cj + (ii)->c);
				const double mrho = 0.5 * (rho_j + particlesa.rho[i]);				

				if (vr < 0) {
					muv = mhsml * vr / (r2 + mhsml * mhsml * 0.01);//FAI_ij < 0
					//piv = (0.5 * muv - 0.5 * mc) * muv / mrho;//beta项-alpha项
					piv = (0.5 * muv - 1.0 * mc) * muv / mrho;//piv > 0
					//piv = (0.5 * muv) * muv / mrho;//只有beta项，加速度会一直很大，停不下来，穿透。
				}
				//人工黏性项，好像过大
				avx += - massj * piv * particlesa.wMxijx[i][j];
				avy += - massj * piv * particlesa.wMxijy[i][j];
			}
			
			(ii)->fintx = temp_sigemax / rho_i;//不对
			(ii)->finty = temp_sigemay /rho_i;
			(ii)->turbx = turbxx + turbxy;
			(ii)->turby = turbyy + turbyx;
			(ii)->avx = avx;
			(ii)->avy = avy;
			///if (particlesa.iotype[i] == InoutType::Fluid|| particlesa.iotype[i] == InoutType::Outlet) 
			if (particlesa.ftype[i] != sph::FixType::Fixed)//inlet粒子的加速度场为0，只有初始速度。outlet粒子呢？
			{
				//particlesa.ax[i] = (ii)->fintx + avx + (ii)->turbx;
				//particlesa.ay[i] = (ii)->finty + avy + (ii)->turby;
				particlesa.ax[i] = (ii)->fintx + avx;
				particlesa.ay[i] = (ii)->finty + avy;	
				//particlesa.ay[i] = 0;
				//particlesa.ax[i] = (ii)->fintx + avx + turbx + (ii)->replx;
				//particlesa.ay[i] = (ii)->finty + avy + turby + (ii)->reply + gravity;				
			}
			
		}

		const std::clock_t end = std::clock();
		if (time_measure) std::cout << "single step costs " << double(end - begin) / TICKS_PER_SEC << "s\n";
	}//end single_step

	//仅用于求温度变化率(一阶)不对，放弃
	inline void domain::single_step_temperature()
	{
		const std::clock_t begin = std::clock();
		this->updateWeight();
		// boundary   		
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (particlesa.btype[i] != sph::BoundaryType::Boundary) continue;
			if (particlesa.iotype[i] == sph::InoutType::Buffer) continue;
			double vcc = 0;			
			//边界的温度梯度
			double t_x = 0, t_y = 0;
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:vcc)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				//if (particlesa.btype[i] == particlesa.btype[jj]) continue;//Buffer粒子也在这里，导致流出边界固壁的压力不正常
				if (particlesa.ftype[jj] == sph::FixType::Fixed) continue;//其他固壁粒子不参与，ghost不参与，buffer参与								
				vcc += particlesa.bweight[i][j];
				t_x += particlesa.temperature_x[jj] * particlesa.bweight[i][j];
				t_y += particlesa.temperature_y[jj] * particlesa.bweight[i][j];
			}
			t_x = 0;//边界是恒温
			t_y = vcc > 0.00000001 ? (t_y / vcc) : 0;//最外层边界的t_y比第一层流体粒子的t_y小，所以导致第二层粒子的温度高		
			particlesa.temperature_x[i] = t_x;
			particlesa.temperature_y[i] = t_y;
		}
		// inlet  目前流入处的温度梯度不对 		
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];			
			if (particlesa.iotype[i] != sph::InoutType::Inlet) continue;
			double vcc = 0;		
			double t_x = 0, t_y = 0;
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:vcc)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.iotype[jj] == sph::InoutType::Inlet) continue;//其他Inlet粒子不参与								
				vcc += particlesa.bweight[i][j];
				t_x += particlesa.temperature_x[jj] * particlesa.bweight[i][j];
				t_y += particlesa.temperature_y[jj] * particlesa.bweight[i][j];
			}
			t_x = vcc > 0.00000001 ? t_x / vcc : 0;
			t_y = vcc > 0.00000001 ? t_y / vcc : 0;				
			particlesa.temperature_x[i] = t_x;
			particlesa.temperature_y[i] = t_y;
		}
		this->updateGhostInfo();
		this->Ghost2bufferInfo();
		//shap matrix
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];

			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			const double rho_i = particlesa.rho[i];
			const double hsml = particlesa.hsml[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];

			double k_11 = 0;//K
			double k_12 = 0;
			double k_21 = 0;
			double k_22 = 0;
			double m_11 = 0;
			double m_12 = 0;
			double m_21 = 0;
			double m_22 = 0;

			//k
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:k_11,k_12,k_21,k_22)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				const double dx = xj - xi;
				const double dy = yj - yi;
				const double rho_j = particlesa.rho[jj];
				const double massj = particlesa.mass[jj];
				k_11 += dx * dx * massj / rho_j * particlesa.bweight[i][j];	//k11
				k_12 += dx * dy * massj / rho_j * particlesa.bweight[i][j];//k_12=k_21
				k_21 += dy * dx * massj / rho_j * particlesa.bweight[i][j];
				k_22 += dy * dy * massj / rho_j * particlesa.bweight[i][j];
			}
			const double det = k_11 * k_22 - k_12 * k_21;//不等于0
			m_11 = k_22 / det;
			m_12 = -k_12 / det;
			m_21 = -k_21 / det;
			m_22 = k_11 / det;
			particlesa.m_11[i] = m_11;//k矩阵求逆
			particlesa.m_12[i] = m_12;
			particlesa.m_21[i] = m_21;
			particlesa.m_22[i] = m_22;    //shape_inverse(:,:,i),iѭ����һ��

		}
		
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];		
			//----temperature----grad(T)
			double vcc_temperature_x = 0;
			double vcc_temperature_y = 0;			
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
//--------wMxij-------
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				const double dx = xj - xi;
				const double dy = yj - yi;
				particlesa.wMxijx[i][j] = (particlesa.m_11[i] * dx + particlesa.m_12[i] * dy) * particlesa.bweight[i][j];
				particlesa.wMxijy[i][j] = (particlesa.m_21[i] * dx + particlesa.m_22[i] * dy) * particlesa.bweight[i][j];
			}
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:vcc_temperature_x,vcc_temperature_y)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;				
				const double rho_j = particlesa.rho[jj];	
				const double massj = particlesa.mass[jj];
				//----temperature-----grad(T)
				//const double tj = particlesa.temperature[jj];
				//const double ti = particlesa.temperature[i];
				const double tij = particlesa.temperature[jj] - particlesa.temperature[i];
				vcc_temperature_x += tij * particlesa.wMxijx[i][j] * massj / rho_j;//流体粒子的温度梯度
				vcc_temperature_y += tij * particlesa.wMxijy[i][j] * massj / rho_j;				
			}//end circle j			
			if (particlesa.iotype[i] == InoutType::Fluid) {

				particlesa.temperature_x[i] = vcc_temperature_x;
				particlesa.temperature_y[i] = vcc_temperature_y;
			}

		}//end circle i		

#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;			
			//----temperature-----目前简化版的，ki和ci为常熟
			const double ki = sph::Fluid::coefficient_heat(particlesa.fltype[i]);
			const double ci = sph::Fluid::specific_heat(particlesa.fltype[i]);
			double vcc_temperature_t = 0;									
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:vcc_temperature_t)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;	
				const double rho_j = particlesa.rho[jj];
				//----temperature-----div(ki*grad(T))
				const double kj = sph::Fluid::coefficient_heat(particlesa.fltype[jj]);
				const double kij = 0.5 * (ki + kj);				
				const double temperature_dvx = particlesa.temperature_x[jj] - particlesa.temperature_x[i];//dTxj-dTxi 温度梯度的再运算
				const double temperature_dvy = particlesa.temperature_y[jj] - particlesa.temperature_y[i];
				const double vcc_temperature_tt = temperature_dvx * particlesa.wMxijx[i][j] + temperature_dvy * particlesa.wMxijy[i][j];
				vcc_temperature_t += vcc_temperature_tt * kij * particlesa.mass[jj] / rho_j;//温度的拉普拉斯运算
			}			
			if (particlesa.iotype[i] == InoutType::Fluid) {
				double temp_t = vcc_temperature_t / particlesa.rho[i] / ci;
				if (temp_t < 0)temp_t = 0;
				particlesa.temperature_t[i] = temp_t;
			}
			else {				
				particlesa.temperature_t[i] = 0;
			}
		}
		const std::clock_t end = std::clock();
		if (time_measure) std::cout << "single step costs " << double(end - begin) / TICKS_PER_SEC << "s\n";
	}
	//不考虑温度 加上湍流，改成态，相比温度，不需要求5阶的M矩阵，只需求2阶矩阵
	inline void domain::single_step()  //
	{
		const std::clock_t begin = std::clock();

		singlestep_rhoeos_dev0(particleNum(), particlesa.btype, particlesa.rho, particlesa.rho_min, particlesa.c0, particlesa.rho0, particlesa.gamma, particlesa.back_p, particlesa.press);

		this->updateWeight();

		// boundary pressure and 速度     		
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			//particle* ii = particles[i];
			if (particlesa.btype[i] != sph::BoundaryType::Boundary) continue;
			//if (particlesa.iotype[i] == sph::InoutType::Buffer) continue;			
			double p = 0, vcc = 0;
			double v_x = 0, v_y = 0;
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:p,vcc,v_x,v_y)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				//if (particlesa.btype[i] == particlesa.btype[jj]) continue;//Buffer粒子也在这里，导致流出边界固壁的压力不正常
				if (particlesa.ftype[jj] == sph::FixType::Fixed) continue;//其他固壁粒子不参与，ghost不参与，buffer参与
				const double mass_j = particlesa.mass[jj];
				const double rho_j = particlesa.rho[jj];
				const double p_k = particlesa.press[jj];
				p += p_k * particlesa.bweight[i][j];//
				vcc += particlesa.bweight[i][j];
				v_x += particlesa.vx[jj] * particlesa.bweight[i][j];//需要将速度沿法线分解，还没分
				v_y += particlesa.vy[jj] * particlesa.bweight[i][j];
			}
			p = vcc > 0.00000001 ? p / vcc : 0;//p有值
			v_x = vcc > 0.00000001 ? v_x / vcc : 0;//v_x一直为0！待解决
			v_y = vcc > 0.00000001 ? v_y / vcc : 0;
			double vx0 = 0;
			double vy0 = 0;
			particlesa.vcc[i] = vcc;
			particlesa.press[i] = p;
			particlesa.vx[i] = 2.0*vx0 - v_x;//无滑移，要改成径向，切向
			particlesa.vy[i] = 2.0*vy0 - v_y;
					
		}
		//this->updateGhostInfo();
		//this->Ghost2bufferInfo();

		//shap matrix
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			//particle* ii = particles[i];

			//if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			//const double rho_i = particlesa.rho[i];
			//const double hsml = particlesa.hsml[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];

			double k_11 = 0;//K
			double k_12 = 0;
			double k_21 = 0;
			double k_22 = 0;
			double m_11 = 0;
			double m_12 = 0;
			double m_21 = 0;
			double m_22 = 0;

			//k
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:k_11,k_12,k_21,k_22)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}
				const double rho_j = particlesa.rho[jj];
				const double massj = particlesa.mass[jj];
				k_11 += dx * dx * massj / rho_j * particlesa.bweight[i][j];	//k11
				k_12 += dx * dy * massj / rho_j * particlesa.bweight[i][j];//k_12=k_21
				k_21 += dy * dx * massj / rho_j * particlesa.bweight[i][j];
				k_22 += dy * dy * massj / rho_j * particlesa.bweight[i][j];
			}
			const double det = k_11 * k_22 - k_12 * k_21;
			m_11 = k_22 / det;
			m_12 = -k_12 / det;
			m_21 = -k_21 / det;
			m_22 = k_11 / det;
			particlesa.m_11[i] = m_11;//k矩阵求逆
			particlesa.m_12[i] = m_12;
			particlesa.m_21[i] = m_21;
			particlesa.m_22[i] = m_22;    			
		}


		// density
		const double chi = 0.2;

		// boundary viscosity due to no-slip condition 对边界的计算，算了边界的黏性力、湍流力
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (particlesa.btype[i] != sph::BoundaryType::Boundary) continue;
			const double rho_i = particlesa.rho[i];
			const double hsml = particlesa.hsml[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];

#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
			//--------wMxij-------
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}
				particlesa.wMxijx[i][j] = (particlesa.m_11[i] * dx + particlesa.m_12[i] * dy) * particlesa.bweight[i][j];
				particlesa.wMxijy[i][j] = (particlesa.m_21[i] * dx + particlesa.m_22[i] * dy) * particlesa.bweight[i][j];
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
			const double p_i = particlesa.press[i];

#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:epsilon_2_11,epsilon_2_12,epsilon_2_21,epsilon_2_22)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.btype[jj] == sph::BoundaryType::Boundary) continue;
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}
				const double r2 = dx * dx + dy * dy;
				const double rho_j = particlesa.rho[jj];
				const double rho_ij = rho_j - rho_i;
				const double diffx = rho_ij * dx / r2;
				const double diffy = rho_ij * dx / r2;
				//const double dvx = particlesa.vx[jj] - ((ii)->bctype == BoundaryConditionType::NoSlip ? particlesa.vx[i] : 0);
				//const double dvy = particlesa.vy[jj] - ((ii)->bctype == BoundaryConditionType::NoSlip ? particlesa.vy[i] : 0);
				const double dvx = particlesa.vx[jj] - particlesa.vx[i];
				const double dvy = particlesa.vy[jj] - particlesa.vy[i];
				const double massj = particlesa.mass[jj];								
				epsilon_2_11 += dvx * particlesa.wMxijx[i][j] * massj / rho_j;//du/dx
				epsilon_2_12 += dvx * particlesa.wMxijy[i][j] * massj / rho_j;//du/dy
				epsilon_2_21 += dvy * particlesa.wMxijx[i][j] * massj / rho_j;//dv/dx
				epsilon_2_22 += dvy * particlesa.wMxijy[i][j] * massj / rho_j;//dv/dy			
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
			particlesa.tau11[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot11;//边界粒子的黏性力
			particlesa.tau12[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot12;
			particlesa.tau21[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot21;
			particlesa.tau22[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_dot22;

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
			const double kenergy = (particlesa.vx[i] * particlesa.vx[i] + particlesa.vy[i] * particlesa.vy[i]) * 0.5;
			particlesa.turb11[i] = 2.0 * mut * dvdx11 * rho_i - 2.0 / 3.0 * kenergy * rho_i;//����tao= ( 2*Vt*Sij-2/3Ksps*�����˺��� )*rho_i// �൱��turbulence(:,:,i)
			particlesa.turb12[i] = 2.0 * mut * dvdx12 * rho_i;
			particlesa.turb21[i] = 2.0 * mut * dvdx21 * rho_i;
			particlesa.turb22[i] = 2.0 * mut * dvdx22 * rho_i - 2.0 / 3.0 * kenergy * rho_i;
		}//end circle i

		// for fluid particles（包括inlet）
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];

			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			const double rho_i = particlesa.rho[i];
			const double hsml = particlesa.hsml[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];
			
			//---------速度算子----			
			double epsilon_2_11 = 0;
			double epsilon_2_12 = 0;
			double epsilon_2_21 = 0;
			double epsilon_2_22 = 0;     //=dudx11			
			const double p_i = particlesa.press[i];
			
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
//--------wMxij-------
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				//周期边界
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;;//xi(1)
				}

				particlesa.wMxijx[i][j] = (particlesa.m_11[i] * dx + particlesa.m_12[i] * dy) * particlesa.bweight[i][j];
				particlesa.wMxijy[i][j] = (particlesa.m_21[i] * dx + particlesa.m_22[i] * dy) * particlesa.bweight[i][j];
			}



#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:epsilon_2_11,epsilon_2_12,epsilon_2_21,epsilon_2_22)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;
				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}
				const double r2 = dx * dx + dy * dy;
				const double rho_j = particlesa.rho[jj];
				const double rho_ij = rho_j - rho_i;
				const double diffx = rho_ij * dx / r2;
				const double diffy = rho_ij * dy / r2;
				const double dvx = particlesa.vx[jj] - particlesa.vx[i];
				const double dvy = particlesa.vy[jj] - particlesa.vy[i];
				const double massj = particlesa.mass[jj];				
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
				epsilon_2_11 += dvx * particlesa.wMxijx[i][j] * massj / rho_j;//du/dx
				epsilon_2_12 += dvx * particlesa.wMxijy[i][j] * massj / rho_j;//du/dy
				epsilon_2_21 += dvy * particlesa.wMxijx[i][j] * massj / rho_j;//dv/dx
				epsilon_2_22 += dvy * particlesa.wMxijy[i][j] * massj / rho_j;//dv/dy				
			}//end circle j
			//-----------------------------速度散度			
			particlesa.divvel[i] = epsilon_2_11 + epsilon_2_22;//弱可压缩，散度应该不会很大		
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
			particlesa.tau11[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_2_11;//这里的tau11代表速度的偏导
			particlesa.tau12[i] = sph::Fluid::Viscosity(particlesa.fltype[i]) * (epsilon_2_12 + epsilon_2_21);//
			particlesa.tau21[i] = particlesa.tau12[i];//
			particlesa.tau22[i] = 2. * sph::Fluid::Viscosity(particlesa.fltype[i]) * epsilon_2_22;//
			//-----------------------------涡量 W=0.5(dudy-dvdx)    应变率张量（正应力张量）: S=0.5(dudx+dvdy) divvel
			double vort = 0.5 * (epsilon_2_12 - epsilon_2_21);
			//particlesa.vort[i] = (epsilon_2_21 - epsilon_2_12);
			//运用Q准则 Q=0.5*(sqrt(W)-sqrt(S))
			//particlesa.vort[i] =0.5* (vort* vort - (0.5* particlesa.divvel[i])* (0.5 * particlesa.divvel[i]));
			particlesa.vort[i] = vort;
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
			const double kenergy = (particlesa.vx[i] * particlesa.vx[i] + particlesa.vy[i] * particlesa.vy[i]) * 0.5;
			particlesa.turb11[i] = 2.0 * mut * dvdx11 * rho_i - 2.0 / 3.0 * kenergy * rho_i;//����tao= ( 2*Vt*Sij-2/3Ksps*�����˺��� )*rho_i// �൱��turbulence(:,:,i)
			particlesa.turb12[i] = 2.0 * mut * dvdx12 * rho_i;
			particlesa.turb21[i] = 2.0 * mut * dvdx21 * rho_i;
			particlesa.turb22[i] = 2.0 * mut * dvdx22 * rho_i - 2.0 / 3.0 * kenergy * rho_i;
			//连续性方程，温度方程
			if (particlesa.ftype[i] != sph::FixType::Fixed) {
				//particlesa.drho[i] = drhodt + drhodiff;
				particlesa.drho[i] = -particlesa.rho[i] * particlesa.divvel[i];	//没有密度耗散项			
			}
			else {				
				particlesa.drho[i] = 0;
			}			    
			//(ii)->replx = replx;                
			//(ii)->reply = reply;
		}//end circle i
		// for fluid particles（包括inlet）       -----动量方程-----
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (int i = 0; i < particles.size(); i++)
		{
			//particle* ii = particles[i];

			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			const double rho_i = particlesa.rho[i];
			const double hsml = particlesa.hsml[i];
			const double p_i = particlesa.press[i];
			const double xi = particlesa.x[i];
			const double yi = particlesa.y[i];
			const double c0 = particlesa.c0[i];
			const double c = particlesa.c[i];
			const double massi = particlesa.mass[i];			
			//------kexi_force-------			
			double temp_sigemax = 0;//动量方程中的T 
			double temp_sigemay = 0;//Tj-Ti			
			//----------artificial viscosity---
			double avx = 0;
			double avy = 0;						
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:temp_sigemax,temp_sigemay,avx,avy)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				if (particlesa.bweight[i][j] < 0.000000001) continue;

				const double xj = particlesa.x[jj];
				const double yj = particlesa.y[jj];
				double dx = xj - xi;
				double dy = yj - yi;
				if (particlesa.iotype[i] == InoutType::Inlet && particlesa.iotype[jj] == InoutType::Outlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] - lengthofx;//xi(1)
				}
				if (particlesa.iotype[i] == InoutType::Outlet && particlesa.iotype[jj] == InoutType::Inlet)
				{
					dx = particlesa.x[jj] - particlesa.x[i] + lengthofx;//xi(1)
				}
				const double hsmlj = particlesa.hsml[jj];
				const double mhsml = (hsml + hsmlj) / 2;
				const double r2 = dx * dx + dy * dy;
				const double rho_j = particlesa.rho[jj];
				const double massj = particlesa.mass[jj];
				const double wMxijxV_iix = particlesa.wMxijx[i][j] * massj / rho_j;
				const double wMxijyV_iiy = particlesa.wMxijy[i][j] * massj / rho_j;
				//const double wMxijxV_jjx = (jj)->wMxijx[j] * massj / rho_j;//(jj)->wMxijx[j] 可能错了
				//const double wMxijyV_jjy = (jj)->wMxijy[j] * massj / rho_j;
				const double wMxijxV_jjx = (particlesa.m_11[jj] * dx + particlesa.m_12[jj] * dy) * particlesa.bweight[i][j] * massj / rho_j;
				const double wMxijyV_jjy = (particlesa.m_21[jj] * dx + particlesa.m_22[jj] * dy) * particlesa.bweight[i][j] * massj / rho_j;
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
				const double sigema_j11 = -particlesa.press[jj] + particlesa.tau11[jj]+particlesa.turb11[jj];//加上湍流应力
				const double sigema_j12 = particlesa.tau12[jj]+ particlesa.turb12[jj];//particlesa.tau12[jj]=particlesa.tau21[jj];particlesa.turb12[jj]=particlesa.turb21[jj]
				const double sigema_j21 = particlesa.tau21[jj]+ particlesa.turb21[jj];
				const double sigema_j22 = -particlesa.press[jj] + particlesa.tau22[jj]+ particlesa.turb22[jj];
				//i粒子
				/*const double sigema_i11 = -p_i + particlesa.tau11[i];
				const double sigema_i12 = particlesa.tau12[i];
				const double sigema_i21 = particlesa.tau21[i];
				const double sigema_i22 = -p_i + particlesa.tau22[i];*/
				const double sigema_i11 = -p_i + particlesa.tau11[i] + particlesa.turb11[i];
				const double sigema_i12 = particlesa.tau12[i] + particlesa.turb12[i];
				const double sigema_i21 = particlesa.tau21[i] + particlesa.turb21[i];
				const double sigema_i22 = -p_i + particlesa.tau22[i] + particlesa.turb22[i];
				temp_sigemax += (sigema_j11 * wMxijxV_jjx + sigema_j12 * wMxijyV_jjy) + (sigema_i11 * wMxijxV_iix + sigema_j12 * wMxijyV_iiy);
				temp_sigemay += (sigema_j21 * wMxijxV_jjx + sigema_j22 * wMxijyV_jjy) + (sigema_i21 * wMxijxV_iix + sigema_j22 * wMxijyV_iiy);
				//-----turbulence-------湍流力加到柯西应力中去了
				
				//--------artificial viscosity-------
				const double dvx = particlesa.vx[jj] - particlesa.vx[i];    //(Vj-Vi)
				const double dvy = particlesa.vy[jj] - particlesa.vy[i];
				double muv = 0;
				double piv = 0;
				const double cj = particlesa.c[jj];
				const double vr = dvx * dx + dvy * dy;     //(Vj-Vi)(Rj-Ri)
				const double mc = 0.5 * (cj + particlesa.c[i]);
				const double mrho = 0.5 * (rho_j + particlesa.rho[i]);				
				if (vr < 0) {
					muv = mhsml * vr / (r2 + mhsml * mhsml * 0.01);//FAI_ij
					//piv = (0.5 * muv - 0.5 * mc) * muv / mrho;//beta项-alpha项
					piv = (0.5 * muv - 1.0 * mc) * muv / mrho;
					//piv = (0.5 * muv) * muv / mrho;//只有beta项，加速度会一直很大，停不下来，穿透。
				}
				avx += -massj * piv * particlesa.wMxijx[i][j];
				avy += -massj * piv * particlesa.wMxijy[i][j];
			}
			
			particlesa.fintx[i] = temp_sigemax / rho_i;
			particlesa.finty[i] = temp_sigemay / rho_i;
			//(ii)->turbx = turbx;// / particlesa.rho[i];
			//(ii)->turby = turby;// / particlesa.rho[i];
			particlesa.avx[i] = avx;
			particlesa.avy[i] = avy;
			//(ii)->avy = 0;							
			if (particlesa.ftype[i] != sph::FixType::Fixed)//inlet粒子的加速度场为0，只有初始速度。outlet粒子呢？
			{				
				particlesa.ax[i] = particlesa.fintx[i] + avx;
				particlesa.ay[i] = particlesa.finty[i] + avy;
				//particlesa.ay[i] = 0;
				//particlesa.ax[i] = (ii)->fintx + avx + turbx + (ii)->replx;
				//particlesa.ay[i] = (ii)->finty + avy + turby + (ii)->reply + gravity;				
			}
		}
		const std::clock_t end = std::clock();
		if (time_measure) std::cout << "single step costs " << double(end - begin) / TICKS_PER_SEC << "s\n";
	}
	
	inline void domain::usingProcessbar(bool _b)
	{
		usingProgressbar = _b;
	}

	inline int domain::getConsoleWidth()
	{
#ifdef _WIN_
		CONSOLE_SCREEN_BUFFER_INFO csbi;

		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
		const int columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
		const int rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
		return columns;
#endif
#ifdef _LINUX_
		struct winsize w;
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
		return w.ws_col;
#endif
	}

	inline void domain::setInitVel(const double _v)
	{
		this->vel = _v;
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (auto i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			particlesa.vx[i] = vel * cos(angle);
			particlesa.vy[i] = vel * sin(angle);
		}
	}

	inline void domain::applyVelBc(const double _v)//inlen区的最左边3层，设为流入区域，并且将粒子设置为Inlet粒子
	{

		this->vel = _v;
		const double RR = 0.25*(this->indiameter + dp) * (this->indiameter + dp);//非完整抛物线
		//const double RR = 0.25 * (this->indiameter) * this->indiameter;//完整抛物线
#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (auto i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			if (particlesa.x[i] < 0) {
				//将速度施加为定速条件
				//particlesa.ftype[i] = sph::FixType::Moving;				
				//particlesa.vx[i] = vel * cos(angle)*(1- particlesa.y[i] * particlesa.y[i] /RR);//vel*(1-rr/RR)
				//particlesa.vy[i] = vel * sin(angle) * (1 - particlesa.y[i] * particlesa.y[i] / RR);
				particlesa.vx[i] = vel * cos(angle);
				particlesa.vy[i] = vel * sin(angle);
				particlesa.iotype[i] = sph::InoutType::Inlet;
			}
			else
			{
				//particlesa.ftype[i] = sph::FixType::Free;
				particlesa.iotype[i] = sph::InoutType::Fluid;
			}
		}
	}

	inline void domain::applyVelBc()
	{
		const double ctime = this->time;
		double vel = 0;
		if (ctime > std::prev(velprofile.end())->first)
			vel = 0;
		else {
			for (auto i = velprofile.begin(); i != velprofile.end(); i++)
			{
				const auto j = std::next(i, 1);
				if (j == velprofile.end())break;
				if (ctime >= i->first && ctime < j->first) {
					vel = i->second + (ctime - i->first) * (j->second - i->second) / (j->first - i->first);
					break;
				}
			}
		}
		this->vel = vel;

#ifdef OMP_USE
#pragma omp parallel for schedule (guided)
#endif
		for (auto i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (particlesa.btype[i] == sph::BoundaryType::Boundary) continue;
			if (particlesa.x[i] < -dp * (int(inlen / dp) - 4.0)) {
				particlesa.ftype[i] = sph::FixType::Moving;
				particlesa.vx[i] = vel * cos(angle);
				particlesa.vy[i] = vel * sin(angle);
				particlesa.iotype[i] = sph::InoutType::Inlet;
			}
			else
			{
				particlesa.ftype[i] = sph::FixType::Free;
				particlesa.iotype[i] = sph::InoutType::Fluid;
			}
		}
	}

	inline void domain::debugNeib(const int _i)
	{
		const int id = _i;
		std::string filename;
		filename = "neib" + std::to_string(id) + ".dat";
		std::ofstream ofile(filename);
		ofile << "Title: \"neighbor list for particle " << id << "\"\n";
		ofile << "Variables = \"X\", \"Y\", \"Id\"\n";
		ofile << "Zone I=" << particles[id - 1]->neiblist.size() + 1 << std::endl;
		ofile << particles[id - 1]->getX() << '\t' << particles[id - 1]->getY() << '\t' << id << std::endl;
		//for (auto i = particles[id - 1]->neiblist.begin(); i != particles[id - 1]->neiblist.end(); i++)
		for (int i = 0; i < particlesa.neibNum[id - 1]; i++)
		{
			ofile << particlesa.getX(particlesa.neiblist[id-1][i]) << '\t' << particlesa.getY(particlesa.neiblist[id - 1][i]) << '\t' << particlesa.getIdx(particlesa.neiblist[id - 1][i]) << std::endl;
		}
		ofile.close();
	}

	inline void domain::setParallel(bool _p)
	{
		parallel = _p;
	}

	inline void domain::setnbskin(double _k)
	{
		neibskin = _k;
	}

	inline void domain::adjustC0()
	{
		double c0 = vmax * 10.0 * 1.1;
		if (vmax == 0 || c0 < sph::Fluid::SoundSpeed(FluidType::Water) * 0.4) {
			c0 = sph::Fluid::SoundSpeed(FluidType::Water) * 0.4;
		}

		adjustC0_dev0(particlesa.c0, c0, particleNum());
	}

	inline bool domain::inlet()//不再需要新增新的粒子，只需要更新粒子的参数
	{			
		inlet_dev0(particleNum(), particlesa.x, particlesa.iotype, outletBcx);
		return true;
			//return false;
		
	}
	//当粒子过了outlet区域，粒子回到Inlet区域
	inline bool domain::outlet()//
	{
		bool _b = false;
		outlet_dev0(particleNum(), particlesa.x, particlesa.iotype, outletBcx, outletBcxEdge, lengthofx);
		return _b;
	}

	inline bool domain::buffoutlet(double dt)
	{
		this->updateGhostPos();
		this->updateGhostInfo();
		const bool _b = this->updateBufferPos(dt);

		return _b;
	}

	inline void domain::updateGhostPos()
	{
		for (int i = 0; i < buffer.size(); i++)
		{
			particle* ii = buffer[i];
			particle* jj = buffer2ghost[ii];
			const double px = particlesa.x[i];
			const double py = ii->y;
			const int udx = outletd == Direction::Left ? -1 : outletd == Direction::Right ? 1 : 0;
			const int udy = 0;
			const double gx = outletBcx - udx * (px - outletBcx);//对称分布
			//现将对称分布改为平移分布

			const double gy = py;
			jj->x = gx;
			jj->y = gy;
		}
	}

	inline void domain::updateGhostInfo()//ghost的信息,温度偏小
	{
		double pmin = DBL_MAX;
		for (int i = 0; i < ghosts.size(); i++)
		{
			particle* ii = ghosts[i];
			const double gamma = ii->gamma;
			const double c0 = particlesa.c0[i];
			const double rho0 = ii->rho0;
			const double b = c0 * c0 * rho0 / gamma;
			const double yi = ii->y;
			const double xi = particlesa.x[i];
			double p = 0, vcc = 0, vx = 0, vy = 0, ax=0;
			double vcc1 = 0, t = 0, t_x = 0, t_y = 0;
			int vcc2 = 0;
			double distmin = 100;
			if (ii->neiblist.size() == 0) {
				std::cerr << "\nError: neighbor list empty\n";
			}
#ifdef OMP_USE
#pragma omp parallel for schedule (guided) reduction(+:p,vx,vy,vcc)
#endif
			for (int j = 0; j < particlesa.neibNum[i]; j++)
			{
				const int jj = particlesa.neiblist[i][j];
				//if (particlesa.btype[i] == particlesa.btype[jj]) continue;
				//if (particlesa.btype[jj] == sph::BoundaryType::Boundary) continue;//排除边界粒子
				if (particlesa.iotype[jj] == sph::InoutType::Ghost) continue;//在建近邻时，ghost邻域中不会有ghost粒子
				const double mass_j = particlesa.mass[jj];
				const double rho_j = particlesa.rho[jj];
				const double p_k = particlesa.press[jj];
				p += p_k * particlesa.bweight[i][j] * mass_j / rho_j;
				vx += particlesa.vx[jj] * particlesa.bweight[i][j] * mass_j / rho_j;//同温度，不同的y，粒子的速度也不同。
				vy += particlesa.vy[jj] * particlesa.bweight[i][j] * mass_j / rho_j;
				vcc += particlesa.bweight[i][j] * mass_j / rho_j;
				//ax += (jj)->ax * particlesa.bweight[i][j] * mass_j / rho_j;
				//double detay = abs(jj->y - yi) / dp;//判断是否属于同一个y
				//if (detay < 0.3)
				//{
				//	vcc1 += particlesa.bweight[i][j] * mass_j / rho_j;
				//	vcc2++;
				//	//t += particlesa.temperature[jj] * particlesa.bweight[i][j] * mass_j / rho_j;
				//	t += particlesa.temperature[jj];
				//	t_x += particlesa.temperature_x[jj] * particlesa.bweight[i][j] * mass_j / rho_j;
				//	t_y += particlesa.temperature_y[jj] * particlesa.bweight[i][j] * mass_j / rho_j;
				//}
				//t += particlesa.temperature[jj] * particlesa.bweight[i][j] * mass_j / rho_j;
				//t_x += particlesa.temperature_x[jj] * particlesa.bweight[i][j] * mass_j / rho_j;
				//t_y += particlesa.temperature_y[jj] * particlesa.bweight[i][j] * mass_j / rho_j;//错的，靠近边界处的不同y粒子的温度梯度区别很大
				//double dist = (jj->y - yi)* (jj->y - yi)+ (jj->x - xi)* (jj->x - xi);
				////double distmin = 100;
				////distmin = distmin < dist ? distmin : dist;//考虑边界温度的时候加上
				//if (distmin > dist) {
				//	t = particlesa.temperature[jj];
				//	t_x = particlesa.temperature_x[jj];
				//	t_y = particlesa.temperature_y[jj];
				//	distmin = dist;
				//}
			}
			/*if(vcc< 0.00000001)
				std::cout << std::endl << "\nvcc of ghost < 0: " << vcc << std::endl;*/
			const double vcc0 = 0.000000001;
			p = vcc > vcc0 ? p / vcc : ii->back_p;//vcc=2.5*10-9 < 1*10-8
			vx = vcc > vcc0 ? vx / vcc : 0;
			vy = vcc > vcc0 ? vy / vcc : 0;
			//ax = vcc > vcc0 ? ax / vcc : 0;
			//t = vcc1 > vcc0 ? t / vcc1 : 0;//t=0				
			//t_x = vcc1 > vcc0 ? t_x / vcc1 : 0;
			//t_y = vcc1 > vcc0 ? t_y / vcc1 : 0;
			if (pmin > p) pmin = p;
			particlesa.vcc[i] = vcc;
			//压力的修改，如何修改使得buffer的压力是递减的
			particlesa.press[i] = p;
			//particlesa.press[i] = 0; 
			particlesa.ax[i] = 0;
			particlesa.ay[i] = 0;
			particlesa.vx[i] = vx;
			particlesa.vy[i] = vy;
			//particlesa.rho[i] = (p - (ii)->back_p) / c0 / c0 + rho0;	
			particlesa.rho[i] = pow(p / b + 1.0, 1 / 7.0) * rho0;//状态方程来的，错了吧，这是gamma=1计算得到的
			(ii)->c = sqrt(b * (p + gamma) / ii->rho);
			particlesa.temperature[i] = t;//如果不传温度，那么Buffer粒子的温度将一直不变，即为流体粒子变为Buffer粒子时的温度，感觉也可以。
			particlesa.temperature_x[i] = t_x;
			particlesa.temperature_y[i] = t_y;
//#ifdef OMP_USE
//#pragma omp parallel for schedule (guided) reduction(+:t,t_x,t_y,vcc1)
//#endif
//			for (int j = 0; j < particlesa.neibNum[i]; j++)
//			{
//				const int jj = particlesa.neiblist[i][j];
//				//if (particlesa.btype[i] == particlesa.btype[jj]) continue;
//				//if (particlesa.btype[jj] == sph::BoundaryType::Boundary) continue;//排除边界粒子
//				if (particlesa.iotype[jj] == sph::InoutType::Ghost) continue;//在建近邻时，ghost邻域中不会有ghost粒子
//				double detay = abs(jj->y - yi)/dp;
//				if (detay>0.3) continue;
//				const double mass_j = particlesa.mass[jj];
//				const double rho_j = particlesa.rho[jj];				
//				vcc1 += particlesa.bweight[i][j] * mass_j / rho_j;				
//				t += particlesa.temperature[jj] * particlesa.bweight[i][j] * mass_j / rho_j;
//				t_x += particlesa.temperature_x[jj] * particlesa.bweight[i][j] * mass_j / rho_j;
//				t_y += particlesa.temperature_y[jj] * particlesa.bweight[i][j] * mass_j / rho_j;//错的，靠近边界处的不同y粒子的温度梯度区别很大
//			}						
//			t = t > 0.00000001 ? t / vcc1 : 0;
//			t_x = vcc1 > 0.00000001 ? t_x / vcc1 : 0;
//			t_y = vcc1 > 0.00000001 ? t_y / vcc1 : 0;
//			particlesa.temperature[i] = t;//如果不传温度，那么Buffer粒子的温度将一直不变，即为流体粒子变为Buffer粒子时的温度，感觉也可以。
//			particlesa.temperature_x[i] = t_x;
//			particlesa.temperature_y[i] = t_y;			
		}
		//for (int i = 0; i < ghosts.size(); i++)//流出边界设为最小压力
		//{
		//	particle* ii = ghosts[i];
		//	particlesa.press[i] = pmin;
		//}
	}

	inline bool domain::updateBufferPos(double dt)
	{
		bool _b = false;
		const double dt2 = dt * 0.5;
		for (int i = 0; i < buffer.size(); i++)
		{
			particle* ii = buffer[i];
			particlesa.half_x[i] = particlesa.x[i];
			particlesa.half_vx[i] = particlesa.vx[i];
			particlesa.half_temperature[i] = particlesa.temperature[i];
			particlesa.vx[i] = particlesa.half_vx[i] + particlesa.ax[i] * dt2;
			particlesa.x[i] = particlesa.half_x[i] + particlesa.vx[i] * dt2;
			//温度也更新下
			particlesa.temperature[i] = particlesa.half_temperature[i] + particlesa.temperature_t[i] * dt2;
			//particlesa.x[i] += ii->vx * dt;
			ii->y += ii->vy * dt;
			if (outletd == Direction::Left) {
				if (particlesa.x[i] < outletBcxEdge) {//discard
					_b = true;
					particle* jj = buffer2ghost[ii];
					ghosts.erase(std::remove(ghosts.begin(), ghosts.end(), jj), ghosts.end());
					ghost2buffer.erase(jj);
					free(jj);
					particles.erase(std::remove(particles.begin(), particles.end(), ii), particles.end());
					buffer2ghost.erase(ii);
					free(ii);
					buffer[i] = NULL;
				}
				else if (particlesa.x[i] > outletBcx) {//convert to fluid
					particle* jj = buffer2ghost[ii];
					ghosts.erase(std::remove(ghosts.begin(), ghosts.end(), jj), ghosts.end());
					ghost2buffer.erase(jj);
					buffer2ghost.erase(ii);
					free(jj);
					particles.push_back(ii);
					ii->setBtype(BoundaryType::Bulk);
					ii->setFtype(FixType::Free);
					ii->setIotype(InoutType::Fluid);
					ii->setIdx(idp++);
					const double ux = outletd == Direction::Left ? -1 : 1;
					const double x = particlesa.x[i] + ux * outletlayer;
					const double y = ii->y;
					const double gux = abs(x - outletBcx);
					const double gx = outletBcx - ux * gux;
					const double gy = y;
					const double temperature1 = ii->temperature;
					const double temperature2 = jj->temperature;
					buffer[i] = NULL;
					// add new buffer and ghost pair
					ii = new particle(x, y, 0, 0, 0, p_back, Fluid::Gamma(FluidType::Water), Fluid::FluidDensity(FluidType::Water),
						dp * dp, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), dp * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature1);
					ii->setBtype(BoundaryType::Boundary);
					ii->setFtype(FixType::Free);
					ii->setFltype(FluidType::Water);
					ii->setIotype(InoutType::Buffer);
					ii->setIdx(static_cast<unsigned int>(buffer.size()) + 1);
					ii->setDensityMin();
					buffer.push_back(ii);
					particles.push_back(ii);
					//--
					jj = new particle(x, y, 0, 0, 0, p_back, Fluid::Gamma(FluidType::Water), Fluid::FluidDensity(FluidType::Water),
						dp * dp, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), dp * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature2);
					jj->setBtype(BoundaryType::Boundary);
					jj->setFtype(FixType::Fixed);
					jj->setFltype(FluidType::Water);
					jj->setIotype(InoutType::Ghost);
					jj->setIdx(static_cast<unsigned int>(buffer.size()) + 1);
					jj->setDensityMin();
					ghosts.push_back(jj);
					ghost2buffer.insert(std::pair<particle*, particle*>(jj, ii));
					buffer2ghost.insert(std::pair<particle*, particle*>(ii, jj));
				}
			}
			else if (outletd == Direction::Right) {
				if (particlesa.x[i] > outletBcxEdge) {//discard超出右边界，删除
					_b = true;
					particle* jj = buffer2ghost[ii];
					ghosts.erase(std::remove(ghosts.begin(), ghosts.end(), jj), ghosts.end());
					ghost2buffer.erase(jj);
					free(jj);
					buffer2ghost.erase(ii);
					particles.erase(std::remove(particles.begin(), particles.end(), ii), particles.end());
					free(ii);
					buffer[i] = NULL;
				}
				else if (particlesa.x[i] < outletBcx) {//convert to fluid
					particle* jj = buffer2ghost[ii];
					ghosts.erase(std::remove(ghosts.begin(), ghosts.end(), jj), ghosts.end());
					ghost2buffer.erase(jj);
					buffer2ghost.erase(ii);
					free(jj);
					particles.push_back(ii);
					ii->setBtype(BoundaryType::Bulk);
					ii->setFtype(FixType::Free);
					ii->setIotype(InoutType::Fluid);
					//ii->setIdx(idp++);
					const double ux = outletd == Direction::Left ? -1 : 1;
					const double x = particlesa.x[i] + ux * outletlayer;
					const double y = ii->y;
					const double gux = abs(x - outletBcx);
					const double gx = outletBcx - ux * gux;
					const double gy = y;
					const double temperature1 = ii->temperature;
					const double temperature2 = jj->temperature;
					buffer[i] = NULL;
					// add new buffer and ghost pair
					ii = new particle(x, y, 0, 0, 0, p_back, Fluid::Gamma(FluidType::Water), Fluid::FluidDensity(FluidType::Water),
						dp * dp, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), dp * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature1);
					ii->setBtype(BoundaryType::Boundary);
					ii->setFtype(FixType::Free);
					ii->setFltype(FluidType::Water);
					ii->setIotype(InoutType::Buffer);
					ii->setIdx(static_cast<unsigned int>(buffer.size()) + 1);
					ii->setDensityMin();
					buffer.push_back(ii);
					particles.push_back(ii);
					//--
					jj = new particle(x, y, 0, 0, 0, p_back, Fluid::Gamma(FluidType::Water), Fluid::FluidDensity(FluidType::Water),
						dp * dp, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), dp * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature2);
					jj->setBtype(BoundaryType::Boundary);
					jj->setFtype(FixType::Fixed);
					jj->setFltype(FluidType::Water);
					jj->setIotype(InoutType::Ghost);
					jj->setIdx(static_cast<unsigned int>(buffer.size()) + 1);
					jj->setDensityMin();
					ghosts.push_back(jj);
					ghost2buffer.insert(std::pair<particle*, particle*>(jj, ii));
					buffer2ghost.insert(std::pair<particle*, particle*>(ii, jj));
				}
			}
		}

		if (_b) {
			buffer.erase(std::remove(buffer.begin(), buffer.end(), (particle*)NULL), buffer.end());
		}
		return _b;
	}

	inline bool domain::checkFluid2Buffer()
	{
		bool _b = false;
		for (int i = 0; i < particles.size(); i++)
		{
			particle* ii = particles[i];
			if (ii->btype == sph::BoundaryType::Boundary) continue;
			if (particlesa.iotype[i] == sph::InoutType::Buffer) continue;
			const double x = particlesa.x[i];
			const double y = ii->y;
			const double vx = ii->vx;
			const double vy = ii->vy;

			const double temperature = ii->temperature;
			if (outletd == Direction::Left) {
				if (x < outletBcx) {
					_b = true;
					// fluid 2 buffer
					ii->setBtype(BoundaryType::Boundary);
					ii->setFtype(FixType::Free);
					ii->setIotype(InoutType::Buffer);
					buffer.push_back(ii);
					const double gx = outletBcx + abs(outletBcx - x);//将对称分布改为平移分布
					const double gy = y;
					particle* gp = new particle(gx, gy, vx, vy, 0, p_back, Fluid::Gamma(FluidType::Water),
						Fluid::FluidDensity(FluidType::Water), dp * dp, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), dp * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature);
					gp->setBtype(BoundaryType::Boundary);
					gp->setFtype(FixType::Fixed);
					gp->setFltype(FluidType::Water);
					gp->setIotype(InoutType::Ghost);
					gp->setIdx(static_cast<unsigned int>(ghosts.size()) + 1);
					gp->setDensityMin();
					ghost2buffer.insert(std::pair<particle*, particle*>(gp, ii));
					buffer2ghost.insert(std::pair<particle*, particle*>(ii, gp));
					ghosts.push_back(gp);
				}
			}
			else if (outletd == Direction::Right) {
				if (x > outletBcx) {
					_b = true;
					// fluid 2 buffer
					ii->setBtype(BoundaryType::Boundary);//现将Buffer粒子改为流体。
					ii->setFtype(FixType::Free);
					ii->setIotype(InoutType::Buffer);
					buffer.push_back(ii);
					const double gx = outletBcx - abs(x - outletBcx);
					const double gy = y;
					particle* gp = new particle(gx, gy, vx, vy, 0, p_back, Fluid::Gamma(FluidType::Water),
						Fluid::FluidDensity(FluidType::Water), dp * dp, Fluid::SoundSpeed(FluidType::Water), Fluid::Viscosity(FluidType::Water), dp * 1.01
						, Fluid::specific_heat(FluidType::Water), Fluid::coefficient_heat(FluidType::Water), temperature);
					gp->setBtype(BoundaryType::Boundary);
					gp->setFtype(FixType::Fixed);
					gp->setFltype(FluidType::Water);
					gp->setIotype(InoutType::Ghost);
					gp->setIdx(static_cast<unsigned int>(ghosts.size()) + 1);
					gp->setDensityMin();
					ghost2buffer.insert(std::pair<particle*, particle*>(gp, ii));
					buffer2ghost.insert(std::pair<particle*, particle*>(ii, gp));
					ghosts.push_back(gp);
				}
			}
		}
		return _b;
	}

	inline void domain::Ghost2bufferInfo()
	{
		for (int i = 0; i < buffer.size(); i++)
		{
			particle* ii = buffer[i];
			const particle* jj = buffer2ghost[ii];
			if (jj == NULL) {
				std::cerr << "\nError: no ghost attached to buffer\n";
			}
			ii->vx = jj->vx;
			ii->vy = jj->vy;
			//ii->vy = 0;
			ii->ax = jj->ax;
			//ii->ax = 0;//不传速度，而是传加速度，会怎么样
			ii->ay = 0;
			ii->c = jj->c;
			ii->press = (jj->press);//传负的press会崩溃
			ii->rho = jj->rho;
			//温度不传、温度梯度传
			//if (jj->temperature != 0)
			//ii->temperature = jj->temperature;
			////ii->temperature = 10;
			////ii->temperature_x = 0;
			//if (jj->temperature_x != 0)
			//ii->temperature_x = jj->temperature_x;//Ghost插值过来的温度梯度是错的！
			//if (jj->temperature_y != 0)
			//ii->temperature_y = jj->temperature_y;//Ghost插值过来的温度梯度是错的！
		}
	}

	inline void domain::readinVel(std::string _f)
	{
		std::ifstream ifile(_f);
		if (ifile.fail())
		{
			std::cerr << "\nvelocity file not exists...";
			exit(1);
		}
		if (ifile.is_open())
		{
			std::string line;
			while (std::getline(ifile, line))
			{
				if (line.rfind('x', 0) == 0) continue;
				std::istringstream iss(line);
				double time, vel;
				char c;
				iss >> time >> c >> vel;
				velprofile.insert(std::pair<double, double>(time, vel));
			}
		}
		ifile.close();
		std::cout << "read in velocity profile\n";
	}

	inline void domain::setShifting(ShiftingType _s)
	{
		stype = _s;
	}

	inline void domain::setShiftingCoe(double _c)
	{
		shiftingCoe = _c;
	}

	domain::~domain()
	{
		for (std::vector<class particle*>::iterator i = particles.begin(); i < particles.end(); i++)
		{
			free(*i);
		}
		particles.clear();
	}

}

