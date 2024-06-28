#pragma once
#include <vector>
#include "vector.h"
#include "matrix.h"
#include "wfunc.h"
#include "fluid.h"
#include "classes_type.h"



namespace sph {
	const double C_s = 0.15;
	const double C_v = 0.08;
	const double C_e = 1.0;

	const double karman = 0.4;


	class particle
	{
		friend class domain;
		friend class particleSOA; 
	public:
		particle(double, double);  //构造函数
		particle(double, double, double, double, double, double, double, double, double, double, double, double, double, double, double);
		void setvolume(double);   //赋值函数（体积）
		void setdensity(double);
		void setInitPressure(double);
		void setInitSoundSpd(double);
		void setIdx(unsigned int _id) { idx = _id; };
		void setVisco(double);    //粘度
		void sethsml(double);
		void setBtype(BoundaryType);
		void setFtype(FixType);
		void setFltype(FluidType);
		void setIotype(InoutType);
		void setP_back(double _p) { back_p = _p; };
		void setDensityMin();
		const BoundaryType getBtype() { return btype; };
		const FixType getFtype() { return ftype; };
		const FluidType getFltype() { return fltype; };
		const double getX() { return x; };
		const double getY() { return y; };
		const double getVx() { return vx; };
		const double getVy() { return vy; };
		const double getFx() { return fintx; };
		const double getFy() { return finty; };
		const double getAx() { return ax; };
		const double getAy() { return ay; };
		const double getAvx() { return avx; };
		const double getAvy() { return avy; };
		const double getTx() { return turbx; };
		const double getTy() { return turby; };
		const double getPress() { return press; };
		const double getP_back() { return back_p; };
		const double getInitPressure() { return press0; };
		const double getInitSoundSpd() { return c0; };
		const double getSoundSpd() { return c; };
		const double getInitDensity() { return rho0; };
		const double getMass() { return mass; };
		const double getDensity() { return rho; };
		const double gethsml() { return hsml; };
		const double getShiftc() { return shift_c; };
		const double getDisp() { return sqrt(ux * ux + uy * uy); };
		const double getdt();
		const double getvort() { return vort; };
		const double gettemperature() { return temperature; };
		const double gettempx() { return temperature_x; };
		const double gettempy() { return temperature_y; };
		const double gettempt() { return temperature_t; };
		const unsigned int getIdx() { return idx; };
		const math::matrix getTurbM() const { return turbmat; };
		void updateVolume() { this->vol = this->mass / this->rho; };
		void storeHalf();   //
		void density_filter();
		void densityRegulation();
		void updatePressure();
		void updateDDensity();
		void updateRepulsion();
		void updateFint();
		void getTurbulence();
		void getTurbForce();
		void getArticial();
		void updateAcc();
		void integration1sthalf(const double);
		void integrationfull(const double);
		void shifting_c();
		void shifting(const double);
		void clearNeiblist() { neiblist.clear(); };
		void add2Neiblist(particle* _p) { neiblist.push_back(_p); neibNum = neiblist.size(); };
		void setZeroDisp() { ux = uy = 0; };
		~particle();

	protected:
		unsigned int idx;
		double x;
		double y;
		double ux;
		double uy;
		double vx;
		double vy;
		double ax;
		double ay;
		double drho;
		double replx;//repulsive x
		double reply;//repulsive y
		double fintx;
		double finty;
		double turbx;
		double turby;
		math::matrix turbmat;
		double turb11;
		double turb12;
		double turb21;
		double turb22;
		double avx;
		double avy;
		double press0;//initial pressure
		double press;
		double back_p;//back pressure
		double rho;//density
		double rho0;//initial density
		double rho_min;//minimal density
		double vol;//volume
		double c;//sound speed
		double c0;//initial sound speed
		double visco;//viscosity
		double mass;
		double hsml;//smooth length
		double gamma;
		double specific_heat;//比热容
		double coefficient_heat;//传热系数
		double temperature;
		double temperature_t;   //温度对时间的导数
		double temperature_x;
		double temperature_y;
		double vcc;
		double shift_c;
		double shift_x;
		double shift_y;
		unsigned int neibNum;
		std::vector<class particle*> neiblist;
		double* bweight;
		double* dbweightx;
		double* dbweighty;
		double* wMxijx;
		double* wMxijy;
		double m_11;//M的一阶逆矩阵的元素
		double m_12;
		double m_21;
		double m_22;
		double M_11;//M的二阶逆矩阵的部分元素
		double M_12;
		double M_13;
		double M_14;
		double M_15;
		double M_21;
		double M_22;
		double M_23;
		double M_24;
		double M_25;
		double M_31 ;
		double M_32 ;
		double M_33 ;
		double M_34 ;
		double M_35 ;
		double M_51 ;
		double M_52 ;
		double M_53 ;
		double M_54 ;
		double M_55 ;
		double tau11;
		double tau12;
		double tau21;
		double tau22;
		double vort;//vorticity涡量
		double divvel;//速度散度
		//std::vector<double> bweight;
		//std::vector<double> dbweightx;
		//std::vector<double> dbweighty;
		BoundaryType btype;
		BoundaryConditionType bctype;
		FixType ftype;
		FluidType fltype;
		InoutType iotype;
		// predictor - corrector
		double half_rho;
		double half_vx;
		double half_vy;
		double half_x;
		double half_y;
		double half_temperature;
	};

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
		M_31=M_32=M_33=M_34=M_35=M_51=M_52=M_53=M_54=M_55=0;		
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

	inline void particle::setvolume(double _v)
	{
		vol = _v;
	}

	inline void particle::setdensity(double _d)
	{
		rho = _d;
	}

	inline void particle::setInitPressure(double _p)
	{
		press0 = _p;
	}

	inline void particle::setInitSoundSpd(double _c0)
	{
		c0 = _c0;
	}

	inline void particle::setVisco(double _v)
	{
		visco = _v;
	}

	inline void particle::sethsml(double _hsml)
	{
		hsml = _hsml;
	}

	inline void particle::setBtype(BoundaryType _b)
	{
		btype = _b;
	}

	inline void particle::setFtype(FixType _f)
	{
		ftype = _f;
	}

	inline void particle::setFltype(FluidType _f)
	{
		fltype = _f;
	}

	inline void particle::setIotype(InoutType _i)
	{
		iotype = _i;
	}

	inline void particle::setDensityMin()
	{
		rho_min = rho0 - back_p / c0 / c0;
	}

	inline const double particle::getdt()
	{
		const double alpha_pi = 1.0;
		const double hsml = this->gethsml();
		double divv = 0;

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			const math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double r = xi.length();
			const double q = r / hsml;

			math::vector dv(this->getVx() - (*i)->getVx(), this->getVy() - (*i)->getVy());
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			divv += dv * dbweight * (*i)->getMass() / (*i)->getDensity();
		}
		const double beta_pi = 10.0 * hsml;
		return 0.3 * hsml / (hsml * divv + c + 1.2 * (alpha_pi * c + beta_pi * abs(divv)));
	}

	inline void particle::storeHalf()
	{
		half_x = x;
		half_y = y;
		half_vx = vx;
		half_vy = vy;
		half_rho = rho;
		half_temperature = temperature;
	}

	inline void particle::density_filter()
	{
		double beta0_mls = 0;
		double rhop_sum_mls = 0;
		const double hsml = this->gethsml();
		for (std::vector<class particle*>::iterator i = neiblist.begin(); i != neiblist.end(); i++)
		{
			const math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double r = xi.length();
			const double hsml = (*i)->gethsml();
			const double q = r / hsml;
			const double rho_b = ((*i)->getPress() - this->getP_back()) / this->getInitSoundSpd() / this->getInitSoundSpd() + this->getInitDensity();
			const double v_j = (*i)->getMass() / (*i)->getDensity();
			const double mass_b = rho_b * v_j;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);

			if (q >= 0 && q <= 3.0)
			{
				beta0_mls += bweight * v_j;
				rhop_sum_mls += bweight * mass_b;
			}
		}

		beta0_mls += sph::wfunc::factor(hsml, 2) * this->getMass() / this->getDensity();
		rhop_sum_mls += sph::wfunc::factor(hsml, 2) * this->getMass();
		this->setdensity(rhop_sum_mls / beta0_mls);
	}

	inline void particle::densityRegulation()
	{
		if (rho < rho_min)
		{
			rho = rho_min;
		}
	}

	inline void particle::updatePressure()
	{
		if (this->getBtype() == BoundaryType::Boundary)
		{
			press = 0;
			double vcc = 0;
			for (auto i = neiblist.begin(); i != neiblist.end(); i++)
			{
				const math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
				const double r = xi.length();
				const double hsml = (*i)->gethsml();
				const double q = r / hsml;
				const double rho_b = ((*i)->getPress() - this->getP_back()) / this->getInitSoundSpd() / this->getInitSoundSpd() + this->getInitDensity();
				const double v_j = (*i)->getMass() / (*i)->getDensity();
				const double mass_b = rho_b * v_j;
				const double bweight = sph::wfunc::bweight(q, hsml, 2);

				press += (*i)->getPress() * bweight * (*i)->getMass() / (*i)->getDensity();
				vcc += bweight * (*i)->getMass() / (*i)->getDensity();
			}
			press = vcc > 0.00000001 ? press / vcc : 0;
			press = press > 0.00000001 ? press : 0;
			rho = (press - back_p) / c0 / c0 + rho0;
		}
		else {
			const double p = c0 * c0 * (rho - rho0) + back_p;
			const double b = c0 * c0 * rho0 / gamma;
			press = b * (pow((rho / rho0), gamma) - 1.0) + back_p;
			c = sqrt(b * gamma / rho0);
			press = press < back_p ? back_p : press;
		}
	}

	inline void particle::updateDDensity()
	{
		// diffusion term
		const double chi = 0.2;

		double drhodt = 0;
		double drhodiff = 0;
		const double rho_i = this->getDensity();
		const double hsml = this->gethsml();

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double r = xi.length();
			if (r == 0) {
				std::cerr << "distance between particle " << this->getIdx() << " and " << (*i)->getIdx() << " is zero\n";
				std::cerr << "this particle " << this->getIdx() << " : " << this->getX() << '\t' << this->getY() << std::endl;
				std::cerr << "that particle " << (*i)->getIdx() << " : " << (*i)->getX() << '\t' << (*i)->getY() << std::endl;
			}
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;
			const double rho_j = (*i)->getDensity();

			const math::vector diff = (rho_j - rho_i) * xi / r2;
			const math::vector diff2((rho_j - rho_i) * xi / r2);
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			math::vector dv(this->getVx() - (*i)->getVx(), this->getVy() - (*i)->getVy());
			const double vcc_rho = dv * dbweight;
			const double vcc_diff = diff * dbweight;
			drhodiff += chi * this->getInitSoundSpd() * this->gethsml() * vcc_diff * (*i)->getMass() / (*i)->getDensity();
			if (drhodiff != drhodiff) {
				std::cerr << "nan\n";
			}
			drhodt -= this->getDensity() * vcc_rho * (*i)->getMass() / (*i)->getDensity();
		}
		drho = drhodt + drhodiff;
	}

	inline void particle::updateRepulsion()
	{
		// diffusion term
		const double chi = 0.2;

		replx = reply = 0;
		if (this->getBtype() == BoundaryType::Boundary) return;

		const double hsml = this->gethsml();

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			if (this->getFltype() == (*i)->getFltype()) continue;

			const math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
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
			replx += 0.01 * c0 * c0 * chi * f_eta * xi.getX() / (r * r) * this->getMass();
			reply += 0.01 * c0 * c0 * chi * f_eta * xi.getY() / (r * r) * this->getMass();
		}
	}

	inline void particle::updateFint()
	{
		fintx = finty = 0;
		const double hsml = this->gethsml();

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			const double p_j = (*i)->getBtype() == BoundaryType::Boundary ? (*i)->getP_back() : (*i)->getPress();

			fintx -= (p_j + this->getPress()) / this->getDensity() * (*i)->getMass() / (*i)->getDensity() * dbweight.getX();//这个地方也要改,xi.getX() = dx;
			//fintx += (ii)->bweight[j] * (temp_ik11 * dx + temp_ik12 * dy) * particlesa.mass[jj] / rho_j / rho_i;//是不是要在上面加上四个temp_ik的函数呢
			finty -= (p_j + this->getPress()) / this->getDensity() * (*i)->getMass() / (*i)->getDensity() * dbweight.getY();
		}
	}

	inline void particle::getTurbulence()
	{
		math::matrix dudx(2);
		math::matrix unit(2);
		unit(1, 1) = unit(2, 2) = 1.0;
		const double hsml = this->gethsml();

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;
			math::vector dv(this->getVx() - (*i)->getVx(), this->getVy() - (*i)->getVy());
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			dudx += dyadic(dv, dbweight) * (*i)->getMass() / (*i)->getDensity();
		}

		double dist;
		if (this->getX() >= 0.0075 && this->getX() <= 0.19) {
			math::vector xi(0.0, abs(this->getY()) - 0.0125);
			dist = xi.length();
		}
		else if (this->getX() < 0.0075) {
			math::vector xi(0.0075 - this->getX(), abs(this->getY()) - 0.0125);
			dist = xi.length();
		}
		else if (this->getX() > 0.19) {
			math::vector xi(0.19 - this->getX(), abs(this->getY()) - 0.0125);
			dist = xi.length();
		}

		const math::matrix ell = (dudx + transpose(dudx)) * 0.5;
		const double sij = sqrt(2.0) * ell.norm();
		const double xjian = this->gethsml() / 1.01;
		const double mut = std::min(dist * dist * karman * karman, xjian * xjian * C_s * C_s) * sij;
		const double kenergy = C_v / C_e * xjian * xjian * sij * sij;

		const math::matrix kmatrix = -2.0 / 3.0 * kenergy * unit;
		this->turbmat = 2.0 * mut * ell + kmatrix * this->getDensity();
	}

	inline void particle::getTurbForce()
	{
		turbx = turby = 0;

		const double rho_i = this->getDensity();
		const double hsml = this->gethsml();

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;

			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			const double rho_j = (*i)->getDensity();
			const math::matrix turb_i = this->getTurbM() / rho_i / rho_i;
			const math::matrix turb_j = (*i)->getTurbM() / rho_j / rho_j;
			const math::matrix turb_ij = turb_i + turb_j;

			const math::vector t_ij = turb_ij * dbweight;

			turbx += t_ij(1) * (*i)->getMass();
			turby += t_ij(2) * (*i)->getMass();
		}
	}

	inline void particle::getArticial()
	{
		avx = avy = 0;

		if (this->getFltype() == FluidType::Moisture) return;

		const double rho_i = this->getDensity();
		const double hsml = this->gethsml();
		const double zeta = 2.0 * sph::Fluid::Viscosity(this->getFltype()) * (3.0 + 2.0) / hsml / c0;

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double mhsml = (hsml + (*i)->gethsml()) * 0.5;
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;

			const double rho_j = (*i)->getDensity();
			math::vector dv(this->getVx() - (*i)->getVx(), this->getVy() - (*i)->getVy());
			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);

			const double vr = dv * xi;
			const double xjian = this->gethsml() / 1.01;

			avx += zeta * hsml * (*i)->getMass() * (this->getSoundSpd() + (*i)->getSoundSpd()) * vr / ((rho_i + rho_j) * (r2 + 0.01 * xjian * xjian)) * dbweight(1);
			avy += zeta * hsml * (*i)->getMass() * (this->getSoundSpd() + (*i)->getSoundSpd()) * vr / ((rho_i + rho_j) * (r2 + 0.01 * xjian * xjian)) * dbweight(2);
		}
	}

	inline void particle::updateAcc()
	{
		ax = fintx + avx + turbx + replx;
		ay = finty + avy + turby + reply;
	}

	inline void particle::integration1sthalf(const double _dt2)
	{
		if (this->getBtype() == BoundaryType::Boundary) return;

		const double dt2 = _dt2;
		if (this->getFtype() != FixType::Fixed) {
			rho = half_rho + drho * dt2;
			vx = half_vx + ax * dt2;
			vy = half_vy + ay * dt2;
		}
		x = half_x + vx * dt2;
		y = half_y + vy * dt2;

		this->updateVolume();
	}

	inline void particle::integrationfull(const double _dt)
	{
		if (this->getBtype() == BoundaryType::Boundary) return;

		const double dt = _dt;
		if (this->getFtype() == FixType::Free)
		{
			rho = half_rho + drho * dt;
			vx = half_vx + ax * dt;
			vy = half_vy + ay * dt;
			ux += vx * dt;
			uy += vy * dt;
			this->updateVolume();
		}
	}

	inline void particle::shifting_c()
	{
		shift_c = 0;
		if (this->getBtype() == BoundaryType::Boundary) return;

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double mhsml = (hsml + (*i)->gethsml()) * 0.5;
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;

			shift_c += bweight * (*i)->getMass() / (*i)->getDensity();
		}
	}

	inline void particle::shifting(const double dt)
	{
		shift_x = shift_y = 0;

		if (this->getBtype() == BoundaryType::Boundary) return;
		if (this->getFtype() != FixType::Free) return;

		math::vector dr(0.0, 0.0);

		for (auto i = neiblist.begin(); i != neiblist.end(); i++)
		{
			math::vector xi(this->getX() - (*i)->getX(), this->getY() - (*i)->getY());
			const double mhsml = (hsml + (*i)->gethsml()) * 0.5;
			const double r = xi.length();
			const double r2 = r * r;
			const double q = r / hsml;
			const double bweight = sph::wfunc::bweight(q, hsml, 2);
			if (bweight < 0.0000000001) continue;

			const math::vector dbweight = sph::wfunc::dbweight(q, hsml, 2, xi);
			const double conc_ij = (*i)->getShiftc() - this->getShiftc();

			dr += conc_ij * (*i)->getMass() / (*i)->getDensity() * dbweight;
		}

		const double xjian = this->gethsml() / 1.01;
		const double vel = sqrt(vx * vx + vy * vy);
		dr = dr * 2.0 * xjian * vel * dt;

		shift_x = -dr(1);
		shift_y = -dr(2);

		x += shift_x;
		y += shift_y;
		ux += shift_x;
		uy += shift_y;
	}

	particle::~particle()
	{
		if (bweight) free(bweight);
		if (dbweightx) free(dbweightx);
		if (dbweighty) free(dbweighty);
		neiblist.clear();
		//bweight.clear();
	}

}
