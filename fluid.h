#pragma once
#include "classes_type.h"
namespace sph {


	class Fluid
	{
	public:
		static const double FluidDensity(FluidType _f) {
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
		static const double SoundSpeed(FluidType _f) {
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
		static const double Gamma(FluidType _f) {
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
		static const double Viscosity(FluidType _f) {
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
		static const double specific_heat(FluidType _f) {        //比热容ci
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
		static const double coefficient_heat(FluidType _f) {     //传热系数ki
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


	};

}
