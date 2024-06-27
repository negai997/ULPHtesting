#pragma once

#include "vector.h"

namespace sph {

	enum class WeightFuncType
	{
		Gaussian = 0,
		Cubic = 1
	};

	class wfunc
	{
	public:                                    //factor(ii->hsml,2),改为mhsml?
		static inline const double factor(double _hsml, int _dim) {
			return 1.0 / (math::PI * pow(_hsml, _dim)) * (1.0 - exp(-9.0)) / (1.0 - 10.0 * exp(-9.0));
		}
		//bweight(q,hsml,2)
		static inline const double bweight(double q, double _hsml, int _dim) {
			return q <= 3.0 ? factor(_hsml, _dim) * (exp(-q * q) - exp(-9.0)) : 0;
		}

		static inline const math::vector dbweight(double q, double _hsml, int _dim, math::vector xi) {
			const double fac = q <= 3.0 ? factor(_hsml, _dim) * exp(-q * q) * (-2.0 / _hsml / _hsml) : 0;
			return math::vector(fac * xi.getX(), fac * xi.getY());
		}

	};

}
