#pragma once
#include <cmath>
#define OMP_USE

namespace math {

	const double PI = 3.141592653589793238463;

	class vector
	{
	public:
		vector(double, double);
		vector();
		vector(const math::vector& _v);
		void setX(double);
		void setY(double);
		const double getX() const { return x; };
		const double getY() const { return y; };
		const double length() const;
		double& operator()(const unsigned int idx);
		const double operator()(const unsigned int idx) const;
		const vector operator*(const double);
		const vector operator/(const double);
		const vector operator+(const vector&);
		const vector operator-(const vector&);
		const vector operator+=(const vector&);
		const vector operator-=(const vector&);
		~vector();

	private:
		double x;
		double y;
	};

	vector::vector(double _x, double _y) :x{ _x }, y{ _y }
	{
	}

	inline vector::vector() : x{ 0 }, y{ 0 }
	{
	}

	inline vector::vector(const math::vector& _v)
	{
		x = _v.x;
		y = _v.y;
	}

	inline void vector::setX(double _x)
	{
		x = _x;
	}

	inline void vector::setY(double _y)
	{
		y = _y;
	}

	inline const double vector::length() const
	{
		return sqrt(x * x + y * y);
	}

	inline double& vector::operator()(const unsigned int idx)
	{
		if (idx == 1) return x;
		if (idx == 2) return y;
	}

	inline const double vector::operator()(const unsigned int idx) const
	{
		if (idx == 1) return x;
		if (idx == 2) return y;
	}

	inline const vector vector::operator*(const double _x)
	{
		return vector(x * _x, y * _x);
	}

	inline const vector vector::operator/(const double _x)
	{
		return vector(x / _x, y / _x);
	}

	inline const vector vector::operator+(const vector& _v)
	{
		return vector(x + _v.getX(), y + _v.getY());
	}

	inline const vector vector::operator-(const vector& _v)
	{
		return vector(x - _v.getX(), y - _v.getY());
	}

	inline const vector vector::operator+=(const vector& _v)
	{
		x = x + _v.getX();
		y = y + _v.getY();
		return vector(*this);
	}

	inline const vector vector::operator-=(const vector& _v)
	{
		x = x - _v.getX();
		y = y - _v.getY();
		return vector(*this);
	}

	vector::~vector()
	{
	}

}

const double operator*(const math::vector& v1, const math::vector& v2) {
	return v1.getX() * v2.getX() + v1.getY() * v2.getY();
}

const math::vector operator*(const double _x, const math::vector& v) {
	return math::vector(_x * v.getX(), _x * v.getY());
}

const math::vector operator*(const math::vector& v, const double _x) {
	return math::vector(_x * v.getX(), _x * v.getY());
}

const math::vector operator/(const math::vector& v, const double _x) {
	return math::vector(v.getX() / _x, v.getY() / _x);
}
