#pragma once
//#include <stdafx.h>
#include <iostream>
#include <cmath>
#include "vector.h"

namespace math {

	class matrix
	{
	public:
		matrix(const unsigned int);
		matrix(const unsigned int, const unsigned int);
		matrix(const matrix&);
		matrix();
		const unsigned int getLen() const { return slen; };
		const unsigned int getLen2()const { return slen2; };
		void setZero();
		const double norm() const;
		~matrix();
		double& operator[](int index);
		const double operator[](const int index) const;
		double& operator()(int idx1, int idx2);
		const double operator()(const int idx1, const int idx2) const;
		const matrix operator+(const matrix& _m) const;
		const matrix operator-(const matrix& _m) const;
		const matrix operator*(const double _x) const;
		const matrix operator/(const double _x) const;
		const vector operator*(const vector& _v) const;
		matrix operator+=(const matrix& _m);
		matrix operator-=(const matrix& _m);
		const matrix operator=(const matrix& _m);
		const matrix inverse();//自己加的，好像也有问题
	private:
		unsigned int slen;//行
		unsigned int slen2;//列
		double* elems;//单元数
	};

	matrix::matrix(const unsigned int _l) :slen{ _l }
	{
		slen2 = slen;//方阵
		elems = new double[slen * slen2];
		this->setZero();
	}

	inline matrix::matrix(const unsigned int _l, const unsigned int _l2) :slen{ _l }, slen2{ _l2 }
	{
		elems = new double[slen * slen2];
		this->setZero();
	}

	inline matrix::matrix(const math::matrix& _m)//输入_m矩阵
	{
		slen = _m.getLen();
		slen2 = _m.getLen2();
		elems = new double[slen * slen2];
		for (int i = 0; i < slen * slen2; i++)
		{
			elems[i] = _m[i];//读取_m矩阵
		}
	}

	inline matrix::matrix()//二维矩阵
	{
		slen = 2;
		slen2 = slen;
		elems = new double[slen * slen];
		this->setZero();
	}

	inline void matrix::setZero()//0矩阵
	{
		if (elems) {
			for (size_t i = 0; i < slen * slen2; i++)
			{
				elems[i] = 0;
			}
		}
	}

	inline const double matrix::norm() const//长度
	{
		double sum = 0.0;
		for (size_t i = 0; i < slen * slen2; i++)
		{
			sum += elems[i] * elems[i];
		}
		return sqrt(sum);
	}

	matrix::~matrix()
	{
		if (elems) free(elems);
	}

	inline double& matrix::operator[](int index)
	{
		return elems[index];
	}

	inline const double matrix::operator[](const int index) const
	{
		return elems[index];
	}

	inline double& matrix::operator()(int idx1, int idx2)
	{
		if (idx1 > slen || idx2 > slen2) {
			std::cerr << std::endl << "Error: index out of range\n";
			std::cout << "xxcell=" << idx1 << std::endl;
			std::cout << "slen=" << slen << std::endl;
			std::cout << "yycell=" << idx2 << std::endl;
			std::cout << "slen2=" << slen2 << std::endl;
		}
		return elems[(idx1 - 1) * slen2 + idx2 - 1];
	}

	inline const double matrix::operator()(const int idx1, const int idx2) const
	{
		if (idx1 > slen || idx2 > slen2) {
			std::cerr << "Error: index out of range\n";
		}
		return elems[(idx1 - 1) * slen2 + idx2 - 1];
	}

	inline const matrix matrix::operator+(const matrix& _m) const//矩阵加法
	{
		matrix mm(*this);
		for (int i = 0; i < slen * slen2; i++)
		{
			mm[i] = mm[i] + _m[i];
		}
		return matrix(mm);
	}

	inline matrix matrix::operator+=(const matrix& _m)
	{
		for (int i = 0; i < slen * slen2; i++)
		{
			elems[i] += _m[i];
		}
		return matrix(*this);
	}

	inline matrix matrix::operator-=(const matrix& _m)
	{
		for (int i = 0; i < slen * slen2; i++)
		{
			elems[i] -= _m[i];
		}
		return matrix(*this);
	}

	inline const matrix matrix::operator=(const matrix& _m)
	{
		if (slen != _m.getLen() || slen2 != _m.getLen2()) {
			free(this->elems);
			slen = _m.getLen();
			slen2 = _m.getLen2();
			this->elems = new double[slen * slen2];
		}
		for (int i = 0; i < slen * slen2; i++)
		{
			elems[i] = _m[i];
		}
		return matrix(*this);
	}

	inline const matrix matrix::operator-(const matrix& _m) const
	{
		matrix mm(*this);
		for (int i = 0; i < slen * slen2; i++)
		{
			mm[i] = mm[i] - _m[i];
		}
		return matrix(mm);
	}

	inline const matrix matrix::operator*(const double _x) const
	{
		matrix mm(*this);
		for (int i = 0; i < slen * slen2; i++)
		{
			mm[i] = mm[i] * _x;
		}
		return matrix(mm);
	}

	inline const matrix matrix::operator/(const double _x) const
	{
		matrix mm(*this);
		for (int i = 0; i < slen * slen2; i++)
		{
			mm[i] = mm[i] / _x;
		}
		return matrix(mm);
	}

	inline const vector matrix::operator*(const vector& _v) const
	{
		math::vector xi = math::vector(0.0, 0.0);
		for (int i = 1; i <= slen; i++)
		{
			xi(i) = 0.0;
			for (int j = 1; j <= slen; j++)
			{
				xi(i) += (*this)(i, j) * _v(j);
			}
		}
		return vector(xi);
	}

}

inline const math::matrix dyadic(const math::vector& v1, const math::vector& v2)  //二阶矩阵
{
	math::matrix m = math::matrix(2);
	for (size_t i = 1; i <= 2; i++)
		for (size_t j = 1; j <= 2; j++)
			m(i, j) = v1(i) * v2(j);
	return math::matrix(m);
}

inline const math::matrix transpose(const math::matrix& _m)  //矩阵的转置
{
	math::matrix m(_m);
	for (size_t i = 1; i <= 2; i++)
		for (size_t j = 1; j <= 2; j++)
			m(i, j) = _m(j, i);
	return math::matrix(m);
}

inline const math::matrix operator*(const double _x, const math::matrix& _m)  //矩阵的算子
{
	math::matrix m(_m);
	const unsigned int slen = _m.getLen();
	for (int i = 0; i < slen * slen; i++)
	{
		m[i] = m[i] * _x;
	}
	return math::matrix(m);
}

inline const math::matrix inverse(const math::matrix& _m)   //矩阵求逆,也不知道对不对！
{
	math::matrix b(_m);

	double det, j11, j12, j21, j22, con;
	det = _m(1, 1) * _m(2, 2) - _m(1, 2) * _m(2, 1);
	b(1, 1) = _m(2, 2);
	b(2, 2) = _m(1, 1);
	b(1, 2) = -_m(1, 2);
	b(2, 1) = -_m(2, 1);
	b = b / det;

	if (det == 0)
		std::cerr << "*WARNING* Zero pivot in Inverse\n";
	else
	{
		b = _m;
		for (int k = 1; k < 3; k++)
		{
			con = b(k, k);
			b(k, k) = 1.0;
			b(k, 1) = b(k, 1) / con;
			b(k, 2) = b(k, 2) / con;
			for (int i = 1; i < 3; i++)
				if (i /= k)
				{
					con = b(i, k);
					b(i, k) = 0.0;
					b(i, 1) = b(i, 1) - b(k, 1) * con;
					b(i, 2) = b(i, 2) - b(k, 2) * con;
				}
		}
	}
	return math::matrix(b);
}
