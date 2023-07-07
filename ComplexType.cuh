#pragma once

#include "Simulation.cuh"

// ------------------------------------ Complex Type Class ------------------------------------ //
class ComplexType
{
public:
	// Constructors
	__host__ __device__ ComplexType() : x(0), y(0) {};
	__host__ __device__ ~ComplexType() {};
	__host__ __device__ ComplexType(float_type realPart, float_type complexPart) : x(realPart), y(complexPart) {};
	__host__ __device__ ComplexType(float_type realPart) : x(realPart), y(0) {};
	__host__ __device__ ComplexType(const ComplexType& cNum): x(cNum.x), y(cNum.y) {};
	__host__ __device__ ComplexType operator=(const ComplexType& b) { x = b.x; y = b.y; return *this; };
	__host__ __device__ ComplexType operator=(float_type& b) { x = b; y = 0; return *this;};

	// Operators
	__host__ __device__ float_type real() const { return x; };
	__host__ __device__ float_type imag() const { return y; };

	__host__ __device__ void setReal(float_type real) { x = real; };
	__host__ __device__ void setImag(float_type imag) { y = imag; };
	__host__ __device__ void conj() { y = -y; };

	__host__ __device__ ComplexType operator+(const ComplexType& b) const { return ComplexType(x + b.x, y + b.y); };
	__host__ __device__ ComplexType operator+(float_type b) const { return ComplexType(x + b, y); };
	__host__ __device__ friend ComplexType operator+(float_type a, const ComplexType& b);

	__host__ __device__ ComplexType operator-() const { return ComplexType( -x, -y ); }
	__host__ __device__ ComplexType operator-(const ComplexType& b) const { return ComplexType(x - b.x, y - b.y); };
	__host__ __device__ ComplexType operator-(float_type b) const { return ComplexType(x - b, y); };
	__host__ __device__ friend ComplexType operator-(float_type a, const ComplexType& b);

	__host__ __device__ ComplexType operator*(const ComplexType& b) const { return ComplexType(fma(x, b.x, -y * b.y), fma(x, b.y, y * b.x)); };
	__host__ __device__ ComplexType operator*(float_type b) const { return { x * b, y * b }; };
	__host__ __device__ friend ComplexType operator*(float_type a, const ComplexType& b);
	__host__ __device__ friend float_type realMult(const ComplexType& a, const ComplexType& b);

	__host__ __device__ ComplexType operator/(const ComplexType& b) const {
		float_type denominator = 1.0 / (fma(b.x, b.x, b.y * b.y));
		return ComplexType((fma(x, b.x, y * b.y)) * denominator, (fma(y, b.x, -x * b.y)) * denominator);
	};
	__host__ __device__ ComplexType operator/(float_type b) const { return ComplexType(x/b, y/b); }
	__host__ __device__ friend ComplexType operator/(float_type a, const ComplexType& b);

	__host__ __device__ friend ComplexType cfma(const ComplexType& a, const ComplexType& b, const ComplexType& c);
	__host__ __device__ friend ComplexType cfma(float_type a, const ComplexType& b, const ComplexType& c);
	__host__ __device__ friend ComplexType cfma(float_type a, float_type b, const ComplexType& c);

	// Friend funcitons
	__host__ __device__ friend ComplexType complexSqrt(const ComplexType& a);
	__host__ __device__ friend ComplexType rComplexSqrt(const ComplexType& a);
	__host__ __device__ friend ComplexType complexExponent(const ComplexType& a);
	__host__ __device__ friend float_type abs(const ComplexType& a);
	__host__ __device__ friend float_type absSquare(const ComplexType& a);
	__host__ __device__ friend ComplexType complex2timesSinh(const ComplexType& expPart, float_type realPart);
	__host__ __device__ friend ComplexType complex2timesSinh(const ComplexType& expPart, const ComplexType& realPart);
	__host__ __device__ friend ComplexType complexSquare(const ComplexType& a);
	__host__ __device__ friend float_type realComplexLog(const ComplexType& val);
	__host__ __device__ friend ComplexType complexLog(const ComplexType& val);

private:
	float_type x;
	float_type y;
};

// ------------------------------------ Friend Functions implementations ------------------------------------ //
__host__ __device__ ComplexType operator+(float_type a, const ComplexType& b)
{
	return ComplexType(a + b.x, b.y);
}

__host__ __device__ ComplexType operator-(float_type a, const ComplexType& b)
{
	return ComplexType(a - b.x, -b.y);
}

__host__ __device__ ComplexType operator*(float_type a, const ComplexType& b)
{
	return ComplexType(a * b.x, a * b.y);
}

__host__ __device__ float_type realMult(const ComplexType& a, const ComplexType& b)
{
	return fma(a.x, b.x, -a.y * b.y);
}

__host__ __device__ ComplexType operator/(float_type a, const ComplexType& b)
{
	a /= (fma(b.x, b.x, b.y * b.y));
	return ComplexType(a * b.x, -a * b.y);
}

__host__ __device__ ComplexType cfma(const ComplexType& a, const ComplexType& b, const ComplexType& c)
{
	return ComplexType( fma(a.x, b.x, fma(-a.y, b.y, c.x)) , fma(a.x, b.y, fma(a.y, b.x, c.y)) );
}

__host__ __device__ ComplexType cfma(float_type a, const ComplexType& b, const ComplexType& c)
{
	return ComplexType( fma(a, b.x, c.x) , fma(a, b.y, c.y) );
}

__host__ __device__ ComplexType cfma(float_type a, float_type b, const ComplexType& c)
{
	return ComplexType( fma(a, b, c.x) , c.y );
}

// Algorithm: https://www.boost.org/doc/libs/1_78_0/boost/multiprecision/complex_adaptor.hpp: eval_sqrt
   // Use the following:
   // sqrt(z) = (s, zi / 2s)       for zr >= 0
   //           (|zi| / 2s, +-s)   for zr <  0
   // where s = sqrt{ [ |zr| + sqrt(zr^2 + zi^2) ] / 2 },
   // and the +- sign is the same as the sign of zi.
__host__ __device__ ComplexType complexSqrt(const ComplexType& a)
{
	if (a.y == 0 && a.x == 0)
	{
		return ComplexType(0., 0.);
	}

	float_type s = sqrt((abs(a.x) + sqrt(fma(a.x,a.x,a.y * a.y))) / 2.0);

	if (a.x >= 0.)
	{
		return(ComplexType(s, a.y / (2.0 * s)));
	}

	return(ComplexType(abs(a.y) / (2.0 * s), copysign(s, a.y)));
}

__host__ __device__ ComplexType rComplexSqrt(const ComplexType& a)
{
	float_type s2 = (abs(a.x) + sqrt(fma(a.x, a.x, a.y * a.y))) / 2.0;
	float_type mult = 1.0 / (s2 + (a.y * a.y) / (4.0 * s2));
	float_type s = sqrt(s2);

	if (a.x >= 0.)
	{
		return(ComplexType(s * mult, -a.y * mult / (2.0 * s)));
	}

	return(ComplexType((abs(a.y) * mult) / (2.0 * s), copysign(s * mult, -a.y)));
}

__host__ __device__ ComplexType complexExponent(const ComplexType& a)
{
	float_type expr, sina, cosa;
	expr = exp(a.x);
#ifdef __CUDA_ARCH__
	sincos(a.y, &sina, &cosa);
#else
	sina = sin(a.y);
	cosa = cos(a.y);
#endif  
	return ComplexType( expr * cosa , expr * sina );
}

__host__ __device__ float_type abs(const ComplexType& a)
{
	return sqrt(fma(a.x, a.x , a.y * a.y));
}

__host__ __device__ float_type absSquare(const ComplexType& a)
{
	return fma(a.x, a.x, a.y * a.y);
}

__host__ __device__ ComplexType complex2timesSinh(const ComplexType& expPart, float_type realPart)
{
	ComplexType positiveArgument = realPart + expPart;
	ComplexType negativeArgument = realPart - expPart;

	ComplexType expRes;
	if (positiveArgument.x > (float_type)-10.)
	{
		expRes = expRes + complexExponent(positiveArgument);
	}
	if (negativeArgument.x > (float_type)-10.)
	{
		expRes = expRes - complexExponent(negativeArgument);
	}

	return expRes;
}

__host__ __device__ ComplexType complex2timesSinh(const ComplexType& expPart, const ComplexType& realPart)
{
	ComplexType positiveArgument = realPart + expPart;
	ComplexType negativeArgument = realPart - expPart;

	ComplexType expRes;
	if (positiveArgument.x > (float_type)-10.)
	{
		expRes = expRes + complexExponent(positiveArgument);
	}
	if (negativeArgument.x > (float_type)-10.)
	{
		expRes = expRes - complexExponent(negativeArgument);
	}

	return expRes;
}

__host__ __device__ ComplexType complexSquare(const ComplexType& a)
{
	return ComplexType( (a.x - a.y) * (a.x + a.y) , (float_type)2.0 * a.x * a.y );
}

__host__ __device__ float_type realComplexLog(const ComplexType& val)
{
	return log(abs(val));
}

__host__ __device__ ComplexType complexLog(const ComplexType& val)
{
	ComplexType logRes;
	logRes.x = log(abs(val));
	logRes.y = atan2(val.y, val.x);
	return logRes;
}

// ------------------------------------ Forced double Complex Type Class ------------------------------------ //
class DoubleComplexType
{
public:
	// Constructors
	__host__ __device__ DoubleComplexType() : x(0), y(0) {};
	__host__ __device__ ~DoubleComplexType() {};
	__host__ __device__ DoubleComplexType(double realPart, double complexPart) : x(realPart), y(complexPart) {};
	__host__ __device__ DoubleComplexType(double realPart) : x(realPart), y(0) {};
	__host__ __device__ DoubleComplexType(const DoubleComplexType& cNum) : x(cNum.x), y(cNum.y) {};
	__host__ __device__ DoubleComplexType operator=(const DoubleComplexType& b) { x = b.x; y = b.y; return *this; };
	__host__ __device__ DoubleComplexType operator=(double& b) { x = b; y = 0; return *this; };

	// Operators
	__host__ __device__ double real() const { return x; };
	__host__ __device__ double imag() const { return y; };

	__host__ __device__ DoubleComplexType operator+(const DoubleComplexType& b) const { return DoubleComplexType(x + b.x, y + b.y); };
	__host__ __device__ DoubleComplexType operator+(double b) const { return DoubleComplexType(x + b, y); };
	__host__ __device__ friend DoubleComplexType operator+(double a, const DoubleComplexType& b);

	__host__ __device__ DoubleComplexType operator-() const { return DoubleComplexType(-x, -y); }
	__host__ __device__ DoubleComplexType operator-(const DoubleComplexType& b) const { return DoubleComplexType(x - b.x, y - b.y); };
	__host__ __device__ DoubleComplexType operator-(double b) const { return DoubleComplexType(x - b, y); };
	__host__ __device__ friend DoubleComplexType operator-(double a, const DoubleComplexType& b);

	__host__ __device__ DoubleComplexType operator*(const DoubleComplexType& b) const { return DoubleComplexType(fma(x, b.x, -y * b.y), fma(x, b.y, y * b.x)); };
	__host__ __device__ DoubleComplexType operator*(double b) const { return { x * b, y * b }; };
	__host__ __device__ friend DoubleComplexType operator*(double a, const DoubleComplexType& b);

	// Friend funcitons
	__host__ __device__ friend DoubleComplexType complexExponent(const DoubleComplexType& a);

private:
	double x;
	double y;
};

__host__ __device__ DoubleComplexType operator+(double a, const DoubleComplexType& b)
{
	return DoubleComplexType(a + b.x, b.y);
}

__host__ __device__ DoubleComplexType operator-(double a, const DoubleComplexType& b)
{
	return DoubleComplexType(a - b.x, -b.y);
}

__host__ __device__ DoubleComplexType operator*(double a, const DoubleComplexType& b)
{
	return DoubleComplexType(a * b.x, a * b.y);
}

__host__ __device__ DoubleComplexType complexExponent(const DoubleComplexType& a)
{
	double expr, sina, cosa;
	expr = exp(a.x);
#ifdef __CUDA_ARCH__
	sincos(a.y, &sina, &cosa);
#else
	sina = sin(a.y);
	cosa = cos(a.y);
#endif  
	return DoubleComplexType(expr * cosa, expr * sina);
}
