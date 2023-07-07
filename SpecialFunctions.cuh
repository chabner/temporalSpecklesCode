#pragma once

#define BESSEL_ITERATIONS 20

#include "Simulation.cuh"
#include "ComplexType.cuh"

// ------------------------------------ Complex Bessel Functions ------------------------------------ //

// Implementation of some useful COMPLEX bessel functions

// A shortcut for log(besseli(0.5,x))
__host__ __device__ float_type logbesseli_05(float_type x)
{
	if (x == 0)
	{
		return 0;
	}
	else if (x < 5.0)
	{
		return log(sqrt((float_type)(2.0 / CUDART_PI)) * sinh(x) / sqrt(x));
	}
	else
	{
		return x - (float_type)(0.5) * log((float_type)(CUDART_PI * x));
	}
}

// The implementation of besseli(v,x)
// Assume v >= 0 and real
// Algorithm: see similar algorith to boost implemenation https://www.boost.org/doc/libs/1_72_0/libs/math/doc/html/math_toolkit/bessel/bessel_first.html
// Boost - Bessel Functions of the First and Second Kinds
// 
// For small x values we use the series in http://mhtlab.uwaterloo.ca/courses/me755/web_chap4.pdf
// Waterloo university, ME755 special functions class - Bessel Functions of the First and Second Kind - page 22
//
// For large x values, we use the formula in https://personal.math.ubc.ca/~cbm/aands/abramowitz_and_stegun.pdf - 9.2.19.
// 
// The limit between small and large values can be chosen similarly to https://dl.acm.org/doi/pdf/10.1145/7921.214331
// ALGORITHM 644 A Portable Package for Bessel Functions of a Complex Argumentand Nonnegative Order

__host__ __device__ ComplexType besseli(float_type nu, ComplexType x)
{
	float_type valLim = 3.0 + 2.0 * sqrt(nu + 1.0);
	ComplexType resI;

	if (abs(x) < valLim)
	{
		// Choose the small value path
		ComplexType Y = complexSquare(0.5 * x);
		ComplexType Zk = ComplexType(1.0);
		ComplexType Bk = ComplexType(1.0);
		resI = ComplexType(1.0);

		for (ub32 k = 1; k <= BESSEL_ITERATIONS; k++)
		{
			Zk = Y / ((float_type)k * ((float_type)k + nu));
			Bk = Zk * Bk;
			resI = resI + Bk;
		}

		// resI *= (x/2)^nu / gamma(1+nu)
		resI = (1.0/tgamma(1.0 + nu)) * complexExponent(nu * complexLog(0.5 * x)) * resI;
	}
	else
	{
		// Choose large value path - similar to boost implemenation.
		// See https://www.boost.org/doc/libs/1_56_0/boost/math/special_functions/detail/bessel_jy_asym.hpp
		
		ComplexType ix = ComplexType(-x.imag(), x.real());

		// Approximate phase:
		float_type mu = 4.0 * nu * nu;
		ComplexType denom = 4.0 * ix;
		ComplexType denom_mult = complexSquare(denom);

		ComplexType theta_v = ix - (0.5 * nu + 0.25) * CUDART_PI;

		theta_v = theta_v + (mu - 1.0) / (2.0 * denom);
		denom = denom * denom_mult;

		theta_v = theta_v + (mu - 1.0) * (mu - 25.0) / (6.0 * denom);
		denom = denom * denom_mult;

		theta_v = theta_v + (mu - 1.0) * (mu * mu - 114.0 * mu + 1073.0) / (5.0 * denom);
		denom = denom * denom_mult;

		theta_v = theta_v + (mu - 1.0) * (5.0 * mu * mu * mu - 1535.0 * mu * mu + 54703.0 * mu - 375733.0) / (14.0 * denom);
		ComplexType cosine_theta_v = ComplexType(cos(theta_v.real()) * cosh(theta_v.imag()), -sin(theta_v.real()) * sinh(theta_v.imag()));

		// Amplitude approximation
		bool isOverFlow = false;
		ComplexType Mv2 = ComplexType(1.0, 0.0);
		ComplexType txq = complexSquare(2.0 * ix);
		ComplexType txq_mult = txq;
		float_type muMult = mu - 1.0;

		Mv2 = Mv2 + 0.5 * muMult / txq;
		txq = txq * txq_mult;
		muMult *= (mu - 9.0);
		if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.375 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 25.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.3125 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 49.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.2734375 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 81.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.24609375 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 121.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.2255859375 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 169.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.20947265625 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 225.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.196380615234375 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 289.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.1854705810546875 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 361.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		if (!isOverFlow)
		{
			Mv2 = Mv2 + 0.176197052001953125 * muMult / txq;
			txq = txq * txq_mult;
			muMult *= (mu - 441.0);
			if (abs(txq) > 1e20 || muMult > 1e20) isOverFlow = true;
		}

		ComplexType Mv = complexSqrt((2.0 / CUDART_PI) * (Mv2 / ix));
		resI = complexExponent(ComplexType(0, -0.5 * nu * CUDART_PI)) * Mv * cosine_theta_v;
	}

	return resI;
}

__host__ __device__ ComplexType logbesseli(float_type nu, ComplexType x)
{
	return complexLog(besseli(nu, x));
}

__host__ __device__ float_type vMFnormalizationLog(float_type x)
{
#if DIMS==3
	if (x < EPSILON)
	{
		return -log((float_type)(4.0 * CUDART_PI));
	}
	else if (x < 5.0)
	{
		return log(x) - log((float_type)(4.0 * CUDART_PI) * sinh(x));
	}
	else
	{
		return log(x) - log((float_type)(2.0 * CUDART_PI)) - x;
	}
#else
	return -log(2.0 * CUDART_PI) - logbesseli(0, x).real();
#endif
}
