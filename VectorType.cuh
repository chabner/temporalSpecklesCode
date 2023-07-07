#pragma once

#include "Simulation.cuh"
#include "ComplexType.cuh"

// ------------------------------------ Vector Type Class ------------------------------------ //
class VectorType
{
public:
	// Constructors
#if DIMS==3
	__host__ __device__ VectorType() {};
	__host__ __device__ ~VectorType() {};
	__host__ __device__ VectorType(float_type x, float_type y, float_type z) : _x(x), _y(y), _z(z) {};
	__host__ __device__ VectorType(float_type x) : _x(x), _y(x), _z(x) {};
	__host__ __device__ VectorType(const VectorType& v): _x(v._x), _y(v._y), _z(v._z) {};
	__host__ __device__ VectorType operator=(const VectorType& v) { _x = v._x; _y = v._y; _z = v._z; return *this; };
#else
	__host__ __device__ VectorType() {};
	__host__ __device__ ~VectorType() {};
	__host__ __device__ VectorType(float_type x, float_type y) : _x(x), _y(y) {};
	__host__ __device__ VectorType(float_type x) : _x(x), _y(x) {};
	__host__ __device__ VectorType(const VectorType& v) : _x(v._x), _y(v._y) {};
	__host__ __device__ VectorType operator=(const VectorType& v) { _x = v._x; _y = v._y; return *this; };
#endif

	// Operators
	__host__ __device__ float_type setx(float_type x) { _x = x; return _x; };
	__host__ __device__ float_type sety(float_type y) { _y = y; return _y; };
#if DIMS==3
	__host__ __device__ float_type setz(float_type z) { _z = z; return _z; };
#endif

	__host__ __device__ float_type x() const { return _x; };
	__host__ __device__ float_type y() const { return _y; };
#if DIMS==3
	__host__ __device__ float_type z() const { return _z; };
#endif

#if DIMS==3
	__host__ __device__ VectorType operator+(const VectorType& v) const { return VectorType(_x + v._x, _y + v._y, _z + v._z); };
#else
	__host__ __device__ VectorType operator+(const VectorType& v) const { return VectorType(_x + v._x, _y + v._y); };
#endif

#if DIMS==3
	__host__ __device__ VectorType operator-(const VectorType& v) const { return VectorType(_x - v._x, _y - v._y, _z - v._z); };
#else
	__host__ __device__ VectorType operator-(const VectorType& v) const { return VectorType(_x - v._x, _y - v._y); };
#endif

#if DIMS==3
	__host__ __device__ VectorType operator-() const { return VectorType(-_x, -_y, -_z); };
#else
	__host__ __device__ VectorType operator-() const { return VectorType(-_x, -_y); };
#endif

#if DIMS==3
	__host__ __device__ float_type operator*(const VectorType& v) const { return(fma(_x, v._x, fma(_y, v._y, _z * v._z))); };
#else
	__host__ __device__ float_type operator*(const VectorType& v) const { return(fma(_x, v._x, _y * v._y)); };
#endif

#if DIMS==3
	__host__ __device__ VectorType operator*(float_type a) const { return(VectorType(a * _x, a * _y, a * _z)); };
#else
	__host__ __device__ VectorType operator*(float_type a) const { return(VectorType(a * _x, a * _y)); };
#endif

	__host__ __device__ friend VectorType operator*(float_type a, const VectorType& v);

	// Friend funcitons
	__host__ __device__ friend float_type abs(const VectorType& v);
	__host__ __device__ friend float_type rabs(const VectorType& v);
	__host__ __device__ friend float_type maxComponent(const VectorType& v);
	__host__ __device__ friend float_type sumSquare(const VectorType& v);

private:
	float_type _x;
	float_type _y;
#if DIMS==3
	float_type _z;
#endif
};

// ------------------------------------ Friend Functions implementations ------------------------------------ //
__host__ __device__ VectorType operator*(float_type a, const VectorType& v)
{
#if DIMS==3
	return(VectorType(a * v._x, a * v._y, a * v._z));
#else
	return(VectorType(a * v._x, a * v._y));
#endif
}

__host__ __device__ float_type abs(const VectorType& v)
{
#ifdef __CUDA_ARCH__
#if DIMS==3
	return norm3d(v._x, v._y, v._z);
#else
	return hypot(v._x, v._y);
#endif

#else

#if DIMS==3
	return sqrt(v._x * v._x + v._y * v._y + v._z * v._z);
#else
	return sqrt(v._x * v._x + v._y * v._y);
#endif

#endif
}
__host__ __device__ float_type rabs(const VectorType& v)
{
#ifdef __CUDA_ARCH__
#if DIMS==3
	return rnorm3d(v._x, v._y, v._z);
#else
	return rhypot(v._x, v._y);
#endif

#else

#if DIMS==3
	return sqrt(v._x * v._x + v._y * v._y + v._z * v._z);
#else
	return sqrt(v._x * v._x + v._y * v._y);
#endif

#endif
}

__host__ __device__ float_type maxComponent(const VectorType& v)
{

#if DIMS==3
	return v._x > v._y ? (v._x > v._z ? v._x : v._z) : (v._y > v._z ? v._y : v._z);
#else
	return v._x > v._y ? v._x : v._y;
#endif
}

__host__ __device__ float_type sumSquare(const VectorType& v)
{
#if DIMS==3
	return fma(v._x, v._x, fma(v._y, v._y, v._z * v._z));
#else
	return fma(v._x, v._x, v._y * v._y);
#endif
}

// ------------------------------------ Other Functions implementations ------------------------------------ //
__device__ VectorType normalize(const VectorType& v)
{
	float_type rr = rabs(v);

#if DIMS==3
	return (rr == INFINITY ? VectorType( (float_type)0.0, (float_type)0.0, (float_type)1.0 ) : rr * v);

#else
	return (rr == INFINITY ? VectorType( (float_type)0.0, (float_type)1.0 ) : rr * v);
#endif
}



__device__ float_type random_vMF_cosine(float_type kappa, curandState_t* state)
{
#if DIMS==3
	float_type b = -kappa + hypot(kappa, (float_type)1.0);
#else
	float_type b = -2.0 * kappa + hypot(2.0 * kappa, (float_type)1.0);
#endif
	float_type dir_x0 = (1.0 - b) / (1.0 + b);

#if DIMS==3
	float_type s_c = kappa * dir_x0 + 2.0 * log(1.0 - dir_x0 * dir_x0);
#else
	float_type s_c = kappa * dir_x0 + 1.0 * log(1.0 - dir_x0 * dir_x0);
#endif
	float_type u, w_z, t;

	do
	{
#if DIMS==3
		float_type z = randUniform(state); // beta(1,1) is uniform
#else
		float_type z = sin(CUDART_PIO2 * randUniform(state)); // beta(0.5,0.5) is arcsin distribution
		z *= z;
#endif
		u = randUniform(state);
		w_z = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);

#if DIMS==3
		t = kappa * w_z + 2.0 * log(1.0 - dir_x0 * w_z) - s_c;
#else
		t = kappa * w_z + log(1.0 - dir_x0 * w_z) - s_c;
#endif
	} while (t < log(u));

	return w_z;
}

#if 0
__device__ VectorType vector_rotation(const VectorType& center, float_type cosineAngle, curandState_t* state)
{
	float_type b;

	float_type divFactor = 0.0;
	float_type r_divFactor;
	float2_type v;

	while (divFactor < EPSILON)
	{
		v.x = randNormal(state);
		v.y = randNormal(state);
		divFactor = hypot(v.x, v.y);
	}

	r_divFactor = 1 / divFactor;
	v.x *= r_divFactor;
	v.y *= r_divFactor;

	float_type sqrt_w = sqrt(abs(1.0 - cosineAngle * cosineAngle));
	VectorType sample(sqrt_w * v.x, sqrt_w * v.y, cosineAngle);

	// house the center
	float_type s = center.x() * center.x() + center.y() * center.y();

	VectorType mu_housed(center.x(), center.y(), 1.0);

	if (abs(s) < 1e-8)
	{
		b = 0.0;
	}
	else
	{
		float_type z = (center.z() <= 0.0 ? center.z() - 1.0 : -s / (center.z() + 1.0));
		b = 2.0 * (z * z) / (s + z * z);
		mu_housed = VectorType(mu_housed.x() / z, mu_housed.y() / z, 1.0);
	}

	VectorType sample_housed = VectorType(
		(1.0 - b * mu_housed.x() * mu_housed.x()) * sample.x() +
		(-b * mu_housed.x() * mu_housed.y()) * sample.y() +
		(-b * mu_housed.x() * mu_housed.z()) * sample.z(),

		(-b * mu_housed.y() * mu_housed.x()) * sample.x() +
		(1.0 - b * mu_housed.y() * mu_housed.y()) * sample.y() +
		(-b * mu_housed.y() * mu_housed.z()) * sample.z(),

		(-b * mu_housed.z() * mu_housed.x()) * sample.x() +
		(-b * mu_housed.z() * mu_housed.y()) * sample.y() +
		(1.0 - b * mu_housed.z() * mu_housed.z()) * sample.z());

	return sample_housed;
}
#else
__device__ VectorType vector_rotation(const VectorType& prevDir, float_type cosineAngle, curandState_t* state)
{
	float_type sinAngle = (randUniform(state) < 0.5 ? -1.0 : 1.0) * sqrt(1.0 - cosineAngle * cosineAngle);
#if DIMS==2
	return VectorType(prevDir.x() * cosineAngle - prevDir.y() * sinAngle, prevDir.x() * sinAngle + prevDir.y() * cosineAngle);
#else
	float_type phi = 2.0 * randUniform(state), sinphi, cosphi;
	sincospi(phi, &sinphi, &cosphi);

	if (abs(prevDir.z()) > (1.0 - EPSILON))
	{
		return VectorType(sinAngle * cosphi, sinAngle * sinphi, cosineAngle * (float_type)(signbit(prevDir.z()) ? -1.0 : 1.0));
	}
	else
	{
		float_type sintheta_w = sqrt(1.0 - prevDir.z() * prevDir.z());
		return VectorType(sinAngle * (prevDir.x() * prevDir.z() * cosphi - prevDir.y() * sinphi) / sintheta_w + prevDir.x() * cosineAngle,
			sinAngle * (prevDir.y() * prevDir.z() * cosphi + prevDir.x() * sinphi) / sintheta_w + prevDir.y() * cosineAngle,
			-sinAngle * cosphi * sintheta_w + prevDir.z() * cosineAngle);
	}
#endif
}
#endif

__device__ VectorType random_vMF_direction(const VectorType& center, float_type kappa, curandState_t* state)
{
	return(vector_rotation(center, random_vMF_cosine(kappa, state), state));
}


// In 2D, base have one element, in 3D base have two elements
// Assume v is normalized
__device__ void orthogonalBase(const VectorType& v, VectorType* base)
{
#if DIMS==3
	if (abs((float_type)1.0 - abs(v.z())) <= EPSILON)
	{
		base[0] = VectorType( 1.0, 0.0, 0.0 );
		base[1] = VectorType( 0.0, 1.0, 0.0 );
	}
	else
	{
		base[0] = VectorType( -v.y(), v.x(), 0.0);
		base[0] = normalize(base[0]);
		base[1] = VectorType(
			base[0].y()* v.z() - base[0].z() * v.y(),
			base[0].z()* v.x() - base[0].x() * v.z(),
			base[0].x()* v.y() - base[0].y() * v.x());

	}
#else
	base[0] = VectorType( -v.y(), v.x());
#endif
}

__device__ VectorType randomDirection(curandState_t* state)
{
#if 0
	float_type divFactor = 0.0;
	float_type r_divFactor;
	vector_type w;

	while (divFactor < EPSILON)
	{
		w.x = randNormal(state);
		w.y = randNormal(state);

#if DIMS==3
		w.z = randNormal(state);
		divFactor = norm3d(w.x, w.y, w.z);
#else
		divFactor = hypot(w.x, w.y);
#endif

	}

	r_divFactor = 1 / divFactor;
	w.x *= r_divFactor;
	w.y *= r_divFactor;

#if DIMS==3
	w.z *= r_divFactor;
#endif

	return w;
#else

#if DIMS==2
	float_type sinTheta, cosTheta;
	sincospi(2.0 * randUniform(state), &sinTheta, &cosTheta);
	return(VectorType( sinTheta , cosTheta ));
#else
	float_type sinPhi, cosPhi;
	sincospi(2.0 * randUniform(state), &sinPhi, &cosPhi);

	float_type cosTheta = fma(randUniform(state), (float_type)2.0, -(float_type)1.0);
	float_type positiveSinTheta = sqrt((float_type)1.0 - cosTheta * cosTheta);

	return(VectorType( positiveSinTheta * cosPhi, positiveSinTheta * sinPhi , cosTheta ));
#endif
#endif
}

// ------------------------------------ Meduim Point Type Struct ------------------------------------ //
// Specify info about a point inside a medium
typedef struct
{
	VectorType position;

	// Negative value for inactive path
	// Value 0 is outside the volume, the rest is the material number
	// material number used for medium, scatterer, temporal settings etc...
	ib32 material;

	// the idx of wavelengh of the point
	ub32 lambdaIdx;

	// sampled source idx: the number of the source which used to sample this point
	ub32 sourceIdx;

	// sampled source dt times diffusive argument D: for correlation algorithm, we pass the dt times D to the source
	// which is responsible to compute the last node temporal correlation
	// The source only needs to multiply with the momentum contribution
	float_type dtD;

} MediumPoint;

// ------------------------------------ Complex Vector Type Class ------------------------------------ //
class ComplexVectorType
{
public:
	// Constructors
#if DIMS==3
	__host__ __device__ ComplexVectorType() {};
	__host__ __device__ ~ComplexVectorType() {};
	__host__ __device__ ComplexVectorType(ComplexType x, ComplexType y, ComplexType z) : _x(x), _y(y), _z(z) {};
	__host__ __device__ ComplexVectorType(VectorType real, VectorType imag) : _x(real.x(), imag.x()), _y(real.y(), imag.y()), _z(real.z(), imag.z()) {};
	__host__ __device__ ComplexVectorType(const ComplexVectorType& v) : _x(v._x), _y(v._y), _z(v._z) {};
	__host__ __device__ ComplexVectorType operator=(const ComplexVectorType& v) { _x = v._x; _y = v._y; _z = v._z; return *this; };
#else
	__host__ __device__ ComplexVectorType() {};
	__host__ __device__ ~ComplexVectorType() {};
	__host__ __device__ ComplexVectorType(ComplexType x, ComplexType y) : _x(x), _y(y) {};
	__host__ __device__ ComplexVectorType(VectorType real, VectorType imag) : _x(real.x(), imag.x()), _y(real.y(), imag.y()) {};
	__host__ __device__ ComplexVectorType(const ComplexVectorType& v) : _x(v._x), _y(v._y) {};
	__host__ __device__ ComplexVectorType operator=(const ComplexVectorType& v) { _x = v._x; _y = v._y; return *this; };
#endif

	// Operators
	__host__ __device__ ComplexType x() const { return _x; };
	__host__ __device__ ComplexType y() const { return _y; };
#if DIMS==3
	__host__ __device__ ComplexType z() const { return _z; };
#endif

	__host__ __device__ void setxr(float_type x) { _x.setReal(x); };
	__host__ __device__ void setxc(float_type x) { _x.setImag(x); };
	__host__ __device__ void setyr(float_type y) { _y.setReal(y); };
	__host__ __device__ void setyc(float_type y) { _y.setImag(y); };
#if DIMS==3
	__host__ __device__ void setzr(float_type z) { _z.setReal(z); };
	__host__ __device__ void setzc(float_type z) { _z.setImag(z); };
#endif

#if DIMS==3
	__host__ __device__ ComplexVectorType operator+(const ComplexVectorType& v) const { return ComplexVectorType(_x + v._x, _y + v._y, _z + v._z); };
#else
	__host__ __device__ ComplexVectorType operator+(const ComplexVectorType& v) const { return ComplexVectorType(_x + v._x, _y + v._y); };
#endif

#if DIMS==3
	__host__ __device__ ComplexVectorType operator-(const ComplexVectorType& v) const { return ComplexVectorType(_x - v._x, _y - v._y, _z - v._z); };
#else
	__host__ __device__ ComplexVectorType operator-(const ComplexVectorType& v) const { return ComplexVectorType(_x - v._x, _y - v._y); };
#endif

#if DIMS==3
	__host__ __device__ ComplexVectorType operator-() const { return ComplexVectorType(-_x, -_y, -_z); };
#else
	__host__ __device__ ComplexVectorType operator-() const { return ComplexVectorType(-_x, -_y); };
#endif

	
#if DIMS==3
	__host__ __device__ ComplexType operator*(const VectorType& v) const { return(cfma(_x, v.x(), cfma(_y, v.y(), _z * v.z()))); };
#else
	__host__ __device__ ComplexType operator*(const VectorType& v) const { return(cfma(_x, v.x(), _y * v.y())); };
#endif

	// Friend funcitons
	__device__ friend ComplexVectorType cfma(float_type a, const VectorType& b, const ComplexVectorType& c);
	__device__ friend ComplexVectorType cfma(ComplexType a, const ComplexVectorType& b, const ComplexVectorType& c);
	__device__ friend ComplexType sumSquare(const ComplexVectorType& v);
	__device__ friend VectorType realMult(const ComplexType& a, const ComplexVectorType& v);

private:
	ComplexType _x;
	ComplexType _y;
#if DIMS==3
	ComplexType _z;
#endif
};

// ------------------------------------ Friend Functions implementations ------------------------------------ //
__device__ ComplexVectorType cfma(float_type a, const VectorType& b, const ComplexVectorType& c)
{
#if DIMS==3
	return ComplexVectorType(cfma(a, b.x(), c._x), cfma(a, b.y(), c._y), cfma(a, b.z(), c._z));
#else
	return ComplexVectorType(cfma(a, b.x(), c._x), cfma(a, b.y(), c._y));
#endif
}

__device__ ComplexVectorType cfma(ComplexType a, const ComplexVectorType& b, const ComplexVectorType& c)
{
#if DIMS==3
	return ComplexVectorType(cfma(a, b.x(), c._x), cfma(a, b.y(), c._y), cfma(a, b.z(), c._z));
#else
	return ComplexVectorType(cfma(a, b.x(), c._x), cfma(a, b.y(), c._y));
#endif
}

__device__ ComplexType sumSquare(const ComplexVectorType& v)
{
#if DIMS==3
	return cfma(v._x, v._x, cfma(v._y, v._y, v._z * v._z));
#else
	return cfma(v._x, v._x, v._y * v._y);
#endif
}

__device__ VectorType realMult(const ComplexType& a, const ComplexVectorType& v)
{
#if DIMS==3
	return VectorType(realMult(a, v._x), realMult(a, v._y), realMult(a, v._z));
#else
	return VectorType(realMult(a, v._x), realMult(a, v._y));
#endif
}
