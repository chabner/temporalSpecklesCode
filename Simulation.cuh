#pragma once

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "math_constants.h"
#include "cublas_v2.h"

// ------------------------------------ Hyper Parameters ------------------------------------ //
#define THREADS_NUM 128
#define SMALL_THREADS_NUM 128

/**
 * PRECISION:
 * - DOUBLE
 * - SINGLE
 */
#define SINGLE 0
#define DOUBLE 1
#define PRECISION DOUBLE

#ifndef DIMS
#define DIMS 3
#endif

// maximal number of materials
// first material is allways outside the medium
#define MATERIAL_NUM 10

// #define TIME_REC

#if PRECISION==DOUBLE
#define EPSILON 1E-12
#else
#define EPSILON 1E-5
#endif

// #define MEMORYCHECK

#ifdef MEMORYCHECK
size_t free_byte, total_byte;
double free_db, total_db, used_db;

#define MEMORY_CHECK(prName)  cudaDeviceSynchronize(); \
	cudaMemGetInfo(&free_byte, &total_byte); \
	free_db = (double)free_byte; \
	total_db = (double)total_byte; \
	used_db = total_db - free_db; \
	printf("%s GPU memory usage: used = %f, free = %f MB, total = %f MB\n", \
		prName, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
#else
#define MEMORY_CHECK(prName) ;
#endif

// ------------------------------------ Data Structures ------------------------------------ //

// Standart types
typedef uint16_t ub16;
typedef uint32_t ub32;
typedef uint64_t ub64;
typedef int32_t  ib32;

#if PRECISION==DOUBLE
	typedef double  float_type;
	typedef double2 float2_type;
	typedef double3 float3_type;
	
#else
	typedef float  float_type;
	typedef float2 float2_type;
	typedef float3 float3_type;
#endif

// Illumination / view connection type
// in case of correlation, we also can decide for illumination and view for u1 and u2
typedef enum
{
	ConnectionTypeIllumination,
	ConnectionTypeView,
	ConnectionTypeIllumination2,
	ConnectionTypeView2
} ConnectionType;

// Connection Type for correlation only - u1 or u2
// The correlation C = u1 .* conj(u2)
typedef enum
{
	// multiple scattering, first connection of (l1->x0...->xb->v1) * (l2->x0...->xb->v2)
	C1_l1l2, 
	// multiple scattering, last  connection of (l1->x0...->xb->v1) * (l2->x0...->xb->v2)
	C1_v1v2, 
	// multiple scattering, first connection of (v1->x0...->xb->l1) * (l2->x0...->xb->v2)
	C2_v1l2, 
	// multiple scattering, last  connection of (v1->x0...->xb->l1) * (l2->x0...->xb->v2)
	C2_l1v2, 
	// multiple scattering, first connection of (l1->x0...->xb->v1) * (v2->x0...->xb->l2)
	C3_l1v2, 
	// multiple scattering, last  connection of (l1->x0...->xb->v1) * (v2->x0...->xb->l2)
	C3_v1l2, 
	// multiple scattering, first connection of (v1->x0...->xb->l1) * (v2->x0...->xb->l2)
	C4_v1v2, 
	// multiple scattering, last  connection of (v1->x0...->xb->l1) * (v2->x0...->xb->l2)
	C4_l1l2  
} ConnectionTypeCorrelation;

// Possible error types
typedef enum
{
	NO_ERROR,
	DEVICE_ERROR,
	ALLOCATION_ERROR,
	KERNEL_ERROR,
	CUBLAS_ERROR,
	NOT_SUPPORTED,
	MISC_ERROR,
	KERNEL_ERROR_GaussianBeamSource_initGaussianBeamSource,
	KERNEL_ERROR_GaussianBeamSource_randomizeViewDirection,
	KERNEL_ERROR_GaussianBeamSource_sampleFirstBeam,
	KERNEL_ERROR_GaussianBeamSource_firstPointProbabilityKernel,
	KERNEL_ERROR_GaussianBeamSource_initSamplingTable,
	KERNEL_ERROR_GaussianBeamSource_fillTimesInSamplingTable,
	KERNEL_ERROR_GaussianBeamSource_getMixtureIdx,
	KERNEL_ERROR_GaussianBeamSource_attachBuffersToSampler,
	KERNEL_ERROR_GaussianBeamSource_alphaPdfIcdfKernel,
	KERNEL_ERROR_GaussianBeamSource_samplingPdfKernel,
	KERNEL_ERROR_GaussianBeamSource_markBeamsKernel,
	KERNEL_ERROR_GaussianBeamSource_getPdfSum,
	KERNEL_ERROR_GaussianBeamSource_normalizeSamplingBuffers,
	KERNEL_ERROR_GaussianBeamSource_computeCdfBuffer,
	KERNEL_ERROR_GaussianBeamSource_normalizeCdfBuffer,
	KERNEL_ERROR_GaussianBeamSource_sampleCdfBuffer,
	KERNEL_ERROR_GaussianBeamSource_sampleDirectionProbability,
	KERNEL_ERROR_GaussianBeamSource_gGaussianBeam,
	KERNEL_ERROR_GaussianBeamSource_fGaussianBeam,
	KERNEL_ERROR_GaussianBeamSource_fsGaussianBeam,
	KERNEL_ERROR_GaussianBeamSource_isotropicGaussianSampling,
	KERNEL_ERROR_GaussianBeamSource_isotropicGaussianProbability,
	KERNEL_ERROR_Sampler_setTemporalPathsNum,
	KERNEL_ERROR_Sampler_copyBuffersToPoints,
	KERNEL_ERROR_Sampler_copyPointsToBuffers,
	KERNEL_ERROR_Sampler_copyPointsToBuffersThroughput,
	KERNEL_ERROR_Sampler_copyPointsToBuffersThreePoints,
	KERNEL_ERROR_Sampler_dtShiftKernel,
	KERNEL_ERROR_Sampler_copyPointsFromBuffersTemporal,
	KERNEL_ERROR_Sampler_randomizeDirectionsKernel,
	KERNEL_ERROR_Source_multPathContribution,
	KERNEL_ERROR_NEE_buildPermuteMatrix,
	KERNEL_ERROR_NEE_complexMultiplicativeInverse,
	KERNEL_ERROR_Scattering_amplitudeKernel,
	KERNEL_ERROR_Scattering_pdfKernel,
	KERNEL_ERROR_Scattering_newDirectionKernel,
	KERNEL_ERROR_Scattering_hgInnerScatteringNormalization,
	KERNEL_ERROR_HetroMedium_heterogeneousAttenuationKernel,
	KERNEL_ERROR_HetroMedium_heterogeneousSampleKernel,
	KERNEL_ERROR_HetroMedium_getMaterialKernel,
	KERNEL_ERROR_Medium_sampleRandomInsideKernel

} ErrorType;

// Point holder to be imlemented by the chosen source
typedef void* Point;

// ------------------------------------ GPU Constants ------------------------------------ //
cublasHandle_t cublasHandle; // Handle for cublas operations

__constant__ curandState_t* statePool; // a pool of states for generating random numbers

__constant__ ub32 lambdaNum; // number of wavelenghs
__constant__ float_type* lambdaValues; // the wavelengh of the simulation

// ------------------------------------ Simulation Class ------------------------------------ //
class Simulation
{
public:
	Simulation(ErrorType* err, ub32 batchNum, const float_type* lambdaIn, ub32 lambdaNumIn, ub32 gpuDeviceNumber = 0);
	~Simulation();

	ub32 getBatchSize() const { return _batchNum; }
	ub32 getWavelenghSize() const { return Nw; }

private:
	float_type* _lambda;
	ub32 _batchNum;
	ub32 Nw;
	ub32 _gpuDeviceNumber;

	ub32 allocationCount;

	Simulation() {};
};

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
Simulation::Simulation(ErrorType* err, ub32 batchNum, const float_type* lambdaIn, ub32 lambdaNumIn, ub32 gpuDeviceNumber):
	_batchNum(batchNum), Nw(lambdaNumIn), _gpuDeviceNumber(gpuDeviceNumber)
{
	allocationCount = 0;

	if (cudaSetDevice(gpuDeviceNumber) != cudaSuccess)
	{
		*err = ErrorType::DEVICE_ERROR;
		return;
	}

	MEMORY_CHECK("Simulation allocation begin");

	if (cudaMalloc(&_lambda, sizeof(float_type) * lambdaNumIn) != cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
	{
		*err = ErrorType::CUBLAS_ERROR;
		return;
	}
	allocationCount++;
	
	if(cudaMemcpy(_lambda, lambdaIn, sizeof(float_type) * lambdaNumIn, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	if (cudaMemcpyToSymbol(lambdaValues, &_lambda, sizeof(float_type*), 0, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	if (cudaMemcpyToSymbol(lambdaNum, &lambdaNumIn, sizeof(ub32), 0, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	MEMORY_CHECK("Simulation allocation end");

	*err = ErrorType::NO_ERROR;
}

Simulation::~Simulation()
{
	MEMORY_CHECK("Simulation free begin");

	switch (allocationCount)
	{
	case 2:
		cublasDestroy(cublasHandle);
	case 1:
		cudaFree(_lambda);
	default:
		break;
	}
	cudaDeviceReset();
	MEMORY_CHECK("Simulation free end");
}

// ------------------------------------ Kernels ------------------------------------ //

#if DIMS==2
	#define ISOTROPIC_AMPLITUDE 0.398942280401432678
	#define ISOTROPIC_PDF       0.159154943091895336
#else
	#define ISOTROPIC_AMPLITUDE 0.282094791773878143
	#define ISOTROPIC_PDF       0.079577471545947668
#endif

__inline__ __device__ float_type safeAcos(float_type x)
{
	if (x < -1.0 + EPSILON)
	{
		return CUDART_PI;
	}
	else if (x > 1.0 - EPSILON)
	{
		return 0.0;
	}
	return acos(x);
}

__device__ ub32 round_num(float_type a, ub32 N)
{
	int n;
#if PRECISION==DOUBLE
	n = __double2int_rn(a);
#else
	n = __float2int_rn(a);
#endif

	return ub32(n < 0 ? 0 : (n >= N ? N - 1 : n));
}

__device__ float_type randUniform(curandState_t* state)
{
#if PRECISION==DOUBLE
	return curand_uniform_double(state);
#else
	return curand_uniform(state);
#endif
}

// random number from [0, N)
__device__ ub32 randUniformInteger(curandState_t* state, ub32 N)
{
	return ceil(randUniform(state) * N) - 1;
}

__device__ float_type randNormal(curandState_t* state)
{
#if PRECISION==DOUBLE
	return curand_normal_double(state);
#else
	return curand_normal(state);
#endif
}

// code from https://www.geeksforgeeks.org/binary-search/
// Function to implement binary search
__device__ ub32 binarySearchKernel(const float_type* arr, ub32 N, float_type x)
{
	ub32 r = N - 1;
	ub32 l = 0;

	if (arr[0] > x)
	{
		return 0;
	}

	if (arr[r] < x)
	{
		return r;
	}

	while (l <= r) {
		ub32 m = l + (r - l) / 2;

		if (m == 0)
		{
			return 0;
		}

		if (arr[m] == x)
			return m;

		// Check if x is present at mid
		if (arr[m - 1] <= x && arr[m] > x)
			return m - 1;

		// If x greater, ignore left half
		if (arr[m] < x)
			l = m + 1;

		// If x is smaller, ignore right half
		else
			r = m - 1;
	}

	// if we reach here, then element was
	// not present
	return 0;
}

__device__ ub32 binarySearchKernel(const ub32* arr, ub32 N, ub32 x)
{
	ub32 r = N - 1;
	ub32 l = 0;

	if (arr[0] > x)
	{
		return 0;
	}

	if (arr[r] <= x)
	{
		return r;
	}

	while (l <= r) {
		ub32 m = l + (r - l) / 2;

		if (m == 0)
		{
			return 0;
		}

		if (arr[m] == x)
			return m;

		// Check if x is present at mid
		if (arr[m - 1] <= x && arr[m] > x)
			return m - 1;

		// If x greater, ignore left half
		if (arr[m] < x)
			l = m + 1;

		// If x is smaller, ignore right half
		else
			r = m - 1;
	}

	// if we reach here, then element was
	// not present
	return 0;
}
