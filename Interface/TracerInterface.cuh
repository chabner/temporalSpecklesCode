#pragma once

#include "../Simulation.cuh"

#include "SamplerInterface.cuh"

// ------------------------------------ Tracer Class ------------------------------------ //

class Tracer
{
public:
	// samplerHandler - handler to sampler handler class.
	// seed           - seed to generate random numbers
	Tracer(ErrorType* err, Sampler* samplerHandler, bool isCBS, ub64 seed);

	~Tracer();

	// HOST FUNCTION //
	// Run the tracing process.
	// v  - allocated in CPU, after tracing the result will be stored in v. size of Nl x Nv x Nt.
	virtual ErrorType trace(ComplexType* v) = 0;

	ub64 getZeroContributionPaths() { return zeroContributionPaths; };
	ub64 getPhotonsBudget() { return photonsBudget; };

	bool getCBS() { return isCBS; };

	ErrorType updateSeed(ub64 seed);

protected:
	Sampler* _samplerHandler;
	ComplexType* vDevice;

	ub32 Nl, Nv, Nw, Nt, P;
	ub64 zeroContributionPaths;
	ub32 maxDim;
	ub64 photonsBudget;

	ub32 allocationCount;

	bool isCBS;
	bool isCorrelation;

	bool isScalarFs;

	Tracer() {};

private:
	curandState_t* statePoolHost;
};


// ------------------------------------ Kernels ------------------------------------ //
namespace TracerNS
{
	__global__ void initPool(ub32 threadsNum, ub64 seed)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		if (threadNum < threadsNum)
		{
			curand_init(seed, threadNum, 0, statePool + threadNum);
		}
	}
}


// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
Tracer::Tracer(ErrorType* err, Sampler* samplerHandler, bool isCBS, ub64 seed): isCBS(isCBS)
{
	MEMORY_CHECK("Tracer allocation begin");
	_samplerHandler = samplerHandler;
	Nl = samplerHandler->getIlluminationSize();
	Nv = samplerHandler->getViewSize();
	Nw = samplerHandler->getWavelenghSize();
	Nt = samplerHandler->getSamplerSize();
	P  = samplerHandler->getBatchSize();

	isCorrelation = samplerHandler->isCorrelation();

	// Init tracing pool
	// The tracing tool for now is the maximal dim
	maxDim = 0;

	maxDim = (maxDim > Nl ? maxDim : Nl);
	maxDim = (maxDim > Nv ? maxDim : Nv);
	maxDim = (maxDim > (Nt * Nw * P) ? maxDim : (Nt * Nw * P));

	allocationCount = 0;

	if (cudaMalloc(&statePoolHost, sizeof(curandState_t) * maxDim) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	cudaMemcpyToSymbol(statePool, &statePoolHost, sizeof(curandState_t*), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

	if (cudaMalloc(&statePoolHost, sizeof(curandState_t) * maxDim) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	if (cudaMalloc(&vDevice, sizeof(ComplexType) * Nl * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	*err = updateSeed(seed);

	MEMORY_CHECK("Tracer allocation end");

	isScalarFs = samplerHandler->isScalarFs();
}

Tracer::~Tracer()
{
	MEMORY_CHECK("Tracer free begin");
	switch (allocationCount)
	{
	case 2:
		cudaFree(vDevice);
	case 1:
		cudaFree(statePoolHost);
	default:
		break;
	}

	MEMORY_CHECK("Tracer allocation end");
}

// ------------------------------------ Class Function Implementations ------------------------------------ //
ErrorType Tracer::updateSeed(ub64 seed)
{
	ub32 threadsNum = maxDim < THREADS_NUM ? maxDim : THREADS_NUM;
	ub32 blocksNum = (maxDim - 1) / THREADS_NUM + 1;

	TracerNS::initPool <<< blocksNum, threadsNum >>> (maxDim, seed);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		cudaFree(statePoolHost);

		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}
