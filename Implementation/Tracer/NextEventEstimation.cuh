#pragma once

#include "../../Interface/TracerInterface.cuh"
#include "../../MatrixOperations.cuh"

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/gather.h>

#ifdef TIME_REC
#include <ctime>
#endif

#define TRACER_CHECK_ERR(res) if ((retVal = res) != ErrorType::NO_ERROR ) \
	{ \
		return retVal; \
	} 

#ifndef TIME_REC
#define TRACER_TIME_REC(inCode, timeVar)  inCode;
#else
#define TRACER_TIME_REC(inCode, timeVar) { \
	std::clock_t t_start_timeVar = std::clock(); \
	inCode; \
	cudaDeviceSynchronize(); \
	timeTracer.timeVar += std::clock() - t_start_timeVar; }
#endif

// printType: 0 - float
//            1 - complex
//            2 - ub32
//            3 - ib32
//            4 - float vector
#define PRINTVAR(varName, printType, varSize, headName) ; /* if(printType == 0) \
	{ \
		float_type* inVarCpu = (float_type*) malloc(sizeof(float_type) * varSize); \
		cudaMemcpy(inVarCpu, varName, sizeof(float_type) * varSize, cudaMemcpyKind::cudaMemcpyDeviceToHost); \
        printf("%s [", headName); \
		for(ub32 ii = 0; ii < varSize; ii++) \
		{ \
			printf("%e", inVarCpu[ii]); if(ii < (varSize - 1)) printf(", "); \
		} \
		printf("];\n"); free(inVarCpu); \
	} if(printType == 1) \
	{ \
		ComplexType* inVarCpu = (ComplexType*) malloc(sizeof(ComplexType) * varSize); \
		cudaMemcpy(inVarCpu, varName, sizeof(ComplexType) * varSize, cudaMemcpyKind::cudaMemcpyDeviceToHost); \
		printf("%s [", headName); \
		for(ub32 ii = 0; ii < varSize; ii++) \
		{ \
			printf("%e + %ei", inVarCpu[ii].real(), inVarCpu[ii].imag()); if(ii < (varSize - 1)) printf(", "); \
		} \
		printf("];\n"); free(inVarCpu); \
	} if(printType == 2) \
	{ \
		ub32* inVarCpu = (ub32*) malloc(sizeof(ub32) * varSize); \
		cudaMemcpy(inVarCpu, varName, sizeof(ub32) * varSize, cudaMemcpyKind::cudaMemcpyDeviceToHost); \
        printf("%s [", headName); \
		for(ub32 ii = 0; ii < varSize; ii++) \
		{ \
			printf("%d", inVarCpu[ii]); if(ii < (varSize - 1)) printf(", "); \
		} \
		printf("];\n"); free(inVarCpu); \
	} if(printType == 3) \
	{ \
		ib32* inVarCpu = (ib32*) malloc(sizeof(ib32) * varSize); \
		cudaMemcpy(inVarCpu, varName, sizeof(ib32) * varSize, cudaMemcpyKind::cudaMemcpyDeviceToHost); \
        printf("%s [", headName); \
		for(ub32 ii = 0; ii < varSize; ii++) \
		{ \
			printf("%d", inVarCpu[ii]); if(ii < (varSize - 1)) printf(", "); \
		} \
		printf("];\n"); free(inVarCpu); \
	} if(printType == 4) \
	{ \
		VectorType* inVarCpu = (VectorType*) malloc(sizeof(VectorType) * varSize); \
		cudaMemcpy(inVarCpu, varName, sizeof(VectorType) * varSize, cudaMemcpyKind::cudaMemcpyDeviceToHost); \
		for(ub32 ii = 0; ii < varSize; ii++) \
		{ \
			printf("%s %d: [%f, %f, %f] \n", headName, ii, inVarCpu[ii].x(), inVarCpu[ii].y(), inVarCpu[ii].z()); \
		} \
		free(inVarCpu); \
	}*/

// Let P be the batched paths number
// Let Nl be the illumination pixels
// Let Nv be the view pixels
// Let Nw be the wavelengh pixels
// Let Nt be the sampler pixels (time)

// ------------------------------------ Public Data Structures ------------------------------------ //

struct NextEventEstimationOptions
{
	bool isCBS;
	ub32 fullIterationsNumber;
	ub64 seed;
};

// Time tracking
#ifdef TIME_REC
class TimeTracer
{
public:
	std::clock_t t_start;
	std::clock_t gTime;
	std::clock_t fTime;
	std::clock_t fsTime;
	std::clock_t sampleSingleTime;
	std::clock_t sampleNextTime;
	std::clock_t probabilitySingleTime;
	std::clock_t probabilityMultipleTime;
	std::clock_t pathContributionTime;
	std::clock_t multiplySingle;
	std::clock_t multiplyMultiple;
	std::clock_t pointwiseMultipication;
	std::clock_t batchFinilize;

	std::clock_t multiplyMultiple_pointwiseMultipication;
	std::clock_t multiplyMultiple_matrixMultipication;
	
	TimeTracer() :
		t_start(0), gTime(0), fTime(0), fsTime(0), sampleSingleTime(0), sampleNextTime(0), 
		probabilitySingleTime(0), probabilityMultipleTime(0), pathContributionTime(0), multiplySingle(0), multiplyMultiple(0),
		pointwiseMultipication(0), batchFinilize(0), multiplyMultiple_pointwiseMultipication(0), multiplyMultiple_matrixMultipication(0)
	{}

	void printTimes() {
		printf("*** Time Report *** \n");
		printf("gTime: %f seconds \n", (float_type)gTime / (float_type)CLOCKS_PER_SEC);
		printf("fTime: %f seconds \n", (float_type)fTime / (float_type)CLOCKS_PER_SEC);
		printf("fsTime: %f seconds \n", (float_type)fsTime / (float_type)CLOCKS_PER_SEC);
		printf("sampleSingleTime: %f seconds \n", (float_type)sampleSingleTime / (float_type)CLOCKS_PER_SEC);
		printf("sampleNextTime: %f seconds \n", (float_type)sampleNextTime / (float_type)CLOCKS_PER_SEC);
		printf("probabilitySingleTime: %f seconds \n", (float_type)probabilitySingleTime / (float_type)CLOCKS_PER_SEC);
		printf("probabilityMultipleTime: %f seconds \n", (float_type)probabilityMultipleTime / (float_type)CLOCKS_PER_SEC);
		printf("pointwiseMultipication: %f seconds \n", (float_type)pointwiseMultipication / (float_type)CLOCKS_PER_SEC);
		printf("pathContributionTime: %f seconds \n", (float_type)pathContributionTime / (float_type)CLOCKS_PER_SEC);
		printf("multiplySingle: %f seconds \n", (float_type)multiplySingle / (float_type)CLOCKS_PER_SEC);
		printf("multiplyMultiple: %f seconds \n", (float_type)multiplyMultiple / (float_type)CLOCKS_PER_SEC);
		printf(" ----- pointwiseMultipication: %f seconds \n", (float_type)multiplyMultiple_pointwiseMultipication / (float_type)CLOCKS_PER_SEC);
		printf(" ----- matrixMultipication: %f seconds \n", (float_type)multiplyMultiple_matrixMultipication / (float_type)CLOCKS_PER_SEC);
		printf("batchFinilize: %f seconds \n", (float_type)batchFinilize / (float_type)CLOCKS_PER_SEC);
		printf("total recored time: %f seconds \n", (float_type)(gTime + fTime + fsTime + sampleSingleTime +
			sampleNextTime + probabilitySingleTime + probabilityMultipleTime + pointwiseMultipication + pathContributionTime +
			multiplySingle + multiplyMultiple + batchFinilize) / (float_type)CLOCKS_PER_SEC);
	};
};
#endif

// ------------------------------------ Next Event Estimation Tracer Class ------------------------------------ //

class NextEventEstimation : public Tracer
{
public:
	virtual ErrorType trace(ComplexType* v);

	NextEventEstimation(ErrorType* err, Sampler* samplerHandler, const NextEventEstimationOptions* options);
	~NextEventEstimation();

private:
	NextEventEstimation() {};

	// input
	ub32 fullIterationsNumber;
	ub32 allocationCount;	

	// current path lengh
	ub32 pL;

	// Sampled points
	Point* x;                                    // DEVICE POINTER, size of P
	Point* xPrev;                                // DEVICE POINTER, size of P
	ib32* pathsIdx;                              // DEVICE POINTER, size of P
	ib32* pathsNum;                              // DEVICE POINTER, size of P
	ib32* pathsIdxMs;                            // DEVICE POINTER, size of P
	ub32 participatingPathsNum;

	// Optimization - unless we deal with CBS + correlation, we can delay the mulatipication of the illumination with the view
	bool isFinilizeSummation;
	ub32 multipleScatteringPathNumBegin;

	// path contribution
	ComplexType* pathContb;                      // DEVICE POINTER, size of P * Nw * Nt

	// Sampling probability
	float_type* pl_raw;                          // DEVICE POINTER, size of P * Nl * Nw
	float_type* pl_rawHelper;                    // DEVICE POINTER, size of P * Nl * Nw
	float_type* pl_calculated;                   // DEVICE POINTER, size of P

	// Edge contributions: direct
	ComplexType* Gl0Buffer;                      // DEVICE POINTER, size of P * Nl * Nw * Nt
	ComplexType* GvBuffer;                       // DEVICE POINTER, size of P * Nv * Nw * Nt
	ComplexType* FsBuffer;                       // DEVICE POINTER, size of P * Nl * Nv * Nw * Nt
	ComplexType* GvSumBuffer;                    // DEVICE POINTER, size of P * Nv * Nw * Nt

	// Edge contributions: CBS
	ComplexType* Gv0Buffer;                      // DEVICE POINTER, size of P * Nv * Nw * Nt
	ComplexType* GlBuffer;                       // DEVICE POINTER, size of P * Nl * Nw * Nt
	ComplexType* GlSumBuffer;                    // DEVICE POINTER, size of P * Nl * Nw * Nt

	// Edge contributions: Correlation
	ComplexType* u2Gl0Buffer;                    // DEVICE POINTER, size of P * Nl * Nw * Nt
	ComplexType* u2GvBuffer;                     // DEVICE POINTER, size of P * Nv * Nw * Nt
	ComplexType* u2FsBuffer;                     // DEVICE POINTER, size of P * Nl * Nv * Nw * Nt

	// Edge contributions: CBS Correlation
	ComplexType* u2Gv0Buffer;                    // DEVICE POINTER, size of P * Nv * Nw * Nt
	ComplexType* u2GlBuffer;                     // DEVICE POINTER, size of P * Nl * Nw * Nt

	// Edge contributions: Temporal correlation
	ComplexType* C1_l1l2Buffer;                  // DEVICE POINTER, size of P * Nl * Nw * Nt

	// Edge contributions: Temporal correlation CBS
	ComplexType* C2_v1l2Buffer;                  // DEVICE POINTER, size of P * Nl * Nv * Nw * Nt
	ComplexType* C3_l1v2Buffer;                  // DEVICE POINTER, size of P * Nl * Nv * Nw * Nt
	ComplexType* C4_v1v2Buffer;                  // DEVICE POINTER, size of P * Nv * Nw * Nt

	// Helpers
	ComplexType* permutationMatrix;              // DEVICE POINTER, size of P * P
	float_type* deviceOne;                       // DEVICE POINTER, size of Nl * Nw
	ComplexType* deviceComplexOne;               // DEVICE POINTER, size of P
	ComplexType* multRes_1;                      // DEVICE POINTER, size of P * Nl * Nv * Nw * Nt
	ComplexType* multRes_2;                      // DEVICE POINTER, size of P * Nl * Nv * Nw * Nt

	// help functions
	ErrorType sapmleFirstPathPoints();
	ErrorType sapmleNextPathPoints();
	ErrorType twoPointsThroughput(ComplexType* target, ConnectionType cType);
	ErrorType threePointsThroughput(ComplexType* target, ConnectionType cType, ub32 isLastConnection);
	ErrorType threePointsThroughputSingle(ComplexType* target, ConnectionType cType);
	ErrorType pathProbabilitySingle();
	ErrorType pathProbabilityMultiple();
	ErrorType multipleResSingle(ComplexType* v);
	ErrorType multipleResMultiple(ComplexType* v);
	ErrorType multipleResFinal(ComplexType* v);
	ErrorType temporalCorrelationThroughput(ComplexType* target, ConnectionTypeCorrelation ccType, ub32 isLastConnection);
	ErrorType temporalCorrelationThroughputSingle(ComplexType* target);
	ErrorType getPathContribution(ComplexType* target, ConnectionType cType);

#ifdef TIME_REC
	TimeTracer timeTracer;
#endif
};

// ------------------------------------ Private Data Structures ------------------------------------ //
namespace NextEventEstimationNS
{
	struct IsNegative
	{
		__host__ __device__
			bool operator()(ib32 x)
		{
			return x < 0;
		}
	};

	struct FloatSqrt
	{
		__host__ __device__
			float_type operator()(float_type x) const
		{
			return sqrt(x);
		}
	};
}

// ------------------------------------ Kernels ------------------------------------ //
namespace NextEventEstimationNS
{
	__global__ void buildPermuteMatrix(ComplexType* permutationMatrix, const ib32* permutation, ub32 pathsNum, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < pathsNum)
		{
			permutationMatrix[threadNum * totalPathsNum + permutation[threadNum]] = 1.0;
		}
	}

	__global__ void buildPermuteMatrix(float_type* permutationMatrix, const ib32* permutation, ub32 pathsNum, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < pathsNum)
		{
			permutationMatrix[threadNum * totalPathsNum + permutation[threadNum]] = float_type(1.0);
		}
	}

	__global__ void complexMultiplicativeInverse(ComplexType* xOut, const float_type* xIn, ub32 xSize)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < xSize)
		{
			xOut[threadNum] = (float_type)1.0 / xIn[threadNum];
		}
	}

	__global__ void complexMultiplicativeInverse(ComplexType* xOut, const float_type* xIn, ub32 xSize, const ComplexType* constFactor, ub32 isCorrelation)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < xSize)
		{
			if (isCorrelation)
			{
				xOut[threadNum] = (*constFactor) * (*constFactor) / xIn[threadNum];
			}
			else
			{
				xOut[threadNum] = (*constFactor) / xIn[threadNum];
			}

		}
	}
}

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
NextEventEstimation::NextEventEstimation(ErrorType* err, Sampler* samplerHandler, const NextEventEstimationOptions* options):
	Tracer(err, samplerHandler, options->isCBS, options->seed)
{
	MEMORY_CHECK("nee allocation begin");
	if (*err != ErrorType::NO_ERROR)
	{
		return;
	}

	NextEventEstimation::fullIterationsNumber = options->fullIterationsNumber;
	allocationCount = 0;

	isFinilizeSummation = !(options->isCBS && isCorrelation);

	// *** Sampling *** //

	// sampling paths points data
	if ((*err = samplerHandler->allocatePoints(&x, P)) != ErrorType::NO_ERROR)
	{
		return;
	}
	allocationCount++;

	if ((*err = samplerHandler->allocatePoints(&xPrev, P)) != ErrorType::NO_ERROR)
	{
		return;
	}
	allocationCount++;

	if (cudaMalloc(&pathsIdx, sizeof(ib32) * P) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&pathsNum, sizeof(ib32) * P) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (isFinilizeSummation) if (cudaMalloc(&pathsIdxMs, sizeof(ib32) * P) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Path contribution *** //
	if (cudaMalloc(&pathContb, sizeof(ComplexType) * P * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Sampling probability *** //
	if (cudaMalloc(&pl_raw, sizeof(float_type) * P * Nl * Nw) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&pl_calculated, sizeof(float_type) * P) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&pl_rawHelper, sizeof(float_type) * P * Nl * Nw) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Edge contributions: direct *** //
	if (cudaMalloc(&Gl0Buffer, sizeof(ComplexType) * P * Nl * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&GvBuffer, sizeof(ComplexType) * P * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if(isFinilizeSummation) if (cudaMalloc(&GvSumBuffer, sizeof(ComplexType) * P * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&FsBuffer, sizeof(ComplexType) * P * Nl * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Edge contributions: CBS *** //
	if (isCBS) if (cudaMalloc(&Gv0Buffer, sizeof(ComplexType) * P * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (isCBS) if (cudaMalloc(&GlBuffer, sizeof(ComplexType) * P * Nl * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (isCBS && isFinilizeSummation) if (cudaMalloc(&GlSumBuffer, sizeof(ComplexType) * P * Nl * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Edge contributions: Correlation *** //
	if (isCorrelation) if (cudaMalloc(&u2Gl0Buffer, sizeof(ComplexType) * P * Nl * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (isCorrelation) if (cudaMalloc(&u2GvBuffer, sizeof(ComplexType) * P * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (isCorrelation) if (cudaMalloc(&u2FsBuffer, sizeof(ComplexType) * P * Nl * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Edge contributions: CBS Correlation *** //
	if (isCorrelation && isCBS) if (cudaMalloc(&u2Gv0Buffer, sizeof(ComplexType) * P * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (isCorrelation && isCBS) if (cudaMalloc(&u2GlBuffer, sizeof(ComplexType) * P * Nl * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Edge contributions: Temporal correlation *** //
	if (isCorrelation) if (cudaMalloc(&C1_l1l2Buffer, sizeof(ComplexType) * P * Nl * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Edge contributions: Temporal correlation CBS *** //

	if (isCorrelation && isCBS) if (cudaMalloc(&C2_v1l2Buffer, sizeof(ComplexType) * P * Nl * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (isCorrelation && isCBS) if (cudaMalloc(&C3_l1v2Buffer, sizeof(ComplexType) * P * Nl * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (isCorrelation && isCBS) if (cudaMalloc(&C4_v1v2Buffer, sizeof(ComplexType) * P * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	// *** Helpers *** //
	if (cudaMalloc(&deviceOne, sizeof(float_type) * Nl * Nw) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;
	thrust::fill_n(thrust::device, deviceOne, Nl * Nw, float_type{ 1.0 });

	if (cudaMalloc(&deviceComplexOne, sizeof(ComplexType) * P) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;
	thrust::fill_n(thrust::device, deviceComplexOne, P, ComplexType(1.0));

	if (cudaMalloc(&multRes_1, sizeof(ComplexType) * P * Nl * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&multRes_2, sizeof(ComplexType) * P * Nl * Nv * Nw * Nt) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&permutationMatrix, sizeof(ComplexType) * P * P) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
    allocationCount++;

	MEMORY_CHECK("nee allocation end");

	*err = ErrorType::NO_ERROR;
}

NextEventEstimation::~NextEventEstimation()
{
	MEMORY_CHECK("nee free begin");
	switch (allocationCount)
	{
	case 30:
		cudaFree(permutationMatrix);
	case 29:
		cudaFree(multRes_2);
	case 28:
		cudaFree(multRes_1);
	case 27:
		cudaFree(deviceComplexOne);
	case 26:
		cudaFree(deviceOne);
	case 25:
		if (isCorrelation && isCBS) cudaFree(C4_v1v2Buffer);
	case 24:
		if (isCorrelation && isCBS) cudaFree(C3_l1v2Buffer);
	case 23:
		if (isCorrelation && isCBS) cudaFree(C2_v1l2Buffer);
	case 22:
		if (isCorrelation) cudaFree(C1_l1l2Buffer);
	case 21:
		if (isCorrelation && isCBS) cudaFree(u2GlBuffer);
	case 20:
		if (isCorrelation && isCBS) cudaFree(u2Gv0Buffer);
	case 19:
		if (isCorrelation) cudaFree(u2FsBuffer);
	case 18:
		if (isCorrelation) cudaFree(u2GvBuffer);
	case 17:
		if (isCorrelation) cudaFree(u2Gl0Buffer);
	case 16:
		if (isCBS && isFinilizeSummation) cudaFree(GlSumBuffer);
	case 15:
		if (isCBS) cudaFree(GlBuffer);
	case 14:
		if (isCBS) cudaFree(Gv0Buffer);
	case 13:
		cudaFree(FsBuffer);
	case 12:
		if(isFinilizeSummation) cudaFree(GvSumBuffer);
	case 11:
		cudaFree(GvBuffer);
	case 10:
		cudaFree(Gl0Buffer);
	case 9:
		cudaFree(pl_rawHelper);
	case 8:
		cudaFree(pl_calculated);
	case 7:
		cudaFree(pl_raw);
	case 6:
		cudaFree(pathContb);
	case 5:
		if (isFinilizeSummation) cudaFree(pathsIdxMs);
	case 4:
		cudaFree(pathsNum);
	case 3:
		cudaFree(pathsIdx);
	case 2:
		_samplerHandler->freePoints(xPrev, P);
	case 1:
		_samplerHandler->freePoints(x, P);
	case 0:
	default:
		break;
	}
	MEMORY_CHECK("nee free end");
}

// ------------------------------------ Private Class Function Implementations ------------------------------------ //
ErrorType NextEventEstimation::sapmleFirstPathPoints()
{
	ub32 pathsLeftToSample = P;

	// pathsIdx: count the number of paths, beginning from 0.

	participatingPathsNum = P;
	thrust::copy(thrust::device, thrust::counting_iterator<ib32>(0), thrust::counting_iterator<ib32>(0) + P, pathsIdx);
	cudaDeviceSynchronize();

	while (pathsLeftToSample > 0)
	{
		thrust::copy(thrust::device, pathsIdx, pathsIdx + pathsLeftToSample, pathsNum);
		cudaDeviceSynchronize();

		ErrorType err = _samplerHandler->sampleFirst(xPrev, x, pathsIdx, pathsLeftToSample);
		cudaDeviceSynchronize();
		if (err != ErrorType::NO_ERROR)
		{
			return err;
		}
		
		// All negative paths are outside - make sure we sample them again.
		pathsLeftToSample = (ub32)(thrust::copy_if(thrust::device, pathsNum, pathsNum + pathsLeftToSample, pathsIdx,
			pathsIdx, NextEventEstimationNS::IsNegative()) - pathsIdx);
		cudaDeviceSynchronize();

		// Add number of missed paths
		zeroContributionPaths += pathsLeftToSample;
	}

	// pathsIdx: count the number of paths, beginning from 0.
	thrust::copy(thrust::device, thrust::counting_iterator<ib32>(0), thrust::counting_iterator<ib32>(0) + P, pathsIdx);
	cudaDeviceSynchronize();

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::sapmleNextPathPoints()
{
	ub32 pathsLeftToSample = participatingPathsNum;
	ErrorType err;

	if (pL == 1)
	{
		err = _samplerHandler->sampleSecond(xPrev, x, pathsIdx, pathsLeftToSample);
	}
	else
	{
		err = _samplerHandler->sampleNext(xPrev, x, pathsIdx, pathsLeftToSample, pL);
	}
	
	cudaDeviceSynchronize();
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// new paths numbers
	thrust::copy(thrust::device, thrust::counting_iterator<ib32>(0), thrust::counting_iterator<ib32>(0) + pathsLeftToSample, pathsNum);
	cudaDeviceSynchronize();

	participatingPathsNum = (ub32)(thrust::remove_if(thrust::device, pathsNum, pathsNum + pathsLeftToSample, pathsIdx,
		NextEventEstimationNS::IsNegative()) - pathsNum);
	cudaDeviceSynchronize();

	if (pL == 1 && isFinilizeSummation)
	{
		multipleScatteringPathNumBegin = participatingPathsNum;
		thrust::copy(thrust::device, thrust::counting_iterator<ib32>(0), thrust::counting_iterator<ib32>(0) + participatingPathsNum, pathsIdxMs);
		cudaDeviceSynchronize();
	}

	// if no path is removed or no paths are left - don't squeeze
	if (participatingPathsNum == pathsLeftToSample || participatingPathsNum == 0)
	{
		return ErrorType::NO_ERROR;
	}

	if (pL == 1 || !isFinilizeSummation)
	{
		// Build permutation matrix
		cudaMemset(permutationMatrix, 0, sizeof(ComplexType) * pathsLeftToSample * pathsLeftToSample);

		ub32 threadsNum = participatingPathsNum < THREADS_NUM ? participatingPathsNum : THREADS_NUM;
		ub32 blocksNum = (participatingPathsNum - 1) / THREADS_NUM + 1;

		NextEventEstimationNS::buildPermuteMatrix <<<blocksNum, threadsNum >>> (permutationMatrix,
			pathsNum, participatingPathsNum, pathsLeftToSample);
		cudaDeviceSynchronize();

		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_NEE_buildPermuteMatrix;
		}

		if ((err = matrixMultipication(multRes_1, Gl0Buffer, permutationMatrix, Nl * Nw * Nt, pathsLeftToSample, pathsLeftToSample, true)) != ErrorType::NO_ERROR) return err;
		if (isCBS && (err = matrixMultipication(multRes_1, Gv0Buffer, permutationMatrix, Nv * Nw * Nt, pathsLeftToSample, pathsLeftToSample, true)) != ErrorType::NO_ERROR) return err;
		if (isCorrelation && (err = matrixMultipication(multRes_1, u2Gl0Buffer, permutationMatrix, Nl * Nw * Nt, pathsLeftToSample, pathsLeftToSample, true)) != ErrorType::NO_ERROR) return err;
		if (isCorrelation && (err = matrixMultipication(multRes_1, C1_l1l2Buffer, permutationMatrix, Nl * Nw * Nt, pathsLeftToSample, pathsLeftToSample, true)) != ErrorType::NO_ERROR) return err;
		if ((isCorrelation && isCBS) && (err = matrixMultipication(multRes_1, u2Gv0Buffer, permutationMatrix, Nv * Nw * Nt, pathsLeftToSample, pathsLeftToSample, true)) != ErrorType::NO_ERROR) return err;
		if ((isCorrelation && isCBS) && (err = matrixMultipication(multRes_1, C2_v1l2Buffer, permutationMatrix, Nl * Nv * Nw * Nt, pathsLeftToSample, pathsLeftToSample, true)) != ErrorType::NO_ERROR) return err;
		if ((isCorrelation && isCBS) && (err = matrixMultipication(multRes_1, C3_l1v2Buffer, permutationMatrix, Nl * Nv * Nw * Nt, pathsLeftToSample, pathsLeftToSample, true)) != ErrorType::NO_ERROR) return err;
		if ((isCorrelation && isCBS) && (err = matrixMultipication(multRes_1, C4_v1v2Buffer, permutationMatrix, Nv * Nw * Nt, pathsLeftToSample, pathsLeftToSample, true)) != ErrorType::NO_ERROR) return err;
	}

	if (pL > 1 && isFinilizeSummation)
	{
		thrust::remove_if(thrust::device, pathsIdxMs, pathsIdxMs + pathsLeftToSample, pathsIdx, NextEventEstimationNS::IsNegative());
		cudaDeviceSynchronize();
	}

	thrust::remove_if(thrust::device, pathsIdx, pathsIdx + pathsLeftToSample, pathsIdx, NextEventEstimationNS::IsNegative());
	cudaDeviceSynchronize();

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::twoPointsThroughput(ComplexType* target, ConnectionType cType)
{
	ErrorType err = _samplerHandler->twoPointThroughput(target, x, pathsIdx, participatingPathsNum, cType, pL);
	cudaDeviceSynchronize();
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::threePointsThroughput(ComplexType* target, ConnectionType cType, ub32 isLastConnection)
{
	Point* current_x;
	Point* current_xPrev;

	if (isLastConnection)
	{
		current_x = xPrev;
		current_xPrev = x;
	}
	else
	{
		current_x = x;
		current_xPrev = xPrev;
	}

	ErrorType err = _samplerHandler->threePointThroughput(target, current_xPrev, current_x, pathsIdx, participatingPathsNum, cType, pL);
	cudaDeviceSynchronize();
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	//ub32 sourceSize = (cType == ConnectionType::ConnectionTypeIllumination ? Nl : Nv);
	//PRINTVAR(target, 1, P * sourceSize * Nt * Nw, "f res ");

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::threePointsThroughputSingle(ComplexType* target, ConnectionType cType)
{
	ErrorType err = _samplerHandler->threePointThroughputSingle(target, x, participatingPathsNum, cType);
	cudaDeviceSynchronize();
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::temporalCorrelationThroughput(ComplexType* target, ConnectionTypeCorrelation ccType, ub32 isLastConnection)
{
	Point* current_x;
	Point* current_xPrev;

	if (isLastConnection)
	{
		current_x = xPrev;
		current_xPrev = x;
	}
	else
	{
		current_x = x;
		current_xPrev = xPrev;
	}

	ErrorType err = _samplerHandler->temporalCorrelationThroughput(target, current_xPrev, current_x, pathsIdx, participatingPathsNum, ccType, pL);
	cudaDeviceSynchronize();
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::temporalCorrelationThroughputSingle(ComplexType* target)
{
	ErrorType err = _samplerHandler->temporalCorrelationThroughputSingle(target, x, participatingPathsNum);
	cudaDeviceSynchronize();
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	return ErrorType::NO_ERROR;
}


ErrorType NextEventEstimation::pathProbabilitySingle()
{
	ErrorType err = _samplerHandler->pathSingleProbability(pl_raw, x, participatingPathsNum);
	cudaDeviceSynchronize();
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// compute mean probability over all illuminations
	cublasStatus_t cuStat;

	float_type alpha = (float_type)(1.0 / (float_type)(Nl));
	float_type beta = 0.0;

#if PRECISION==DOUBLE
	cuStat = cublasDgemv(cublasHandle, CUBLAS_OP_T, Nl * Nw, participatingPathsNum, &alpha, pl_raw, Nl * Nw, deviceOne, 1, &beta, pl_calculated, 1);
#else
	cuStat = cublasSgemv(cublasHandle, CUBLAS_OP_T, Nl * Nw, participatingPathsNum, &alpha, pl_raw, Nl * Nw, deviceOne, 1, &beta, pl_calculated, 1);
#endif
	cudaDeviceSynchronize();
	if (cuStat != CUBLAS_STATUS_SUCCESS)
	{
		return ErrorType::CUBLAS_ERROR;
	}

	if (!isCorrelation)
	{
		// Square root for probability
		thrust::transform(thrust::device, pl_calculated, pl_calculated + participatingPathsNum, pl_calculated, NextEventEstimationNS::FloatSqrt());
		cudaDeviceSynchronize();
	}

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::pathProbabilityMultiple()
{
	cudaMemset(permutationMatrix, 0, sizeof(ComplexType) * P * P);

	ub32 threadsNum = participatingPathsNum < THREADS_NUM ? participatingPathsNum : THREADS_NUM;
	ub32 blocksNum = (participatingPathsNum - 1) / THREADS_NUM + 1;

	// Reduce pl_raw to active paths
	NextEventEstimationNS::buildPermuteMatrix <<<blocksNum, threadsNum >>> ((float_type*)permutationMatrix,
		pathsIdx, participatingPathsNum, P);
	cudaDeviceSynchronize();

	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_NEE_buildPermuteMatrix;
	}

	cublasStatus_t cuErr;
	float_type alpha = 1.0;
	float_type beta = 0.0;

#if PRECISION==DOUBLE
	cuErr = cublasDgemm(cublasHandle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		Nl * Nw, P, P,
		&alpha,
		pl_raw, Nl * Nw,
		(float_type*)permutationMatrix, P,
		&beta,
		pl_rawHelper, Nl * Nw);
#else
	cuErr = cublasSgemm(cublasHandle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		Nl * Nw, P, P,
		&alpha,
		pl_raw, Nl * Nw,
		(float_type*)permutationMatrix, P,
		&beta,
		pl_rawHelper, Nl * Nw);
#endif

	if (cuErr != CUBLAS_STATUS_SUCCESS)
	{
		return ErrorType::CUBLAS_ERROR;
	}

	PRINTVAR(pl_rawHelper, 0, P * Nl * Nw, "path probability before direction");

	// Multiply sampling pdf with the first point sampling probability
	ErrorType err = _samplerHandler->pathMultipleProbability(pl_rawHelper, xPrev, x, pathsIdx, participatingPathsNum);
	cudaDeviceSynchronize();
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	PRINTVAR(pl_rawHelper, 0, P * Nl * Nw, "path probability after direction");

	// compute mean probability over all illuminations
	alpha = (float_type)(1.0 / (float_type)(Nl));

#if PRECISION==DOUBLE
	cuErr = cublasDgemv(cublasHandle, CUBLAS_OP_T, Nl * Nw, P, &alpha, pl_rawHelper, Nl * Nw, deviceOne, 1, &beta, pl_raw, 1);
#else
	cuErr = cublasSgemv(cublasHandle, CUBLAS_OP_T, Nl * Nw, P, &alpha, pl_rawHelper, Nl * Nw, deviceOne, 1, &beta, pl_raw, 1);
#endif
	cudaDeviceSynchronize();
	if (cuErr != CUBLAS_STATUS_SUCCESS)
	{
		return ErrorType::CUBLAS_ERROR;
	}

	if (!isCorrelation)
	{
		// Square root for probability
		thrust::transform(thrust::device, pl_raw, pl_raw + participatingPathsNum, pl_calculated, NextEventEstimationNS::FloatSqrt());
	}
	else
	{
		cudaMemcpy(pl_calculated, pl_raw, sizeof(float_type) * participatingPathsNum, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	}
	
	cudaDeviceSynchronize();

	// Inverse the probability
	// pl_calculated: float   size of 1  x 1  x 1  x 1  x P
	// multRes_1:     complex size of 1  x 1  x 1  x 1  x P
	// multRes_1 = 1./pl_calculated + 0i
	NextEventEstimationNS::complexMultiplicativeInverse <<<blocksNum, threadsNum >>> (multRes_1, pl_calculated, participatingPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_NEE_complexMultiplicativeInverse;
	}

	// Multiple the probability with buffers
	// Gl0Buffer: size of Nl x 1  x Nw x Nt x P
	// multRes_1: size of 1  x 1  x 1  x 1  x P
	// Gl0Buffer = repmat(multRes_1, [Nl,1,Nw,Nt,1]) .* Gl0Buffer;
	if ((err = pointwiseMultipication(multRes_2,
		Gl0Buffer, Nl * Nw * Nt,
		multRes_1, 1, participatingPathsNum, true)) != ErrorType::NO_ERROR) return err;

	if (isCBS)
	{
		// Multiple the probability with buffers
		// Gv0Buffer: size of 1  x Nv x Nw x Nt x P
		// multRes_1: size of 1  x 1  x 1  x 1  x P
		// Gv0Buffer = repmat(multRes_1, [1,Nv,Nw,Nt,1]) .* Gv0Buffer;
		if ((err = pointwiseMultipication(multRes_2,
			Gv0Buffer, Nv * Nw * Nt,
			multRes_1, 1, participatingPathsNum, true)) != ErrorType::NO_ERROR) return err;
	}
	
	cudaDeviceSynchronize();

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::getPathContribution(ComplexType* target, ConnectionType cType)
{
	return _samplerHandler->pathContribution(target, x, pathsIdx, participatingPathsNum, cType, pL);
}

ErrorType NextEventEstimation::multipleResSingle(ComplexType* v)
{
	ErrorType retVal = ErrorType::NO_ERROR;
	ub32 threadsNum = P < THREADS_NUM ? P : THREADS_NUM;
	ub32 blocksNum = (P - 1) / THREADS_NUM + 1;

	if (isScalarFs)
	{
		// Inverse the probability
		// pl_calculated: float   size of 1  x 1  x 1  x 1  x P
		// multRes_1:     complex size of 1  x 1  x 1  x 1  x P
		// multRes_1 = FsBuffer(1)./pl_calculated + 0i
		NextEventEstimationNS::complexMultiplicativeInverse <<<blocksNum, threadsNum >>> (multRes_1, pl_calculated, P, FsBuffer, isCorrelation);
	}
	else
	{
		// Inverse the probability
		// pl_calculated: float   size of 1  x 1  x 1  x 1  x P
		// multRes_1:     complex size of 1  x 1  x 1  x 1  x P
		// multRes_1 = 1./pl_calculated + 0i
		NextEventEstimationNS::complexMultiplicativeInverse <<<blocksNum, threadsNum >>> (multRes_1, pl_calculated, P);
	}

	cudaDeviceSynchronize();

	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_NEE_complexMultiplicativeInverse;
	}

	// Multiple the probability
	// Gl0Buffer: size of Nl x 1  x Nw x Nt x P
	// multRes_1: size of 1  x 1  x 1  x 1  x P
	// multRes_2 = repmat(multRes_1, [Nl,1,Nw,Nt,1]) .* Gl0Buffer;
	TRACER_CHECK_ERR(pointwiseMultipication(multRes_2,
		Gl0Buffer, Nl * Nw * Nt,
		multRes_1, 1, P));

	// Compute the path contribution
	TRACER_TIME_REC(TRACER_CHECK_ERR(getPathContribution(pathContb, ConnectionType::ConnectionTypeIllumination)), pathContributionTime);

	// Multiple path contribution
	// multRes_2: size of Nl x 1  x Nw x Nt x P
	// pathContb: size of 1  x 1  x Nw x Nt x P
	// multRes_1 = repmat(pathContb, [Nl,1,1,1,1]) .* multRes_2;
	TRACER_CHECK_ERR(pointwiseMultipication(multRes_1,
		multRes_2, Nl,
		pathContb, 1, Nw * Nt * P));

	// Multiple Gl0 and Gv0
	// multRes_1: size of Nl x 1  x Nw, Nt x P
	// Gv0Buffer: size of 1  x Nv x Nw, Nt x P
	// multRes_2 = repmat(multRes_2, [1,Nv,1,1,1]) .* repmat(Gv0Buffer, [Nl,1,1,1,1]);
	TRACER_CHECK_ERR(pointwiseMultipication(multRes_2,
		multRes_1, Nl,
		isCBS ? Gv0Buffer : GvBuffer, Nv,
		Nw * Nt * P));

	ComplexType* resSingle;
	ComplexType* freeBuffer;

	if (isScalarFs)
	{
		resSingle = multRes_2;
		freeBuffer = multRes_1;
	}
	else
	{
		// Multiple the phase function
		// multRes_2: size of Nl x Nv x Nw x Nt x P
		// FsBuffer:  size of Nl x Nv x Nw x Nt x P
		// multRes_1 = multRes_1 .* FsBuffer;
		TRACER_CHECK_ERR(pointwiseMultipication(multRes_1, FsBuffer, multRes_2, Nl * Nv * Nw * Nt * P));
		resSingle = multRes_1;
		freeBuffer = multRes_2;
	}

	if (isCorrelation)
	{
		// !! In single scattering correlation, the path contribution equals to 1 !! //
		
		// Multiple u2Gl0 and u2Gv0
		// u2Gl0Buffer: size of Nl x 1  x Nw, Nt x P
		// u2Gv0Buffer: size of 1  x Nv x Nw, Nt x P
		// freeBuffer = repmat(u2Gl0Buffer, [1,Nv,1,1,1]) .* repmat(Gv0Buffer, [Nl,1,1,1,1]);
		TRACER_CHECK_ERR(pointwiseMultipication(freeBuffer,
			u2Gl0Buffer, Nl,
			isCBS ? u2Gv0Buffer : u2GvBuffer, Nv,
			Nw * Nt * P));

		ComplexType* currentBuffer;
		if (!isScalarFs)
		{
			currentBuffer = FsBuffer;
			// Multiple the phase function
			// freeBuffer: size of Nl x Nv x Nw x Nt x P
			// u2FsBuffer: size of Nl x Nv x Nw x Nt x P
			// currentBuffer =  freeBuffer .* u2FsBuffer;
			TRACER_CHECK_ERR(pointwiseMultipication(currentBuffer, u2FsBuffer, freeBuffer, Nl * Nv * Nw * Nt * P));
		}
		else
		{
			currentBuffer = freeBuffer;
		}

		// Multiple u1 with u2
		// resSingle:     size of Nl x Nv x Nw x Nt x P
		// currentBuffer: size of Nl x Nv x Nw x Nt x P
		// u2FsBuffer = multRes_2 .* conj(FsBuffer)
		TRACER_CHECK_ERR(pointwiseMultipication(u2FsBuffer, resSingle, false, currentBuffer, true, Nl * Nv * Nw * Nt * P));
		resSingle = u2FsBuffer;
	}

	// Sum all paths values
	// resSingle: size of Nl x Nv x Nw x Nt x P
	// v:         size of Nl x Nv x Nw x Nt x 1
	// v = v + sum(resSingle,5);

	return matrixMultipication(v, resSingle, deviceComplexOne,
		Nl * Nv * Nw * Nt, P, 1, false, (float_type) 1.0, (float_type) 1.0);
}

ErrorType NextEventEstimation::multipleResMultiple(ComplexType* v)
{
	ErrorType retVal = ErrorType::NO_ERROR;

	// *** CBS01 - Direct *** ///
	// field: l -> x0 -> x1 -> .... -> xb -> v
	// correlation: (l1 -> x0 -> x1 -> .... -> xb -> v1) * conj(l2 -> x0 -> x1 -> .... -> xb -> v2)

	ComplexType* currentGl0;
	ComplexType* currentGv;
	if (isCorrelation)
	{
		// multiply xb-1 -> xb -> v1 contriubution with conjugate of xb-1 -> xb -> v2 contriubution
		//   GvBuffer: size of 1 x Nv  x Nw x Nt x P
		// u2GvBuffer: size of 1 x Nv  x Nw x Nt x P
		// multRes_1 = GvBuffer .* conj(u2GvBuffer)
		if ((retVal = pointwiseMultipication(multRes_1, GvBuffer, false, u2GvBuffer, true, Nv * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Compute temporal contribution
		TRACER_CHECK_ERR(temporalCorrelationThroughput(FsBuffer, ConnectionTypeCorrelation::C1_v1v2, true));

		// Multiply with temporal contribution
		// FsBuffer : size of 1 x Nv  x Nw x Nt x P
		// multRes_1: size of 1 x Nv  x Nw x Nt x P
		// u2FsBuffer = FsBuffer .* multRes_1
		if ((retVal = pointwiseMultipication(u2FsBuffer, FsBuffer, multRes_1, Nv * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		currentGv = u2FsBuffer;

		if (!isFinilizeSummation)
		{
			// multiply l1 -> x0 -> x1 contriubution with conjugate of l2 -> x0 -> x1 contriubution
			// Gl0Buffer  : size of Nl x 1  x Nw x Nt x P
			// u2Gl0Buffer: size of Nl x 1  x Nw x Nt x P
			// multRes_1 = Gl0Buffer .* conj(u2Gl0Buffer)
			if ((retVal = pointwiseMultipication(multRes_1, Gl0Buffer, false, u2Gl0Buffer, true, Nl * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

			// Multiply with temporal contribution
			// multRes_1    : size of Nl x 1  x Nw x Nt x P
			// C1_l1l2Buffer: size of Nl x 1  x Nw x Nt x P
			// FsBuffer = multRes_1 .* C1_l1l2Buffer;
			if ((retVal = pointwiseMultipication(FsBuffer, multRes_1, C1_l1l2Buffer, Nl * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;
		}

		currentGl0 = FsBuffer;
	}
	else
	{
		currentGl0 = Gl0Buffer;
		currentGv = GvBuffer;
	}

	if (isFinilizeSummation)
	{
		// Sum Gv to GvSumBuffer
		// currentGv  : size of 1  x Nv x Nw x Nt x P
		// GvSumBuffer: size of 1  x Nv x Nw x Nt x P
		// GvSumBuffer = GvSumBuffer + currentGv;
		TRACER_CHECK_ERR(matrixSum(GvSumBuffer,
			currentGv, pathsIdxMs, Nv * Nw * Nt, Nv * Nw * Nt * participatingPathsNum));
	}
	else
	{
		// Multiple Gl0 and Gv
		// currentGl0: size of Nl x 1  x Nw x Nt x P
		// currentGv : size of 1  x Nv x Nw x Nt x P
		// multRes_2 = repmat(currentGl0, [1,Nv,1,1,1]) .* repmat(currentGv, [Nl,1,1,1,1]);
		TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2,
			currentGl0, Nl,
			currentGv, Nv,
			Nw * Nt * participatingPathsNum)), multiplyMultiple_pointwiseMultipication);
	}

	// *** Field Coherent back scattering *** //
	if (!isCorrelation && isCBS)
	{
		if (isFinilizeSummation)
		{
			// Sum Gl to GlSumBuffer
			// GlBuffer   : size of Nl x 1  x Nw x Nt x P
			// GlSumBuffer: size of Nl x 1  x Nw x Nt x P
			// GlSumBuffer = GlSumBuffer + GlBuffer;
			TRACER_CHECK_ERR(matrixSum(GlSumBuffer,
				GlBuffer, pathsIdxMs, Nl * Nw * Nt, Nl * Nw * Nt * participatingPathsNum));
		}
		else
		{
			// Multiple Gv0 and Gl
			// GlBuffer : size of Nl x 1  x Nw x Nt x P
			// Gv0Buffer: size of 1  x Nv x Nw x Nt x P
			// multRes_2 = multRes_2 + repmat(GlBuffer, [1,Nv,1,1,1]) .* repmat(Gv0Buffer, [Nl,1,1,1,1]);
			if ((retVal = pointwiseMultipication(multRes_2,
				GlBuffer, Nl,
				Gv0Buffer, Nv,
				Nw * Nt * participatingPathsNum,
				false, (float_type)1.0, (float_type)1.0)) != ErrorType::NO_ERROR) return retVal;
		}
	}

	// *** CBS02 - Opposite u1, Direct u2 *** ///
	// correlation: (v1 -> x0 -> x1 -> .... -> xb -> l1) * conj(l2 -> x0 -> x1 -> .... -> xb -> v2)
;
	if (isCorrelation && isCBS)
	{
		// multiply xb-1 -> xb -> l1 contriubution with conjugate of xb-1 -> xb -> v2 contriubution
		//   GlBuffer: size of Nl x 1  x Nw x Nt x P
		// u2GvBuffer: size of 1  x Nv x Nw x Nt x P
		// multRes_1 = GlBuffer .* conj(u2GvBuffer)
		if ((retVal = pointwiseMultipication(multRes_1,
			GlBuffer, Nl, false,
			u2GvBuffer, Nv, true,
			Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Compute temporal contribution
		TRACER_CHECK_ERR(temporalCorrelationThroughput(FsBuffer, ConnectionTypeCorrelation::C2_l1v2, true));

		// Multiply with temporal contribution
		// FsBuffer : size of Nl x Nv x Nw x Nt x P
		// multRes_1: size of Nl x Nv x Nw x Nt x P
		// u2FsBuffer = FsBuffer .* multRes_1
		if ((retVal = pointwiseMultipication(u2FsBuffer, FsBuffer, multRes_1, Nl * Nv * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// multiply v1 -> x0 -> x1 contriubution with conjugate of l2 -> x0 -> x1 contriubution
		// u2Gl0Buffer: size of Nl x 1  x Nw x Nt x P
		//   Gv0Buffer: size of 1  x Nv x Nw x Nt x P
		// multRes_1 = conj(u2Gl0Buffer) .* Gv0Buffer
		if ((retVal = pointwiseMultipication(multRes_1,
			u2Gl0Buffer, Nl, true,
			Gv0Buffer, Nv, false,
			Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Multiply with temporal contribution
		// multRes_1    : size of Nl x Nv x Nw x Nt x P
		// C2_v1l2Buffer: size of Nl x Nv x Nw x Nt x P
		// FsBuffer = multRes_1 .* C2_v1l2Buffer;
		if ((retVal = pointwiseMultipication(FsBuffer, multRes_1, C2_v1l2Buffer, Nl * Nv * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Add CBS02 contribution
		// Multiple FsBuffer and u2FsBuffer
		// FsBuffer:   size of Nl x Nv x Nw x Nt x P
		// u2FsBuffer: size of Nl x Nv x Nw x Nt x P
		// multRes_2 = multRes_2 + FsBuffer .* u2FsBuffer;
		if ((retVal = pointwiseMultipication(multRes_2, FsBuffer, u2FsBuffer, Nl * Nv * Nw * Nt * participatingPathsNum,
			false, (float_type)1.0, (float_type)1.0)) != ErrorType::NO_ERROR) return retVal;
	}

	// *** CBS03 - Direct u1, Opposite u2 *** ///
	// correlation: (l1 -> x0 -> x1 -> .... -> xb -> v1) * conj(v2 -> x0 -> x1 -> .... -> xb -> l2)
	
	if (isCorrelation && isCBS)
	{
		// multiply xb-1 -> xb -> v1 contriubution with conjugate of xb-1 -> xb -> l2 contriubution
		// u2GlBuffer: size of Nl x 1  x Nw x Nt x P
		//   GvBuffer: size of 1  x Nv x Nw x Nt x P
		// multRes_1 = GvBuffer .* conj(u2GlBuffer)
		if ((retVal = pointwiseMultipication(multRes_1,
			u2GlBuffer, Nl, true,
			GvBuffer, Nv, false,
			Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Compute temporal contribution
		TRACER_CHECK_ERR(temporalCorrelationThroughput(FsBuffer, ConnectionTypeCorrelation::C3_v1l2, true));

		// Multiply with temporal contribution
		// FsBuffer : size of Nl x Nv x Nw x Nt x P
		// multRes_1: size of Nl x Nv x Nw x Nt x P
		// u2FsBuffer = FsBuffer .* multRes_1
		if ((retVal = pointwiseMultipication(u2FsBuffer, FsBuffer, multRes_1, Nl * Nv * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// multiply l1 -> x0 -> x1 contriubution with conjugate of v2 -> x0 -> x1 contriubution
		//   Gl0Buffer: size of Nl x 1  x Nw x Nt x P
		// u2Gv0Buffer: size of 1  x Nv x Nw x Nt x P
		// multRes_1 = Gl0Buffer .* conj(u2Gv0Buffer)
		if ((retVal = pointwiseMultipication(multRes_1,
			Gl0Buffer, Nl, false,
			u2Gv0Buffer, Nv, true,
			Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Multiply with temporal contribution
		// multRes_1    : size of Nl x Nv x Nw x Nt x P
		// C3_l1v2Buffer: size of Nl x Nv x Nw x Nt x P
		// FsBuffer = multRes_1 .* C2_v1l2Buffer;
		if ((retVal = pointwiseMultipication(FsBuffer, multRes_1, C3_l1v2Buffer, Nl * Nv * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Add CBS03 contribution
		// Multiple FsBuffer and u2FsBuffer
		// FsBuffer:   size of Nl x Nv x Nw x Nt x P
		// u2FsBuffer: size of Nl x Nv x Nw x Nt x P
		// multRes_2 = multRes_2 + FsBuffer .* u2FsBuffer;
		if ((retVal = pointwiseMultipication(multRes_2, FsBuffer, u2FsBuffer, Nl * Nv * Nw * Nt * participatingPathsNum,
			false, (float_type)1.0, (float_type)1.0)) != ErrorType::NO_ERROR) return retVal;
	}

	// *** CBS04 - Opposite u1, Opposite u2 *** ///
	// correlation: (v1 -> x0 -> x1 -> .... -> xb -> l1) * conj(v2 -> x0 -> x1 -> .... -> xb -> l2)

	if (isCorrelation && isCBS)
	{
		// multiply xb-1 -> xb -> l1 contriubution with conjugate of xb-1 -> xb -> l2 contriubution
		//   GlBuffer: size of Nl x 1  x Nw x Nt x P
		// u2GlBuffer: size of Nl x 1  x Nw x Nt x P
		// multRes_1 = GlBuffer .* conj(u2GlBuffer)
		if ((retVal = pointwiseMultipication(multRes_1,
			GlBuffer, false,
			u2GlBuffer, true,
			Nl * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Compute temporal contribution
		TRACER_CHECK_ERR(temporalCorrelationThroughput(u2FsBuffer, ConnectionTypeCorrelation::C4_l1l2, true));

		// Multiply with temporal contribution
		// u2FsBuffer: size of Nl x 1 x Nw x Nt x P
		// multRes_1 : size of Nl x 1 x Nw x Nt x P
		// FsBuffer = u2FsBuffer .* multRes_1
		if ((retVal = pointwiseMultipication(FsBuffer, u2FsBuffer, multRes_1, Nl * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// multiply v1 -> x0 -> x1 contriubution with conjugate of v2 -> x0 -> x1 contriubution
		//   Gv0Buffer: size of 1  x Nv x Nw x Nt x P
		// u2Gv0Buffer: size of 1  x Nv x Nw x Nt x P
		// multRes_1 = Gv0Buffer .* conj(u2Gv0Buffer)
		if ((retVal = pointwiseMultipication(multRes_1,
			Gv0Buffer, false,
			u2Gv0Buffer, true,
			Nv * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Multiply with temporal contribution
		// multRes_1    : size of 1  x Nv x Nw x Nt x P
		// C4_v1v2Buffer: size of 1  x Nv x Nw x Nt x P
		// u2FsBuffer = multRes_1 .* C4_v1v2Buffer;
		if ((retVal = pointwiseMultipication(u2FsBuffer, multRes_1, C4_v1v2Buffer, Nv * Nw * Nt * participatingPathsNum)) != ErrorType::NO_ERROR) return retVal;

		// Multiple Gv0 and Gl
		// FsBuffer  : size of Nl x 1  x Nw x Nt x P
		// u2FsBuffer: size of 1  x Nv x Nw x Nt x P
		// multRes_2 = rmultRes_2 + epmat(FsBuffer, [1,Nv,1,1,1]) .* repmat(u2FsBuffer, [Nl,1,1,1,1]);
		if ((retVal = pointwiseMultipication(multRes_2,
			FsBuffer, Nl,
			u2FsBuffer, Nv,
			Nw * Nt * participatingPathsNum,
			false, (float_type)1.0, (float_type)1.0)) != ErrorType::NO_ERROR) return retVal;
	}

	if (!isFinilizeSummation)
	{
		// Sum all paths values and multiple by sqrt 2
		// multRes_2:       size of Nl x Nv x Nw x Nt x P
		// v:               size of Nl x Nv x Nw x Nt x 1
		// FIELD CBS:       v = 1/sqrt(2) * sum(multRes_2,5);
		// CORRELATION CBS: v = 0.5 * sum(multRes_2,5);
		// NO CBS:          v = sum(multRes_2,5);
		TRACER_TIME_REC(TRACER_CHECK_ERR(matrixMultipication(v, multRes_2, deviceComplexOne,
			Nl * Nv * Nw * Nt, participatingPathsNum, 1, false,
			isCBS ? (isCorrelation ? (float_type)0.5 : (float_type)CUDART_SQRT_HALF) : (float_type)1.0,
			(float_type)1.0)), multiplyMultiple_matrixMultipication);
	}

	return ErrorType::NO_ERROR;
}

ErrorType NextEventEstimation::multipleResFinal(ComplexType* v)
{
	ErrorType retVal = ErrorType::NO_ERROR;
	ComplexType* currentGl0;

	if (isFinilizeSummation && multipleScatteringPathNumBegin > 0)
	{
		if (isCorrelation)
		{
			// multiply l1 -> x0 -> x1 contriubution with conjugate of l2 -> x0 -> x1 contriubution
			//   Gl0Buffer: size of Nl x 1  x Nw x Nt x P
			// u2Gl0Buffer: size of Nl x 1  x Nw x Nt x P
			// multRes_1 = Gl0Buffer .* conj(u2Gl0Buffer)
			if ((retVal = pointwiseMultipication(multRes_1, Gl0Buffer, false, u2Gl0Buffer, true, Nl * Nw * Nt * multipleScatteringPathNumBegin)) != ErrorType::NO_ERROR) return retVal;

			// Multiply with temporal contribution
			// multRes_1    : size of Nl x 1  x Nw x Nt x P
			// C1_l1l2Buffer: size of Nl x 1  x Nw x Nt x P
			// FsBuffer = multRes_1 .* C1_l1l2Buffer;
			if ((retVal = pointwiseMultipication(FsBuffer, multRes_1, C1_l1l2Buffer, Nl * Nw * Nt * multipleScatteringPathNumBegin)) != ErrorType::NO_ERROR) return retVal;
			currentGl0 = FsBuffer;
		}
		else
		{
			currentGl0 = Gl0Buffer;
		}
		// Multiple Gl0 and Gv
		// currentGl0  : size of Nl x 1  x Nw x Nt x P
		// GvSumBuffer : size of 1  x Nv x Nw x Nt x P
		// multRes_2 = repmat(currentGl0, [1,Nv,1,1,1]) .* repmat(GvSumBuffer, [Nl,1,1,1,1]);
		TRACER_CHECK_ERR(pointwiseMultipication(multRes_2,
			currentGl0, Nl,
			GvSumBuffer, Nv,
			Nw * Nt * multipleScatteringPathNumBegin));

		if (isCBS)
		{
			// Multiple Gv0 and Gl
			// GlBuffer : size of Nl x 1  x Nw x Nt x P
			// Gv0Buffer: size of 1  x Nv x Nw x Nt x P
			// multRes_2 = multRes_2 + repmat(GlBuffer, [1,Nv,1,1,1]) .* repmat(Gv0Buffer, [Nl,1,1,1,1]);
			if ((retVal = pointwiseMultipication(multRes_2,
				GlSumBuffer, Nl,
				Gv0Buffer, Nv,
				Nw * Nt * multipleScatteringPathNumBegin,
				false, (float_type)1.0, (float_type)1.0)) != ErrorType::NO_ERROR) return retVal;
		}

		// Sum all paths values and multiple by sqrt 2
		// multRes_2:       size of Nl x Nv x Nw x Nt x P
		// v:               size of Nl x Nv x Nw x Nt x 1
		// FIELD CBS:       v = 1/sqrt(2) * sum(multRes_2,5);
		// CORRELATION CBS: v = 0.5 * sum(multRes_2,5);
		// NO CBS:          v = sum(multRes_2,5);
		TRACER_CHECK_ERR(matrixMultipication(v, multRes_2, deviceComplexOne,
			Nl * Nv * Nw * Nt, multipleScatteringPathNumBegin, 1, false,
			isCBS ? (isCorrelation ? (float_type)0.5 : (float_type)CUDART_SQRT_HALF) : (float_type)1.0,
			(float_type)1.0));
	}

	return ErrorType::NO_ERROR;
}

// ------------------------------------ Public Class Function Implementations ------------------------------------ //
ErrorType NextEventEstimation::trace(ComplexType* v)
{
	// init tracing values
	ErrorType retVal = ErrorType::NO_ERROR;
	zeroContributionPaths = 0;
	photonsBudget = 0;

	if (cudaMemset(vDevice, 0, sizeof(ComplexType) * Nl * Nv * Nw * Nt) != cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	// rendering loop

	for (ub32 itr = 0; itr < fullIterationsNumber; itr++)
	{
		pL = 1;

		// Sample paths
		TRACER_TIME_REC(TRACER_CHECK_ERR(sapmleFirstPathPoints()), sampleSingleTime);

		// Two point throughput of edge points
		TRACER_TIME_REC(TRACER_CHECK_ERR(twoPointsThroughput(Gl0Buffer, ConnectionType::ConnectionTypeIllumination)), gTime);
		TRACER_TIME_REC(TRACER_CHECK_ERR(twoPointsThroughput(isCBS ? Gv0Buffer : GvBuffer, ConnectionType::ConnectionTypeView)), gTime);

		if (isCorrelation)
		{
			TRACER_TIME_REC(TRACER_CHECK_ERR(twoPointsThroughput(u2Gl0Buffer, ConnectionType::ConnectionTypeIllumination2)), gTime);
			TRACER_TIME_REC(TRACER_CHECK_ERR(twoPointsThroughput(isCBS ? u2Gv0Buffer : u2GvBuffer, ConnectionType::ConnectionTypeView2)), gTime);
		}

		// First sampled point probability
		TRACER_TIME_REC(TRACER_CHECK_ERR(pathProbabilitySingle()), probabilitySingleTime);

		// Put zeros on all sum buffers
		if (isFinilizeSummation)
		{
			if (cudaMemset(GvSumBuffer, 0, sizeof(ComplexType) * Nv * Nw * Nt * P) != cudaSuccess)
			{
				return ErrorType::ALLOCATION_ERROR;
			}

			if (isCBS)
			{
				if (cudaMemset(GlSumBuffer, 0, sizeof(ComplexType) * Nl * Nw * Nt * P) != cudaSuccess)
				{
					return ErrorType::ALLOCATION_ERROR;
				}
			}
		}

		while (participatingPathsNum > 0)
		{
			if (pL == 1)
			{
				// Single scattering
				TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughputSingle(FsBuffer, ConnectionType::ConnectionTypeIllumination)), fsTime);

				if (isCorrelation)
				{
					TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughputSingle(u2FsBuffer, ConnectionType::ConnectionTypeIllumination2)), fsTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(temporalCorrelationThroughputSingle(u2FsBuffer)), fsTime);
				}
			}
			else if (pL == 2)
			{
				// Three point througphputs of edge points
				TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughput(multRes_1, ConnectionType::ConnectionTypeIllumination, false)), fTime);
				PRINTVAR(multRes_1, 1, Nl * Nw * Nt * participatingPathsNum, "multiple scattering illumination f pL = 2");
				TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2, Gl0Buffer, multRes_1, Nl * Nw * Nt * participatingPathsNum, true)), pointwiseMultipication);
				PRINTVAR(multRes_2, 1, Nl * Nw * Nt * participatingPathsNum, "multiple scattering illumination Gl0Buffer pL = 2");

				if (isCBS)
				{
					TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughput(multRes_1, ConnectionType::ConnectionTypeView, false)), fTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2, Gv0Buffer, multRes_1, Nv * Nw * Nt * participatingPathsNum, true)), pointwiseMultipication);
				}
				if (isCorrelation)
				{
					TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughput(multRes_1, ConnectionType::ConnectionTypeIllumination2, false)), fTime);
					PRINTVAR(multRes_1, 1, Nl * Nw * Nt * participatingPathsNum, "multiple scattering illumination2 f pL = 2");
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2, u2Gl0Buffer, multRes_1, Nl * Nw * Nt * participatingPathsNum, true)), pointwiseMultipication);
					PRINTVAR(multRes_1, 1, Nl * Nw * Nt * participatingPathsNum, "multiple scattering illumination2 Gl0Buffer pL = 2");
					TRACER_TIME_REC(TRACER_CHECK_ERR(temporalCorrelationThroughput(C1_l1l2Buffer, ConnectionTypeCorrelation::C1_l1l2, false)), fTime);
					PRINTVAR(multRes_1, 1, Nl * Nw * Nt * participatingPathsNum, "multiple scattering temporal correlation pL = 2");
				}
				if (isCorrelation && isCBS)
				{
					TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughput(multRes_1, ConnectionType::ConnectionTypeView2, false)), fTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2, u2Gv0Buffer, multRes_1, Nv * Nw * Nt * participatingPathsNum, true)), pointwiseMultipication);

					TRACER_TIME_REC(TRACER_CHECK_ERR(temporalCorrelationThroughput(C2_v1l2Buffer, ConnectionTypeCorrelation::C2_v1l2, false)), fTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(temporalCorrelationThroughput(C3_l1v2Buffer, ConnectionTypeCorrelation::C3_l1v2, false)), fTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(temporalCorrelationThroughput(C4_v1v2Buffer, ConnectionTypeCorrelation::C4_v1v2, false)), fTime);
				}

				// path probability
				TRACER_TIME_REC(TRACER_CHECK_ERR(pathProbabilityMultiple()), probabilityMultipleTime);
			}
			if (pL >= 2)
			{
				TRACER_TIME_REC(TRACER_CHECK_ERR(twoPointsThroughput(GvBuffer, ConnectionType::ConnectionTypeView)), gTime);
				PRINTVAR(GvBuffer, 1, Nv * Nw * Nt * participatingPathsNum, "multiple scattering view g pL >= 2");
				TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughput(multRes_1, ConnectionType::ConnectionTypeView, true)), fTime);
				PRINTVAR(multRes_1, 1, Nv * Nw * Nt * participatingPathsNum, "multiple scattering view f pL >= 2");
				TRACER_TIME_REC(TRACER_CHECK_ERR(getPathContribution(pathContb, ConnectionType::ConnectionTypeView)), pathContributionTime);
				PRINTVAR(pathContb, 1, participatingPathsNum, "path contribution pL >= 2");
				TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2, multRes_1, Nv, pathContb, 1, Nw * Nt * participatingPathsNum)), pointwiseMultipication);
				PRINTVAR(multRes_2, 1, Nv * Nw * Nt * participatingPathsNum, "path contribution * view f pL >= 2");
				TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_1, GvBuffer, multRes_2, Nv * Nw * Nt * participatingPathsNum, true)), pointwiseMultipication);
				PRINTVAR(GvBuffer, 1, Nv* Nw* Nt* participatingPathsNum, "path contribution * view f * g pL >= 2");

				if (isCBS)
				{
					TRACER_TIME_REC(TRACER_CHECK_ERR(twoPointsThroughput(GlBuffer, ConnectionType::ConnectionTypeIllumination)), gTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughput(multRes_1, ConnectionType::ConnectionTypeIllumination, true)), fTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(getPathContribution(pathContb, ConnectionType::ConnectionTypeIllumination)), pathContributionTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2, multRes_1, Nl, pathContb, 1, Nw * Nt * participatingPathsNum)), pointwiseMultipication);
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_1, GlBuffer, multRes_2, Nl * Nw * Nt * participatingPathsNum, true)), pointwiseMultipication);
				}

				if (isCorrelation)
				{
					TRACER_TIME_REC(TRACER_CHECK_ERR(twoPointsThroughput(u2GvBuffer, ConnectionType::ConnectionTypeView2)), gTime);
					PRINTVAR(u2GvBuffer, 1, Nv * Nw * Nt * participatingPathsNum, "multiple scattering view2 g pL >= 2");
					TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughput(multRes_1, ConnectionType::ConnectionTypeView2, true)), fTime);
					PRINTVAR(multRes_1, 1, Nv * Nw * Nt * participatingPathsNum, "multiple scattering view2 f pL >= 2");
					TRACER_TIME_REC(TRACER_CHECK_ERR(getPathContribution(pathContb, ConnectionType::ConnectionTypeView2)), pathContributionTime);
					PRINTVAR(pathContb, 1, participatingPathsNum, "path contribution2 pL >= 2");
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2, multRes_1, Nv, pathContb, 1, Nw * Nt * participatingPathsNum)), pointwiseMultipication);
					PRINTVAR(multRes_2, 1, Nv * Nw * Nt * participatingPathsNum, "path contribution * view2 f pL >= 2");
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_1, u2GvBuffer, multRes_2, Nv * Nw * Nt * participatingPathsNum, true)), pointwiseMultipication);
					PRINTVAR(GvBuffer, 1, Nv * Nw * Nt * participatingPathsNum, "path contribution * view2 f * g pL >= 2");
				}

				if (isCorrelation && isCBS)
				{
					TRACER_TIME_REC(TRACER_CHECK_ERR(twoPointsThroughput(u2GlBuffer, ConnectionType::ConnectionTypeIllumination2)), gTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(threePointsThroughput(multRes_1, ConnectionType::ConnectionTypeIllumination2, true)), fTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(getPathContribution(pathContb, ConnectionType::ConnectionTypeIllumination2)), pathContributionTime);
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_2, multRes_1, Nl, pathContb, 1, Nw * Nt * participatingPathsNum)), pointwiseMultipication);
					TRACER_TIME_REC(TRACER_CHECK_ERR(pointwiseMultipication(multRes_1, u2GlBuffer, multRes_2, Nl * Nw * Nt * participatingPathsNum, true)), pointwiseMultipication);
				}
			}

			// Multiple values
			if (pL == 1)
			{
				TRACER_TIME_REC(TRACER_CHECK_ERR(multipleResSingle(vDevice)), multiplySingle);
			}
			else
			{
				TRACER_TIME_REC(TRACER_CHECK_ERR(multipleResMultiple(vDevice)), multiplyMultiple);
			}
			photonsBudget = photonsBudget + participatingPathsNum;

			// Sample next scattering event
			TRACER_TIME_REC(TRACER_CHECK_ERR(sapmleNextPathPoints()), sampleNextTime);

			pL++;
		}

		// Finalize batch tracking
		TRACER_TIME_REC(TRACER_CHECK_ERR(multipleResFinal(vDevice)), batchFinilize);
	}

	if (cudaMemcpy(v, vDevice, sizeof(ComplexType) * Nl * Nv * Nw * Nt, cudaMemcpyKind::cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

#ifdef TIME_REC
	timeTracer.printTimes();
#endif

	return retVal;
}


