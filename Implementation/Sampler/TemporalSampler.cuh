#pragma once

#include "../../Interface/SamplerInterface.cuh"

#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// Let P be the batched paths number
// Let Nt be the time pixels
// Let Nw be the wavelengh pixels
// The convention is first dim is wavelengh, other dim is time

// ------------------------------------ GPU Constants ------------------------------------ //
namespace TemporalSamplerNS
{
	// Temporal point structre
	typedef struct
	{
		ub32 ticksNum;
		ub32 batchSize;
		float_type* dt;  // size of Nt
		float_type meanTick;
		ub32 negativeIdxNum;
		ub32 positiveIdxNum;
	} TemporalSamplerGpuDataStruct;

	__constant__ TemporalSamplerGpuDataStruct temporalSamplerGpuData;

	// Temporal material structre
	typedef struct {
		float_type D;
		VectorType U;
	} TemporalMaterialDataStructre;

	__constant__ TemporalMaterialDataStructre temporalMaterialGpuData[MATERIAL_NUM];

}

// ------------------------------------ Data Structures ------------------------------------ //
namespace TemporalSamplerNS
{
	struct TemporalPathPoint {
		MediumPoint xSampled;
		float_type randomNum;
		ub32 randomIdx;

		VectorType* xT;               // size of Nt
		float_type* totalDist;        // size of Nt
	};
}

// ------------------------------------ Temporal Sampler Class ------------------------------------ //
class TemporalSampler : public Sampler
{
public:
	// constructor
	// HOST FUNCION //
	// Init the functions for generating temporal tracer
	// ticks is CPU pointer in size of ticksNum
	TemporalSampler(ErrorType* err, Source* illuminationsHandle, Source* viewsHandle, const Simulation* simulationHandler, const Medium* mediumHandler,
		const Scattering* scatteringHandler, const float_type* ticks, ub32 ticksNum, bool isTemporalAmplitude = true, bool isForcedInsideBin = false);

	// HOST FUNCION //
	// Free allocated memory for tracer
	~TemporalSampler();

	// HOST FUNCION //
	// Add matrial temporal data
	ErrorType setMaterial(float_type D, VectorType U, ub32 materialNum);

	// Declerations
	virtual ErrorType allocatePoints(Point** p, ub32 pointsNumber);
	virtual void freePoints(Point* p, ub32 pointsNumber);
	virtual ErrorType sampleNext(Point* p0, Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ub32 pL);
	virtual ErrorType sampleFirst(Point* pa, Point* pb, ib32* pathsIdx, ub32 totalPathsNum);
	virtual ErrorType sampleSecond(Point* p1, Point* p2, ib32* pathsIdx, ub32 totalPathsNum);
	virtual ErrorType pathSingleProbability(float_type* probabilityRes, const Point* p1, ub32 totalPathsNum);
	virtual ErrorType pathMultipleProbability(float_type* probabilityRes, const Point* p1, const Point* p2, const ib32* pathsIdx, ub32 totalPathsNum);
	virtual ErrorType twoPointThroughput(ComplexType* gRes, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cType, ub32 pL);
	virtual ErrorType threePointThroughput(ComplexType* fRes, const Point* p1, const Point* p2, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cType, ub32 pL);
	virtual ErrorType threePointThroughputSingle(ComplexType* fsRes, const Point* p1, ub32 totalPathsNum, ConnectionType cType);
	virtual ErrorType temporalCorrelationThroughput(ComplexType* tRes, const Point* p1, const Point* p2, ib32* pathsIdx, ub32 totalPathsNum, ConnectionTypeCorrelation ccType, ub32 pL) {
		return ErrorType::NOT_SUPPORTED;
	};
	virtual ErrorType temporalCorrelationThroughputSingle(ComplexType* tsRes, const Point* p1, ub32 totalPathsNum) {
		return ErrorType::NOT_SUPPORTED;
	};
	virtual bool isCorrelation() { return false; };
	virtual ErrorType pathContribution(ComplexType* pContrb, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cn, ub32 pL);

	// DEBUG
	virtual void printAllPoints(Point* p, ub32 pointsNumber){
		TemporalSamplerNS::TemporalPathPoint* pp = (TemporalSamplerNS::TemporalPathPoint*)malloc(sizeof(TemporalSamplerNS::TemporalPathPoint) * pointsNumber);
		cudaMemcpy(pp, p, sizeof(TemporalSamplerNS::TemporalPathPoint) * pointsNumber, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		for (ub32 ii = 0; ii < pointsNumber; ii++)
		{
			printf("%d: (%f %f) %d \n", ii, pp[ii].xSampled.position.x(), pp[ii].xSampled.position.y(), pp[ii].xSampled.material);
		}

        std::free(pp);
	}
	// END DEBUG

private:
    ub32 allocationsCount;
    ub32 negativeIdxNum;
    ub32 positiveIdxNum;
    float_type mean_t;
	float_type* dt_cpu;                // size of Nt
    ub32* positiveIdxList;             // less than Nt
    ub32* negativeIdxList;             // less than Nt

	bool _isTemporalAmplitude;         // The amplitude function is computed on the shifted point, or just on the central sampled point.
	bool _isForcedInsideBin;           // The temporal path is forced to be sampled on the same bin of the sampled point

	ub32* temporalPathsNum;            // size of P * Nt, GPU buffer, contians Nt / 2 zeros, then Nt / 2 ones, and so on
	ub32* pathsNum;                    // size of P * Nt, GPU buffer, contians Nt zeros, then Nt ones, and so on

    // sampling buffers
	MediumPoint* mediumPointsBuffer;   // size of P * Nw * Nt
	MediumPoint* mediumPointsBuffer_2; // size of P
	VectorType* previousPointsBuffer;  // size of P * Nw * Nt
	VectorType* pathsVectorBuffer;     // size of P * Nw * Nt
	bool* isValidBuffer;               // size of P * Nt

	// Function helpers
	template<bool isFirstPoint>
	ErrorType dtshift_host(TemporalSamplerNS::TemporalPathPoint* sPa, TemporalSamplerNS::TemporalPathPoint* sPb, ib32* pathsIdx, ub32 totalPathsNum);

	// Block default constructor
	TemporalSampler();
};

// ------------------------------------ Kernels ------------------------------------ //
namespace TemporalSamplerNS
{
	__global__ void setTemporalPathsNum(ub32* temporalPathsNum, ub32* pathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 temporalNum = threadNum % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / temporalSamplerGpuData.ticksNum;

		if (pathNum < temporalSamplerGpuData.batchSize)
		{
			if (temporalNum < temporalSamplerGpuData.negativeIdxNum)
			{
				temporalPathsNum[threadNum] = 2 * pathNum;
			}
			else
			{
				temporalPathsNum[threadNum] = 2 * pathNum + 1;
			}
			
			pathsNum[threadNum] = pathNum;
		}
	}

	// COPY TO / FROM BUFFERS
	__global__ void copyBuffersToPoints(TemporalPathPoint* pa, const VectorType* pa_buffer,
		TemporalPathPoint* pb, const MediumPoint* pb_buffer, ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathsNum)
		{
			ub32 currentPathIdx = pathsIdx[pathNum];
			pa[currentPathIdx].xSampled.position = pa_buffer[pathNum];
			pa[currentPathIdx].xSampled.material = pb[currentPathIdx].xSampled.material;
			pa[currentPathIdx].randomNum = randUniform(statePool + pathNum);

			pb[currentPathIdx].xSampled = pb_buffer[pathNum];
			pb[currentPathIdx].randomNum = randUniform(statePool + pathNum);

			if (pb_buffer[pathNum].material == 0)
			{
				pathsIdx[pathNum] = -1;
			}
		}
	}

	__global__ void copyBuffersToPoints(TemporalPathPoint* pa, TemporalPathPoint* pb, const MediumPoint* pb_buffer, ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathsNum)
		{
			ub32 currentPathIdx = pathsIdx[pathNum];
			pa[currentPathIdx].xSampled = pb[currentPathIdx].xSampled;
			pa[currentPathIdx].randomNum = randUniform(statePool + pathNum);

			pb[currentPathIdx].xSampled = pb_buffer[pathNum];
			pb[currentPathIdx].randomNum = randUniform(statePool + pathNum);

			if (pb_buffer[pathNum].material == 0)
			{
				pathsIdx[pathNum] = -1;
			}
		}
	}

	__global__ void copyPointsToBuffers(const TemporalPathPoint* pa, VectorType* pa_buffer,
		const TemporalPathPoint* pb, MediumPoint* pb_buffer,
		const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;
		if (pathNum < totalPathsNum)
		{
			ub32 currentPath = pathsIdx[pathNum];

			pa_buffer[pathNum] = pa[currentPath].xSampled.position;
			pb_buffer[pathNum] = pb[currentPath].xSampled;
			pb_buffer[pathNum].dtD = 0;
		}
	}

	__global__ void copyPointsToBuffers(const TemporalPathPoint* pa, MediumPoint* pa_buffer,
		const TemporalPathPoint* pb, VectorType* pb_buffer,
		const ib32* pathsIdx, ub32 wavelenghNum, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 nw = threadNum % wavelenghNum;
		ub32 pathNum = threadNum / wavelenghNum;

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath = pathsIdx[pathNum];

			pa_buffer[pathNum] = pa[currentPath].xSampled;
			pb_buffer[pathNum] = pb[currentPath].xSampled.position;
			pa_buffer[pathNum].lambdaIdx = nw;
		}
	}

	__global__ void copyPointsToBuffers(const TemporalPathPoint* pb, MediumPoint* pb_buffer, ub32 wavelenghNum, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 nw = threadNum % wavelenghNum;
		ub32 pathNum = threadNum / wavelenghNum;

		if (pathNum < totalPathsNum)
		{
			pb_buffer[pathNum] = pb[pathNum].xSampled;
			pb_buffer[pathNum].lambdaIdx = nw;
		}
	}

	__global__ void copyPointsToBuffersThroughput(const TemporalPathPoint* pb, MediumPoint* pb_buffer,
		const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath = pathsIdx[pathNum];

			pb_buffer[threadNum].position = pb[currentPath].xT[temporalNum];
			pb_buffer[threadNum].material = pb[currentPath].xSampled.material;
			pb_buffer[threadNum].lambdaIdx = wavelenghNum;
		}
	}

	template <bool isTemporalAmplitude>
	__global__ void copyPointsToBuffersThreePoints(const TemporalPathPoint* pa, MediumPoint* pa_buffer,
		const TemporalPathPoint* pb, VectorType* pb_buffer, const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath = pathsIdx[pathNum];

			if (isTemporalAmplitude)
			{
				pa_buffer[threadNum].position = pa[currentPath].xT[temporalNum];
				pb_buffer[threadNum] = pb[currentPath].xT[temporalNum];
			}
			else
			{
				pa_buffer[threadNum].position = pa[currentPath].xSampled.position;
				pb_buffer[threadNum] = pb[currentPath].xSampled.position;
			}

			pa_buffer[threadNum].material = pa[currentPath].xSampled.material;
			pa_buffer[threadNum].lambdaIdx = wavelenghNum;
		}
	}

	template <bool isTemporalAmplitude>
	__global__ void copyPointsToBuffersThreePoints(const TemporalPathPoint* pa, MediumPoint* pa_buffer, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalSamplerGpuData.ticksNum * lambdaNum);
		
		if (pathNum < totalPathsNum)
		{
			if (isTemporalAmplitude)
			{
				pa_buffer[threadNum].position = pa[pathNum].xT[temporalNum];
			}
			else
			{
				pa_buffer[threadNum].position = pa[pathNum].xSampled.position;
			}

			pa_buffer[threadNum].material = pa[pathNum].xSampled.material;
			pa_buffer[threadNum].lambdaIdx = wavelenghNum;
		}
	}

	__global__ void copyPointsToBuffersPathContribution(const TemporalPathPoint* pa, MediumPoint* pa_buffer,
		VectorType* temporalPoints_buffer, ib32* negativePathOutsideIdx, ib32* positivePathOutsideIdx,
		const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 temporalNum = threadNum % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / temporalSamplerGpuData.ticksNum;

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath = pathsIdx[pathNum];
			if (temporalNum == 0)
			{
				pa_buffer[pathNum] = pa[currentPath].xSampled;
			}
			
			temporalPoints_buffer[threadNum] = pa[currentPath].xT[temporalNum];
			negativePathOutsideIdx[threadNum] = -1;
			positivePathOutsideIdx[threadNum] = ~(1 << 31);
		}
	}

	__global__ void initValidBuffer(bool* validBuffer, ub32 bufferSize)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < bufferSize)
		{
			validBuffer[threadNum] = true;
		}
	}

	// allocate and free
	__global__ void pairPointToBuffers(TemporalPathPoint* p,
		float_type* totalDistBuffer,
		VectorType* xTbuffer,
		ub32 pointsNumber,
		ub32 ticksNumber)
	{
		ub32 currentPathNumber = threadIdx.x + blockDim.x * blockIdx.x;

		if (currentPathNumber < pointsNumber)
		{
			p[currentPathNumber].totalDist = totalDistBuffer + currentPathNumber * ticksNumber;
			p[currentPathNumber].xT = xTbuffer + currentPathNumber * ticksNumber;
		}
	}

	// dt sampling
	__global__ void dtShiftKernel(float_type* pos_x, float_type* pos_y,
#if DIMS==3
		float_type* pos_z,
#endif
		const ub32* materialBuffer, const VectorType* wBuffer, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 temporalNum = threadNum % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / temporalSamplerGpuData.ticksNum;
		if (pathNum < totalPathsNum)
		{
			ub32 materialNum = materialBuffer[pathNum];

			if (materialNum > 0)
			{
				ub32 curretntTemporal;
				if (temporalNum < temporalSamplerGpuData.negativeIdxNum)
				{
					curretntTemporal = temporalSamplerGpuData.negativeIdxNum - 1 - temporalNum;
				}
				else
				{
					curretntTemporal = temporalNum;
				}

				curandState_t* state = statePool + threadNum;
				float_type D = temporalMaterialGpuData[materialNum].D;
				VectorType U = temporalMaterialGpuData[materialNum].U;
				float_type dt = temporalSamplerGpuData.dt[curretntTemporal];

				// shift temporal data points

				// MATLAB code to demonstrate:
				// 
				// D = 10;
				// dt = 2;
				// dr = randn(1e6, 1) * sqrt(2.0 * D * 3 * abs(dt));
				// w = randn(1e6, 3);
				// w = w . / sqrt(sum(w. ^ 2, 2));
				// X = dr.*w;
				// mean_r_sq = mean(sum(X.^ 2, 2))
				// 
				// mean_r_sq ~ 120, which is 6 * D * dt.

				float_type dr = randNormal(state) * sqrt(2.0 * DIMS * D * fabs(dt));

				VectorType w = wBuffer[threadNum];
				ub32 posIdx = curretntTemporal + pathNum * temporalSamplerGpuData.ticksNum;

				pos_x[posIdx] = fma(dr, w.x(), dt * U.x());
				pos_y[posIdx] = fma(dr, w.y(), dt * U.y());
#if DIMS==3
				pos_z[posIdx] = fma(dr, w.z(), dt * U.z());
#endif

				//printf("path: %d, material = %d, dt = %f ,randW = [%f %f %f], dr = %e, D = %e, U = [%f %f %f], dp = [%f %f %f] \n",
				//	pathNum, materialNum,
				//	dt, w.x(), w.y(), w.z(), dr, D, U.x(), U.y(), U.z(), pos_x[threadNum], pos_y[threadNum], pos_z[threadNum]);
			}
		}
	}

	__global__ void getMaterialNumTemporal(ub32* materialNum, TemporalPathPoint* pb,
		const ib32* pathsIdx, ub32 totalPathsNum, bool isForcedInsideBin)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathsNum)
		{
			ib32 currentPath = pathsIdx[pathNum];
			if (currentPath >= 0)
			{
				materialNum[pathNum] = pb[currentPath].xSampled.material;
			}
			else
			{
				materialNum[pathNum] = 0;
			}

			if (isForcedInsideBin)
			{
				pb[currentPath].randomIdx = randUniformInteger(statePool + pathNum, temporalSamplerGpuData.ticksNum);
				// printf("pathNum - %d, currentPath - %d, ticksNum - %d, randomIdx - %d \n", pathNum, currentPath, temporalSamplerGpuData.ticksNum, pb[currentPath].randomIdx);
			}
		}
	}

	__global__ void shiftFirstTemporalPoint(TemporalPathPoint* pb, float_type* pos_x, float_type* pos_y,
#if DIMS==3
		float_type* pos_z,
#endif
		const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathsNum)
		{
			ib32 currentPath = pathsIdx[pathNum];

			if (currentPath >= 0)
			{
				pos_x[pathNum * temporalSamplerGpuData.ticksNum] += pb[currentPath].xSampled.position.x();
				pos_y[pathNum * temporalSamplerGpuData.ticksNum] += pb[currentPath].xSampled.position.y();
#if DIMS==3
				pos_z[pathNum * temporalSamplerGpuData.ticksNum] += pb[currentPath].xSampled.position.z();
#endif

				if (temporalSamplerGpuData.negativeIdxNum > 0)
				{
					pos_x[pathNum * temporalSamplerGpuData.ticksNum + temporalSamplerGpuData.negativeIdxNum] += pb[currentPath].xSampled.position.x();
					pos_y[pathNum * temporalSamplerGpuData.ticksNum + temporalSamplerGpuData.negativeIdxNum] += pb[currentPath].xSampled.position.y();
#if DIMS==3
					pos_z[pathNum * temporalSamplerGpuData.ticksNum + temporalSamplerGpuData.negativeIdxNum] += pb[currentPath].xSampled.position.z();
#endif
				}
			}
		}
	}

	__global__ void shiftRandomSampledIndex(TemporalPathPoint* pb, float_type* pos_x, float_type* pos_y,
#if DIMS==3
		float_type* pos_z,
#endif
		const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		// ub32 temporalNum = threadNum % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / temporalSamplerGpuData.ticksNum;

		if (pathNum < totalPathsNum)
		{
			ib32 currentPath = pathsIdx[pathNum];
			if (currentPath >= 0)
			{
				ub32 temporalCenterNum = pb[currentPath].randomIdx;
				ub32 posIdx;
				if (temporalCenterNum < temporalSamplerGpuData.negativeIdxNum)
				{
					posIdx = temporalSamplerGpuData.negativeIdxNum - 1 - temporalCenterNum;
				}
				else
				{
					posIdx = temporalCenterNum;
				}

				VectorType shiftPoint = VectorType(
					pb[currentPath].xSampled.position.x() - pos_x[pathNum * temporalSamplerGpuData.ticksNum + posIdx],
					pb[currentPath].xSampled.position.y() - pos_y[pathNum * temporalSamplerGpuData.ticksNum + posIdx]

#if DIMS==3
					, pb[currentPath].xSampled.position.z() - pos_z[pathNum * temporalSamplerGpuData.ticksNum + posIdx]
#endif
				);

				pos_x[threadNum] += shiftPoint.x();
				pos_y[threadNum] += shiftPoint.y();
#if DIMS==3
				pos_z[threadNum] += shiftPoint.z();
#endif
			}
		}
	}

	__global__ void copyPointsFromBuffersTemporal(TemporalPathPoint* pb, const float_type* pos_x, const float_type* pos_y,
#if DIMS==3
		const float_type* pos_z,
#endif
		const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 temporalNum = threadNum % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / temporalSamplerGpuData.ticksNum;

		if (pathNum < totalPathsNum)
		{
			ib32 currentPath = pathsIdx[pathNum];
			if (currentPath >= 0)
			{
				ub32 posIdx;

				if (temporalNum < temporalSamplerGpuData.negativeIdxNum)
				{
					posIdx = temporalSamplerGpuData.negativeIdxNum - 1 - temporalNum + pathNum * temporalSamplerGpuData.ticksNum;
				}
				else
				{
					posIdx = threadNum;
				}

				VectorType p(pos_x[posIdx], pos_y[posIdx]
#if DIMS==3
					, pos_z[posIdx]
#endif
					);

#if DIMS==3
				//printf("**update pb** thread: %d, pathNum: %d, currentPath: %d, totalPathsNum: %d, temporalNum: %d, |pos| = %f, pb_sampled = [%f %f %f], pos = [%f %f %f], p = [%f %f %f], d = %f, D  = %f \n",
				//	threadNum, pathNum, currentPath, totalPathsNum, temporalNum, sqrt(pos_x[posIdx] * pos_x[posIdx] + pos_y[posIdx] * pos_y[posIdx] + pos_z[posIdx] * pos_z[posIdx]),
				//	pb[currentPath].xSampled.position.x(), pb[currentPath].xSampled.position.y(), pb[currentPath].xSampled.position.z(),
				//	pos_x[posIdx], pos_y[posIdx], pos_z[posIdx],
				//	p.x(), p.y(), p.z(),
				//	abs(p - pb[currentPath].xT[temporalNum]), pb[currentPath].totalDist[temporalNum]);
#else
				//printf("**update pb** thread: %d, pathNum: %d, currentPath: %d, totalPathsNum: %d, pb = [%f %f], p = [%f %f], d = %f, D  = %f \n",
				//	threadNum, pathNum, currentPath, totalPathsNum,
				//	pb[currentPath].xT[temporalNum].x(), pb[currentPath].xT[temporalNum].y(),
				//	p.x(), p.y(),
				//	abs(p - pb[currentPath].xT[temporalNum]), pb[currentPath].totalDist[temporalNum]);
#endif

				pb[currentPath].xT[temporalNum] = p;
				pb[currentPath].totalDist[temporalNum] = 0;
			}
		}
	}

	__global__ void copyPointsFromBuffersPrevTemporal(TemporalPathPoint* pb, TemporalPathPoint* pa, const float_type* pos_x, const float_type* pos_y,
#if DIMS==3
		const float_type* pos_z,
#endif
		const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 temporalNum = threadNum % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / temporalSamplerGpuData.ticksNum;

		if (pathNum < totalPathsNum)
		{
			ib32 currentPath = pathsIdx[pathNum];
			if (currentPath >= 0)
			{
				ub32 posIdx;

				if (temporalNum < temporalSamplerGpuData.negativeIdxNum)
				{
					posIdx = temporalSamplerGpuData.negativeIdxNum - 1 - temporalNum + pathNum * temporalSamplerGpuData.ticksNum;
				}
				else
				{
					posIdx = threadNum;
				}

				VectorType p(pos_x[posIdx],
					pos_y[posIdx]
#if DIMS==3
					, pos_z[posIdx]
#endif
					);

#if DIMS==3
				//printf("**update pb** thread: %d, pathNum: %d, currentPath: %d, totalPathsNum: %d, pa = [%f %f %f], pb = [%f %f %f], p = [%f %f %f], d = %f, D  = %f \n",
				//	threadNum, pathNum, currentPath, totalPathsNum,
				//	pa[currentPath].xT[temporalNum].x(), pa[currentPath].xT[temporalNum].y(), pa[currentPath].xT[temporalNum].z(),
				//	pb[currentPath].xT[temporalNum].x(), pb[currentPath].xT[temporalNum].y(), pb[currentPath].xT[temporalNum].z(),
				//	pos_x[threadNum], pos_y[threadNum], pos_z[threadNum],
				//	abs(p - pb[currentPath].xT[temporalNum]), pb[currentPath].totalDist[temporalNum]);
#else
				//printf("**update pb** thread: %d, pathNum: %d, currentPath: %d, totalPathsNum: %d, pa = [%f %f], pb = [%f %f], p = [%f %f], d = %f, D  = %f \n",
				//	threadNum, pathNum, currentPath, totalPathsNum,
				//	pa[currentPath].xT[temporalNum].x(), pa[currentPath].xT[temporalNum].y(),
				//	pb[currentPath].xT[temporalNum].x(), pb[currentPath].xT[temporalNum].y(),
				//	p.x(), p.y(),
				//	abs(p - pb[currentPath].xT[temporalNum]), pb[currentPath].totalDist[temporalNum]);
#endif

				pa[currentPath].xT[temporalNum] = pb[currentPath].xT[temporalNum];
				pb[currentPath].totalDist[temporalNum] += abs(p - pb[currentPath].xT[temporalNum]);
				pb[currentPath].xT[temporalNum] = p;
			}
		}
	}

	__global__ void randomizeDirectionsKernel(VectorType* wBuffer,
		const ib32* pathsIdx, ub32 totalPathsNum, ub32 Nt)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		// ub32 nt = threadNum % Nt;
		ub32 pathNum = threadNum / Nt;

		if (pathNum < totalPathsNum)
		{
			ib32 currentPath = pathsIdx[pathNum];
			if (currentPath >= 0)
			{
				curandState_t* state = statePool + threadNum;
				wBuffer[threadNum] = randomDirection(state);
			}
		}
	}

	__global__ void pathContriobutionKernel(ComplexType* pContrb, const TemporalPathPoint* pa, const bool* isValid, const ib32* pathsIdx, ub32 totalPathsNum, ub32 pL)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath;
			if (pL > 1)
			{
				currentPath = pathsIdx[pathNum];

			}
			else
			{
				currentPath = pathNum;
			}

			if (isValid[pathNum * temporalSamplerGpuData.ticksNum + temporalNum] == false)
			{
				pContrb[threadNum] = ComplexType(0., 0.);
			}
			else
			{
				float_type currentLambda = lambdaValues[wavelenghNum];

				float_type sinPhaseRandom, cosPhaseRandom;
				float_type phaseRandom = pa[currentPath].randomNum;
				sincospi(2.0 * phaseRandom, &sinPhaseRandom, &cosPhaseRandom);

				if (pL > 1)
				{
					float_type sinPhaseDistance, cosPhaseDistance;
					float_type phaseDistance = (float_type)((1.0) / currentLambda) * pa[currentPath].totalDist[temporalNum];
					sincospi(2.0 * phaseDistance, &sinPhaseDistance, &cosPhaseDistance);

					pContrb[threadNum] = ComplexType(cosPhaseDistance * cosPhaseRandom - sinPhaseDistance * sinPhaseRandom,
						cosPhaseDistance * sinPhaseRandom + cosPhaseRandom * sinPhaseDistance);
				}
				else
				{
					pContrb[threadNum] = ComplexType(cosPhaseRandom, sinPhaseRandom);
				}

				//printf("temporalNum - %d, total dist - %f , [%f %f %f] \n", temporalNum, pa[currentPath].totalDist[temporalNum],
				//	pa[currentPath].xT[temporalNum].x(), pa[currentPath].xT[temporalNum].y(), pa[currentPath].xT[temporalNum].z());
			}
		}
	}

	__global__ void copyOutsidePointIdx(ib32* negativePathOutsideIdx, ib32* positivePathOutsideIdx,
		const TemporalPathPoint* pa, const bool* isValidBuffer, const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 temporalNum = threadNum % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / temporalSamplerGpuData.ticksNum;

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath = pathsIdx[pathNum];
			if (isValidBuffer[threadNum] == false)
			{
				if (temporalNum <= pa[currentPath].randomIdx)
				{
					negativePathOutsideIdx[threadNum] = temporalNum;
				}
				else
				{
					positivePathOutsideIdx[threadNum] = temporalNum;
				}
			}

			//printf("path - %d, tmp - %d, neg - %d, pos - %d, sampled - [%f %f %f], xt - [%f, %f, %f], rIdx - %d, isValid - %d \n",
			//	pathNum, temporalNum, negativePathOutsideIdx[threadNum], positivePathOutsideIdx[threadNum], 
			//	pa[currentPath].xSampled.position.x(), pa[currentPath].xSampled.position.y(), pa[currentPath].xSampled.position.z(),
			//	pa[currentPath].xT[temporalNum].x(), pa[currentPath].xT[temporalNum].y(), pa[currentPath].xT[temporalNum].z(),
			//	pa[currentPath].randomIdx, isValidBuffer[threadNum]);
		}
	}

	__global__ void updateIsValidBuffer(bool* isValidBuffer, const ib32* negativePathMax, const ib32* positivePathMin, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 temporalNum = threadNum % temporalSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / temporalSamplerGpuData.ticksNum;

		if (pathNum < totalPathsNum)
		{
			if (negativePathMax[pathNum] != -1 && temporalNum <= negativePathMax[pathNum])
			{
				isValidBuffer[threadNum] = false;
			}

			if (positivePathMin[pathNum] != ~(1 << 31) && temporalNum >= positivePathMin[pathNum])
			{
				isValidBuffer[threadNum] = false;
			}

			//printf("path - %d, tmp - %d, neg - %d, pos - %d, isValid - %d \n",
			//	pathNum, temporalNum, negativePathMax[pathNum], positivePathMin[pathNum], isValidBuffer[threadNum]);
		}
	}
}

// ------------------------------------ Function Helpers ------------------------------------ //
template<bool isFirstPoint>
__host__ ErrorType TemporalSampler::dtshift_host(TemporalSamplerNS::TemporalPathPoint* sPa, TemporalSamplerNS::TemporalPathPoint* sPb, ib32* pathsIdx, ub32 totalPathsNum)
{
	ub32 totalThreads = totalPathsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	// Build temporal xyz buffers
	ub32* marerialNumBuffer = (ub32*)mediumPointsBuffer;
	float_type* buffer_x = ((float_type*)previousPointsBuffer);
	float_type* buffer_y = ((float_type*)previousPointsBuffer) + totalPathsNum * samplerSize;
#if DIMS==3
	float_type* buffer_z = ((float_type*)previousPointsBuffer) + 2 * totalPathsNum * samplerSize;
#endif

	// Copy data to buffers
	TemporalSamplerNS::getMaterialNumTemporal <<<blocksNum, threadsNum >>> (marerialNumBuffer, sPb,
		pathsIdx, totalPathsNum, _isForcedInsideBin);

	totalThreads = totalPathsNum * samplerSize;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	// randomize directions
	TemporalSamplerNS::randomizeDirectionsKernel <<<blocksNum, threadsNum >>> (pathsVectorBuffer,
		pathsIdx, totalPathsNum, samplerSize);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_randomizeDirectionsKernel;
	}

	// Get randomized shifts accrording to dt
	TemporalSamplerNS::dtShiftKernel <<<blocksNum, threadsNum >>> (buffer_x, buffer_y,
#if DIMS==3
		buffer_z,
#endif
		marerialNumBuffer, pathsVectorBuffer, totalPathsNum);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_dtShiftKernel;
	}

	// cumsum each dim

	// DEBUG
	/*float_type* buffer_x_before = (float_type*)malloc(sizeof(float_type) * totalPathsNum * samplerSize);
	float_type* buffer_y_before = (float_type*)malloc(sizeof(float_type) * totalPathsNum * samplerSize);
	float_type* buffer_z_before = (float_type*)malloc(sizeof(float_type) * totalPathsNum * samplerSize);

	float_type* buffer_x_after = (float_type*)malloc(sizeof(float_type) * totalPathsNum * samplerSize);
	float_type* buffer_y_after = (float_type*)malloc(sizeof(float_type) * totalPathsNum * samplerSize);
	float_type* buffer_z_after = (float_type*)malloc(sizeof(float_type) * totalPathsNum * samplerSize);

	ub32* temporalPathsNum_cpu = (ub32*)malloc(sizeof(ub32) * totalPathsNum * samplerSize);
	TemporalSamplerNS::TemporalPathPoint* sPb_cpu = (TemporalSamplerNS::TemporalPathPoint*)malloc(sizeof(TemporalSamplerNS::TemporalPathPoint) * totalPathsNum);
	ub32* marerialNumBuffer_cpu = (ub32*)malloc(sizeof(ub32) * totalPathsNum);

	cudaMemcpy(buffer_x_before, buffer_x, sizeof(float_type) * totalPathsNum * samplerSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(buffer_y_before, buffer_y, sizeof(float_type) * totalPathsNum * samplerSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(buffer_z_before, buffer_z, sizeof(float_type) * totalPathsNum * samplerSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaMemcpy(temporalPathsNum_cpu, temporalPathsNum, sizeof(ub32) * totalPathsNum * samplerSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(sPb_cpu, sPb, sizeof(TemporalSamplerNS::TemporalPathPoint) * totalPathsNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(marerialNumBuffer_cpu, marerialNumBuffer, sizeof(ub32) * totalPathsNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);*/
	// END DEBUG

	totalThreads = totalPathsNum;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::shiftFirstTemporalPoint <<<blocksNum, threadsNum >>> (sPb, buffer_x, buffer_y,
#if DIMS==3
		buffer_z,
#endif
		pathsIdx, totalPathsNum);


	thrust::inclusive_scan_by_key(thrust::device, temporalPathsNum, temporalPathsNum + totalPathsNum * samplerSize, buffer_x, buffer_x);
	thrust::inclusive_scan_by_key(thrust::device, temporalPathsNum, temporalPathsNum + totalPathsNum * samplerSize, buffer_y, buffer_y);
#if DIMS==3
	thrust::inclusive_scan_by_key(thrust::device, temporalPathsNum, temporalPathsNum + totalPathsNum * samplerSize, buffer_z, buffer_z);
#endif
	cudaDeviceSynchronize();

	if (_isForcedInsideBin)
	{
		totalThreads = totalPathsNum * samplerSize;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		// shift path to random sampled index
		TemporalSamplerNS::shiftRandomSampledIndex <<<blocksNum, threadsNum >>> (sPb, buffer_x, buffer_y,
#if DIMS==3
			buffer_z,
#endif
			pathsIdx, totalPathsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}
	}

	// DEBUG
	/*cudaMemcpy(buffer_x_after, buffer_x, sizeof(float_type) * totalPathsNum * samplerSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(buffer_y_after, buffer_y, sizeof(float_type) * totalPathsNum * samplerSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(buffer_z_after, buffer_z, sizeof(float_type) * totalPathsNum * samplerSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	for (int ii = 0; ii < totalPathsNum * samplerSize; ii++)
	{
		printf("temporalPathsNum: %d, randomIdx: %d, materialNum: %d - [%f %f %f] - [%f %f %f] \n", temporalPathsNum_cpu[ii],
			sPb_cpu[ii / samplerSize].randomIdx, marerialNumBuffer_cpu[ii / samplerSize],
			buffer_x_before[ii], buffer_y_before[ii], buffer_z_before[ii],
			buffer_x_after[ii], buffer_y_after[ii], buffer_z_after[ii]);
	}

	std::free(buffer_x_before);
	std::free(buffer_y_before);
	std::free(buffer_z_before);
	std::free(buffer_x_after);
	std::free(buffer_y_after);
	std::free(buffer_z_after);
	std::free(temporalPathsNum_cpu);
	std::free(sPb_cpu);
	std::free(marerialNumBuffer_cpu);*/
	// END DEBUG

	totalThreads = totalPathsNum * samplerSize;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	if (isFirstPoint)
	{
		TemporalSamplerNS::copyPointsFromBuffersTemporal <<<blocksNum, threadsNum >>> (sPb, buffer_x, buffer_y,
#if DIMS==3
			buffer_z,
#endif
			pathsIdx, totalPathsNum);
	}
	else
	{
		TemporalSamplerNS::copyPointsFromBuffersPrevTemporal <<<blocksNum, threadsNum >>> (sPb, sPa, buffer_x, buffer_y,
#if DIMS==3
			buffer_z,
#endif
			pathsIdx, totalPathsNum);
	}

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyPointsFromBuffersTemporal;
	}

	return ErrorType::NO_ERROR;
}


// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
TemporalSampler::TemporalSampler(ErrorType* err, Source* illuminationsHandle, Source* viewsHandle, const Simulation* simulationHandler, const Medium* mediumHandler,
	const Scattering* scatteringHandler, const float_type* ticks, ub32 ticksNum, bool isTemporalAmplitude, bool isForcedInsideBin):
	Sampler(illuminationsHandle, viewsHandle, simulationHandler, mediumHandler, scatteringHandler, ticksNum),
	_isTemporalAmplitude(isTemporalAmplitude), _isForcedInsideBin(isForcedInsideBin)
{
	MEMORY_CHECK("temporal sampler allocation begin");
	allocationsCount = 0;

	// copy ticks to CPU
	dt_cpu = (float_type*)malloc(sizeof(float_type) * ticksNum);

	if(dt_cpu == NULL)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	// Compute mean time
	mean_t = 0;
	for (ub32 tt = 0; tt < ticksNum; tt++)
	{
		mean_t += ticks[tt];
	}
	mean_t = mean_t / ticksNum;

	// count negative / positive idx num
	negativeIdxNum = 0;
	positiveIdxNum = 0;
	for (ub32 tt = 0; tt < ticksNum; tt++)
	{
		if (ticks[tt] - mean_t < 0)
		{
			negativeIdxNum++;
		}
		else
		{
			positiveIdxNum++;
		}
	}

	// allocate
	positiveIdxList = (ub32*)malloc(sizeof(ub32) * positiveIdxNum);
	if (positiveIdxList == NULL)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	negativeIdxList = (ub32*)malloc(sizeof(ub32) * negativeIdxNum);
	if (negativeIdxList == NULL)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	ub32* isUsed = (ub32*)malloc(sizeof(ub32) * ticksNum);
	if (isUsed == NULL)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	// init isUsed to zeros
	memset(isUsed, 0, sizeof(ub32) * ticksNum);

	// sort according to smallest td - negative index
	for (ub32 tt_1 = 0; tt_1 < negativeIdxNum; tt_1++)
	{
		bool isFirstIdx = true;
		float_type smallestTd;
		ub32 tdIdx;

		for (ub32 tt_2 = 0; tt_2 < ticksNum; tt_2++)
		{
			if ((isUsed[tt_2] == 0) && (ticks[tt_2] - mean_t < 0))
			{
				if (isFirstIdx)
				{
					isFirstIdx = false;
					smallestTd = mean_t - ticks[tt_2];
					tdIdx = tt_2;
				}
				else if (mean_t - ticks[tt_2] < smallestTd)
				{
					smallestTd = mean_t - ticks[tt_2];
					tdIdx = tt_2;
				}
			}
		}
		isUsed[tdIdx] = 1;
		negativeIdxList[tt_1] = tdIdx;
	}

	// sort according to smallest td - positive index
	for (ub32 tt_1 = 0; tt_1 < positiveIdxNum; tt_1++)
	{
		bool isFirstIdx = true;
		float_type smallestTd;
		ub32 tdIdx;

		for (ub32 tt_2 = 0; tt_2 < ticksNum; tt_2++)
		{
			if ((isUsed[tt_2] == 0) && (ticks[tt_2] - mean_t >= 0))
			{
				if (isFirstIdx)
				{
					isFirstIdx = false;
					smallestTd = ticks[tt_2] - mean_t;
					tdIdx = tt_2;
				}
				else if (ticks[tt_2] - mean_t < smallestTd)
				{
					smallestTd = ticks[tt_2] - mean_t;
					tdIdx = tt_2;
				}
			}
		}
		isUsed[tdIdx] = 1;
		positiveIdxList[tt_1] = tdIdx;
	}

	std::free(isUsed);

	if (cudaMalloc(&mediumPointsBuffer, sizeof(MediumPoint) * batchSize * wavelenghSize * samplerSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	if (cudaMalloc(&mediumPointsBuffer_2, sizeof(MediumPoint) * batchSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;
	
	if (cudaMalloc(&previousPointsBuffer, sizeof(VectorType) * batchSize * wavelenghSize * samplerSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	if (cudaMalloc(&pathsVectorBuffer, sizeof(VectorType) * batchSize * wavelenghSize * samplerSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	if (cudaMalloc(&isValidBuffer, sizeof(bool) * batchSize * samplerSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	// get gpu data
	TemporalSamplerNS::TemporalSamplerGpuDataStruct gpuDataCpu;
	gpuDataCpu.ticksNum = ticksNum;
	gpuDataCpu.meanTick = mean_t;
	gpuDataCpu.batchSize = batchSize;
	gpuDataCpu.negativeIdxNum = negativeIdxNum;
	gpuDataCpu.positiveIdxNum = positiveIdxNum;

	if (cudaMalloc(&gpuDataCpu.dt, sizeof(float_type) * ticksNum) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	// Compute the dt for each pixel
	ub32 currentNegativeIdxNum = 0;
	ub32 currentPositiveIdxNum = 0;
	ib32 tickIdx;
	ib32 prevTickIdx;

	for (ub32 tIdx = 0; tIdx < samplerSize; tIdx++)
	{
		if (tIdx < negativeIdxNum)
		{
			tickIdx = (ib32)negativeIdxList[currentNegativeIdxNum];

			if (currentNegativeIdxNum == 0)
			{
				prevTickIdx = -1;
			}

			currentNegativeIdxNum++;
		}
		else
		{
			tickIdx = (ib32)positiveIdxList[currentPositiveIdxNum];

			if (currentPositiveIdxNum == 0)
			{
				prevTickIdx = -1;
			}

			currentPositiveIdxNum++;
		}

		if (prevTickIdx < 0)
		{
			dt_cpu[tIdx] = ticks[tickIdx] - mean_t;
		}
		else
		{
			dt_cpu[tIdx] = ticks[tickIdx] - ticks[prevTickIdx];
		}

		prevTickIdx = tickIdx;
	}


	if (cudaMemcpy(gpuDataCpu.dt, dt_cpu, sizeof(float_type) * ticksNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	if (cudaMalloc(&temporalPathsNum, sizeof(ub32) * batchSize * samplerSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	if (cudaMalloc(&pathsNum, sizeof(ub32) * batchSize * samplerSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	if (cudaMemcpyToSymbol(TemporalSamplerNS::temporalSamplerGpuData, &gpuDataCpu, sizeof(TemporalSamplerNS::TemporalSamplerGpuDataStruct), 0, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	ub32 totalThreadsNum = batchSize * samplerSize;
	ub32 threadsNum = totalThreadsNum < THREADS_NUM ? totalThreadsNum : THREADS_NUM;
	ub32 blocksNum = (totalThreadsNum - 1) / THREADS_NUM + 1;

	// copy points data into buffers
	TemporalSamplerNS::setTemporalPathsNum <<<blocksNum, threadsNum >>> (temporalPathsNum, pathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::KERNEL_ERROR_Sampler_setTemporalPathsNum;
		return;
	}

	*err = ErrorType::NO_ERROR;

	MEMORY_CHECK("temporal sampler allocation end");
}

TemporalSampler::~TemporalSampler()
{
	MEMORY_CHECK("temporal sampler free begin");

	TemporalSamplerNS::TemporalSamplerGpuDataStruct gpuDataCpu;

	if (allocationsCount >= 9)
	{
		cudaMemcpyFromSymbol(&gpuDataCpu, TemporalSamplerNS::temporalSamplerGpuData, sizeof(TemporalSamplerNS::TemporalSamplerGpuDataStruct), 0, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}

	switch (allocationsCount)
	{
	case 11:
		cudaFree(pathsNum);
	case 10:
		cudaFree(temporalPathsNum);
	case 9:
		cudaFree(gpuDataCpu.dt);
	case 8:
		cudaFree(isValidBuffer);
	case 7:
		cudaFree(pathsVectorBuffer);
	case 6:
		cudaFree(previousPointsBuffer);
	case 5:
		cudaFree(mediumPointsBuffer_2);
	case 4:
		cudaFree(mediumPointsBuffer);
	case 3:
		std::free(negativeIdxList);
	case 2:
		std::free(positiveIdxList);
	case 1:
		std::free(dt_cpu);
	default:
		break;
	}

	MEMORY_CHECK("temporal sampler free end");
}

// ------------------------------------ Class Function Implementations ------------------------------------ //
__host__ ErrorType TemporalSampler::setMaterial(float_type D, VectorType U, ub32 materialNum)
{
	TemporalSamplerNS::TemporalMaterialDataStructre currentMatrialData{ D, U };

	cudaMemcpyToSymbol(TemporalSamplerNS::temporalMaterialGpuData, &currentMatrialData,
		sizeof(TemporalSamplerNS::TemporalMaterialDataStructre),
		sizeof(TemporalSamplerNS::TemporalMaterialDataStructre) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ void TemporalSampler::freePoints(Point* p, ub32 pointsNumber)
{
	TemporalSamplerNS::TemporalPathPoint sPa_cpu, * sPa_gpu;
	sPa_gpu = (TemporalSamplerNS::TemporalPathPoint*)p;
	cudaMemcpy(&sPa_cpu, sPa_gpu, sizeof(TemporalSamplerNS::TemporalPathPoint), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaFree(sPa_cpu.totalDist);
	cudaFree(sPa_cpu.xT);
	cudaFree(sPa_gpu);
}

__host__ ErrorType TemporalSampler::allocatePoints(Point** p, ub32 pointsNumber)
{
	TemporalSamplerNS::TemporalPathPoint* sPa_gpu;
	float_type* totalDistBuffer;
	VectorType* xTbuffer;

	// allocate gpu points
	if (cudaMalloc(&sPa_gpu, sizeof(TemporalSamplerNS::TemporalPathPoint) * pointsNumber) != cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	// allocate gpu bufferes
	if (cudaMalloc(&totalDistBuffer, sizeof(float_type) * pointsNumber * samplerSize) != cudaSuccess)
	{
		freePoints((Point*)sPa_gpu, pointsNumber);
		return ErrorType::ALLOCATION_ERROR;
	}

	if (cudaMalloc(&xTbuffer, sizeof(VectorType) * pointsNumber * samplerSize) != cudaSuccess)
	{
		freePoints((Point*)sPa_gpu, pointsNumber);
		return ErrorType::ALLOCATION_ERROR;
	}

	ub32 totalThreads = pointsNumber;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::pairPointToBuffers <<<blocksNum, threadsNum >>> (sPa_gpu,
		totalDistBuffer,
		xTbuffer,
		pointsNumber,
		samplerSize);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	*p = (Point*)sPa_gpu;

	return ErrorType::NO_ERROR;
}

// pathsIdx: if negative value -> point is missed the medium.
__host__ ErrorType TemporalSampler::sampleFirst(Point* pa, Point* pb, ib32* pathsIdx, ub32 totalPathsNum)
{
	// Let the illumination source to sample the points

	// sample new point
	// previousPointsBuffer: new sampled source point
	// pathsVectorBuffer   : new sampled source direction
	// mediumPointsBuffer  : new sampled point in medium
	ErrorType err = illuminationsHandle->sampleFirstPoint(previousPointsBuffer,
		pathsVectorBuffer,
		mediumPointsBuffer,
		totalPathsNum);

	// Attach the resulted samples to the points structre

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	ub32 threadsNum = totalPathsNum < THREADS_NUM ? totalPathsNum : THREADS_NUM;
	ub32 blocksNum = (totalPathsNum - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sPa = (TemporalSamplerNS::TemporalPathPoint*)pa;
	TemporalSamplerNS::TemporalPathPoint* sPb = (TemporalSamplerNS::TemporalPathPoint*)pb;

	TemporalSamplerNS::copyBuffersToPoints <<<blocksNum, threadsNum >>> (
		sPa, previousPointsBuffer, sPb, mediumPointsBuffer, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyBuffersToPoints;
	}

	return dtshift_host<true>(sPa, sPb, pathsIdx, totalPathsNum);
}

__host__ ErrorType TemporalSampler::sampleSecond(Point* pa, Point* pb, ib32* pathsIdx, ub32 totalPathsNum)
{
	ub32 threadsNum = totalPathsNum < THREADS_NUM ? totalPathsNum : THREADS_NUM;
	ub32 blocksNum = (totalPathsNum - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sPa = (TemporalSamplerNS::TemporalPathPoint*)pa;
	TemporalSamplerNS::TemporalPathPoint* sPb = (TemporalSamplerNS::TemporalPathPoint*)pb;

	// copy points data into buffers
	TemporalSamplerNS::copyPointsToBuffers <<<blocksNum, threadsNum >>> (
		sPa, previousPointsBuffer, sPb, mediumPointsBuffer, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffers;
	}

	// sample new point
	// previousPointsBuffer: sampled source point
	// mediumPointsBuffer  : sampled point in medium
	// mediumPointsBuffer_2: new sampled second point in medium
	ErrorType err = illuminationsHandle->sampleSecondPoint(previousPointsBuffer,
		mediumPointsBuffer,
		mediumPointsBuffer_2,
		totalPathsNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// copy back buffers to points
	TemporalSamplerNS::copyBuffersToPoints <<<blocksNum, threadsNum >>> (
		sPa, sPb, mediumPointsBuffer_2, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyBuffersToPoints;
	}

	return dtshift_host<false>(sPa, sPb, pathsIdx, totalPathsNum);
}

__host__ ErrorType TemporalSampler::sampleNext(Point* pa, Point* pb, ib32* pathsIdx, ub32 totalPathsNum, ub32 pL)
{
	ub32 threadsNum = totalPathsNum < THREADS_NUM ? totalPathsNum : THREADS_NUM;
	ub32 blocksNum = (totalPathsNum - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sPa = (TemporalSamplerNS::TemporalPathPoint*)pa;
	TemporalSamplerNS::TemporalPathPoint* sPb = (TemporalSamplerNS::TemporalPathPoint*)pb;

	// copy points data into buffers
	TemporalSamplerNS::copyPointsToBuffers <<<blocksNum, threadsNum >>> (
		sPa, previousPointsBuffer, sPb, mediumPointsBuffer, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffers;
	}

	// sample new direction
	// previousPointsBuffer: previous sampled point
	// mediumPointsBuffer  : current sampled point
	// pathsVectorBuffer   : new sampled direction
	ErrorType err = scatteringHandler->newDirection(pathsVectorBuffer,
		previousPointsBuffer,
		mediumPointsBuffer,
		totalPathsNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// sample new point
	// mediumPointsBuffer_2: sampled new point
	// mediumPointsBuffer  : current sampled point
	// pathsVectorBuffer   : new sampled direction
	err = mediumHandler->sample(mediumPointsBuffer_2,
		mediumPointsBuffer, pathsVectorBuffer, totalPathsNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// copy back buffers to points
	TemporalSamplerNS::copyBuffersToPoints <<<blocksNum, threadsNum >>> (
		sPa, sPb, mediumPointsBuffer_2, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyBuffersToPoints;
	}

	return dtshift_host<false>(sPa, sPb, pathsIdx, totalPathsNum);
}

__host__ ErrorType TemporalSampler::pathSingleProbability(float_type* probabilityRes, const Point* p1, ub32 totalPathsNum)
{
	ub32 totalThreads = totalPathsNum * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sP1 = (TemporalSamplerNS::TemporalPathPoint*)p1;

	// copy point p1 to buffers

	TemporalSamplerNS::copyPointsToBuffers <<< blocksNum, threadsNum >>> (sP1, mediumPointsBuffer, wavelenghSize, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffers;
	}
	
	// compute the probability regarding to the illumination source
	return illuminationsHandle->firstPointProbability(probabilityRes,
		mediumPointsBuffer,
		totalPathsNum);
}

// probabilityRes are the probabilities computed for the single scattering case
__host__ ErrorType TemporalSampler::pathMultipleProbability(float_type* probabilityRes, const Point* p1, const Point* p2, const ib32* pathsIdx, ub32 totalPathsNum)
{
	ub32 totalThreads = totalPathsNum * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sP1 = (TemporalSamplerNS::TemporalPathPoint*)p1;
	TemporalSamplerNS::TemporalPathPoint* sP2 = (TemporalSamplerNS::TemporalPathPoint*)p2;

	// copy points data into buffers
	TemporalSamplerNS::copyPointsToBuffers <<<blocksNum, threadsNum >>> (
		sP1, mediumPointsBuffer, sP2, previousPointsBuffer, pathsIdx, wavelenghSize, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffers;
	}
	
	return illuminationsHandle->secondPointProbability(probabilityRes,
		mediumPointsBuffer,
		previousPointsBuffer,
		totalPathsNum);
}

__host__ ErrorType TemporalSampler::twoPointThroughput(ComplexType* gRes, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cType, ub32 pL)
{
	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sP1 = (TemporalSamplerNS::TemporalPathPoint*)p1;

	// copy points data into buffers
	TemporalSamplerNS::copyPointsToBuffersThroughput <<<blocksNum, threadsNum >>> (
		sP1, mediumPointsBuffer, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffersThroughput;
	}

	Source* currentSource = ((cType == ConnectionType::ConnectionTypeIllumination) ? illuminationsHandle : viewsHandle);

	return currentSource->throughputFunction(gRes, mediumPointsBuffer, totalThreads);
}

__host__ ErrorType TemporalSampler::threePointThroughput(ComplexType* fRes, const Point* p1, const Point* p2, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cType, ub32 pL)
{
	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sP1 = (TemporalSamplerNS::TemporalPathPoint*)p1;
	TemporalSamplerNS::TemporalPathPoint* sP2 = (TemporalSamplerNS::TemporalPathPoint*)p2;

	// copy points data into buffers
	if (_isTemporalAmplitude)
	{
		TemporalSamplerNS::copyPointsToBuffersThreePoints<true> <<<blocksNum, threadsNum >>> (
			sP1, mediumPointsBuffer, sP2, previousPointsBuffer, pathsIdx, totalPathsNum);
	}
	else
	{
		TemporalSamplerNS::copyPointsToBuffersThreePoints<false> <<<blocksNum, threadsNum >>> (
			sP1, mediumPointsBuffer, sP2, previousPointsBuffer, pathsIdx, totalPathsNum);
	}

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffersThreePoints;
	}

	Source* currentSource = ((cType == ConnectionType::ConnectionTypeIllumination) ? illuminationsHandle : viewsHandle);

	ErrorType err = currentSource->threePointFunction(fRes, mediumPointsBuffer, previousPointsBuffer, totalThreads);
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType TemporalSampler::threePointThroughputSingle(ComplexType* fsRes, const Point* p1, ub32 totalPathsNum, ConnectionType cType)
{
	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sP1 = (TemporalSamplerNS::TemporalPathPoint*)p1;

	if (!isScalarFs())
	{
		// copy points data into buffers
		if (_isTemporalAmplitude)
		{
			TemporalSamplerNS::copyPointsToBuffersThreePoints<true> <<<blocksNum, threadsNum >>> (sP1, mediumPointsBuffer, totalPathsNum);
		}
		else
		{
			TemporalSamplerNS::copyPointsToBuffersThreePoints<false> <<<blocksNum, threadsNum >>> (sP1, mediumPointsBuffer, totalPathsNum);
		}


		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffersThreePoints;
		}
	}
	
	ErrorType err = illuminationsHandle->threePointFunctionSingle(fsRes, mediumPointsBuffer, viewsHandle, totalThreads);
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	return ErrorType::NO_ERROR;
}

ErrorType TemporalSampler::pathContribution(ComplexType* pContrb, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cn, ub32 pL)
{
	ErrorType err;
	ub32 totalThreads = totalPathsNum * samplerSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::TemporalPathPoint* sP1 = (TemporalSamplerNS::TemporalPathPoint*)p1;

	// Build buffers
	
	// negative index - points which are before the reference sapmled point.
	// We search for the highest index where the path is outside.
	// init to -1, and search for maximal index
	ib32* negativePathOutsideIdx = (ib32*)pathsVectorBuffer;

	// positive index - points which are after the reference sapmled point.
	// We search for the smallest index where the path is outside.
	// init to inf, and search for minimal index
	ib32* positivePathOutsideIdx = ((ib32*)pathsVectorBuffer) + totalPathsNum * samplerSize;

	ib32* negativePathMax = (ib32*)mediumPointsBuffer_2;
	ib32* positivePathMin = ((ib32*)mediumPointsBuffer_2) + totalPathsNum;
	ub32* outKeys         = ((ub32*)mediumPointsBuffer_2) + 2 * totalPathsNum;

	if (_isForcedInsideBin)
	{
		// Copy all sampled points
		TemporalSamplerNS::copyPointsToBuffersPathContribution <<<blocksNum, threadsNum >>> (sP1, mediumPointsBuffer,
			previousPointsBuffer, negativePathOutsideIdx, positivePathOutsideIdx, pathsIdx, totalPathsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}

		err = mediumHandler->isSameBin(isValidBuffer, mediumPointsBuffer, previousPointsBuffer, totalPathsNum, samplerSize);
		if (err != ErrorType::NO_ERROR)
		{
			return err;
		}

		// Mark all idx for points which are outside the bin

		totalThreads = totalPathsNum * samplerSize;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		TemporalSamplerNS::copyOutsidePointIdx <<<blocksNum, threadsNum >>> (negativePathOutsideIdx, positivePathOutsideIdx,
			sP1, isValidBuffer,pathsIdx, totalPathsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}

		thrust::equal_to<ub32> equalOp;
		thrust::maximum<ib32> maxBinaryOp;
		thrust::minimum<ib32> minBinaryOp;

		// Compute the maximal and minimal values for outside points
		thrust::reduce_by_key(thrust::device, pathsNum, pathsNum + totalPathsNum * samplerSize, negativePathOutsideIdx, outKeys, negativePathMax, equalOp, maxBinaryOp);
		thrust::reduce_by_key(thrust::device, pathsNum, pathsNum + totalPathsNum * samplerSize, positivePathOutsideIdx, outKeys, positivePathMin, equalOp, minBinaryOp);

		totalThreads = totalPathsNum * samplerSize;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		// Copy back to is valid buffer
		TemporalSamplerNS::updateIsValidBuffer <<<blocksNum, threadsNum >>> (isValidBuffer, negativePathMax, positivePathMin, totalPathsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}
	}
	else
	{
		TemporalSamplerNS::initValidBuffer <<<blocksNum, threadsNum >>> (isValidBuffer, totalThreads);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}
	}
	
	totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerNS::pathContriobutionKernel <<<blocksNum, threadsNum >>> (pContrb, sP1, isValidBuffer, pathsIdx, totalPathsNum, pL);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}
