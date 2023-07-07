#pragma once

#include "../../Interface/SamplerInterface.cuh"
#include "../../MatrixOperations.cuh"

// Let P be the batched paths number
// Let Nt be the time pixels
// Let Nw be the wavelengh pixels
// The convention is first dim is wavelengh, other dim is time

// ------------------------------------ GPU Constants ------------------------------------ //
namespace TemporalSamplerCorrelationNS
{
	// Temporal point structre
	typedef struct
	{
		ub32 ticksNum;
		ub32 batchSize;
		float_type* ticks;  // size of Nt
	} TemporalCorrelationSamplerGpuDataStruct;

	__constant__ TemporalCorrelationSamplerGpuDataStruct temporalCorrelationSamplerGpuData;

	// Temporal material structre
	typedef struct {
		float_type D;
		VectorType U;
	} TemporalCorrelationMaterialDataStructre;

	__constant__ TemporalCorrelationMaterialDataStructre temporalCorrelationMaterialGpuData[MATERIAL_NUM];

}

// ------------------------------------ Data Structures ------------------------------------ //
namespace TemporalSamplerCorrelationNS
{
	struct TemporalCorrelationPathPoint
	{
		MediumPoint xSampled;
		float_type momentTransfer;
		float_type totalDist;
	};
}

// ------------------------------------ Temporal Sampler Class ------------------------------------ //
class TemporalCorrelationSampler : public Sampler
{
public:
	// constructor
	// HOST FUNCION //
	// Init the functions for generating temporal tracer
	// ticks is CPU pointer in size of ticksNum
	TemporalCorrelationSampler(ErrorType* err, Source* u1IlluminationsHandle, Source* u2IlluminationsHandle,
		Source* u1ViewsHandle, Source* u2ViewsHandle, const Simulation* simulationHandler, const Medium* mediumHandler,
		const Scattering* scatteringHandler, const float_type* ticks, ub32 ticksNum);

	// HOST FUNCION //
	// Free allocated memory for tracer
	~TemporalCorrelationSampler();

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
	virtual ErrorType temporalCorrelationThroughput(ComplexType* tRes, const Point* p1, const Point* p2, ib32* pathsIdx, ub32 totalPathsNum, ConnectionTypeCorrelation ccType, ub32 pL);
	virtual ErrorType temporalCorrelationThroughputSingle(ComplexType* tsRes, const Point* p1, ub32 totalPathsNum);
	virtual bool isCorrelation() { return true; };
	virtual ErrorType pathContribution(ComplexType* pContrb, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cn, ub32 pL);

	// DEBUG
	virtual void printAllPoints(Point* p, ub32 pointsNumber) {}
	// END DEBUG

private:
    ub32 allocationsCount;
	float_type* ticks_cpu;             // size of Nt

    // sampling buffers
	MediumPoint* mediumPointsBuffer;   // size of P * Nw * Nt
	MediumPoint* mediumPointsBuffer_2; // size of P * Nw * Nt
	VectorType* previousPointsBuffer;  // size of P * Nw * Nt
	VectorType* pathsVectorBuffer;     // size of P * Nw * Nt

	void* tmpBuffer;                   // size of P * max(Nl,Nv) * Nw * Nt * sizeof(ComplexType)

	// Source handles
	Source* u1IlluminationsHandle;
	Source* u2IlluminationsHandle;
	Source* u1ViewsHandle;
	Source* u2ViewsHandle;

	// private functions
	ErrorType initInnerPhase();

	// Block default constructor
	TemporalCorrelationSampler();
};

// ------------------------------------ Kernels ------------------------------------ //
namespace TemporalSamplerCorrelationNS
{
	// COPY TO / FROM BUFFERS
	__global__ void copyBuffersToPoints(TemporalCorrelationPathPoint* pa, const VectorType* pa_buffer,
		TemporalCorrelationPathPoint* pb, const MediumPoint* pb_buffer, ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathsNum)
		{
			ub32 currentPathIdx = pathsIdx[pathNum];
			pa[currentPathIdx].xSampled.position = pa_buffer[pathNum];
			pa[currentPathIdx].xSampled.material = pb[currentPathIdx].xSampled.material;

			pb[currentPathIdx].xSampled = pb_buffer[pathNum];
			pb[currentPathIdx].momentTransfer = (float_type)0.0;
			pb[currentPathIdx].totalDist = (float_type)0.0;

			pa[currentPathIdx].momentTransfer = (float_type)0.0;
			pa[currentPathIdx].totalDist = (float_type)0.0;

			if (pb_buffer[pathNum].material == 0)
			{
				pathsIdx[pathNum] = -1;
			}

			//printf("copyBuffersToPoints single, pathNum: %d, currentPathIdx: %d, pa = [%f %f %f], pb = [%f %f %f] \n",
			//	pathNum, currentPathIdx, pa[currentPathIdx].xSampled.position.x(), pa[currentPathIdx].xSampled.position.y(), pa[currentPathIdx].xSampled.position.z(),
			//	pb[currentPathIdx].xSampled.position.x(), pb[currentPathIdx].xSampled.position.y(), pb[currentPathIdx].xSampled.position.z());
		}
	}

	// Inner path - sampling from the third scatterer
	template <bool isInnerPath>
	__global__ void copyBuffersToPoints_momentumTransfer(TemporalCorrelationPathPoint* pa, TemporalCorrelationPathPoint* pb, const MediumPoint* pb_buffer,
		ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathsNum)
		{
			ub32 currentPathIdx = pathsIdx[pathNum];

			if (isInnerPath)
			{
				// in inner path, if we don't change the phase function, we have to accumilate the mumentom of the inner path
				pa[currentPathIdx].momentTransfer = pb[currentPathIdx].momentTransfer;
				VectorType momoent = (normalize(pb[currentPathIdx].xSampled.position - pa[currentPathIdx].xSampled.position) -
					normalize(pb_buffer[pathNum].position - pb[currentPathIdx].xSampled.position));
				pb[currentPathIdx].momentTransfer += (momoent * momoent) * temporalCorrelationMaterialGpuData[pb[currentPathIdx].xSampled.material].D;
			}

			pa[currentPathIdx].totalDist = pb[currentPathIdx].totalDist;
			pb[currentPathIdx].totalDist += abs(pb[currentPathIdx].xSampled.position - pb_buffer[pathNum].position);
			VectorType aa = pa[currentPathIdx].xSampled.position;
			pa[currentPathIdx].xSampled = pb[currentPathIdx].xSampled;
			pb[currentPathIdx].xSampled = pb_buffer[pathNum];

			if (pb_buffer[pathNum].material == 0)
			{
				pathsIdx[pathNum] = -1;
			}

			//printf("copyBuffersToPoints multiple, pathNum: %d, currentPathIdx: %d\n [%f %f %f] -> [%f %f %f] -> [%f %f %f], mt_pa = %e, mt_pb = %e \n",
			//	pathNum, currentPathIdx, aa.x(), aa.y(), aa.z(),
			//	pa[currentPathIdx].xSampled.position.x(), pa[currentPathIdx].xSampled.position.y(), pa[currentPathIdx].xSampled.position.z(),
			//	pb[currentPathIdx].xSampled.position.x(), pb[currentPathIdx].xSampled.position.y(), pb[currentPathIdx].xSampled.position.z(), pa[currentPathIdx].momentTransfer, pb[currentPathIdx].momentTransfer);
		}
	}
	__global__ void copyPointsToBuffers(const TemporalCorrelationPathPoint* pa, VectorType* pa_buffer,
		const TemporalCorrelationPathPoint* pb, MediumPoint* pb_buffer,
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

	__global__ void copyPointsToBuffers(const TemporalCorrelationPathPoint* pa, MediumPoint* pa_buffer,
		const TemporalCorrelationPathPoint* pb, VectorType* pb_buffer,
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
	
	__global__ void copyPointsToBuffers(const TemporalCorrelationPathPoint* pb, MediumPoint* pb_buffer, ub32 wavelenghNum, ub32 totalPathsNum)
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
	
	__global__ void copyPointsToBuffersThroughput(const TemporalCorrelationPathPoint* pb, MediumPoint* pb_buffer,
		const ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cn)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalCorrelationSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalCorrelationSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath;

			currentPath = pathsIdx[pathNum];
			
			ib32 currentMaterial = pb[currentPath].xSampled.material;
			float_type dt = temporalCorrelationSamplerGpuData.ticks[temporalNum];
			VectorType U = temporalCorrelationMaterialGpuData[currentMaterial].U;

			pb_buffer[threadNum].position = pb[currentPath].xSampled.position +
				((cn == ConnectionType::ConnectionTypeIllumination || cn == ConnectionType::ConnectionTypeView) ? (-dt/2.0) : (dt / 2.0)) * U;

			pb_buffer[threadNum].material = currentMaterial;
			pb_buffer[threadNum].lambdaIdx = wavelenghNum;
		}
	}
	
	__global__ void copyPointsToBuffersThreePoints(const TemporalCorrelationPathPoint* pa, MediumPoint* pa_buffer,
		const TemporalCorrelationPathPoint* pb, VectorType* pb_buffer,
		const ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cn)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalCorrelationSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalCorrelationSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath;

			currentPath = pathsIdx[pathNum];

			ib32 currntMaterialNum = pa[currentPath].xSampled.material;

			if (currntMaterialNum > 0)
			{
				float_type dt = (cn == ConnectionType::ConnectionTypeIllumination || ConnectionType::ConnectionTypeView ? (float_type)-1.0 : (float_type)1.0) *
					temporalCorrelationSamplerGpuData.ticks[temporalNum];

				VectorType currntU = temporalCorrelationMaterialGpuData[currntMaterialNum].U;

				ib32 prevMaterialNum = pb[currentPath].xSampled.material;
				VectorType prevU = temporalCorrelationMaterialGpuData[prevMaterialNum].U;

				pa_buffer[threadNum].position = pa[currentPath].xSampled.position + (dt / 2.0) * currntU;
				pb_buffer[threadNum] = pb[currentPath].xSampled.position + (dt / 2.0) * prevU;

				pa_buffer[threadNum].material = currntMaterialNum;
				pa_buffer[threadNum].lambdaIdx = wavelenghNum;
			}
		}
	}

	__global__ void copyPointsToBuffersThreePoints(const TemporalCorrelationPathPoint* pa, MediumPoint* pa_buffer, ub32 totalPathsNum, ConnectionType cn)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalCorrelationSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalCorrelationSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			float_type dt = (cn == ConnectionType::ConnectionTypeIllumination || ConnectionType::ConnectionTypeView ? (float_type)-1.0 : (float_type)1.0) *
				temporalCorrelationSamplerGpuData.ticks[temporalNum];

			ub32 currntMaterialNum = pa[pathNum].xSampled.material;
			VectorType currntU = temporalCorrelationMaterialGpuData[currntMaterialNum].U;

			pa_buffer[threadNum].position = pa[pathNum].xSampled.position + (dt / 2.0) * currntU;

			pa_buffer[threadNum].material = currntMaterialNum;
			pa_buffer[threadNum].lambdaIdx = wavelenghNum;
		}
	}

	__global__ void copyPointsToBuffersTemporal(const TemporalCorrelationPathPoint* pa, MediumPoint* pa_buffer,
		const TemporalCorrelationPathPoint* pb, VectorType* pb_buffer, const ib32* pathsIdx, ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalCorrelationSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalCorrelationSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			ub32 currentPath;

			currentPath = pathsIdx[pathNum];

			float_type currentLambda = lambdaValues[wavelenghNum];
			float_type k = (2.0 * CUDART_PI) / (currentLambda);
			float_type dt = temporalCorrelationSamplerGpuData.ticks[temporalNum];

			ib32 currntMaterialNum = pa[currentPath].xSampled.material;
			float_type currntD = temporalCorrelationMaterialGpuData[currntMaterialNum].D;

			pa_buffer[threadNum] = pa[currentPath].xSampled;
			pb_buffer[threadNum] = pb[currentPath].xSampled.position;

			pa_buffer[threadNum].dtD = k * k * fabs(currntD * dt);
		}
	}

	__global__ void copyPointsToBuffersTemporalSingle(const TemporalCorrelationPathPoint* pa, MediumPoint* pa_buffer,
		ub32 totalPathsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalCorrelationSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalCorrelationSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			float_type currentLambda = lambdaValues[wavelenghNum];
			float_type k = (2.0 * CUDART_PI) / (currentLambda);
			float_type dt = temporalCorrelationSamplerGpuData.ticks[temporalNum];

			ib32 currntMaterialNum = pa[pathNum].xSampled.material;
			float_type currntD = temporalCorrelationMaterialGpuData[currntMaterialNum].D;

			pa_buffer[threadNum] = pa[pathNum].xSampled;
			pa_buffer[threadNum].dtD = k * k * fabs(currntD * dt);
		}
	}

	// Assuming running this only with one thread
	__global__ void randomNumberOfSources(ub32* outParam, ub32 sourcesNum)
	{
		*outParam = randUniformInteger(statePool, sourcesNum + 1);
	}

	__global__ void meanProbabilities(float_type* probA, const float_type* probB, ub32 probSize)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < probSize)
		{
			probA[threadNum] = 0.5 * (probA[threadNum] + probB[threadNum]);
		}
	}

	__global__ void pathContriobutionKernel(ComplexType* pContrb, const TemporalCorrelationPathPoint* pa, ib32* pathsIdx,
		ub32 totalPathsNum, ConnectionType cn, ub32 pL)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 wavelenghNum = threadNum % lambdaNum;
		ub32 temporalNum = (threadNum / lambdaNum) % temporalCorrelationSamplerGpuData.ticksNum;
		ub32 pathNum = threadNum / (temporalCorrelationSamplerGpuData.ticksNum * lambdaNum);

		if (pathNum < totalPathsNum)
		{
			if (pL == 1)
			{
				pContrb[threadNum] = (float_type)1.0;
			}
			else
			{
				ub32 currentPath;

				currentPath = pathsIdx[pathNum];

				ib32 currntMaterialNum = pa[currentPath].xSampled.material;

				if (currntMaterialNum > 0)
				{
					float_type currentLambda = lambdaValues[wavelenghNum];
					float_type k = (2.0 * CUDART_PI) / (currentLambda);
					float_type dt = (cn == ConnectionType::ConnectionTypeIllumination || ConnectionType::ConnectionTypeView ? (float_type)-1.0 : (float_type)1.0) *
						temporalCorrelationSamplerGpuData.ticks[temporalNum];

					float_type sinPhaseDistance, cosPhaseDistance;
					float_type phaseDistance = (float_type)((1.0) / currentLambda) * pa[currentPath].totalDist;
					sincospi(2.0 * phaseDistance, &sinPhaseDistance, &cosPhaseDistance);

					// We don't need to calculate the shared path momentum transfer twice, let's just use u1 path
					float_type temporalPathTransfer = (cn == ConnectionType::ConnectionTypeIllumination || cn == ConnectionType::ConnectionTypeView) ?
						(exp(-k * k * fabs(dt) * pa[currentPath].momentTransfer)) : 1.0;

					pContrb[threadNum] = ComplexType(temporalPathTransfer * cosPhaseDistance, temporalPathTransfer * sinPhaseDistance);

					// DEBUG
					//printf("Three points copy %s, thread: %d, path: %d, temporal: %d, wl = %d, currentPathNum = %d\n [%f %f %f] -> [%f, %f, %f]\n middle temportal atten = %e, const path = %f %f, totalDist = %f, momentum = %e \n",
					//	cn == ConnectionTypeIllumination ? "Illumination" : cn == ConnectionTypeIllumination2 ? "Illumination2" : cn == ConnectionTypeView ? "View" : "View2",
					//	threadNum, pathNum, temporalNum, wavelenghNum, currentPath,
					//	pa_buffer[currentPath].position.x(), pa_buffer[threadNum].position.y(), pa_buffer[threadNum].position.z(),
					//	pb_buffer[threadNum].x(), pb_buffer[threadNum].y(), pb_buffer[threadNum].z(), temporalPathTransfer, pathContribution[threadNum].real(), pathContribution[threadNum].imag(), pa[currentPath].totalDist, pa[currentPath].momentTransfer);
					
					//printf("Path contribution: %s, thread: %d, path: %d, temporal: %d, wl = %d, currentPathNum = %d\n middle temportal atten = %e, const path = %f %f, totalDist = %f, momentum = %e \n",
					//	cn == ConnectionTypeIllumination ? "Illumination" : cn == ConnectionTypeIllumination2 ? "Illumination2" : cn == ConnectionTypeView ? "View" : "View2",
					//	threadNum, pathNum, temporalNum, wavelenghNum, currentPath,
					//	temporalPathTransfer, pContrb[threadNum].real(), pContrb[threadNum].imag(), pa[currentPath].totalDist, pa[currentPath].momentTransfer);

					// END DEBUG
				}
				else
				{
					pContrb[threadNum] = (float_type)0.0;

					//printf("Three points copy out %s, thread: %d, path: %d, temporal: %d, wl = %d, currentPathNum = %d\n [%f %f %f] -> [%f, %f, %f]\n const path = %f %f, totalDist = %f, momentum = %e \n",
					//	cn == ConnectionTypeIllumination ? "Illumination" : cn == ConnectionTypeIllumination2 ? "Illumination2" : cn == ConnectionTypeView ? "View" : "View2",
					//	threadNum, pathNum, temporalNum, wavelenghNum, currentPath,
					//	pa[currentPath].xSampled.position.x(), pa[threadNum].xSampled.position.y(), pa[threadNum].xSampled.position.z(),
					//	pb[currentPath].xSampled.position.x(), pb[currentPath].xSampled.position.y(), pb[currentPath].xSampled.position.z(), pathContribution[threadNum].real(), pathContribution[threadNum].imag(), pa[currentPath].totalDist, pa[currentPath].momentTransfer);
				}
			}
		}
	}
}

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
TemporalCorrelationSampler::TemporalCorrelationSampler(ErrorType* err, Source* u1IlluminationsHandle, Source* u2IlluminationsHandle,
	Source* u1ViewsHandle, Source* u2ViewsHandle, const Simulation* simulationHandler, const Medium* mediumHandler,
	const Scattering* scatteringHandler, const float_type* ticks, ub32 ticksNum):
	Sampler(u1IlluminationsHandle, u1ViewsHandle, simulationHandler, mediumHandler, scatteringHandler, ticksNum),
	u1IlluminationsHandle(u1IlluminationsHandle), u2IlluminationsHandle(u2IlluminationsHandle), u1ViewsHandle(u1ViewsHandle), u2ViewsHandle(u2ViewsHandle)
{
	MEMORY_CHECK("temporal sampler correlation allocation begin");
	allocationsCount = 0;

	// copy ticks to CPU
	ticks_cpu = (float_type*)malloc(sizeof(float_type) * ticksNum);

	if(ticks_cpu == NULL)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	memcpy(ticks_cpu, ticks, sizeof(float_type) * ticksNum);

	if (cudaMalloc(&mediumPointsBuffer, sizeof(MediumPoint) * batchSize * wavelenghSize * samplerSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	if (cudaMalloc(&mediumPointsBuffer_2, sizeof(MediumPoint) * batchSize * wavelenghSize * samplerSize) != cudaError_t::cudaSuccess)
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

	ub32 maxSourceSize = illuminationSize > viewSize ? illuminationSize : viewSize;
	if (cudaMalloc(&tmpBuffer, sizeof(ComplexType) * batchSize * wavelenghSize * maxSourceSize * samplerSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	// get gpu data
	TemporalSamplerCorrelationNS::TemporalCorrelationSamplerGpuDataStruct gpuDataCpu;
	gpuDataCpu.ticksNum = ticksNum;
	gpuDataCpu.batchSize = batchSize;

	if (cudaMalloc(&gpuDataCpu.ticks, sizeof(float_type) * ticksNum) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationsCount++;

	if (cudaMemcpy(gpuDataCpu.ticks, ticks, sizeof(float_type) * ticksNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	if (cudaMemcpyToSymbol(TemporalSamplerCorrelationNS::temporalCorrelationSamplerGpuData, &gpuDataCpu, sizeof(TemporalSamplerCorrelationNS::TemporalCorrelationSamplerGpuDataStruct), 0, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	*err = ErrorType::NO_ERROR;

	MEMORY_CHECK("temporal sampler correlation allocation end");
}

TemporalCorrelationSampler::~TemporalCorrelationSampler()
{
	MEMORY_CHECK("temporal sampler correlation free begin");

	TemporalSamplerCorrelationNS::TemporalCorrelationSamplerGpuDataStruct gpuDataCpu;

	if (allocationsCount >= 7)
	{
		cudaMemcpyFromSymbol(&gpuDataCpu, TemporalSamplerCorrelationNS::temporalCorrelationSamplerGpuData, sizeof(TemporalSamplerCorrelationNS::TemporalCorrelationSamplerGpuDataStruct), 0, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}

	switch (allocationsCount)
	{
	case 7:
		cudaFree(gpuDataCpu.ticks);
	case 6:
		cudaFree(tmpBuffer);
	case 5:
		cudaFree(pathsVectorBuffer);
	case 4:
		cudaFree(previousPointsBuffer);
	case 3:
		cudaFree(mediumPointsBuffer_2);
	case 2:
		cudaFree(mediumPointsBuffer);
	case 1:
		free(ticks_cpu);
	default:
		break;
	}

	MEMORY_CHECK("temporal sampler correlation free end");
}

// ------------------------------------ Class Function Implementations ------------------------------------ //
__host__ ErrorType TemporalCorrelationSampler::setMaterial(float_type D, VectorType U, ub32 materialNum)
{
	TemporalSamplerCorrelationNS::TemporalCorrelationMaterialDataStructre currentMatrialData{ D, U };

	cudaMemcpyToSymbol(TemporalSamplerCorrelationNS::temporalCorrelationMaterialGpuData, &currentMatrialData,
		sizeof(TemporalSamplerCorrelationNS::TemporalCorrelationMaterialDataStructre),
		sizeof(TemporalSamplerCorrelationNS::TemporalCorrelationMaterialDataStructre) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ void TemporalCorrelationSampler::freePoints(Point* p, ub32 pointsNumber)
{
	cudaFree((TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p);
}

__host__ ErrorType TemporalCorrelationSampler::allocatePoints(Point** p, ub32 pointsNumber)
{
	// allocate gpu points
	if (cudaMalloc(p, sizeof(TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint) * pointsNumber) != cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	

	return ErrorType::NO_ERROR;
}

// pathsIdx: if negative value -> point is missed the medium.
__host__ ErrorType TemporalCorrelationSampler::sampleFirst(Point* pa, Point* pb, ib32* pathsIdx, ub32 totalPathsNum)
{
	ErrorType err;
	// Let the illumination source to sample the points

	// Use mediumPointsBuffer as a temporal buffer to get random number of illuminations used for source u1
	TemporalSamplerCorrelationNS::randomNumberOfSources<<<1,1>>>((ub32*)mediumPointsBuffer, totalPathsNum);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}


	ub32 u1RandomIlluminationNum;
	cudaMemcpy(&u1RandomIlluminationNum, mediumPointsBuffer, sizeof(ub32), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// sample new points
	// previousPointsBuffer: new sampled source point
	// pathsVectorBuffer   : new sampled source direction
	// mediumPointsBuffer  : new sampled point in medium
	if (u1RandomIlluminationNum > 0)
	{
		err = u1IlluminationsHandle->sampleFirstPoint(previousPointsBuffer,
			pathsVectorBuffer,
			mediumPointsBuffer,
			u1RandomIlluminationNum);

		if (err != ErrorType::NO_ERROR)
		{
			return err;
		}
	}

	if (totalPathsNum - u1RandomIlluminationNum > 0)
	{
		err = u2IlluminationsHandle->sampleFirstPoint(previousPointsBuffer + u1RandomIlluminationNum,
			pathsVectorBuffer + u1RandomIlluminationNum,
			mediumPointsBuffer + u1RandomIlluminationNum,
			totalPathsNum - u1RandomIlluminationNum);

		if (err != ErrorType::NO_ERROR)
		{
			return err;
		}
	}

	// Attach the resulted samples to the points structre

	ub32 threadsNum = totalPathsNum < THREADS_NUM ? totalPathsNum : THREADS_NUM;
	ub32 blocksNum = (totalPathsNum - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sPa = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)pa;
	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sPb = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)pb;

	TemporalSamplerCorrelationNS::copyBuffersToPoints <<<blocksNum, threadsNum >>> (
		sPa, previousPointsBuffer, sPb, mediumPointsBuffer, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType TemporalCorrelationSampler::sampleSecond(Point* pa, Point* pb, ib32* pathsIdx, ub32 totalPathsNum)
{
	ub32 threadsNum = totalPathsNum < THREADS_NUM ? totalPathsNum : THREADS_NUM;
	ub32 blocksNum = (totalPathsNum - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sPa = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)pa;
	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sPb = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)pb;

	// copy points data into buffers
	TemporalSamplerCorrelationNS::copyPointsToBuffers <<<blocksNum, threadsNum >>> (
		sPa, previousPointsBuffer, sPb, mediumPointsBuffer, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	// sample new point - assuming u1 and u2 have the same source type
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
	TemporalSamplerCorrelationNS::copyBuffersToPoints_momentumTransfer<false> <<<blocksNum, threadsNum >>> (
		sPa, sPb, mediumPointsBuffer_2, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType TemporalCorrelationSampler::sampleNext(Point* pa, Point* pb, ib32* pathsIdx, ub32 totalPathsNum, ub32 pL)
{
	ErrorType err;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sPa = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)pa;
	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sPb = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)pb;

	ub32 totalThreads = totalPathsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	// copy points data into buffers
	TemporalSamplerCorrelationNS::copyPointsToBuffers << <blocksNum, threadsNum >> > (
		sPa, previousPointsBuffer, sPb, mediumPointsBuffer, pathsIdx, totalPathsNum);


	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	// sample new direction
	// previousPointsBuffer: previous sampled point
	// mediumPointsBuffer  : current sampled point
	// pathsVectorBuffer   : new sampled direction
	err = scatteringHandler->newDirection(pathsVectorBuffer,
		previousPointsBuffer,
		mediumPointsBuffer,
		totalThreads);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// sample new point
	// mediumPointsBuffer_2: sampled new point
	// mediumPointsBuffer  : current sampled point
	// pathsVectorBuffer   : new sampled direction
	err = mediumHandler->sample(mediumPointsBuffer_2,
		mediumPointsBuffer, pathsVectorBuffer, totalThreads);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// copy back buffers to points
	TemporalSamplerCorrelationNS::copyBuffersToPoints_momentumTransfer<true> <<<blocksNum, threadsNum >>> (
		sPa, sPb, mediumPointsBuffer_2, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType TemporalCorrelationSampler::pathSingleProbability(float_type* probabilityRes, const Point* p1, ub32 totalPathsNum)
{
	ub32 totalThreads = totalPathsNum * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP1 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p1;

	// copy point p1 to buffers

	TemporalSamplerCorrelationNS::copyPointsToBuffers <<< blocksNum, threadsNum >>> (sP1, mediumPointsBuffer, wavelenghSize, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}
	
	// compute the probability regarding to the illumination sources
	ErrorType err = u1IlluminationsHandle->firstPointProbability((float_type*)tmpBuffer,
		mediumPointsBuffer,
		totalPathsNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	err = u2IlluminationsHandle->firstPointProbability(probabilityRes,
		mediumPointsBuffer,
		totalPathsNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	totalThreads = totalPathsNum * wavelenghSize * illuminationSize;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
	TemporalSamplerCorrelationNS::meanProbabilities <<< blocksNum, threadsNum >>>(probabilityRes, (float_type*)tmpBuffer, totalThreads);
	
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

// probabilityRes are the probabilities computed for the single scattering case
__host__ ErrorType TemporalCorrelationSampler::pathMultipleProbability(float_type* probabilityRes, const Point* p1, const Point* p2, const ib32* pathsIdx, ub32 totalPathsNum)
{
	ub32 totalThreads = totalPathsNum * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP1 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p1;
	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP2 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p2;

	// copy points data into buffers
	TemporalSamplerCorrelationNS::copyPointsToBuffers <<<blocksNum, threadsNum >>> (
		sP1, mediumPointsBuffer, sP2, previousPointsBuffer, pathsIdx, wavelenghSize, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}
	
	// Assuming u1 and u2 are made of the same source type
	return illuminationsHandle->secondPointProbability(probabilityRes,
		mediumPointsBuffer,
		previousPointsBuffer,
		totalPathsNum);
}

__host__ ErrorType TemporalCorrelationSampler::twoPointThroughput(ComplexType* gRes, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cType, ub32 pL)
{
	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP1 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p1;

	// copy points data into buffers, in throughput the result is the same for each time step
	TemporalSamplerCorrelationNS::copyPointsToBuffersThroughput <<<blocksNum, threadsNum >>> (
		sP1, mediumPointsBuffer, pathsIdx, totalPathsNum, cType);
	

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	Source* currentSource =
		cType == ConnectionType::ConnectionTypeIllumination  ? u1IlluminationsHandle :
		cType == ConnectionType::ConnectionTypeIllumination2 ? u2IlluminationsHandle :
		cType == ConnectionType::ConnectionTypeView          ? u1ViewsHandle         :
		cType == ConnectionType::ConnectionTypeView2         ? u2ViewsHandle         :
		NULL;

	return currentSource->throughputFunction(gRes, mediumPointsBuffer, totalThreads);
}

__host__ ErrorType TemporalCorrelationSampler::threePointThroughput(ComplexType* fRes, const Point* p1, const Point* p2, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cType, ub32 pL)
{
	ErrorType err;

	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP1 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p1;
	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP2 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p2;

	// copy points data into buffers
	TemporalSamplerCorrelationNS::copyPointsToBuffersThreePoints <<<blocksNum, threadsNum >>> (
		sP1, mediumPointsBuffer, sP2, previousPointsBuffer, pathsIdx, totalPathsNum, cType);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	Source* currentSource =
		cType == ConnectionType::ConnectionTypeIllumination  ? u1IlluminationsHandle :
		cType == ConnectionType::ConnectionTypeIllumination2 ? u2IlluminationsHandle :
		cType == ConnectionType::ConnectionTypeView          ? u1ViewsHandle         :
		cType == ConnectionType::ConnectionTypeView2         ? u2ViewsHandle         :
		NULL;

	err = currentSource->threePointFunction(fRes, mediumPointsBuffer, previousPointsBuffer, totalThreads);
	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType TemporalCorrelationSampler::threePointThroughputSingle(ComplexType* fsRes, const Point* p1, ub32 totalPathsNum, ConnectionType cType)
{
	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP1 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p1;

	// copy points data into buffers
	if (!isScalarFs())
	{
		TemporalSamplerCorrelationNS::copyPointsToBuffersThreePoints <<<blocksNum, threadsNum >>> (sP1, mediumPointsBuffer, totalPathsNum, cType);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}
	}
	
	if (cType == ConnectionType::ConnectionTypeIllumination)
	{
		return u1IlluminationsHandle->threePointFunctionSingle(fsRes, mediumPointsBuffer, u1ViewsHandle, totalThreads);
	}
	
	return u2IlluminationsHandle->threePointFunctionSingle(fsRes, mediumPointsBuffer, u2ViewsHandle, totalThreads);
}

ErrorType TemporalCorrelationSampler::temporalCorrelationThroughput(ComplexType* tRes, const Point* p1, const Point* p2, ib32* pathsIdx, ub32 totalPathsNum, ConnectionTypeCorrelation ccType, ub32 pL)
{
	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP1 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p1;
	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP2 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p2;

	// copy points data into buffers
	TemporalSamplerCorrelationNS::copyPointsToBuffersTemporal <<<blocksNum, threadsNum >>> (sP1, mediumPointsBuffer,
		sP2, previousPointsBuffer, pathsIdx, totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	const Source* thisSource;
	const Source* otherSource;
	bool isP1P2BeginsPaths;

	switch (ccType)
	{
	case ConnectionTypeCorrelation::C1_l1l2:
	{
		thisSource =  u1IlluminationsHandle;
		otherSource = u2IlluminationsHandle;
		isP1P2BeginsPaths = true;
		break;
	}
	case ConnectionTypeCorrelation::C1_v1v2:
	{
		thisSource  = u1ViewsHandle;
		otherSource = u2ViewsHandle;
		isP1P2BeginsPaths = false;
		break;
	}
	case ConnectionTypeCorrelation::C2_v1l2:
	{
		thisSource  = u1ViewsHandle;
		otherSource = u2IlluminationsHandle;
		isP1P2BeginsPaths = true;
		break;
	}
	case ConnectionTypeCorrelation::C2_l1v2:
	{
		thisSource  = u1IlluminationsHandle;
		otherSource = u2ViewsHandle;
		isP1P2BeginsPaths = false;
		break;
	}
	case ConnectionTypeCorrelation::C3_l1v2:
	{
		thisSource  = u1IlluminationsHandle;
		otherSource = u2ViewsHandle;
		isP1P2BeginsPaths = true;
		break;
	}
	case ConnectionTypeCorrelation::C3_v1l2:
	{
		thisSource  = u1ViewsHandle;
		otherSource = u2IlluminationsHandle;
		isP1P2BeginsPaths = false;
		break;
	}
	case ConnectionTypeCorrelation::C4_v1v2:
	{
		thisSource  = u1ViewsHandle;
		otherSource = u2ViewsHandle;
		isP1P2BeginsPaths = true;
		break;
	}
	case ConnectionTypeCorrelation::C4_l1l2:
	{
		thisSource  = u1IlluminationsHandle;
		otherSource = u2IlluminationsHandle;
		isP1P2BeginsPaths = false;
		break;
	}
	default:
		break;
	}

	return thisSource->temporalTransferFunction(tRes,
		mediumPointsBuffer,
		previousPointsBuffer,
		otherSource,
		totalThreads,
		isP1P2BeginsPaths);
}

ErrorType TemporalCorrelationSampler::temporalCorrelationThroughputSingle(ComplexType* tsRes, const Point* p1, ub32 totalPathsNum)
{
	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP1 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p1;

	// copy points data into buffers
	TemporalSamplerCorrelationNS::copyPointsToBuffersTemporalSingle <<<blocksNum, threadsNum >>> (sP1, mediumPointsBuffer,
		totalPathsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return u1IlluminationsHandle->temporalTransferFunctionSingle(tsRes, u2IlluminationsHandle, mediumPointsBuffer, u1ViewsHandle, u2ViewsHandle, totalThreads);
}

ErrorType TemporalCorrelationSampler::pathContribution(ComplexType* pContrb, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cn, ub32 pL)
{
	ub32 totalThreads = totalPathsNum * samplerSize * wavelenghSize;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint* sP1 = (TemporalSamplerCorrelationNS::TemporalCorrelationPathPoint*)p1;

	TemporalSamplerCorrelationNS::pathContriobutionKernel <<<blocksNum, threadsNum >>> (pContrb, sP1, pathsIdx, totalPathsNum, cn, pL);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}
