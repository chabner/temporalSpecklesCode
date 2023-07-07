#pragma once

#include "../../Interface/SourceInterface.cuh"
#include "../Scattering.cuh"
#include "../../Interface/MediumInterface.cuh"

// ------------------------------------ Far Field Class ------------------------------------ //
class FarField : public Source
{
public:
	// cpu pointers
	FarField(ErrorType* err, const Simulation* simulationHandler, const Medium* mediumHandler, const Scattering* scatteringHandler,
		const VectorType* directions, ub32 directionsNum, ConnectionType cType);

	~FarField();

	// Declerations
	virtual ErrorType sampleFirstPoint(VectorType* randomSourcePoint,
		VectorType* randomSourceDirection,
		MediumPoint* randomFirstScatterer,
		ub32 samplesNum);

	virtual ErrorType firstPointProbability(float_type* probabilityRes,
		const MediumPoint* firstPoint,
		ub32 pointsNum) const;

	virtual ErrorType secondPointProbability(float_type* probabilityRes,
		const MediumPoint* firstPoint,
		const VectorType* secondPoint,
		ub32 pointsNum) const;

	virtual ErrorType throughputFunction(ComplexType* gRes,
		const MediumPoint* p1,
		ub32 pointsNum) const;

	// source -> p1 -> p2 contribution, likewise scattering amplitude
	virtual ErrorType threePointFunction(ComplexType* fRes,
		const MediumPoint* p1,
		const VectorType* p2,
		ub32 pointsNum) const;

	// illumination -> p1 -> view contribution, likewise scattering amplitude
	virtual ErrorType threePointFunctionSingle(ComplexType* fsRes,
		const MediumPoint* p1,
		const Source* otherSource,
		ub32 pointsNum) const;

	virtual ErrorType temporalTransferFunction(ComplexType* tRes,
		const MediumPoint* p1,
		const VectorType* p2,
		const Source* otherSource,
		ub32 pointsNum,
		bool isP1P2BeginsPaths) const;

	virtual ErrorType temporalTransferFunctionSingle(ComplexType* tsRes,
		const Source* source1,
		const MediumPoint* p0,
		const Source* source2,
		const Source* source3,
		ub32 pointsNum) const;

protected:
	ub32 allocationCount;
};



// ------------------------------------ Kernels ------------------------------------ //
namespace FarFieldNS
{
	__global__ void fillBuffers(Source::FarFieldStructType sourceData, BlockingCircle blockingCircle, BlockingBox box, ConnectionType cn)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < sourceData.directionsNum)
		{
			sourceData.directions[threadNum] = normalize(sourceData.directions[threadNum]);
			sourceData.oppositeDirections[threadNum] = (float_type)(-1.0) * sourceData.directions[threadNum];

			if (cn == ConnectionType::ConnectionTypeView || cn == ConnectionType::ConnectionTypeView2)
			{
				return;
			}

			sourceData.inverseDirections[threadNum] = {
				abs(sourceData.directions[threadNum].x()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / sourceData.directions[threadNum].x(),
				abs(sourceData.directions[threadNum].y()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / sourceData.directions[threadNum].y()
#if DIMS==3
				,abs(sourceData.directions[threadNum].z()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / sourceData.directions[threadNum].z()
#endif
			};

			sourceData.tangentPositions[threadNum] = blockingCircle.center + blockingCircle.radius * sourceData.oppositeDirections[threadNum];
			
#if DIMS==3
			VectorType orthDir[2];
			orthogonalBase(sourceData.oppositeDirections[threadNum], orthDir);
			sourceData.perpendicularDirectionsT[threadNum] = orthDir[0];
			sourceData.perpendicularDirectionsS[threadNum] = orthDir[1];
#else
			VectorType orthDir;
			orthogonalBase(sourceData.oppositeDirections[threadNum], &orthDir);
			sourceData.perpendicularDirectionsT[threadNum] = orthDir;
#endif

			// Compute all minimal and maximal times from the box boundary to the perpendicular directions
			projectBoxToLine(box, sourceData.tangentPositions[threadNum], sourceData.perpendicularDirectionsT[threadNum], sourceData.minT + threadNum, sourceData.maxT + threadNum);

#if DIMS==3
			projectBoxToLine(box, sourceData.tangentPositions[threadNum], sourceData.perpendicularDirectionsS[threadNum], sourceData.minS + threadNum, sourceData.maxS + threadNum);
#endif

#if DIMS==3
			sourceData.samplingProbability[threadNum] = 1.0 / ((sourceData.maxT[threadNum] - sourceData.minT[threadNum]) * (sourceData.maxS[threadNum] - sourceData.minS[threadNum]));
#else
			sourceData.samplingProbability[threadNum] = 1.0 / (sourceData.maxT[threadNum] - sourceData.minT[threadNum]);
#endif

			//printf("build %d: direction: [%f %f %f], tangent: [%f %f %f], orthDir1: [%f %f %f], t: [%f %f], orthDir2: [%f %f %f], s: [%f %f], samplingProb: %f \n",
			//	threadNum, sourceData.directions[threadNum].x, sourceData.directions[threadNum].y, sourceData.directions[threadNum].z,
			//	sourceData.tangentPositions[threadNum].x, sourceData.tangentPositions[threadNum].y, sourceData.tangentPositions[threadNum].z,
			//	sourceData.perpendicularDirectionsT[threadNum].x, sourceData.perpendicularDirectionsT[threadNum].y, sourceData.perpendicularDirectionsT[threadNum].z,
			//	sourceData.minT[threadNum], sourceData.maxT[threadNum],
			//	sourceData.perpendicularDirectionsS[threadNum].x, sourceData.perpendicularDirectionsS[threadNum].y, sourceData.perpendicularDirectionsS[threadNum].z,
			//	sourceData.minS[threadNum], sourceData.maxS[threadNum], sourceData.samplingProbability[threadNum]);
		}
	}

	__global__ void sampleFarFieldKernel(VectorType* randomChosenDirection,
		VectorType* randomChosenPoint,
		ub32 totalPathNum,
		Source::FarFieldStructType farFieldSourceStruct,
		BlockingBox box)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathNum)
		{
			curandState_t* currentState = statePool + pathNum;

			// randomize an illuminator among all options
			ub32 illuminatorNum = randUniformInteger(currentState, farFieldSourceStruct.directionsNum);
			VectorType chosenDir = farFieldSourceStruct.directions[illuminatorNum];

			VectorType randomPointOnLine = farFieldSourceStruct.tangentPositions[illuminatorNum] +
				(farFieldSourceStruct.minT[illuminatorNum] + randUniform(currentState) * (farFieldSourceStruct.maxT[illuminatorNum] - farFieldSourceStruct.minT[illuminatorNum])) *
				farFieldSourceStruct.perpendicularDirectionsT[illuminatorNum]
#if DIMS==3
				+ (farFieldSourceStruct.minS[illuminatorNum] + randUniform(currentState) * (farFieldSourceStruct.maxS[illuminatorNum] - farFieldSourceStruct.minS[illuminatorNum])) *
				farFieldSourceStruct.perpendicularDirectionsS[illuminatorNum]
#endif
				;

			VectorType shiftedRandomPointOnLine = randomPointOnLine - box.center;
			VectorType invRayDirection = farFieldSourceStruct.inverseDirections[illuminatorNum];

			ib32 sgn[DIMS] = { -(chosenDir.x() < 0. ? -1 : 1), -(chosenDir.y() < 0. ? -1 : 1)
	#if DIMS==3
				, -(chosenDir.z() < 0. ? -1 : 1)
	#endif
			};

			VectorType distanceToPlane = VectorType(
				(invRayDirection.x()) * (box.radius.x() * sgn[0] - shiftedRandomPointOnLine.x()),
				(invRayDirection.y()) * (box.radius.y() * sgn[1] - shiftedRandomPointOnLine.y())
	#if DIMS==3
			   ,(invRayDirection.z()) * (box.radius.z() * sgn[2] - shiftedRandomPointOnLine.z())
	#endif
			);

			bool test[DIMS];
			float_type bd;

#if DIMS==2
			test[0] = (distanceToPlane.x() >= 0.0) &&
				(abs(fma(chosenDir.y(), distanceToPlane.x(), shiftedRandomPointOnLine.y())) < box.radius.y());

			test[1] = (distanceToPlane.y() >= 0.0) &&
				(abs(fma(chosenDir.x(), distanceToPlane.y(), shiftedRandomPointOnLine.x())) < box.radius.x());

			bd = test[0] ? distanceToPlane.x() : (test[1] ? distanceToPlane.y() : 0);
#else
			test[0] = (distanceToPlane.x() >= 0.0) &&
				(abs(fma(chosenDir.y(), distanceToPlane.x(), shiftedRandomPointOnLine.y())) < box.radius.y()) &&
				(abs(fma(chosenDir.z(), distanceToPlane.x(), shiftedRandomPointOnLine.z())) < box.radius.z());

			test[1] = (distanceToPlane.y() >= 0.0) &&
				(abs(fma(chosenDir.x(), distanceToPlane.y(), shiftedRandomPointOnLine.x())) < box.radius.x()) &&
				(abs(fma(chosenDir.z(), distanceToPlane.y(), shiftedRandomPointOnLine.z())) < box.radius.z());

			test[2] = (distanceToPlane.z() >= 0.0) &&
				(abs(fma(chosenDir.x(), distanceToPlane.z(), shiftedRandomPointOnLine.x())) < box.radius.x()) &&
				(abs(fma(chosenDir.y(), distanceToPlane.z(), shiftedRandomPointOnLine.y())) < box.radius.y());

			bd = test[0] ? distanceToPlane.x() : (test[1] ? distanceToPlane.y() : (test[2] ? distanceToPlane.z() : 0));
#endif

			randomChosenDirection[pathNum] = chosenDir;
			randomChosenPoint[pathNum] = randomPointOnLine + (bd + EPSILON) * chosenDir;

			//printf("sapmle %d: chosen dir: [%f %f %f], randomPointOnLine: [%f %f %f], bd: %f, randomChosenPoint: [%f %f %f] \n",
			//	pathNum, chosenDir.x, chosenDir.y, chosenDir.z, randomPointOnLine.x, randomPointOnLine.y, randomPointOnLine.z,
			//	bd, randomChosenPoint[pathNum].x, randomChosenPoint[pathNum].y, randomChosenPoint[pathNum].z);
		}
	}

	__global__ void probabilityDirectionNormalizeKernel(float_type* probabilityRes,
		ub32 pointsNum,
		Source::FarFieldStructType farFieldSourceStruct)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 sourceNum = threadNum % farFieldSourceStruct.directionsNum;
		// ub32 wavelenghNum = (threadNum / pointSourceStruct.pointsNum) % lambdaNum;
		ub32 scattererNum = threadNum / (farFieldSourceStruct.directionsNum * lambdaNum);

		if (scattererNum < pointsNum)
		{
			probabilityRes[threadNum] *= farFieldSourceStruct.samplingProbability[sourceNum];
		}
	}

	__global__ void gDirectionKernel(ComplexType* gRes,
		const MediumPoint* p1, ub32 pointsNum, Source::FarFieldStructType directionSourceStruct, ConnectionType cType)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 totalPixels = directionSourceStruct.directionsNum;

		// threads order: (pixel,time,path)
		ub32 pixelNumber = threadNum % totalPixels;
		ub32 currentPathNumber = threadNum / totalPixels;

		if (currentPathNumber < pointsNum)
		{
			if (p1[currentPathNumber].material > 0)
			{
				VectorType dir = directionSourceStruct.directions[pixelNumber];
				VectorType x1 = p1[currentPathNumber].position;

				float_type atten = gRes[threadNum].real();

				// and since we are computing field, we need the square root of the attenuation
				atten = sqrt(atten);

				float_type multSign = (float_type)((cType == ConnectionType::ConnectionTypeIllumination || cType == ConnectionType::ConnectionTypeIllumination2) ? 1.0 : -1.0);

				float_type r = dir * x1;
				float_type sinPhase, cosPhase;

				sincospi(multSign * 2.0 * r / lambdaValues[p1[currentPathNumber].lambdaIdx], &sinPhase, &cosPhase);
				gRes[threadNum] = ComplexType(atten * cosPhase, atten * sinPhase);

				//printf("g %d: atten: %f, sourceContribution: [%f %f], currentPathContribution: [%f %f] \n",
				//	threadNum, atten, sourceContribution.x, sourceContribution.y,
				//	currentPathContribution.x, currentPathContribution.y);
			}
			else
			{
				gRes[threadNum] = 0;
			}
			
		}
	}

}

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
FarField::FarField(ErrorType* err, const Simulation* simulationHandler, const Medium* mediumHandler, const Scattering* scatteringHandler,
	const VectorType* directions, ub32 directionsNum, ConnectionType cType) :
	Source(simulationHandler, mediumHandler, scatteringHandler, cType)
{

	MEMORY_CHECK("ff allocation begin");

	sourceType = SourceType::FarFieldType;
	sourceSize = directionsNum;
	allocationCount = 0;

	FarFieldStructType* sourceData;

	sourceData = (FarFieldStructType*)malloc(sizeof(FarFieldStructType));

	if (sourceData == 0)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	sourceData->directionsNum = directionsNum;

	if (cudaMalloc(&sourceData->directions, sizeof(VectorType) * directionsNum) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&sourceData->oppositeDirections, sizeof(VectorType) * directionsNum) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cType == ConnectionType::ConnectionTypeIllumination || cType == ConnectionType::ConnectionTypeIllumination2)
	{
		if (cudaMalloc(&sourceData->inverseDirections, sizeof(VectorType) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		if (cudaMalloc(&sourceData->samplingProbability, sizeof(float_type) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		if (cudaMalloc(&sourceData->tangentPositions, sizeof(VectorType) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		if (cudaMalloc(&sourceData->perpendicularDirectionsT, sizeof(VectorType) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		if (cudaMalloc(&sourceData->minT, sizeof(float_type) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		if (cudaMalloc(&sourceData->maxT, sizeof(float_type) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

#if DIMS==3
		if (cudaMalloc(&sourceData->perpendicularDirectionsS, sizeof(VectorType) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		if (cudaMalloc(&sourceData->minS, sizeof(float_type) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		if (cudaMalloc(&sourceData->maxS, sizeof(float_type) * directionsNum) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;
#endif
	}

	if (cudaMemcpy(sourceData->directions, directions, sizeof(VectorType) * directionsNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	ub32 threadsNum = directionsNum < THREADS_NUM ? directionsNum : THREADS_NUM;
	ub32 blocksNum = (directionsNum - 1) / THREADS_NUM + 1;;

	FarFieldNS::fillBuffers <<<blocksNum, threadsNum >>>(*sourceData, mediumHandler->getBlockingCircle(), mediumHandler->getBlockingBox(), cType);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::KERNEL_ERROR;
		return;
	}

	data = sourceData;
	*err = NO_ERROR;

	MEMORY_CHECK("ff allocation end");
}

FarField::~FarField()
{
	MEMORY_CHECK("ff free begin");

	FarFieldStructType* sourceData = (FarFieldStructType*)data;

	switch (allocationCount)
	{
#if DIMS==3
	case 12:
		cudaFree(sourceData->maxS);
	case 11:
		cudaFree(sourceData->minS);
	case 10:
		cudaFree(sourceData->perpendicularDirectionsS);
#endif
	case 9:
		cudaFree(sourceData->maxT);
	case 8:
		cudaFree(sourceData->minT);
	case 7:
		cudaFree(sourceData->perpendicularDirectionsT);
	case 6:
		cudaFree(sourceData->tangentPositions);
	case 5:
		cudaFree(sourceData->samplingProbability);
	case 4:
		cudaFree(sourceData->inverseDirections);
	case 3:
		cudaFree(sourceData->oppositeDirections);
	case 2:
		cudaFree(sourceData->directions);
	case 1:
		free(sourceData);
    default:
        break;
	}
	MEMORY_CHECK("ff free end");
}

// ------------------------------------ Function Implementations ------------------------------------ //
__host__ ErrorType FarField::sampleFirstPoint(VectorType* randomSourcePoint,
	VectorType* randomSourceDirection,
	MediumPoint* randomFirstScatterer,
	ub32 samplesNum)
{
	ub32 threadsNum = samplesNum < THREADS_NUM ? samplesNum : THREADS_NUM;
	ub32 blocksNum = (samplesNum - 1) / THREADS_NUM + 1;

	FarFieldStructType* directionData = (FarFieldStructType*)data;

	// sample a point inside the medium
	FarFieldNS::sampleFarFieldKernel <<<blocksNum, threadsNum >>> (randomSourceDirection,
		randomSourcePoint,
		samplesNum,
		*directionData,
		mediumHandler->getBlockingBox());

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return mediumHandler->sample(randomFirstScatterer,
		randomSourcePoint,
		randomSourceDirection,
		samplesNum);
}


__host__ ErrorType FarField::firstPointProbability(float_type* probabilityRes,
	const MediumPoint* firstPoint,
	ub32 pointsNum) const
{
	FarFieldStructType* directionData = (FarFieldStructType*)data;

	// compute attenuation
	ErrorType err = mediumHandler->attenuationDirection(probabilityRes,
		directionData->oppositeDirections, directionData->directionsNum,
		firstPoint, pointsNum * wavelenghNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	ub32 totalThreads = pointsNum * wavelenghNum * directionData->directionsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	// divide by sampling probability of the projection on box
	FarFieldNS::probabilityDirectionNormalizeKernel <<<blocksNum, threadsNum >>> (probabilityRes, pointsNum, *directionData);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType FarField::secondPointProbability(float_type* probabilityRes,
	const MediumPoint* firstPoint,
	const VectorType* secondPoint,
	ub32 pointsNum) const
{
	FarFieldStructType* directionData = (FarFieldStructType*)data;

	return scatteringHandler->multiplyPdfDirection(probabilityRes,
		directionData->directions, directionData->directionsNum,
		firstPoint,
		secondPoint, pointsNum * wavelenghNum);
}

__host__ ErrorType FarField::throughputFunction(ComplexType* gRes,
	const MediumPoint* p1,
	ub32 pointsNum) const
{
	FarFieldStructType* directionData = (FarFieldStructType*)data;

	// compute attenuation
	ErrorType err = mediumHandler->attenuationDirection(gRes,
		(connectionType == ConnectionType::ConnectionTypeIllumination || connectionType == ConnectionType::ConnectionTypeIllumination2) ? directionData->oppositeDirections : directionData->directions,
		directionData->directionsNum,
		p1, pointsNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	ub32 totalThreads = pointsNum * directionData->directionsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
	
	FarFieldNS::gDirectionKernel <<< blocksNum, threadsNum >>> (gRes, p1, pointsNum, *directionData, connectionType);
	
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType FarField::threePointFunction(ComplexType* fRes,
	const MediumPoint* p1,
	const VectorType* p2,
	ub32 pointsNum) const
{
	FarFieldStructType* directionData = (FarFieldStructType*)data;

	return scatteringHandler->amplitude(fRes,
		(connectionType == ConnectionType::ConnectionTypeIllumination || connectionType == ConnectionType::ConnectionTypeIllumination2) ? directionData->directions : directionData->oppositeDirections, true, directionData->directionsNum,
		p1, p2, false, pointsNum);
}

__host__ ErrorType FarField::threePointFunctionSingle(ComplexType* fsRes,
	const MediumPoint* p1,
	const Source* otherSource,
	ub32 pointsNum) const
{
	FarFieldStructType* directionData = (FarFieldStructType*)data;

	if (otherSource->sourceType == SourceType::PointSourceType)
	{
		PointSourceStructType* otherSourceData = (PointSourceStructType*)otherSource->data;

		return scatteringHandler->amplitude(fsRes,
			directionData->directions, true, directionData->directionsNum,
			p1, pointsNum,
			otherSourceData->points, false, otherSourceData->pointsNum);
	}
	else if (otherSource->sourceType == SourceType::FarFieldType)
	{
		// can be cached
		FarFieldStructType* otherSourceData = (FarFieldStructType*)otherSource->data;

		return scatteringHandler->amplitude(fsRes,
			directionData->directions, true, directionData->directionsNum,
			p1, pointsNum,
			otherSourceData->directions, true, otherSourceData->directionsNum);
	}
	else if (otherSource->sourceType == SourceType::PencilType)
	{
		PencilStructType* otherSourceData = (PencilStructType*)otherSource->data;

		return scatteringHandler->amplitude(fsRes,
			directionData->directions, true, directionData->directionsNum,
			p1, pointsNum,
			otherSourceData->directions, true, otherSourceData->directionsNum);
	}
	else
	{
		return ErrorType::NOT_SUPPORTED;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType FarField::temporalTransferFunction(ComplexType* tRes,
	const MediumPoint* p1,
	const VectorType* p2,
	const Source* otherSource,
	ub32 pointsNum,
	bool isP1P2BeginsPaths) const
{
	ub32 totalThreads = pointsNum * sourceSize * (connectionType == otherSource->connectionType ? 1 : otherSource->getSourceSize());
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	FarFieldStructType* directionData = (FarFieldStructType*)data;

	if (otherSource->sourceType == SourceType::PointSourceType)
	{
		PointSourceStructType* otherSourceData = (PointSourceStructType*)otherSource->data;
		SourceNS::temporalCorrelationSourceContributionMultiple<true, false> <<< blocksNum, threadsNum >>> (tRes,
			directionData->directions, directionData->directionsNum, connectionType,
			otherSourceData->points, otherSourceData->pointsNum, otherSource->connectionType,
			p1, p2, pointsNum, isP1P2BeginsPaths);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}

		return ErrorType::NO_ERROR;
	}
	else if (otherSource->sourceType == SourceType::FarFieldType)
	{
		FarFieldStructType* otherSourceData = (FarFieldStructType*)otherSource->data;
		SourceNS::temporalCorrelationSourceContributionMultiple<true, true> <<< blocksNum, threadsNum >>> (tRes,
			directionData->directions, directionData->directionsNum, connectionType,
			otherSourceData->directions, otherSourceData->directionsNum, otherSource->connectionType,
			p1, p2, pointsNum, isP1P2BeginsPaths);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}

		return ErrorType::NO_ERROR;
	}
	else if (otherSource->sourceType == SourceType::PencilType)
	{
		PencilStructType* otherSourceData = (PencilStructType*)otherSource->data;
		SourceNS::temporalCorrelationSourceContributionMultiple<true, true> <<< blocksNum, threadsNum >>> (tRes,
			directionData->directions, directionData->directionsNum, connectionType,
			otherSourceData->directions, otherSourceData->directionsNum, otherSource->connectionType,
			p1, p2, pointsNum, isP1P2BeginsPaths);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}

		return ErrorType::NO_ERROR;
	}
	else
	{
		return ErrorType::NOT_SUPPORTED;
	}

	return ErrorType::NOT_SUPPORTED;
}

// this source and source2 may be other type, but this source and source1 must be same type, as same for source2 and source3
__host__ ErrorType FarField::temporalTransferFunctionSingle(ComplexType* tsRes,
	const Source* source1,
	const MediumPoint* p0,
	const Source* source2,
	const Source* source3,
	ub32 pointsNum) const
{
	ub32 totalThreads = pointsNum * sourceSize * source2->getSourceSize();
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	FarFieldStructType* directionData = (FarFieldStructType*)data;
	FarFieldStructType* source1directionData = (FarFieldStructType*)source1->data;

	if (source2->sourceType == SourceType::PointSourceType)
	{
		PointSourceStructType* source2Data = (PointSourceStructType*)source2->data;
		PointSourceStructType* source3Data = (PointSourceStructType*)source3->data;

		SourceNS::temporalCorrelationSourceContributionSingle<true, false> <<< blocksNum, threadsNum >>> (tsRes,
			directionData->directions, source1directionData->directions, directionData->directionsNum,
			p0, pointsNum,
			source2Data->points, source3Data->points, source2Data->pointsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}

		return ErrorType::NO_ERROR;
	}
	else if (source2->sourceType == SourceType::FarFieldType)
	{
		FarFieldStructType* source2Data = (FarFieldStructType*)source2->data;
		FarFieldStructType* source3Data = (FarFieldStructType*)source3->data;
		SourceNS::temporalCorrelationSourceContributionSingle<true, true> <<< blocksNum, threadsNum >>> (tsRes,
			directionData->directions, source1directionData->directions, directionData->directionsNum,
			p0, pointsNum,
			source2Data->directions, source3Data->directions, source2Data->directionsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}

		return ErrorType::NO_ERROR;
	}
	else if (source2->sourceType == SourceType::PencilType)
	{
		PencilStructType* source2Data = (PencilStructType*)source2->data;
		PencilStructType* source3Data = (PencilStructType*)source3->data;
		SourceNS::temporalCorrelationSourceContributionSingle<true, true> <<< blocksNum, threadsNum >>> (tsRes,
			directionData->directions, source1directionData->directions, directionData->directionsNum,
			p0, pointsNum,
			source2Data->directions, source3Data->directions, source2Data->directionsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR;
		}

		return ErrorType::NO_ERROR;
	}
	else
	{
		return ErrorType::NOT_SUPPORTED;
	}

	return ErrorType::NOT_SUPPORTED;
}
