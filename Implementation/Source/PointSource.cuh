#pragma once

#include "../../Interface/SourceInterface.cuh"
#include "../Scattering.cuh"
#include "../../Interface/MediumInterface.cuh"

// ------------------------------------ Point Source Class ------------------------------------ //
class PointSource : public Source
{
public:
	// cpu pointers
	PointSource(ErrorType* err, const Simulation* simulationHandler, const Medium* mediumHandler, const Scattering* scatteringHandler,
		const VectorType* points, ub32 pointsNum, bool isMulyiplyDirection, ConnectionType cType);

	~PointSource();

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

	virtual bool isScalarFs() const { return scatteringHandler->isIsotropicScattering(); };

protected:
	ub32 allocationCount;
	bool _isMulyiplyDirection;
};

// ------------------------------------ Kernels ------------------------------------ //
namespace PointSourceNS
{
	__global__ void sampleFirstPointKernel(VectorType* chosenSource,
		VectorType* randomChosenDirection,
		ub32 totalPathNum,
		Source::PointSourceStructType pointSourceStruct)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathNum)
		{
			curandState_t* currentState = statePool + pathNum;

			// randomize an illuminator among all options
			ub32 illuminatorNum = randUniformInteger(currentState, pointSourceStruct.pointsNum);
			chosenSource[pathNum] = pointSourceStruct.points[illuminatorNum];

			// sample a random direction
			randomChosenDirection[pathNum] = randomDirection(currentState);
		}
	}

	__global__ void probabilityPointNormalizeKernel(float_type* probabilityRes,
		const MediumPoint* firstPoint,
		ub32 pointsNum,
		Source::PointSourceStructType pointSourceStruct)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 sourceNum = threadNum % pointSourceStruct.pointsNum;
		// ub32 wavelenghNum = (threadNum / pointSourceStruct.pointsNum) % lambdaNum;
		ub32 scattererNum = threadNum / (pointSourceStruct.pointsNum * lambdaNum);

		if (scattererNum < pointsNum)
		{
			float_type rr = rabs(firstPoint[scattererNum].position - pointSourceStruct.points[sourceNum]);

#if DIMS==2
			// probabilityRes[threadNum] *= rr / (float_type(2.0 * CUDART_PI));
            probabilityRes[threadNum] *= (rr * (float_type(8.0 * CUDART_PI)));
#else
			probabilityRes[threadNum] *= (rr * rr) / (float_type(4.0 * CUDART_PI));
#endif
		}

	}

	template<bool isDirectionMult>
	__global__ void gPointKernel(ComplexType* gRes,
		const MediumPoint* p1, ub32 pointsNum, Source::PointSourceStructType pointSourceStruct)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 totalPixels = pointSourceStruct.pointsNum;

		// threads order: (pixel,time,path)
		ub32 pixelNumber = threadNum % totalPixels;
		ub32 currentPathNumber = threadNum / totalPixels;

		if (currentPathNumber < pointsNum)
		{
			if (p1[currentPathNumber].material > 0)
			{
				VectorType x0 = pointSourceStruct.points[pixelNumber];
				VectorType x1 = p1[currentPathNumber].position;

				float_type atten = gRes[threadNum].real();

				float_type rr = rabs(x1 - x0);

				// in point source we devide the attenuation with the distance
				atten *= (rr
#if DIMS==3
					* rr * ISOTROPIC_AMPLITUDE
#endif
					);

				if (isDirectionMult)
				{
#if DIMS==3
					atten *= (rr * abs(x1.z() - x0.z()));
#else
					atten *= (rr * abs(x1.y() - x0.y()));
#endif
				}

				// and since we are computing field, we need the square root of the attenuation
				atten = sqrt(atten);

				float_type r = 1 / rr;
				float_type sinPhase, cosPhase;
				sincospi((float_type)(2.0) * (r / lambdaValues[p1[currentPathNumber].lambdaIdx]), &sinPhase, &cosPhase);
				gRes[threadNum] = ComplexType(atten * cosPhase, atten * sinPhase);

				//printf("Point source throughput %d: x0 = [%f %f %f], x1 = [%f %f %f], attenCalc = %f, r = %f, rr = %f, phaseContb = %f + %fi, cosAng = %f, gRes = [%f %f] \n",
				//	threadNum, x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, atten, r, rr, sinPhase, cosPhase,
				//	isDirectionMult ? (rr * abs(x1.z - x0.z)) : 0.0, gRes[threadNum].x, gRes[threadNum].y);
			}
			else
			{
				gRes[threadNum] = 0;
			}
		}
	}
}

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
PointSource::PointSource(ErrorType* err, const Simulation* simulationHandler, const Medium* mediumHandler, const Scattering* scatteringHandler,
	const VectorType* points, ub32 pointsNum, bool isMulyiplyDirection, ConnectionType cType):
	Source(simulationHandler, mediumHandler, scatteringHandler, cType), _isMulyiplyDirection(isMulyiplyDirection)
{
	MEMORY_CHECK("ps allocation begin");

	sourceType = SourceType::PointSourceType;
	sourceSize = pointsNum;
	allocationCount = 0;

	PointSourceStructType* sourceData;

	sourceData = (PointSourceStructType*)malloc(sizeof(PointSourceStructType));

	if (sourceData == 0)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	sourceData->pointsNum = pointsNum;

	if (cudaMalloc(&sourceData->points, sizeof(VectorType) * pointsNum) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMemcpy(sourceData->points, points, sizeof(VectorType) * pointsNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	data = sourceData;
	*err = NO_ERROR;

	MEMORY_CHECK("ff allocation end");
}

PointSource::~PointSource()
{
	MEMORY_CHECK("ff free begin");
	PointSourceStructType* sourceData = (PointSourceStructType*)data;

	switch (allocationCount)
	{
	case 2:
		cudaFree(sourceData->points);
	case 1:
		free(sourceData);
	default:
		break;
	}
	MEMORY_CHECK("ff free end");
}

// ------------------------------------ Function Implementations ------------------------------------ //
__host__ ErrorType PointSource::sampleFirstPoint(VectorType* randomSourcePoint,
	VectorType* randomSourceDirection,
	MediumPoint* randomFirstScatterer,
	ub32 samplesNum)
{
	ub32 threadsNum = samplesNum < THREADS_NUM ? samplesNum : THREADS_NUM;
	ub32 blocksNum = (samplesNum - 1) / THREADS_NUM + 1;

	PointSourceStructType* pointData = (PointSourceStructType*)data;

	PointSourceNS::sampleFirstPointKernel <<<blocksNum, threadsNum >>> (randomSourcePoint,
		randomSourceDirection,
		samplesNum,
		*pointData);

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

__host__ ErrorType PointSource::firstPointProbability(float_type* probabilityRes,
	const MediumPoint* firstPoint,
	ub32 pointsNum) const
{
	PointSourceStructType* pointData = (PointSourceStructType*)data;

	// compute attenuation
	ErrorType err = mediumHandler->attenuationPoint(probabilityRes,
		pointData->points, pointData->pointsNum,
		firstPoint, pointsNum * wavelenghNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	ub32 totalThreads = pointsNum * wavelenghNum * pointData->pointsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	// divide by distance
	PointSourceNS::probabilityPointNormalizeKernel <<<blocksNum, threadsNum >>> (probabilityRes,
		firstPoint, pointsNum, *pointData);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType PointSource::secondPointProbability(float_type* probabilityRes,
	const MediumPoint* firstPoint,
	const VectorType* secondPoint,
	ub32 pointsNum) const
{
	PointSourceStructType* pointData = (PointSourceStructType*)data;

	return scatteringHandler->multiplyPdf(probabilityRes,
		pointData->points, pointData->pointsNum,
		firstPoint,
		secondPoint, pointsNum * wavelenghNum);
}

__host__ ErrorType PointSource::throughputFunction(ComplexType* gRes,
	const MediumPoint* p1,
	ub32 pointsNum) const
{
	PointSourceStructType* pointData = (PointSourceStructType*)data;

	// compute attenuation
	ErrorType err = mediumHandler->attenuationPoint(gRes,
		pointData->points, pointData->pointsNum,
		p1, pointsNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	ub32 totalThreads = pointsNum * pointData->pointsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
	
	if (_isMulyiplyDirection == true)
	{
		PointSourceNS::gPointKernel<true> <<< blocksNum, threadsNum >>> (gRes, p1, pointsNum, *pointData);
	}
	else
	{
		PointSourceNS::gPointKernel<false> <<< blocksNum, threadsNum >>> (gRes, p1, pointsNum, *pointData);
	}
	
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType PointSource::threePointFunction(ComplexType* fRes,
	const MediumPoint* p1,
	const VectorType* p2,
	ub32 pointsNum) const
{
	PointSourceStructType* pointData = (PointSourceStructType*)data;

	if (scatteringHandler->isIsotropicScattering())
	{
		return matrixFill(fRes, ISOTROPIC_AMPLITUDE, pointsNum * pointData->pointsNum);
	}
	else
	{
		ErrorType err = scatteringHandler->amplitude(fRes,
			pointData->points, false, pointData->pointsNum,
			p1,
			p2, false, pointsNum);

		if (err != ErrorType::NO_ERROR)
		{
			return err;
		}
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType PointSource::threePointFunctionSingle(ComplexType* fsRes,
	const MediumPoint* p1,
	const Source* otherSource,
	ub32 pointsNum) const
{
	PointSourceStructType* pointData = (PointSourceStructType*)data;

	if (otherSource->sourceType == SourceType::PointSourceType)
	{
		PointSourceStructType* otherSourceData = (PointSourceStructType*)otherSource->data;

		if (scatteringHandler->isIsotropicScattering())
		{
			ub32 fillSize;

			if (isScalarFs() && otherSource->isScalarFs())
			{
				fillSize = 1;
			}
			else
			{
				fillSize = pointsNum * pointData->pointsNum * otherSourceData->pointsNum;
			}
			
			return matrixFill(fsRes, ISOTROPIC_AMPLITUDE, fillSize);
		}
		else
		{
			return scatteringHandler->amplitude(fsRes,
				pointData->points, false, pointData->pointsNum,
				p1, pointsNum,
				otherSourceData->points, false, otherSourceData->pointsNum);
		}
	}
	else if (otherSource->sourceType == SourceType::FarFieldType)
	{
		FarFieldStructType* otherSourceData = (FarFieldStructType*)otherSource->data;

		if (scatteringHandler->isIsotropicScattering())
		{
			return matrixFill(fsRes, ISOTROPIC_AMPLITUDE, pointsNum * pointData->pointsNum * otherSourceData->directionsNum);
		}
		else
		{
			return scatteringHandler->amplitude(fsRes,
				pointData->points, false, pointData->pointsNum,
				p1, pointsNum,
				otherSourceData->directions, true, otherSourceData->directionsNum);
		}
	}
	else if (otherSource->sourceType == SourceType::PencilType)
	{
		PencilStructType* otherSourceData = (PencilStructType*)otherSource->data;

		if (scatteringHandler->isIsotropicScattering())
		{
			return matrixFill(fsRes, ISOTROPIC_AMPLITUDE, pointsNum * pointData->pointsNum * otherSourceData->directionsNum);
		}
		else
		{
			return scatteringHandler->amplitude(fsRes,
				pointData->points, false, pointData->pointsNum,
				p1, pointsNum,
				otherSourceData->directions, true, otherSourceData->directionsNum);
		}
	}
	else
	{
		return ErrorType::NOT_SUPPORTED;
	}

	return ErrorType::NO_ERROR;
}

__host__ ErrorType PointSource::temporalTransferFunction(ComplexType* tRes,
	const MediumPoint* p1,
	const VectorType* p2,
	const Source* otherSource,
	ub32 pointsNum,
	bool isP1P2BeginsPaths) const
{
	ub32 totalThreads = pointsNum * sourceSize * (connectionType == otherSource->connectionType ? 1 : otherSource->getSourceSize());
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	PointSourceStructType* pointData = (PointSourceStructType*)data;

	if (otherSource->sourceType == SourceType::PointSourceType)
	{
		PointSourceStructType* otherSourceData = (PointSourceStructType*)otherSource->data;
		SourceNS::temporalCorrelationSourceContributionMultiple<false, false> <<< blocksNum, threadsNum>>> (tRes,
			pointData->points, pointData->pointsNum, connectionType,
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
		SourceNS::temporalCorrelationSourceContributionMultiple<false, true> <<< blocksNum, threadsNum>>> (tRes,
			pointData->points, pointData->pointsNum, connectionType,
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
		SourceNS::temporalCorrelationSourceContributionMultiple<false, true> <<< blocksNum, threadsNum>>> (tRes,
			pointData->points, pointData->pointsNum, connectionType,
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
__host__ ErrorType PointSource::temporalTransferFunctionSingle(ComplexType* tsRes,
	const Source* source1,
	const MediumPoint* p0,
	const Source* source2,
	const Source* source3,
	ub32 pointsNum) const
{
	ub32 totalThreads = pointsNum * sourceSize * source2->getSourceSize();
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	PointSourceStructType* pointData = (PointSourceStructType*)data;
	PointSourceStructType* source1pointData = (PointSourceStructType*)source1->data;

	if (source2->sourceType == SourceType::PointSourceType)
	{
		PointSourceStructType* source2Data = (PointSourceStructType*)source2->data;
		PointSourceStructType* source3Data = (PointSourceStructType*)source3->data;

		SourceNS::temporalCorrelationSourceContributionSingle<false, false> <<< blocksNum, threadsNum >>> (tsRes,
			pointData->points, source1pointData->points, pointData->pointsNum,
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
		SourceNS::temporalCorrelationSourceContributionSingle<false, true> <<< blocksNum, threadsNum >>> (tsRes,
			pointData->points, source1pointData->points, pointData->pointsNum,
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
		SourceNS::temporalCorrelationSourceContributionSingle<false, true> <<< blocksNum, threadsNum >>> (tsRes,
			pointData->points, source1pointData->points, pointData->pointsNum,
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
