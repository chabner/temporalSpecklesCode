#pragma once

#include "../Simulation.cuh"
#include "../ComplexType.cuh"
#include "../Implementation/Scattering.cuh"
#include "MediumInterface.cuh"

// ------------------------------------ Source Class ------------------------------------ //

class Source {
public:
	ub32 getSourceSize() const {return sourceSize; };

	virtual ErrorType sampleFirstPoint(VectorType* randomSourcePoint,
		VectorType* randomSourceDirection,
		MediumPoint* randomFirstScatterer,
		ub32 samplesNum) = 0;

	virtual ErrorType firstPointProbability(float_type* probabilityRes,
		const MediumPoint* firstPoint,
		ub32 pointsNum) const = 0;

	virtual ErrorType sampleSecondPoint(VectorType* sampledSourcePoint,
		MediumPoint* firstSampledPoint,
		MediumPoint* secondSampledPoint,
		ub32 samplesNum);

	virtual ErrorType secondPointProbability(float_type* probabilityRes,
		const MediumPoint* firstPoint,
		const VectorType* secondPoint,
		ub32 pointsNum) const = 0;

	virtual ErrorType throughputFunction(ComplexType* gRes,
		const MediumPoint* p1,
		ub32 pointsNum) const = 0;

	// source -> p1 -> p2 contribution, likewise scattering amplitude
	virtual ErrorType threePointFunction(ComplexType* fRes,
		const MediumPoint* p1,
		const VectorType* p2,
		ub32 pointsNum) const = 0;

	// illumination -> p1 -> view contribution, likewise scattering amplitude
	virtual ErrorType threePointFunctionSingle(ComplexType* fsRes,
		const MediumPoint* p1,
		const Source* otherSource,
		ub32 pointsNum) const = 0;

	// tRes according to the moment transfer, regrading to multiple scattering.
	// tRes = (thisSource,otherSource)->p1->p2.
	// p1 and p2 in size of pointsNum.
	// If both thisSource and otherSource are illumination, tRes resulted size Nl x 1  x pointsNum
	// If both thisSource and otherSource are views, tRes resulted size        1  x Nv x pointsNum
	// If thisSource and otherSource are different kinds, tRes resulted size   Nl x Nv x pointsNum
	virtual ErrorType temporalTransferFunction(ComplexType* tRes,
		const MediumPoint* p1,
		const VectorType* p2,
		const Source* otherSource,
		ub32 pointsNum,
		bool isP1P2BeginsPaths) const = 0;

	// tsRes according to the moment transfer, regrading to single scattering.
	// tsRes = (thisSource,source1)->p0->(source2,source3).
	// p0 in size of pointsNum.
	// tsRes resulted size Nl x Nv x pointsNum
	virtual ErrorType temporalTransferFunctionSingle(ComplexType* tsRes,
		const Source* source1,
		const MediumPoint* p0,
		const Source* source2,
		const Source* source3,
		ub32 pointsNum) const = 0;

	// return if fs can be treated as a scalar for performance boosting
	virtual bool isScalarFs() const { return false; };

	// Data need to be accessed because of threePointFunctionSingle

	// Data Structures
	enum SourceType
	{
		PointSourceType,
		FarFieldType,
		GaussianBeamType,
		PencilType
	};

	struct PointSourceStructType
	{
		VectorType* points;
		ub32 pointsNum;
	};

	struct FarFieldStructType
	{
		ub32 directionsNum;

		VectorType* directions;
		VectorType* oppositeDirections;
		VectorType* inverseDirections;

		float_type* samplingProbability;
		VectorType* tangentPositions;

		VectorType* perpendicularDirectionsT;
		float_type* minT;
		float_type* maxT;

#if DIMS==3
		VectorType* perpendicularDirectionsS;
		float_type* minS;
		float_type* maxS;
#endif
	};

	struct GaussianBeamStructType
	{
		MediumPoint* focalPoints;
		VectorType* beamDirections;
		VectorType* beamOppositeDirections;
		
		float_type gaussianAperture;
		float_type apertureNormalization;

		ub32 beamsNum;
		ub32 beamSampleTableEntries;

#if DIMS==2
		VectorType* focalPointsVector;
#endif
	};

	struct PencilStructType
	{
		ub32 directionsNum;
		float_type interfaceWidth;

		VectorType* directions;
		VectorType* oppositeDirections;
		VectorType* inverseDirections;

		float_type* samplingProbability;
		VectorType* tangentPositions;

		VectorType* perpendicularDirectionsT;
		float_type* minT;
		float_type* maxT;

#if DIMS==3
		VectorType* perpendicularDirectionsS;
		float_type* minS;
		float_type* maxS;
#endif

		VectorType* interfacePoints;
		BlockingBox* interfaceBox;
	};

	// Variables
	ConnectionType connectionType;
	SourceType sourceType;
	void* data; // CPU pointer

protected:

	Source(const Simulation* simulationHandler, const Medium* mediumHandler, const Scattering* scatteringHandler, ConnectionType cType) :
		mediumHandler(mediumHandler), scatteringHandler(scatteringHandler), wavelenghNum(simulationHandler->getWavelenghSize()) {
	
		connectionType = (cType == ConnectionType::ConnectionTypeIllumination || cType == ConnectionType::ConnectionTypeIllumination2) ?
			ConnectionType::ConnectionTypeIllumination : ConnectionType::ConnectionTypeView;
	};

	const Medium* mediumHandler;
	const Scattering* scatteringHandler;
	ub32 sourceSize;
	ub32 wavelenghNum;

private:
	Source() {};
};

// ------------------------------------ Function Implementations ------------------------------------ //

// Default second point sampling is according to the phase function
ErrorType Source::sampleSecondPoint(VectorType* sampledSourcePoint,
	MediumPoint* firstSampledPoint,
	MediumPoint* secondSampledPoint,
	ub32 samplesNum)
{
	// use secondSampledPoint as temporal buffer for the sampled directoin
	VectorType* sampledDirection = (VectorType*)secondSampledPoint; // sizeof(MediumPoint) > sizeof(VectorType)

	ErrorType err = scatteringHandler->newDirection(sampledDirection,
		sampledSourcePoint,
		firstSampledPoint,
		samplesNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// we don't need sampledSourcePoint anymore, copy sampledDirection to sampledSourcePoint
	if (cudaMemcpy(sampledSourcePoint, sampledDirection, sizeof(VectorType) * samplesNum, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	// sample new point
	return mediumHandler->sample(secondSampledPoint, firstSampledPoint, sampledSourcePoint, samplesNum);
}


// ------------------------------------ Kernels ------------------------------------ //

namespace SourceNS
{
	// p0 -> p1 -> p2
	// p0 is illuminations, p1 is a scatterer inside the volume, p2 is views
	// Multiply the current value in tsRes
	template<bool isP0Direction, bool isP2Direction>
	__global__ void temporalCorrelationSourceContributionSingle(ComplexType* tsRes,
		const VectorType* p0u1, const VectorType* p0u2, ub32 p0Size,
		const MediumPoint* p1, ub32 p1Size,
		const VectorType* p2u1, const VectorType* p2u2, ub32 p2Size)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < p0Size * p1Size * p2Size)
		{
			ub32 p0Idx = threadNum % p0Size;
			ub32 p1Idx = (threadNum / (p0Size * p2Size));
			ub32 p2Idx = (threadNum / p0Size) % p2Size;

			// float_type k0 = 2.0 * CUDART_PI / lambdaValues[p1[p1Idx].lambdaIdx];

			VectorType p0p1DirU1, p0p1DirU2, p1p2DirU1, p1p2DirU2;
			if (isP0Direction)
			{
				p0p1DirU1 = p0u1[p0Idx];
				p0p1DirU2 = p0u2[p0Idx];
			}
			else
			{
				p0p1DirU1 = normalize(p1[p1Idx].position - p0u1[p0Idx]);
				p0p1DirU2 = normalize(p1[p1Idx].position - p0u2[p0Idx]);
			}

			if (isP2Direction)
			{
				p1p2DirU1 = p2u1[p2Idx];
				p1p2DirU2 = p2u2[p2Idx];
			}
			else
			{
				p1p2DirU1 = normalize(p2u1[p2Idx] - p1[p1Idx].position);
				p1p2DirU2 = normalize(p2u2[p2Idx] - p1[p1Idx].position);
			}

			VectorType momoent = 0.5 * (p0p1DirU1 + p0p1DirU2 - p1p2DirU1 - p1p2DirU2);
			tsRes[threadNum] = tsRes[threadNum] * exp(-p1[p1Idx].dtD * (momoent * momoent));

			// DEBUG
			//printf("temporal single: thread: %d, p0Idx: %d, p1Idx: %d, p2Idx: %d, dtD = %e, moment^2 = %e, \n Points: [%f %f %f] -> [%f %f %f] -> [%f %f %f]\n Directions: [%f %f %f] - [%f %f %f] \n",
			//threadNum, p0Idx, p1Idx, p2Idx, p1[p1Idx].dtD, momoent * momoent, p0u1[p0Idx].x, p0u1[p0Idx].y, p0u1[p0Idx].z, p1[p1Idx].position.x, p1[p1Idx].position.y, p1[p1Idx].position.z,
			//	p2u1[p2Idx].x, p2u1[p2Idx].y, p2u1[p2Idx].z, p0p1DirU1.x, p0p1DirU1.y, p0p1DirU1.z, p1p2DirU1.x, p1p2DirU1.y, p1p2DirU1.z);
			// END DEBUG
		}
	}

	// p0 -> p1 -> p2
	// p0 is illumination / view, p1 and p2 are scatterers inside the volume
	template<bool isP0u1Direction, bool isP0u2Direction>
	__global__ void temporalCorrelationSourceContributionMultiple(ComplexType* tRes,
		const VectorType* p0u1, ub32 p0u1Size, ConnectionType u1Connection,
		const VectorType* p0u2, ub32 p0u2Size, ConnectionType u2Connection,
		const MediumPoint* p1, const VectorType* p2, ub32 p1p2Size, bool isP1P2BeginsPaths)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 totalThreads, u1Idx, u2Idx, pathIdx;

		// ts is separatable, p0u1Size = p0u2Size = Nl or Nv, tsRes size is p0u1Size x p1p2Size.
		if (u1Connection == u2Connection)
		{
			totalThreads = p0u1Size * p1p2Size;
			u1Idx = u2Idx = threadNum % p0u1Size;
			pathIdx = threadNum / p0u1Size;
		}
		// non separatable, tsRes size is Nl x Nv x p1p2Size.
		else
		{
			totalThreads = p0u1Size * p0u2Size * p1p2Size;

			// Nl is allways the first dim
			if (u1Connection == ConnectionType::ConnectionTypeIllumination)
			{
				u1Idx = threadNum % p0u1Size;
				u2Idx = (threadNum / p0u1Size) % p0u2Size;
			}
			else
			{
				u2Idx = threadNum % p0u2Size;
				u1Idx = (threadNum / p0u2Size) % p0u1Size;
			}

			pathIdx = threadNum / (p0u1Size * p0u2Size) ;
		}

		if (threadNum < totalThreads)
		{
			if (p1[pathIdx].material != 0)
			{
				float_type u1Sign = (u1Connection == ConnectionType::ConnectionTypeIllumination ? (float_type)1.0 : (float_type)-1.0);
				float_type u2Sign = (u2Connection == ConnectionType::ConnectionTypeIllumination ? (float_type)1.0 : (float_type)-1.0);
				float_type pathSign = (isP1P2BeginsPaths ? (float_type)1.0 : (float_type)1.0);

				VectorType p0p1DirU1, p0p1DirU2;

				if (isP0u1Direction)
				{
					p0p1DirU1 = p0u1[u1Idx];
				}
				else
				{
					p0p1DirU1 = u1Sign * normalize(p1[pathIdx].position - p0u1[u1Idx]);
				}

				if (isP0u2Direction)
				{
					p0p1DirU2 = p0u2[u2Idx];
				}
				else
				{
					p0p1DirU2 = u2Sign * normalize(p1[pathIdx].position - p0u2[u2Idx]);
				}

				VectorType p1p2Dir = pathSign * normalize(p2[pathIdx] - p1[pathIdx].position);
				VectorType momoent = 0.5 * (u1Sign * p0p1DirU1 + u2Sign * p0p1DirU2) - p1p2Dir;
				tRes[threadNum] = exp(-p1[pathIdx].dtD * (momoent * momoent));

				// DEBUG
				//printf("temporal ms %s %s: thread: %d, u1Idx: %d, u2Idx: %d, pathIdx: %d, dtD = %e, tRes = %e %e \n Points: ([%f %f %f], [%f %f %f]) -> [%f %f %f] -> [%f %f %f]\n Directions: [%f %f %f] - [%f %f %f] \nmomoent = [%f %f %f], momoent^2 = %f, -p1[pathIdx].dtD * (momoent * momoent)) = %f\n",
				//	u1Connection == ConnectionTypeIllumination ? "Illumination" : "view", u2Connection == ConnectionTypeIllumination ? "Illumination" : "view",
				//	threadNum, u1Idx, u2Idx, pathIdx, p1[pathIdx].dtD, tRes[threadNum].x, exp(-p1[pathIdx].dtD * (momoent * momoent)), p0u1[u1Idx].x, p0u1[u1Idx].y, p0u1[u1Idx].z, p0u2[u2Idx].x, p0u2[u2Idx].y, p0u2[u2Idx].z, p1[pathIdx].position.x, p1[pathIdx].position.y, p1[pathIdx].position.z,
				//	p2[pathIdx].x, p2[pathIdx].y, p2[pathIdx].z, u1Sign * p0p1DirU1.x, u1Sign * p0p1DirU1.y, u1Sign * p0p1DirU1.z, p1p2Dir.x, p1p2Dir.y, p1p2Dir.z, momoent.x, momoent.y, momoent.z, momoent * momoent, -p1[pathIdx].dtD * (momoent * momoent));
				// END DEBUG
			}
			else
			{
				tRes[threadNum] = 0;
			}

		}
	}
}
