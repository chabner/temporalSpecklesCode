#pragma once

#include "../Simulation.cuh"

#include "SourceInterface.cuh"
#include "MediumInterface.cuh"
#include "../Implementation/Scattering.cuh"

// ------------------------------------ Sampler Class ------------------------------------ //

class Sampler
{
public:
	// DEBUG
	virtual void printAllPoints(Point* p, ub32 pointsNumber) = 0;
	// END DEBUG

	// HOST FUNCTION //
	// Allocate GPU memory for points
	// No need to initiate value
	virtual ErrorType allocatePoints(Point** p, ub32 pointsNumber) = 0;

	// HOST FUNCTION //
	// Free GPU memory for points
	virtual void freePoints(Point* p, ub32 pointsNumber) = 0;

	/* Sampling */

	// HOST FUNCTION //
	// Sample the first scattering point.
	// 
	// INPUT:
	// pathsIdx: index of participating paths, which need to be sampled, in size of totalPathsNum.
	// 
	// OUTPUT:
	// p0: the sampled source point
	// p1: the sampled first scatterer
	// pathsIdx: negative value for failed sampled path

	virtual ErrorType sampleFirst(Point* p0, Point* p1, ib32* pathsIdx, ub32 totalPathsNum) = 0;

	// HOST FUNCTION //
	// Sample second scattering point.
	// 
	// INPUT:
	// p1: the sampled source point.
	// p2: the sampled first scatterer
	// pathsIdx: index of participating paths, in size of totalPathsNum.
	// 
	// OUTPUT:
	// p1 <- p2.
	// p2: new next sampled point.
	// pathsIdx: negative value if new point is outside the volume.
	virtual ErrorType sampleSecond(Point* p1, Point* p2, ib32* pathsIdx, ub32 totalPathsNum) = 0;
	
	// HOST FUNCTION //
	// Sample next scattering point.
	// 
	// INPUT:
	// pa: the previous sampled point.
	// pb: the the current sampled point
	// pathsIdx: index of participating paths, in size of totalPathsNum.
	// 
	// OUTPUT:
	// pa <- pb.
	// pb: new next sampled point.
	// pathsIdx: negative value if new point is outside the volume.
	virtual ErrorType sampleNext(Point* pa, Point* pb, ib32* pathsIdx, ub32 totalPathsNum, ub32 pL) = 0;

	/* Probability */

	// HOST FUNCTION //
	// The sampling probability of the first scatterer as if the path is sampled from each possible illumination point.
	// 
	// INPUT:
	// p1: first scatterer point.
	// totalPathsNum: number of paths to sample.
	// 
	// OUTPUT:
	// probabilityRes: the path probability, in size of Nl x P,
	//                 where each entry is the sampling probability
	//                 as if it was sampled from the corresponding illumination.
	//

	virtual ErrorType pathSingleProbability(float_type* probabilityRes, const Point* p1, ub32 totalPathsNum) = 0;

	// HOST FUNCTION //
	// The sampling probability of the first and second scatterer as if the path is sampled from each possible illumination point.
	// 
	// INPUT:
	// probabilityRes: single scattering event path probabilities, as calculated in PathSingleProbability.
	// p1: first scatterer point.
	// p2: second scatterer point.
	// pathsIdx: index of participating paths, in size of totalPathsNum.
	// 
	// totalPathsNum: number of paths to sample.
	// 
	// OUTPUT:
	// probabilityRes: the path probability, in size of Nl x P,
	//                 where each entry is the sampling probability
	//                 as if it was sampled from the corresponding illumination.
	//

	virtual ErrorType pathMultipleProbability(float_type* probabilityRes, const Point* p1, const Point* p2, const ib32* pathsIdx, ub32 totalPathsNum) = 0;

	/* Throuput */

	// HOST FUNCTION //
	// Two points throuput for edge point
	virtual ErrorType twoPointThroughput(ComplexType* gRes, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cType, ub32 pL) = 0;

	// HOST FUNCTION //
	// Three points throuput for edge point
	// The path is p0 -> p1 -> p2, where p0 is the considered pixel
	virtual ErrorType threePointThroughput(ComplexType* fRes, const Point* p1, const Point* p2, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cType, ub32 pL) = 0;

	// HOST FUNCTION //
	// Three points throuput for single scattering
	virtual ErrorType threePointThroughputSingle(ComplexType* fsRes, const Point* p1, ub32 totalPathsNum, ConnectionType cType) = 0;

	// HOST FUNCTION //
	// Three points throuput for edge point
	// The path is p0 -> p1 -> p2, where p0 is the considered pixel
	virtual ErrorType temporalCorrelationThroughput(ComplexType* tRes, const Point* p1, const Point* p2, ib32* pathsIdx, ub32 totalPathsNum, ConnectionTypeCorrelation ccType, ub32 pL) = 0;

	// HOST FUNCTION //
	// Three points throuput for single scattering
	virtual ErrorType temporalCorrelationThroughputSingle(ComplexType* tsRes, const Point* p1, ub32 totalPathsNum) = 0;

	// HOST FUNCTION //
	// The contribution of a path, regardless of the source points
	virtual ErrorType pathContribution(ComplexType* pContrb, const Point* p1, ib32* pathsIdx, ub32 totalPathsNum, ConnectionType cn, ub32 pL) = 0;

	// Constructor
	Sampler(Source* illuminationsHandle, Source* viewsHandle, const Simulation* simulationHander, const Medium* mediumHandler, const Scattering* scatteringHandler, ub32 samplerSize):
		illuminationsHandle(illuminationsHandle),
		viewsHandle(viewsHandle),
		mediumHandler(mediumHandler),
		scatteringHandler(scatteringHandler),
		samplerSize(samplerSize),
		batchSize(simulationHander->getBatchSize()),
		wavelenghSize(simulationHander->getWavelenghSize()),
		illuminationSize(illuminationsHandle->getSourceSize()),
		viewSize(viewsHandle->getSourceSize()) { }

	// Access data
	ub32 getWavelenghSize() { return wavelenghSize; }
	ub32 getSamplerSize() { return samplerSize; }
	ub32 getIlluminationSize() { return illuminationSize; }
	ub32 getViewSize() { return viewSize; }
	ub32 getBatchSize() { return batchSize; }
	virtual bool isCorrelation() = 0;

	// return if fs can be treated as a scalar for performance boosting
	virtual bool isScalarFs() const { return illuminationsHandle->isScalarFs() && viewsHandle->isScalarFs(); };

protected:
	// sources handlers
	Source* illuminationsHandle;
	Source* viewsHandle;

	// Medium handler
	const Medium* mediumHandler;

	// Scattering handler
	const Scattering* scatteringHandler;

	ub32 wavelenghSize;
	ub32 samplerSize;
	ub32 batchSize;
	ub32 illuminationSize;
	ub32 viewSize;

private:
	Sampler() {};
};
