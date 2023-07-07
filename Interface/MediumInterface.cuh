#pragma once

#include "../Simulation.cuh"
#include "../ComplexType.cuh"
#include "../VectorType.cuh"

// ------------------------------------ Public Data Structures ------------------------------------ //
typedef struct
{
	float_type sigs;
	float_type siga;
} MediumMaterial;

struct BlockingCircle
{
	VectorType center;
	float_type radius;
};

struct BlockingBox
{
	VectorType boxMin;
	VectorType boxMax;

	VectorType center;
	VectorType radius;
};

// ------------------------------------ GPU Constants ------------------------------------ //
namespace MediumNS
{
	__constant__ MediumMaterial mediumGpuData[MATERIAL_NUM];
	__constant__ VectorType boxMin;
	__constant__ VectorType boxMax;
}

// ------------------------------------ Medium Class ------------------------------------ //

class Medium
{
public:
	// HOST FUNCTION //
	// Get the attenuation between two points.
	// Use the material number of p2.
	// points order p1 -> p2. number of points is pointsNum.
	ErrorType attenuationPoint(float_type* atteunationRes,
		const VectorType* p1,
		const MediumPoint* p2,
		ub32 pointsNum) const;

	ErrorType attenuationPoint(ComplexType* atteunationRes,
		const VectorType* p1,
		const MediumPoint* p2,
		ub32 pointsNum) const;

	ErrorType attenuationPoint(float_type* atteunationRes,
		const VectorType* p1, ub32 p1num,
		const MediumPoint* p2, ub32 p2num) const;

	ErrorType attenuationPoint(ComplexType* atteunationRes,
		const VectorType* p1, ub32 p1num,
		const MediumPoint* p2, ub32 p2num) const;

	ErrorType attenuationDirection(float_type* atteunationRes,
		const VectorType* direction,
		const MediumPoint* p2,
		ub32 pointsNum) const;

	ErrorType attenuationDirection(ComplexType* atteunationRes,
		const VectorType* direction,
		const MediumPoint* p2,
		ub32 pointsNum) const;

	ErrorType attenuationDirection(float_type* atteunationRes,
		const VectorType* direction, ub32 directionNum,
		const MediumPoint* p2, ub32 p2num) const;

	ErrorType attenuationDirection(ComplexType* atteunationRes,
		const VectorType* direction, ub32 directionNum,
		const MediumPoint* p2, ub32 p2num) const;

	// HOST FUNCTION //
	// Sample next point
	// Material value 0 is miss
	virtual ErrorType sample(MediumPoint* sampledPoint,
		const VectorType* sourcePoint, const VectorType* sapmleDirection,
		ub32 pointsNum) const;
	virtual ErrorType sample(MediumPoint* sampledPoint,
		const MediumPoint* sourcePoint, const VectorType* sapmleDirection,
		ub32 pointsNum) const;

	// HOST FUNCTION //
	// Set material to medium
	// material is CPU pointer
	virtual ErrorType setMaterial(const MediumMaterial* material, ub32 materialNum);

	Medium(VectorType boxMin, VectorType boxMax);

	// HOST FUNCTION //
	// Sample random points inside the volume
	// sampledPoint is device pointer
	ErrorType sampleRandomInside(VectorType* sampledPoint, ub32 pointsNum) const;

	// HOST FUNCTION //
	// get material info of a medium point
	virtual ErrorType getMaterialInfo(MediumPoint* sampledPoint, ub32 pointsNum) const = 0;

	// HOST FUNCTION //
	// Check if points (x,y,z) are in the same bin as the ref point
	// Assume ref point is inside the medium
	// We have pointsNum x refNum points to check
	virtual ErrorType isSameBin(bool* isSameBuffer, const MediumPoint* refPoint,
		const VectorType* pos, ub32 refNum, ub32 pointsNum) const;

	BlockingBox getBlockingBox() const { return _blockingBox; }
	BlockingCircle getBlockingCircle() const { return _blockingCircle; }

protected:

	virtual ErrorType attenuation(void* atteunationRes,
		const VectorType* p1, ub32 p1num,
		const MediumPoint* p2, ub32 p2num,
		bool isComplex, bool isSeparatable, bool isP1direction) const = 0;

	virtual ErrorType sampleImplementation(MediumPoint* sampledPoint,
		const void* sourcePoint, const VectorType* sapmleDirection,
		ub32 pointsNum, bool isSourcePointIsMediumPoint) const = 0;

	Medium() {};
	VectorType _boxMin;
	VectorType _boxMax;
	
	BlockingBox _blockingBox;
	BlockingCircle _blockingCircle;
};
// ------------------------------------ Bounding Box Help Kernels ------------------------------------ //

__device__ float_type pointToLineProjectionDistance(VectorType p, VectorType p0, VectorType v)
{
#if DIMS==3
	VectorType orthDir = {
		v.x() * v.x() * (p.x() - p0.x()) + v.x() * v.y() * (p.y() - p0.y()) + v.x() * v.z() * (p.z() - p0.z()),
		v.y() * v.x() * (p.x() - p0.x()) + v.y() * v.y() * (p.y() - p0.y()) + v.y() * v.z() * (p.z() - p0.z()),
		v.z() * v.x() * (p.x() - p0.x()) + v.z() * v.y() * (p.y() - p0.y()) + v.z() * v.z() * (p.z() - p0.z()) };
#else
	VectorType orthDir = {
		v.x() * v.x() * (p.x() - p0.x()) + v.x() * v.y() * (p.y() - p0.y()),
		v.y() * v.x() * (p.x() - p0.x()) + v.y() * v.y() * (p.y() - p0.y()) };
#endif

	return orthDir * v;
}


// project a bounding box to a line, defined with center and direction
__device__ void projectBoxToLine(BlockingBox box, VectorType linePoint, VectorType lineDirection, float_type* tMin, float_type* tMax)
{
#if DIMS==3
	ub32 edgePoints = 8;
	float_type tBox[8];
	tBox[0] = pointToLineProjectionDistance(VectorType( box.boxMin.x(), box.boxMin.y(), box.boxMin.z()), linePoint, lineDirection);
	tBox[1] = pointToLineProjectionDistance(VectorType( box.boxMax.x(), box.boxMin.y(), box.boxMin.z()), linePoint, lineDirection);
	tBox[2] = pointToLineProjectionDistance(VectorType( box.boxMin.x(), box.boxMax.y(), box.boxMin.z()), linePoint, lineDirection);
	tBox[3] = pointToLineProjectionDistance(VectorType( box.boxMax.x(), box.boxMax.y(), box.boxMin.z()), linePoint, lineDirection);
	tBox[4] = pointToLineProjectionDistance(VectorType( box.boxMin.x(), box.boxMin.y(), box.boxMax.z()), linePoint, lineDirection);
	tBox[5] = pointToLineProjectionDistance(VectorType( box.boxMax.x(), box.boxMin.y(), box.boxMax.z()), linePoint, lineDirection);
	tBox[6] = pointToLineProjectionDistance(VectorType( box.boxMin.x(), box.boxMax.y(), box.boxMax.z()), linePoint, lineDirection);
	tBox[7] = pointToLineProjectionDistance(VectorType( box.boxMax.x(), box.boxMax.y(), box.boxMax.z()), linePoint, lineDirection);
#else
	ub32 edgePoints = 4;

	float_type tBox[4];
	tBox[0] = pointToLineProjectionDistance(VectorType( box.boxMin.x(), box.boxMin.y()), linePoint, lineDirection);
	tBox[1] = pointToLineProjectionDistance(VectorType( box.boxMax.x(), box.boxMin.y()), linePoint, lineDirection);
	tBox[2] = pointToLineProjectionDistance(VectorType( box.boxMin.x(), box.boxMax.y()), linePoint, lineDirection);
	tBox[3] = pointToLineProjectionDistance(VectorType( box.boxMax.x(), box.boxMax.y()), linePoint, lineDirection);
#endif

	float_type tMinLocal, tMaxLocal;

	tMinLocal = tBox[0];
	tMaxLocal = tBox[0];

	for (ub32 boxPointNum = 1; boxPointNum < edgePoints; boxPointNum++)
	{
		if (tBox[boxPointNum] < tMinLocal)
		{
			tMinLocal = tBox[boxPointNum];
		}

		if (tBox[boxPointNum] > tMaxLocal)
		{
			tMaxLocal = tBox[boxPointNum];
		}
	}

	*tMin = tMinLocal;
	*tMax = tMaxLocal;
}

// ------------------------------------ Kernels ------------------------------------ //
namespace MediumNS
{
	__device__ bool isInsideKernel(VectorType p)
	{
		return p.x() >= boxMin.x() && p.x() <= boxMax.x() &&
			p.y() >= boxMin.y() && p.y() <= boxMax.y()
#if DIMS==3
			&& p.z() >= boxMin.z() && p.z() <= boxMax.z()
#endif
			;
	}

	__device__ bool isInsideToleranceKernel(VectorType p, float_type tol)
	{
		return p.x() >= (boxMin.x() + tol) && p.x() <= (boxMax.x() - tol) &&
			p.y() >= (boxMin.y() + tol) && p.y() <= (boxMax.y() - tol)
#if DIMS==3
			&& p.z() >= (boxMin.z() + tol) && p.z() <= (boxMax.z() - tol)
#endif
			;
	}

	__global__ void sampleRandomInsideKernel(VectorType* sampledPoint, ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < pointsNum)
		{
			curandState_t* currentState = statePool + threadNum;

			sampledPoint[threadNum] = VectorType(boxMin.x() + randUniform(currentState) * (boxMax.x() - boxMin.x()),
				boxMin.y() + randUniform(currentState) * (boxMax.y() - boxMin.y())
#if DIMS==3
			, boxMin.z() + randUniform(currentState) * (boxMax.z() - boxMin.z())
#endif
			);
		}
	}

	__global__ void getIfInside(bool* isInsideRes, const VectorType* pos, ub32 totalPoints)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < totalPoints)
		{
			isInsideRes[threadNum] = isInsideKernel(pos[threadNum]);
		}
	}
}
// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
Medium::Medium(VectorType boxMin, VectorType boxMax): _boxMin(boxMin), _boxMax(boxMax)
{
	MEMORY_CHECK("Medium allocation begin");
	// init all sigs and siga to 0
	MediumMaterial mat0;
	mat0.sigs = (float_type)0.;
	mat0.siga = (float_type)0.;

	for (ub32 materialNum = 0; materialNum < MATERIAL_NUM; materialNum++)
	{
		cudaMemcpyToSymbol(MediumNS::mediumGpuData, &mat0, sizeof(MediumMaterial), sizeof(MediumMaterial) * materialNum, cudaMemcpyKind::cudaMemcpyHostToDevice);
	}

	cudaMemcpyToSymbol(MediumNS::boxMin, &boxMin, sizeof(VectorType), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(MediumNS::boxMax, &boxMax, sizeof(VectorType), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

	VectorType boxRadius = 0.5 * (boxMax - boxMin);
	VectorType boxCenter = 0.5 * (boxMax + boxMin);
	float_type maxRadius = maxComponent(boxRadius) + (float_type)(EPSILON);

	_blockingBox.boxMax = boxMax;
	_blockingBox.boxMin = boxMin;
	_blockingBox.center = boxCenter;
	_blockingBox.radius = boxRadius;

	_blockingCircle.center = boxCenter;
	_blockingCircle.radius = maxRadius;

	MEMORY_CHECK("Medium allocation end");
}

// ------------------------------------ Class Function Implementations ------------------------------------ //
ErrorType Medium::attenuationPoint(float_type* atteunationRes,
	const VectorType* p1,
	const MediumPoint* p2,
	ub32 pointsNum) const
{
	return attenuation((void*)atteunationRes,
		p1, pointsNum,
		p2, pointsNum,
		false, false, false);
}

ErrorType Medium::attenuationPoint(ComplexType* atteunationRes,
	const VectorType* p1,
	const MediumPoint* p2,
	ub32 pointsNum) const
{
	return attenuation((void*)atteunationRes,
		p1, pointsNum,
		p2, pointsNum,
		true, false, false);
}

ErrorType Medium::attenuationPoint(float_type* atteunationRes,
	const VectorType* p1, ub32 p1num,
	const MediumPoint* p2, ub32 p2num) const
{
	return attenuation((void*)atteunationRes,
		p1, p1num,
		p2, p2num,
		false, true, false);
}

ErrorType Medium::attenuationPoint(ComplexType* atteunationRes,
	const VectorType* p1, ub32 p1num,
	const MediumPoint* p2, ub32 p2num) const
{
	return attenuation((void*)atteunationRes,
		p1, p1num,
		p2, p2num,
		true, true, false);
}

ErrorType Medium::attenuationDirection(float_type* atteunationRes,
	const VectorType* direction,
	const MediumPoint* p2,
	ub32 pointsNum) const
{
	return attenuation((void*)atteunationRes,
		direction, pointsNum,
		p2, pointsNum,
		false, false, true);
}

ErrorType Medium::attenuationDirection(ComplexType* atteunationRes,
	const VectorType* direction,
	const MediumPoint* p2,
	ub32 pointsNum) const
{
	return attenuation((void*)atteunationRes,
		direction, pointsNum,
		p2, pointsNum,
		true, false, true);
}

ErrorType Medium::attenuationDirection(float_type* atteunationRes,
	const VectorType* direction, ub32 directionNum,
	const MediumPoint* p2, ub32 p2num) const
{
	return attenuation((void*)atteunationRes,
		direction, directionNum,
		p2, p2num,
		false, true, true);
}

ErrorType Medium::attenuationDirection(ComplexType* atteunationRes,
	const VectorType* direction, ub32 directionNum,
	const MediumPoint* p2, ub32 p2num) const
{
	return attenuation((void*)atteunationRes,
		direction, directionNum,
		p2, p2num,
		true, true, true);
}

ErrorType Medium::sample(MediumPoint* sampledPoint,
	const VectorType* sourcePoint, const VectorType* sapmleDirection,
	ub32 pointsNum) const
{
	return sampleImplementation(sampledPoint,
		(const void*)sourcePoint, sapmleDirection,
		pointsNum, false);
}

ErrorType Medium::sample(MediumPoint* sampledPoint,
	const MediumPoint* sourcePoint, const VectorType* sapmleDirection,
	ub32 pointsNum) const
{
	return sampleImplementation(sampledPoint,
		(const void*)sourcePoint, sapmleDirection,
		pointsNum, true);
}

ErrorType Medium::setMaterial(const MediumMaterial* material, ub32 materialNum)
{
	cudaMemcpyToSymbol(MediumNS::mediumGpuData, material, sizeof(MediumMaterial), sizeof(MediumMaterial) * materialNum, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Medium::sampleRandomInside(VectorType* sampledPoint, ub32 pointsNum) const
{
	ub32 threadsNum = pointsNum < THREADS_NUM ? pointsNum : THREADS_NUM;
	ub32 blocksNum = (pointsNum - 1) / THREADS_NUM + 1;

	MediumNS::sampleRandomInsideKernel <<< blocksNum, threadsNum>>> (sampledPoint, pointsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Medium_sampleRandomInsideKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Medium::isSameBin(bool* isSameBuffer, const MediumPoint* refPoint,
	const VectorType* pos, ub32 refNum, ub32 pointsNum) const
{
	ub32 totalThreads = refNum * pointsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	MediumNS::getIfInside <<< blocksNum, threadsNum>>> (isSameBuffer, pos, totalThreads);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::DEVICE_ERROR;
	}

	return ErrorType::NO_ERROR;
}
