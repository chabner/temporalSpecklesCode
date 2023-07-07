#pragma once

#include "../../Interface/MediumInterface.cuh"

#include <thrust/execution_policy.h>
//#include <thrust/binary_search.h>

// box intersection algorithm: https://jcgt.org/published/0007/03/04/
// A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering

// xAxis, yAxis and zAxis are the bins defining the box
// resulting in (xAxisSize - 1) x (yAxisSize - 1) x (zAxisSize - 1) box grid values
// the values are integers, representing the material number. these values are stored in materialNum.
// material number 0 is outside the box, thus all values of materialNum are equal or greater than 1.

// ------------------------------------ GPU Constants ------------------------------------ //
namespace HeterogeneousMediumNS
{
	// Heterogeneous medium structre
	typedef struct
	{
		const float_type* xAxis;
		ub32 xAxisSize;
		const float_type* yAxis;
		ub32 yAxisSize;

#if DIMS==3
		const float_type* zAxis;
		ub32 zAxisSize;
#endif

		const ub32* materialNum;

	} HeterogeneousMediumDataStructre;

	__constant__ HeterogeneousMediumDataStructre heterogeneousMediumGpuData;

	__constant__ float_type maxSigt;

	// Axis type
	typedef enum
	{
		axisX,
		axisY
#if DIMS==3
		,axisZ
#endif
	} AxisType;
}

// ------------------------------------ Heterogeneous Medium Class ------------------------------------ //

class HeterogeneousMedium: public Medium
{
public:

	// All pointers in CPU
	HeterogeneousMedium(ErrorType* err, const ub32* materialNum,
		const float_type* xAxis, ub32 xAxisSize,
		const float_type* yAxis, ub32 yAxisSize
#if DIMS==3
		, const  float_type* zAxis, ub32 zAxisSize
#endif
	);

	~HeterogeneousMedium();

	// HOST FUNCTION //
	// Sample next point
	// Material value 0 is miss
	virtual ErrorType sampleImplementation(MediumPoint* sampledPoint,
		const void* sourcePoint, const VectorType* sapmleDirection,
		ub32 pointsNum, bool isSourcePointIsMediumPoint) const;

	virtual ErrorType setMaterial(const MediumMaterial* material, ub32 materialNum);

	virtual ErrorType getMaterialInfo(MediumPoint* sampledPoint, ub32 pointsNum) const;

	virtual ErrorType isSameBin(bool* isSameBuffer, const MediumPoint* refPoint,
		const VectorType* pos, ub32 refNum, ub32 pointsNum) const;

protected:
	virtual ErrorType attenuation(void* atteunationRes,
		const VectorType* p1, ub32 p1num,
		const MediumPoint* p2, ub32 p2num,
		bool isComplex, bool isSeparatable, bool isP1direction) const;

	HeterogeneousMedium() {};

	float_type* _xAxis;
	ub32 _xAxisSize;
	float_type* _yAxis;
	ub32 _yAxisSize;

#if DIMS==3
	float_type* _zAxis;
	ub32 _zAxisSize;
#endif
	ub32* _materialNum;

	ub32 allocationCount;
	ub32 totalValuesInBox;
	float_type _maxSigt;
};

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
HeterogeneousMedium::HeterogeneousMedium(ErrorType* err, const ub32* materialNum,
	const float_type* xAxis, ub32 xAxisSize,
	const float_type* yAxis, ub32 yAxisSize
#if DIMS==3
	, const  float_type* zAxis, ub32 zAxisSize
#endif
) : Medium(VectorType( xAxis[0] , yAxis[0]
#if DIMS==3
	, zAxis[0]
#endif
	), VectorType( xAxis[xAxisSize - 1] , yAxis[yAxisSize - 1]
#if DIMS==3
	, zAxis[zAxisSize - 1]
#endif
	))
{
	MEMORY_CHECK("heterogeneous medium allocation begin");

	allocationCount = 0;

	totalValuesInBox = (xAxisSize - 1) * (yAxisSize - 1);
#if DIMS==3
	totalValuesInBox *= (zAxisSize - 1);
#endif

	// allocate memory
	if (cudaMalloc(&_xAxis, sizeof(float_type) * xAxisSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&_yAxis, sizeof(float_type) * yAxisSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

#if DIMS==3
	if (cudaMalloc(&_zAxis, sizeof(float_type) * zAxisSize) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
#endif
	allocationCount++;

	if (cudaMalloc(&_materialNum, sizeof(ub32) * totalValuesInBox) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	_xAxisSize = xAxisSize;
	_yAxisSize = yAxisSize;
#if DIMS==3
	_zAxisSize = zAxisSize;
#endif
	_maxSigt = (float_type)0.;


	HeterogeneousMediumNS::HeterogeneousMediumDataStructre heterogeneousDataHost;

	heterogeneousDataHost.xAxis = _xAxis;
	heterogeneousDataHost.xAxisSize = xAxisSize;
	heterogeneousDataHost.yAxis = _yAxis;
	heterogeneousDataHost.yAxisSize = yAxisSize;
#if DIMS==3
	heterogeneousDataHost.zAxis = _zAxis;
	heterogeneousDataHost.zAxisSize = zAxisSize;
#endif
	heterogeneousDataHost.materialNum = _materialNum;

	// copy from host to device
	if (cudaMemcpy(_xAxis, xAxis, sizeof(float_type) * xAxisSize, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	if (cudaMemcpy(_yAxis, yAxis, sizeof(float_type) * yAxisSize, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

#if DIMS==3
	if (cudaMemcpy(_zAxis, zAxis, sizeof(float_type) * zAxisSize, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
#endif

	if (cudaMemcpy(_materialNum, materialNum, sizeof(ub32) * totalValuesInBox, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	cudaMemcpyToSymbol(HeterogeneousMediumNS::heterogeneousMediumGpuData, &heterogeneousDataHost,
		sizeof(HeterogeneousMediumNS::HeterogeneousMediumDataStructre), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	cudaMemcpyToSymbol(HeterogeneousMediumNS::maxSigt, &_maxSigt,
		sizeof(float_type), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	*err = ErrorType::NO_ERROR;

	MEMORY_CHECK("heterogeneous medium allocation end");
}

HeterogeneousMedium::~HeterogeneousMedium()
{
	MEMORY_CHECK("heterogeneous medium free begin");
	switch (allocationCount)
	{
	case 4:
		cudaFree(_materialNum);
	case 3:
#if DIMS==3
		cudaFree(_zAxis);
#endif
	case 2:
		cudaFree(_yAxis);
	case 1:
		cudaFree(_xAxis);
	default:
		break;
	}
	MEMORY_CHECK("heterogeneous medium free end");
}

// ------------------------------------ Kernels ------------------------------------ //
namespace HeterogeneousMediumNS
{
	// translate 2d / 3d index to 1d index
	__device__ ub32 materialIdx(ub32* idx)
	{
		return idx[0] + idx[1] * (heterogeneousMediumGpuData.xAxisSize - 1)
#if DIMS==3
			+ idx[2] * (heterogeneousMediumGpuData.xAxisSize - 1) * (heterogeneousMediumGpuData.yAxisSize - 1)
#endif
			;
	}

	__device__ float_type bdDistance(ib32* hitSide, VectorType boxMin, VectorType boxMax, VectorType rayOrigin, VectorType rayDirection)
	{
		VectorType boxRadius = 0.5 * (boxMax - boxMin);
		rayOrigin = rayOrigin - 0.5 * (boxMax + boxMin); // Centerlize origin

		VectorType boxInvRadius = VectorType(
			abs(boxRadius.x()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / boxRadius.x(),
			abs(boxRadius.y()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / boxRadius.y()
#if DIMS==3
			, abs(boxRadius.z()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / boxRadius.z()
#endif
		);

		VectorType invRayDirection = {
			abs(rayDirection.x()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / rayDirection.x(),
			abs(rayDirection.y()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / rayDirection.y()
#if DIMS==3
		   ,abs(rayDirection.z()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / rayDirection.z()
#endif
		};

		ib32 winding = (maxComponent(
			VectorType{ abs(rayOrigin.x()) * boxInvRadius.x() ,
				abs(rayOrigin.y()) * boxInvRadius.y()
#if DIMS==3
				,abs(rayOrigin.z()) * boxInvRadius.z()
#endif
			}) <= 1.0) ? -1 : 1;



		ib32 sgn[DIMS] = { -(rayDirection.x() < 0. ? -1 : 1), -(rayDirection.y() < 0. ? -1 : 1)
#if DIMS==3
			, -(rayDirection.z() < 0. ? -1 : 1)
#endif
		};

		VectorType distanceToPlain = VectorType(
			(invRayDirection.x()) * (boxRadius.x() * (winding * sgn[0]) - rayOrigin.x()),
			(invRayDirection.y()) * (boxRadius.y() * (winding * sgn[1]) - rayOrigin.y())
#if DIMS==3
		   ,(invRayDirection.z()) * (boxRadius.z() * (winding * sgn[2]) - rayOrigin.z())
#endif
		);

		bool test[DIMS];

#if DIMS==2
		test[0] = (distanceToPlain.x() >= 0.0) &&
			(abs(fma(rayDirection.y(), distanceToPlain.x(), rayOrigin.y())) < boxRadius.y());

		test[1] = (distanceToPlain.y() >= 0.0) &&
			(abs(fma(rayDirection.x(), distanceToPlain.y(), rayOrigin.x())) < boxRadius.x());

		if (test[0] > 0)
		{
			sgn[1] = 0;
		}
		else if (test[1] > 0)
		{
			sgn[0] = 0;
		}
		else
		{
			sgn[0] = 0;
			sgn[1] = 0;
		}
#else
		test[0] = (distanceToPlain.x() >= 0.0) &&
			(abs(fma(rayDirection.y(), distanceToPlain.x(), rayOrigin.y())) < boxRadius.y()) &&
			(abs(fma(rayDirection.z(), distanceToPlain.x(), rayOrigin.z())) < boxRadius.z());

		test[1] = (distanceToPlain.y() >= 0.0) &&
			(abs(fma(rayDirection.x(), distanceToPlain.y(), rayOrigin.x())) < boxRadius.x()) &&
			(abs(fma(rayDirection.z(), distanceToPlain.y(), rayOrigin.z())) < boxRadius.z());

		test[2] = (distanceToPlain.z() >= 0.0) &&
			(abs(fma(rayDirection.x(), distanceToPlain.z(), rayOrigin.x())) < boxRadius.x()) &&
			(abs(fma(rayDirection.y(), distanceToPlain.z(), rayOrigin.y())) < boxRadius.y());

		if (test[0] > 0)
		{
			sgn[1] = 0;
			sgn[2] = 0;
		}
		else if (test[1] > 0)
		{
			sgn[0] = 0;
			sgn[2] = 0;
		}
		else if (test[2] > 0)
		{
			sgn[0] = 0;
			sgn[1] = 0;
		}
		else
		{
			sgn[0] = 0;
			sgn[1] = 0;
			sgn[2] = 0;
		}
#endif

		hitSide[0] = sgn[0] * winding;
		hitSide[1] = sgn[1] * winding;
#if DIMS==3
		hitSide[2] = sgn[2] * winding;
#endif

#if DIMS==2
		return test[0] ? distanceToPlain.x() : (test[1] ? distanceToPlain.y() : 0);
#else
		return test[0] ? distanceToPlain.x() : (test[1] ? distanceToPlain.y() : (test[2] ? distanceToPlain.z() : 0));
#endif
	}

	// TODO: Kernel is very large, try to split it to sereval calls
	template <bool isComplex, bool isSeparatable, bool isP1direction>
	__global__ void heterogeneousAttenuationKernel(void* atteunationRes,
		const VectorType* p1, ub32 p1num,
		const MediumPoint* p2, ub32 p2num)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 p1idx, p2idx;

		if (isSeparatable)
		{
			p1idx = threadNum % p1num;
			p2idx = threadNum / p1num;
		}
		else
		{
			p1idx = threadNum;
			p2idx = threadNum;
		}

		if (p2idx < p2num)
		{
			VectorType rayDirection;
			if (isP1direction)
			{
				rayDirection = p1[p1idx];
			}
			else
			{
				rayDirection = normalize(p1[p1idx] - p2[p2idx].position);
			}
			VectorType p2Current = p2[p2idx].position;
			float_type d_sigt = (float_type)0.;

			ib32 hitSide[DIMS];
			ub32 currentBin[DIMS];

			// DEBUG
			//float_type bd_homo = bdDistance(hitSide, MediumNS::boxMin, MediumNS::boxMax, p2[p2idx], rayDirection);
			//float_type bd_hetro = (float_type)0.;
			// END DEBUG

			if (MediumNS::isInsideKernel(p2Current))
			{
				VectorType invRayDirection = {
					abs(rayDirection.x()) < EPSILON ? (rayDirection.x() >= 0 ? (float_type)1e12 : (float_type)-1e12) : (float_type)1.0 / rayDirection.x(),
					abs(rayDirection.y()) < EPSILON ? (rayDirection.y() >= 0 ? (float_type)1e12 : (float_type)-1e12) : (float_type)1.0 / rayDirection.y()
#if DIMS==3
					,abs(rayDirection.z()) < EPSILON ? (rayDirection.z() >= 0 ? (float_type)1e12 : (float_type)-1e12) : (float_type)1.0 / rayDirection.z()
#endif
				};
				ib32 winding = -1;

				ib32 sgn[DIMS] = { -(rayDirection.x() < 0. ? -1 : 1), -(rayDirection.y() < 0. ? -1 : 1)
#if DIMS==3
					, -(rayDirection.z() < 0. ? -1 : 1)
#endif
				};

				currentBin[0] = binarySearchKernel(heterogeneousMediumGpuData.xAxis, heterogeneousMediumGpuData.xAxisSize, p2Current.x());
				currentBin[1] = binarySearchKernel(heterogeneousMediumGpuData.yAxis, heterogeneousMediumGpuData.yAxisSize, p2Current.y());
#if DIMS==3
				currentBin[2] = binarySearchKernel(heterogeneousMediumGpuData.zAxis, heterogeneousMediumGpuData.zAxisSize, p2Current.z());
#endif
				MediumMaterial* currentMaterialData = MediumNS::mediumGpuData + heterogeneousMediumGpuData.materialNum[materialIdx(currentBin)];
				float_type currentSigt = currentMaterialData->sigs + currentMaterialData->siga;

				while (true)
				{
					VectorType boxMin = VectorType( heterogeneousMediumGpuData.xAxis[currentBin[0]]
						, heterogeneousMediumGpuData.yAxis[currentBin[1]]
#if DIMS==3
						, heterogeneousMediumGpuData.zAxis[currentBin[2]]
#endif
					);

					VectorType boxMax = VectorType( heterogeneousMediumGpuData.xAxis[currentBin[0] + 1]
						, heterogeneousMediumGpuData.yAxis[currentBin[1] + 1]
#if DIMS==3
						, heterogeneousMediumGpuData.zAxis[currentBin[2] + 1]
#endif
					);

					// float_type bd = bdDistance(hitSide, boxMin, boxMax, p2Current, rayDirection, true);

					VectorType boxRadius = 0.5 * (boxMax - boxMin);
					VectorType rayOrigin = p2Current - 0.5 * (boxMax + boxMin); // Centerlize origin

					VectorType distanceToPlain = VectorType(
						(invRayDirection.x()) * (boxRadius.x() * (winding * sgn[0]) - rayOrigin.x()),
						(invRayDirection.y()) * (boxRadius.y() * (winding * sgn[1]) - rayOrigin.y())
#if DIMS==3
					   ,(invRayDirection.z()) * (boxRadius.z() * (winding * sgn[2]) - rayOrigin.z())
#endif
					);

#if DIMS==2
					float_type bd;
					if (distanceToPlain.x() < distanceToPlain.y())
					{
						bd = distanceToPlain.x();
						hitSide[0] = sgn[0] * winding;
						hitSide[1] = 0;
					}
					else
					{
						bd = distanceToPlain.y();
						hitSide[0] = 0;
						hitSide[1] = sgn[1] * winding;
					}
#else
					float_type bd;
					if (distanceToPlain.x() < distanceToPlain.y() && distanceToPlain.x() < distanceToPlain.z())
					{
						bd = distanceToPlain.x();
						hitSide[0] = sgn[0] * winding;
						hitSide[1] = 0;
						hitSide[2] = 0;
					}
					else if(distanceToPlain.y() < distanceToPlain.z())
					{
						bd = distanceToPlain.y();
						hitSide[0] = 0;
						hitSide[1] = sgn[1] * winding;
						hitSide[2] = 0;
					}
					else
					{
						bd = distanceToPlain.z();
						hitSide[0] = 0;
						hitSide[1] = 0;
						hitSide[2] = sgn[2] * winding;
					}
#endif

					// DEBUG
					//bd_hetro += bd;
					// END DEBUG

					d_sigt += bd * currentSigt;

					// Check if hit the edge
					// xAxisSize is number of edges
					// xAxisSize - 1 is number of cells
					// xAxisSize - 2 is last cell idx
					if ((currentBin[0] == 0 && hitSide[0] == -1) || (currentBin[0] == (heterogeneousMediumGpuData.xAxisSize - 2) && hitSide[0] == 1) ||
						(currentBin[1] == 0 && hitSide[1] == -1) || (currentBin[1] == (heterogeneousMediumGpuData.yAxisSize - 2) && hitSide[1] == 1)
#if DIMS==3
						|| (currentBin[2] == 0 && hitSide[2] == -1) || (currentBin[2] == (heterogeneousMediumGpuData.zAxisSize - 2) && hitSide[2] == 1)
#endif
						)
					{
						break;
					}
					
					p2Current = p2Current + bd * rayDirection;

					currentBin[0] = (ub32)(currentBin[0] + hitSide[0]);
					currentBin[1] = (ub32)(currentBin[1] + hitSide[1]);
#if DIMS==3
					currentBin[2] = (ub32)(currentBin[2] + hitSide[2]);
#endif
					currentMaterialData = MediumNS::mediumGpuData + heterogeneousMediumGpuData.materialNum[materialIdx(currentBin)];
					currentSigt = currentMaterialData->sigs + currentMaterialData->siga;
				}
			}

			//if (MediumNS::isInsideKernel(p2[p2idx]) && abs(bd_hetro - bd_homo) > 10 * EPSILON)
			//{
			//	printf("%d: bd hetro: %f, bd homo: %f, p2 = %f %f, dir = %f %f \n",
			//		threadNum, bd_hetro, bd_homo, p2[p2idx].x, p2[p2idx].y, rayDirection.x, rayDirection.y);
			//}

			//printf("attenuation: %d, p1 = [%f %f %f], p2 = [%f %f %f], d_sigt = %f \n", threadNum,
			//	p1[p1idx].x, p1[p1idx].y, p1[p1idx].z, p2[p2idx].position.x, p2[p2idx].position.y, p2[p2idx].position.z, d_sigt);

			if (isComplex)
			{
				ComplexType* atteunationResPtr = (ComplexType*)atteunationRes;
				atteunationResPtr[threadNum] = exp(-d_sigt);
			}
			else
			{
				float_type* atteunationResPtr = (float_type*)atteunationRes;
				atteunationResPtr[threadNum] = exp(-d_sigt);
			}
		}
	}

	// Woodcock sampling
	template <bool isSourcePointIsMediumPoint>
	__global__ void heterogeneousSampleKernel(MediumPoint* sampledPoint,
		const void* sourcePoint, const VectorType* sapmleDirection, ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		//materialIdx
		if (threadNum < pointsNum)
		{
			VectorType rayOrigin;
			if (isSourcePointIsMediumPoint)
			{
				const MediumPoint* currentMediumPoint = ((const MediumPoint*)sourcePoint) + threadNum;
				rayOrigin = currentMediumPoint->position;

				// inactive path
				if (currentMediumPoint->material < 0)
				{
					sampledPoint[threadNum].position = rayOrigin;
					sampledPoint[threadNum].material = currentMediumPoint->material;
					return;
				}
			}
			else
			{
				rayOrigin = ((const VectorType*)sourcePoint)[threadNum];
			}
			VectorType rayDirection = sapmleDirection[threadNum];
			VectorType currentSampled;
			float_type d = (float_type)0.;
			ib32 hitSide[DIMS];
			ub32 currentSampledIdx[DIMS];
			ub32 currentMaterial;

			while (true)
			{
				d += -log(-randUniform(statePool + threadNum) + 1) / maxSigt;

				currentSampled = rayOrigin + d * rayDirection;


				if (!MediumNS::isInsideKernel(currentSampled))
				{

					// if outside volume
					// check if pointing towards the volume
					bdDistance(hitSide, MediumNS::boxMin, MediumNS::boxMax, currentSampled, rayDirection);

					if (hitSide[0] == 0 && hitSide[1] == 0
#if DIMS==3
						&& hitSide[2] == 0
#endif
						)
					{
						// if ray is not pointing towards the medium, we have finished
						break;
					}
				}
				else
				{

					// if inside the volume
					// find where the current point is located
					currentSampledIdx[0] = binarySearchKernel(heterogeneousMediumGpuData.xAxis, heterogeneousMediumGpuData.xAxisSize, currentSampled.x());
					currentSampledIdx[1] = binarySearchKernel(heterogeneousMediumGpuData.yAxis, heterogeneousMediumGpuData.yAxisSize, currentSampled.y());
#if DIMS==3
					currentSampledIdx[2] = binarySearchKernel(heterogeneousMediumGpuData.zAxis, heterogeneousMediumGpuData.zAxisSize, currentSampled.z());
#endif

					// get the current material
					currentMaterial = heterogeneousMediumGpuData.materialNum[materialIdx(currentSampledIdx)];

					// check if finish sampling
					if (randUniform(statePool + threadNum) <= ((MediumNS::mediumGpuData[currentMaterial].sigs + MediumNS::mediumGpuData[currentMaterial].siga) / maxSigt))
					{
						break;
					}
				}
			}

			// update after finish sampling
			sampledPoint[threadNum].position = currentSampled;
			// materialNum[threadNum].value = (MediumNS::isInsideToleranceKernel(currentSampled, EPSILON)) ? currentMaterial : 0;

			if (MediumNS::isInsideKernel(currentSampled))
			{
				// If inside, we still need to check if we finish the path due to attenuation
				if (randUniform(statePool + threadNum) > (MediumNS::mediumGpuData[currentMaterial].sigs / (MediumNS::mediumGpuData[currentMaterial].sigs + MediumNS::mediumGpuData[currentMaterial].siga)))
				{
					sampledPoint[threadNum].material = 0;
				}
				else
				{
					sampledPoint[threadNum].material = currentMaterial;
				}
			}
			else
			{
				sampledPoint[threadNum].material = 0;
			}

			// printf("Sampled Point %d: [%f %f %f], material: %d \n", threadNum, currentSampled.x, currentSampled.y, currentSampled.z, currentMaterial);
		}
	}
		

	__global__ void getMaterialKernel(MediumPoint* sampledPoint, ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < pointsNum)
		{
			VectorType currentSampled = sampledPoint[threadNum].position;

			ub32 currentSampledIdx[DIMS];

			currentSampledIdx[0] = binarySearchKernel(heterogeneousMediumGpuData.xAxis, heterogeneousMediumGpuData.xAxisSize, currentSampled.x());
			currentSampledIdx[1] = binarySearchKernel(heterogeneousMediumGpuData.yAxis, heterogeneousMediumGpuData.yAxisSize, currentSampled.y());
#if DIMS==3
			currentSampledIdx[2] = binarySearchKernel(heterogeneousMediumGpuData.zAxis, heterogeneousMediumGpuData.zAxisSize, currentSampled.z());
#endif

			if (MediumNS::isInsideKernel(currentSampled))
			{
				sampledPoint[threadNum].material = heterogeneousMediumGpuData.materialNum[materialIdx(currentSampledIdx)];
			}
			else
			{
				sampledPoint[threadNum].material = 0;
			}
		}
	}

	__global__ void isSameBinKernel(bool* isSameBuffer, const MediumPoint* refPoint,
		const VectorType* pos, ub32 refNum, ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 temporalNum = threadNum % pointsNum;
		ub32 pathNum = threadNum / pointsNum;

		if (pathNum < refNum)
		{
			ub32 refBin[DIMS], pBin[DIMS];
			refBin[0] = binarySearchKernel(heterogeneousMediumGpuData.xAxis, heterogeneousMediumGpuData.xAxisSize, refPoint[pathNum].position.x());
			refBin[1] = binarySearchKernel(heterogeneousMediumGpuData.yAxis, heterogeneousMediumGpuData.yAxisSize, refPoint[pathNum].position.y());
#if DIMS==3
			refBin[2] = binarySearchKernel(heterogeneousMediumGpuData.zAxis, heterogeneousMediumGpuData.zAxisSize, refPoint[pathNum].position.z());
#endif

			pBin[0] = binarySearchKernel(heterogeneousMediumGpuData.xAxis, heterogeneousMediumGpuData.xAxisSize, pos[threadNum].x());
			pBin[1] = binarySearchKernel(heterogeneousMediumGpuData.yAxis, heterogeneousMediumGpuData.yAxisSize, pos[threadNum].y());
#if DIMS==3
			pBin[2] = binarySearchKernel(heterogeneousMediumGpuData.zAxis, heterogeneousMediumGpuData.zAxisSize, pos[threadNum].z());
#endif
			isSameBuffer[threadNum] = (
				(refBin[0] == pBin[0]) &&
				(refBin[1] == pBin[1])
#if DIMS==3
				 && (refBin[2] == pBin[2])
#endif
				) && MediumNS::isInsideKernel(pos[threadNum]);
		}
	}
}

// ------------------------------------ Class Function Implementations ------------------------------------ //
ErrorType HeterogeneousMedium::attenuation(void* atteunationRes,
	const VectorType* p1, ub32 p1num,
	const MediumPoint* p2, ub32 p2num,
	bool isComplex, bool isSeparatable, bool isP1direction) const
{

	ub32 totalThreads;
	if (isSeparatable)
	{
		totalThreads = p1num * p2num;
	}
	else
	{
		totalThreads = p1num;
	}

	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	if(isComplex == true && isSeparatable == true && isP1direction == true)
		HeterogeneousMediumNS::heterogeneousAttenuationKernel<true, true, true>    <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == true && isSeparatable == true && isP1direction == false)
		HeterogeneousMediumNS::heterogeneousAttenuationKernel<true, true, false>   <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == true && isSeparatable == false && isP1direction == true)
		HeterogeneousMediumNS::heterogeneousAttenuationKernel<true, false, true>   <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == true && isSeparatable == false && isP1direction == false)
		HeterogeneousMediumNS::heterogeneousAttenuationKernel<true, false, false>  <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == false && isSeparatable == true && isP1direction == true)
		HeterogeneousMediumNS::heterogeneousAttenuationKernel<false, true, true>   <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == false && isSeparatable == true && isP1direction == false)
		HeterogeneousMediumNS::heterogeneousAttenuationKernel<false, true, false>  <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == false && isSeparatable == false && isP1direction == true)
		HeterogeneousMediumNS::heterogeneousAttenuationKernel<false, false, true>  <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == false && isSeparatable == false && isP1direction == false)
		HeterogeneousMediumNS::heterogeneousAttenuationKernel<false, false, false> <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_HetroMedium_heterogeneousAttenuationKernel;
	}
	
	return ErrorType::NO_ERROR;
}

ErrorType HeterogeneousMedium::sampleImplementation(MediumPoint* sampledPoint,
	const void* sourcePoint, const VectorType* sapmleDirection,
	ub32 pointsNum, bool isSourcePointIsMediumPoint) const
{
	ub32 threadsNum = pointsNum < THREADS_NUM ? pointsNum : THREADS_NUM;
	ub32 blocksNum = (pointsNum - 1) / THREADS_NUM + 1;

	if (isSourcePointIsMediumPoint == false)
	{
		HeterogeneousMediumNS::heterogeneousSampleKernel<false> <<< blocksNum, threadsNum >>> (sampledPoint,
			sourcePoint, sapmleDirection, pointsNum);
	}
	else
	{
		HeterogeneousMediumNS::heterogeneousSampleKernel<true> <<< blocksNum, threadsNum >>> (sampledPoint,
			sourcePoint, sapmleDirection, pointsNum);
	}

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_HetroMedium_heterogeneousSampleKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType HeterogeneousMedium::setMaterial(const MediumMaterial* material, ub32 materialNum)
{
	// set the material
	ErrorType err = Medium::setMaterial(material, materialNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// calculate max sigt
	_maxSigt = (float_type)0.;
	MediumMaterial currentMaterial;

	for (ub32 matNum = 0; matNum < MATERIAL_NUM; matNum++)
	{
		cudaMemcpyFromSymbol(&currentMaterial, MediumNS::mediumGpuData, sizeof(MediumMaterial), sizeof(MediumMaterial) * matNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::ALLOCATION_ERROR;
		}

		if (_maxSigt < (currentMaterial.siga + currentMaterial.sigs))
		{
			_maxSigt = currentMaterial.siga + currentMaterial.sigs;
		}
	}

	cudaMemcpyToSymbol(HeterogeneousMediumNS::maxSigt, &_maxSigt,
		sizeof(float_type), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	return ErrorType::NO_ERROR;
}

ErrorType HeterogeneousMedium::getMaterialInfo(MediumPoint* sampledPoint, ub32 pointsNum) const
{
	ub32 threadsNum = pointsNum < THREADS_NUM ? pointsNum : THREADS_NUM;
	ub32 blocksNum = (pointsNum - 1) / THREADS_NUM + 1;

	HeterogeneousMediumNS::getMaterialKernel <<< blocksNum, threadsNum >>> (sampledPoint, pointsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_HetroMedium_getMaterialKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType HeterogeneousMedium::isSameBin(bool* isSameBuffer, const MediumPoint* refPoint,
	const VectorType* pos, ub32 refNum, ub32 pointsNum) const
{
	ub32 totalThreads = refNum * pointsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	HeterogeneousMediumNS::isSameBinKernel <<< blocksNum, threadsNum >>> (isSameBuffer, refPoint, pos,
		refNum, pointsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::DEVICE_ERROR;
	}

	return ErrorType::NO_ERROR;
}
