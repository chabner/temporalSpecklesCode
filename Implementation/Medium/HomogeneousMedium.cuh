#pragma once

#include "../../Interface/MediumInterface.cuh"

// box intersection algorithm: https://jcgt.org/published/0007/03/04/
// A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering

// ------------------------------------ GPU Constants ------------------------------------ //
namespace HomogeneousMediumNS
{
	// Homogeneous medium structre
	typedef struct
	{
		VectorType boxRadius;
		VectorType boxInvRadius;
		VectorType boxCenter;

	} HomogeneousMediumDataStructre;

	__constant__ HomogeneousMediumDataStructre homogeneousMediumGpuData;
}

// ------------------------------------ Homogeneous Medium Class ------------------------------------ //

class HomogeneousMedium: public Medium
{
public:

	HomogeneousMedium(ErrorType* err, VectorType boxMin, VectorType boxMax);

	// HOST FUNCTION //
	// Sample next point
	// Material value 0 is miss
	virtual ErrorType sampleImplementation(MediumPoint* sampledPoint,
		const void* sourcePoint, const VectorType* sapmleDirection,
		ub32 pointsNum, bool isSourcePointIsMediumPoint) const;

	virtual ErrorType getMaterialInfo(MediumPoint* sampledPoint, ub32 pointsNum) const;

protected:
	virtual ErrorType attenuation(void* atteunationRes,
		const VectorType* p1, ub32 p1num,
		const MediumPoint* p2, ub32 p2num,
		bool isComplex, bool isSeparatable, bool isP1direction) const;

	HomogeneousMedium() {};
};

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
HomogeneousMedium::HomogeneousMedium(ErrorType *err, VectorType boxMin, VectorType boxMax): Medium(boxMin, boxMax)
{
	MEMORY_CHECK("homogeneous medium allocation begin");

	HomogeneousMediumNS::HomogeneousMediumDataStructre homogeneousDataHost;

	homogeneousDataHost.boxCenter = 0.5 * (boxMax + boxMin);
	homogeneousDataHost.boxRadius = 0.5 * (boxMax - boxMin);

#if DIMS==2
	homogeneousDataHost.boxInvRadius = (
		abs(homogeneousDataHost.boxRadius.x()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / homogeneousDataHost.boxRadius.x(),
		abs(homogeneousDataHost.boxRadius.y()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / homogeneousDataHost.boxRadius.y() );
#else
	homogeneousDataHost.boxInvRadius = VectorType(
		abs(homogeneousDataHost.boxRadius.x()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / homogeneousDataHost.boxRadius.x(),
		abs(homogeneousDataHost.boxRadius.y()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / homogeneousDataHost.boxRadius.y(),
		abs(homogeneousDataHost.boxRadius.z()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / homogeneousDataHost.boxRadius.z() );
#endif

	cudaMemcpyToSymbol(HomogeneousMediumNS::homogeneousMediumGpuData, &homogeneousDataHost,
		sizeof(HomogeneousMediumNS::HomogeneousMediumDataStructre), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	*err = ErrorType::NO_ERROR;

	MEMORY_CHECK("homogeneous medium allocation end");
}

// ------------------------------------ Kernels ------------------------------------ //
namespace HomogeneousMediumNS
{
	__device__ float_type bdDistance(VectorType rayOrigin, VectorType rayDirection)
	{
		rayOrigin = rayOrigin - homogeneousMediumGpuData.boxCenter;
		VectorType invRayDirection = VectorType(
			abs(rayDirection.x()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / rayDirection.x(),
			abs(rayDirection.y()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / rayDirection.y()
#if DIMS==3
		   ,abs(rayDirection.z()) < EPSILON ? (float_type)1e12 : (float_type)1.0 / rayDirection.z()
#endif
		);

		ib32 winding = (maxComponent(
			VectorType( abs(rayOrigin.x()) * homogeneousMediumGpuData.boxInvRadius.x(),
						 abs(rayOrigin.y()) * homogeneousMediumGpuData.boxInvRadius.y()
#if DIMS==3
						,abs(rayOrigin.z()) * homogeneousMediumGpuData.boxInvRadius.z()
#endif
			)) < 1.0) ? -1 : 1;

		ib32 sgn[DIMS] = { -(rayDirection.x() < 0. ? -1 : 1), -(rayDirection.y() < 0. ? -1 : 1)
#if DIMS==3
			, -(rayDirection.z() < 0. ? -1 : 1)
#endif
		};

		VectorType distanceToPlain = VectorType(
			(invRayDirection.x()) * (homogeneousMediumGpuData.boxRadius.x() * (winding * sgn[0]) - rayOrigin.x()),
			(invRayDirection.y()) * (homogeneousMediumGpuData.boxRadius.y() * (winding * sgn[1]) - rayOrigin.y())
#if DIMS==3
		   ,(invRayDirection.z()) * (homogeneousMediumGpuData.boxRadius.z() * (winding * sgn[2]) - rayOrigin.z())
#endif
		);

		bool test[DIMS];

#if DIMS==2
		test[0] = (distanceToPlain.x() >= 0.0) &&
			(abs(fma(rayDirection.y(), distanceToPlain.x(), rayOrigin.y())) < homogeneousMediumGpuData.boxRadius.y());

		test[1] = (distanceToPlain.y() >= 0.0) &&
			(abs(fma(rayDirection.x(), distanceToPlain.y(), rayOrigin.x())) < homogeneousMediumGpuData.boxRadius.x());

		return test[0] ? distanceToPlain.x() : (test[1] ? distanceToPlain.y() : 0);
#else
		test[0] = (distanceToPlain.x() >= 0.0) &&
			(abs(fma(rayDirection.y(), distanceToPlain.x(), rayOrigin.y())) < homogeneousMediumGpuData.boxRadius.y()) &&
			(abs(fma(rayDirection.z(), distanceToPlain.x(), rayOrigin.z())) < homogeneousMediumGpuData.boxRadius.z());

		test[1] = (distanceToPlain.y() >= 0.0) &&
			(abs(fma(rayDirection.x(), distanceToPlain.y(), rayOrigin.x())) < homogeneousMediumGpuData.boxRadius.x()) &&
			(abs(fma(rayDirection.z(), distanceToPlain.y(), rayOrigin.z())) < homogeneousMediumGpuData.boxRadius.z());

		test[2] = (distanceToPlain.z() >= 0.0) &&
			(abs(fma(rayDirection.x(), distanceToPlain.z(), rayOrigin.x())) < homogeneousMediumGpuData.boxRadius.x()) &&
			(abs(fma(rayDirection.y(), distanceToPlain.z(), rayOrigin.y())) < homogeneousMediumGpuData.boxRadius.y());

		return test[0] ? distanceToPlain.x() : (test[1] ? distanceToPlain.y() : (test[2] ? distanceToPlain.z() : 0));
#endif
	}

	template <bool isComplex, bool isSeparatable, bool isP1direction>
	__global__ void homogeneousAttenuationKernel(void* atteunationRes,
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
			if(isP1direction)
			{
				rayDirection = p1[p1idx];
			}
			else
			{
                rayDirection = normalize(p1[p1idx] - p2[p2idx].position);
            }

			float_type d = bdDistance(p2[p2idx].position, rayDirection);
			if (isComplex)
			{
				ComplexType* atteunationResPtr = (ComplexType*)atteunationRes;
				atteunationResPtr[threadNum] = exp(-d * (MediumNS::mediumGpuData[1].sigs + MediumNS::mediumGpuData[1].siga));
			}
			else
			{
				float_type* atteunationResPtr = (float_type*)atteunationRes;
				atteunationResPtr[threadNum] = exp(-d * (MediumNS::mediumGpuData[1].sigs + MediumNS::mediumGpuData[1].siga));
			}

			/*
			printf("attenuation %d: bd: %f rayDirection: [%f %f %f]. point: [%f %f %f]. \n", threadNum,
				d, rayDirection.x, rayDirection.y, rayDirection.z, p2[p2idx].position.x, p2[p2idx].position.y, p2[p2idx].position.z);
				*/
		}
	}

	template <bool isSourcePointIsMediumPoint>
	__global__ void homogeneousSampleKernel(MediumPoint* sampledPoint,
		const void* sourcePoint, const VectorType* sapmleDirection, ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

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
			float_type d = (float_type) 0.;

			// if origin inside, just sample next step
			if (MediumNS::isInsideKernel(rayOrigin))
			{
				d = -log(-randUniform(statePool + threadNum) + 1) / (MediumNS::mediumGpuData[1].sigs + MediumNS::mediumGpuData[1].siga);
			}
			else
			{
				float_type dmin;

				// check minimum distance to hit the volume
				// in case of missing dmin will be 0
				dmin = bdDistance(rayOrigin, rayDirection);

				// trace until hit the volume

				while (d < dmin)
				{
					d += -log(-randUniform(statePool + threadNum) + 1) / (MediumNS::mediumGpuData[1].sigs + MediumNS::mediumGpuData[1].siga);
				}
			}

			VectorType currentSampledPoint = rayOrigin + d * rayDirection;
			sampledPoint[threadNum].position = currentSampledPoint;

			if (MediumNS::isInsideToleranceKernel(currentSampledPoint, EPSILON))
			{
				// If inside, we still need to check if we finish the path due to attenuation
				if (randUniform(statePool + threadNum) > (MediumNS::mediumGpuData[1].sigs / (MediumNS::mediumGpuData[1].sigs + MediumNS::mediumGpuData[1].siga)))
				{
					sampledPoint[threadNum].material = 0;
				}
				else
				{
					sampledPoint[threadNum].material = 1;
				}
			}
			else
			{
				sampledPoint[threadNum].material = 0;
			}
			
			/*
			printf("sample %d: Sampled point: [%f %f %f], material: %d \n",
				threadNum, sampledPoint[threadNum].position.x, sampledPoint[threadNum].position.y, sampledPoint[threadNum].position.z,
				sampledPoint[threadNum].material);
				*/
		}
	}

	__global__ void getMaterialKernel(MediumPoint* sampledPoint, ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < pointsNum)
		{
			sampledPoint[threadNum].material = MediumNS::isInsideKernel(sampledPoint[threadNum].position) ? 1 : 0;
		}
	}
}

// ------------------------------------ Class Function Implementations ------------------------------------ //
ErrorType HomogeneousMedium::attenuation(void* atteunationRes,
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
		HomogeneousMediumNS::homogeneousAttenuationKernel<true, true, true>    <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == true && isSeparatable == true && isP1direction == false)
		HomogeneousMediumNS::homogeneousAttenuationKernel<true, true, false>   <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == true && isSeparatable == false && isP1direction == true)
		HomogeneousMediumNS::homogeneousAttenuationKernel<true, false, true>   <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == true && isSeparatable == false && isP1direction == false)
		HomogeneousMediumNS::homogeneousAttenuationKernel<true, false, false>  <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == false && isSeparatable == true && isP1direction == true)
		HomogeneousMediumNS::homogeneousAttenuationKernel<false, true, true>   <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == false && isSeparatable == true && isP1direction == false)
		HomogeneousMediumNS::homogeneousAttenuationKernel<false, true, false>  <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == false && isSeparatable == false && isP1direction == true)
		HomogeneousMediumNS::homogeneousAttenuationKernel<false, false, true>  <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	else if (isComplex == false && isSeparatable == false && isP1direction == false)
		HomogeneousMediumNS::homogeneousAttenuationKernel<false, false, false> <<< blocksNum, threadsNum >>> (atteunationRes, p1, p1num, p2, p2num);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

ErrorType HomogeneousMedium::sampleImplementation(MediumPoint* sampledPoint,
	const void* sourcePoint, const VectorType* sapmleDirection,
	ub32 pointsNum, bool isSourcePointIsMediumPoint) const
{
	ub32 threadsNum = pointsNum < THREADS_NUM ? pointsNum : THREADS_NUM;
	ub32 blocksNum = (pointsNum - 1) / THREADS_NUM + 1;

	if (isSourcePointIsMediumPoint == false)
	{
		HomogeneousMediumNS::homogeneousSampleKernel<false> <<< blocksNum, threadsNum >>> (sampledPoint,
			sourcePoint, sapmleDirection, pointsNum);
	}
	else
	{
		HomogeneousMediumNS::homogeneousSampleKernel<true> <<< blocksNum, threadsNum >>> (sampledPoint,
			sourcePoint, sapmleDirection, pointsNum);
	}

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}

ErrorType HomogeneousMedium::getMaterialInfo(MediumPoint* sampledPoint, ub32 pointsNum) const
{
	ub32 threadsNum = pointsNum < THREADS_NUM ? pointsNum : THREADS_NUM;
	ub32 blocksNum = (pointsNum - 1) / THREADS_NUM + 1;

	HomogeneousMediumNS::getMaterialKernel <<< blocksNum, threadsNum >>> (sampledPoint, pointsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
}
