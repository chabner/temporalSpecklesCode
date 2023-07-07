#pragma once

#define MAX_HG_ESTIMATION_3D_M (float_type)5.0

#include "../Simulation.cuh"
#include "../SpecialFunctions.cuh"
#include "../VectorType.cuh"

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>

// ------------------------------------ GPU Constants ------------------------------------ //
namespace ScatteringNS
{
	enum ScatteringType {
		NOT_DEFINED,
		ISOTROPIC,
		TABULAR,
		HENYEY_GREENSTEIN,
		VMF_MIXTURE
	};
	__constant__ ScatteringType scatteringTypesList[MATERIAL_NUM];

	// parametrs for each scattering type
	typedef void* ScatteringParmetersType;
	__constant__ ScatteringParmetersType scatteringParameters[MATERIAL_NUM];
}

// ------------------------------------ Scattering Class ------------------------------------ //
class Scattering
{
public:
	Scattering();
	~Scattering();

	// Add scatterer type:
	ErrorType setIsotropicScattering(ub32 materialNum);
	ErrorType setTabularScattering(const ComplexType* f, ub32 entriesNum, ub32 materialNum); // f is a CPU pointer
	ErrorType setHenyeyGreensteinScattering(float_type g, ub32 materialNum);
	ErrorType setVonMisesFisherScattering(const float_type* mixtureMu, const float_type* mixtureC, const float_type* mixtureAlpha, ub32 mixturesNum, ub32 materialNum); // all CPU pointers

	// Remove scatterer
	ErrorType remove(ub32 materialNum);

	// Declerations:
	
	// Return the scattering amplitude of points p1 -> p2 -> p3, where the scattering function defined by scatteresList.
	// All are device pointers in size of pointsNum.
	// The scattering function is defined according to p2 position inside the material.
	ErrorType amplitude(ComplexType* amplitudeRes,
		const VectorType* p1, bool isP1Direction,
		const MediumPoint* p2,
		const VectorType* p3, bool isP3Direction, ub32 p1p2p3num) const;

	ErrorType amplitude(ComplexType* amplitudeRes,
		const VectorType* p1, bool isP1Direction, ub32 p1num,
		const MediumPoint* p2,
		const VectorType* p3, bool isP3Direction, ub32 p2p3num) const;

	ErrorType amplitude(ComplexType* amplitudeRes,
		const VectorType* p1, bool isP1Direction, ub32 p1num,
		const MediumPoint* p2, ub32 p2num,
		const VectorType* p3, bool isP3Direction, ub32 p3num) const;

	// Return the scattering pdf of points p1 -> p2 -> p3, where the scattering function defined by scatteresList.
	// The resulted pdf is multiplied to pdf.
	// All are device pointers in size of pointsNum.
	// The scattering function is defined according to p2 position inside the material.
	ErrorType multiplyPdf(float_type* pdfRes,
		const VectorType* p1,
		const MediumPoint* p2,
		const VectorType* p3, ub32 p1p2p3num) const;

	ErrorType multiplyPdf(float_type* pdfRes,
		const VectorType* p1, ub32 p1num,
		const MediumPoint* p2,
		const VectorType* p3, ub32 p2p3num) const;

	ErrorType multiplyPdfDirection(float_type* pdfRes,
		const VectorType* direction,
		const MediumPoint* p2,
		const VectorType* p3, ub32 p1p2p3num) const;

	ErrorType multiplyPdfDirection(float_type* pdfRes,
		const VectorType* direction, ub32 directionNum,
		const MediumPoint* p2,
		const VectorType* p3, ub32 p2p3num) const;

	// Sample new scattering direction
	ErrorType newDirection(VectorType* sampledDirection,
		const VectorType* p1, const MediumPoint* p2, ub32 pointsNum) const;

	ScatteringNS::ScatteringType getScatteringType(ub32 materialNum) const { return scatteringTypesListCPU[materialNum]; };

	// Check if the scatterer is isotropic - including g = 0 HG or kappa = 0 vMF
 	bool isIsotropicScattering (ub32 materialNum) const;

	// Check if all scatterers are isotropic
	bool isIsotropicScattering() const { return isIsotropicScatteringBool; };

private:
	ScatteringNS::ScatteringType scatteringTypesListCPU[MATERIAL_NUM];
	ScatteringNS::ScatteringParmetersType scatteringParametersCPU[MATERIAL_NUM];
	ScatteringNS::ScatteringParmetersType scatteringParametersGPU[MATERIAL_NUM];

	// release all allocated memory
	ErrorType removeIsotropicScattering(ub32 materialNum);
	ErrorType removeTabularScattering(ub32 materialNum);
	ErrorType removeHenyeyGreensteinScattering(ub32 materialNum);
	ErrorType removeVonMisesFisherScattering(ub32 materialNum);

	// ask each scatterer if it is isotropic scattering
	bool isIsotropicScatteringIsotropic(ub32 materialNum) const;
	bool isIsotropicScatteringTabular(ub32 materialNum) const;
	bool isIsotropicScatteringHenyeyGreenstein(ub32 materialNum) const;
	bool isIsotropicScatteringVonMisesFisher(ub32 materialNum) const;

	ub32 allocationCount[MATERIAL_NUM];

	// Scan all scatterers and check if they are isotropic
	void updateIsIsotropic();
	ub32 isIsotropicScatteringBool;
};

// \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ //
// ----------------------------------------------------- Scattering type definitions ----------------------------------------------------- //
// \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ //

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Isotropic Scattering >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> //

// ------------------------------------ Class Function Implementations ------------------------------------ //
ErrorType Scattering::setIsotropicScattering(ub32 materialNum)
{
	// First remove any existing material
	ErrorType err;
	err = remove(materialNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	ScatteringNS::ScatteringType isotropicType = ScatteringNS::ScatteringType::ISOTROPIC;
	scatteringTypesListCPU[materialNum] = isotropicType;

	cudaMemcpyToSymbol(ScatteringNS::scatteringTypesList,
		&isotropicType,
		sizeof(ScatteringNS::ScatteringType),
		sizeof(ScatteringNS::ScatteringType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	updateIsIsotropic();

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::removeIsotropicScattering(ub32 materialNum)
{
	ScatteringNS::ScatteringType removeType = ScatteringNS::ScatteringType::NOT_DEFINED;
	scatteringTypesListCPU[materialNum] = removeType;

	cudaMemcpyToSymbol(ScatteringNS::scatteringTypesList,
		&removeType,
		sizeof(ScatteringNS::ScatteringType),
		sizeof(ScatteringNS::ScatteringType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	updateIsIsotropic();
	return ErrorType::NO_ERROR;
}

bool Scattering::isIsotropicScatteringIsotropic(ub32 materialNum) const
{
	return true;
}

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Tabular Scattering >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> //

// ------------------------------------ Data Structures ------------------------------------ //
namespace TabularScatteringNS
{
	typedef struct {
		ub32 tableSize;
		float_type* icdf;
		float_type* pdf;
		ComplexType* amplitudeFunction;
	} TabularAmplitudeFunction;
}

// ------------------------------------ Kernels ------------------------------------ //
namespace TabularScatteringNS
{
	// compute the sampling pdf, without the normalization
	__global__ void getSamplePdfRaw(float_type* sapmlingPdf, const ComplexType* amplitudeFunction, ub32 entriesNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < entriesNum)
		{
#if DIMS==2
			sapmlingPdf[threadNum] = absSquare(amplitudeFunction[threadNum]);
#else
			// angle from 0 to pi
			float_type theta = ((float_type)threadNum / (float_type)(entriesNum - 1)) * CUDART_PI;
			sapmlingPdf[threadNum] = absSquare(amplitudeFunction[threadNum]) * sin(theta);
#endif
		}
	}

	// normalize vector according to the multiplier
	__global__ void normalizeKernel(float_type* inVector, float_type multiplier, ub32 entriesNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < entriesNum)
		{
			inVector[threadNum] *= multiplier;
		}
	}
	__global__ void normalizeKernel(ComplexType* inVector, float_type multiplier, ub32 entriesNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < entriesNum)
		{
			inVector[threadNum] = inVector[threadNum] * multiplier;
		}
	}

	// comput the icdf from the cdf
	__global__ void setThetaKernel(float_type* theta, ub32 entriesNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < entriesNum)
		{
			theta[threadNum] = ((float_type)threadNum) / ((float_type)(entriesNum - 1));
		}
	}

	// normalize cdf after inverting cdf
	__global__ void icdfNormalizeKernel(float_type* icdf, ub32 entriesNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < entriesNum)
		{
			icdf[threadNum] *= (float_type)((
#if DIMS==2
				2.0 *
#endif
				CUDART_PI) / ((float_type)(entriesNum - 1)));
		}
	}
}

// ------------------------------------ Class Function Implementations ------------------------------------ //
ErrorType Scattering::setTabularScattering(const ComplexType* f, ub32 entriesNum, ub32 materialNum)
{
	// First remove any existing material
	ErrorType err;
	err = remove(materialNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	allocationCount[materialNum] = 0;

	scatteringParametersCPU[materialNum] = malloc(sizeof(TabularScatteringNS::TabularAmplitudeFunction));
	if (scatteringParametersCPU[materialNum] == NULL)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;
	
	TabularScatteringNS::TabularAmplitudeFunction* cpuTabularFunction = (TabularScatteringNS::TabularAmplitudeFunction*)scatteringParametersCPU[materialNum];
	float_type *cdf, *theta;

	// allocate memory
	if (cudaMalloc(&cpuTabularFunction->amplitudeFunction, sizeof(ComplexType) * entriesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMalloc(&cpuTabularFunction->pdf, sizeof(float_type) * entriesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMalloc(&cpuTabularFunction->icdf, sizeof(float_type) * entriesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMalloc(&cdf, sizeof(float_type) * entriesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	if (cudaMalloc(&theta, sizeof(float_type) * entriesNum) != cudaError_t::cudaSuccess)
	{
		cudaFree(cdf);
		return ErrorType::ALLOCATION_ERROR;
	}

	// copy the input amplitude function to gpu
	if (cudaMemcpy(cpuTabularFunction->amplitudeFunction,
		f,
		sizeof(ComplexType) * entriesNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		cudaFree(cdf);
		cudaFree(theta);

		return ErrorType::ALLOCATION_ERROR;
	}

	ub32 threadsNum = entriesNum < THREADS_NUM ? entriesNum : THREADS_NUM;
	ub32 blocksNum = (entriesNum - 1) / THREADS_NUM + 1;

	// compute the sampling pdf without normalization
	TabularScatteringNS::getSamplePdfRaw <<<blocksNum, threadsNum >>> (cpuTabularFunction->pdf, cpuTabularFunction->amplitudeFunction, entriesNum);

	// sum all entries
	float_type samplePdfSum = thrust::reduce(thrust::device, cpuTabularFunction->pdf, cpuTabularFunction->pdf + entriesNum, (float_type)0.0);
	cudaDeviceSynchronize();

	// normalize sampling pdf
	TabularScatteringNS::normalizeKernel <<<blocksNum, threadsNum >>> (cpuTabularFunction->pdf, 1 / samplePdfSum, entriesNum);

	// compute cross section
	float_type crossSection = (float_type)((samplePdfSum / (float_type)(entriesNum)) * 2.0 * CUDART_PI
#if DIMS==3
		* CUDART_PI
#endif
		);

	// normalize the amplitude function
	TabularScatteringNS::normalizeKernel <<<blocksNum, threadsNum >>> (cpuTabularFunction->amplitudeFunction, sqrt(1 / crossSection), entriesNum);

	// compute the cdf
	thrust::inclusive_scan(thrust::device, cpuTabularFunction->pdf, cpuTabularFunction->pdf + entriesNum, cdf);
	cudaDeviceSynchronize();

	// compute theta values to search on cdf
	TabularScatteringNS::setThetaKernel <<<blocksNum, threadsNum >>> (theta, entriesNum);
	cudaDeviceSynchronize();

	// compute the icdf
	thrust::lower_bound(thrust::device, cdf, cdf + entriesNum, theta, theta + entriesNum, cpuTabularFunction->icdf);
	cudaDeviceSynchronize();

	// normalize icdf
	TabularScatteringNS::icdfNormalizeKernel <<<blocksNum, threadsNum >>> (cpuTabularFunction->icdf, entriesNum);
	cudaDeviceSynchronize();

	// translate pdf to eval pdf
	TabularScatteringNS::getSamplePdfRaw <<<blocksNum, threadsNum >>> (cpuTabularFunction->pdf, cpuTabularFunction->amplitudeFunction, entriesNum);

	cudaFree(cdf);
	cudaFree(theta);

	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	// copy to amplitude function
	cpuTabularFunction->tableSize = entriesNum;

	if (cudaMalloc(&scatteringParametersGPU[materialNum], sizeof(TabularScatteringNS::TabularAmplitudeFunction)) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	if (cudaMemcpy(scatteringParametersGPU[materialNum], cpuTabularFunction, sizeof(TabularScatteringNS::TabularAmplitudeFunction), cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	cudaMemcpyToSymbol(ScatteringNS::scatteringParameters,
		&scatteringParametersGPU[materialNum],
		sizeof(ScatteringNS::ScatteringParmetersType),
		sizeof(ScatteringNS::ScatteringParmetersType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	ScatteringNS::ScatteringType tabularType = ScatteringNS::ScatteringType::TABULAR;
	scatteringTypesListCPU[materialNum] = tabularType;

	cudaMemcpyToSymbol(ScatteringNS::scatteringTypesList,
		&tabularType,
		sizeof(ScatteringNS::ScatteringType),
		sizeof(ScatteringNS::ScatteringType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	updateIsIsotropic();

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::removeTabularScattering(ub32 materialNum)
{
	TabularScatteringNS::TabularAmplitudeFunction* cpuTabularFunction =
		(TabularScatteringNS::TabularAmplitudeFunction*) scatteringParametersCPU[materialNum];

	switch (allocationCount[materialNum])
	{
	case 5:
		cudaFree(scatteringParametersGPU[materialNum]);
	case 4:
		cudaFree(cpuTabularFunction->icdf);
	case 3:
		cudaFree(cpuTabularFunction->pdf);
	case 2:
		cudaFree(cpuTabularFunction->amplitudeFunction);
	case 1:
		free(scatteringParametersCPU[materialNum]);
	default:
		break;
	}

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	ScatteringNS::ScatteringType removeType = ScatteringNS::ScatteringType::NOT_DEFINED;
	scatteringTypesListCPU[materialNum] = removeType;

	cudaMemcpyToSymbol(ScatteringNS::scatteringTypesList,
		&removeType,
		sizeof(ScatteringNS::ScatteringType),
		sizeof(ScatteringNS::ScatteringType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	updateIsIsotropic();

	return ErrorType::NO_ERROR;
}

bool Scattering::isIsotropicScatteringTabular(ub32 materialNum) const
{
	// Can be implemented wiser
	return false;
}

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Henyey Greenstein >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> //

// ------------------------------------ Data Structures ------------------------------------ //
namespace HenyeyGreensteinNS
{
	typedef struct {
		float_type g;

		float_type g_up;
		float_type g_down1;
		float_type g_down2;
	} HenyeyGreensteinFunction;
}

// ------------------------------------ Class Function Implementations ------------------------------------ //

ErrorType Scattering::setHenyeyGreensteinScattering(float_type g, ub32 materialNum)
{
	// First remove any existing material
	ErrorType err;
	err = remove(materialNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	HenyeyGreensteinNS::HenyeyGreensteinFunction hgScatterer;
	allocationCount[materialNum] = 0;

	hgScatterer.g = g;
#if DIMS==2
	hgScatterer.g_up = (float_type)(2.0 * ((1.0 - g * g) / (4.0 * CUDART_PI)));
#else
	hgScatterer.g_up = (float_type)((1.0 - g * g) / (4.0 * CUDART_PI));
#endif
	
	hgScatterer.g_down1 = (float_type)(1.0 + g * g);
	hgScatterer.g_down2 = (float_type)(-2.0 * g);

	scatteringParametersCPU[materialNum] = malloc(sizeof(HenyeyGreensteinNS::HenyeyGreensteinFunction));
	if (scatteringParametersCPU[materialNum] == NULL)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;
	memcpy(scatteringParametersCPU[materialNum], &hgScatterer, sizeof(HenyeyGreensteinNS::HenyeyGreensteinFunction));

	if (cudaMalloc(&scatteringParametersGPU[materialNum], sizeof(HenyeyGreensteinNS::HenyeyGreensteinFunction)) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;
	if (cudaMemcpy(scatteringParametersGPU[materialNum], &hgScatterer, sizeof(HenyeyGreensteinNS::HenyeyGreensteinFunction), cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	cudaMemcpyToSymbol(ScatteringNS::scatteringParameters,
		&scatteringParametersGPU[materialNum],
		sizeof(ScatteringNS::ScatteringParmetersType),
		sizeof(ScatteringNS::ScatteringParmetersType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	ScatteringNS::ScatteringType hgType = ScatteringNS::ScatteringType::HENYEY_GREENSTEIN;
	scatteringTypesListCPU[materialNum] = hgType;

	cudaMemcpyToSymbol(ScatteringNS::scatteringTypesList,
		&hgType,
		sizeof(ScatteringNS::ScatteringType),
		sizeof(ScatteringNS::ScatteringType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	updateIsIsotropic();

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::removeHenyeyGreensteinScattering(ub32 materialNum)
{
	switch (allocationCount[materialNum])
	{
	case 2:
		cudaFree(scatteringParametersGPU[materialNum]);
	case 1:
		free(scatteringParametersCPU[materialNum]);
	default:
		break;
	}

	ScatteringNS::ScatteringType removeType = ScatteringNS::ScatteringType::NOT_DEFINED;
	scatteringTypesListCPU[materialNum] = removeType;

	cudaMemcpyToSymbol(ScatteringNS::scatteringTypesList,
		&removeType,
		sizeof(ScatteringNS::ScatteringType),
		sizeof(ScatteringNS::ScatteringType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	updateIsIsotropic();

	return ErrorType::NO_ERROR;
}

bool Scattering::isIsotropicScatteringHenyeyGreenstein(ub32 materialNum) const
{
	HenyeyGreensteinNS::HenyeyGreensteinFunction* cpuHgFunction =
		(HenyeyGreensteinNS::HenyeyGreensteinFunction*)scatteringParametersCPU[materialNum];
	return (abs(cpuHgFunction->g) < 0.0001);
}

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< von Mises–Fisher Mixture >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> //
// ------------------------------------ Data Structures ------------------------------------ //
namespace vmfScatteringNS
{
	// Gaussian Beam Material Structre
	// GaussianBeamFunction is a vector in size of the maximal materials allowed
	typedef struct
	{
		// mixture amplitude data
		float_type* mixtureMu;
		float_type* mixtureC;

		// mixture phase function data (size of mixturesNum^2)
		float_type* mixturePhaseMu;
		float_type* mixturePdf;
		float_type* mixtureCdf;
		float_type* mixtureLogNormalization;

		ub32 mixturesNum;
		ub32 isIsotropic;
	} GaussianBeamFunction;
}

// ------------------------------------ Class Function Implementations ------------------------------------ //
ErrorType Scattering::setVonMisesFisherScattering(const float_type* mixtureMu, const float_type* mixtureC, const float_type* mixtureAlpha, ub32 mixturesNum, ub32 materialNum)
{
	if (mixturesNum == 0)
	{
		return ErrorType::NO_ERROR;
	}

	// First remove any existing material
	ErrorType err;
	err = remove(materialNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	allocationCount[materialNum] = 0;

	scatteringParametersCPU[materialNum] = malloc(sizeof(vmfScatteringNS::GaussianBeamFunction));
	if (scatteringParametersCPU[materialNum] == NULL)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	vmfScatteringNS::GaussianBeamFunction* cpuVmfFunction = (vmfScatteringNS::GaussianBeamFunction*)scatteringParametersCPU[materialNum];

	// allocate buffers
	cpuVmfFunction->mixturesNum = mixturesNum;

	if (cudaMalloc(&cpuVmfFunction->mixtureMu, sizeof(float_type) * mixturesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMalloc(&cpuVmfFunction->mixtureC, sizeof(float_type) * mixturesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMalloc(&cpuVmfFunction->mixturePhaseMu, sizeof(float_type) * mixturesNum * mixturesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMalloc(&cpuVmfFunction->mixturePdf, sizeof(float_type) * mixturesNum * mixturesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMalloc(&cpuVmfFunction->mixtureCdf, sizeof(float_type) * mixturesNum * mixturesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMalloc(&cpuVmfFunction->mixtureLogNormalization, sizeof(float_type) * mixturesNum * mixturesNum) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;

	if (cudaMemcpy(cpuVmfFunction->mixtureMu, mixtureMu, sizeof(float_type) * mixturesNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	float_type* tmpVec = (float_type*)malloc(sizeof(float_type) * mixturesNum * mixturesNum);
	if (tmpVec == NULL)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	for (ub32 mixNum = 0; mixNum < mixturesNum; mixNum++)
	{
		tmpVec[mixNum] = mixtureC[mixNum] + log(mixtureAlpha[mixNum]);
	}

	if (cudaMemcpy(cpuVmfFunction->mixtureC, tmpVec, sizeof(float_type) * mixturesNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		free(tmpVec);
		return ErrorType::ALLOCATION_ERROR;
	}

	for (ub32 mixNum = 0; mixNum < mixturesNum * mixturesNum; mixNum++)
	{
		tmpVec[mixNum] = mixtureMu[mixNum % mixturesNum] + mixtureMu[mixNum / mixturesNum];
	}
	if (cudaMemcpy(cpuVmfFunction->mixturePhaseMu, tmpVec, sizeof(float_type) * mixturesNum * mixturesNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	for (ub32 mixNum = 0; mixNum < mixturesNum * mixturesNum; mixNum++)
	{
		tmpVec[mixNum] = vMFnormalizationLog(mixtureMu[mixNum % mixturesNum] + mixtureMu[mixNum / mixturesNum]);
	}
	if (cudaMemcpy(cpuVmfFunction->mixtureLogNormalization, tmpVec, sizeof(float_type) * mixturesNum * mixturesNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	// Check if isotropic
	cpuVmfFunction->isIsotropic = true;
	for (ub32 mixNum = 0; mixNum < mixturesNum; mixNum++)
	{
		if (abs(mixtureMu[mixNum]) > 0.0001)
		{
			cpuVmfFunction->isIsotropic = false;
		}
	}

	// mixture weight values
	float_type sumWeightsVal = 0.;
	for (ub32 mixNum = 0; mixNum < mixturesNum * mixturesNum; mixNum++)
	{
		ub32 mix1 = mixNum % mixturesNum;
		ub32 mix2 = mixNum / mixturesNum;

		tmpVec[mixNum] = exp(mixtureC[mix1] + log(mixtureAlpha[mix1]) + mixtureC[mix2] + log(mixtureAlpha[mix2]) - 
			vMFnormalizationLog(mixtureMu[mix1] + mixtureMu[mix2]));
		sumWeightsVal += tmpVec[mixNum];
	}

	// divide sum
	for (ub32 mixNum = 0; mixNum < mixturesNum * mixturesNum; mixNum++)
	{
		tmpVec[mixNum] /= sumWeightsVal;
	}

	if (cudaMemcpy(cpuVmfFunction->mixturePdf, tmpVec, sizeof(float_type) * mixturesNum * mixturesNum, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		free(tmpVec);
		return ErrorType::ALLOCATION_ERROR;
	}

	free(tmpVec);

	// compute cumsum
	thrust::exclusive_scan(thrust::device, cpuVmfFunction->mixturePdf, cpuVmfFunction->mixturePdf + mixturesNum * mixturesNum, cpuVmfFunction->mixtureCdf, 0.);
	cudaDeviceSynchronize();

	// Copy to GPU const memory
	if (cudaMalloc(&scatteringParametersGPU[materialNum], sizeof(vmfScatteringNS::GaussianBeamFunction)) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount[materialNum]++;
	if (cudaMemcpy(scatteringParametersGPU[materialNum], cpuVmfFunction, sizeof(vmfScatteringNS::GaussianBeamFunction), cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	cudaMemcpyToSymbol(ScatteringNS::scatteringParameters,
		&scatteringParametersGPU[materialNum],
		sizeof(ScatteringNS::ScatteringParmetersType),
		sizeof(ScatteringNS::ScatteringParmetersType)* materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	ScatteringNS::ScatteringType vmfType = ScatteringNS::ScatteringType::VMF_MIXTURE;
	scatteringTypesListCPU[materialNum] = vmfType;

	cudaMemcpyToSymbol(ScatteringNS::scatteringTypesList,
		&vmfType,
		sizeof(ScatteringNS::ScatteringType),
		sizeof(ScatteringNS::ScatteringType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	updateIsIsotropic();

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::removeVonMisesFisherScattering(ub32 materialNum)
{
	vmfScatteringNS::GaussianBeamFunction* cpuVmfFunction =
		(vmfScatteringNS::GaussianBeamFunction*)scatteringParametersCPU[materialNum];

	if (cpuVmfFunction->mixturesNum == 0)
	{
		return ErrorType::NO_ERROR;
	}

	switch (allocationCount[materialNum])
	{
	case 8:
		cudaFree(scatteringParametersGPU[materialNum]);
	case 7:
		cudaFree(cpuVmfFunction->mixtureLogNormalization);
	case 6:
		cudaFree(cpuVmfFunction->mixtureCdf);
	case 5:
		cudaFree(cpuVmfFunction->mixturePdf);
	case 4:
		cudaFree(cpuVmfFunction->mixturePhaseMu);
	case 3:
		cudaFree(cpuVmfFunction->mixtureC);
	case 2:
		cudaFree(cpuVmfFunction->mixtureMu);
	case 1:
		free(scatteringParametersCPU[materialNum]);
	default:
		break;
	}

	ScatteringNS::ScatteringType removeType = ScatteringNS::ScatteringType::NOT_DEFINED;
	scatteringTypesListCPU[materialNum] = removeType;

	cudaMemcpyToSymbol(ScatteringNS::scatteringTypesList,
		&removeType,
		sizeof(ScatteringNS::ScatteringType),
		sizeof(ScatteringNS::ScatteringType) * materialNum,
		cudaMemcpyKind::cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	updateIsIsotropic();

	return ErrorType::NO_ERROR;
}

bool Scattering::isIsotropicScatteringVonMisesFisher(ub32 materialNum) const
{
	vmfScatteringNS::GaussianBeamFunction* cpuVmfFunction =
		(vmfScatteringNS::GaussianBeamFunction*)scatteringParametersCPU[materialNum];

	return cpuVmfFunction->isIsotropic;
}

// \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ //
// --------------------------------------------------- Scattering class implementation --------------------------------------------------- //
// \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ //

// ------------------------------------ Device function implementions for each scattering type ------------------------------------ //
namespace ScatteringNS
{
	// get the amplitude function of the scattering in particular direction
	// cosang is the cosine of the scattering direction, where cosAng = 1 is forward direction, and -1 is backscattering
	__device__ ComplexType amplitudeDeviceImplementation(ScatteringType scatteringType, ScatteringParmetersType amplitudeFunction, float_type cosAng)
	{
		switch (scatteringType)
		{
		case ScatteringType::ISOTROPIC:
#if DIMS==2
		case ScatteringType::VMF_MIXTURE:
#endif
		{
			return ComplexType{ ISOTROPIC_AMPLITUDE, 0.0 };
			break;
		}
		case ScatteringType::TABULAR:
		{
			const TabularScatteringNS::TabularAmplitudeFunction* inAmpliteudeFunctionStruct =
				(const TabularScatteringNS::TabularAmplitudeFunction*)amplitudeFunction;

			return inAmpliteudeFunctionStruct->amplitudeFunction[round_num(safeAcos(cosAng) * (inAmpliteudeFunctionStruct->tableSize - 1) / (
#if DIMS==2
				2 *
#endif
				CUDART_PI), inAmpliteudeFunctionStruct->tableSize)];
			break;
		}
		case ScatteringType::HENYEY_GREENSTEIN:
		{
			const HenyeyGreensteinNS::HenyeyGreensteinFunction* inAmpliteudeFunctionStruct =
				(const HenyeyGreensteinNS::HenyeyGreensteinFunction*)amplitudeFunction;

			float_type downCalc = fma(inAmpliteudeFunctionStruct->g_down2, cosAng, inAmpliteudeFunctionStruct->g_down1);
#if DIMS==2
			return ComplexType( sqrt((inAmpliteudeFunctionStruct->g_up) / (downCalc)));
#else
			return ComplexType( sqrt((inAmpliteudeFunctionStruct->g_up) / (downCalc * sqrt(downCalc))));
#endif
			break;
		}
#if DIMS==3
		case ScatteringType::VMF_MIXTURE:
		{
			const vmfScatteringNS::GaussianBeamFunction* inAmpliteudeFunctionStruct =
				(const vmfScatteringNS::GaussianBeamFunction*)amplitudeFunction;
			
			// For each mixture, compute the scattering contribution
			float_type res = 0.;
			for (ub32 mixNum = 0; mixNum < inAmpliteudeFunctionStruct->mixturesNum; mixNum++)
			{
				res += exp(cosAng * inAmpliteudeFunctionStruct->mixtureMu[mixNum] + inAmpliteudeFunctionStruct->mixtureC[mixNum]);
			}

			return ComplexType(res);
		}
#endif
		default:
		{
			break;
		}
		};

		return ComplexType();
	}

	// get the pdf function of the scattering in particular direction
	// cosang is the cosine of the scattering direction, where cosAng = 1 is forward direction, and -1 is backscattering
	__device__ float_type pdfDeviceImplementation(ScatteringType scatteringType, ScatteringParmetersType amplitudeFunction, float_type cosAng)
	{
		switch (scatteringType)
		{
		case ScatteringType::ISOTROPIC:
#if DIMS==2
		case ScatteringType::VMF_MIXTURE:
#endif
		{
			return ISOTROPIC_PDF;
			break;
		}
		case ScatteringType::TABULAR:
		{
			const TabularScatteringNS::TabularAmplitudeFunction* inAmpliteudeFunctionStruct =
				(const TabularScatteringNS::TabularAmplitudeFunction*)amplitudeFunction;

			return (inAmpliteudeFunctionStruct->pdf[round_num(safeAcos(cosAng) * (inAmpliteudeFunctionStruct->tableSize - 1) / (
#if DIMS==2
				2 *
#endif
				CUDART_PI), inAmpliteudeFunctionStruct->tableSize)]);
			break;
		}
		case ScatteringType::HENYEY_GREENSTEIN:
		{
			const HenyeyGreensteinNS::HenyeyGreensteinFunction* inAmpliteudeFunctionStruct =
				(const HenyeyGreensteinNS::HenyeyGreensteinFunction*)amplitudeFunction;

			float_type downCalc = fma(inAmpliteudeFunctionStruct->g_down2, cosAng, inAmpliteudeFunctionStruct->g_down1);
#if DIMS==2
			return (inAmpliteudeFunctionStruct->g_up) / (downCalc);
#else
			return (inAmpliteudeFunctionStruct->g_up) / (downCalc * sqrt(downCalc));
#endif
			break;
		}
#if DIMS==3
		case ScatteringType::VMF_MIXTURE:
		{
			const vmfScatteringNS::GaussianBeamFunction* inAmpliteudeFunctionStruct =
				(const vmfScatteringNS::GaussianBeamFunction*)amplitudeFunction;

			// float_type sinAng = (abs(cosAng) - float_type(1.0) < EPSILON) ? 0.0 : sqrt((float_type)(1.0) - cosAng * cosAng);
			
			// For each mixture, compute the scattering contribution
			ub32 mixturesNum = inAmpliteudeFunctionStruct->mixturesNum;
			float_type res = 0.;
			for (ub32 mixNum = 0; mixNum < mixturesNum * mixturesNum; mixNum++)
			{
				res += inAmpliteudeFunctionStruct->mixturePdf[mixNum] *
					exp(cosAng * inAmpliteudeFunctionStruct->mixturePhaseMu[mixNum] + inAmpliteudeFunctionStruct->mixtureLogNormalization[mixNum]);
			}

			return res;
		}
#endif
		default:
		{
			break;
		}
		}

		return 0.0;
	}

	__device__ float_type sampleCosineDeviceImplementation(ScatteringType scatteringType, ScatteringParmetersType amplitudeFunction, curandState_t* state)
	{
		float_type cosangle;
		switch (scatteringType)
		{
		case ScatteringType::TABULAR:
		{
			const TabularScatteringNS::TabularAmplitudeFunction* inAmpliteudeFunctionStruct =
				(const TabularScatteringNS::TabularAmplitudeFunction*)amplitudeFunction;

			ub32 randomIdx = round_num(randUniform(state) * inAmpliteudeFunctionStruct->tableSize, inAmpliteudeFunctionStruct->tableSize);
			cosangle = cos(inAmpliteudeFunctionStruct->icdf[randomIdx]);
			break;
		}
		case ScatteringType::HENYEY_GREENSTEIN:
		{
			const HenyeyGreensteinNS::HenyeyGreensteinFunction* inAmpliteudeFunctionStruct =
				(const HenyeyGreensteinNS::HenyeyGreensteinFunction*)amplitudeFunction;

			float_type g = inAmpliteudeFunctionStruct->g;
			float_type randomNum;

			if (abs(g) < 0.001)
			{
				cosangle = randUniform(state) * 2.0 - 1.0;
			}
			else
			{
				randomNum = randUniform(state);

#if DIMS==2
				cosangle = cos(2.0 * atan(((1.0 - g) / (1.0 + g)) * tan((CUDART_PI / 2.0) * (1 - 2.0 * randomNum))));
#else
				float_type s = (1.0 - g * g) / (1.0 - g + 2.0 * g * randomNum);
				cosangle = (1.0 + g * g - s * s) / (2.0 * g);
#endif
			}
			break;
		}
#if DIMS==3
		case ScatteringType::VMF_MIXTURE:
		{
			const vmfScatteringNS::GaussianBeamFunction* inAmpliteudeFunctionStruct =
				(const vmfScatteringNS::GaussianBeamFunction*)amplitudeFunction;

			// Choose random scatterer
			ub32 mixturesNum = inAmpliteudeFunctionStruct->mixturesNum;

			float_type randomNum = randUniform(state);
			ub32 sampledMixture = binarySearchKernel(inAmpliteudeFunctionStruct->mixtureCdf, mixturesNum * mixturesNum, randomNum);

			cosangle = random_vMF_cosine(inAmpliteudeFunctionStruct->mixturePhaseMu[sampledMixture], state);
			break;
		}
#endif
		default:
		{
			break;
		}
		}

		return cosangle;
	}

	// sample new scattring direction reletivly to the prevDir
	// state is a pointer for a current random number generator state
	__device__ VectorType sampleDeviceImplementation(ScatteringType scatteringType, ScatteringParmetersType amplitudeFunction, VectorType prevDir, curandState_t* state)
	{
		// isotropic - random direction
#if DIMS==2
		if (scatteringType == ScatteringType::ISOTROPIC || scatteringType == ScatteringType::VMF_MIXTURE)
#else
		if (scatteringType == ScatteringType::ISOTROPIC)
#endif
		{
			return randomDirection(state);
		}
		
		// otherwise, sample new scattering direction
		float_type cosangle = sampleCosineDeviceImplementation(scatteringType, amplitudeFunction, state);

		// and rotate according to new sampled direction
		return vector_rotation(prevDir, cosangle, state);

	}
}

// ------------------------------------ Kernels ------------------------------------ //
namespace ScatteringNS
{
	// seperateMode: 0 - single vector p1 = p2 = p3.
	//               1 - matrix p1 x (p2 = p3)
	//               2 - matrix p1 x p3 x p2
	template <ub32 seperateMode>
	__global__ void amplitudeKernel(ComplexType* amplitude,
		const VectorType* p1, bool isP1Direction, ub32 p1num,
		const MediumPoint* p2, ub32 p2num,
		const VectorType* p3, bool isP3Direction, ub32 p3num)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 p1idx, p2idx, p3idx;

		if (seperateMode == 0)
		{
			p1idx = threadNum;
			p2idx = threadNum;
			p3idx = threadNum;
		}
		else if (seperateMode == 1)
		{
			p1idx = threadNum % p1num;
			p2idx = threadNum / p1num;
			p3idx = threadNum / p1num;
		}
		else
		{
			p1idx = threadNum % p1num;
			p2idx = (threadNum / (p1num * p3num));
			p3idx = (threadNum / p1num) % p3num;
		}

		if (p2idx < p2num)
		{
			ib32 materialNum = p2[p2idx].material;

			if (materialNum > 0)
			{
				VectorType p1p2dir;
				if (isP1Direction)
				{
					p1p2dir = p1[p1idx];
				}
				else
				{
					p1p2dir = normalize(p2[p2idx].position - p1[p1idx]);
				}
				VectorType p2p3dir;
				if (isP3Direction)
				{
					p2p3dir = p3[p3idx];
				}
				else
				{
					p2p3dir = normalize(p3[p3idx] - p2[p2idx].position);
				}

				amplitude[threadNum] = amplitudeDeviceImplementation(scatteringTypesList[materialNum],
					scatteringParameters[materialNum], p1p2dir * p2p3dir);

				/*
				printf("Scattering amplitude %d: p1: [%f %f %f], p2: [%f %f %f], p3: [%f %f %f], p1p2dir: [%f %f %f], p2p3dir: [%f %f %f], cosDir: %f material: %d \n",
					threadNum, p1[p1idx].x, p1[p1idx].y, p1[p1idx].z,
					p2[p2idx].position.x, p2[p2idx].position.y, p2[p2idx].position.z,
					p3[p3idx].x, p3[p3idx].y, p3[p3idx].z,
					p1p2dir.x, p1p2dir.y, p1p2dir.z,
					p2p3dir.x, p2p3dir.y, p2p3dir.z, p1p2dir * p2p3dir, materialNum);
					*/
			}
			else
			{
				amplitude[threadNum] = 0;
			}
		}
	}

	template <bool isSeparatable, bool isDirection>
	__global__ void pdfKernel(float_type* pdf,
		const VectorType* p1, ub32 p1num,
		const MediumPoint* p2,
		const VectorType* p3, ub32 p2p3num)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 p1idx, p2p3idx;

		if (isSeparatable)
		{
			p1idx = threadNum % p1num;
			p2p3idx = threadNum / p1num;
		}
		else
		{
			p1idx = threadNum;
			p2p3idx = threadNum;
		}

		if (p2p3idx < p2p3num)
		{
			ib32 materialNum = p2[p2p3idx].material;

			VectorType inDir;
			if (isDirection)
			{
				inDir = p1[p1idx];
			}
			else
			{
				inDir = normalize(p2[p2p3idx].position - p1[p1idx]);
			}

			pdf[threadNum] *= pdfDeviceImplementation(scatteringTypesList[materialNum],
				scatteringParameters[materialNum],
				inDir * normalize(p3[p2p3idx] - p2[p2p3idx].position));

			/*
			printf("Scattering pdf %d: p1: [%f %f %f], p2: [%f %f %f], p3: [%f %f %f], indir: [%f %f %f], p2p3dir: [%f %f %f], cosDir: %f material: %d \n",
				threadNum, p1[p1idx].x, p1[p1idx].y, p1[p1idx].z,
				p2[p2p3idx].position.x, p2[p2p3idx].position.y, p2[p2p3idx].position.z,
				p3[p2p3idx].x, p3[p2p3idx].y, p3[p2p3idx].z,
				inDir.x, inDir.y, inDir.z,
				normalize(p3[p2p3idx] - p2[p2p3idx].position).x, normalize(p3[p2p3idx] - p2[p2p3idx].position).y, normalize(p3[p2p3idx] - p2[p2p3idx].position).z,
				inDir * normalize(p3[p2p3idx] - p2[p2p3idx].position), materialNum);
				*/
		}
	}

	__global__ void newDirectionKernel(VectorType* sampledDirection,
		const VectorType* p1, const MediumPoint* p2, ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < pointsNum)
		{
			ib32 materialNum = p2[threadNum].material;

			if (materialNum >= 0)
			{
				sampledDirection[threadNum] = sampleDeviceImplementation(scatteringTypesList[materialNum],
					scatteringParameters[materialNum],
					normalize(p2[threadNum].position - p1[threadNum]),
					statePool + threadNum);
			}
			else
			{
				sampledDirection[threadNum] = VectorType(0);
			}
		}
	}
}

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
Scattering::Scattering()
{
	MEMORY_CHECK("Scattering allocation begin");
	for (ub32 materialNum = 0; materialNum < MATERIAL_NUM; materialNum++)
	{
		scatteringTypesListCPU[materialNum] = ScatteringNS::ScatteringType::NOT_DEFINED;
	}
	MEMORY_CHECK("Scattering allocation end");
}

Scattering::~Scattering()
{
	MEMORY_CHECK("Scattering free begin");
	for (ub32 materialNum = 0; materialNum < MATERIAL_NUM; materialNum++)
	{
		remove(materialNum);
	}
	MEMORY_CHECK("Scattering free end");
}

// ------------------------------------ Function Implementations ------------------------------------ //

// Return the scattering amplitude of points p1 -> p2 -> p3, where the scattering function defined by scatteresList.
// All are device pointers in size of pointsNum.
// The scattering function is defined according to p2 position inside the material.
ErrorType Scattering::amplitude(ComplexType* amplitudeRes,
	const VectorType* p1, bool isP1Direction,
	const MediumPoint* p2,
	const VectorType* p3, bool isP3Direction, ub32 p1p2p3num) const
{
	ub32 threadsNum = p1p2p3num < THREADS_NUM ? p1p2p3num : THREADS_NUM;
	ub32 blocksNum = (p1p2p3num - 1) / THREADS_NUM + 1;

	ScatteringNS::amplitudeKernel <0> <<<blocksNum, threadsNum >>> (amplitudeRes,
		p1, isP1Direction, p1p2p3num, p2, p1p2p3num, p3, isP3Direction, p1p2p3num);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Scattering_amplitudeKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::amplitude(ComplexType* amplitudeRes,
	const VectorType* p1, bool isP1Direction, ub32 p1num,
	const MediumPoint* p2,
	const VectorType* p3, bool isP3Direction, ub32 p2p3num) const
{
	ub32 totalThreads = p1num * p2p3num;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	ScatteringNS::amplitudeKernel <1> <<<blocksNum, threadsNum >>> (amplitudeRes,
		p1, isP1Direction, p1num, p2, p2p3num, p3, isP3Direction, p2p3num);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Scattering_amplitudeKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::amplitude(ComplexType* amplitudeRes,
	const VectorType* p1, bool isP1Direction, ub32 p1num,
	const MediumPoint* p2, ub32 p2num,
	const VectorType* p3, bool isP3Direction, ub32 p3num) const
{
	ub32 totalThreads = p1num * p2num * p3num;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	ScatteringNS::amplitudeKernel <2> <<<blocksNum, threadsNum >>> (amplitudeRes,
		p1, isP1Direction, p1num, p2, p2num, p3, isP3Direction, p3num);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Scattering_amplitudeKernel;
	}

	return ErrorType::NO_ERROR;
}

// Return the scattering pdf of points p1 -> p2 -> p3, where the scattering function defined by scatteresList.
// All are device pointers in size of pointsNum.
// The scattering function is defined according to p2 position inside the material.
ErrorType Scattering::multiplyPdf(float_type* pdfRes,
	const VectorType* p1,
	const MediumPoint* p2,
	const VectorType* p3, ub32 p1p2p3num) const
{
	ub32 threadsNum = p1p2p3num < THREADS_NUM ? p1p2p3num : THREADS_NUM;
	ub32 blocksNum = (p1p2p3num - 1) / THREADS_NUM + 1;

	ScatteringNS::pdfKernel <false,false> <<<blocksNum, threadsNum >>> (pdfRes,
		p1, p1p2p3num, p2, p3, p1p2p3num);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Scattering_pdfKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::multiplyPdf(float_type* pdfRes,
	const VectorType* p1, ub32 p1num,
	const MediumPoint* p2,
	const VectorType* p3, ub32 p2p3num) const
{
	ub32 totalThreads = p1num * p2p3num;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	ScatteringNS::pdfKernel <true,false> <<<blocksNum, threadsNum >>> (pdfRes,
		p1, p1num, p2, p3, p2p3num);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Scattering_pdfKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::multiplyPdfDirection(float_type* pdfRes,
	const VectorType* direction,
	const MediumPoint* p2,
	const VectorType* p3, ub32 p1p2p3num) const
{
	ub32 threadsNum = p1p2p3num < THREADS_NUM ? p1p2p3num : THREADS_NUM;
	ub32 blocksNum = (p1p2p3num - 1) / THREADS_NUM + 1;

	ScatteringNS::pdfKernel <false,true> <<<blocksNum, threadsNum >>> (pdfRes,
		direction, p1p2p3num, p2, p3, p1p2p3num);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Scattering_pdfKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::multiplyPdfDirection(float_type* pdfRes,
	const VectorType* direction, ub32 directionNum,
	const MediumPoint* p2,
	const VectorType* p3, ub32 p2p3num) const
{
	ub32 totalThreads = directionNum * p2p3num;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	ScatteringNS::pdfKernel <true,true> <<<blocksNum, threadsNum >>> (pdfRes,
		direction, directionNum, p2, p3, p2p3num);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Scattering_pdfKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::newDirection(VectorType* sampledDirection,
	const VectorType* p1, const MediumPoint* p2,
	ub32 pointsNum) const
{
	ub32 threadsNum = pointsNum < THREADS_NUM ? pointsNum : THREADS_NUM;
	ub32 blocksNum = (pointsNum - 1) / THREADS_NUM + 1;

	ScatteringNS::newDirectionKernel <<<blocksNum, threadsNum >>> (sampledDirection,
		p1, p2, pointsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_Scattering_newDirectionKernel;
	}

	return ErrorType::NO_ERROR;
}

ErrorType Scattering::remove(ub32 materialNum)
{
	switch (scatteringTypesListCPU[materialNum])
	{
	case ScatteringNS::ScatteringType::ISOTROPIC:
	{
		return removeIsotropicScattering(materialNum);
		break;
	}
	case ScatteringNS::ScatteringType::TABULAR:
	{
		return removeTabularScattering(materialNum);
		break;
	}
	case ScatteringNS::ScatteringType::HENYEY_GREENSTEIN:
	{
		return removeHenyeyGreensteinScattering(materialNum);
		break;
	}
	case ScatteringNS::ScatteringType::VMF_MIXTURE:
	{
		return removeVonMisesFisherScattering(materialNum);
		break;
	}
	default:
	{

	}
	}

	return ErrorType::NO_ERROR;
}

bool Scattering::isIsotropicScattering(ub32 materialNum) const
{
	switch (scatteringTypesListCPU[materialNum])
	{
	case ScatteringNS::ScatteringType::ISOTROPIC:
	{
		return isIsotropicScatteringIsotropic(materialNum);
	}
	case ScatteringNS::ScatteringType::HENYEY_GREENSTEIN:
	{
		return isIsotropicScatteringHenyeyGreenstein(materialNum);
	}
	case ScatteringNS::ScatteringType::VMF_MIXTURE:
	{
		return isIsotropicScatteringVonMisesFisher(materialNum);
	}
	case ScatteringNS::ScatteringType::TABULAR:
	{
		return isIsotropicScatteringTabular(materialNum);
	}
	default:
	{
		return false;
	}
	}

	return false;
}

void Scattering::updateIsIsotropic()
{
	isIsotropicScatteringBool = true;

	for (ub32 materialNum = 0; materialNum < MATERIAL_NUM; materialNum++)
	{
		if (scatteringTypesListCPU[materialNum] != ScatteringNS::ScatteringType::NOT_DEFINED)
		{
			if (!isIsotropicScattering(materialNum))
			{
				isIsotropicScatteringBool = false;
			}
		}
	}
}
