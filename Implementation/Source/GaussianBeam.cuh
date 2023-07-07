#pragma once

#include "../../Interface/SourceInterface.cuh"
#include "../../SpecialFunctions.cuh"
#include "../Scattering.cuh"
#include "../../Interface/MediumInterface.cuh"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

// Let Nl be number of illuminations
// Let Nw be the number of wavelenghs
// let P be the maximal paths in batch
// let B be the number of bins to sample z axis of the beam
// let M be the maximal mixture number of a material

#if DIMS==3
// ------------------------------------ Private data structres ------------------------------------ //
namespace GaussianBeamNS
{
	// GaussianDistanceSampler is a vector in size of Nl * Nw
	typedef struct
	{
		MediumPoint focalPoint;                // The corresponding focal point

		// Sample first position
		MediumPoint* zPoints;                  // size of B
		float_type* pdf;                       // size of M * B
		float_type* icdf;                      // size of M * B
		float_type* alphaPdf;                  // size of M
		float_type* alphaIcdf;                 // size of M
		

		// t = 0 is the focal point
		// negative t is opposite to beamDirections
		// positive t is in beamDirections

		float_type tMin;
		float_type tMax;
		float_type dt;

		VectorType xyBase[2];
	} GaussianDistanceSampler;
}
#endif

// ------------------------------------ GPU Constants ------------------------------------ //
namespace GaussianBeamNS
{

	__constant__ VectorType meanViewDirection;
	__constant__ ub32 maxMixtures;
}

// ------------------------------------ Gaussian Beam Class ------------------------------------ //
class GaussianBeam : public Source
{
public:
	// cpu pointers
	GaussianBeam(ErrorType* err, const Simulation* simulationHandler, const Medium* mediumHandler, const Scattering* scatteringHandler,
		const VectorType* focalPoints, const VectorType* beamDirections, float_type gaussianAperture, ub32 beamsNum,
		ConnectionType cType, bool apertureNormalized = true, ub32 zSamplesNum = 1000);

	~GaussianBeam();

	ErrorType updateSampler(GaussianBeamStructType* gaussianBeamData);

	// Declerations
	virtual ErrorType sampleFirstPoint(VectorType* randomSourcePoint,
		VectorType* randomSourceDirection,
		MediumPoint* randomFirstScatterer,
		ub32 samplesNum);

	virtual ErrorType firstPointProbability(float_type* probabilityRes,
		const MediumPoint* firstPoint,
		ub32 pointsNum) const;

	virtual ErrorType sampleSecondPoint(VectorType* sampledSourcePoint,
		MediumPoint* firstSampledPoint,
		MediumPoint* secondSampledPoint,
		ub32 samplesNum);

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
		bool isP1P2BeginsPaths) const {
		return ErrorType::NOT_SUPPORTED;
	};

	virtual ErrorType temporalTransferFunctionSingle(ComplexType* tsRes,
		const Source* source1,
		const MediumPoint* p0,
		const Source* source2,
		const Source* source3,
		ub32 pointsNum) const {
		return ErrorType::NOT_SUPPORTED;
	};

protected:
	ub32 allocationCount;
	ub32 beamSampleTableEntries;
	bool isFirstSampling;
	bool isRandomizedDirection;
	ub32 batchNum;

#if DIMS==3
	ub32 M;                                                         // maximal mixture number
	GaussianBeamNS::GaussianDistanceSampler* gaussianSamplingTable; // size of Nl * Nw

	MediumPoint* zBuffer;                                           // size of B * Nl * Nw

	float_type* pdfBuffer;                                          // size of M * B * Nl * Nw
	float_type* icdfBuffer;                                         // size of M * B * Nl * Nw

	float_type* alphaPdfBuffer;                                     // size of M * Nl * Nw
	float_type* alphaIcdfBuffer;                                    // size of M * Nl * Nw

	float_type* sampleProbabilitiesCdfBuffer;                       // size of M * M * P * Nl * Nw
	VectorType* real_conv_muBuffer;                                 // size of M * M * P * Nl * Nw
	float_type* log_CBuffer;                                        // size of M * M * P * Nl * Nw
	ub32* beamsNumbering;                                           // size of M * M * P * Nl * Nw
	float_type* sampleProbabilitiesCdfSumBuffer;                    // size of P * Nl * Nw
	ub32* tmpKeysBuffer;                                            // size of P * Nl * Nw
#endif
};

// ------------------------------------ Kernels ------------------------------------ //
namespace GaussianBeamNS
{
	__global__ void initGaussianBeamSource(MediumPoint* gaussianFocalPoint,
		VectorType* oppDir,
		const VectorType* inDir,
		const VectorType* focalPoints,
		ub32 totalDir)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalDir)
		{
			oppDir[pathNum] = (float_type)(-1.0) * inDir[pathNum];
			gaussianFocalPoint[pathNum].position = focalPoints[pathNum];
		}
	}

#if DIMS==3
	__global__ void randomizeViewDirection(VectorType* vec)
	{
		*vec = randomDirection(statePool);
	}

	// Attach to buffer and calculate tMin and tMax
	__global__ void initSamplingTable(GaussianDistanceSampler* gaussianSampler, MediumPoint* zBuffer,
		const MediumPoint* focalPoint, const VectorType* focalDirection, BlockingBox box,
		ub32 B, ub32 Nl, ub32 Nw)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 nl = threadNum % Nl;
		ub32 nw = threadNum / Nl;

		if (nw < Nw)
		{
			gaussianSampler[threadNum].focalPoint = focalPoint[nl];
			gaussianSampler[threadNum].focalPoint.lambdaIdx = nw;

			gaussianSampler[threadNum].zPoints = zBuffer + threadNum * B;

			VectorType p0 = focalPoint[nl].position;
			VectorType v = focalDirection[nl];

			projectBoxToLine(box, p0, v, &gaussianSampler[threadNum].tMin, &gaussianSampler[threadNum].tMax);
			gaussianSampler[threadNum].dt = (gaussianSampler[threadNum].tMax - gaussianSampler[threadNum].tMin) / (float_type)(B);

			// Compute xy base
			orthogonalBase(v, gaussianSampler[threadNum].xyBase);
		}
	}

	__global__ void fillTimesInSamplingTable(GaussianDistanceSampler* gaussianSampler, const MediumPoint* focalPoint,
		const VectorType* focalDirection, ub32 B, ub32 Nl, ub32 Nw)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 b = threadNum % B;
		ub32 nl = (threadNum / B) % Nl;
		ub32 nw = threadNum / (B * Nl);

		ub32 beamIdx = threadNum / B;

		if (nw < Nw)
		{
			float_type currentT = gaussianSampler[beamIdx].tMin + (float_type)(b)*gaussianSampler[beamIdx].dt;
			gaussianSampler[beamIdx].zPoints[b].position = focalPoint[nl].position + currentT * focalDirection[nl];

			//printf("%d - z pos: [%f %f %f], tMin = %f, dt = %f \n", threadNum,
			//	gaussianSampler[beamIdx].zPoints[b].position.x(), gaussianSampler[beamIdx].zPoints[b].position.y(), gaussianSampler[beamIdx].zPoints[b].position.z(),
			//	gaussianSampler[beamIdx].tMin, (float_type)(b)*gaussianSampler[beamIdx].dt);
		}
	}

	__global__ void getMixtureIdx(ub32* mixtureIdxSize, const MediumPoint* focalPoint, ub32 Nl)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < Nl)
		{
			const vmfScatteringNS::GaussianBeamFunction* currentStructre =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[focalPoint[threadNum].material];

			mixtureIdxSize[threadNum] = currentStructre->mixturesNum;
		}
	}

	__global__ void attachBuffersToSampler(GaussianDistanceSampler* gaussianSampler, float_type* pdfBuffer,
		float_type* icdfBuffer, float_type* alphaPdfBuffer, float_type* alphaIcdfBuffer,
		ub32 M, ub32 Nl, ub32 Nw, ub32 B)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < Nl * Nw)
		{
			gaussianSampler[threadNum].pdf = pdfBuffer + threadNum * M * B;
			gaussianSampler[threadNum].icdf = icdfBuffer + threadNum * M * B;
			gaussianSampler[threadNum].alphaPdf = alphaPdfBuffer + threadNum * M;
			gaussianSampler[threadNum].alphaIcdf = alphaIcdfBuffer + threadNum * M;
		}
	}

	__global__ void alphaPdfIcdfKernel(GaussianDistanceSampler* gaussianSampler, ub32 beamsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		if (threadNum < beamsNum)
		{
			const vmfScatteringNS::GaussianBeamFunction* currentStructre =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[gaussianSampler[threadNum].focalPoint.material];

			ub32 mixNum = currentStructre->mixturesNum;
			float_type alphaSum = (float_type)0.;
			
			// compute sum all alpha values
			for (ub32 alphaNum = 0; alphaNum < mixNum; alphaNum++)
			{
				float_type kappa = abs(currentStructre->mixtureMu[alphaNum]);
				float_type c = currentStructre->mixtureC[alphaNum];
				float_type log_vMF_norm_factor = kappa < 1e-5 ? -log(4.0 * CUDART_PI) :
					log(kappa) - (kappa > 10.0 ? log(2.0 * CUDART_PI) * kappa : log(4.0 * CUDART_PI * sinh(kappa)));
				float_type alpha = exp(c + log_vMF_norm_factor);

				alphaSum += alpha;
				gaussianSampler[threadNum].alphaPdf[alphaNum] = alpha;
			}

			// compute alpha pdf
			for (ub32 alphaNum = 0; alphaNum < mixNum; alphaNum++)
			{
				gaussianSampler[threadNum].alphaPdf[alphaNum] /= alphaSum;
			}

			// compute icdf
			for (ub32 alphaNum = 0; alphaNum < mixNum; alphaNum++)
			{
				float_type cdf = (float_type)0.;
				for (ub32 alphaNum_inner = 0; alphaNum_inner < alphaNum; alphaNum_inner++)
				{
					cdf += gaussianSampler[threadNum].alphaPdf[alphaNum_inner];
				}
				gaussianSampler[threadNum].alphaIcdf[alphaNum] = cdf;
			}
		}
	}

	__global__ void samplingPdfKernel(GaussianDistanceSampler* gaussianSampler, const float_type* allAttenuationBuffer,
		const VectorType* beamDirections, float_type gaussianAperture, ub32 beamSampleTableEntries, ub32 beamsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 entryNum = threadNum % beamSampleTableEntries;
		ub32 currentBeam = threadNum / beamSampleTableEntries;

		if (currentBeam < beamsNum)
		{
			const vmfScatteringNS::GaussianBeamFunction* currentStructre =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[gaussianSampler[currentBeam].focalPoint.material];

			float_type k = (float_type)2.0 * CUDART_PI / lambdaValues[gaussianSampler[currentBeam].focalPoint.lambdaIdx];
			float_type dz = (gaussianSampler[currentBeam].zPoints[entryNum].position - gaussianSampler[currentBeam].focalPoint.position) * beamDirections[currentBeam];

			for (ub32 alphaNum = 0; alphaNum < currentStructre->mixturesNum; alphaNum++)
			{
				float_type gamma_s = abs(currentStructre->mixtureMu[alphaNum]);
				float_type wz_square = (gaussianAperture + gamma_s) / (k * k) + (dz * dz) / gaussianAperture;
				gaussianSampler[currentBeam].pdf[entryNum + alphaNum * beamSampleTableEntries] = allAttenuationBuffer[threadNum] / wz_square;

				//printf("%d %d: z = [%f %f %f], dz = %f, k = %f, gaussianAperture = %f, gamma_s = %f, attenuation = %f \n",
				//	threadNum, alphaNum, gaussianSampler[currentBeam].zPoints[entryNum].position.x(), gaussianSampler[currentBeam].zPoints[entryNum].position.y(),
				//	gaussianSampler[currentBeam].zPoints[entryNum].position.z(), dz, k, gaussianAperture, gamma_s, allAttenuationBuffer[threadNum]);
			}
		}
	}

	__global__ void markBeamsKernel(ub32* markers, ub32 beamSampleTableEntries, ub32 totalThreads)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < totalThreads)
		{
			markers[threadNum] = threadNum / beamSampleTableEntries;
		}
	}

	__global__ void getPdfSum(float_type* pdfSum, const float_type* pdf, const float_type* icdf, ub32 beamSampleTableEntries, ub32 totalThreads)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < totalThreads)
		{
			pdfSum[threadNum] = pdf[(threadNum + 1) * beamSampleTableEntries - 1] + icdf[(threadNum + 1) * beamSampleTableEntries - 1];
		}
	}

	__global__ void normalizeSamplingBuffers(float_type* pdf, float_type* icdf, const float_type* pdfSum, ub32 beamSampleTableEntries, ub32 totalThreads)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < totalThreads)
		{
			pdf[threadNum] *= (((float_type)(beamSampleTableEntries - 1)) / pdfSum[threadNum / beamSampleTableEntries]);
			icdf[threadNum] /= pdfSum[threadNum / beamSampleTableEntries];
		}
	}

	__global__ void sampleFirstBeam(VectorType* randomChosenFocalPoint,
		MediumPoint* randomScatterer,
		const GaussianDistanceSampler* gaussianSamplingTable,
		ub32 totalPathNum,
		ub32 beamSampleTableEntries,
		Source::GaussianBeamStructType gaussianBeamSourceStruct)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathNum)
		{
			curandState_t* currentState = statePool + pathNum;

			// randomize an illuminator among all options
			ub32 illuminatorNum = randUniformInteger(currentState, gaussianBeamSourceStruct.beamsNum);
			ub32 wavelenghSampledNum = randUniformInteger(currentState, lambdaNum);

			ub32 currntEntryInSamplingTable = illuminatorNum + wavelenghSampledNum * gaussianBeamSourceStruct.beamsNum;
			const GaussianDistanceSampler* gaussianSamplingTableEntry = gaussianSamplingTable + currntEntryInSamplingTable;

			MediumPoint currentFocalPoint = gaussianSamplingTableEntry->focalPoint;
			const vmfScatteringNS::GaussianBeamFunction* currentStructre =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[currentFocalPoint.material];

			// Choose random mixture
			ub32 mixNum = binarySearchKernel(gaussianSamplingTableEntry->alphaIcdf,
				currentStructre->mixturesNum, curand_uniform(currentState) );
			
			// Choose random z point
			float_type randNum = curand_uniform(currentState);
			ub32 randomCdfIdx = binarySearchKernel(gaussianSamplingTableEntry->icdf + mixNum * beamSampleTableEntries,
				beamSampleTableEntries, randNum);

			// Move z point randomly in dt range
			VectorType zBase = gaussianBeamSourceStruct.beamDirections[illuminatorNum];
			VectorType z = gaussianSamplingTableEntry->zPoints[randomCdfIdx].position +
				curand_uniform(currentState) * gaussianSamplingTableEntry->dt * zBase;

			// Compute xy shift
			float_type k = (float_type)2.0 * CUDART_PI / lambdaValues[wavelenghSampledNum];
			float_type gamma_a = gaussianBeamSourceStruct.gaussianAperture;
			float_type gamma_s = abs(currentStructre->mixtureMu[mixNum]);
			float_type dz = (z - currentFocalPoint.position) * zBase;
			float_type wz = sqrt((gamma_a + gamma_s) / (k * k) + dz * dz / gamma_a);
			VectorType randScat = z + wz * (
				randNormal(currentState) * gaussianSamplingTableEntry->xyBase[0] +
				randNormal(currentState) * gaussianSamplingTableEntry->xyBase[1]);

			//printf("%d: mixNum: %d, randomCdfIdx: %d, zBase:[%f, %f, %f], z:[%f, %f, %f] currentFocalPoint:[%f, %f, %f], k = %f, gamma_a = %f, gamma_s = %f, dz = %f, wz = %f, randScat:[%f, %f, %f], xyBase[0] [%f %f %f], xyBase[1] [%f %f %f], randNum %f, iilumNum = %d, beamsNum = %d \n",
			//	pathNum, mixNum, randomCdfIdx,
			//	zBase.x(), zBase.y(), zBase.z(), z.x(), z.y(), z.z(), currentFocalPoint.position.x(), currentFocalPoint.position.y(), currentFocalPoint.position.z(), k, gamma_a, gamma_s,
			//	dz, wz, randScat.x(), randScat.y(), randScat.z(), gaussianSamplingTableEntry->xyBase[0].x(), gaussianSamplingTableEntry->xyBase[0].y(), gaussianSamplingTableEntry->xyBase[0].z(),
			//	gaussianSamplingTableEntry->xyBase[1].x(), gaussianSamplingTableEntry->xyBase[1].y(), gaussianSamplingTableEntry->xyBase[1].z(), randNum, illuminatorNum, gaussianBeamSourceStruct.beamsNum);
			
			// Take the first scattering point as a point in the opposite direction
			randomChosenFocalPoint[pathNum] = randScat + gaussianBeamSourceStruct.beamOppositeDirections[illuminatorNum];
			randomScatterer[pathNum].position = randScat;
			randomScatterer[pathNum].lambdaIdx = wavelenghSampledNum;
			randomScatterer[pathNum].material = currentFocalPoint.material;
			randomScatterer[pathNum].sourceIdx = currntEntryInSamplingTable;
		}
	}

	__global__ void firstPointProbabilityKernel(float_type* probabilityRes,
		const GaussianDistanceSampler* gaussianSamplingTable,
		const MediumPoint* firstPoint,
		ub32 pointsNum,
		Source::GaussianBeamStructType gaussianBeamStruct)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 sourceNum = threadNum % gaussianBeamStruct.beamsNum;
		ub32 wavelenghNum = (threadNum / gaussianBeamStruct.beamsNum) % lambdaNum;
		ub32 scattererNum = threadNum / (gaussianBeamStruct.beamsNum * lambdaNum);

		if (scattererNum < pointsNum)
		{
			ub32 beamNum = sourceNum + wavelenghNum * gaussianBeamStruct.beamsNum;

			VectorType beamCenter = gaussianBeamStruct.focalPoints[sourceNum].position;
			VectorType beamDirection = gaussianBeamStruct.beamDirections[sourceNum];
			VectorType currentScatterer = firstPoint[scattererNum].position;

			float_type gamma_a = gaussianBeamStruct.gaussianAperture;
			float_type k = (float_type)2.0 * CUDART_PI / lambdaValues[wavelenghNum];
			
			const vmfScatteringNS::GaussianBeamFunction* currentGaussianBeamGpuData =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[gaussianBeamStruct.focalPoints[sourceNum].material];
			
			float_type dz = pointToLineProjectionDistance(currentScatterer, beamCenter, beamDirection);
			VectorType zPointScattererVector = beamCenter + dz * beamDirection - currentScatterer;
			float_type dxy2 = zPointScattererVector * zPointScattererVector;
			
			ub32 zIdx;
			if (dz < gaussianSamplingTable[beamNum].tMin)
			{
				zIdx = 0;
			}
			else if (dz > gaussianSamplingTable[beamNum].tMax)
			{
				zIdx = gaussianBeamStruct.beamSampleTableEntries - 1;
			}
			else
			{
				zIdx = (ub32)((dz - gaussianSamplingTable[beamNum].tMin) / gaussianSamplingTable[beamNum].dt);
			}

			float_type pdf = 0.;

			for (ub32 mixIdx = 0; mixIdx < currentGaussianBeamGpuData->mixturesNum; mixIdx++)
			{
				// z idx probability
				float_type pz = gaussianSamplingTable[beamNum].pdf[zIdx + gaussianBeamStruct.beamSampleTableEntries * mixIdx];

				// xy distance probability
				float_type gamma_s = abs(currentGaussianBeamGpuData->mixtureMu[mixIdx]);
				float_type wz_squared = (gamma_a + gamma_s) / (k * k) + dz * dz / gamma_a;
				float_type px_py = exp(-dxy2 / (2.0 * wz_squared)) / ((2.0 * wz_squared) * CUDART_PI);

				pdf += gaussianSamplingTable[beamNum].alphaPdf[mixIdx] * pz * px_py;

				//printf("pdf %d: mixNum: %d, currentScatterer = [%f %f %f], beamCenter = [%f %f %f], beamDirection = [%f %f %f], dz = %f, zPointScattererVector = [%f %f %f], dxy2 = %f, zIdx = %d, pz = %f, px_py = %f, alphaPdf = %f, pdf = %f \n",
				//	threadNum, mixIdx, currentScatterer.x, currentScatterer.y, currentScatterer.z, beamCenter.x, beamCenter.y, beamCenter.z, beamDirection.x, beamDirection.y, beamDirection.z,
				//	dz, zPointScattererVector.x, zPointScattererVector.y, zPointScattererVector.z, dxy2, zIdx, pz, px_py, gaussianSamplingTable[beamNum].alphaPdf[mixIdx], gaussianSamplingTable[beamNum].alphaPdf[mixIdx] * pz * px_py);
			}

			probabilityRes[threadNum] = pdf  / (MediumNS::mediumGpuData[firstPoint[scattererNum].material].sigs * (MediumNS::boxMax.z() - MediumNS::boxMin.z()));
		}

	}

	template <bool isRaySampled>
	__global__ void computeCdfBuffer(const MediumPoint* p0,
		Source::GaussianBeamStructType gaussianBeamStruct,
		float_type* sampleProbabilitiesCdfBuffer,
		VectorType* real_conv_muBuffer,
		float_type* log_CBuffer,
		ub32 M, ub32 P, ub32 Nl, ub32 totalThreads)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		
		if (threadNum < totalThreads)
		{
			ub32 mixtureIdx = threadNum % (M * M);
			ub32 p = (threadNum / (M * M)) % P;
			ub32 nl, nw, beamNum;

			if (isRaySampled)
			{
				beamNum = p0[p].sourceIdx;
				nl = beamNum % Nl;
				nw = beamNum / Nl;
			}
			else
			{
				nl = (threadNum / (M * M * P)) % Nl;
				nw = threadNum / (M * M * P * Nl);
				beamNum = nl + nw * Nl;
			}

			const vmfScatteringNS::GaussianBeamFunction* currentGaussianBeamGpuData =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[gaussianBeamStruct.focalPoints[nl].material];

			ub32 mixturesNum = currentGaussianBeamGpuData->mixturesNum;
			float_type alpha = 0.;

			if (mixtureIdx < mixturesNum * mixturesNum)
			{
				VectorType x0 = p0[p].position;
				float_type k = (float_type)2.0 * CUDART_PI / lambdaValues[nw];

				// The complex throughput
				ComplexVectorType throughput = ComplexVectorType(gaussianBeamStruct.beamDirections[nl] * gaussianBeamStruct.gaussianAperture,
					k * (x0 - gaussianBeamStruct.focalPoints[nl].position));

				ub32 mixtureIdx1 = mixtureIdx % mixturesNum;
				ub32 mixtureIdx2 = mixtureIdx / mixturesNum;

				float_type gamma_s = currentGaussianBeamGpuData->mixtureMu[mixtureIdx1];
				float_type real_c_1;
				VectorType real_conv_mu_1(0.0);

				//	printf("%d: mixtureIdx = %d, p = %d, nl =  %d, nw = %d, beamNum = %d gamma_s_1 = %f \n", threadNum, mixtureIdx, p, nl, nw, beamNum, gamma_s);

				if (abs(gamma_s) < 0.000000001)
				{
					ComplexType C_tmp = complexSqrt(sumSquare(throughput));
					real_c_1 = C_tmp.real() - realComplexLog(C_tmp) + currentGaussianBeamGpuData->mixtureC[mixtureIdx1];
				}
				else
				{
					ComplexType gamma_s_over_beta_0 = gamma_s * rComplexSqrt(sumSquare(cfma(gamma_s, meanViewDirection, throughput)));

					real_conv_mu_1 = realMult(gamma_s_over_beta_0, throughput);

					float_type rkappa = rabs(real_conv_mu_1);
					VectorType w = rkappa * real_conv_mu_1;

					ComplexType C_tmp = complexSqrt(sumSquare(cfma(gamma_s, w, throughput)));
					real_c_1 = C_tmp.real() - realComplexLog(C_tmp) + currentGaussianBeamGpuData->mixtureC[mixtureIdx1] - w * real_conv_mu_1;
				}

				gamma_s = currentGaussianBeamGpuData->mixtureMu[mixtureIdx2];
				float_type real_c_2;
				VectorType real_conv_mu_2(0.0);

				if (abs(gamma_s) < 0.000000001)
				{
					ComplexType C_tmp = complexSqrt(sumSquare(throughput));
					real_c_2 = C_tmp.real() - realComplexLog(C_tmp) + currentGaussianBeamGpuData->mixtureC[mixtureIdx2];
				}
				else
				{
					ComplexType gamma_s_over_beta_0 = gamma_s * rComplexSqrt(sumSquare(cfma(gamma_s, meanViewDirection, throughput)));
					real_conv_mu_2 = realMult(gamma_s_over_beta_0, throughput);

					float_type rkappa = rabs(real_conv_mu_2);
					VectorType w = rkappa * real_conv_mu_2;

					ComplexType C_tmp = complexSqrt(sumSquare(cfma(gamma_s, w, throughput)));

					real_c_2 = C_tmp.real() - realComplexLog(C_tmp) + currentGaussianBeamGpuData->mixtureC[mixtureIdx2] - w * real_conv_mu_2;
				}

				// multiple two mixtures
				VectorType real_conv_mu = real_conv_mu_1 + real_conv_mu_2;
				float_type real_c = real_c_1 + real_c_2;

				// cdf (not normalized) for each mixture
				float_type kappa = abs(real_conv_mu);
				float_type log_C;

				if (kappa < 0.0001)
				{
					log_C = 1.0 / (4.0 * CUDART_PI);
				}
				else
				{
					// log_C = 0.5 * log(kappa) - log((float_type)(2.0 * CUDART_PI)) * (float_type)(1.5) - logbesseli_05(kappa);
					log_C = vMFnormalizationLog(kappa);
				}

				alpha = exp(real_c - log_C - 2.0 * gaussianBeamStruct.gaussianAperture);

				real_conv_muBuffer[threadNum] = real_conv_mu;

				if (!isRaySampled)
				{
					log_CBuffer[threadNum] = log_C;
				}
			}

			sampleProbabilitiesCdfBuffer[threadNum] = alpha;


			//printf("%d: %d %d %d cdf: %e, mu: [%f %f %f] log_C: %f, gamma_s_2 = %f, real_conv_mu_1 = [%f %f %f], real_conv_mu_2 = [%f %f %f], real_c_1: %f, real_c_2: %f \n",
			//	threadNum, beamNum, mixtureIdx, mixtureIdx + p * mixturesNum * mixturesNum,
			//	gaussianSamplingTable[beamNum].sampleProbabilitiesCdf[mixtureIdx + p * mixturesNum * mixturesNum],
			//	gaussianSamplingTable[beamNum].real_conv_mu[mixtureIdx + p * mixturesNum * mixturesNum].x,
			//	gaussianSamplingTable[beamNum].real_conv_mu[mixtureIdx + p * mixturesNum * mixturesNum].y,
			//	gaussianSamplingTable[beamNum].real_conv_mu[mixtureIdx + p * mixturesNum * mixturesNum].z,
			//	gaussianSamplingTable[beamNum].log_C[mixtureIdx + p * mixturesNum * mixturesNum], gamma_s,
			//	real_conv_mu_1.x, real_conv_mu_1.y, real_conv_mu_1.z, real_conv_mu_2.x, real_conv_mu_2.y, real_conv_mu_2.z, real_c_1, real_c_2);
		}
	}

	__global__ void normalizeCdfBuffer(float_type* sampleProbabilitiesCdfBuffer,
		const float_type* sampleProbabilitiesCdfSumBuffer,
		ub32 M,
		ub32 totalThreads)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < totalThreads)
		{
			sampleProbabilitiesCdfBuffer[threadNum] /= sampleProbabilitiesCdfSumBuffer[threadNum / (M * M)];
		}
	}

	__global__ void sampleCdfBuffer(VectorType* dirSapmled,
		const MediumPoint* firstFocalPoint,
		const float_type* sampleProbabilitiesCdfBuffer,
		const VectorType* real_conv_muBuffer,
		ub32 M,
		ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < pointsNum)
		{
			const vmfScatteringNS::GaussianBeamFunction* currentGaussianBeamGpuData =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[firstFocalPoint[threadNum].material];

			ub32 mixturesNum = currentGaussianBeamGpuData->mixturesNum;

			curandState_t* currentState = statePool + threadNum;
			float_type randomNum = randUniform(currentState);

			ub32 sapmledIdx;
			float_type cdfSum = 0.;

			// Lazy CDF summing - to make sure we don't pass the last mixture
			for (sapmledIdx = 0; sapmledIdx < (mixturesNum * mixturesNum); sapmledIdx++)
			{
				if (randomNum < cdfSum)
				{
					break;
				}

				cdfSum += sampleProbabilitiesCdfBuffer[sapmledIdx + M * M * threadNum];
			}
			sapmledIdx--;

			VectorType real_conv_mu = real_conv_muBuffer[sapmledIdx + M * M * threadNum];
			// float_type log_C = gaussianSamplingTable[beamNum].log_C[sapmledIdx + mixturesNum * mixturesNum * threadNum];
			float_type kappa = abs(real_conv_mu);
			
			if (kappa < 0.0001)
			{
				real_conv_mu = VectorType(0.0, 0.0, 1.0);
			}
			else
			{
				real_conv_mu = real_conv_mu * ((float_type)1.0 / kappa);
			}

			// sample a direction
			dirSapmled[threadNum] = random_vMF_direction(real_conv_mu, kappa, currentState);

			//printf("%d: randomNum = %f, sapmledIdx = %d, kappa = %f, real_conv_mu = [%f %f %f], dirSapmled = [%f %f %f] \n",
			//	threadNum, randomNum, sapmledIdx, kappa, real_conv_mu.x(), real_conv_mu.y(), real_conv_mu.z(),
			//	dirSapmled[threadNum].x(), dirSapmled[threadNum].y(), dirSapmled[threadNum].z());
		}
	}

	// We multiply the current probability with the sampled direction probability
	__global__ void sampleDirectionProbability(float_type* sampledProbability,
		const MediumPoint* firstPoint,
		const VectorType* secondPoint,
		const float_type* sampleProbabilitiesCdfBuffer,
		const VectorType* real_conv_muBuffer,
		const float_type* log_CBuffer,
		ub32 M,
		ub32 beamsNum,
		ub32 pointsNum)
	{

		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 sourceNum = threadNum % beamsNum;
		ub32 wavelenghNum = (threadNum / beamsNum) % lambdaNum;
		ub32 scattererNum = threadNum / (beamsNum * lambdaNum);

		if (scattererNum < pointsNum)
		{
			ub32 beamNum = sourceNum + wavelenghNum * beamsNum;

			const vmfScatteringNS::GaussianBeamFunction* currentGaussianBeamGpuData =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[firstPoint[scattererNum].material];

			ub32 mixturesNum = currentGaussianBeamGpuData->mixturesNum;

			float_type current_pw0 = 0;

			VectorType dirSapmled = normalize(secondPoint[scattererNum] - firstPoint[scattererNum].position);

			for (ub32 mixNum = 0; mixNum < (mixturesNum * mixturesNum); mixNum++)
			{
				ub32 bufferIdxNum = mixNum + scattererNum * M * M + beamNum * M * M * pointsNum;
				VectorType real_conv_mu = real_conv_muBuffer[bufferIdxNum];
				float_type log_C = log_CBuffer[bufferIdxNum];
				float_type kappa = abs(real_conv_mu);

				float_type muTimesX = real_conv_mu * dirSapmled;

				current_pw0 += sampleProbabilitiesCdfBuffer[bufferIdxNum] * exp(muTimesX + log_C);

				//printf("%d %d %d %d: real_conv_mu = [%f,%f,%f], log_C: %f, cdf: %e \n", threadNum, mixNum, beamNum, scattererNum,
				//	real_conv_mu.x(), real_conv_mu.y(), real_conv_mu.z(), log_C, sampleProbabilitiesCdfBuffer[bufferIdxNum]);
			}

			//printf("%d: dirSapmled = [%f,%f,%f], scatterer: [%f %f %f], scattererProbability: %e, directionProbability: %e, probability: %e \n", threadNum, 
			//	dirSapmled.x(), dirSapmled.y(), dirSapmled.z(), firstPoint[scattererNum].position.x(), firstPoint[scattererNum].position.y(), firstPoint[scattererNum].position.z(),
			//	sampledProbability[threadNum], current_pw0, sampledProbability[threadNum] * current_pw0);

			float_type updateProbability = sampledProbability[threadNum] * current_pw0;

			if (updateProbability == 0)
			{
				// Kill the sample
				updateProbability = 1e20;

			}

			sampledProbability[threadNum] = updateProbability;
		}
	}
#else
	__global__ void sampleFirstBeam(VectorType* chosenSource,
		VectorType* randomChosenDirection,
		ub32 totalPathNum,
		Source::GaussianBeamStructType gaussianBeamStruct)
	{
		ub32 pathNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (pathNum < totalPathNum)
		{
			curandState_t* currentState = statePool + pathNum;

			// randomize an illuminator among all beams
			ub32 illuminatorNum = randUniformInteger(currentState, gaussianBeamStruct.beamsNum);
			chosenSource[pathNum] = gaussianBeamStruct.focalPoints[illuminatorNum].position;

			// sample a random direction
			randomChosenDirection[pathNum] = randomDirection(currentState);
		}
	}

	__global__ void probabilityPointNormalizeKernel(float_type* probabilityRes,
		const MediumPoint* firstPoint,
		ub32 pointsNum,
		Source::GaussianBeamStructType gaussianBeamStruct)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 sourceNum = threadNum % gaussianBeamStruct.beamsNum;
		// ub32 wavelenghNum = (threadNum / pointSourceStruct.pointsNum) % lambdaNum;
		ub32 scattererNum = threadNum / (gaussianBeamStruct.beamsNum * lambdaNum);

		if (scattererNum < pointsNum)
		{
			float_type rr = rabs(firstPoint[scattererNum].position - gaussianBeamStruct.focalPoints[sourceNum].position);;
			probabilityRes[threadNum] *= (rr * (float_type(8.0 * CUDART_PI)));
		}

	}
#endif

	__global__ void gGaussianBeam(ComplexType* gRes,
		const MediumPoint* p1,
		ub32 pointsNum,
		ub32 beamsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		// ub32 sourceNum = threadNum % beamsNum;
		ub32 scattererNum = threadNum / beamsNum;

		if (scattererNum < pointsNum)
		{
			if (p1[scattererNum].material > 0)
			{
				gRes[threadNum] = sqrt(gRes[threadNum].real());
			}
			else
			{
				gRes[threadNum] = 0;
			}
		}
	}

	__global__ void fGaussianBeam(ComplexType* fRes,
		const MediumPoint* p1, // p1 is the central scattering point, so scattering goes from beam -> p1 -> p2
		const VectorType* p2,
		Source::GaussianBeamStructType gaussianBeamStruct,
		ub32 pointsNum,
		ConnectionType cType)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 sourceNum = threadNum % gaussianBeamStruct.beamsNum;
		ub32 scattererNum = threadNum / gaussianBeamStruct.beamsNum;

		if (scattererNum < pointsNum)
		{
			if (p1[scattererNum].material > 0)
			{
				const vmfScatteringNS::GaussianBeamFunction* currentGaussianBeamGpuData =
					(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[p1[scattererNum].material];

				VectorType x0 = p1[scattererNum].position;
				VectorType w0;

				if (cType == ConnectionType::ConnectionTypeIllumination)
				{
					w0 = normalize(p2[scattererNum] - x0);
				}
				else
				{
					w0 = normalize(x0 - p2[scattererNum]);
				}

				float_type k = cType == ConnectionType::ConnectionTypeIllumination ?
					(float_type)2.0 * CUDART_PI / lambdaValues[p1[scattererNum].lambdaIdx] :
					-(float_type)2.0 * CUDART_PI / lambdaValues[p1[scattererNum].lambdaIdx];

				// The complex throughput
				ComplexVectorType throughput = ComplexVectorType(gaussianBeamStruct.beamDirections[sourceNum] * gaussianBeamStruct.gaussianAperture,
					k * (x0 - gaussianBeamStruct.focalPoints[sourceNum].position));

				float_type throughput_c = gaussianBeamStruct.apertureNormalization + log((float_type)(2.0 * CUDART_PI));
				ComplexType fCalculated;

				for (ub32 mixtureNum = 0; mixtureNum < currentGaussianBeamGpuData->mixturesNum; mixtureNum++)
				{
					float_type gamma_s = currentGaussianBeamGpuData->mixtureMu[mixtureNum];
					float_type log_nu = throughput_c + currentGaussianBeamGpuData->mixtureC[mixtureNum];

					ComplexType sqrtMu = complexSqrt(sumSquare(cfma(gamma_s, w0, throughput)));

#if DIMS == 3
					if (abs(sqrtMu.real()) < 0.0001 && abs(sqrtMu.imag()) < 0.0001)
					{
						fCalculated = fCalculated + ((float_type)2.0) * exp(log_nu);
					}
					else
					{
						fCalculated = fCalculated + ((complex2timesSinh(sqrtMu, log_nu)) / sqrtMu);
					}
#else
					fCalculated = fCalculated + complexExponent(logbesseli(0, sqrtMu) + log_nu);
#endif

					//printf("%d %d: throughput = [%f+%fi,%f+%fi,%f+%fi], log_nu = %f, sqrtMu = %f+%fi \n",
					//	threadNum, mixtureNum, throughput[0].x, throughput[0].y, throughput[1].x, throughput[1].y, throughput[2].x, throughput[2].y,
					//	log_nu, sqrtMu.x, sqrtMu.y);
				}

				fRes[threadNum] = fCalculated;
			}
			else
			{
				fRes[threadNum] = 0;
			}
		}
	}

	__global__ void fsGaussianBeam(ComplexType* fsRes,
		const MediumPoint* p0,
		Source::GaussianBeamStructType thisGaussianBeamStruct,
		Source::GaussianBeamStructType otherGaussianBeamStruct,
		ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		ub32 aSourceSize = thisGaussianBeamStruct.beamsNum;
		ub32 bSourceSize = otherGaussianBeamStruct.beamsNum;

		ub32 aSourceNum = threadNum % aSourceSize;
		ub32 bSourceNum = (threadNum / aSourceSize) % bSourceSize;
		ub32 scattererNum = threadNum / (aSourceSize * bSourceSize);

		if (scattererNum < pointsNum)
		{
			const vmfScatteringNS::GaussianBeamFunction* currentGaussianBeamGpuData =
				(const vmfScatteringNS::GaussianBeamFunction*)ScatteringNS::scatteringParameters[p0[scattererNum].material];

			VectorType x0 = p0[scattererNum].position;
			float_type k = (float_type)2.0 * CUDART_PI / lambdaValues[p0[scattererNum].lambdaIdx];

			// The complex throughput
			ComplexVectorType aThroughput = ComplexVectorType(thisGaussianBeamStruct.beamDirections[aSourceNum] * thisGaussianBeamStruct.gaussianAperture,
				k * (x0 - thisGaussianBeamStruct.focalPoints[aSourceNum].position));

			ComplexVectorType bThroughput = ComplexVectorType(otherGaussianBeamStruct.beamDirections[bSourceNum] * otherGaussianBeamStruct.gaussianAperture,
				k * (otherGaussianBeamStruct.focalPoints[bSourceNum].position - x0));

			float_type throughput_c = thisGaussianBeamStruct.apertureNormalization + otherGaussianBeamStruct.apertureNormalization + (float_type)(2.0) * log((float_type)(2.0 * CUDART_PI));
			ComplexType fsCalculated;

			for (ub32 mixtureNum = 0; mixtureNum < currentGaussianBeamGpuData->mixturesNum; mixtureNum++)
			{
				float_type gamma_s = currentGaussianBeamGpuData->mixtureMu[mixtureNum];
				float_type log_nu = throughput_c + currentGaussianBeamGpuData->mixtureC[mixtureNum];

				ComplexType C, sqrtMu, c;

				// Convolution with the illumination throughput
				if (abs(gamma_s) < 0.000001)
				{
					// c = ComplexType;
					C = complexSqrt(sumSquare(aThroughput));
					sqrtMu = complexSqrt(sumSquare(bThroughput));

					//printf("-- 1 -- %d,%d: log_nu = %f, gamma_s = %f, c = %f+%fi, C = %f+%fi, sqrtMu = %f+%fi \n", threadNum, mixtureNum, log_nu,
					//	gamma_s, c.x, c.y, C.x, C.y, sqrtMu.x, sqrtMu.y);
				}
				else
				{
					ComplexType gamma_s_over_beta_0 = gamma_s * rComplexSqrt(
						sumSquare(cfma(gamma_s, otherGaussianBeamStruct.beamDirections[bSourceNum], aThroughput)));

					sqrtMu = complexSqrt(sumSquare(cfma(gamma_s_over_beta_0, aThroughput, bThroughput)));

					VectorType w = realMult(gamma_s_over_beta_0, aThroughput);

					float_type real_abs_mu = rabs(w);

					gamma_s *= real_abs_mu;

					C = complexSqrt(sumSquare(cfma(gamma_s, w, aThroughput)));
					c = (-real_abs_mu) * (gamma_s_over_beta_0) * (aThroughput * w);

					//printf("-- 2 -- %d,%d: log_nu = %f, gamma_s = %f, c = %f+%fi, C = %f+%fi, sqrtMu = %f+%fi, gamma_s_over_beta_0 = %f+%fi, real_abs_mu = %f, w = [%f %f %f] \n",
					//	threadNum, mixtureNum, log_nu,
					//	gamma_s, c.x, c.y, C.x, C.y, sqrtMu.x, sqrtMu.y, gamma_s_over_beta_0.x, gamma_s_over_beta_0.y, real_abs_mu, w.x, w.y, w.z);
				}

				// integrate
#if DIMS == 3
				if ((abs(sqrtMu.real()) > 0.0001 || abs(sqrtMu.imag()) > 0.0001) && (abs(C.real()) > 0.0001 || abs(C.imag()) > 0.0001))
				{
					fsCalculated = fsCalculated + complex2timesSinh(C + sqrtMu, c + log_nu) / (C * sqrtMu);
				}
				else
				{
					if (abs(sqrtMu.real()) < 0.0001 && abs(sqrtMu.imag()) < 0.0001)
					{
						if (abs(C.real()) < 0.0001 && abs(C.imag()) < 0.0001)
						{
							fsCalculated = fsCalculated + ((float_type)4.0) * complexExponent(c + log_nu);
						}
						else
						{
							fsCalculated = fsCalculated + ((float_type)2.0) * complex2timesSinh(C, c + log_nu) / (C);
						}
					}
					else
					{
						fsCalculated = fsCalculated + ((float_type)2.0) * complex2timesSinh(sqrtMu, log_nu) / (sqrtMu);
					}
				}
#else
				fsCalculated = fsCalculated + complexExponent(logbesseli(0, C) + logbesseli(0, sqrtMu) + c + log_nu);
#endif
			}

			fsRes[threadNum] = fsCalculated;

			//printf("fs %d: aSourceNum - %d, bSourceNum - %d, scattererNum - %d, k = %f, x0 = [%f %f %f] fs = %f+%fi \n", threadNum, aSourceNum, bSourceNum, scattererNum,
			//	k, x0.x, x0.y, x0.z, fsCalculated.x, fsCalculated.y);
		}
	}

	__global__ void isotropicGaussianSampling(VectorType* dirSapmled,
		ub32 pointsNum)
	{
		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

		if (threadNum < pointsNum)
		{
			curandState_t* currentState = statePool + threadNum;
			dirSapmled[threadNum] = randomDirection(currentState);
		}
	}

	__global__ void isotropicGaussianProbability(float_type* sampledProbability,
		ub32 beamsNum,
		ub32 pointsNum)
	{

		ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
		ub32 scattererNum = threadNum / (beamsNum * lambdaNum);

		if (scattererNum < pointsNum)
		{
			sampledProbability[threadNum] *= ISOTROPIC_PDF;
		}
	}
}

// ------------------------------------ Class Constructor & Destructor Implementations ------------------------------------ //
GaussianBeam::GaussianBeam(ErrorType* err, const Simulation* simulationHandler, const Medium* mediumHandler, const Scattering* scatteringHandler,
	const VectorType* focalPoints, const VectorType* beamDirections, float_type gaussianAperture, ub32 beamsNum,
	ConnectionType cType, bool apertureNormalized, ub32 zSamplesNum) :
	Source(simulationHandler, mediumHandler, scatteringHandler, cType),
	beamSampleTableEntries(zSamplesNum),
	batchNum(simulationHandler->getBatchSize())
{
	MEMORY_CHECK("Gaussian beam allocation begin");
	sourceType = SourceType::GaussianBeamType;
	sourceSize = beamsNum;
	allocationCount = 0;
	isFirstSampling = true;

	// All size which will be used here
	ub32 Nl = sourceSize;

	GaussianBeamStructType* sourceData;

	sourceData = (GaussianBeamStructType*)malloc(sizeof(GaussianBeamStructType));

	if (sourceData == 0)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	sourceData->beamsNum = beamsNum;
	// Gaussian to vMF parameter
	sourceData->gaussianAperture = 1.0 / (gaussianAperture * gaussianAperture);
	sourceData->beamSampleTableEntries = beamSampleTableEntries;

	if (apertureNormalized)
	{	
		sourceData->apertureNormalization = vMFnormalizationLog(sourceData->gaussianAperture);
	}
	else
	{
		sourceData->apertureNormalization = -sourceData->gaussianAperture;
	}

	// allocate the Gaussian Beam sources
	if (cudaMalloc(&sourceData->focalPoints, sizeof(MediumPoint) * Nl) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&sourceData->beamDirections, sizeof(VectorType) * Nl) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMalloc(&sourceData->beamOppositeDirections, sizeof(VectorType) * Nl) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	allocationCount++;

	if (cudaMemcpy(sourceData->beamDirections, beamDirections, sizeof(VectorType) * Nl, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
#if DIMS == 2
	if (cudaMalloc(&sourceData->focalPointsVector, sizeof(VectorType) * Nl) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	if (cudaMemcpy(sourceData->focalPointsVector, focalPoints, sizeof(VectorType) * Nl, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
#endif
	allocationCount++;

	data = sourceData;
	VectorType* focalPointsGPU;
	if (cudaMalloc(&focalPointsGPU, sizeof(VectorType) * Nl) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}
	if (cudaMemcpy(focalPointsGPU, focalPoints, sizeof(VectorType) * Nl, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::ALLOCATION_ERROR;
		return;
	}

	ub32 totalThreads = Nl;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
	GaussianBeamNS::initGaussianBeamSource <<<blocksNum, threadsNum >>> (sourceData->focalPoints, sourceData->beamOppositeDirections,
		sourceData->beamDirections, focalPointsGPU, Nl);

	cudaDeviceSynchronize();
	cudaFree(focalPointsGPU);

	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		*err = ErrorType::KERNEL_ERROR_GaussianBeamSource_initGaussianBeamSource;
	}

#if DIMS == 3
	ub32 B = beamSampleTableEntries;
	ub32 Nw = wavelenghNum;
	// ub32 P = batchNum;

	// For now, only illuminations are capable to sample
	if (connectionType == ConnectionType::ConnectionTypeIllumination)
	{
		// Position sampling buffers
		if (cudaMalloc(&zBuffer, sizeof(MediumPoint) * Nl * Nw * B) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		if (cudaMalloc(&gaussianSamplingTable, sizeof(GaussianBeamNS::GaussianDistanceSampler) * Nl * Nw) != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
		allocationCount++;

		*err = mediumHandler->getMaterialInfo(sourceData->focalPoints, Nl);

		if (*err != ErrorType::NO_ERROR)
		{
			return;
		}

		totalThreads = Nl * Nw;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
		
		GaussianBeamNS::initSamplingTable <<<blocksNum, threadsNum >>> (gaussianSamplingTable, zBuffer, sourceData->focalPoints, sourceData->beamDirections,
			mediumHandler->getBlockingBox(), B, Nl, Nw);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::KERNEL_ERROR_GaussianBeamSource_initSamplingTable;
			return;
		}

		totalThreads = Nl * Nw * B;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		GaussianBeamNS::fillTimesInSamplingTable <<<blocksNum, threadsNum>>> (gaussianSamplingTable, 
			sourceData->focalPoints, sourceData->beamDirections, B, Nl, Nw);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::KERNEL_ERROR_GaussianBeamSource_fillTimesInSamplingTable;
			return;
		}
	}

	// For now, we compute the mean viewing direction
	if (connectionType == ConnectionType::ConnectionTypeView)
	{
		VectorType meanViewDirectionCPU(0.0);
		float_type sourceSizeNormalization = (float_type)(1.0) / sourceSize;

		for (ub32 viewNum = 0; viewNum < sourceSize; viewNum++)
		{
			meanViewDirectionCPU = meanViewDirectionCPU + sourceSizeNormalization * beamDirections[viewNum];
		}

		// normalize meanViewCPU
		float_type r = abs(meanViewDirectionCPU);
		meanViewDirectionCPU = meanViewDirectionCPU * (float_type)(1.0 / r);
		isRandomizedDirection = (r < (float_type)0.0001);

		// copy to symbol
		cudaMemcpyToSymbol(GaussianBeamNS::meanViewDirection, &meanViewDirectionCPU, sizeof(VectorType), 0, cudaMemcpyKind::cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			*err = ErrorType::ALLOCATION_ERROR;
			return;
		}
	}
#endif
	*err = NO_ERROR;

	MEMORY_CHECK("Gaussian beam allocation end");
}

GaussianBeam::~GaussianBeam()
{
	MEMORY_CHECK("Gaussian beam free begin");
	GaussianBeamStructType* sourceData = (GaussianBeamStructType*)data;

	switch (allocationCount)
	{
#if DIMS == 3
	case 17:
		cudaFree(tmpKeysBuffer);
	case 16:
		cudaFree(sampleProbabilitiesCdfSumBuffer);
	case 15:
		cudaFree(beamsNumbering);
	case 14:
		cudaFree(log_CBuffer);
	case 13:
		cudaFree(real_conv_muBuffer);
	case 12:
		cudaFree(sampleProbabilitiesCdfBuffer);
	case 11:
		cudaFree(alphaIcdfBuffer);
	case 10:
		cudaFree(alphaPdfBuffer);
	case 9:
		cudaFree(icdfBuffer);
	case 8:
		cudaFree(pdfBuffer);
	case 7:
		cudaFree(gaussianSamplingTable);
	case 6:
		cudaFree(zBuffer);
#endif
	case 5:
#if DIMS == 2
		cudaFree(sourceData->focalPointsVector);
#endif
	case 4:
		cudaFree(sourceData->beamOppositeDirections);
	case 3:
		cudaFree(sourceData->beamDirections);
	case 2:
		cudaFree(sourceData->focalPoints);
	case 1:
		free(sourceData);
	default:
		break;
	}

	MEMORY_CHECK("Gaussian beam free end");
}

// ------------------------------------ Function Implementations ------------------------------------ //

ErrorType GaussianBeam::updateSampler(GaussianBeamStructType* gaussianBeamData)
{
	// make sure that all scatterers are vMF type
	for (ub32 materialNum = 0; materialNum < MATERIAL_NUM; materialNum++)
	{
		if (scatteringHandler->getScatteringType(materialNum) != ScatteringNS::ScatteringType::VMF_MIXTURE && 
			scatteringHandler->getScatteringType(materialNum) != ScatteringNS::ScatteringType::NOT_DEFINED)
		{
			return ErrorType::NOT_SUPPORTED;
		}
	}

#if DIMS==3
	ErrorType err;

	ub32 B = beamSampleTableEntries;
	ub32 Nl = sourceSize;
	ub32 Nw = wavelenghNum;
	ub32 P = batchNum;

	// Free previous allocated buffers
	switch (allocationCount)
	{
	case 17:
		cudaFree(tmpKeysBuffer);
		allocationCount--;
	case 16:
		cudaFree(sampleProbabilitiesCdfSumBuffer);
		allocationCount--;
	case 15:
		cudaFree(beamsNumbering);
		allocationCount--;
	case 14:
		cudaFree(log_CBuffer);
		allocationCount--;
	case 13:
		cudaFree(real_conv_muBuffer);
		allocationCount--;
	case 12:
		cudaFree(sampleProbabilitiesCdfBuffer);
		allocationCount--;
	case 11:
		cudaFree(alphaIcdfBuffer);
		allocationCount--;
	case 10:
		cudaFree(alphaPdfBuffer);
		allocationCount--;
	case 9:
		cudaFree(icdfBuffer);
		allocationCount--;
	case 8:
		cudaFree(pdfBuffer);
		allocationCount--;
	default:
		break;
	}

	// ------- Goal 1: get the number entries of pdf and icdf buffers ------ //

	ub32 totalThreads = Nl;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	ub32* mixturesNumTmpBuffer; // Storing M for each entry in table

	if (cudaMalloc(&mixturesNumTmpBuffer, sizeof(ub32) * Nl) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	// Get the number of mixture entries for each focal point
	GaussianBeamNS::getMixtureIdx <<< blocksNum, threadsNum >>> (mixturesNumTmpBuffer, gaussianBeamData->focalPoints, Nl);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		cudaFree(mixturesNumTmpBuffer);
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_getMixtureIdx;
	}

	// Get the maximal number of mixtures
	ub32 maxIdx = (ub32)(thrust::max_element(thrust::device, mixturesNumTmpBuffer, mixturesNumTmpBuffer + Nl) - mixturesNumTmpBuffer);

	if (cudaMemcpy(&M, mixturesNumTmpBuffer + maxIdx, sizeof(ub32), cudaMemcpyKind::cudaMemcpyDeviceToHost) != cudaError_t::cudaSuccess)
	{
		cudaFree(mixturesNumTmpBuffer);
		return ErrorType::ALLOCATION_ERROR;
	}
	cudaFree(mixturesNumTmpBuffer);

	if (cudaMemcpyToSymbol(GaussianBeamNS::maxMixtures, &M, sizeof(ub32), 0, cudaMemcpyKind::cudaMemcpyHostToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	
	// ------- Goal 2: allocate and attach pdf and icdf buffers ------ //

	// Allocate buffers
	if (cudaMalloc(&pdfBuffer, sizeof(float_type) * M * Nl * Nw * B) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	if (cudaMalloc(&icdfBuffer, sizeof(float_type) * M * Nl * Nw * B) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	if (cudaMalloc(&alphaPdfBuffer, sizeof(float_type) * M * Nl * Nw ) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	if (cudaMalloc(&alphaIcdfBuffer, sizeof(float_type) * M * Nl * Nw) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	totalThreads = Nl * Nw;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	// Attach buffers to entries
	GaussianBeamNS::attachBuffersToSampler <<< blocksNum, threadsNum >>> (gaussianSamplingTable,
		pdfBuffer, icdfBuffer, alphaPdfBuffer, alphaIcdfBuffer, 
		M, Nl, Nw, B);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_attachBuffersToSampler;
	}

	// ------- Goal 3: Allocate and attach the sample probabilities cdf ------ //
	if (cudaMalloc(&sampleProbabilitiesCdfBuffer, sizeof(float_type) * M * M * Nl * Nw * P) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	if (cudaMalloc(&real_conv_muBuffer, sizeof(VectorType) * M * M * Nl * Nw * P) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	if (cudaMalloc(&log_CBuffer, sizeof(float_type) * M * M * Nl * Nw * P) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	if (cudaMalloc(&beamsNumbering, sizeof(ub32) * M * M * Nl * Nw * P) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	if (cudaMalloc(&sampleProbabilitiesCdfSumBuffer, sizeof(float_type) * Nl * Nw * P) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	if (cudaMalloc(&tmpKeysBuffer, sizeof(ub32) * Nl * Nw * P) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	allocationCount++;

	totalThreads = P * M * M * Nl * Nw;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	GaussianBeamNS::markBeamsKernel <<< blocksNum, threadsNum >>> (beamsNumbering, M * M, P * M * M * Nl * Nw);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_markBeamsKernel;
	}

	// ------- Goal 4: Compute alpha pdf and icdf ------ //
	totalThreads = Nl * Nw;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
	GaussianBeamNS::alphaPdfIcdfKernel <<< blocksNum, threadsNum >>> (gaussianSamplingTable, Nl * Nw);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_alphaPdfIcdfKernel;
	}

	// ------- Goal 5: Compute attenuation of all points in beams ------ //
	float_type* allAttenuationBuffer;

	// Compute all attenuation
	if (cudaMalloc(&allAttenuationBuffer, sizeof(float_type) * Nl * Nw * B) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	err = mediumHandler->attenuationDirection(allAttenuationBuffer, gaussianBeamData->beamOppositeDirections, 1, zBuffer, Nl * Nw * B);

	if (err != ErrorType::NO_ERROR)
	{
		cudaFree(allAttenuationBuffer);
		return err;
	}	

	// ------- Goal 6: Compute sampling pdf ------ //
	totalThreads = B * Nl * Nw;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
	GaussianBeamNS::samplingPdfKernel <<< blocksNum, threadsNum >>> (gaussianSamplingTable, allAttenuationBuffer,
		gaussianBeamData->beamDirections, gaussianBeamData->gaussianAperture, B, Nl * Nw);

	cudaDeviceSynchronize();
	cudaFree(allAttenuationBuffer);
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_samplingPdfKernel;
	}
	
	// ------- Goal 7: Normalize sampling pdf ------ //
	// mark beams numbers
	ub32* beamsMark;

	if (cudaMalloc(&beamsMark, sizeof(ub32) * M * B * Nl * Nw) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	totalThreads = B * M * Nl * Nw;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
	
	GaussianBeamNS::markBeamsKernel <<< blocksNum, threadsNum >>> (beamsMark, B, B * M * Nl * Nw);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		cudaFree(beamsMark);
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_markBeamsKernel;
	}
	
	// mark beams keys
	/*ub32* beamsMarkKeys;
	if (cudaMalloc(&beamsMarkKeys, sizeof(ub32) * sumAllIdx) != cudaError_t::cudaSuccess)
	{
		cudaFree(beamsMark);
		return ErrorType::ALLOCATION_ERROR;
	}

	thrust::reduce_by_key(thrust::device, beamsMark, beamsMark + sumAllIdx * B, pdfBuffer, beamsMarkKeys, pdfSum);
	cudaDeviceSynchronize();
	cudaFree(beamsMarkKeys);
	
	// divide all sum values
	GaussianBeamNS::dividePdfKernel <<< blocksNum, threadsNum >>> (pdfBuffer, pdfSum, B, sumAllIdx * B);
	
	cudaDeviceSynchronize();
	cudaFree(pdfSum);
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		cudaFree(beamsMark);
		return ErrorType::KERNEL_ERROR;
	}*/

	// ------- Goal 8: compute icdf and normalize ------ //
	thrust::exclusive_scan_by_key(thrust::device, beamsMark, beamsMark + B * M * Nl * Nw, pdfBuffer, icdfBuffer);

	// get sum of each beam probability
	totalThreads = M * Nl * Nw;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	float_type* pdfSum;
	if (cudaMalloc(&pdfSum, sizeof(float_type) * M * Nl * Nw) != cudaError_t::cudaSuccess)
	{
		cudaFree(beamsMark);
		return ErrorType::ALLOCATION_ERROR;
	}

	GaussianBeamNS::getPdfSum <<< blocksNum, threadsNum >>> (pdfSum, pdfBuffer, icdfBuffer, B, M * Nl * Nw);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		cudaFree(beamsMark);
		cudaFree(pdfSum);
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_getPdfSum;
	}

	// normalize buffers
	totalThreads = B * M * Nl * Nw;
	threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
	GaussianBeamNS::normalizeSamplingBuffers <<< blocksNum, threadsNum >>> (pdfBuffer, icdfBuffer, pdfSum, B, M * Nl * Nw * B);

	cudaDeviceSynchronize();
	cudaFree(beamsMark);
	cudaFree(pdfSum);
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_normalizeSamplingBuffers;
	}

	// DEBUG
	/*float_type* pdfCPU = (float_type*)malloc(sizeof(float_type) * B * sumAllIdx);
	float_type* icdfCPU = (float_type*)malloc(sizeof(float_type) * B * sumAllIdx);

	cudaMemcpy(pdfCPU, pdfBuffer, sizeof(float_type) * B * sumAllIdx, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(icdfCPU, icdfBuffer, sizeof(float_type) * B * sumAllIdx, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	for (ub32 ii = 0; ii < B * sumAllIdx; ii++)
	{
		printf("%d: %e, %e \n", ii, pdfCPU[ii], icdfCPU[ii]);
	}

	free(pdfCPU);
	free(icdfCPU);
	*/
	// END DEBUG
#endif
	return ErrorType::NO_ERROR;
}


__host__ ErrorType GaussianBeam::sampleFirstPoint(VectorType* randomSourcePoint,
	VectorType* randomSourceDirection,
	MediumPoint* randomFirstScatterer,
	ub32 samplesNum)
{
	GaussianBeamStructType* gaussianBeamData = (GaussianBeamStructType*)data;

	ub32 threadsNum = samplesNum < THREADS_NUM ? samplesNum : THREADS_NUM;
	ub32 blocksNum = (samplesNum - 1) / THREADS_NUM + 1;

	ErrorType err;
	if (isFirstSampling)
	{
		err = updateSampler(gaussianBeamData);

		if (err != ErrorType::NO_ERROR)
		{
			return err;
		}
	}
	isFirstSampling = false;
#if DIMS==3
	if (isRandomizedDirection)
	{
		// use randomSourcePoint as a temporal buffer
		GaussianBeamNS::randomizeViewDirection<<<1,1>>>(randomSourcePoint);
		cudaMemcpyToSymbol(GaussianBeamNS::meanViewDirection, randomSourcePoint, sizeof(VectorType), 0, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
	}

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_randomizeViewDirection;
	}
	
	GaussianBeamNS::sampleFirstBeam <<<blocksNum, threadsNum >>> (randomSourcePoint,
		randomFirstScatterer,
		gaussianSamplingTable,
		samplesNum,
		beamSampleTableEntries,
		*gaussianBeamData);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_sampleFirstBeam;
	}

	// Check of sampled point is inside
	return mediumHandler->getMaterialInfo(randomFirstScatterer, samplesNum);
#else
	// In 2D, sample the first point relatively to the focal point
	GaussianBeamNS::sampleFirstBeam <<<blocksNum, threadsNum >>> (randomSourcePoint,
		randomSourceDirection,
		samplesNum,
		*gaussianBeamData);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return mediumHandler->sample(randomFirstScatterer,
		randomSourcePoint,
		randomSourceDirection,
		samplesNum);
#endif
}

ErrorType GaussianBeam::sampleSecondPoint(VectorType* sampledSourcePoint,
	MediumPoint* firstSampledPoint,
	MediumPoint* secondSampledPoint,
	ub32 samplesNum)
{
	if (scatteringHandler->isIsotropicScattering())
	{
		ub32 totalThreads = samplesNum;
		ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;
		GaussianBeamNS::isotropicGaussianSampling <<< blocksNum, threadsNum >>> (sampledSourcePoint, samplesNum);
		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_isotropicGaussianSampling;
		}
	}
	else
	{
#if DIMS==3
		GaussianBeamStructType* gaussianBeamData = (GaussianBeamStructType*)data;

		ub32 totalThreads = M * M * samplesNum;
		ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		// compute cdf buffer
		GaussianBeamNS::computeCdfBuffer<true> <<< blocksNum, threadsNum >>> (firstSampledPoint, *gaussianBeamData,
			sampleProbabilitiesCdfBuffer, real_conv_muBuffer, log_CBuffer,
			M, samplesNum, sourceSize, totalThreads);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_computeCdfBuffer;
		}

		// normalize cdf buffer
		thrust::reduce_by_key(thrust::device, beamsNumbering, beamsNumbering + totalThreads,
			sampleProbabilitiesCdfBuffer, tmpKeysBuffer, sampleProbabilitiesCdfSumBuffer);
		cudaDeviceSynchronize();

		GaussianBeamNS::normalizeCdfBuffer <<< blocksNum, threadsNum >>> (sampleProbabilitiesCdfBuffer,
			sampleProbabilitiesCdfSumBuffer, M, M * M * samplesNum);
		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_normalizeCdfBuffer;
		}

		// DEBUG
		/*
		ub32* allIdxSumBufferCPU = (ub32*)malloc(sizeof(ub32) * sumAllIdxSquated * batchNum);
		float_type* sampleProbabilitiesCdfBufferCPU = (float_type*)malloc(sizeof(float_type) * sumAllIdxSquated * batchNum);

		cudaMemcpy(allIdxSumBufferCPU, allIdxSumBuffer, sizeof(ub32) * sumAllIdxSquated * batchNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(sampleProbabilitiesCdfBufferCPU, sampleProbabilitiesCdfBuffer, sizeof(float_type) * sumAllIdxSquated * batchNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		for (ub32 ii = 0; ii < sumAllIdxSquated * batchNum; ii++)
		{
			printf("%d: %d, %f \n", ii, allIdxSumBufferCPU[ii], sampleProbabilitiesCdfBufferCPU[ii]);
		}

		for (ub32 ii = 0; ii < sumAllIdxSquated * batchNum; ii++)
		{
			printf("%f ", sampleProbabilitiesCdfBufferCPU[ii]);
			if (ii % sumAllIdxSquated == (sumAllIdxSquated - 1))
			{
				printf("\n");
			}
		}

		free(allIdxSumBufferCPU);
		free(sampleProbabilitiesCdfBufferCPU);
		*/
		// END DEBUG

		// sample direction from cdf buffer, store in sampledSourcePoint

		totalThreads = samplesNum;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		GaussianBeamNS::sampleCdfBuffer <<< blocksNum, threadsNum >>> (sampledSourcePoint,
			firstSampledPoint, sampleProbabilitiesCdfBuffer, real_conv_muBuffer, M, samplesNum);
		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_sampleCdfBuffer;
		}
#endif
	}

#if DIMS==3
	// sample new point
	return mediumHandler->sample(secondSampledPoint, firstSampledPoint, sampledSourcePoint, samplesNum);
#else
	return Source::sampleSecondPoint(sampledSourcePoint, firstSampledPoint, secondSampledPoint, samplesNum);
#endif
	
}

__host__ ErrorType GaussianBeam::firstPointProbability(float_type* probabilityRes,
	const MediumPoint* firstPoint,
	ub32 pointsNum) const
{
	GaussianBeamStructType* gaussianBeamData = (GaussianBeamStructType*)data;

	ub32 totalThreads = pointsNum * wavelenghNum * gaussianBeamData->beamsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

#if DIMS==3
	GaussianBeamNS::firstPointProbabilityKernel <<<blocksNum, threadsNum >>> (probabilityRes,
		gaussianSamplingTable, firstPoint, pointsNum, *gaussianBeamData);
	
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_firstPointProbabilityKernel;
	}

	return ErrorType::NO_ERROR;
#else
	// compute attenuation
	ErrorType err = mediumHandler->attenuationPoint(probabilityRes,
		gaussianBeamData->focalPointsVector, gaussianBeamData->beamsNum,
		firstPoint, pointsNum * wavelenghNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	// divide by distance
	GaussianBeamNS::probabilityPointNormalizeKernel <<<blocksNum, threadsNum >>> (probabilityRes,
		firstPoint, pointsNum, *gaussianBeamData);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	return ErrorType::NO_ERROR;
#endif
}

__host__ ErrorType GaussianBeam::secondPointProbability(float_type* probabilityRes,
	const MediumPoint* firstPoint,
	const VectorType* secondPoint,
	ub32 pointsNum) const
{
	GaussianBeamStructType* gaussianBeamData = (GaussianBeamStructType*)data;

	if (scatteringHandler->isIsotropicScattering())
	{
		ub32 totalThreads = pointsNum * wavelenghNum * sourceSize;
		ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		GaussianBeamNS::isotropicGaussianProbability <<<blocksNum, threadsNum >>> (probabilityRes, sourceSize, pointsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_isotropicGaussianProbability;
		}
		return ErrorType::NO_ERROR;
	}
	else
	{
#if DIMS==3
		ub32 totalThreads = M * M * pointsNum * wavelenghNum * sourceSize;
		ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		// compute cdf buffer
		GaussianBeamNS::computeCdfBuffer<false> << < blocksNum, threadsNum >> > (firstPoint, *gaussianBeamData,
			sampleProbabilitiesCdfBuffer, real_conv_muBuffer, log_CBuffer,
			M, pointsNum, sourceSize, totalThreads);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_computeCdfBuffer;
		}

		// normalize cdf buffer
		thrust::reduce_by_key(thrust::device, beamsNumbering, beamsNumbering + totalThreads,
			sampleProbabilitiesCdfBuffer, tmpKeysBuffer, sampleProbabilitiesCdfSumBuffer);
		cudaDeviceSynchronize();

		GaussianBeamNS::normalizeCdfBuffer <<< blocksNum, threadsNum >>> (sampleProbabilitiesCdfBuffer,
			sampleProbabilitiesCdfSumBuffer, M, totalThreads);
		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_normalizeCdfBuffer;
		}


		// DEBUG
		/*float_type* probabilityRes_1_CPU = (float_type*)malloc(sizeof(float_type) * pointsNum * wavelenghNum * sourceSize);
		float_type* probabilityRes_2_CPU = (float_type*)malloc(sizeof(float_type) * pointsNum * wavelenghNum * sourceSize);
		MediumPoint* firstPoint_CPU = (MediumPoint*)malloc(sizeof(MediumPoint) * pointsNum);
		VectorType* secondPoint_CPU = (VectorType*)malloc(sizeof(VectorType) * pointsNum);

		cudaMemcpy(probabilityRes_1_CPU, probabilityRes, sizeof(float_type) * pointsNum * wavelenghNum * sourceSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(firstPoint_CPU, firstPoint, sizeof(MediumPoint) * pointsNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(secondPoint_CPU, secondPoint, sizeof(VectorType) * pointsNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);*/
		// END DEBUG

		totalThreads = pointsNum * wavelenghNum * sourceSize;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		GaussianBeamNS::sampleDirectionProbability <<<blocksNum, threadsNum >>> (probabilityRes,
			firstPoint, secondPoint, sampleProbabilitiesCdfBuffer, real_conv_muBuffer, log_CBuffer,
			M, sourceSize, pointsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_sampleDirectionProbability;
		}

		// DEBUG
		/*cudaMemcpy(probabilityRes_2_CPU, probabilityRes, sizeof(float_type) * pointsNum * wavelenghNum * sourceSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		for (ub32 ii = 0; ii < pointsNum * wavelenghNum * sourceSize; ii++)
		{
			printf("prob %d: first point: [%f %f %f], second point: [%f %f %f], p1: %e, p_rotate: %e, p2: %e \n", ii,
				firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.x(), firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.y(), firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.z(),
				secondPoint_CPU[ii / gaussianBeamData->beamsNum].x(), secondPoint_CPU[ii / gaussianBeamData->beamsNum].y(), secondPoint_CPU[ii / gaussianBeamData->beamsNum].z(),
				probabilityRes_1_CPU[ii], probabilityRes_2_CPU[ii] / probabilityRes_1_CPU[ii], probabilityRes_2_CPU[ii]);
		}

		free(probabilityRes_1_CPU);
		free(probabilityRes_2_CPU);
		free(firstPoint_CPU);
		free(secondPoint_CPU);*/
		// END DEBUG

		return ErrorType::NO_ERROR;

#else
		return scatteringHandler->multiplyPdf(probabilityRes,
			gaussianBeamData->focalPointsVector, gaussianBeamData->beamsNum,
			firstPoint,
			secondPoint, pointsNum * wavelenghNum);
#endif
	}
}

__host__ ErrorType GaussianBeam::throughputFunction(ComplexType* gRes,
	const MediumPoint* p1,
	ub32 pointsNum) const
{
	GaussianBeamStructType* gaussianBeamData = (GaussianBeamStructType*)data;

	// compute attenuation
	ErrorType err = mediumHandler->attenuationDirection(gRes,
		connectionType == ConnectionType::ConnectionTypeIllumination ? gaussianBeamData->beamOppositeDirections : gaussianBeamData->beamDirections,
		gaussianBeamData->beamsNum,
		p1, pointsNum);

	if (err != ErrorType::NO_ERROR)
	{
		return err;
	}

	ub32 totalThreads = pointsNum * gaussianBeamData->beamsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	GaussianBeamNS::gGaussianBeam <<<blocksNum, threadsNum >>> (gRes, p1, pointsNum, gaussianBeamData->beamsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_gGaussianBeam;
	}

	// DEBUG
	/*
	ComplexType* g_CPU = (ComplexType*)malloc(sizeof(ComplexType) * totalThreads);
	MediumPoint* firstPoint_CPU = (MediumPoint*)malloc(sizeof(MediumPoint) * pointsNum);

	cudaMemcpy(g_CPU, gRes, sizeof(ComplexType) * totalThreads, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(firstPoint_CPU, p1, sizeof(MediumPoint) * pointsNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	for (ub32 ii = 0; ii < totalThreads; ii++)
	{
		printf("g %d: first point: [%f %f %f], g: %e \n", ii,
			firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.x, firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.y, firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.z,
			sqrt(g_CPU[ii].x * g_CPU[ii].real() + g_CPU[ii].y * g_CPU[ii].imag()));
	}

	free(firstPoint_CPU);
	free(g_CPU);
	*/
	// END DEBUG

	return ErrorType::NO_ERROR;
}

__host__ ErrorType GaussianBeam::threePointFunction(ComplexType* fRes,
	const MediumPoint* p1,
	const VectorType* p2,
	ub32 pointsNum) const
{
	GaussianBeamStructType* gaussianBeamData = (GaussianBeamStructType*)data;

	ub32 totalThreads = pointsNum * gaussianBeamData->beamsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	GaussianBeamNS::fGaussianBeam <<<blocksNum, threadsNum >>> (fRes, p1, p2, *gaussianBeamData, pointsNum, connectionType);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR_GaussianBeamSource_fGaussianBeam;
	}

	// DEBUG
	/*
	complex_type* f_CPU = (complex_type*)malloc(sizeof(complex_type) * totalThreads);
	MediumPoint* firstPoint_CPU = (MediumPoint*)malloc(sizeof(MediumPoint) * pointsNum);
	vector_type* secondPoint_CPU = (vector_type*)malloc(sizeof(vector_type) * pointsNum);

	cudaMemcpy(f_CPU, fRes, sizeof(complex_type) * totalThreads, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(firstPoint_CPU, p1, sizeof(MediumPoint) * pointsNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaMemcpy(secondPoint_CPU, p2, sizeof(vector_type) * pointsNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	for (ub32 ii = 0; ii < totalThreads; ii++)
	{
		printf("f %d: first point: [%f %f %f], second point: [%f %f %f], f: %e \n", ii,
			firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.x, firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.y, firstPoint_CPU[ii / gaussianBeamData->beamsNum].position.z,
			secondPoint_CPU[ii / gaussianBeamData->beamsNum].x, secondPoint_CPU[ii / gaussianBeamData->beamsNum].y, secondPoint_CPU[ii / gaussianBeamData->beamsNum].z,
			sqrt(f_CPU[ii].x * f_CPU[ii].x + f_CPU[ii].y * f_CPU[ii].y));
	}

	free(firstPoint_CPU);
	free(f_CPU);
	free(secondPoint_CPU);
	*/
	// END DEBUG

	return ErrorType::NO_ERROR;
}

__host__ ErrorType GaussianBeam::threePointFunctionSingle(ComplexType* fsRes,
	const MediumPoint* p1,
	const Source* otherSource,
	ub32 pointsNum) const
{
	GaussianBeamStructType* gaussianBeamData = (GaussianBeamStructType*)data;

	if (otherSource->sourceType == SourceType::GaussianBeamType)
	{
		GaussianBeamStructType* otherGaussianBeamData = (GaussianBeamStructType*)otherSource->data;

		ub32 totalThreads = pointsNum * gaussianBeamData->beamsNum * otherGaussianBeamData->beamsNum;

		ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		GaussianBeamNS::fsGaussianBeam <<<blocksNum, threadsNum >>> (fsRes,
			p1,
			*gaussianBeamData,
			*otherGaussianBeamData,
			pointsNum);

		cudaDeviceSynchronize();
		if (cudaGetLastError() != cudaError_t::cudaSuccess)
		{
			return ErrorType::KERNEL_ERROR_GaussianBeamSource_fsGaussianBeam;
		}

		// DEBUG
		/*
		complex_type* fs_CPU = (complex_type*)malloc(sizeof(complex_type) * totalThreads);
		MediumPoint* firstPoint_CPU = (MediumPoint*)malloc(sizeof(MediumPoint) * pointsNum);

		cudaMemcpy(fs_CPU, fsRes, sizeof(complex_type) * totalThreads, cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(firstPoint_CPU, p1, sizeof(MediumPoint) * pointsNum, cudaMemcpyKind::cudaMemcpyDeviceToHost);

		for (ub32 ii = 0; ii < totalThreads; ii++)
		{
			printf("fs %d: first point: [%f %f %f], fs: %e \n", ii,
				firstPoint_CPU[ii / (gaussianBeamData->beamsNum * otherGaussianBeamData->beamsNum)].position.x,
				firstPoint_CPU[ii / (gaussianBeamData->beamsNum * otherGaussianBeamData->beamsNum)].position.y,
				firstPoint_CPU[ii / (gaussianBeamData->beamsNum * otherGaussianBeamData->beamsNum)].position.z,
				sqrt(fs_CPU[ii].x * fs_CPU[ii].x + fs_CPU[ii].y * fs_CPU[ii].y));
		}

		free(firstPoint_CPU);
		free(fs_CPU);
		*/
		// END DEBUG
	}
	else
	{
		return ErrorType::NOT_SUPPORTED;
	}

	return ErrorType::NO_ERROR;
}
