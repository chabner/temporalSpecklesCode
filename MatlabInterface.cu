#include "Simulation.cuh"
#include "Implementation/Tracer/NextEventEstimation.cuh"
#include "Implementation/Scattering.cuh"
#include "Implementation/Medium/HomogeneousMedium.cuh"
#include "Implementation/Medium/HeterogeneousMedium.cuh"
#include "Implementation/Sampler/TemporalSampler.cuh"
#include "Implementation/Sampler/TemporalSamplerCorrelation.cuh"
#include "Implementation/Source/PointSource.cuh"
#include "Implementation/Source/FarField.cuh"
#include "Implementation/Source/GaussianBeam.cuh"

#include <time.h>

#include "math_constants.h"
#include <stdio.h>
#include <memory>
#include <string.h>
#include "mex.h"
#include "matrix.h"
#include "mat.h"

ub32 cudaDeviceNum = 0;

void getErrorMsg(ErrorType errT, char** err)
{
    if(errT == ErrorType::DEVICE_ERROR)
    {
        int gpuCount;
        cudaGetDeviceCount( &gpuCount );
        mexErrMsgIdAndTxt( "SSTMC:CUDA:deviceNum",
            "Bad device number. GPU number is %d, where available gpu numbers from begins from 0 to %d.",
            cudaDeviceNum, gpuCount - 1);

    }
    if(errT == ErrorType::ALLOCATION_ERROR)
    {
        strcpy(*err, "Allocation error: CPU / GPU exceed maximal memory. Try to decrease batchSize.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR)
    {
        strcpy(*err, "Error occurred during a kernel opration.");
        return;
    }
    if(errT == ErrorType::CUBLAS_ERROR)
    {
        strcpy(*err, "Error occurred during a cublas opration.");
        return;
    }
    if(errT == ErrorType::NOT_SUPPORTED)
    {
        strcpy(*err, "Illumination and view combination is not supported.");
        return;
    }
    if(errT == ErrorType::MISC_ERROR)
    {
        strcpy(*err, "Unknown error.");
        return;
    }
    if(errT == ErrorType::NO_ERROR)
    {
        strcpy(*err, "No error.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_initGaussianBeamSource)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_initGaussianBeamSource.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_randomizeViewDirection)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_randomizeViewDirection.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_sampleFirstBeam)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_sampleFirstBeam.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_firstPointProbabilityKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_firstPointProbabilityKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_initSamplingTable)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_initSamplingTable.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_fillTimesInSamplingTable)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_fillTimesInSamplingTable.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_getMixtureIdx)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_getMixtureIdx.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_attachBuffersToSampler)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_attachBuffersToSampler.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_alphaPdfIcdfKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_alphaPdfIcdfKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_samplingPdfKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_samplingPdfKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_markBeamsKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_markBeamsKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_getPdfSum)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_getPdfSum.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_normalizeSamplingBuffers)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_normalizeSamplingBuffers.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_computeCdfBuffer)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_computeCdfBuffer.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_normalizeCdfBuffer)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_normalizeCdfBuffer.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_sampleCdfBuffer)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_sampleCdfBuffer.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_sampleDirectionProbability)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_sampleDirectionProbability.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_gGaussianBeam)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_gGaussianBeam.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_fGaussianBeam)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_fGaussianBeam.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_fsGaussianBeam)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_fsGaussianBeam.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_isotropicGaussianSampling)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_isotropicGaussianSampling.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_GaussianBeamSource_isotropicGaussianProbability)
    {
        strcpy(*err, "Error occurred during a kernel opration. GaussianBeamSource_isotropicGaussianProbability.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Sampler_setTemporalPathsNum)
    {
        strcpy(*err, "Error occurred during a kernel opration. Sampler_setTemporalPathsNum.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Sampler_copyBuffersToPoints)
    {
        strcpy(*err, "Error occurred during a kernel opration. Sampler_copyBuffersToPoints.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffers)
    {
        strcpy(*err, "Error occurred during a kernel opration. Sampler_copyPointsToBuffers.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffersThroughput)
    {
        strcpy(*err, "Error occurred during a kernel opration. Sampler_copyPointsToBuffersThroughput.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Sampler_copyPointsToBuffersThreePoints)
    {
        strcpy(*err, "Error occurred during a kernel opration. Sampler_copyPointsToBuffersThreePoints.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Sampler_dtShiftKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. Sampler_dtShiftKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Sampler_copyPointsFromBuffersTemporal)
    {
        strcpy(*err, "Error occurred during a kernel opration. Sampler_copyPointsFromBuffersTemporal.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Sampler_randomizeDirectionsKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. Sampler_randomizeDirectionsKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Source_multPathContribution)
    {
        strcpy(*err, "Error occurred during a kernel opration. Source_multPathContribution.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_NEE_buildPermuteMatrix)
    {
        strcpy(*err, "Error occurred during a kernel opration. NEE_buildPermuteMatrix.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_NEE_complexMultiplicativeInverse)
    {
        strcpy(*err, "Error occurred during a kernel opration. NEE_complexMultiplicativeInverse.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Scattering_amplitudeKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. Scattering_amplitudeKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Scattering_pdfKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. Scattering_pdfKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Scattering_newDirectionKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. Scattering_newDirectionKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Scattering_hgInnerScatteringNormalization)
    {
        strcpy(*err, "Error occurred during a kernel opration. Scattering_hgInnerScatteringNormalization.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_HetroMedium_heterogeneousAttenuationKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. HetroMedium_heterogeneousAttenuationKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_HetroMedium_heterogeneousSampleKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. HetroMedium_heterogeneousSampleKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_HetroMedium_getMaterialKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. HetroMedium_getMaterialKernel.");
        return;
    }
    if(errT == ErrorType::KERNEL_ERROR_Medium_sampleRandomInsideKernel)
    {
        strcpy(*err, "Error occurred during a kernel opration. Medium_sampleRandomInsideKernel.");
        return;
    }

    strcpy(*err, "New unknown error.");
}

mxArray* getStruct(const mxArray* inArray, const char* fieldName, bool isRequried)
{
    mxArray* outStruct = mxGetField(inArray, 0, fieldName);

    if(outStruct == NULL && isRequried == false)
    {
        return NULL;
    }

    if(outStruct == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequriedStruct",
            "%s is a requried field.", fieldName);
    }

    if(!mxIsStruct(outStruct))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noStruct",
            "%s is not a struct.",fieldName);
    }

    return outStruct;
}

// reuturn cell dims
mxArray* getCellArray(ub32* cellSize, const mxArray* inArray, const char* fieldName, bool isRequried, 
    bool forceMinimalValue, ub32 minimalVal, bool forceMaximalValue, ub32 maximalVal)
{
    mxArray* outCell = mxGetField(inArray, 0, fieldName);

    if(outCell == NULL && isRequried == false)
    {
        return NULL;
    }

    if(outCell == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequriedStruct",
            "%s is a requried field.", fieldName);
    }

    if(!mxIsCell(outCell))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noCell",
            "%s must be a cell array.", fieldName);
    }

    *cellSize = mxGetNumberOfElements(outCell);

    if(forceMinimalValue && *cellSize < minimalVal)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:cellMinimalElements",
            "%s must have at least %d elements, input has %d elements.", fieldName, minimalVal, *cellSize);
    }

    if(forceMaximalValue && *cellSize > maximalVal)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:cellMaximalElements",
            "%s must have at the most %d elements, input has %d elements.", fieldName, maximalVal, *cellSize);
    }

    return outCell;
}

void* getDataFromCell(ub32* N, ub32 cellNum, const mxArray* inArray, bool isComplex, bool isRequried)
{
    mxArray* cellArray;
    cellArray = mxGetCell(inArray, cellNum);

    if(cellArray == NULL && isRequried == false)
    {
        return 0;
    }

    if(cellArray == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequriedCellNumber",
            "cell num %d is not appeared.", cellNum);
    }

    if(mxGetClassID(cellArray) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:cellNoDouble",
            "cell num %d must be double type.", cellNum);
    }

    if(isComplex && !mxIsComplex(cellArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:cellNoComplex",
            "cell num %d must be complex type.", cellNum);
    }

    if(!isComplex && mxIsComplex(cellArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:cellNoComplex",
            "cell num %d must be real type.", cellNum);
    }

    *N = mxGetNumberOfElements(cellArray);

    if(isComplex)
    {
        mxComplexDouble *cellValues = mxGetComplexDoubles(cellArray);

        if(cellValues == NULL) 
        {
            mexErrMsgIdAndTxt( "SSTMC:MATLAB:cellReadValue",
                "Failed to read %d value in the cell array.", cellNum);
        }

        ComplexType* outRes;

        outRes = (ComplexType*) mxCalloc(*N, sizeof(ComplexType));
        if(outRes == 0)
        {
            mexErrMsgIdAndTxt( "SSTMC:Host:allocation", "Host allocation error");
        }

        for(ub32 vIdx = 0; vIdx < *N; vIdx++)
        {
            outRes[vIdx] = ComplexType((float_type)(cellValues[vIdx].real), (float_type)(cellValues[vIdx].imag));
        }

        return (void*)outRes;
    }
    else
    {
        mxDouble *cellValues = mxGetDoubles(cellArray);

        if(cellValues == NULL) 
        {
            mexErrMsgIdAndTxt( "SSTMC:MATLAB:cellReadValue",
                "Failed to read %d value in the cell array.", cellNum);
        }

        float_type* outRes;

        outRes = (float_type*) mxCalloc(*N, sizeof(float_type));
        if(outRes == 0)
        {
            mexErrMsgIdAndTxt( "SSTMC:Host:allocation", "Host allocation error");
        }

        for(ub32 vIdx = 0; vIdx < *N; vIdx++)
        {
            outRes[vIdx] = (float_type)(cellValues[vIdx]);
        }

        return (void*) outRes;
    }

    return NULL;
}

bool isEqualToName(const mxArray* inArray, const char* fieldName, const char* targetName, bool isRequried)
{
    bool isEqual = false;

    mxArray* nameArray = mxGetField(inArray, 0, fieldName);

    if(nameArray == NULL && isRequried == false)
    {
        return isEqual;
    }

    if(nameArray == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequiredField",
            "%s is a requried field.", fieldName);
    }
        
    if (!mxIsChar(nameArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidFieldType",
            "%s must be a string.", fieldName);
    }

    char* nameBuffer = mxArrayToString(nameArray);
    if(nameBuffer == NULL) 
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidNameBuufer",
            "Could not convert %s to string.", fieldName);
    }

    if(strcmp(nameBuffer, targetName) == 0)
    {
        isEqual = true;
    }

    mxFree(nameBuffer);

    return isEqual;
}

bool getVectorScalar(VectorType* res, const mxArray* inArray, const char* fieldName, bool isRequried)
{
    mxArray* valuesArray = mxGetField(inArray, 0, fieldName);

    if(valuesArray == NULL && isRequried == false)
    {
        return false;
    }

    if(valuesArray == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequiredField",
            "%s is a requried field.", fieldName);
    }

    if(mxGetClassID(valuesArray) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidType",
            "%s must be a double type.", fieldName);
    }

    if(mxIsComplex(valuesArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidComplexType",
            "%s must be a real type.", fieldName);
    }

    if(mxGetNumberOfElements(valuesArray) != DIMS)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidSize",
            "%s must be size %d.", fieldName, DIMS);
    }

    *res = VectorType((float_type)(mxGetDoubles(valuesArray)[0]),
        (float_type)(mxGetDoubles(valuesArray)[1])
#if DIMS==3
        , (float_type)(mxGetDoubles(valuesArray)[2])
#endif
    );

    return true;
}

bool getScalar(float_type* res, const mxArray* inArray, const char* fieldName, bool isRequried)
{
    mxArray* valuesArray = mxGetField(inArray, 0, fieldName);

    if(valuesArray == NULL && isRequried == false)
    {
        return false;
    }

    if(valuesArray == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequiredField",
            "%s is a requried field.", fieldName);
    }

    if(mxGetClassID(valuesArray) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidType",
            "%s must be a double type.", fieldName);
    }

    if(mxIsComplex(valuesArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidComplexType",
            "%s must be a real type.", fieldName);
    }

    if(mxGetNumberOfElements(valuesArray) != 1)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidSize",
            "%s must be size %d.", fieldName, 1);
    }

    *res = (float_type)(mxGetDoubles(valuesArray)[0]);

    return true;
}

VectorType* getVectorArray(ub32* arraySize, const mxArray* inArray, const char* fieldName_x, const char* fieldName_y
#if DIMS==3
    , const char* fieldName_z
#endif
    , bool isRequried, bool forceElementsNumber, ub32 forcedElementsNum)
{
    mxArray* valuesArray = mxGetField(inArray, 0, fieldName_x);

    if(valuesArray == NULL && isRequried == false)
    {
        return NULL;
    }

    if(valuesArray == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequiredField",
            "%s is a requried field.", fieldName_x);
    }

    if(mxGetClassID(valuesArray) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidType",
            "%s must be a double type.", fieldName_x);
    }

    if(mxIsComplex(valuesArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidComplexType",
            "%s must be a real type.", fieldName_x);
    }

    if(forceElementsNumber && mxGetNumberOfElements(valuesArray) != forcedElementsNum)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidSize",
            "%s must be size %d.", fieldName_x, forcedElementsNum);
    }

    if(mxGetNumberOfElements(valuesArray) == 0)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:zeroSize",
            "%s must be non-empty.", fieldName_x);
    }

    *arraySize = mxGetNumberOfElements(valuesArray);
    VectorType* res = (VectorType*) mxCalloc(*arraySize, sizeof(VectorType));

    for (ub32 elemNum = 0; elemNum < *arraySize; elemNum++)
    {
        res[elemNum].setx((float_type)(mxGetDoubles(valuesArray)[elemNum]));
    }

    valuesArray = mxGetField(inArray, 0, fieldName_y);

    if(valuesArray == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequiredField",
            "%s is a requried field.", fieldName_y);
    }

    if(mxGetClassID(valuesArray) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidType",
            "%s must be a double type.", fieldName_y);
    }

    if(mxIsComplex(valuesArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidComplexType",
            "%s must be a real type.", fieldName_y);
    }

    if(mxGetNumberOfElements(valuesArray) != *arraySize)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidSize",
            "%s must be size %d.", fieldName_y, *arraySize);
    }

    for (ub32 elemNum = 0; elemNum < *arraySize; elemNum++)
    {
        res[elemNum].sety((float_type)(mxGetDoubles(valuesArray)[elemNum]));
    }

#if DIMS==3
    valuesArray = mxGetField(inArray, 0, fieldName_z);

    if(valuesArray == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequiredField",
            "%s is a requried field.", fieldName_z);
    }

    if(mxGetClassID(valuesArray) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidType",
            "%s must be a double type.", fieldName_z);
    }

    if(mxIsComplex(valuesArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidComplexType",
            "%s must be a real type.", fieldName_z);
    }

    if(mxGetNumberOfElements(valuesArray) != *arraySize)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidSize",
            "%s must be size %d.", fieldName_z, *arraySize);
    }

    for (ub32 elemNum = 0; elemNum < *arraySize; elemNum++)
    {
        res[elemNum].setz((float_type)(mxGetDoubles(valuesArray)[elemNum]));
    }
#endif

    return res;
}

template<typename T>
T* getScalarArray(ub32 *sizeArray, const mxArray* inArray, const char* fieldName,
    bool isRequried, bool forceMinimalNumber, ub32 forcedElementsNum)
{
    mxArray* valuesArray = mxGetField(inArray, 0, fieldName);

    if(valuesArray == NULL && isRequried == false)
    {
        return NULL;
    }

    if(valuesArray == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noRequiredField",
            "%s is a requried field.", fieldName);
    }

    if(mxGetClassID(valuesArray) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidType",
            "%s must be a double type.", fieldName);
    }

    if(mxIsComplex(valuesArray))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidComplexType",
            "%s must be a real type.", fieldName);
    }

    if(forceMinimalNumber && mxGetNumberOfElements(valuesArray) < forcedElementsNum)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidSize",
            "%s must be at least size %d.", fieldName, forcedElementsNum);
    }

    *sizeArray = mxGetNumberOfElements(valuesArray);
    T* res = (T*) mxCalloc(*sizeArray, sizeof(T));

    for (ub32 elemNum = 0; elemNum < *sizeArray; elemNum++)
    {
        res[elemNum] = (T)(mxGetDoubles(valuesArray)[elemNum]);
    }
    return res;
}

Scattering* buildScattering(const mxArray* prhs[], ub32* materialNum)
{
    ErrorType err;
    char* errorBuffer = (char*) mxCalloc(200, sizeof(char));

    Scattering* scatterer = new Scattering();

    mxArray* scatteringStruct = getStruct(prhs[0], "scattering", true);
    mxArray* scatteringType = getCellArray(materialNum, scatteringStruct, "type", true, true, 0, true, MATERIAL_NUM);

    for(ub32 matNum = 0; matNum < *materialNum; matNum++)
    {
        mxArray* scatteringTypeName = mxGetCell(scatteringType, matNum);
        
        if ( scatteringTypeName == NULL || !mxIsChar(scatteringTypeName))
        {
            mexErrMsgIdAndTxt( "SSTMC:MATLAB:scatteringTypeName",
                "The scattering type name in cell %d must be a string.", matNum + 1);
        }

        char* scatteringNameBuffer = mxArrayToString(scatteringTypeName);
        if(scatteringNameBuffer == NULL) 
        {
            mexErrMsgIdAndTxt( "SSTMC:MATLAB:scatteringTypeNameStr",
                "Could not convert scattering name to string.");
        }

        if(strcmp(scatteringNameBuffer, "Isotropic") == 0)
        {
            err = scatterer->setIsotropicScattering(matNum);

            if(err != ErrorType::NO_ERROR)
            {
                getErrorMsg(err, &errorBuffer);
                mexErrMsgIdAndTxt( "SSTMC:Scattering:setIsotropic",
                    "Error occurred during adding isotropic scattering to material number %d. Error msg: %s", matNum, errorBuffer);
            }

            if(*materialNum == 1)
            {
                err = scatterer->setIsotropicScattering(1);

                if(err != ErrorType::NO_ERROR)
                {
                    getErrorMsg(err, &errorBuffer);
                    mexErrMsgIdAndTxt( "SSTMC:Scattering:setIsotropic",
                        "Error occurred during adding isotropic scattering to material number %d. Error msg: %s", matNum, errorBuffer);
                }
            }
        }
        else if(strcmp(scatteringNameBuffer, "Tabular") == 0)
        {
            ub32 totalCells;
            mxArray* scatteringAmplitude = getCellArray(&totalCells, scatteringStruct, "amplitudeFunction", true, true, *materialNum, true, *materialNum);
                
            ub32 N;
            ComplexType* fHost = (ComplexType*) getDataFromCell(&N, matNum, scatteringAmplitude, true, true);

            err = scatterer->setTabularScattering(fHost, N, matNum);

            if(err != ErrorType::NO_ERROR)
            {
                getErrorMsg(err, &errorBuffer);
                mexErrMsgIdAndTxt( "SSTMC:Scattering:setTabular",
                    "Error occurred during adding tabular scattering to material number %d. Error msg: %s", matNum, errorBuffer);
            }

            if(*materialNum == 1)
            {
                err = scatterer->setTabularScattering(fHost, N, 1);

                if(err != ErrorType::NO_ERROR)
                {
                    getErrorMsg(err, &errorBuffer);
                    mexErrMsgIdAndTxt( "SSTMC:Scattering:setTabular",
                        "Error occurred during adding tabular scattering to material number %d. Error msg: %s", matNum, errorBuffer);
                }
            }

            mxFree(fHost);
        }
        else if((strcmp(scatteringNameBuffer, "HenyeyGreenstein") == 0) || (strcmp(scatteringNameBuffer, "HG") == 0) )
        {
            ub32 totalCells;
            mxArray* scatteringAmplitude = getCellArray(&totalCells, scatteringStruct, "amplitudeFunction", true, true, *materialNum, true, *materialNum);

            ub32 N;
            float_type* gHost = (float_type*) getDataFromCell(&N, matNum, scatteringAmplitude, false, true);

            // mexPrintf("add HG scattring, gHost = %f, matNum = %d \n", *gHost, matNum);

            err = scatterer->setHenyeyGreensteinScattering(*gHost, matNum);

            if(err != ErrorType::NO_ERROR)
            {
                getErrorMsg(err, &errorBuffer);
                mexErrMsgIdAndTxt( "SSTMC:Scattering:setHenyeyGreenstein",
                    "Error occurred during adding Henyey Greenstein scattering to material number %d. Error msg: %s", matNum, errorBuffer);
            }

            if(*materialNum == 1)
            {
                // mexPrintf("add HG scattring, gHost = %f, matNum = %d \n", *gHost, 1);
                err = scatterer->setHenyeyGreensteinScattering(*gHost, 1);

                if(err != ErrorType::NO_ERROR)
                {
                    getErrorMsg(err, &errorBuffer);
                    mexErrMsgIdAndTxt( "SSTMC:Scattering:setHenyeyGreenstein",
                        "Error occurred during adding Henyey Greenstein scattering to material number %d. Error msg: %s", matNum, errorBuffer);
                }
            }

            mxFree(gHost);
        }
        else if((strcmp(scatteringNameBuffer, "VonMisesFisher") == 0) || (strcmp(scatteringNameBuffer, "vMF") == 0) )
        {
            ub32 totalCells;
            mxArray* scatteringAmplitude = getCellArray(&totalCells, scatteringStruct, "amplitudeFunction", true, true, *materialNum, true, *materialNum);

            mxArray* vMfcellDataArray = mxGetCell(scatteringAmplitude, matNum);

            ub32 mixturesNum, currentMixturesNum;

            float_type* muHost = getScalarArray<float_type>(&mixturesNum, vMfcellDataArray, "mixtureMu", true, false, 1);
            float_type* cHost = getScalarArray<float_type>(&currentMixturesNum, vMfcellDataArray, "mixtureC", true, true, mixturesNum);
            float_type* alphaHost = getScalarArray<float_type>(&currentMixturesNum, vMfcellDataArray, "mixtureAlpha", true, true, mixturesNum);


            //mexPrintf("Add Gaussian Beam material %d: mixtures num: %d, muHost: %f %f %f %f, cHost: %f %f %f %f, alphaHost: %f %f %f %f \n",
            //    matNum, mixturesNum, muHost[0], muHost[1], muHost[2], muHost[3], cHost[0], cHost[1], cHost[2], cHost[3]
            //   , alphaHost[0], alphaHost[1], alphaHost[2], alphaHost[3]);

            err = scatterer->setVonMisesFisherScattering(muHost, cHost, alphaHost, mixturesNum, matNum);

            if(err != ErrorType::NO_ERROR)
            {
                getErrorMsg(err, &errorBuffer);
                mexErrMsgIdAndTxt( "SSTMC:Scattering:setHenyeyGreenstein",
                    "Error occurred during adding Henyey Greenstein scattering to material number %d. Error msg: %s", matNum, errorBuffer);
            }

            if(*materialNum == 1)
            {
                err = scatterer->setVonMisesFisherScattering(muHost, cHost, alphaHost, mixturesNum, 1);

                if(err != ErrorType::NO_ERROR)
                {
                    getErrorMsg(err, &errorBuffer);
                    mexErrMsgIdAndTxt( "SSTMC:Scattering:setHenyeyGreenstein",
                        "Error occurred during adding Henyey Greenstein scattering to material number %d. Error msg: %s", matNum, errorBuffer);
                }
            }

            mxFree(muHost);
            mxFree(cHost);
            mxFree(alphaHost);

        }
        else
        {
            mexErrMsgIdAndTxt( "SSTMC:MATLAB:unsupportedScattering",
                "Unsupported scattering type in material %d. Type entered: %s. Available scattering types are Isotropic, Tabular, HenyeyGreenstein (HG), and VonMisesFisher (vMF)",
                matNum, scatteringNameBuffer);
        }

        if(*materialNum == 1)
        {
            (*materialNum)++;
        }
        
        mxFree(scatteringNameBuffer);
    }
    
    mxFree(errorBuffer);
    return scatterer;
}

Medium* buildMedium(const mxArray* prhs[], ub32 materialNum)
{
    ErrorType err;
    char* errorBuffer = (char*) mxCalloc(200, sizeof(char));

    Medium* medium = NULL;
    mxArray* mediumStruct = getStruct(prhs[0], "medium", true);

    if(isEqualToName(mediumStruct, "type", "Homogeneous", true))
    {
        if(materialNum != 2)
        {
            mexErrMsgIdAndTxt( "SSTMC:MATLAB:homogeneousMaterials",
                "Homogeneous medium cannot have heterogeneous material.");
        }

        VectorType boxMinHost;
        getVectorScalar(&boxMinHost, mediumStruct, "boxMin", true);

        VectorType boxMaxHost;
        getVectorScalar(&boxMaxHost, mediumStruct, "boxMax", true);

        float_type sigsHost;
        getScalar(&sigsHost, mediumStruct, "sigs", true);

        float_type sigaHost;
        if(getScalar(&sigaHost, mediumStruct, "siga", false) == false)
        {
            sigaHost = 0.;
        }

        medium = new HomogeneousMedium(&err, boxMinHost, boxMaxHost);

        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Medium:makeHomogeneous",
                "Error occurred during making an Homogeneous medium. Error msg: %s", errorBuffer);
        }

        MediumMaterial homogeneousMaterial;
        homogeneousMaterial.sigs = (float_type)0.;

        err = medium->setMaterial(&homogeneousMaterial, 0);
        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Medium:setMaterial",
                "Error occurred during setting material 0 to medium. Error msg: %s", errorBuffer);
        }

        homogeneousMaterial.sigs = sigsHost;
        err = medium->setMaterial(&homogeneousMaterial, 1);
        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Medium:setMaterial",
                "Error occurred during setting material 1 to medium. Error msg: %s", errorBuffer);
        }

    }
    else if(isEqualToName(mediumStruct, "type", "Heterogeneous", true))
    {
        ub32 xAxisElements, yAxisElements;

        float_type *xAxisHost = getScalarArray<float_type>(&xAxisElements, mediumStruct, "xAxis", true, true, 2);
        float_type *yAxisHost = getScalarArray<float_type>(&yAxisElements, mediumStruct, "yAxis", true, true, 2);

        ub32 totalGridElements = ((ub32)(xAxisElements - 1) * (yAxisElements - 1));
#if DIMS==3
        ub32 zAxisElements;
        float_type *zAxisHost = getScalarArray<float_type>(&zAxisElements, mediumStruct, "zAxis", true, true, 2);
        totalGridElements *= ((ub32)(zAxisElements - 1));
#endif

        ub32 sigsElements;
        ub32 sigaElements;

        float_type *sigsHost = getScalarArray<float_type>(&sigsElements, mediumStruct, "sigs", true, true, materialNum);

        float_type* sigaHost = getScalarArray<float_type>(&sigaElements, mediumStruct, "siga", false, true, materialNum);

        ub32 materialGridElements;
        ub32* materialGridHost = getScalarArray<ub32>(&materialGridElements, mediumStruct, "materialGrid", true, true, totalGridElements);

        for(ub32 mElement = 0 ; mElement < totalGridElements; mElement++)
        {
            ub32 currentMaterial = materialGridHost[mElement];

            if(currentMaterial == 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:heterogeneousZeroMaterial",
                    "Material 0 is only allowed outside the grid. Allowed material numbers are from 1 to %d.",
                    materialNum - 1);
            }

            if(currentMaterial >= materialNum)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:heterogeneousLargeMaterial",
                    "Material %d is not defined. Allowed material numbers are from 1 to %d.",
                    currentMaterial, materialNum - 1);
            }
        }

        //mexPrintf("Medium: materialGrid - %d %d xAxis %d - %f %f yAxis %d - %f %f zAxis %d - %f %f %f \n",
        //   materialGridHost[0], materialGridHost[1], xAxisElements, xAxisHost[0], xAxisHost[1],
        //  yAxisElements, yAxisHost[0], yAxisHost[1], zAxisElements, zAxisHost[0], zAxisHost[1], zAxisHost[2]);

        medium = new HeterogeneousMedium(&err, materialGridHost,
            xAxisHost, xAxisElements, yAxisHost, yAxisElements
#if DIMS==3
            , zAxisHost, zAxisElements
#endif
        );

        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Medium:makeHeterogeneous",
                "Error occurred during making an heterogeneous medium. Error msg: %s", errorBuffer);
        }

        for(ub32 mNum = 0; mNum < materialNum; mNum++)
        {
            MediumMaterial heterogeneousMaterial;
            heterogeneousMaterial.sigs = sigsHost[mNum];

            if(sigaHost == NULL)
            {
                heterogeneousMaterial.siga = 0.;
            }
            else
            {
                heterogeneousMaterial.siga = sigaHost[mNum];
            }

            // mexPrintf("Medium material %d: sigs = %f \n", mNum, heterogeneousMaterial.sigs);

            err = medium->setMaterial(&heterogeneousMaterial, mNum);
            if(err != ErrorType::NO_ERROR)
            {
                getErrorMsg(err, &errorBuffer);
                mexErrMsgIdAndTxt( "SSTMC:Medium:setMaterial",
                    "Error occurred during setting material %d to medium. Error msg: %s", mNum, errorBuffer);
            }
        }

        mxFree(xAxisHost);
        mxFree(yAxisHost);
#if DIMS==3
        mxFree(zAxisHost);
#endif
        mxFree(sigsHost);
        mxFree(materialGridHost);

        if(sigaHost != NULL)
        {
            mxFree(sigaHost);
        }

    }
    else
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:unsupportedMedium",
            "Unsupported medium type. Available medium types are Homogeneous and Heterogeneous");
    }

    mxFree(errorBuffer);

    return medium;
}

Source* buildSource(const mxArray* prhs[], const Scattering* scatterer, const Simulation* simulation,
    const Medium* medium, ub32 materialNum, ConnectionType cn)
{
    ErrorType err;
    bool isU2;
    char* errorBuffer = (char*) mxCalloc(200, sizeof(char));

    char sourceName[20];
    if(cn == ConnectionType::ConnectionTypeIllumination)
    {
        strcpy(sourceName, "illumination");
        isU2 = false;
    }
    else if(cn == ConnectionType::ConnectionTypeView)
    {
        strcpy(sourceName, "view");
        isU2 = false;
    }
    else if(cn == ConnectionType::ConnectionTypeIllumination2)
    {
        strcpy(sourceName, "illumination2");
        isU2 = true;
    }
    else
    {
        strcpy(sourceName, "view2");
        isU2 = true;
    }

    Source* source = NULL;
    bool isActive = true;

    mxArray* sourceStruct = getStruct(prhs[0], sourceName, !isU2);

    if(sourceStruct == NULL)
    {
        isActive = false;
        source = NULL;
    }

    if(isActive && isEqualToName(sourceStruct, "type", "PointSource", true))
    {
        mxArray* locationStruct = getStruct(sourceStruct, "location", true);

        ub32 locationSize;
        VectorType* pointSourceLocationHost = getVectorArray(&locationSize, locationStruct, "x", "y"
#if DIMS==3
        , "z"
#endif
        , true, false, 1);

        // optional
        bool isAreaSource = false;

        mxArray* isAreaSourceMatlab = mxGetField(sourceStruct, 0, "isAreaSource");
        if(isAreaSourceMatlab != NULL)
        {
            if(!mxIsLogicalScalar(isAreaSourceMatlab))
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:pointSourceIsAeraNotLogical",
                "%s point source isAreaSource field is a logical scalar.", sourceName);
            }

            isAreaSource = mxIsLogicalScalarTrue(isAreaSourceMatlab);
        }

        source = new PointSource(&err, simulation, medium, scatterer,
            pointSourceLocationHost, locationSize, isAreaSource, cn);

        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Source:makePointSource",
                "Error occurred during making %s point source. Error msg: %s", sourceName, errorBuffer);
        }
        
        mxFree(pointSourceLocationHost);
    }
    else if(isActive && isEqualToName(sourceStruct, "type", "FarField", true))
    {
        mxArray* directionStruct = getStruct(sourceStruct, "direction", true);

        ub32 directionSize;
        VectorType* farFieldDirectionHost = getVectorArray(&directionSize, directionStruct, "x", "y"
#if DIMS==3
    , "z"
#endif
        , true, false, 1);

        source = new FarField(&err, simulation, medium, scatterer,
            farFieldDirectionHost, directionSize, cn);

        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Source:makeFarFieldSource",
                "Error occurred during making %s far field source. Error msg: %s", sourceName, errorBuffer);
        }

        mxFree(farFieldDirectionHost);
    }
    else if(isActive && isEqualToName(sourceStruct, "type", "GaussianBeam", true))
    {
        mxArray* focalPointStruct = getStruct(sourceStruct, "focalPoint", true);

        ub32 focalPointsSize;
        VectorType* focalPointHost = getVectorArray(&focalPointsSize, focalPointStruct, "x", "y"
#if DIMS==3
    , "z"
#endif
        , true, false, 1);

        mxArray* focalDirectionStruct = getStruct(sourceStruct, "focalDirection", true);

        ub32 focalDirectionsSize;
        VectorType* focalDirectionHost = getVectorArray(&focalDirectionsSize, focalDirectionStruct, "x", "y"
#if DIMS==3
        , "z"
#endif
        , true, true, focalPointsSize);

        float_type gaussianApertureHost;
        getScalar(&gaussianApertureHost, sourceStruct, "aperture", true);

        //mexPrintf("Add Gaussian Beam %s: beams: %d, aperture: %f.\n", sourceName, focalPointsSize, gaussianApertureHost);
        //for(int ii = 0; ii < focalPointsSize; ii++)
        //{
        //    printf("%d: focalPoint: %f %f focal direction: %f %f \n", ii,
        //        focalPointHost[ii].x(), focalPointHost[ii].y(), focalDirectionHost[ii].x(), focalDirectionHost[ii].y());
        //}

        source = new GaussianBeam(&err, simulation, medium, scatterer,
            focalPointHost, focalDirectionHost, gaussianApertureHost, focalPointsSize ,cn);

        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Source:makeGaussainBeamSource",
                "Error occurred during making %s gaussian beam source. Error msg: %s", sourceName, errorBuffer);
        }

        mxFree(focalDirectionHost);
        mxFree(focalPointHost);
    }
    else if(isActive)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:unsupportedSource",
            "Unsupported source type. Available %s types are PointSource, FarField and GaussianBeam.",
            sourceName);
    }
    mxFree(errorBuffer);

    return source;
}

Sampler* buildSampler(const mxArray* prhs[], Source* illuminationsHandle, Source* illuminationsHandle2,
        Source* viewsHandle, Source* viewsHandle2,
		const Simulation* simulation, const Medium* mediumHandler, const Scattering* scatteringHandler, ub32 materialNum)
{
    ErrorType err;
    char* errorBuffer = (char*) mxCalloc(200, sizeof(char));

    Sampler* sampler = NULL;
    mxArray* samplerStruct = getStruct(prhs[0], "sampler", true);

    ub32 tSize;
    float_type* tHost;
    float_type* dHost;
    VectorType* uHost;

    if(isEqualToName(samplerStruct, "type", "TemporalSampler", true) || isEqualToName(samplerStruct, "type", "TemporalCorrelationSampler", true) )
    {
        ub32 tmp;
        tHost = getScalarArray<float_type>(&tSize, samplerStruct, "t", true, false, 1);
        dHost = getScalarArray<float_type>(&tmp, samplerStruct, "D", true, true, materialNum);
        mxArray* uStruct = getStruct(samplerStruct, "U", true);
        uHost = getVectorArray(&tmp, uStruct, "x", "y"
#if DIMS==3
    , "z"
#endif
        , true, true, materialNum);
    }

    if(isEqualToName(samplerStruct, "type", "TemporalSampler", true))
    {
        // optional
        bool isTemporalAmplitude = true;
        mxArray* isTemporalAmplitudeMatlab = mxGetField(samplerStruct, 0, "isTemporalAmplitude");
        if(isTemporalAmplitudeMatlab != NULL)
        {
            if(!mxIsLogicalScalar(isTemporalAmplitudeMatlab))
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:samplerIsTemporalAmplitudeNotLogical",
                    "sampler isTemporalAmplitude field must be a logical scalar.");
            }

            isTemporalAmplitude = mxIsLogicalScalarTrue(isTemporalAmplitudeMatlab);
        }

        bool isForcedInsideBin = false;
        mxArray* isForcedInsideBinMatlab = mxGetField(samplerStruct, 0, "isForcedInsideBin");
        if(isForcedInsideBinMatlab != NULL)
        {
            if(!mxIsLogicalScalar(isForcedInsideBinMatlab))
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:samplerIsForcedInsideBinNotLogical",
                    "sampler isForcedInsideBin field must be a logical scalar.");
            }

            isForcedInsideBin = mxIsLogicalScalarTrue(isForcedInsideBinMatlab);
        }

        //mexPrintf("Build temporal sampler with ticks number of %d. t = [", tSize);
        //for(int ii = 0; ii < tSize; ii++)
        //{
        //    mexPrintf("%f ",tHost[ii]);
        //}
        //mexPrintf("];\n");

        sampler = new TemporalSampler(&err, illuminationsHandle, viewsHandle, simulation,
            mediumHandler, scatteringHandler, tHost, tSize, isTemporalAmplitude, isForcedInsideBin);

        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Sampler:makeTemporalSampler",
                "Error occurred during making a temporal sampler. Error msg: %s", errorBuffer);
        }

        for(ub32 mNum = 0; mNum < materialNum; mNum++)
        {
            if(dHost[mNum] < 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:temporalSamplerNegativeD",
                    "D values in temporal sampler must be non-negative");
            }

            err = (dynamic_cast<TemporalSampler*>(sampler))->setMaterial(dHost[mNum], uHost[mNum], mNum);
            if(err != ErrorType::NO_ERROR)
            {
                getErrorMsg(err, &errorBuffer);
                mexErrMsgIdAndTxt( "SSTMC:Sampler:setMaterial",
                    "Error occurred during setting material %d to temporal sampler. Error msg: %s", mNum, errorBuffer);
            }
        }

        mxFree(tHost);
        mxFree(dHost);
        mxFree(uHost);

    }
    else if(isEqualToName(samplerStruct, "type", "TemporalCorrelationSampler", true))
    {
        if(illuminationsHandle2 == NULL || viewsHandle2 == NULL)
        {
            mexErrMsgIdAndTxt( "SSTMC:Sampler:temporalCorrelationNo2",
                "For correlation sampler, source.illuminationsHandle2 and source.viewsHandle2 must be entered");
        }

        if(illuminationsHandle->getSourceSize() != illuminationsHandle2->getSourceSize())
        {
            mexErrMsgIdAndTxt( "SSTMC:Sampler:badIllumination",
                "Illumination 1 in size of %d, Illumination 2 in size of %d. Both illuminations must be at the same size", illuminationsHandle->getSourceSize() , illuminationsHandle2->getSourceSize());
        }

        if(viewsHandle->getSourceSize() != viewsHandle2->getSourceSize())
        {
            mexErrMsgIdAndTxt( "SSTMC:Sampler:badView",
                "View 1 in size of %d, View 2 in size of %d. Both views must be at the same size", viewsHandle->getSourceSize() , viewsHandle2->getSourceSize());
        }

        sampler = new TemporalCorrelationSampler(&err, illuminationsHandle, illuminationsHandle2, viewsHandle, viewsHandle2, simulation,
            mediumHandler, scatteringHandler, tHost, tSize);

        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Sampler:makeTemporalCorrelationSampler",
                "Error occurred during making a temporal sampler. Error msg: %s", errorBuffer);
        }

        for(ub32 mNum = 0; mNum < materialNum; mNum++)
        {
            if(dHost[mNum] < 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:temporalSamplerNegativeD",
                    "D values in temporal sampler must be non-negative");
            }
            err = (dynamic_cast<TemporalCorrelationSampler*>(sampler))->setMaterial(dHost[mNum], uHost[mNum], mNum);
            if(err != ErrorType::NO_ERROR)
            {
                getErrorMsg(err, &errorBuffer);
                mexErrMsgIdAndTxt( "SSTMC:Sampler:setMaterial",
                    "Error occurred during setting material %d to temporal sampler. Error msg: %s", mNum, errorBuffer);
            }
        }

        mxFree(tHost);
        mxFree(dHost);
        mxFree(uHost);
    }
    else
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:unsupportedSampler",
            "Unsupported sapmler type. Available sampler types are TemporalSampler and TemporalCorrelationSampler");
    }

    mxFree(errorBuffer);

    return sampler;
}

Tracer* buildTracer(const mxArray* prhs[], Sampler* sampler, ub32 fullIteration, ub64 seed)
{
    ErrorType err;
    char* errorBuffer = (char*) mxCalloc(200, sizeof(char));

    mxArray* tracerStruct = mxGetField(prhs[0], 0, "tracer");

    Tracer* tracer = NULL;

    if(tracerStruct == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noTracer",
            "Input must have field named tracer.");
    }

    if(!mxIsStruct(tracerStruct))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:tracerNotStruct",
            "tracer must be a structure.");
    }

    mxArray* tracerType = mxGetField(tracerStruct, 0, "type");

    if(tracerType == NULL)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:noTracerType",
            "tracer must have field named type.");
    }
        
    if (!mxIsChar(tracerType))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidTracerType",
            "The tracer type must be a string.");
    }

    char* tracerNameBuffer = mxArrayToString(tracerType);
    if(tracerNameBuffer == NULL) 
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:invalidTracerTypeStr",
            "Could not convert tracer type name to string.");
    }

    // optional
    bool isCBS = false;
    mxArray* isCBSMatlab = mxGetField(tracerStruct, 0, "isCBS");
    if(isCBSMatlab != NULL)
    {
        if(!mxIsLogicalScalar(isCBSMatlab))
        {
            mexErrMsgIdAndTxt( "SSTMC:MATLAB:tracerCBSNotLogical",
                "tracer isCBS field must be a logical scalar.");
        }

        isCBS = mxIsLogicalScalarTrue(isCBSMatlab);
    }

    if(strcmp(tracerNameBuffer, "nee") == 0)
    {
        NextEventEstimationOptions neeOptions;

        neeOptions.fullIterationsNumber = fullIteration;
        neeOptions.seed = seed;
        neeOptions.isCBS = isCBS;

        tracer = new NextEventEstimation(&err, sampler, &neeOptions);
// mexPrintf("seed: %llu \n", seed);
        if(err != ErrorType::NO_ERROR)
        {
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Tracer:makeNextEventEstimationTracer",
                "Error occurred during making a Next Event Estimation Tracer. Error msg: %s", errorBuffer);
        }
    }
    else
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:unsupportedTracer",
            "Unsupported tracer type. Type entered: %s. Available tracer type is nee",
            tracerNameBuffer);
    }

    mxFree(tracerNameBuffer);
    mxFree(errorBuffer);
    return tracer;
}

Simulation* buildSimulation(const mxArray* prhs[], ub32* batchSize)
{
    char* errorBuffer = (char*) mxCalloc(200, sizeof(char));
    ErrorType err;

    // *** Get simulation structure (optional) *** //

    // Default values
    *batchSize = 1024;
    ub32 lambdaNum = 1;
    float_type* lambdaValues;
    bool isLambdaAssigned = false;

    mxArray* simulationStruct = mxGetField(prhs[0], 0, "simulation");

    if(simulationStruct != NULL)
    {
        mxArray* deviceNumberMatlab = mxGetField(simulationStruct, 0, "deviceNumber");
        if(deviceNumberMatlab != NULL)
        {
            if(mxGetClassID(deviceNumberMatlab) != mxDOUBLE_CLASS || mxIsComplex(deviceNumberMatlab))
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:deviceNumberType",
                    "Device number type must be non-complex double.");
            }
            cudaDeviceNum = (ub32)(*mxGetDoubles(deviceNumberMatlab));
        }

        mxArray* batchSizeMatlab = mxGetField(simulationStruct, 0, "batchSize");
        if(batchSizeMatlab != NULL)
        {
            if(mxGetClassID(batchSizeMatlab) != mxDOUBLE_CLASS || mxIsComplex(batchSizeMatlab))
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:batchSizeType",
                    "Batch size type must be non-complex positive double.");
            }
            double batchSizeDouble = (*mxGetDoubles(batchSizeMatlab));

            if(batchSizeDouble <= 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:batchSizeType",
                    "Batch size type must be non-complex positive double.");
            }

            *batchSize = (ub32)(batchSizeDouble);

            if(batchSize == 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:batchSizeType",
                    "Batch size type must be non-complex positive double.");
            }
        }

        mxArray* lambdaMatlab = mxGetField(simulationStruct, 0, "lambda");

        if(lambdaMatlab != NULL)
        {
            if ( mxGetClassID(lambdaMatlab) != mxDOUBLE_CLASS || mxIsComplex(lambdaMatlab) || mxGetNumberOfElements(lambdaMatlab) == 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:simulationInvalidLambda",
                    "lambda must be double non-complex non-empty type.");
            }

            mxDouble *lambdaMatlabDouble = mxGetDoubles(lambdaMatlab);

            if(lambdaMatlabDouble == NULL) 
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:simulationVarLambda",
                    "Could not convert lambda to double array.");
            }

            lambdaNum = mxGetNumberOfElements(lambdaMatlab);
            lambdaValues = (float_type*)mxCalloc(lambdaNum, sizeof(float_type));

            for(ub32 lambdaElement = 0 ; lambdaElement < lambdaNum; lambdaElement++)
            {
                lambdaValues[lambdaElement] = (float_type)(lambdaMatlabDouble[lambdaElement]);
            }

            isLambdaAssigned = true;
        }
    }

    if(isLambdaAssigned == false)
    {
        lambdaValues = (float_type*)mxCalloc(1, sizeof(float_type));
        lambdaValues[0] = 1.0;
    }


    // mexPrintf("Simulation: batch: %d, lambdaVal = %f, lambdaNum = %d, gpu = %d \n", *batchSize, lambdaValues[0], lambdaNum, cudaDeviceNum);
    Simulation* simulatoin = new Simulation(&err, *batchSize, lambdaValues, lambdaNum, cudaDeviceNum);

    if(err != ErrorType::NO_ERROR)
    {
        getErrorMsg(err, &errorBuffer);
        mexErrMsgIdAndTxt( "SSTMC:Simulation:makeSimulation",
            "Error occurred during making Simulation class. Error msg: %s", errorBuffer);
    }

    mxFree(errorBuffer);
    mxFree(lambdaValues);

    return simulatoin;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // check proper input and output
    if(nrhs!=1)
    {
        mexErrMsgIdAndTxt("SSTMC:MATLAB:invalidNumInputs",
            "One input required.");
    }

    if(nlhs>2)
    {
        mexErrMsgIdAndTxt("SSTMC:MATLAB:invalidNumOutput",
            "One or two output required.");
    }


    if(!mxIsStruct(prhs[0]))
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:inputNotStruct",
            "Input must be a structure.");
    }

    char* errorBuffer = (char*) mxCalloc(200, sizeof(char));
    ErrorType err;

    // Default values
    ub32 batchSize;
    ub32 fullIteration = 1;
    ub32 renderingsNum = 1;
    ub64 seed = (ub64)clock();
    mxArray* simulationStruct = mxGetField(prhs[0], 0, "simulation");

    if(simulationStruct != NULL)
    {
        mxArray* fullIterationMatlab = mxGetField(simulationStruct, 0, "fullIteration");
        if(fullIterationMatlab != NULL)
        {
            if(mxGetClassID(fullIterationMatlab) != mxDOUBLE_CLASS || mxIsComplex(fullIterationMatlab))
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:fullIterationType",
                    "Full iteraion type must be non-complex positive double.");
            }
            double fullIterationDouble = (*mxGetDoubles(fullIterationMatlab));

            if(fullIterationDouble <= 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:fullIterationType",
                    "Full iteraion type must be non-complex positive double.");
            }

            fullIteration = (ub32)(fullIterationDouble);

            if(fullIteration == 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:fullIterationType",
                    "Full iteraion type must be non-complex positive double.");
            }
        }

        mxArray* renderingsNumMatlab = mxGetField(simulationStruct, 0, "renderingsNum");
        if(renderingsNumMatlab != NULL)
        {
            if(mxGetClassID(renderingsNumMatlab) != mxDOUBLE_CLASS || mxIsComplex(renderingsNumMatlab))
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:renderingsNumType",
                    "Renderings num type must be non-complex positive double.");
            }
            double renderingsNumDouble = (*mxGetDoubles(renderingsNumMatlab));
            
            if(renderingsNumDouble <= 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:renderingsNumType",
                    "Renderings num type must be non-complex positive double.");
            }

            renderingsNum = (ub32)(renderingsNumDouble);

            if(renderingsNum == 0)
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:renderingsNumType",
                    "Renderings num type must be non-complex positive double.");
            }
        }

        mxArray* seedMatlab = mxGetField(simulationStruct, 0, "seed");
        if(seedMatlab != NULL)
        {
            if(mxIsComplex(seedMatlab))
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:seedComplex",
                    "seed number can't be complex.");
            }
            if(mxGetClassID(seedMatlab) == mxDOUBLE_CLASS)
            {
                seed = (ub64)(abs(*mxGetDoubles(seedMatlab)));
            }
            else if(mxGetClassID(seedMatlab) == mxUINT64_CLASS)
            {
                seed = (ub64)(*mxGetUint64s(seedMatlab));
            }
            else
            {
                mexErrMsgIdAndTxt( "SSTMC:MATLAB:seedType",
                    "Seed number must be a double or unsigned 64 integer.");
            }
        }
    }

    // Build scattering structre
    ub32 materialNum;
    ub32 Nl, Nv, Nt;

    Simulation* simulation = buildSimulation(prhs, &batchSize);
    Scattering* scatterer = buildScattering(prhs, &materialNum);

	// Build medium structre
    Medium* medium = buildMedium(prhs, materialNum);
  
    // Build illumination and view structre
    Source* illuminationSource = buildSource(prhs, scatterer, simulation, medium, materialNum, ConnectionType::ConnectionTypeIllumination);
    Source* viewSource = buildSource(prhs, scatterer, simulation, medium, materialNum, ConnectionType::ConnectionTypeView);

    Source* illuminationSource2 = buildSource(prhs, scatterer, simulation, medium, materialNum, ConnectionType::ConnectionTypeIllumination2);
    Source* viewSource2 = buildSource(prhs, scatterer, simulation, medium, materialNum, ConnectionType::ConnectionTypeView2);
    
    // Build sampler structre
    Sampler* sampler = buildSampler(prhs, illuminationSource, illuminationSource2, viewSource, viewSource2,
		simulation, medium, scatterer, materialNum);
    
    // Build tracer
    Tracer* tracer = buildTracer(prhs, sampler, fullIteration, seed);

    Nl = sampler->getIlluminationSize();
    Nv = sampler->getViewSize();
    Nt = sampler->getSamplerSize();

    // allocate the returned field
     
    mxClassID returnedClassID = 
#if PRECISION==DOUBLE
        mxDOUBLE_CLASS
#else
        mxSINGLE_CLASS
#endif
        ;

    // mexPrintf("%d %d %d %d \n", Nl, Nv, Nt, renderingsNum);
   /*

        mwSize returnedDims[4];
    returnedDims[0] = 1;
    returnedDims[1] = 2;
    returnedDims[2] = 3;
    returnedDims[3] = 4;


    mxArray* returnedField = mxCreateNumericArray(4, returnedDims, returnedClassID, mxCOMPLEX);

    mxArray* iterationsNum;

        mwSize iterationsDims[2];
        iterationsDims[0] = 1;
        iterationsDims[1] = 4;

        iterationsNum = mxCreateNumericArray(2, iterationsDims, returnedClassID, mxREAL);
*/
    mwSize returnedDims[4];
    returnedDims[0] = Nl;
    returnedDims[1] = Nv;
    returnedDims[2] = Nt;
    returnedDims[3] = renderingsNum;


    mxArray* returnedField = mxCreateNumericArray(4, returnedDims, returnedClassID, mxCOMPLEX);

    if(returnedField == 0)
    {
        mexErrMsgIdAndTxt( "SSTMC:MATLAB:fieldAllocation",
            "Allocation of field has failed. Requried allocation size: %d x %d x %d x %d",
            Nl, Nv, Nt, renderingsNum);
    }

    // Get pointer to field
    ComplexType* v = (ComplexType*)
#if PRECISION==DOUBLE
        mxGetComplexDoubles(returnedField)
#else
        mxGetComplexSingles(returnedField)
#endif
        ;

    // allocate number of iterations
    mxArray* iterationsNum;
    float_type* totalIterations;

    if(nlhs >= 2)
    {
        mwSize iterationsDims[2];
        iterationsDims[0] = 1;
        iterationsDims[1] = renderingsNum;

        iterationsNum = mxCreateNumericArray(2, iterationsDims, returnedClassID, mxREAL);

        if(iterationsNum == 0)
        {
            mexErrMsgIdAndTxt( "SSTMC:MATLAB:iterationsAllocation",
                "Allocation of iterations number has failed.");
        }

        totalIterations = (float_type*)
#if PRECISION==DOUBLE
        mxGetDoubles(iterationsNum)
#else
        mxGetSingles(iterationsNum)
#endif
        ;
    }

    for(ub32 fieldNum = 0; fieldNum < renderingsNum; fieldNum++)
    {
        err = tracer->trace(v + fieldNum * Nl * Nv * Nt);

        if (err != ErrorType::NO_ERROR)
		{
            getErrorMsg(err, &errorBuffer);
            mexErrMsgIdAndTxt( "SSTMC:Tracer:trace",
                "Error occurred during the tracing. Error msg: %s", errorBuffer);
		}

        if(nlhs >= 2)
        {
            totalIterations[fieldNum] = (float_type)(batchSize * fullIteration + tracer->getZeroContributionPaths());
        }

        seed++;
        tracer->updateSeed(seed);
    }

    mxFree(errorBuffer);
    delete tracer;
    delete simulation;
    delete scatterer;
    delete medium;
    delete illuminationSource;
    delete viewSource;
    if(illuminationSource2)
    {
        delete illuminationSource2;
    }
    if(viewSource2)
    {
        delete viewSource2;
    }
    delete sampler;

    plhs[0] = returnedField;
    if(nlhs >= 2)
    {
        plhs[1] = iterationsNum;
    }
}


