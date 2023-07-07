#pragma once

#include "Simulation.cuh"
#include "ComplexType.cuh"

// ------------------------------------ Kernels ------------------------------------ //

// A = const
__global__ void fillKernel(ComplexType* A, float_type c, ub32 elementsNum)
{
	ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

	if (threadNum < elementsNum)
	{
		A[threadNum] = ComplexType(c, 0.);
	}
}

// A = A + B
__global__ void sumKernel(ComplexType* A, const ComplexType* B, ub32 elementsNum)
{
	ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

	if (threadNum < elementsNum)
	{
		A[threadNum] = A[threadNum] + B[threadNum];
	}
}

__global__ void sumKernel(ComplexType* A, const ComplexType* B, const ib32* matrixIdx, ub32 N, ub32 elementsNum)
{
	ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;
	ub32 m = threadNum % N;
	ub32 n = threadNum / N;
	ib32 matA_Idx = matrixIdx[n] * N + m;

	if (threadNum < elementsNum)
	{
		A[matA_Idx] = A[matA_Idx] + B[threadNum];
		// printf("thread %d: m = %d, n = %d, N = %d, matA_Idx = %d, matrixIdx[n] = %d \n", threadNum, m, n, N, matA_Idx, matrixIdx[n]);
	}
}

template<bool isConjA, bool isConjB>
__global__ void pointwiseMultipicationKernel(ComplexType* C, const ComplexType* A, const ComplexType* B,
	float_type alpha, float_type beta, ub32 uniqueElementsA, ub32 uniqueElementsB, ub32 sharedElements)
{
	ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

	ub32 aIdx = threadNum % uniqueElementsA;
	ub32 bIdx = (threadNum / uniqueElementsA) % uniqueElementsB;
	ub32 sIdx = threadNum / (uniqueElementsA * uniqueElementsB);

	if (sIdx < sharedElements)
	{
		ComplexType aElem = A[aIdx + sIdx * uniqueElementsA];
		ComplexType bElem = B[bIdx + sIdx * uniqueElementsB];

		if (isConjA)
		{
			aElem.conj();
		}

		if (isConjB)
		{
			bElem.conj();
		}

		C[threadNum] = cfma(alpha * aElem, bElem, beta * C[threadNum]);
	}
}

template<bool isConjA, bool isConjB>
__global__ void pointwiseMultipicationKernel(ComplexType* C, const ComplexType* A, const ComplexType* B,
	ub32 uniqueElementsA, ub32 uniqueElementsB, ub32 sharedElements)
{
	ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

	ub32 aIdx = threadNum % uniqueElementsA;
	ub32 bIdx = (threadNum / uniqueElementsA) % uniqueElementsB;
	ub32 sIdx = threadNum / (uniqueElementsA * uniqueElementsB);

	if (sIdx < sharedElements)
	{
		ComplexType aElem = A[aIdx + sIdx * uniqueElementsA];
		ComplexType bElem = B[bIdx + sIdx * uniqueElementsB];

		if (isConjA)
		{
			aElem.conj();
		}

		if (isConjB)
		{
			bElem.conj();
		}

		C[threadNum] = aElem * bElem;
	}
}

template<bool isConjA, bool isConjB>
__global__ void pointwiseMultipicationSharedKernel(ComplexType* C, const ComplexType* A, const ComplexType* B,
	float_type alpha, float_type beta, ub32 sharedElements)
{
	ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

	if (threadNum < sharedElements)
	{
		ComplexType aElem = A[threadNum];
		ComplexType bElem = B[threadNum];

		if (isConjA)
		{
			aElem.conj();
		}

		if (isConjB)
		{
			bElem.conj();
		}

		C[threadNum] = cfma(alpha * aElem, bElem, beta * C[threadNum]);
	}
}

template<bool isConjA, bool isConjB>
__global__ void pointwiseMultipicationSharedKernel(ComplexType* C, const ComplexType* A, const ComplexType* B, ub32 sharedElements)
{
	ub32 threadNum = threadIdx.x + blockDim.x * blockIdx.x;

	if (threadNum < sharedElements)
	{
		ComplexType aElem = A[threadNum];
		ComplexType bElem = B[threadNum];

		if (isConjA)
		{
			aElem.conj();
		}

		if (isConjB)
		{
			bElem.conj();
		}

		C[threadNum] = aElem * bElem;
	}
}

// ------------------------------------ Declarations ------------------------------------ //

// C = alpha * (A .* B) + beta * C
// A size of uniqueElementsA x 1               x sharedElements
// B size of 1               x uniqueElementsB x sharedElements
// C size of uniqueElementsA x uniqueElementsB x sharedElements
// If isCopyToA is true, we perform A = A .* B, where C is temporal vector
ErrorType pointwiseMultipication(ComplexType* C,
	ComplexType* A, ub32 uniqueElementsA, bool isConjA,
	const ComplexType* B, ub32 uniqueElementsB, bool isConjB,
	ub32 sharedElements, bool isCopyToA = false, float_type alpha = (float_type)1.0, float_type beta = (float_type)0.0);

// Here A, B and C have shared N elements
ErrorType pointwiseMultipication(ComplexType* C, ComplexType* A, bool isConjA, const ComplexType* B, bool isConjB, ub32 N,
	bool isCopyToA = false, float_type alpha = (float_type)1.0, float_type beta = (float_type)0.0);

// Nonconj versions
ErrorType pointwiseMultipication(ComplexType* C,
	ComplexType* A, ub32 uniqueElementsA,
	const ComplexType* B, ub32 uniqueElementsB,
	ub32 sharedElements, bool isCopyToA = false, float_type alpha = (float_type)1.0, float_type beta = (float_type)0.0);

ErrorType pointwiseMultipication(ComplexType* C, ComplexType* A, const ComplexType* B, ub32 N,
	bool isCopyToA = false, float_type alpha = (float_type)1.0, float_type beta = (float_type)0.0);

// C = alpha * (A * B) + beta * C
// A size of rowsA x colsA
// B size of colsA x colsB
// C size of rowsA x colsB
ErrorType matrixMultipication(ComplexType* C, ComplexType* A, const ComplexType* B,
	ub32 rowsA, ub32 colsA, ub32 colsB,
	bool isCopyToA = false, float_type alpha = (float_type)1.0, float_type beta = (float_type)0.0);

// A = A + B
ErrorType matrixSum(ComplexType* A, const ComplexType* B, ub32 elementsNum);

// A = const
ErrorType matrixFill(ComplexType* A, float_type c, ub32 elementsNum);

// ------------------------------------ Implementations ------------------------------------ //

ErrorType pointwiseMultipication(ComplexType* C_in,
	ComplexType* A_in, ub32 uniqueElementsA, bool isConjA,
	const ComplexType* B_in, ub32 uniqueElementsB, bool isConjB,
	ub32 sharedElements, bool isCopyToA, float_type alpha, float_type beta)
{
#if 0
	float2_type* A = (float2_type*)A_in;
	float2_type* B = (float2_type*)B_in;
	float2_type* C = (float2_type*)C_in;
	float2_type complexAlpha = float2_type{ alpha , 0};
	float2_type complexBeta = float2_type{ beta ,0 };

	// Pointwise non-conj multipication non-sum all shared
	if (isConjA == false && isConjB == false && abs(alpha - (float_type)1.0) < EPSILON && abs(beta) < EPSILON && uniqueElementsA == 1 && uniqueElementsB == 1)
	{
		if (
#if PRECISION==DOUBLE
			cublasZdgmm(cublasHandle, CUBLAS_SIDE_RIGHT, 1, sharedElements, A, 1, B, 1, C, 1)
#else
			cublasCdgmm(cublasHandle, CUBLAS_SIDE_RIGHT, 1, sharedElements, A, 1, B, 1, C, 1)
#endif
			!= CUBLAS_STATUS_SUCCESS) return ErrorType::CUBLAS_ERROR;
		cudaDeviceSynchronize();
		if (isCopyToA && cudaMemcpy(A, C, sizeof(ComplexType) * sharedElements, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
		{
			return ErrorType::ALLOCATION_ERROR;
		}
		return ErrorType::NO_ERROR;
	}

	// Pointwise non-conj multipication all shared
	if (isConjA == false && isConjB == false && uniqueElementsA == 1 && uniqueElementsB == 1)
	{
		if (
#if PRECISION==DOUBLE
			cublasZgbmv(cublasHandle, CUBLAS_OP_N, sharedElements, sharedElements, 0, 0, &complexAlpha, A, 1, B, 1, &complexBeta, C, 1)
#else
			cublasCgbmv(cublasHandle, CUBLAS_OP_N, sharedElements, sharedElements, 0, 0, &complexAlpha, A, 1, B, 1, &complexBeta, C, 1)
#endif
			!= CUBLAS_STATUS_SUCCESS) return ErrorType::CUBLAS_ERROR;
		cudaDeviceSynchronize();
		if (isCopyToA && cudaMemcpy(A, C, sizeof(ComplexType) * sharedElements, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
		{
			return ErrorType::ALLOCATION_ERROR;
		}
		return ErrorType::NO_ERROR;
	}

	// Pointwise conj multipication all shared
	if (isConjA == true && isConjB == false && uniqueElementsA == 1 && uniqueElementsB == 1)
	{
		if (
#if PRECISION==DOUBLE
			cublasZgbmv(cublasHandle, CUBLAS_OP_C, sharedElements, sharedElements, 0, 0, &complexAlpha, A, 1, B, 1, &complexBeta, C, 1)
#else
			cublasCgbmv(cublasHandle, CUBLAS_OP_C, sharedElements, sharedElements, 0, 0, &complexAlpha, A, 1, B, 1, &complexBeta, C, 1)
#endif
			!= CUBLAS_STATUS_SUCCESS) return ErrorType::CUBLAS_ERROR;
		cudaDeviceSynchronize();
		if (isCopyToA && cudaMemcpy(A, C, sizeof(ComplexType) * sharedElements, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
		{
			return ErrorType::ALLOCATION_ERROR;
		}
		return ErrorType::NO_ERROR;
	}

	if (isConjA == false && isConjB == true && uniqueElementsA == 1 && uniqueElementsB == 1)
	{
		if (
#if PRECISION==DOUBLE
			cublasZgbmv(cublasHandle, CUBLAS_OP_C, sharedElements, sharedElements, 0, 0, &complexAlpha, B, 1, A, 1, &complexBeta, C, 1)
#else
			cublasCgbmv(cublasHandle, CUBLAS_OP_C, sharedElements, sharedElements, 0, 0, &complexAlpha, B, 1, A, 1, &complexBeta, C, 1)
#endif
			!= CUBLAS_STATUS_SUCCESS) return ErrorType::CUBLAS_ERROR;
		cudaDeviceSynchronize();
		if (isCopyToA && cudaMemcpy(A, C, sizeof(ComplexType) * sharedElements, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
		{
			return ErrorType::ALLOCATION_ERROR;
		}
		return ErrorType::NO_ERROR;
	}

	// Pointwise full multipication
	if (
#if PRECISION==DOUBLE
		cublasZgemmStridedBatched(cublasHandle,
			isConjA ? CUBLAS_OP_C : CUBLAS_OP_N,
			isConjB ? CUBLAS_OP_C : CUBLAS_OP_N,
			uniqueElementsA, uniqueElementsB, 1,
			&complexAlpha,
			A, isConjA ? 1 : uniqueElementsA, uniqueElementsA,
			B, isConjB ? uniqueElementsB : 1, uniqueElementsB,
			&complexBeta,
			C, uniqueElementsA, uniqueElementsA * uniqueElementsB,
			sharedElements)
#else
	cublasCgemmStridedBatched(cublasHandle,
		isConjA ? CUBLAS_OP_C : CUBLAS_OP_N,
		isConjB ? CUBLAS_OP_C : CUBLAS_OP_N,
		uniqueElementsA, uniqueElementsB, 1,
		&complexAlpha,
		A, isConjA ? 1 : uniqueElementsA, uniqueElementsA,
		B, isConjB ? uniqueElementsB : 1, uniqueElementsB,
		&complexBeta,
		C, uniqueElementsA, uniqueElementsA * uniqueElementsB,
		sharedElements)
#endif
		!= CUBLAS_STATUS_SUCCESS) return ErrorType::CUBLAS_ERROR;
		cudaDeviceSynchronize();
#else

	ub32 totalThreads, threadsNum, blocksNum;

	// Pointwise non-conj multipication non-sum all shared
	if (abs(alpha - (float_type)1.0) < EPSILON && abs(beta) < EPSILON && uniqueElementsA == 1 && uniqueElementsB == 1)
	{
		totalThreads = sharedElements;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		if (isConjA == false && isConjB == false)
		{
			pointwiseMultipicationSharedKernel<false, false> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, sharedElements);
		}
		else if (isConjA == true && isConjB == false)
		{
			pointwiseMultipicationSharedKernel<true, false> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, sharedElements);
		}
		else if (isConjA == false && isConjB == true)
		{
			pointwiseMultipicationSharedKernel<false, true> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, sharedElements);
		}
		else
		{
			pointwiseMultipicationSharedKernel<true, true> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, sharedElements);
		}
	}
	else if (uniqueElementsA == 1 && uniqueElementsB == 1)
	{
		totalThreads = sharedElements;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		if (isConjA == false && isConjB == false)
		{
			pointwiseMultipicationSharedKernel<false, false> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, alpha, beta, sharedElements);
		}
		else if (isConjA == true && isConjB == false)
		{
			pointwiseMultipicationSharedKernel<true, false> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, alpha, beta, sharedElements);
		}
		else if (isConjA == false && isConjB == true)
		{
			pointwiseMultipicationSharedKernel<false, true> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, alpha, beta, sharedElements);
		}
		else
		{
			pointwiseMultipicationSharedKernel<true, true> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, alpha, beta, sharedElements);
		}
	}
	else if (abs(alpha - (float_type)1.0) < EPSILON && abs(beta) < EPSILON)
	{
		totalThreads = uniqueElementsA * uniqueElementsB * sharedElements;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		if (isConjA == false && isConjB == false)
		{
			pointwiseMultipicationKernel<false, false> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, uniqueElementsA, uniqueElementsB, sharedElements);
		}
		else if (isConjA == true && isConjB == false)
		{
			pointwiseMultipicationKernel<true, false> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, uniqueElementsA, uniqueElementsB, sharedElements);
		}
		else if (isConjA == false && isConjB == true)
		{
			pointwiseMultipicationKernel<false, true> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, uniqueElementsA, uniqueElementsB, sharedElements);
		}
		else
		{
			pointwiseMultipicationKernel<true, true> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, uniqueElementsA, uniqueElementsB, sharedElements);
		}
	}
	else
	{
		totalThreads = uniqueElementsA * uniqueElementsB * sharedElements;
		threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
		blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

		if (isConjA == false && isConjB == false)
		{
			pointwiseMultipicationKernel<false, false> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, alpha, beta, uniqueElementsA, uniqueElementsB, sharedElements);
		}
		else if (isConjA == true && isConjB == false)
		{
			pointwiseMultipicationKernel<true, false> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, alpha, beta, uniqueElementsA, uniqueElementsB, sharedElements);
		}
		else if (isConjA == false && isConjB == true)
		{
			pointwiseMultipicationKernel<false, true> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, alpha, beta, uniqueElementsA, uniqueElementsB, sharedElements);
		}
		else
		{
			pointwiseMultipicationKernel<true, true> <<<blocksNum, threadsNum >>> (C_in, A_in, B_in, alpha, beta, uniqueElementsA, uniqueElementsB, sharedElements);
		}
	}

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}

	if (isCopyToA && cudaMemcpy(A_in, C_in, sizeof(ComplexType) * uniqueElementsA * uniqueElementsB * sharedElements, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}
	return ErrorType::NO_ERROR;

#endif
}

ErrorType pointwiseMultipication(ComplexType* C, ComplexType* A, bool isConjA, const ComplexType* B, bool isConjB, ub32 N,
	bool isCopyToA, float_type alpha, float_type beta)
{
	return pointwiseMultipication(C,
		A, 1, isConjA,
		B, 1, isConjB,
		N, isCopyToA, alpha, beta);
}

ErrorType pointwiseMultipication(ComplexType* C,
	ComplexType* A, ub32 uniqueElementsA,
	const ComplexType* B, ub32 uniqueElementsB,
	ub32 sharedElements, bool isCopyToA, float_type alpha, float_type beta)
{
	return pointwiseMultipication(C,
		A, uniqueElementsA, false,
		B, uniqueElementsB, false,
		sharedElements, isCopyToA, alpha, beta);
}

ErrorType pointwiseMultipication(ComplexType* C, ComplexType* A, const ComplexType* B, ub32 N,
	bool isCopyToA, float_type alpha, float_type beta)
{
	return pointwiseMultipication(C,
		A, false,
		B, false,
		N, isCopyToA, alpha, beta);
}

ErrorType matrixMultipication(ComplexType* C_in, ComplexType* A_in, const ComplexType* B_in,
	ub32 rowsA, ub32 colsA, ub32 colsB, bool isCopyToA, float_type alpha, float_type beta)
{
	float2_type* A = (float2_type*)A_in;
	float2_type* B = (float2_type*)B_in;
	float2_type* C = (float2_type*)C_in;
	float2_type complexAlpha = float2_type{ alpha, 0.0 };
	float2_type complexBeta = float2_type{ beta, 0.0 };

	// Scalar multipication
	if (colsA == 1 && colsB == 1 && abs(beta - (float_type)1.0) < EPSILON)
	{
		ComplexType b;

		if (cudaMemcpy(&b, B_in, sizeof(ComplexType), cudaMemcpyKind::cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			return ErrorType::ALLOCATION_ERROR;
		}
		b = alpha * b;
		complexAlpha = float2_type{ b.real(), b.imag() };
		
		if (
#if PRECISION==DOUBLE
			cublasZaxpy(cublasHandle, rowsA,
				&complexAlpha,
				A, 1,
				C, 1)
#else
			cublasCaxpy(cublasHandle, rowsA,
				&complexAlpha,
				A, 1,
				C, 1)
#endif
			!= CUBLAS_STATUS_SUCCESS) return ErrorType::CUBLAS_ERROR;

		cudaDeviceSynchronize();
		if (isCopyToA && cudaMemcpy(A, C, sizeof(ComplexType) * rowsA, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
		{
			return ErrorType::ALLOCATION_ERROR;
		}

		return ErrorType::NO_ERROR;
	}

	// Matrix-vector multipication
	if (colsB == 1)
	{
		if (
#if PRECISION==DOUBLE
			cublasZgemv(cublasHandle,
				CUBLAS_OP_N,
				rowsA, colsA,
				&complexAlpha, A,
				rowsA,
				B, 1, &complexBeta, C, 1)
#else
			cublasCgemv(cublasHandle,
				CUBLAS_OP_N,
				rowsA, colsA,
				&complexAlpha, A,
				rowsA,
				B, 1, &complexBeta, C, 1)
#endif
		!= CUBLAS_STATUS_SUCCESS) return ErrorType::CUBLAS_ERROR;
		
		cudaDeviceSynchronize();
		if (isCopyToA && cudaMemcpy(A, C, sizeof(ComplexType) * rowsA, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
		{
			return ErrorType::ALLOCATION_ERROR;
		}

		return ErrorType::NO_ERROR;
	}

	// Matrix-Matrix multipication
	if (
#if PRECISION==DOUBLE
		cublasZgemm(cublasHandle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			rowsA, colsB, colsA,
			&complexAlpha,
			A, rowsA,
			B, colsA,
			&complexBeta,
			C, rowsA)
#else
		cublasCgemm(cublasHandle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			rowsA, colsB, colsA,
			&complexAlpha,
			A, rowsA,
			B, colsA,
			&complexBeta,
			C, rowsA)
#endif
		!= CUBLAS_STATUS_SUCCESS) return ErrorType::CUBLAS_ERROR;

	cudaDeviceSynchronize();
	if (isCopyToA && cudaMemcpy(A, C, sizeof(ComplexType) * rowsA * colsB, cudaMemcpyKind::cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
	{
		return ErrorType::ALLOCATION_ERROR;
	}

	return ErrorType::NO_ERROR;
}

ErrorType matrixSum(ComplexType* A, const ComplexType* B, ub32 elementsNum)
{
	ub32 totalThreads = elementsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	sumKernel <<<blocksNum, threadsNum >>> (A, B, elementsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}
	return ErrorType::NO_ERROR;
}

ErrorType matrixSum(ComplexType* A, const ComplexType* B, const ib32* matrixIdx, ub32 N, ub32 elementsNum)
{
	ub32 totalThreads = elementsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	sumKernel <<<blocksNum, threadsNum >>> (A, B, matrixIdx, N, elementsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}
	return ErrorType::NO_ERROR;
}

// A = const
ErrorType matrixFill(ComplexType* A, float_type c, ub32 elementsNum)
{
	ub32 totalThreads = elementsNum;
	ub32 threadsNum = totalThreads < THREADS_NUM ? totalThreads : THREADS_NUM;
	ub32 blocksNum = (totalThreads - 1) / THREADS_NUM + 1;

	fillKernel <<<blocksNum, threadsNum >>> (A, c, elementsNum);

	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaError_t::cudaSuccess)
	{
		return ErrorType::KERNEL_ERROR;
	}
	return ErrorType::NO_ERROR;
}
