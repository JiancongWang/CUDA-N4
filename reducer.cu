// This file contains the reduce kernel and its wrapper. The code comes from CUDA
// sdk example "CUDA_path/Samples/6_Advanced/reduction". This uses the kernel 4
// since it doesn't require the input to be power of 2 yet still retain most optimization
// be definitely fast enough to handle 500cube image reduction

#include <stdio.h>
#include <iostream>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include "helper_functions.h"
#include "helper_cuda.h"

#include "reducer.h"

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T> struct SharedMemory{
	__device__ inline operator T *(){
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<> struct SharedMemory<double>{
	__device__ inline operator double *(){
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};

/*
    Comments from the SDK.
    This version uses the warp shuffle operation if available to reduce
    warp synchronization. When shuffle is not available the final warp's
    worth of work is unrolled to reduce looping overhead.

    See http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    for additional information about using shuffle to perform a reduction
    within a warp.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
 */
__global__ void reduce_sum_kernel(float * g_idata, float * g_odata, unsigned int n, unsigned int blockSize){
	float * sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	// Each thread corresponding to two values in this kernel. That's where the blockDim.x * 2 and the if comes from
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	float mySum = (i < n) ? g_idata[i] : 0;

	if (i + blockSize < n)
		mySum += g_idata[i+blockSize];

	sdata[tid] = mySum;
	__syncthreads();

	// do reduction in shared mem. When reduced down to single warp shift to
	// warp shuffling.
	for (unsigned int s=blockDim.x/2; s>32; s>>=1){
		if (tid < s){
			sdata[tid] = mySum = mySum + sdata[tid + s];
		}
		__syncthreads();
	}

	if ( tid < 32 ){
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) mySum += sdata[tid + 32];
		for (int offset = warpSize/2; offset > 0; offset /= 2){
			mySum += __shfl_down(mySum, offset);
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = mySum;
}

__global__ void reduce_max_kernel(float * g_idata, float * g_odata, unsigned int n, unsigned int blockSize){
	float * sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	// Each thread corresponding to two values in this kernel. That's where the blockDim.x * 2 and the if comes from
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	float myMax = (i < n) ? g_idata[i] : -10000000000000.0;

	if (i + blockSize < n)
		myMax = max(myMax, g_idata[i+blockSize]);

	sdata[tid] = myMax;
	__syncthreads();

	// do reduction in shared mem. When reduced down to single warp shift to
	// warp shuffling.
	for (unsigned int s=blockDim.x/2; s>32; s>>=1){
		if (tid < s){
			sdata[tid] = myMax = max(myMax , sdata[tid + s]);
		}
		__syncthreads();
	}

	if ( tid < 32 ){
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) myMax = max(myMax, sdata[tid + 32]);
		for (int offset = warpSize/2; offset > 0; offset /= 2){
			myMax = max( myMax, __shfl_down(myMax, offset));
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = myMax;
}

__global__ void reduce_min_kernel(float * g_idata, float * g_odata, unsigned int n, unsigned int blockSize){
	float * sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	// Each thread corresponding to two values in this kernel. That's where the blockDim.x * 2 and the if comes from
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	float myMin = (i < n) ? g_idata[i] : 10000000000000.0;

	if (i + blockSize < n)
		myMin = min(myMin, g_idata[i+blockSize]);

	sdata[tid] = myMin;
	__syncthreads();

	// do reduction in shared mem. When reduced down to single warp shift to
	// warp shuffling.
	for (unsigned int s=blockDim.x/2; s>32; s>>=1){
		if (tid < s){
			sdata[tid] = myMin = min(myMin , sdata[tid + s]);
		}
		__syncthreads();
	}

	if ( tid < 32 ){
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) myMin = min(myMin, sdata[tid + 32]);
		for (int offset = warpSize/2; offset > 0; offset /= 2){
			myMin = min( myMin, __shfl_down(myMin, offset));
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = myMin;
}


/*****************************Kernel above, Helper functions and wrapper below************************************/
bool isPow2(unsigned int x){
	return ((x&(x-1))==0);
}

unsigned int nextPow2(unsigned int x){
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int &blocks, int &threads){
	//get device capability, to avoid block/grid size exceed the upper bound
	int maxThreads = 1024;
	int maxBlock = 2147483647;
	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + threads * 2 - 1) / (threads * 2);

	// Need to notice that there is limit on how large block dimension can be.
	// On my GTX980Ti the maximum blocks number on X direction is 2,147,483,647.
	// So this is completely enough for 500cube and 1000cube image so I delete
	// the check code here. Please check this with
	// CUDA Path/samples/1_Utilities/deviceQuery/deviceQuery

}

void reduce_sum_kernel_wrapper(int size, int threads, int blocks, float * d_idata, float * d_odata){
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads){
	case 1024:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 1024);
		break;
	case 512:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 512);
		break;
	case 256:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 256);
		break;
	case 128:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 128);
		break;
	case 64:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 64);
		break;
	case 32:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 32);
		break;
	case 16:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 16);
		break;
	case  8:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 8);
		break;
	case  4:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 4);
		break;
	case  2:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 2);
		break;
	case  1:
		reduce_sum_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 1);
		break;
	default:
		break;
//		std::cout<<"From function reduce_kernel_wrapper: The number of thread is not a power of 2. Check your code!"<<std::endl;
	}

}

void reduce_max_kernel_wrapper(int size, int threads, int blocks, float * d_idata, float * d_odata){
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads){
	case 1024:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 1024);
		break;
	case 512:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 512);
		break;
	case 256:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 256);
		break;
	case 128:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 128);
		break;
	case 64:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 64);
		break;
	case 32:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 32);
		break;
	case 16:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 16);
		break;
	case  8:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 8);
		break;
	case  4:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 4);
		break;
	case  2:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 2);
		break;
	case  1:
		reduce_max_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 1);
		break;
	default:
		break;
//		std::cout<<"From function reduce_kernel_wrapper: The number of thread is not a power of 2. Check your code!"<<std::endl;
	}

}

void reduce_min_kernel_wrapper(int size, int threads, int blocks, float * d_idata, float * d_odata){
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads){
	case 1024:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 1024);
		break;
	case 512:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 512);
		break;
	case 256:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 256);
		break;
	case 128:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 128);
		break;
	case 64:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 64);
		break;
	case 32:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 32);
		break;
	case 16:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 16);
		break;
	case  8:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 8);
		break;
	case  4:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 4);
		break;
	case  2:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 2);
		break;
	case  1:
		reduce_min_kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 1);
		break;
	default:
		break;
//		std::cout<<"From function reduce_kernel_wrapper: The number of thread is not a power of 2. Check your code!"<<std::endl;
	}

}


/*********************************** Helper functions above, wrapper below ********************************/
float Reducer::reduce_sum_wrapper(int n, float *d_idata, float *d_odata){
	// Create and start timer
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	float gpu_result = 0;
	int numThreads = 0;
	int numBlocks = 0;

	getNumBlocksAndThreads(n, numBlocks, numThreads);

//	std::cout<<"From function reduce_wrapper, numBlocks : "<< numBlocks << "numThreads: " << numThreads << std::endl;

	// First pass of the kernel. Basically do a reduction and copy to d_odata
	int kernelCount = 0;
	reduce_sum_kernel_wrapper(n, numThreads, numBlocks, d_idata, d_odata);
	kernelCount++;

	// sum partial block sums on GPU
	int s = numBlocks;

	// Since we are considering reducing 500cube images, the cpu reduce scheme won't save
	// use much time compared to total kernel launch. So it is omitted here.
	while (s > 1){
		getNumBlocksAndThreads(s, numBlocks, numThreads);
//		std::cout<<"From function reduce_wrapper, numBlocks : "<< numBlocks << "numThreads: " << numThreads << std::endl;
		reduce_sum_kernel_wrapper(s, numThreads, numBlocks, d_odata, d_odata);
		s = (s + (numThreads*2-1)) / (numThreads*2);
		kernelCount++;
	}

	// Stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
	double reduce_time = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	// copy final sum from device to host
	checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost));
//	std::cout<<"From function reduce_wrapper: The reduce kernel has launch "<< kernelCount << " times." << std::endl;
//	std::cout<<"Time: " << reduce_time << " ms." << std::endl;

	return gpu_result;
}

float Reducer::reduce_max_wrapper(int n, float *d_idata, float *d_odata){
	// Create and start timer
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	float gpu_result = 0;
	int numThreads = 0;
	int numBlocks = 0;

	getNumBlocksAndThreads(n, numBlocks, numThreads);

//	std::cout<<"From function reduce_wrapper, numBlocks : "<< numBlocks << "numThreads: " << numThreads << std::endl;

	// First pass of the kernel. Basically do a reduction and copy to d_odata
	int kernelCount = 0;
	reduce_max_kernel_wrapper(n, numThreads, numBlocks, d_idata, d_odata);
	kernelCount++;

	// sum partial block sums on GPU
	int s = numBlocks;

	// Since we are considering reducing 500cube images, the cpu reduce scheme won't save
	// use much time compared to total kernel launch. So it is omitted here.
	while (s > 1){
		getNumBlocksAndThreads(s, numBlocks, numThreads);
//		std::cout<<"From function reduce_wrapper, numBlocks : "<< numBlocks << "numThreads: " << numThreads << std::endl;
		reduce_max_kernel_wrapper(s, numThreads, numBlocks, d_odata, d_odata);
		s = (s + (numThreads*2-1)) / (numThreads*2);
		kernelCount++;
	}

	// Stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
	double reduce_time = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	// copy final sum from device to host
	checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost));
//	std::cout<<"From function reduce_wrapper: The reduce kernel has launch "<< kernelCount << " times." << std::endl;
//	std::cout<<"Time: " << reduce_time << " ms." << std::endl;

	return gpu_result;
}

float Reducer::reduce_min_wrapper(int n, float *d_idata, float *d_odata){
	// Create and start timer
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	float gpu_result = 0;
	int numThreads = 0;
	int numBlocks = 0;

	getNumBlocksAndThreads(n, numBlocks, numThreads);

//	std::cout<<"From function reduce_wrapper, numBlocks : "<< numBlocks << "numThreads: " << numThreads << std::endl;

	// First pass of the kernel. Basically do a reduction and copy to d_odata
	int kernelCount = 0;
	reduce_min_kernel_wrapper(n, numThreads, numBlocks, d_idata, d_odata);
	kernelCount++;

	// sum partial block sums on GPU
	int s = numBlocks;

	// Since we are considering reducing 500cube images, the cpu reduce scheme won't save
	// use much time compared to total kernel launch. So it is omitted here.
	while (s > 1){
		getNumBlocksAndThreads(s, numBlocks, numThreads);
//		std::cout<<"From function reduce_wrapper, numBlocks : "<< numBlocks << "numThreads: " << numThreads << std::endl;
		reduce_min_kernel_wrapper(s, numThreads, numBlocks, d_odata, d_odata);
		s = (s + (numThreads*2-1)) / (numThreads*2);
		kernelCount++;
	}

	// Stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
	double reduce_time = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	// copy final sum from device to host
	checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost));
//	std::cout<<"From function reduce_wrapper: The reduce kernel has launch "<< kernelCount << " times." << std::endl;
//	std::cout<<"Time: " << reduce_time << " ms." << std::endl;

	return gpu_result;
}



