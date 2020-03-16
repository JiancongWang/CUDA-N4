/* CUDA implementation of the N4 algorithm. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h> // memset
#include <complex>
// Without this the complex number multiplication/division doesn't work.
#include <fftw3.h>
#include "helper_functions.h"
#include "helper_cuda.h"
#include "reducer.h"
#include "cudaN4.h"

// This simple kernel set the background pixel to a background_value.
__global__ void set_background_kernel(
		// Input
		const float * mask,
		const float background_value,
		const unsigned int n,
		// Output
		float * data){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;
	if (mask[i]==0){
		data[i]=background_value;
	}
}

// This calculates a[i]+=b[i]
__global__ void sum_inplace_kernel(
		// Input
		float * a, // output
		const float * b,
		const unsigned int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;
	a[i]+=b[i];
}

// calculate a-b
__global__ void subtract(
		// Input
		const float * a,
		const float * b,
		const unsigned int n,
		// Output
		float * out){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;
	out[i] = a[i] - b[i];
}


// calculate exp(logBiasField) in place
__global__ void exp_kernel(
		// Input
		const float * logBiasField,
		const unsigned int n,
		// output
		float * biasField){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;
	biasField[i] = exp(logBiasField[i]);
}


// calculate img/exp(logBiasField) in place
__global__ void exp_and_divide_kernel(
		// Input
		const float * logBiasField,
		const float * im,
		const unsigned int n,
		// output
		float * im_normalized){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;
	im_normalized[i] = im[i]/exp(logBiasField[i]);
}

// This simple kernel simply calculate exp(a-b). Used in calculate bias field convergence.
__global__ void subtract_and_exp_kernel(
		// Input
		const float * a,
		const float * b,
		// Output
		const unsigned int n,
		float * out){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;
	out[i]= exp(a[i] - b[i]);
}

// Calculate a[i] = (a[i]-mean)^2
__global__ void subtract_mean_and_sqr_kernel(
		float * a,
		const float mean,
		const unsigned int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;
	float am = a[i] - mean;
	a[i] = am * am;
}

// Calculate c[i] = a[i] / (b[i]) if b[i]!=0 else 0
__global__ void divide_kernel(
		// Input
		const float * a,
		const float * b,
		const unsigned n,
		// Output
		float * c){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;

	if (b[i]!=0){
		c[i] = a[i]/b[i];
	}else{
		c[i] = 0.;
	}
}

// This function simply logs the image if mask[i]>0
__global__ void log_kernel(
		// Input
		const float * im,
		const float * mask,
		const unsigned int numberOfPixels,
		// Output
		float * im_log){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=numberOfPixels)
		return;

	if (mask[i]!=0){
		im_log[i] = log(im[i]);
	}else{
		im_log[i] = 0;
	}
}

// histogramGPU computes the histogram of an input array on the GPU
// This function is taken from
// https://github.com/kevinzakka/learn-cuda/blob/master/src/histogram.cu
// Modified to support fractional bin as the N4 implementation did.
__global__ void histogramGPU_kernel(
		// Input
		const float * input,
		const float * mask,
		const float in_min,
		const float bin_slope,
		const unsigned int numElems,
		const unsigned int numBin,
		// Output
		float * bins){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int BLOCK_SIZE = blockDim.x;

	// compute global thread coordinates
	int i = (bx * BLOCK_SIZE) + tx;

	// create a private histogram copy for each thread block
	// size same as numBin
	extern __shared__ float hist[];

	// each thread must initialize more than 1 location
	if (numBin > BLOCK_SIZE) {
		for (int j=tx; j<numBin; j+=BLOCK_SIZE) {
			hist[j] = 0.f;
		}
	}
	// use the first `PRIVATE` threads of each block to init
	else {
		if (tx < numBin) {
			hist[tx] = 0.f;
		}
	}
	// wait for all threads in the block to finish
	__syncthreads();

	// update private histogram given the mask value
	// this is safe due to short-circuit evaluation
	if ((i < numElems) && (mask[i]!=0.)) {
		// bin the input
		float cidx =(input[i] - in_min)/bin_slope;
		int idx = (int)floor(cidx);
		float offset = cidx - (float)idx;

		if( offset == 0. ){
			atomicAdd(&(hist[idx]), 1.);
		}
		else if( idx < numBin - 1 ){
			atomicAdd(&(hist[idx]), 1. - offset);
			atomicAdd(&(hist[idx+1]), offset);
		}
	}
	// wait for all threads in the block to finish
	__syncthreads();

	// each thread must update more than 1 location
	if (numBin > BLOCK_SIZE) {
		for (int j=tx; j<numBin; j+=BLOCK_SIZE) {
			atomicAdd(&(bins[j]), hist[j]);
		}
	}
	// use the first `PRIVATE` threads to update final histogram
	else {
		if (tx < numBin) {
			atomicAdd(&(bins[tx]), hist[tx]);
		}
	}
}

// This maps the value based on the histogram adjusted from sharpening the histogram distribution.
__global__ void histogramMapping_kernel(
		// Input
		const float * before_sharpen,
		const float * mask,
		const float * E,
		const float binMinimum,
		const float histogramSlope,
		const unsigned int n, // num pixel in image
		const unsigned int numBin,
		// Output
		float * after_sharpen // Image after sharpening
){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=n)
		return;
	if (mask[i]==0.)
		return;

	float cidx = ( before_sharpen[i] - binMinimum ) / histogramSlope;
	cidx = max(0., cidx); // In case numerical error send cidx<0...
	unsigned int idx = floor( cidx );

	float correctedPixel;

	if( idx < numBin - 1 ){
		correctedPixel = E[idx] + ( E[idx + 1] - E[idx] ) * ( cidx - idx );
	} else {
		correctedPixel = E[numBin - 1];
	}

	after_sharpen[i] = correctedPixel;
}

// This kernel has to be used with the upsampleLattice3D_gpu() with a fixed number of threads to produce the correct result.
__global__ void upsample_lattice_kernel_3D(
		// Input
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,

		const unsigned int nx,
		const unsigned int ny,
		const unsigned int nz,

		const unsigned int n2x,
		const unsigned int n2y,
		const unsigned int n2z,

		const float * lattice,
		// Output
		float * lattice_upsample
){
	// The index notation here is trying to match as closely as possible with the one on the numpy version.

	// blockIdx-1: -1:n+1, -1:m+1, -1:l+1

	// i: -1:n+1
	int i = (int)blockIdx.x-1;
	int j = (int)blockIdx.y-1;
	int k = (int)blockIdx.z-1;

	// blockDim.x
	int xx = threadIdx.x;
	int yy = threadIdx.y;
	int zz = threadIdx.z;
	int tx = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;

	float bw[2][3] = {
			{1./8, 6./8, 1./8},
			{0, 1./2, 1./2}
	};

	__shared__ float lattice_piece[27];

	int BLOCK_SIZE = blockDim.x * blockDim.y * blockDim.z;

	// Each threads load lattice pieces into the shared lattice_piece
	for (int h=tx; h<27; h+=BLOCK_SIZE) {
		int Z = h % 3 + k -1;
		int Y = (h/3)%3 + j -1;
		int X = h / 9 + i -1; // i-1:i+1

		// clip the input lattice point to a constant boundary
		X = min(max(X, -1), (int)(nx+1))+1;
		Y = min(max(Y, -1), (int)(ny+1))+1;
		Z = min(max(Z, -1), (int)(nz+1))+1;

		lattice_piece[h] = lattice[ X * ncpt_y * ncpt_z + Y * ncpt_z + Z ];
	}

	__syncthreads();

	// Now, sum the 27 elements weighted by the weight.
	int idx = 2*i + xx;
	int idy = 2*j + yy;
	int idz = 2*k + zz;

	if( (idx>=-1) && (idx<=(int)(2*nx+1)) &&
			(idy>=-1) && (idy<=(int)(2*ny+1)) &&
			(idz>=-1) && (idz<=(int)(2*nz+1))){
		float ls = 0.;
#pragma unroll
		for (int bx=0; bx<3; bx++){
#pragma unroll
			for (int by=0; by<3; by++){
#pragma unroll
				for (int bz=0; bz<3; bz++){
					ls+=bw[xx][bx]*bw[yy][by]*bw[zz][bz] * lattice_piece[bx * 9 + by * 3 + bz];
				}
			}
		}
		idx+=1;
		idy+=1;
		idz+=1;
		lattice_upsample[idx * n2y*n2z + idy * n2z + idz] = ls;
	}
}

// This function calculates the cubic B spline coefficients.
__device__ __inline__ void cubicBspline(
		// Input
		const float t,
		// Output
		float * B){
	float t2 = t*t;
	float t3 = t*t*t;
	float tm = 1-t;

	B[0] = tm*tm*tm/6;
	B[1] = (3*t3 - 6*t2 + 4)/6;
	B[2] = (-3*t3 + 3*t2 + 3*t + 1)/6;
	B[3] = t3/6;
}


// This evaluates the Bspline on each pixel on a field. This uses shared memory to store
// lattice value shared among nearby pixels.
__global__ void evaluate_bspline_kernel_3D(
		// Input
		const float * lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,

		const unsigned int blocks_per_span_x, // How many block is there per span
		const unsigned int blocks_per_span_y,
		const unsigned int blocks_per_span_z,

		const float span_x, // How long each span is
		const float span_y,
		const float span_z,

		const unsigned int fx, // What is the output field size
		const unsigned int fy,
		const unsigned int fz,

		// Output
		float * fitted
){
	__shared__ float lattice_piece[64];

	// Start Spans index i,j,k. This is the index into the lattice's left top corner.
	int si = blockIdx.x / (int)blocks_per_span_x;
	int sj = blockIdx.y / (int)blocks_per_span_y;
	int sk = blockIdx.z / (int)blocks_per_span_z;

	// Which block it is within this span
	int bnx = blockIdx.x % (int)blocks_per_span_x;
	int bny = blockIdx.y % (int)blocks_per_span_y;
	int bnz = blockIdx.z % (int)blocks_per_span_z;

	int tx = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
	int BLOCK_SIZE = blockDim.x * blockDim.y * blockDim.z;

	// Each thread loads lattice values into the network
	for (int h=tx; h<64; h+=BLOCK_SIZE) {
		// Need to translate the h into span index.
		// So threadidx -> linearize -> for loop -> delinearize -> lattice index
		int skk = sk + h % 4;
		int sjj = sj + (h/4) % 4;
		int sii = si + h/16;

		lattice_piece[h] = lattice[ sii * ncpt_y * ncpt_z + sjj * ncpt_z + skk ];
	}

	__syncthreads();

	if ((blockIdx.x==0) && (blockIdx.y==0) && (blockIdx.z==0) &&
			(threadIdx.x==0) && (threadIdx.y==1) && (threadIdx.z==0)){
		for (int i=0; i<4; i++){
			for (int j=0; j<4; j++){
				for (int k=0; k<4; k++){
				}
			}
		}
	}

	// Determine if the current thread correspond to an actual pixel.
	// Example of starting pixel vs the Bspline points,
	// If this is 72.3, then the right closest center point is 72.5 then pixel index is 72.
	// If this is 72.6, then the right closest center point is 73.5 then pixel index is 73.
	// So it ends up being rounding around 0.5.

	// The start pixel that this span included
	int start_pixel_x = round(span_x * si);
	int start_pixel_y = round(span_y * sj);
	int start_pixel_z = round(span_z * sk);

	// The end pixel that this span included
	int end_pixel_x = round(span_x * (si+1))-1;
	int end_pixel_y = round(span_y * (sj+1))-1;
	int end_pixel_z = round(span_z * (sk+1))-1;

	// Pixel index
	int i = start_pixel_x + bnx* blockDim.x + threadIdx.x;
	int j = start_pixel_y + bny* blockDim.y + threadIdx.y;
	int k = start_pixel_z + bnz* blockDim.z + threadIdx.z;

	// If this thread actually correspond to an actual pixel, calculate the fitting value.
	if ((i <= end_pixel_x) && (j <= end_pixel_y) && (k <= end_pixel_z) &&
			(i < fx) && (j < fy) && (k < fz)){
		// The normalized local coordinates t for calculating bspline coefficients
		int bx = si;
		int by = sj;
		int bz = sk;

		float tx = ((i+0.5) - bx * span_x)/span_x;
		float ty = ((j+0.5) - by * span_y)/span_y;
		float tz = ((k+0.5) - bz * span_z)/span_z;

		// calculate B spline weight
		float wx[4];
		float wy[4];
		float wz[4];

		cubicBspline(tx, wx);
		cubicBspline(ty, wy);
		cubicBspline(tz, wz);

		// Accumulate the values
		float value = 0.;

#pragma unroll
		for (int ix=0; ix<4; ix++){
#pragma unroll
			for (int iy=0; iy<4; iy++){
#pragma unroll
				for (int iz=0; iz<4; iz++){
					value+=wx[ix]* wy[iy]* wz[iz] * lattice_piece[ix * 16 + iy * 4 + iz];
				}
			}
		}

		fitted[i*fy*fz + j * fz + k] = value;
	}
}


// This function accumulate WC2_phi and WC2 for fitting.
__global__ void accumulate_WC2phic_and_WC2(
		// Input
		const float * field,
		const float * mask,

		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,

		const unsigned int blocks_per_span_x, // How many block is there per span
		const unsigned int blocks_per_span_y,
		const unsigned int blocks_per_span_z,

		const float span_x, // How long each span is
		const float span_y,
		const float span_z,

		const unsigned int fx, // What is the input field size
		const unsigned int fy,
		const unsigned int fz,

		// Output
		float * wc2_phic,
		float * wc2
){
//	printf("Accumulate into wc2 = %f\n", wc2_local[h]);
//	printf("Running the accumulation\n");


	// Initialize the local wc2_phic and wc2 to 0
	__shared__ float wc2phic_local[64];
	__shared__ float wc2_local[64];

	// Start Spans index i,j,k. This is the index into the lattice's left top corner.
	int si = blockIdx.x / blocks_per_span_x;
	int sj = blockIdx.y / blocks_per_span_y;
	int sk = blockIdx.z / blocks_per_span_z;

	// Which block it is within this span
	int bnx = blockIdx.x % blocks_per_span_x;
	int bny = blockIdx.y % blocks_per_span_y;
	int bnz = blockIdx.z % blocks_per_span_z;

	int tx = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
	int BLOCK_SIZE = blockDim.x * blockDim.y * blockDim.z;

	// Each thread initializes the wc2phi and wc2
	for (int h=tx; h<64; h+=BLOCK_SIZE) {
		wc2phic_local[h] = 0.;
		wc2_local[h] = 0.;
	}

	__syncthreads();


	// Determine if the current thread correspond to an actual pixel.
	// Example of starting pixel vs the Bspline points,
	// If this is 72.3, then the right closest center point is 72.5 then pixel index is 72.
	// If this is 72.6, then the right closest center point is 73.5 then pixel index is 73.
	// So it ends up being rounding around 0.5.

	// The start pixel that this span included
	int start_pixel_x = round(span_x * si);
	int start_pixel_y = round(span_y * sj);
	int start_pixel_z = round(span_z * sk);

	// The end pixel that this span included
	int end_pixel_x = round(span_x * (si+1))-1;
	int end_pixel_y = round(span_y * (sj+1))-1;
	int end_pixel_z = round(span_z * (sk+1))-1;

	// Pixel index
	int i = start_pixel_x + bnx* blockDim.x + threadIdx.x;
	int j = start_pixel_y + bny* blockDim.y + threadIdx.y;
	int k = start_pixel_z + bnz* blockDim.z + threadIdx.z;

	// If this thread actually correspond to an actual pixel and the mask value is not zero, calculate the fitting value.
	// Note that the (mask[fidx]!=0) has to be put to the end to ensure the input fidx is a valid index due to short
	// circuit logic.

	int fidx= i*fy*fz + j*fz + k;
	if ((i <= end_pixel_x) && (j <= end_pixel_y) && (k <= end_pixel_z) &&
			(i < fx) && (j < fy) && (k < fz) && (mask[fidx]!=0) ){
		// The normalized local coordinates t for calculating bspline coefficients
		int bx = si;
		int by = sj;
		int bz = sk;

		float tx = ((i+0.5) - bx * span_x)/span_x;
		float ty = ((j+0.5) - by * span_y)/span_y;
		float tz = ((k+0.5) - bz * span_z)/span_z;

		// calculate B spline weight
		float wx[4];
		float wy[4];
		float wz[4];

		cubicBspline(tx, wx);
		cubicBspline(ty, wy);
		cubicBspline(tz, wz);

		// Calculate phi
		float wc_sum = 0.;
		float wc;
		float phi_c;

		// calculate wc_sum
#pragma unroll
		for (int ix=0; ix<4; ix++){
#pragma unroll
			for (int iy=0; iy<4; iy++){
#pragma unroll
				for (int iz=0; iz<4; iz++){
					wc = wx[ix]* wy[iy]* wz[iz];
					wc_sum += wc*wc;
				}
			}
		}

		// calculate wc2_phi and wc2
		float fv = field[fidx]; // field value
#pragma unroll
		for (int ix=0; ix<4; ix++){
#pragma unroll
			for (int iy=0; iy<4; iy++){
#pragma unroll
				for (int iz=0; iz<4; iz++){
					wc = wx[ix]* wy[iy]* wz[iz];
					phi_c = fv * wc / wc_sum;

					// Locally accumulate wc2phic and wc2. This eliminates the need for
					// each thread directly accumulate into global memory.
					atomicAdd(&wc2phic_local[ix*16 + iy*4 + iz], wc*wc*phi_c);
					atomicAdd(&wc2_local[ix*16 + iy*4 + iz], wc*wc);
				}
			}
		}
	}

	__syncthreads();

	// Each thread accumulates the local wc2phi and wc2 into the global memory.
	for (int h=tx; h<64; h+=BLOCK_SIZE) {
		// Need to translate the h into span index.
		// So threadidx -> linearize -> for loop -> delinearize -> lattice index
		int skk = sk + h % 4;
		int sjj = sj + (h/4) % 4;
		int sii = si + h/16;

		atomicAdd(&wc2_phic[sii * ncpt_y * ncpt_z + sjj * ncpt_z + skk], wc2phic_local[h]);
		atomicAdd(&wc2[sii * ncpt_y * ncpt_z + sjj * ncpt_z + skk], wc2_local[h]);
	}

}


// This sets the image below certain threshold to 0.
__global__ void lowthreshold(
		const float low_value,
		const unsigned int numberOfPixel,
		float * im,
		float * mask){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=numberOfPixel)
		return;

	if (im[i]<=low_value){
		im[i]=0;
		mask[i]=0;
	}
}


// The function that takes in the sharpening kernel and sharpen the image.
void sharpenImage(
		// Input
		float * d_before_sharpen, // image before sharpening. Does not set const here because the min/max reduction has to set the background value.
		const float * d_mask, // the input mask

		const unsigned int numberOfPixels, // number of pixel within the image
		const unsigned int NumberOfHistogramBins, // number of histogram bin
		const unsigned int paddedHistogramSize, // number of histogram bin after padded
		const unsigned int histogramOffset, // histogram offset
		const float WienerFilterNoise, // when building the wiener filter, the tiny constant at the denominator to prevent divide by 0
		const float BiasFieldFullWidthAtHalfMaximum, // gaussian filter width

		float * h_V, // real values
		float * h_F,
		float * h_U,
		float * h_numerator,
		float * h_denominator,
		float * h_E,

		std::vector<std::complex<float>> & h_Vf, // complex values
		std::vector<std::complex<float>> & h_Ff,
		std::vector<std::complex<float>> & h_Uf,
		std::vector<std::complex<float>> & h_numeratorf,
		std::vector<std::complex<float>> & h_denominatorf,

		fftwf_plan & pf_v, // FFTW plans
		fftwf_plan & pf_f,
		fftwf_plan & pf_numerator,
		fftwf_plan & pf_denominator,

		fftwf_plan & pb_u,
		fftwf_plan & pb_numerator,
		fftwf_plan & pb_denominator,

		// Output
		float * d_after_sharpen // Image after sharpening
){
	// Define variables
	float binMaximum, binMinimum;
	dim3 threads_1d(1024, 1, 1);
	dim3 blocks_1d( (numberOfPixels+1023)/1024, 1, 1);

	// buffer needed for the min/max operation
	float * d_buffer;
	float * d_histogram;
	float * d_E;

	checkCudaErrors(cudaMalloc((void **)&d_buffer, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_histogram, NumberOfHistogramBins*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_E, NumberOfHistogramBins*sizeof(float)));
	checkCudaErrors(cudaMemset(d_histogram, 0, NumberOfHistogramBins*sizeof(float))); // Init histogram to 0

	// Within the range defined by the mask, get the min/max of the before_sharpen image.
	set_background_kernel<<< blocks_1d, threads_1d >>>(d_mask, 10000000000000.0, numberOfPixels, d_before_sharpen);
	binMinimum = Reducer::reduce_min_wrapper(numberOfPixels, d_before_sharpen, d_buffer);
	set_background_kernel<<< blocks_1d, threads_1d >>>(d_mask, -10000000000000.0, numberOfPixels, d_before_sharpen);
	binMaximum = Reducer::reduce_max_wrapper(numberOfPixels, d_before_sharpen, d_buffer);
	set_background_kernel<<< blocks_1d, threads_1d >>>(d_mask, 0.0, numberOfPixels, d_before_sharpen);

	// Calculate how large is each bin
	float histogramSlope = ( binMaximum - binMinimum )/( (float)NumberOfHistogramBins - 1. );

	// Create the intensity profile (within the masked region, if applicable)
	// using a triangular parzen windowing scheme (bullshit parzen windowing. Simply a
	// histogram considering fractional count).
	histogramGPU_kernel<<< blocks_1d, threads_1d, NumberOfHistogramBins*sizeof(float) >>>(
			// Input
			d_before_sharpen,
			d_mask,
			binMinimum,
			histogramSlope,
			numberOfPixels,
			NumberOfHistogramBins,
			// Output
			d_histogram
	);

	checkCudaErrors(cudaMemcpy(&(h_V[histogramOffset]), d_histogram, NumberOfHistogramBins*sizeof(float), cudaMemcpyDeviceToHost));

	// confirmed against the python example that the histogram is correct at this point.
	// Calculate the fft on the histogram, h_V -> h_Vf
	fftwf_execute(pf_v);

	// create a equal-size-to-histogram gaussian filter, fft it.
	// Since the histogram size here is small (for a 200 bin at most 512 is needed. Use the CPU
	// implementation of the fftw instead of cuFFT).

	// Create the Gaussian filter.
	float scaledFWHM = BiasFieldFullWidthAtHalfMaximum / histogramSlope;
	float expFactor = 4.0 * log( 2.0 ) / (scaledFWHM * scaledFWHM);
	float scaleFactor = 2.0 * sqrt( log( 2.0 )/ M_PI ) / scaledFWHM;

	// These parameters matches the python implementation
	//	printf("GPU: Histogram slope/scaledFWHM/expFactor/scaleFactor: (%f, %f, %f, %f)\n", histogramSlope, scaledFWHM, expFactor, scaleFactor);

	h_F[0] = scaleFactor;
	unsigned int halfSize = (unsigned int)(0.5 * paddedHistogramSize);
	for( unsigned int i = 1; i <= halfSize; i++ ){
		h_F[i] = h_F[paddedHistogramSize - i] =
				scaleFactor * exp( -expFactor*i*i );
	}

	if( paddedHistogramSize % 2 == 0 ){
		h_F[halfSize] = scaleFactor * exp(
				-0.25 * paddedHistogramSize*paddedHistogramSize*expFactor );
	}

	// FFT the gaussian kernel, h_F -> h_Ff
	fftwf_execute(pf_f);

	// change the Ff to Gf, multiply that with Vf and output to Uf
	for( unsigned int i = 0; i < (paddedHistogramSize/2+1); i++ ){
		// Make the Wiener deconvolution kernel and multiply with the signal.
		std::complex<float> c = conj(h_Ff[i]);
		std::complex<float> Gf = c / ( c * h_Ff[i] + WienerFilterNoise );
		h_Uf[i] = h_Vf[i] * Gf.real() / (float)paddedHistogramSize ;
	}


	// iFFT the deconvolved histogram and set clip negative real value to 0
	// h_Uf -> h_U. Note that this compared to the python implementation does not
	// do the normalization. So here need to divide the paddedHistogram to do the
	// normalization.
	fftwf_execute(pb_u);
	for( unsigned int i = 0; i < paddedHistogramSize; i++ ){
		h_U[i] = max( h_U[i], 0.0 );
	}

	// The numerator is histBin * U, where U = deconv(V)
	for( unsigned int i = 0; i < paddedHistogramSize; i++ ){
		h_numerator[i] = ( (float)binMinimum + ((float)i - histogramOffset) * histogramSlope ) * h_U[i];
	}

	// This is simply using the gaussian kernel h_Ff to smooth out the numerator.
	// smooth(hisBin * h_U)
	fftwf_execute(pf_numerator);

	for( unsigned int i = 0; i < (paddedHistogramSize/2+1); i++){
		h_numeratorf[i] *= h_Ff[i];
	}
	fftwf_execute(pb_numerator);


	// h_U -> h_denominatorf. This use directly h_U as input.
	// smooth(h_U)
	// Again this simply smooth the denominator with the gaussian kernel h_Ff.
	fftwf_execute(pf_denominator);
	for( unsigned int i = 0; i < (paddedHistogramSize/2+1); i++ ){
		h_denominatorf[i]*= h_Ff[i];
	}
	fftwf_execute(pb_denominator); // h_denominatorf -> h_denominator

	// The divide part.  smooth(hisBin * h_U)/smooth(h_U)
	// Build a map of image from old histogram to new histogram
	// This skip the amount of histogramOffset.
	for( unsigned int i = 0; i < NumberOfHistogramBins; i++ ){
		if( h_denominator[i+histogramOffset] != 0.0 ){
			h_E[i] = h_numerator[i+histogramOffset] / h_denominator[i+histogramOffset];
		} else {
			h_E[i] = 0.0;
		}
	}

	// Map the pixel value using the map.
	checkCudaErrors(cudaMemcpy(d_E, h_E, NumberOfHistogramBins*sizeof(float), cudaMemcpyHostToDevice));

	histogramMapping_kernel<<< blocks_1d, threads_1d >>>(
			// Input
			d_before_sharpen,
			d_mask,
			d_E,
			binMinimum,
			histogramSlope,
			numberOfPixels, // num pixel in image
			NumberOfHistogramBins,
			// Output
			d_after_sharpen // Image after sharpening
	);

	// Clean up
	checkCudaErrors(cudaFree(d_buffer));
	checkCudaErrors(cudaFree(d_histogram));
	checkCudaErrors(cudaFree(d_E));
}

// This function serves to test the log image and sharpenImage.
void testsharpenImage(
		// Input
		const float * h_before_sharpen, // image before sharpening. Does not set const here because the min/max reduction has to set the background value.
		const float * h_mask,
		const unsigned int numberOfPixels,
		N4Param & param,
		N4Data & data,
		// Output
		float * h_before_sharpen_log,
		float * h_after_sharpen,
		float & binMin_out,
		float & binMax_out){
	float * d_before_sharpen;
	float * d_before_sharpen_log;
	float * d_mask;
	float * d_after_sharpen;

	checkCudaErrors(cudaMalloc((void **)&d_before_sharpen, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_mask, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_before_sharpen_log, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_after_sharpen, numberOfPixels*sizeof(float)));

	// Copy the image and mask to gpu
	checkCudaErrors(cudaMemcpy(d_before_sharpen, h_before_sharpen, numberOfPixels*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mask, h_mask, numberOfPixels*sizeof(float), cudaMemcpyHostToDevice));

	dim3 threads_1d(512, 1, 1);
	dim3 blocks_1d((numberOfPixels+511)/512, 1, 1);

	// Set the mask = 0 and im = 0 at the point where im<low_value
	lowthreshold<<< blocks_1d, threads_1d >>>(
			param.low_value,
			numberOfPixels,
			d_before_sharpen,
			d_mask);


	log_kernel<<< blocks_1d, threads_1d >>>(
			// Input
			d_before_sharpen,
			d_mask,
			numberOfPixels,
			// Output
			d_before_sharpen_log);

	// Output the log image for checking
	checkCudaErrors(cudaMemcpy(h_before_sharpen_log, d_before_sharpen_log, numberOfPixels*sizeof(float), cudaMemcpyDeviceToHost));


	// Run the function
	sharpenImage(
			// Input
			d_before_sharpen_log, // image before sharpening. Does not set const here because the min/max reduction has to set the background value.
			d_mask, // the input mask

			numberOfPixels, // number of pixel within the image
			param.NumberOfHistogramBins, // number of histogram bin
			param.paddedHistogramSize, // number of histogram bin after padded
			param.histogramOffset, // histogram offset
			param.WienerFilterNoise, // when building the wiener filter, the tiny constant at the denominator to prevent divide by 0
			param.BiasFieldFullWidthAtHalfMaximum, // gaussian filter width

			data.h_V, // real values
			data.h_F,
			data.h_U,
			data.h_numerator,
			data.h_denominator,
			data.h_E,

			data.h_Vf, // complex values
			data.h_Ff,
			data.h_Uf,
			data.h_numeratorf,
			data.h_denominatorf,

			data.pf_v, // FFTW plans
			data.pf_f,
			data.pf_numerator,
			data.pf_denominator,

			data.pb_u,
			data.pb_numerator,
			data.pb_denominator,

			// Output
			d_after_sharpen // Image after sharpening
	);

	checkCudaErrors(cudaMemcpy(h_after_sharpen, d_after_sharpen, numberOfPixels*sizeof(float), cudaMemcpyDeviceToHost));

	// Clean up
	checkCudaErrors(cudaFree(d_before_sharpen));
	checkCudaErrors(cudaFree(d_before_sharpen_log));
	checkCudaErrors(cudaFree(d_mask));
	checkCudaErrors(cudaFree(d_after_sharpen));
}


void calculateConvergenceMeasurement(
		// Input
		const float * d_field1,
		const float * d_field2,
		const float * d_mask,
		const unsigned int numberOfPixels,
		const float numberOfForeground,
		// output
		float & convergence){
	dim3 threads_1d(512, 1, 1);
	dim3 blocks_1d( (numberOfPixels+511)/512, 1, 1);

	// calculate exp(field1 - field2)
	float * d_pixel;
	float * d_buffer;
	checkCudaErrors(cudaMalloc((void **)&d_pixel, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_buffer, numberOfPixels*sizeof(float)));
	subtract_and_exp_kernel<<< blocks_1d, threads_1d >>>(
			// Input
			d_field1,
			d_field2,
			// Output
			numberOfPixels,
			d_pixel);

	// mean and std of the mask region
	set_background_kernel<<< blocks_1d, threads_1d >>>(d_mask, 0, numberOfPixels, d_pixel);
	float mu = Reducer::reduce_sum_wrapper(numberOfPixels, d_pixel, d_buffer);
	mu /= numberOfForeground;

	// calculate (X - mean)^2
	subtract_mean_and_sqr_kernel<<< blocks_1d, threads_1d >>>(
			d_pixel,
			mu,
			numberOfPixels);

	// Divide sum by N-1 and sqrt
	set_background_kernel<<< blocks_1d, threads_1d >>>(d_mask, 0, numberOfPixels, d_pixel);
	float sigma = Reducer::reduce_sum_wrapper(numberOfPixels, d_pixel, d_buffer);
	sigma /= (numberOfForeground-1);
	sigma = sqrt(sigma);

	// output
	convergence = sigma/mu;

	checkCudaErrors(cudaFree(d_pixel));
	checkCudaErrors(cudaFree(d_buffer));
}


// This function does the upsampling of the lattice on the CPU
// This function assume input to be (nx, ny, nz) number of control points
// and output to be (nx-3)*2 +3 number of control points.
// As suggested by the Bspline paper, this always upsample a lattice of size
// n+3 to 2n+3
void upsampleLattice3D_cpu(
		// Input
		const float * h_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		// Output
		float * h_lattice_upsample){

	float bw[2][3] = {
			{1./8, 6./8, 1./8},
			{0, 1./2, 1./2}
	};

	unsigned int nx = ncpt_x - 3;
	unsigned int ny = ncpt_y - 3;
	unsigned int nz = ncpt_z - 3;

	// Size of the upsample lattice
	unsigned int n2x = 2*nx + 3;
	unsigned int n2y = 2*ny + 3;
	unsigned int n2z = 2*nz + 3;

	float lattice_piece[27];

	// Loop through each point within the low resolution lattice
	for (int i=-1; i<nx+2; i++){
		for (int j=-1; j<ny+2; j++){
			for (int k=-1; k<nz+2; k++){
				// Each takes care the neighboring 8 points on the upsampled points
				// as a linear combination of the neighboring 3x3x3 piece

				// gather the lattice piece
				// i/j/kl: (i-1)~(i+1)
				for (int il = i-1; il<i+2; il++){
					for (int jl = j-1; jl<j+2; jl++){
						for (int kl = k-1; kl<k+2; kl++){
							// i/j/klp: 0~2
							int ilp = il-i+1;
							int jlp = jl-j+1;
							int klp = kl-k+1;


							// clip the dimension within the lattice range
							int ilc = min(max(il, -1), nx+1) +1;
							int jlc = min(max(jl, -1), ny+1) +1;
							int klc = min(max(kl, -1), nz+1) +1;

							lattice_piece[ilp*9 + jlp * 3 + klp] =
									h_lattice[ ilc * ncpt_y * ncpt_z + jlc * ncpt_z + klc];
						}
					}
				}

				// Do the accumulation for the 8 elements
				for (int i2=0; i2<2; i2++){
					for (int j2=0; j2<2; j2++){
						for (int k2=0; k2<2; k2++){
							int i2l = 2*i + i2;
							int j2l = 2*j + j2;
							int k2l = 2*k + k2;

							// if statement to
							if ((i2l>=-1) && (i2l<=2*nx+1) &&
									(j2l>=-1) && (j2l<=2*ny+1) &&
									(k2l>=-1) && (k2l<=2*nz+1)
							){
								// Each elements are sum of 27 terms.
								float ls = 0.;
								for (int bx=0; bx<3; bx++){
									for (int by=0; by<3; by++){
										for (int bz=0; bz<3; bz++){
											ls+=bw[i2][bx]*bw[j2][by]*bw[k2][bz] * lattice_piece[bx * 9 + by * 3 + bz];
										}
									}
								}

								// Put the summed element into the array
								h_lattice_upsample[(i2l+1) * n2y * n2z +
								                   (j2l+1) * n2z +
								                   k2l+1] = ls;
							}

						}
					}
				}
			}
		}
	}
}

void upsampleLattice3D_gpu(
		// Input
		const float * d_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		// Output
		float * d_lattice_upsample){
	unsigned int nx = ncpt_x - 3;
	unsigned int ny = ncpt_y - 3;
	unsigned int nz = ncpt_z - 3;

	// Size of the upsample lattice
	unsigned int n2x = 2*nx + 3;
	unsigned int n2y = 2*ny + 3;
	unsigned int n2z = 2*nz + 3;

	dim3 blocks_3d(ncpt_x, ncpt_y, ncpt_z);
	dim3 threads_3d(2, 2, 2);

	upsample_lattice_kernel_3D<<< blocks_3d, threads_3d >>>(
			// Input
			ncpt_x, // number of control points
			ncpt_y,
			ncpt_z,
			nx, // ncpt - 3, the base number
			ny,
			nz,
			n2x, // number of control points on next level
			n2y,
			n2z,
			d_lattice,
			// Output
			d_lattice_upsample
	);

}


void testUpsampleLattice3D_gpu(
		// Input
		const float * h_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		// Output
		float * h_lattice_upsample){
	unsigned int numberOfLattice = ncpt_x * ncpt_y * ncpt_z;

	// number of control points of the upsampled one.
	unsigned int ncpt_x_n = (ncpt_x - 3) * 2 + 3;
	unsigned int ncpt_y_n = (ncpt_y - 3) * 2 + 3;
	unsigned int ncpt_z_n = (ncpt_z - 3) * 2 + 3;
	unsigned int numberOfLattice_n = ncpt_x_n * ncpt_y_n * ncpt_z_n;

	float * d_lattice;
	float * d_lattice_upsample;

	checkCudaErrors(cudaMalloc((void **)&d_lattice, numberOfLattice*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_lattice_upsample, numberOfLattice_n*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_lattice, h_lattice, numberOfLattice*sizeof(float), cudaMemcpyHostToDevice));

	upsampleLattice3D_gpu(
			// Input
			d_lattice,
			ncpt_x, // number of control points
			ncpt_y,
			ncpt_z,
			// Output
			d_lattice_upsample);

	checkCudaErrors(cudaMemcpy(h_lattice_upsample, d_lattice_upsample, numberOfLattice_n*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_lattice));
	checkCudaErrors(cudaFree(d_lattice_upsample));
}

// This function evaluates Bspline on the field.
// To reduce total amount of global read, this again applies the same memory reading scheme
// as the one on on the fitting function.
void EvaluateBspline3D(
		// Input
		const float * d_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		const unsigned int fx, // size of the output field
		const unsigned int fy,
		const unsigned int fz,
		// Output
		float * d_fitted){
	unsigned int numberOfSpansX = ncpt_x - 3;
	unsigned int numberOfSpansY = ncpt_y - 3;
	unsigned int numberOfSpansZ = ncpt_z - 3;

	float span_x = (float)fx / numberOfSpansX;
	float span_y = (float)fy / numberOfSpansY;
	float span_z = (float)fz / numberOfSpansZ;

	// If the spans is smaller than 8, then use the spans size for each block.
	// Note that the threadblock size 8x8x8 on GTX 1080 with driver 418 doesn't work.
	// The kernel simply silently not launch. Switch to 6x6x6 makes it work.

//	int thread_x = (int)ceil(fmin(span_x, 8.f));
//	int thread_y = (int)ceil(fmin(span_y, 8.f));
//	int thread_z = (int)ceil(fmin(span_z, 8.f));

	int thread_x = (int)ceil(fmin(span_x, 6.f));
	int thread_y = (int)ceil(fmin(span_y, 6.f));
	int thread_z = (int)ceil(fmin(span_z, 6.f));

	int blocks_per_span_x = (int)ceil(span_x / (float)thread_x);
	int blocks_per_span_y = (int)ceil(span_y / (float)thread_y);
	int blocks_per_span_z = (int)ceil(span_z / (float)thread_z);

	int num_block_x = numberOfSpansX * blocks_per_span_x;
	int num_block_y = numberOfSpansY * blocks_per_span_y;
	int num_block_z = numberOfSpansZ * blocks_per_span_z;

	dim3 threads_3d(thread_x, thread_y, thread_z);
	dim3 blocks_3d( num_block_x , num_block_y, num_block_z);


	evaluate_bspline_kernel_3D<<< blocks_3d, threads_3d >>>(
			// Output
			d_lattice,
			ncpt_x, // number of control points
			ncpt_y,
			ncpt_z,
			blocks_per_span_x, // How many block is there per span
			blocks_per_span_y,
			blocks_per_span_z,
			span_x, // How long each span is
			span_y,
			span_z,
			fx, // What is the output field size
			fy,
			fz,
			// Output
			d_fitted
	);
}


void testEvaluateBspline3D(// Input
		const float * h_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		const unsigned int fx, // size of the output field
		const unsigned int fy,
		const unsigned int fz,
		// Output
		float * h_fitted){
	float * d_lattice;
	float * d_fitted;

	unsigned int numberOfLattice = ncpt_x * ncpt_y * ncpt_z;
	unsigned int numberOfPixels = fx * fy * fz;

	checkCudaErrors(cudaMalloc((void **)&d_lattice, numberOfLattice*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_fitted, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_lattice, h_lattice, numberOfLattice*sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(d_fitted, 0, numberOfLattice*sizeof(float)));

	EvaluateBspline3D(
			// Input
			d_lattice,
			ncpt_x, // number of control points
			ncpt_y,
			ncpt_z,
			fx, // size of the output field
			fy,
			fz,
			// Output
			d_fitted);


	checkCudaErrors(cudaMemcpy(h_fitted, d_fitted, numberOfPixels*sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_lattice));
	checkCudaErrors(cudaFree(d_fitted));
}


// This given a field and a mask, fits a Bspline to the field according to basic
// algorithm from paper "Scattered Data Interpolation with Multilevel B-Splines".
// The block/thread/shared memory layout is exactly the same as EvaluateBspline3D
// except read/write were reversed.
void FitBspline3D(
		// Input
		const float * d_field,
		const float * d_mask,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		const unsigned int fx, // size of the output field
		const unsigned int fy,
		const unsigned int fz,
		// Output
		float * d_lattice
){
	// Initialize the numeratora and the denominator
	unsigned int n_lattice = ncpt_x * ncpt_y * ncpt_z;
	float * d_wc2_phic;
	float * d_wc2;
	float * d_buffer;
	checkCudaErrors(cudaMalloc((void **)&d_wc2_phic, n_lattice*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_wc2, n_lattice*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_buffer, n_lattice*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wc2_phic, 0, n_lattice*sizeof(float)));
	checkCudaErrors(cudaMemset(d_wc2, 0, n_lattice*sizeof(float)));

	// Same block/thread/shared memory layout as the EvaluateBspline3D.
	unsigned int numberOfSpansX = ncpt_x - 3;
	unsigned int numberOfSpansY = ncpt_y - 3;
	unsigned int numberOfSpansZ = ncpt_z - 3;

	float span_x = 1.0*fx / numberOfSpansX;
	float span_y = 1.0*fy / numberOfSpansY;
	float span_z = 1.0*fz / numberOfSpansZ;

	// If the spans is smaller than 8, then use the spans size for each block.
//	int thread_x = ceil(fmin(span_x, 8.f));
//	int thread_y = ceil(fmin(span_y, 8.f));
//	int thread_z = ceil(fmin(span_z, 8.f));

	int thread_x = ceil(fmin(span_x, 6.f));
	int thread_y = ceil(fmin(span_y, 6.f));
	int thread_z = ceil(fmin(span_z, 6.f));

	int blocks_per_span_x = ceil(span_x / thread_x);
	int blocks_per_span_y = ceil(span_y / thread_y);
	int blocks_per_span_z = ceil(span_z / thread_z);

	int num_block_x = numberOfSpansX * blocks_per_span_x;
	int num_block_y = numberOfSpansY * blocks_per_span_y;
	int num_block_z = numberOfSpansZ * blocks_per_span_z;

	dim3 threads_3d(thread_x, thread_y, thread_z);
	dim3 blocks_3d(num_block_x , num_block_y, num_block_z);

	// accumulate wc2_phi and wc2
	accumulate_WC2phic_and_WC2<<< blocks_3d, threads_3d >>>(
			// Input
			d_field,
			d_mask,
			ncpt_x, // number of control points
			ncpt_y,
			ncpt_z,
			blocks_per_span_x, // How many block is there per span
			blocks_per_span_y,
			blocks_per_span_z,
			span_x, // How long each span is
			span_y,
			span_z,
			fx, // What is the input field size
			fy,
			fz,
			// Output
			d_wc2_phic,
			d_wc2
	);

	// lattice = wc2_phic / d_wc2
	dim3 threads_1d(512, 1, 1);
	dim3 blocks_1d((n_lattice+511)/512, 1, 1);
	divide_kernel<<< blocks_1d, threads_1d >>>(
			// Input
			d_wc2_phic,
			d_wc2,
			n_lattice,
			// Output
			d_lattice);

	// clean up
	checkCudaErrors(cudaFree(d_wc2_phic));
	checkCudaErrors(cudaFree(d_wc2));
	checkCudaErrors(cudaFree(d_buffer));
}


void testFitBspline3D(
		// Input
		const float * h_field,
		const float * h_mask,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		const unsigned int fx, // size of the output field
		const unsigned int fy,
		const unsigned int fz,
		// Output
		float * h_lattice
){
	float * d_field;
	float * d_mask;
	float * d_lattice;

	unsigned int numberOfLattice = ncpt_x * ncpt_y * ncpt_z;
	unsigned int numberOfPixels = fx * fy * fz;

	checkCudaErrors(cudaMalloc((void **)&d_field, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_mask, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_lattice, numberOfLattice*sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_field, h_field, numberOfPixels*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mask, h_mask, numberOfPixels*sizeof(float), cudaMemcpyHostToDevice));

	FitBspline3D(
			// Input
			d_field,
			d_mask,
			ncpt_x, // number of control points
			ncpt_y,
			ncpt_z,
			fx, // size of the output field
			fy,
			fz,
			// Output
			d_lattice
	);

	checkCudaErrors(cudaMemcpy(h_lattice, d_lattice, numberOfLattice*sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_field));
	checkCudaErrors(cudaFree(d_mask));
	checkCudaErrors(cudaFree(d_lattice));
}


// The overall N4 function
void N4(
		// Input
		float * d_im, // The input image
		float * d_mask, // The mask. Right now only binary mask are supported.

		const unsigned int fx, // What is the input field size
		const unsigned int fy,
		const unsigned int fz,

		N4Param & param,
		N4Data & data, // Some additional data structure needed to run the function,
		// mostly are fft stuff needed for fftw.

		// Output
		float * d_im_normalized, // The image after normalization
		float * d_lattice, // The resulting lattice representing the bias field.
		// Size of this should be equal to 2^(number of level-1) * (initial_point-3) + 3

		// Optional. Also output the bias field
		float * d_biasField
){
	unsigned int numberOfPixels = fx * fy * fz;

	// Define variables
	unsigned int ncpt_x = param.NumberOfControlPoints_x; // number of control points
	unsigned int ncpt_y = param.NumberOfControlPoints_y;
	unsigned int ncpt_z = param.NumberOfControlPoints_z;
	unsigned int numberOfLattice = ncpt_x * ncpt_y * ncpt_z;

	unsigned int ncpt_x_n; // number of control points on next level
	unsigned int ncpt_y_n;
	unsigned int ncpt_z_n;
	unsigned int numberOfLattice_n;

	float CurrentConvergenceMeasurement;
	unsigned int elapsedIterations;

	dim3 threads_1d(512, 1, 1);
	dim3 blocks_1d((numberOfPixels+511)/512, 1, 1);

	// Set the mask = 0 and im = 0 at the point where im<low_value
	lowthreshold<<< blocks_1d, threads_1d >>>(
			param.low_value,
			numberOfPixels,
			d_im,
			d_mask);


	float * h_buffer = new float[numberOfPixels];

	float spacing[] = {1,1,1};
	int size[] = {(int)fz, (int)fy, (int)fx};

	float * d_im_log;
	float * d_logUncorrectedImage;
	float * d_logSharpenedImage;
	float * d_residualBiasField;

	float * d_logBiasField;
	float * d_newLogBiasField;
	float * d_buffer;
	float * d_temp;

	float * d_lattice_c_residual; // current level lattice fitted to the residual
	float * d_lattice_c; // current level lattice accumulated
	float * d_lattice_n; // next level lattice

	// Assign space
	checkCudaErrors(cudaMalloc((void **)&d_im_log, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_logUncorrectedImage, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_logSharpenedImage, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_residualBiasField, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_logBiasField, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_newLogBiasField, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_buffer, numberOfPixels*sizeof(float)));

	checkCudaErrors(cudaMemset(d_im_log, 0, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMemset(d_logBiasField, 0, numberOfPixels*sizeof(float))); // set the logBiasField initially to 0.
	checkCudaErrors(cudaMemset(d_buffer, 0, numberOfPixels*sizeof(float))); // set the logBiasField initially to 0.

	log_kernel<<< blocks_1d, threads_1d >>>(
			// Input
			d_im,
			d_mask,
			numberOfPixels,
			// Output
			d_im_log);

	float numberOfForeground = Reducer::reduce_sum_wrapper(numberOfPixels, d_mask, d_buffer);

	checkCudaErrors(cudaMemcpy(d_logUncorrectedImage, d_im_log, numberOfPixels*sizeof(float), cudaMemcpyDeviceToDevice));


	// assign lattice space for the first level
	checkCudaErrors(cudaMalloc((void **)&d_lattice_c, numberOfLattice*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_lattice_c_residual, numberOfLattice*sizeof(float)));
	checkCudaErrors(cudaMemset(d_lattice_c, 0, numberOfLattice*sizeof(float))); // Init the initial lattice to 0.


	for (int currentLevel = 0; currentLevel<param.NumberOfFittingLevels; currentLevel++){
		// Calculate number of control points at this level
		// Currently the fitting resolution of the 3 axes are tied together. Later maybe
		// we can make it separate. Anyway.
		CurrentConvergenceMeasurement = 10000000000.0;
		elapsedIterations = 0;

		dim3 threads_1d_lattice(512, 1, 1);
		dim3 blocks_1d_lattice((numberOfLattice+511)/512, 1, 1);

		while ((CurrentConvergenceMeasurement > param.ConvergenceThreshold) &&
				(elapsedIterations<param.MaximumNumberOfIterations)){

			printf("Level %d, iter %d\n", currentLevel, elapsedIterations);
			elapsedIterations++;



			// Sharpen the image
			sharpenImage(
					// Input
					d_logUncorrectedImage, // image before sharpening. Does not set const here because the min/max reduction has to set the background value.
					d_mask, // the input mask

					numberOfPixels, // number of pixel within the image
					param.NumberOfHistogramBins, // number of histogram bin
					param.paddedHistogramSize, // number of histogram bin after padded
					param.histogramOffset, // histogram offset
					param.WienerFilterNoise, // when building the wiener filter, the tiny constant at the denominator to prevent divide by 0
					param.BiasFieldFullWidthAtHalfMaximum, // gaussian filter width

					data.h_V, // real values
					data.h_F,
					data.h_U,
					data.h_numerator,
					data.h_denominator,
					data.h_E,

					data.h_Vf, // complex values
					data.h_Ff,
					data.h_Uf,
					data.h_numeratorf,
					data.h_denominatorf,

					data.pf_v, // FFTW plans
					data.pf_f,
					data.pf_numerator,
					data.pf_denominator,

					data.pb_u,
					data.pb_numerator,
					data.pb_denominator,

					// Output
					d_logSharpenedImage // Image after sharpening
			);


			// Benchmark code- Start
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start); // Benchmark code


			subtract<<< blocks_1d, threads_1d >>>(
					// Input
					d_logUncorrectedImage,
					d_logSharpenedImage,
					numberOfPixels,
					// Output
					d_residualBiasField);

			// Fit a new Bspline lattice to the current residual field
			// clear the
			FitBspline3D(
					// Input
					d_residualBiasField,
					d_mask,
					ncpt_x, // number of control points
					ncpt_y,
					ncpt_z,
					fx, // size of the output field
					fy,
					fz,
					// Output
					d_lattice_c_residual
			);

			// Accumulate the residual lattice in overall lattice
			sum_inplace_kernel<<< blocks_1d_lattice, threads_1d_lattice >>>(
					// Input
					d_lattice_c, // output
					d_lattice_c_residual,
					numberOfLattice);

			// calculate the new bias field
			EvaluateBspline3D(
					// Input
					d_lattice_c,
					ncpt_x, // number of control points
					ncpt_y,
					ncpt_z,
					fx, // size of the output field
					fy,
					fz,
					// Output
					d_newLogBiasField);

			// Benchmark code - End
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Smooth image takes %f ms\n.", milliseconds);


			// calculate convergence
			calculateConvergenceMeasurement(
					// Input
					d_logBiasField,
					d_newLogBiasField,
					d_mask,
					numberOfPixels,
					numberOfForeground,
					// output
					param.ConvergenceThreshold);

			// Update the logBiasField. Use pointer swap here to save time on copying data.
			d_temp = d_logBiasField;
			d_logBiasField = d_newLogBiasField;
			d_newLogBiasField = d_temp;

			// Update logUncorrectedImage
			subtract<<< blocks_1d, threads_1d >>>(
					// Input
					d_im_log,
					d_logBiasField,
					numberOfPixels,
					// Output
					d_logUncorrectedImage);
		}

		// Upsample the lattice if not the last level
		if (currentLevel!=param.NumberOfFittingLevels-1){
			ncpt_x_n = (ncpt_x - 3) * 2 + 3; // number of control points. Since we don't how many points initially the user input so we iterate like this.
			ncpt_y_n = (ncpt_y - 3) * 2 + 3;
			ncpt_z_n = (ncpt_z - 3) * 2 + 3;
			numberOfLattice_n = ncpt_x_n * ncpt_y_n * ncpt_z_n;

			// Assign memory for the next level lattice
			checkCudaErrors(cudaMalloc((void **)&d_lattice_n, numberOfLattice_n*sizeof(float)));

			// Upsample
			upsampleLattice3D_gpu(
					// Input
					d_lattice_c,
					ncpt_x, // number of control points
					ncpt_y,
					ncpt_z,
					// Output
					d_lattice_n);

			// Free up the current lattice and pointer swap to next
			checkCudaErrors(cudaFree(d_lattice_c));
			checkCudaErrors(cudaFree(d_lattice_c_residual));
			d_lattice_c = d_lattice_n;
			// lattice_residual for next level
			checkCudaErrors(cudaMalloc((void **)&d_lattice_c_residual, numberOfLattice_n*sizeof(float)));

			// Update number of control points to next level
			ncpt_x = ncpt_x_n;
			ncpt_y = ncpt_y_n;
			ncpt_z = ncpt_z_n;
			numberOfLattice = numberOfLattice_n;
		}
	}



	// Final normalization

	// calculate the final field
	EvaluateBspline3D(
			// Input
			d_lattice_c,
			ncpt_x, // number of control points
			ncpt_y,
			ncpt_z,
			fx, // size of the output field
			fy,
			fz,
			// Output
			d_logBiasField);


	exp_and_divide_kernel<<< blocks_1d, threads_1d >>>(
			// Input
			d_logBiasField,
			d_im,
			numberOfPixels,
			// output
			d_im_normalized);

	// Output the bias field
	if (d_biasField != NULL){
		exp_kernel<<< blocks_1d, threads_1d >>>(
				// Input
				d_logBiasField,
				numberOfPixels,
				// output
				d_biasField);
	}

	delete [] h_buffer;

	// Save the lattice to output
	checkCudaErrors(cudaMemcpy(d_lattice, d_lattice_c, numberOfLattice*sizeof(float), cudaMemcpyDeviceToDevice));

	// clean up
	checkCudaErrors(cudaFree(d_im_log));
	checkCudaErrors(cudaFree(d_logUncorrectedImage));
	checkCudaErrors(cudaFree(d_logSharpenedImage));
	checkCudaErrors(cudaFree(d_residualBiasField));
	checkCudaErrors(cudaFree(d_logBiasField));
	checkCudaErrors(cudaFree(d_newLogBiasField));
	checkCudaErrors(cudaFree(d_buffer));

	checkCudaErrors(cudaFree(d_lattice_c));
	checkCudaErrors(cudaFree(d_lattice_c_residual));
}



void testN4(
		// Input
		float * h_im, // The input image
		float * h_mask, // The mask. Right now only binary mask are supported.

		const unsigned int fx, // What is the input field size
		const unsigned int fy,
		const unsigned int fz,

		N4Param & param,
		N4Data & data, // Some additional data structure needed to run the function,
		// mostly are fft stuff needed for fftw.

		// Output
		float * h_im_normalized, // The image after normalization
		float * h_lattice, // The resulting lattice representing the bias field.
		// Size of this should be equal to 2^(number of level-1) * (initial_point-3) + 3

		// Optional. Also output the bias field
		float * h_biasField
){
	float * d_im;
	float * d_im_normalized;
	float * d_mask;
	float * d_lattice;
	float * d_biasField;

	unsigned int ncpt_x = param.NumberOfControlPoints_x;
	unsigned int ncpt_y = param.NumberOfControlPoints_y;
	unsigned int ncpt_z = param.NumberOfControlPoints_z;

	for (unsigned int i=0; i< param.NumberOfFittingLevels-1; i++){
		ncpt_x=(ncpt_x-3)*2+3;
		ncpt_y=(ncpt_y-3)*2+3;
		ncpt_z=(ncpt_z-3)*2+3;
	}

	unsigned int numberOfLattice = ncpt_x * ncpt_y * ncpt_z;
	unsigned int numberOfPixels = fx * fy * fz;

	printf("Test lattice size: %d\n", numberOfLattice);

	checkCudaErrors(cudaMalloc((void **)&d_im, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_mask, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_im_normalized, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_lattice, numberOfLattice*sizeof(float)));

	// Optional. Also output the bias field
	if (h_biasField!=NULL){
		checkCudaErrors(cudaMalloc((void **)&d_biasField, numberOfPixels*sizeof(float)));
	}else{
		d_biasField = NULL;
	}

	checkCudaErrors(cudaMemcpy(d_im, h_im, numberOfPixels*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mask, h_mask, numberOfPixels*sizeof(float), cudaMemcpyHostToDevice));

	N4(
			// Input
			d_im, // The input image
			d_mask, // The mask. Right now only binary mask are supported.

			fx, // What is the input field size
			fy,
			fz,

			param,
			data, // Some additional data structure needed to run the function,
			// mostly are fft stuff needed for fftw.

			// Output
			d_im_normalized, // The image after normalization
			d_lattice, // The resulting lattice representing the bias field.
			// Size of this should be equal to 2^(number of level-1) * (initial_point-3) + 3

			// Optional. Also output the bias field
			d_biasField
	);


	checkCudaErrors(cudaMemcpy(h_im_normalized, d_im_normalized, numberOfPixels*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_lattice, d_lattice, numberOfLattice*sizeof(float), cudaMemcpyDeviceToHost));

	// Optional. Also output the bias field
	if (h_biasField!=NULL){
		checkCudaErrors(cudaMemcpy(h_biasField, d_biasField, numberOfPixels*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_biasField));
	}

	checkCudaErrors(cudaFree(d_im));
	checkCudaErrors(cudaFree(d_mask));
	checkCudaErrors(cudaFree(d_im_normalized));
	checkCudaErrors(cudaFree(d_lattice));



}
















































