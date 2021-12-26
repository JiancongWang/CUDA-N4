/* Written by Jiancong Wang.
 * This implements the N4 bias correction algorithm in CUDA, with a fixed
 * cubic B spline.
 * The original paper titled "N4ITK: Improved N3 Bias Correction".
 * For the B spline implementation, refer to
 * "Scattered Data Interpolation with Multilevel B-Splines".
 * C++ code is in the ITK package.
 */

#ifndef CUDAN4_H
#define CUDAN4_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>
#include <vector>
#include <string.h>
#include <fftw3.h>
#include <math.h>
#include <iostream>
#include <string>

/* This struct contains parameters for the N4 algorithm. */
struct N4Param{
	/* For histogram based sharpening */
	/**
	 * Set number of bins defining the log input intensity histogram.
	 * Default = 200.
	 */
	unsigned int NumberOfHistogramBins = 200;
	float exponent = ceil( log((float)NumberOfHistogramBins) / log( 2.0 ) ) + 1.;
	unsigned int paddedHistogramSize = (unsigned int)(pow(2.0, exponent ) + 0.5);
	unsigned int histogramOffset = (unsigned int)( 0.5 * ( paddedHistogramSize - NumberOfHistogramBins ) );

	/**
	 * Set the noise estimate defining the Wiener filter.  Default = 0.01.
	 */
	float WienerFilterNoise = 0.01;

	/**
	 * Set the full width at half maximum parameter characterizing the width of
	 * the Gaussian deconvolution.  Default = 0.15.
	 */
	float BiasFieldFullWidthAtHalfMaximum = 0.15;


	/* For bspline smoothing*/
	/**
	 * Set the control point grid size defining the B-spline estimate of the
	 * scalar bias field.  In each dimension, the B-spline mesh size is equal
	 * to the number of control points in that dimension minus the spline order.
	 * Default = 4 control points in each dimension for a mesh size of 1 in each
	 * dimension.
	 * Also note that this means the number of control points at the coarest level.
	 * The number of control point  will be 2^(level-1)+ 3 if initial control is set to
	 * 4.
	 */
	unsigned int NumberOfControlPoints = 4;

	unsigned int NumberOfControlPoints_x = 4; // number of control points in 3 dimensions
	unsigned int NumberOfControlPoints_y = 4;
	unsigned int NumberOfControlPoints_z = 4;


	/**
	 * Set the number of fitting levels.  One of the contributions of N4 is the
	 * introduction of a multi-scale approach to fitting. This allows one to
	 * specify a B-spline mesh size for initial fitting followed by a doubling of
	 * the mesh resolution for each subsequent fitting level.  Default = 1 level.
	 */
	unsigned int NumberOfFittingLevels = 4;


	// Set the maximum number of iterations specified at each fitting level. Default = 50.
	unsigned int MaximumNumberOfIterations = 50;

	/**
	 * Set the convergence threshold.  Convergence is determined by the
	 * coefficient of variation of the difference image between the current bias
	 * field estimate and the previous estimate.  If this value is less than the
	 * specified threshold, the algorithm proceeds to the next fitting level or
	 * terminates if it is at the last level. Default 0.001.
	 */
	float ConvergenceThreshold = 0.001;
	float low_value = 10.;


	/* IO parameters */
	std::string inputImFile;
	std::string outputImFile;
	std::string maskFile;
	std::string outBiasFieldFile;

};


/* This struct contains the data structure needed for the N4 algorithm. */
struct N4Data{
	/* Variables */

	/* For histogram based sharpening */
	// The FFTW FFT plans - forward
	fftwf_plan pf_v;
	fftwf_plan pf_f;
	fftwf_plan pf_numerator;
	fftwf_plan pf_denominator;

	// backward
	fftwf_plan pb_u;
	fftwf_plan pb_numerator;
	fftwf_plan pb_denominator;

	// Real value variables
	float * d_histogram;

	float * h_V; // This variable is the histogram
	float * h_F;
	float * h_U;
	float * h_numerator;
	float * h_denominator;
	float * h_E;

	// Complex value variables
	std::vector<std::complex<float>> h_Vf;
	std::vector<std::complex<float>> h_Ff;
	std::vector<std::complex<float>> h_Uf;
	std::vector<std::complex<float>> h_numeratorf;
	std::vector<std::complex<float>> h_denominatorf;

	/* Functions */
	// This function assign the memory needed for the N4 algorithm
	void assign(const unsigned int NumberOfHistogramBins,
			const unsigned int paddedHistogramSize){
		// Calculate the size needed for the histogram bin

		h_V = new float[paddedHistogramSize];
		memset(h_V, 0, paddedHistogramSize*sizeof(float));
		// Need to init the padded histogram to 0. This makes sure the padded part is all 0 so it is necessary.

		h_F = new float[paddedHistogramSize];
		h_U = new float[paddedHistogramSize];
		h_numerator = new float[paddedHistogramSize];
		h_denominator = new float[paddedHistogramSize];
		h_E = new float[NumberOfHistogramBins];

		h_Vf = std::vector<std::complex<float>>(paddedHistogramSize / 2 + 1);
		h_Ff = std::vector<std::complex<float>>(paddedHistogramSize / 2 + 1);
		h_Uf = std::vector<std::complex<float>>(paddedHistogramSize / 2 + 1);
		h_numeratorf = std::vector<std::complex<float>>(paddedHistogramSize / 2 + 1);
		h_denominatorf = std::vector<std::complex<float>>(paddedHistogramSize / 2 + 1);

		pf_v = fftwf_plan_dft_r2c_1d(paddedHistogramSize, h_V,
				reinterpret_cast<fftwf_complex*>(&h_Vf[0]) ,
				FFTW_MEASURE);
		pf_f = fftwf_plan_dft_r2c_1d(paddedHistogramSize, h_F,
				reinterpret_cast<fftwf_complex*>(&h_Ff[0]),
				FFTW_MEASURE);
		pf_numerator = fftwf_plan_dft_r2c_1d(paddedHistogramSize, h_numerator,
				reinterpret_cast<fftwf_complex*>(&h_numeratorf[0]),
				FFTW_MEASURE);
		pf_denominator = fftwf_plan_dft_r2c_1d(paddedHistogramSize, h_U,
				reinterpret_cast<fftwf_complex*>(&h_denominatorf[0]),
				FFTW_MEASURE);

		pb_u = fftwf_plan_dft_c2r_1d(paddedHistogramSize,
				reinterpret_cast<fftwf_complex*>(&h_Uf[0]),
				h_U, FFTW_MEASURE);
		pb_numerator = fftwf_plan_dft_c2r_1d(paddedHistogramSize,
				reinterpret_cast<fftwf_complex*>(&h_numeratorf[0]),
				h_numerator, FFTW_MEASURE);
		pb_denominator = fftwf_plan_dft_c2r_1d(paddedHistogramSize,
				reinterpret_cast<fftwf_complex*>(&h_denominatorf[0]),
				h_denominator, FFTW_MEASURE);
	}

	// This function cleans up the assigned memory after done.
	void clean() {
		delete [] h_V;
		delete [] h_F;
		delete [] h_U;
		delete [] h_numerator;
		delete [] h_denominator;
		delete [] h_E;

		fftwf_destroy_plan(pf_v);
		fftwf_destroy_plan(pf_f);
		fftwf_destroy_plan(pf_numerator);
		fftwf_destroy_plan(pf_denominator);

		fftwf_destroy_plan(pb_u);
		fftwf_destroy_plan(pb_numerator);
		fftwf_destroy_plan(pb_denominator);
	}

};

// The function that takes in the sharpening kernel and sharpen the image.
void sharpenImage(
		// Input
		float * d_before_sharpen, // image before sharpening. Does not set const here because the min/max reduction has to set the background value.
		const float * d_mask, // the input mask
		const unsigned int n, // number of pixel within the image

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
);

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
		float & binMax_out);

void calculateConvergenceMeasurement(
		// Input
		const float * d_field1,
		const float * d_field2,
		const float * d_mask,
		const unsigned int numberOfPixels,
		const float numberOfForeground,
		// output
		float & convergence);


void upsampleLattice3D_cpu(
		// Input
		const float * h_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		// Output
		float * h_lattice_upsample);

void upsampleLattice3D_gpu(
		// Input
		const float * d_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		// Output
		float * d_lattice_upsample);

void testUpsampleLattice3D_gpu(
		// Input
		const float * h_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		// Output
		float * h_lattice_upsample);


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
		float * d_fitted);


void testEvaluateBspline3D(// Input
		const float * h_lattice,
		const unsigned int ncpt_x, // number of control points
		const unsigned int ncpt_y,
		const unsigned int ncpt_z,
		const unsigned int fx, // size of the output field
		const unsigned int fy,
		const unsigned int fz,
		// Output
		float * h_fitted);


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
);


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
);


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
		float * d_biasField = NULL


);

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
		// Size of this should be qual to 2^(number of level-1) * (initial_point-3) + 3

		// Optional. Also output the bias field
		float * h_biasField = NULL
);


#endif






























