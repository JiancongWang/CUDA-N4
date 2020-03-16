// This script is the main interface for the CUDA N4.
#include <complex>
#include <vnl/algo/vnl_fft_1d.h>
#include <vnl/vnl_complex_traits.h>
#include <algorithm>    // std::max
#include <math.h>
#include <stdio.h>
#include <fftw3.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include "itkLoadVolume.h"
#include "CommandLineHelper.h"
#include "cudaN4.h"

using namespace std;

int usage(){
	cout << "CUDA N4: CUDA version of the N4 bias correction. " << endl;
	cout << "(Considering most of you will run this with GTX/RTX cards, it is available in float32 only )" << endl;
	cout << "Usage:" << endl;
	cout << "  CUDAN4 [options]" << endl;
	cout << "Required Options:" << endl;
	cout << "  -i input.nii.gz            : input image" << endl;
	cout << "  -o norm.nii.gz             : bias corrected input image" << endl;
	cout << "Additional Options" << endl;
	cout << "  -m mask.nii.gz             : the mask used for mask out the brain during the calculation" << endl;
	cout << "  -n N                       : max iterations of each level (deafult 50)" << endl;
	cout << "  -t level_fitting           : level of fitting, default (4)" << endl;
	cout << "  -c number_of_control_point : number of control points, default (4)" << endl;
	cout << "  -b bias_field              : output the resulting bias field to this directory" <<  endl;
	return -1;
}


// OK now this runs. Double check if the function is right or not.
// OK now the gpu version also runs. Need to compare the result now.
int main(int argc, char *argv[]){
	// Randomly produce input variables
	srand (time(NULL));

	// Read in user input
	if(argc < 2)
		return usage();
	N4Param n4_param;

	string n;
	CommandLineHelper cl(argc, argv);
	while(!cl.is_at_end()){
		// Read the next command
		std::string arg = cl.read_command();

		if(arg == "-i"){
			n4_param.inputImFile = cl.read_existing_filename();
		}
		else if(arg == "-o"){
			n4_param.outputImFile = cl.read_output_filename();
		}
		else if(arg == "-m"){
			n4_param.maskFile = cl.read_existing_filename();
		}
		else if(arg == "-b"){
			n4_param.outBiasFieldFile = cl.read_output_filename();
		}
		else if(arg == "-n"){
			n4_param.MaximumNumberOfIterations = (unsigned int) cl.read_integer();
		}
		else if(arg == "-t"){
			n4_param.NumberOfFittingLevels = (unsigned int) cl.read_integer();
		}
		else if(arg == "-c"){
			n4_param.NumberOfControlPoints_x = n4_param.NumberOfControlPoints_y =
					n4_param.NumberOfControlPoints_z = (unsigned int) cl.read_integer();
		}
		else{
			cerr << "Unknown option " << arg << endl;
			return -1;
		}
	}

	// Calculate final number of control points
	unsigned int ncpt_x = n4_param.NumberOfControlPoints_x;
	unsigned int ncpt_y = n4_param.NumberOfControlPoints_y;
	unsigned int ncpt_z = n4_param.NumberOfControlPoints_z;

	for (unsigned int i=0; i< n4_param.NumberOfFittingLevels-1; i++){
		ncpt_x=(ncpt_x-3)*2+3;
		ncpt_y=(ncpt_y-3)*2+3;
		ncpt_z=(ncpt_z-3)*2+3;
	}
	int numberOfLattice = ncpt_x * ncpt_y * ncpt_z;


	// IO: read input image and mask
	float spacing[3];
	int size[3];

	itkLoadVolumeHeader3D(n4_param.inputImFile.c_str(), spacing, size);

	if (!n4_param.maskFile.empty()){
		int size_mask[3];
		float spacing_mask[3];
		itkLoadVolumeHeader3D(n4_param.maskFile.c_str(), spacing_mask, size_mask);

		// Sanity check for size
		if ((size_mask[0]!=size[0]) || (size_mask[1]!=size[1]) || (size_mask[2]!=size[2])){
			printf("Size mismatch. im:(%d, %d, %d), mask:(%d, %d, %d) \n",
					size[0], size[1], size[2],
					size_mask[0], size_mask[1], size_mask[2]
			);

			return -1;
		}
	}
	int numberOfPixels = size[0] * size[1] * size[2];

	// Assign memory
	float * h_im = new float[numberOfPixels];
	float * h_mask = new float[numberOfPixels];
	float * h_im_normalized = new float[numberOfPixels];
	float * h_lattice_n4 = new float[numberOfLattice];
	float * h_bias_field;
	if (!n4_param.outBiasFieldFile.empty()){
		h_bias_field = new float[numberOfPixels];
	}else{
		h_bias_field = NULL;
	}

	//	printf("size (%d, %d, %d)\n", size[0], size[1], size[2]);


	// OK. Other than the direction information is lost, this is working ok.
	itkLoadVolume3D(n4_param.inputImFile.c_str(), h_im);
	if (!n4_param.maskFile.empty()){
		itkLoadVolume3D(n4_param.maskFile.c_str(), h_mask);
	}else{
		for (int i=0; i< numberOfPixels; i++){
			h_mask[i] = 1.f;
		}
	}

	// Initialize the FFTW structure needed for the computation.
	N4Data data;
	data.assign(n4_param.NumberOfHistogramBins, n4_param.paddedHistogramSize);

	// Final test of the overall function
	testN4(
			// Input
			h_im, // The input image
			h_mask, // The mask. Right now only binary mask are supported.

			size[2], // What is the input field size. Note that the sequence of size get flipped here.
			size[1],
			size[0],

			n4_param,
			data, // Some additional data structure needed to run the function,
			// mostly are fft stuff needed for fftw.

			// Output
			h_im_normalized, // The image after normalization
			h_lattice_n4, // The resulting lattice representing the bias field.
			// Size of this should be equal to 2^(number of level-1) * (initial_point-3) + 3

			// Optional. Also output the bias field
			h_bias_field
	);


	itkWrtieVolume3D(n4_param.outputImFile.c_str(), h_im_normalized, spacing, size);

	if (!n4_param.outBiasFieldFile.empty()){
		itkWrtieVolume3D(n4_param.outBiasFieldFile.c_str(), h_bias_field, spacing, size);
		delete [] h_bias_field;
	}


	data.clean();
	delete [] h_im_normalized;
	delete [] h_lattice_n4;
	delete [] h_im;
	delete [] h_mask;

	return 0;
}
