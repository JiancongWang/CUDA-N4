# CUDA-N4
This is a complete CUDA implementation of the N4 bias correction algorithm. It implements the 
1). Histogram sharpening
2). Cubic B-spline fitting on dense volume of values. 
and 3). other auxillary functions using CUDA.

Currently it is roughly 4x faster than the CPU implementation of N4 from ANTs and SimpleITK, benchmarking on a 
machine with i7 4790K and a GTX 1080, running Ubuntu 16.04, CUDA 10.1, driver version 418. 

Dependency and compilation: to compile this code please  
1). Install cmake

2). compile the ITK toolkits and change the ITK directory within the CMakeLists.txt file. The ITK is only for image reading/writing. The core algorithm is completely CUDA.

3). compile and install the FFTW3 library. The cmake file should be able to find the FFTW3 library. If not please manually specify the directory in the cmake file. 

Files: 

N4.py - A simple numpy implementation for the N4 algorithm, For people interested in the algorithm but does not want to go through the CUDA code or the original C++ code. 

cudaN4.h/cudaN4.cu - the core CUDA N4 class. Both the histogram sharpening and B-spline fitting for dense volumes are coded here. 

reducer.h/reducer.cu - a simple shared memory reduction code adapted from the Nvidia SDK example. Used in histogram fitting. 

itkLoadVolume.h - simple itk wrapper for reading/writing images. 

CommandLineHelper.h, GreedyException.h and GreedyParameters.h - command line parser borrowed from the Greedy toolkit by Dr. Paul Yushkevich at PICSL lab, Upenn. 

