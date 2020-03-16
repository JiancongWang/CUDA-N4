# CUDA-N4
This is a CUDA implementation of the N4 bias correction algorithm. It implements the 
1). Histogram sharpening
2). B-spline fitting on dense volume of values.

Currently it is roughly 4x faster than the CPU implementation of N4 from ANTs and SimpleITK, benchmarking on a 
machine with i7 4790K and a GTX 1080. 

To compile this code please also compile the ITK toolkits and change the ITK directory within the CMakeLists.txt file. The ITK is only for image reading/writing. The core algorithm is completely CUDA. 

N4.py - A simple numpy implementation for the N4 algorithm, For people interested in the algorithm but does not want to go through the CUDA code or the original C++ code. 

cudaN4.h/cudaN4.cu - the core CUDA N4 class. Both the histogram sharpening and B-spline fitting for dense volumes are coded here. 

itkLoadVolume.h - simple itk wrapper for reading/writing images. 

reducer.h/reducer.cu - a simple shared memory reduction code adapted from the Nvidia SDK example. Used in histogram fitting. 

CommandLineHelper.h, GreedyException.h and GreedyParameters.h - command line parser borrowed from the Greedy toolkit by Dr. Paul Yushkevich at PICSL lab, Upenn. 

