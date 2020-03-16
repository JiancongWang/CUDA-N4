# CUDA-N4
This is a CUDA implementation of the N4 bias correction algorithm. It implements the 
1). Histogram sharpening
2). B-spline fitting on dense volume of values.

Currently it is roughly 4x faster than the CPU implementation of N4 from ANTs and SimpleITK, benchmarking on a 
machine with i7 4790K and a GTX 1080. 


