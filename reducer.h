#include <iostream>

class Reducer{
public:
	Reducer(){};
	~Reducer(){};
	static float reduce_sum_wrapper(int n, float *d_idata, float *d_odata);
	static float reduce_max_wrapper(int n, float *d_idata, float *d_odata);
	static float reduce_min_wrapper(int n, float *d_idata, float *d_odata);

private:
};
