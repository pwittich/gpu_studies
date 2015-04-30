#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <list>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Matriplex.h"

//using namespace Matriplex;

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, 
		      bool abort=true) 
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, 
	      line);
      if (abort) exit(code);
   }
}





int main()
{

  int num_devices, device;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&num_devices));
  printf("This many devices: %d\n", num_devices);
  int max_multiprocessors = -1, max_device = -1;
  cudaDeviceProp best_prop;
  for ( device = 0; device < num_devices; ++device ) {
    cudaDeviceProp properties;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device));
    if ( max_multiprocessors < properties.multiProcessorCount ) {
      max_multiprocessors = properties.multiProcessorCount;
      max_device = device;
      best_prop = properties;
    }
  }
  if ( max_device >=0 )
    cudaSetDevice(max_device);
  else  {
    printf("problem finding a good device! aborting.\n");
    return 1;
  }
  printf("# Running on device %d (name %s)\n", max_device, best_prop.name);

  const int DIM1 = 3;
  const int DIM2 = 2;
  const int DIM3 = 4;
  const int N = 10;
  const int nmatrix1 = DIM1*DIM2*N;
  const int nmatrix2 = DIM2*DIM3*N;
  const int nmatrixres = DIM1*DIM3*N;

  // fill matrices with random data
  float m1[nmatrix1];
  float m2[nmatrix2];
  float mres[nmatrix2];
  for (int i = 0; i < nmatrix1; ++i ) {
    m1[i] = rand()*20./RAND_MAX;
  }
  for (int i = 0; i < nmatrix2; ++i ) {
    m2[i] = rand()*20./RAND_MAX;
  }

  // these vectors hold the pre-matriplex matrices
  thrust::host_vector<float> h_pos1(nmatrix1);
  thrust::host_vector<float> h_pos2(nmatrix1);
  thrust::device_vector<float> d_pos1(nmatrix1);
  thrust::device_vector<float> d_pos2(nmatrix2);

  // copy to GPU
  printf("copying to GPU .... 1\n");
  d_pos1 = h_pos1;
  printf("copying to GPU .... 2\n");
  d_pos2 = h_pos2;


  float *d_f1 = thrust::raw_pointer_cast(d_pos1.data());
  float *d_f2 = thrust::raw_pointer_cast(d_pos1.data());

  float *h_f1 = thrust::raw_pointer_cast(h_pos1.data());
  float *h_f2 = thrust::raw_pointer_cast(h_pos1.data());

  Matriplex::Matriplex<float, DIM1, DIM2, N> h_matrices1;
  Matriplex::Matriplex<float, DIM2, DIM3, N> h_matrices2;

  // convert random data to matriplex
  for ( int i = 0; i < N; ++i ) {
    h_matrices1.CopyIn(i, m1+i*N-1);
    h_matrices2.CopyIn(i, m2+i*N-1);
  }

  Matriplex::Matriplex<float, DIM1, DIM3, N> h_result;
  Matriplex::MultiplyGeneral(h_matrices1, h_matrices2, h_result);

  // result is now in h_result.fArray;


  
  return 0;
}
