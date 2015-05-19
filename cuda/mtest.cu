#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <list>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "Matriplex.h"

using namespace Matriplex;

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

// this approach runs into the 48 kB limit per shared memory
// for 6x6 matrices this means something like 1300 matrices, so
// for a matrix muptiplication something like 450 size of matriplex.
template<typename T, idx_t DIM1, idx_t DIM2, idx_t DIM3, idx_t N>
__global__ void matrixkern(const T *d1, 
			   const T *d2,
			   T *d3) 
{
  const int gti = blockIdx.x * blockDim.x + threadIdx.x;
  const int gStride = blockDim.x * gridDim.x;

  __shared__ float m1[DIM1*DIM2*N], m2[DIM2*DIM3*N], m3[DIM1*DIM3*N];

#ifdef NOTDEF

  // allocate memory locally
  if ( threadIdx.x == 0 ) {
     // m1 = (float*) malloc(DIM1*DIM2*N*sizeof(float));
     // m2 = (float*) malloc(DIM2*DIM3*N*sizeof(float));
     // m3 = (float*) malloc(DIM1*DIM3*N*sizeof(float));
     // if ( m1 ) 
     //   memset(m1, 0, DIM1*DIM2*N*sizeof(float));
     // if ( m2 ) 
     //   memset(m2, 0, DIM2*DIM3*N*sizeof(float));
     // if ( m3 ) 
     //   memset(m3, 0, DIM1*DIM3*N*sizeof(float));
     
    // malloc one branch, set pointers by hand
    int totsize = (DIM1*DIM2 + DIM2*DIM3 + DIM1*DIM3)*N*sizeof(float);
    m1 = (float*) malloc(totsize);
    
    // need to clear the memory.Remember this works on bytes, not floats.
    if ( m1 ) {
      memset(m1, 0, totsize);
      m2 = m1 + DIM1*DIM2*N;
      m3 = m2 + DIM2*DIM3*N;
    }
      
  }
  __syncthreads();
  // all threads exit if malloc fails
  //if ( m1 == NULL || m2 == NULL || m3 == NULL ) {
  if ( m1 == NULL  ) {
    if ( threadIdx.x == 0 ) printf("malloc failed in block %d\n", blockIdx.x);
    return;
  }
#endif // NOTDEF
  // copy data into matriplex
  Matriplex::MPlex<float, DIM1, DIM2, N> d_matrices1(&(m1[0]));
  Matriplex::MPlex<float, DIM2, DIM3, N> d_matrices2(&(m2[0]));


  // convert random data to matriplex
  for ( idx_t i = gti; i < N; i += gStride ) {
    //printf("gti = %d, i = %d, dest = %p\n", gti, i, d1);
    d_matrices1.CopyIn(i, d1+i*d_matrices1.kSize);
    d_matrices2.CopyIn(i, d2+i*d_matrices2.kSize);
  }

  Matriplex::Matriplex<float, DIM1, DIM3, N> d_result(&(m3[0]));
  // do matrix multiplication
  MultiplyGeneralStride(d_matrices1, d_matrices2, d_result, gti, gStride);

  // copy result back
  for ( idx_t i = gti; i < N; i += gStride ) 
    d_result.CopyOut(i, d3+i*d_result.kSize);

#ifdef NOTDEF
  // one thread to clear them all
  if ( threadIdx.x == 0 ) {
     free(m1);
     // free(m2);
     // free(m3);
  }
#endif // NOTDEF  
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

  // set the memory limits on the device

  const int NBLOCKS = 26;
  const int DIM1 = 3;
  const int DIM2 = 2;
  const int DIM3 = 4;
  //const int N = 103-6;
  const int N = 332;
  const int nmatrix1 = DIM1*DIM2*N;
  const int nmatrix2 = DIM2*DIM3*N;
  const int nmatrixres = DIM1*DIM3*N;
  printf("Size of memory required: %5.1f kB\n",
	 NBLOCKS * sizeof(float)*(nmatrix1+nmatrix2+nmatrixres)/1024.);

  // get the heap size
  size_t curSize = 0;
  cudaDeviceGetLimit(&curSize, cudaLimitMallocHeapSize);
  curSize *=5.0;
  cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, curSize);
  if ( err != cudaSuccess ) {
    printf("failed to set heap size to %d\n", curSize);
    return 1;
  }
  cudaDeviceGetLimit(&curSize, cudaLimitMallocHeapSize);
  printf("Current size: %5.0f kB\n", curSize/1024.);

  // fill matrices with random data
  float mres[nmatrixres];
  float mres_gpu[nmatrixres];
  memset(mres, 0,nmatrixres*sizeof(float));
  memset(mres_gpu, 0,nmatrixres*sizeof(float));



  cudaDeviceSynchronize();


  // these vectors hold the pre-matriplex matrices
  thrust::host_vector<float> h_pos1(nmatrix1);
  thrust::host_vector<float> h_pos2(nmatrix2);
  thrust::device_vector<float> d_pos1(nmatrix1);
  thrust::device_vector<float> d_pos2(nmatrix2);
  thrust::device_vector<float> d_posres(nmatrixres);
  memset(h_pos2.data(), 0, sizeof(float)*nmatrix2);

  printf("nmatrix1 = %d\n", nmatrix1);
  for ( auto i : h_pos1 ) 
    i = -1.0;
  for (int n = 0; n < N; ++n ) {
    for (int i = 0; i < DIM1; ++i ) {
      for (int j = 0; j < DIM2; ++j ) {
	int p = n*(DIM1*DIM2)+i*DIM2+j;
	//printf("n,i,j,p=%d %d %d %d %f\n", n,i,j,p, 100.*n+10*i+j);
	h_pos1[p] = 100.*n+10*i+j;
	//std::cout << "pos1: " << h_pos1[p] << std::endl;
      }
    }
  }
  // printf("after------\n");
  // for (int i = 0; i < nmatrix1; ++i ) {
  //   std::cout << "pos1: "<< i << "\t" << h_pos1[i]  << std::endl;
  // }


  for (int i = 0; i < nmatrix1; ++i ) {
     h_pos1[i] = rand()*20./RAND_MAX;
  }
  for (int i = 0; i < nmatrix2; ++i ) {
    h_pos2[i] = rand()*20./RAND_MAX;
  }

  // for (int n = 0; n < N; ++n ) {
  //   for (int i = 0; i < DIM1; ++i ) {
  //     int j = i; // diagnoal only
  //     int p = n*(DIM1*DIM2)+i*DIM2+j;
  //     //printf("n,i,j,p=%d %d %d %d %f\n", n,i,j,p, 100.*n+10*i+j);
  //     h_pos2[p] = 1.;
  //     //std::cout << "pos1: " << h_pos1[p] << std::endl;
  //   }
  // }



  // copy to GPU
  printf("copying to GPU .... 1\n");
  d_pos1 = h_pos1;
  printf("copying to GPU .... 2\n");
  d_pos2 = h_pos2;


  const float *d_f1 = thrust::raw_pointer_cast(d_pos1.data());
  const float *d_f2 = thrust::raw_pointer_cast(d_pos2.data());
  float *d_fres = thrust::raw_pointer_cast(d_posres.data());

  float *h_f1 = thrust::raw_pointer_cast(h_pos1.data());
  float *h_f2 = thrust::raw_pointer_cast(h_pos2.data());

  Matriplex::MPlex<float, DIM1, DIM2, N> h_matrices1;
  Matriplex::MPlex<float, DIM2, DIM3, N> h_matrices2;
  memset(h_matrices1.fArray, 0, h_matrices1.kTotSize*sizeof(float));
  memset(h_matrices2.fArray, 0, h_matrices2.kTotSize*sizeof(float));

  // // convert random data to matriplex
  for ( idx_t i = 0; i < N; ++i ) {
    h_matrices1.CopyIn(i, h_pos1.data()+i*h_matrices1.kSize);
    h_matrices2.CopyIn(i, h_pos2.data()+i*h_matrices2.kSize);
  }


 Matriplex::Matriplex<float, DIM1, DIM3, N> h_result;
 MultiplyGeneral(h_matrices1, h_matrices2, h_result);
 // convert resulting data from matriplex
  for ( idx_t i = 0; i < N; ++i ) 
    h_result.CopyOut(i, mres+i*(h_result.kSize));

  // result is now in d_fres
  matrixkern<float, DIM1, DIM2, DIM3,N><<<NBLOCKS,32>>>(d_f1,d_f2, 
						    d_fres );
  //cudaThreadSynchronize();
  cudaDeviceSynchronize();
  // check for error. this catches a kernel launch error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
     // print the CUDA error message and exit
     printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,cudaGetErrorString(error));
     exit(-1);
  }
  // copy result back
  CUDA_SAFE_CALL(cudaMemcpy(mres_gpu,d_fres,sizeof(float)*nmatrixres, cudaMemcpyDeviceToHost));



  printf("i:cpu\tgpu\n");
  for (int i = 0;i<nmatrixres; ++i ) {
    printf("%d: (%d) %8.3f\t%8.3f %s\n", i, int(i/h_result.kSize),mres[i], mres_gpu[i], 
 	   ((mres[i]-mres_gpu[i])<1.0e-3)?"":"<<<");
  }

   
  
  return 0;
}
