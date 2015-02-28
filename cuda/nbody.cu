// -*-c++-*-
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <list>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

float4
__device__ __host__ operator*(const float4 a, const float4 b)
{
  return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
float4
__device__ __host__ operator-(const float4 a, const float4 b)
{
  return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

float4
__device__ __host__ operator*(const float a, const float4 b)
{
  return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

float4
__device__ __host__ operator+=(const float4 a, float4 b)
{
  b = make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
  return b;
}

float4
__device__ __host__ operator+(const float4 a, const float4 b)
{
  return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

using namespace thrust;


__device__ __host__ 
void update_particle(const size_t i, const size_t nparticle, 
		     const float4 * pos_old,
		     float4 * pos_new, 
		     float4 * vel)
{
  const float dt = 0.1;
  const float4 dt0 =make_float4(dt,dt,dt,0.0f);
  const float eps = 0.0001;
  float4 p = pos_old[i];
  float4 v = vel[i];
  float4 a = make_float4(0.0f,0.0f,0.0f,0.0f);
  // does this include self-interaction?
  for(size_t j = 0; j < nparticle; ++j  ) { // inner loop over particles
    if ( i == j ) continue;
    const float4 p2 = pos_old[j]; //Read a particle position 
    float4 d = p2 - p;
    float invr = 1./sqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps);
    float f = p2.w*invr*invr*invr;
    a += f*d; // Accumulate acceleration 
  }
  
  p += dt0*v + 0.5f*dt0*dt0*a;
  v += dt0*a;
  
  pos_new[i] = p;
  vel[i] = v;
  
}

// SOA - sorta?
struct MY_ALIGN(16) wrapper_t  {
  int toggle; // pos1->pos2 or pos2->pos1
  size_t nparticle;
  float4 *pos1, *vel, *pos2;
};


class MY_ALIGN(16) functor_thrust {
private:
  const struct wrapper_t _wrapper;
public:
  __host__ __device__
   functor_thrust( struct wrapper_t w ):
     _wrapper(w)
  {}
  
  __device__ __host__ 
  void operator()(const int i) const 
  {
    if ( _wrapper.toggle ) 
      update_particle(i, _wrapper.nparticle, _wrapper.pos1, _wrapper.pos2, _wrapper.vel);
    else
      update_particle(i, _wrapper.nparticle, _wrapper.pos2, _wrapper.pos1, _wrapper.vel);
      
  }

};

#define CUDA_SAFE_CALL(call) call

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



  //uint64_t        t0, t1, t2;
  int nparticle = 8; /* MUST be a nice power of two for simplicity */
  const int nstep = 3;
  //int nburst = 20; /* MUST divide the value of nstep without remainder */
  //int nthread = 64; /* chosen for ATI Radeon HD 5870 */

  const float dt = 0.1;
  const float eps = 0.0001;


  thrust::device_vector<int> d_ivals(nparticle);
  thrust::host_vector<int> h_ivals(nparticle);
  thrust::sequence(d_ivals.begin(), d_ivals.end()); // 0..nparticle-1
  printf("copying from GPU\n");
  h_ivals = d_ivals;
  //return 0;

  thrust::host_vector<float4> h_pos1(nparticle), h_pos2(nparticle), h_vel(nparticle);
  thrust::device_vector<float4> d_pos1(nparticle), d_pos2(nparticle), d_vel(nparticle);

  
  const float4 dt0 = make_float4(dt,dt,dt,0.0f);

  printf("making particles .... \n");
  srand(1232773);
  for ( int i = 0; i < nparticle; ++i ){
    h_pos1[i] = make_float4(100.*rand()/RAND_MAX  - 50.,
			    100.*rand()/RAND_MAX  - 50.,
			    100.*rand()/RAND_MAX - 50.,
			    10.*rand()/RAND_MAX // mass
			    );
    h_pos2[i] = make_float4(0.f,0.f,0.f,0.f);
    h_vel[i]  = make_float4(0.f,0.f,0.f,0.f);
  }
  for ( int i = 0; i < nparticle; ++i ) {
    printf("%d\t%f\n", i,h_pos1[i]);
  }

  struct wrapper_t wrapper;
  struct wrapper_t h_wrapper;
  // copy to GPU
  printf("copying to GPU .... 1\n");
  d_pos1 = h_pos1;
  printf("copying to GPU .... 2\n");
  d_pos2 = h_pos2;
  printf("copying to GPU .... 3\n");
  d_vel  = h_vel;
  printf("done copying to GPU .... \n");
  wrapper.pos1 = thrust::raw_pointer_cast(d_pos1.data());
  wrapper.pos2 = thrust::raw_pointer_cast(d_pos2.data());
  wrapper.vel  = thrust::raw_pointer_cast(d_vel .data());
  wrapper.nparticle = nparticle;
  wrapper.toggle = 0;

  h_wrapper.pos1 = thrust::raw_pointer_cast(h_pos1.data());
  h_wrapper.pos2 = thrust::raw_pointer_cast(h_pos2.data());
  h_wrapper.vel  = thrust::raw_pointer_cast(h_vel .data());
  h_wrapper.nparticle = nparticle;
  h_wrapper.toggle = 0;



  int which = 8;

  
  // loop over time steps
  printf("Starting loop \n");
  for ( int istep = 0; istep<nstep; ++istep ) {
    printf("istep = %d,\n",istep);
    if ( istep%2==0 ) {
      wrapper.toggle = 1;
    }
    else {
      wrapper.toggle = 0;
    }
    try {
      thrust::for_each(d_ivals.begin(), d_ivals.end(), functor_thrust(wrapper));
    }
    catch(thrust::system_error &e) {
      printf("error: %s\n", e.what());
    }
    
    // if ( istep%2==0 ) {
    //   h_wrapper.toggle = 1;
    // }
    // else {
    //   h_wrapper.toggle = 0;
    // }
    // std::for_each(h_ivals.begin(), h_ivals.end(), functor_thrust(h_wrapper));
    
    // tbb::parallel_for(blocked_range<size_t>(0,nparticle),
    //  		      functor_tbb(pos_old, pos_new, &vel,nparticle));
  }


  //t2 = mach_absolute_time();
  printf("done.\n");
  // struct mach_timebase_info info;
  // mach_timebase_info(&info);
  // double          t = 1e-9 * (t2 - t1) * info.numer / info.denom;
  // printf("Time spent = %g\n", t);


  //float4_t endpos = pos1[which];

  //printf("End:   particle %d x=%f, y=%f, z=%f, m=%f\n",
  // which, pos1[which].x, pos1[which].y, pos1[which].z, pos1[which].w);

  // //float4_t sep = endpos-startpos;;
  // float distance = sqrt(sep.x*sep.x + sep.y*sep.y + sep.z*sep.z);
  // printf("Distance travelled = %g\n", distance);

  h_pos1 = d_pos1; // copy back

  //dump
  for ( size_t i = 0; i < nparticle; ++i ) {
    printf("Final: particle %d x=%f, y=%f, z=%f, m=%f\n",
	 i, h_pos1[i].x, h_pos1[i].y, h_pos1[i].z, h_pos1[i].w);
  }


  return 0;
}
