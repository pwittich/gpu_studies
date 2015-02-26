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

class float4_t {
 public:
  float x,y,z,w;
  float4_t operator+=(const float4_t & rhs ) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    this->w += rhs.w;
    return *this ;
  }
   float4_t operator+( const float4_t & rhs ) const {
    float4_t tmp = *this;
    tmp += rhs;
    return tmp;
  }
  float4_t operator-=(const float4_t & rhs ) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    this->w -= rhs.w;
    return *this ;
  }
  float4_t operator-( const float4_t & rhs )  const {
    float4_t tmp = *this;
    tmp -= rhs;
    return tmp;
  }
  float4_t operator*=(const float4_t & rhs ) {
    this->x *= rhs.x;
    this->y *= rhs.y;
    this->z *= rhs.z;
    this->w *= rhs.w;
    return *this ;
  }
  float4_t operator*(const float4_t & rhs ) const {
    float4_t tmp = *this;
    tmp *= rhs;
    return tmp;
  }

  float4_t operator*=(const float & rhs) {
    this->x *= rhs;
    this->y *= rhs;
    this->z *= rhs;
    this->w *= rhs;
    return *this ;
  }
  float4_t operator*(const float & rhs) const {
    float4_t tmp = *this;
    tmp *= rhs;
    return tmp;
  }
    
 float4_t(float ax, float ay, float az, float aw) :
  x(ax), y(ay), z(az), w(aw)
  {}
};

float4_t operator*(const float & lhs, const float4_t & rhs ) {
  float4_t tmp=rhs;
  tmp *= lhs;
  return tmp;
}

// void update_particle(const size_t i, //const size_t nparticle, 
// 		     const thrust::device_vector<float4> * pos_old,
// 		     thrust::device_vector<float4> * pos_new, 
// 		     thrust::device_vector<float4> * vel)
__device__
void update_particle(const size_t i, const size_t nparticle, 
		     const float4* pos_old,
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
  for(size_t i = 0; i < nparticle; ++i  ) { // inner loop over particles
    const float4 p2 = pos_old[i]; //Read a particle position 
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

// SOA
struct wrapper_t {
  //thrust::device_vector<float4> d_pos1, d_pos2, d_vel;
  float4 *d_pos1, *d_pos2, *d_vel;
  int toggle; // pos1->pos2 or pos2->pos1
  size_t nparticle;
};


class functor_thrust {
private:
  struct wrapper_t *_wrapper;
public:
   functor_thrust( struct wrapper_t *w ):
     _wrapper(w)
  {}
  
  __device__ 
  void operator()(const int i) const 
  {
    if ( _wrapper->toggle ) 
    //   update_particle(i, _wrapper.d_pos1, _wrapper.d_pos2, _wrapper.d_vel);
    // else
      update_particle(i, _wrapper->nparticle, _wrapper->d_pos2, _wrapper->d_pos1, _wrapper->d_vel);
  }

};

#ifdef TBB
class functor_tbb {
private:
   const float4_ts * p_pos_old;
   float4_ts * p_pos_new;
   float4_ts * p_vel; 
   int nparticle;
public:
   functor_tbb(   const float4_ts * pos_old,
	      float4_ts * pos_new,
	      float4_ts * vel,
	      int n):
      p_pos_old(pos_old),
      p_pos_new(pos_new),
      p_vel(vel),
      nparticle(n)
   {}

   
   void operator()(const blocked_range<size_t> & r) const 
   {
      for ( size_t i = r.begin(); i != r.end(); ++i ) {
	 update_particle(i, nparticle, p_pos_old, p_pos_new, p_vel);
      }
   }

};
#endif // TBB


int main()
{
  //uint64_t        t0, t1, t2;
  int nparticle = 2*8192; /* MUST be a nice power of two for simplicity */
  const int nstep = 5;
  //int nburst = 20; /* MUST divide the value of nstep without remainder */
  //int nthread = 64; /* chosen for ATI Radeon HD 5870 */

  const float dt = 0.1;
  const float eps = 0.0001;


  thrust::host_vector<float4> h_pos1(nparticle), h_pos2(nparticle), h_vel(nparticle);
  thrust::device_vector<float4> d_pos1(nparticle), d_pos2(nparticle), d_vel(nparticle);
  
  
  const float4 dt0 = make_float4(dt,dt,dt,0.0f);

  for ( int i = 0; i < nparticle; ++i ){
    
    h_pos1.push_back(make_float4((float)rand()/RAND_MAX * 100. - 50.,
				       (float)rand()/RAND_MAX * 100. - 50.,
				       (float)rand()/RAND_MAX * 100. - 50.,
				       (float)rand()/RAND_MAX * 10. // mass
				       ));
    h_pos2.push_back(make_float4(0.f,0.f,0.f,0.f));
    h_vel. push_back(make_float4(0.f,0.f,0.f,0.f));
  }

  struct wrapper_t wrapper;
  // copy to GPU
  d_pos1 = h_pos1;
  d_pos2 = h_pos2;
  d_vel  = h_vel;
  wrapper.d_pos1 = thrust::raw_pointer_cast(d_pos1.data());
  wrapper.d_pos2 = thrust::raw_pointer_cast(d_pos2.data());
  wrapper.d_vel  = thrust::raw_pointer_cast(d_vel .data());
  

  int which = 8;

  
  // loop over time steps
  for ( int istep = 0; istep<nstep; ++istep ) {
    printf("istep = %d,",istep);
    int toggle;  // pos1->pos2 or pos2->pos1
    if ( istep%2==1 ) {
      thrust::for_each(d_pos1.begin(), d_pos1.end(), functor_thrust(&wrapper));
    }
    else {
      thrust::for_each(d_pos2.begin(), d_pos2.end(), functor_thrust(&wrapper));
    }
    
    //std::for_each(parts.begin(), parts.end(), functor_serial(pos_old, pos_new, &vel,nparticle));
    
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
