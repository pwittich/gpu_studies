// -*-c++-*-
#include <cstdio>
#include <cstdlib>
//#include <mach/mach_time.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <list>

#include "tbb/tbb.h"
using namespace tbb;

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

class float4_soa_t {
private:
  float *x, *y, *z, *w;
  size_t nparticle_;
public:
  float4_soa_t(size_t n ) :
    x(0),y(0),z(0),w(0),
    nparticle_(n)
  {
    x = new float[nparticle_];
    y = new float[nparticle_];
    z = new float[nparticle_];
    w = new float[nparticle_];
  }
  ~float4_soa_t()
  {
    delete [] x;
    delete [] y;
    delete [] z;
    delete [] w;
  }
  //operator[](size_t i ) 
};


void update_particle(const size_t i, const size_t nparticle, 
		     const std::vector<float4_t> * pos_old,
		     std::vector<float4_t> * pos_new, 
		     std::vector<float4_t> * vel)
{
  const float dt = 0.1;
  const float4_t dt0(dt,dt,dt,0.0f);
  const float eps = 0.0001;
  float4_t p = (*pos_old)[i];
  float4_t v = (*vel)[i];
  float4_t a = float4_t(0.0f,0.0f,0.0f,0.0f);
  for(int j=0; j<nparticle; j++) { // inner loop over particles
     const float4_t p2 = (*pos_old)[j]; //Read a cached particle position */
     float4_t d = p2 - p;
     float invr = 1./sqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps);
     float f = p2.w*invr*invr*invr;
     a += f*d; // Accumulate acceleration 
  }
  
  p += dt0*v + 0.5f*dt0*dt0*a;
  v += dt0*a;
  
  (*pos_new)[i] = p;
  (*vel)[i] = v;
  
}

typedef std::vector<float4_t> float4_ts;

class functor_serial {
private:
   const float4_ts * p_pos_old;
   float4_ts * p_pos_new;
   float4_ts * p_vel; 
   int nparticle;
public:
   functor_serial(   const float4_ts * pos_old,
	      float4_ts * pos_new,
	      float4_ts * vel,
	      int n):
      p_pos_old(pos_old),
      p_pos_new(pos_new),
      p_vel(vel),
      nparticle(n)
   {}

   
   void operator()(const int i) const 
   {
     update_particle(i, nparticle, p_pos_old, p_pos_new, p_vel);
   }

};

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


int main()
{
  uint64_t        t0, t1, t2;
  int nparticle = 2*8192; /* MUST be a nice power of two for simplicity */
  const int nstep = 500;
  //int nburst = 20; /* MUST divide the value of nstep without remainder */
  //int nthread = 64; /* chosen for ATI Radeon HD 5870 */

  const float dt = 0.1;
  const float eps = 0.0001;


  // create arrays

  float4_ts pos1, pos2, vel;
  std::list<int> parts(nparticle);
  std::iota(parts.begin(), parts.end(), 0);
  pos1.reserve(nparticle);
  pos2.reserve(nparticle);
  vel. reserve(nparticle);
  
  
  const float4_t dt0(dt,dt,dt,0.0f);

  for ( int i = 0; i < nparticle; ++i ){
    pos1.push_back(float4_t(0.f,0.f,0.f,0.f));
    pos1[i].x = (float)rand()/RAND_MAX * 100. - 50.;
    pos1[i].y = (float)rand()/RAND_MAX * 100. - 50.;
    pos1[i].z = (float)rand()/RAND_MAX * 100. - 50.;
    pos1[i].w = (float)rand()/RAND_MAX * 10.; // mass
    pos2.push_back(float4_t(0.f,0.f,0.f,0.f));
    vel. push_back(float4_t(0.f,0.f,0.f,0.f));
  }

  int which = 8;

  float4_t startpos = pos1[which];

  printf("Start: particle %d x=%f, y=%f, z=%f, m=%f\n",
	 which, pos1[which].x, pos1[which].y, pos1[which].z, pos1[which].w);
  
  // start iterating
  //t0 = t1 = mach_absolute_time();
  // loop over time steps
  for ( int istep = 0; istep<nstep; ++istep ) {
    printf("istep = %d,",istep);

    float4_ts * pos_new,  * pos_old;
    //const float4_ts * pos_new,  * pos_old;
    if ( istep%2==1 ) {
      pos_new = &pos1;
      pos_old = &pos2;
    }
    else {
      pos_new = &pos2;
      pos_old = &pos1;
    }
    printf("particle %d x=%f, y=%f, z=%f, m=%f\n",
	   which, (*pos_old)[which].x, (*pos_old)[which].y, (*pos_old)[which].z, (*pos_old)[which].w);

    // for ( int i = 0; i < nparticle; ++i ) { // outer loop over particles
    //   update_particle(i,nparticle, pos_old, pos_new, &vel);
    // }
    //std::for_each(parts.begin(), parts.end(), functor_serial(pos_old, pos_new, &vel,nparticle));

    for( auto i: parts) 
      update_particle(i,nparticle, pos_old, pos_new, &vel);    
    tbb::parallel_for(blocked_range<size_t>(0,nparticle),
      		      functor_tbb(pos_old, pos_new, &vel,nparticle));




  }


  //t2 = mach_absolute_time();
  printf("done.\n");
  // struct mach_timebase_info info;
  // mach_timebase_info(&info);
  // double          t = 1e-9 * (t2 - t1) * info.numer / info.denom;
  // printf("Time spent = %g\n", t);


  float4_t endpos = pos1[which];

  printf("End:   particle %d x=%f, y=%f, z=%f, m=%f\n",
	 which, pos1[which].x, pos1[which].y, pos1[which].z, pos1[which].w);

  float4_t sep = endpos-startpos;;
  float distance = sqrt(sep.x*sep.x + sep.y*sep.y + sep.z*sep.z);
  printf("Distance travelled = %g\n", distance);

  //dump
  for ( size_t i = 0; i < nparticle; ++i ) {
    printf("Final: particle %zd x=%f, y=%f, z=%f, m=%f\n",
	 i, pos1[i].x, pos1[i].y, pos1[i].z, pos1[i].w);
  }


  return 0;
}
