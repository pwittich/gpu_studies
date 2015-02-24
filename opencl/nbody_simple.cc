#include <cstdio>
#include <cstdlib>
#include <mach/mach_time.h>
#include <cmath>
#include <vector>

class float4 {
 public:
  float x,y,z,w;
  float4 operator+=(const float4 & rhs ) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    this->w += rhs.w;
    return *this ;
  }
  float4 operator+( const float4 & rhs ) {
    float4 tmp = *this;
    tmp += rhs;
    return tmp;
  }
  float4 operator-=(const float4 & rhs ) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    this->w -= rhs.w;
    return *this ;
  }
  float4 operator-( const float4 & rhs )  const {
    float4 tmp = *this;
    tmp -= rhs;
    return tmp;
  }
  float4 operator*=(const float4 & rhs ) {
    this->x *= rhs.x;
    this->y *= rhs.y;
    this->z *= rhs.z;
    this->w *= rhs.w;
    return *this ;
  }
  float4 operator*(const float4 & rhs ) const {
    float4 tmp = *this;
    tmp *= rhs;
    return tmp;
  }

  float4 operator*=(const float & rhs) {
    this->x *= rhs;
    this->y *= rhs;
    this->z *= rhs;
    this->w *= rhs;
    return *this ;
  }
  float4 operator*(const float & rhs) const {
    float4 tmp = *this;
    tmp *= rhs;
    return tmp;
  }
    
 float4(float ax, float ay, float az, float aw) :
  x(ax), y(ay), z(az), w(aw)
  {}
};

float4 operator*(const float & lhs, const float4 & rhs ) {
  float4 tmp=rhs;
  tmp *= lhs;
  return tmp;
}


int main()
{
  uint64_t        t0, t1, t2;
  int nparticle = 2*8192; /* MUST be a nice power of two for simplicity */
  const int nstep = 5;
  //int nburst = 20; /* MUST divide the value of nstep without remainder */
  //int nthread = 64; /* chosen for ATI Radeon HD 5870 */

  float dt = 0.1;
  float eps = 0.0001;


  // create arrays

  std::vector<float4> pos1, pos2, vel;
  pos1.reserve(nparticle);
  pos2.reserve(nparticle);
  vel. reserve(nparticle);
  
  const float4 dt0(dt,dt,dt,0.0f);

  for ( int i = 0; i < nparticle; ++i ){
    pos1.push_back(float4(0.f,0.f,0.f,0.f));
    pos1[i].x = (float)rand()/RAND_MAX * 100. - 50.;
    pos1[i].y = (float)rand()/RAND_MAX * 100. - 50.;
    pos1[i].z = (float)rand()/RAND_MAX * 100. - 50.;
    pos1[i].w = (float)rand()/RAND_MAX * 10.; // mass
    pos2.push_back(float4(0.f,0.f,0.f,0.f));
    vel. push_back(float4(0.f,0.f,0.f,0.f));
  }

  int which = 8;

  float4 startpos = pos1[which];

  printf("Start: particle %d x=%f, y=%f, z=%f, m=%f\n",
	 which, pos1[which].x, pos1[which].y, pos1[which].z, pos1[which].w);
  
  // start iterating
  t0 = t1 = mach_absolute_time();
  for ( int istep = 0; istep<nstep; ++istep ) {
    printf("istep = %d,",istep);

    std::vector<float4> * pos_new,  * pos_old;
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

    for ( int i = 0; i < nparticle; ++i ) { // outer loop over particles
      // if ( i%2000 == 0 ) 
      // 	printf("particle = %d\n", i);
      float4 p = (*pos_old)[i];
      float4 v = vel[i];
      float4 a = float4(0.0f,0.0f,0.0f,0.0f);
      for(int j=0; j<nparticle; j++) { // inner loop over particles
	float4 p2 = (*pos_old)[j]; //Read a cached particle position */
	float4 d = p2 - p;
	float invr = 1./sqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps);
	float f = p2.w*invr*invr*invr;
	a += f*d; // Accumulate acceleration 
      }

      p += dt0*v + 0.5f*dt0*dt0*a;
      v += dt0*a;

      (*pos_new)[i] = p;
      vel[i] = v;
    }
  }

  t2 = mach_absolute_time();
  printf("done.\n");
  struct mach_timebase_info info;
  mach_timebase_info(&info);
  double          t = 1e-9 * (t2 - t1) * info.numer / info.denom;
  printf("Time spent = %g\n", t);


  float4 endpos = pos1[which];

  printf("End:   particle %d x=%f, y=%f, z=%f, m=%f\n",
	 which, pos1[which].x, pos1[which].y, pos1[which].z, pos1[which].w);

  float4 sep = endpos-startpos;;
  float distance = sqrt(sep.x*sep.x + sep.y*sep.y + sep.z*sep.z);
  printf("Distance travelled = %g\n", distance);


  return 0;
}
