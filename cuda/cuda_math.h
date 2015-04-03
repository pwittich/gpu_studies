// $Id$
#ifndef CUDA_MATH
#define CUDA_MATH



inline 
__device__ __host__ float4 operator*(const float4 a, const float4 b)
{
  return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
inline
__device__ __host__ float4 operator-(const float4 a, const float4 b)
{
  return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline
__device__ __host__ float4 operator*(const float a, const float4 b)
{
  return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

inline
__device__ __host__ void operator+=(float4 & a, const float4 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

inline
__device__ __host__ float4 operator+(const float4 a, const float4 b)
{
  return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

#endif // CUDA_MATH
