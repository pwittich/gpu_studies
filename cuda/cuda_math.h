// $Id$
#ifndef CUDA_MATH
#define CUDA_MATH



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

#endif // CUDA_MATH
