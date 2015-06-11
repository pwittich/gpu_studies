#ifndef Matriplex_H
#define Matriplex_H

//#include "MatriplexCommon.h"

#ifndef __CUDACC__
#define __host__
#define __device__
#endif // __CUDACC


namespace Matriplex
{
   typedef int idx_t;

//------------------------------------------------------------------------------

   template<typename T, idx_t D1, idx_t D2, idx_t N>
     class Matriplex
   {
   public:
     typedef T value_type;

     enum
     {
       /// return no. of matrix rows
       kRows = D1,
       /// return no. of matrix columns
       kCols = D2,
       /// return no of elements: rows*columns
       kSize = D1 * D2,
       /// size of the whole matriplex
       kTotSize = N * kSize
     };


     //T fArray[kTotSize] __attribute__((aligned(64)));
     T *fArray;
     //__shared__ T fArray[kTotSize];


     __host__
       Matriplex()    { fArray = new T[kTotSize];}
     __device__ __host__
       Matriplex(T * h) {fArray = h; }
     Matriplex(T v) { SetVal(v); }

     idx_t PlexSize() const { return N; }

     void SetVal(T v)
     {
       for (idx_t i = 0; i < kTotSize; ++i)
	 {
	   fArray[i] = v;
	 }
     }
     __host__ __device__
       T  operator[](idx_t xx) const { return fArray[xx]; }
     __host__ __device__
       T& operator[](idx_t xx)       { return fArray[xx]; }

     const T& ConstAt(idx_t n, idx_t i, idx_t j) const { return fArray[(i * D2 + j) * N + n]; }

     T& At(idx_t n, idx_t i, idx_t j) { return fArray[(i * D2 + j) * N + n]; }

     T& operator()(idx_t n, idx_t i, idx_t j) { return fArray[(i * D2 + j) * N + n]; }

     __host__ __device__
       Matriplex& operator=(const Matriplex& m)
       {
	 memcpy(fArray, m.fArray, sizeof(T) * kTotSize); return *this;
       }

     __host__ __device__ inline
     void CopyIn(idx_t n, const T *arr)
     {
#pragma unroll 
       for (idx_t i = n; i < kTotSize; i += N) {
	 fArray[i] = *(arr++);
       }
     }

     __device__ __host__
       void CopyOut(idx_t n, T *arr)
     {
       for (idx_t i = n; i < kTotSize; i += N)
	 {
	   *(arr++) = fArray[i];
	 }
     }

     __device__ 
     inline
       void CopyOutPlex(idx_t n, T *arr)
     {
       T tmp[kSize];
       
       idx_t  j = 0;
#pragma unroll 
       for (idx_t i = n; i < kTotSize; i += N, ++j) {
	 tmp[j] = fArray[i];
       }
       // offset into the output array
       j = n*kSize;
#pragma unroll 
       for ( idx_t i = 0; i < kSize; ++ i) {
	 arr[j+i ] = tmp[i];
       }
       
     }
   };


   template<typename T, idx_t D1, idx_t D2, idx_t N> using MPlex = Matriplex<T, D1, D2, N>;


   //==============================================================================
   // Multiplications
   //==============================================================================
   template<typename T, idx_t D1, idx_t D2, idx_t D3, idx_t N>
   void MultiplyGeneral(const MPlex<T, D1, D2, N>& A,
                        const MPlex<T, D2, D3, N>& B,
                        MPlex<T, D1, D3, N>& C)
   {
     for (idx_t i = 0; i < D1; ++i)
       {
	 for (idx_t j = 0; j < D3; ++j)
	   {
	     const idx_t ijo = N * (i * D3 + j);

	     for (idx_t n = 0; n < N; ++n)
	       {
		 C.fArray[ijo + n] = 0;
	       }

	     //#pragma omp simd collapse(2)
#pragma  ivdep
	     for (idx_t k = 0; k < D2; ++k)
	       {
		 const idx_t iko = N * (i * D2 + k);
		 const idx_t kjo = N * (k * D3 + j);

#pragma ivdep
		 for (idx_t n = 0; n < N; ++n)
		   {
		     // C.fArray[i, j, n] += A.fArray[i, k, n] * B.fArray[k, j, n];
		     C.fArray[ijo + n] += A.fArray[iko + n] * B.fArray[kjo + n];
		   }
	       }
	   }
       }
   } // MutiplyGeneral
   template<typename T, idx_t D1, idx_t D2, idx_t D3, idx_t N>
   __device__
   void MultiplyGeneralStride(const MPlex<T, D1, D2, N>& A,
			      const MPlex<T, D2, D3, N>& B,
			      MPlex<T, D1, D3, N>& C, const int offset, const int stride)
   {
     for (idx_t n = offset; n < N; n += stride)
       {
#pragma unroll
	 for (idx_t i = 0; i < D1; ++i)
	   {
	     for (idx_t j = 0; j < D3; ++j)
	       {
		 const idx_t ijo = N * (i * D3 + j);

		 // commenting this out assumes these are set to zero before  - pw 
		 // for (idx_t nn = 0; nn < N; ++nn)
		 //   {
		 //     C.fArray[ijo + nn] = 0;
		 //   }
		 C.fArray[ijo + n] = 0.f;
#pragma unroll
		 for (idx_t k = 0; k < D2; ++k)
		   {
		     const idx_t iko = N * (i * D2 + k);
		     const idx_t kjo = N * (k * D3 + j);

		     // C.fArray[i, j, n] += A.fArray[i, k, n] * B.fArray[k, j, n];
		     C.fArray[ijo + n] += A.fArray[iko + n] * B.fArray[kjo + n];
		   }
	       }
	   }
       }
   } // MutiplyGeneralStride

   template<typename T, idx_t D1, idx_t D2, idx_t D3, idx_t N>
   __device__
     void AddGeneralStride(const MPlex<T, D1, D2, N>& A, 
			   const MPlex<T, D2, D3, N>& B, 
			   MPlex<T, D1, D3, N>& C, const int offset, const int stride,
			   const float s1 = 1.0, const float s2 = 1.0)
   {
     for (idx_t n = offset; n < N; n += stride)
       {
#pragma unroll
	 for (idx_t i = 0; i < D1; ++i)
	   {
	     for (idx_t j = 0; j < D3; ++j)
	       {
		 const idx_t ijo = N * (i * D3 + j);

		 // commenting this out assumes these are set to zero before  - pw 
		 // for (idx_t nn = 0; nn < N; ++nn)
		 //   {
		 //     C.fArray[ijo + nn] = 0;
		 //   }
#pragma unroll
		 for (idx_t k = 0; k < D2; ++k)
		   {
		     const idx_t iko = N * (i * D2 + k);
		     const idx_t kjo = N * (k * D3 + j);

		     // C.fArray[i, j, n] += A.fArray[i, k, n] * B.fArray[k, j, n];
		     C.fArray[ijo + n] += s1*A.fArray[iko + n] + s2*B.fArray[kjo + n];
		   }
	       }
	   }
       }
   } // MutiplyGeneralStride

   
}// namespace
//------------------------------------------------------------------------------

#endif //  Matriplex_H
