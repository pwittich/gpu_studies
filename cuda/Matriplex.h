#ifndef Matriplex_H
#define Matriplex_H

//#include "MatriplexCommon.h"

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

     T fArray[kTotSize] __attribute__((aligned(64)));

     Matriplex()    {}
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

     __host__ __device__
       void CopyIn(idx_t n, T *arr)
     {
       for (idx_t i = n; i < kTotSize; i += N)
	 {
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
   };


   //template<typename T, idx_t D1, idx_t D2, idx_t N> using MPlex = Matriplex<T, D1, D2, N>;


   //==============================================================================
   // Multiplications
   //==============================================================================
   template<typename T, idx_t D1, idx_t D2, idx_t D3, idx_t N>
     void MultiplyGeneral(const Matriplex<T, D1, D2, N>& A,
			  const Matriplex<T, D2, D3, N>& B,
			  Matriplex<T, D1, D3, N>& C)
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
	     for (idx_t k = 0; k < D2; ++k)
	       {
		 const idx_t iko = N * (i * D2 + k);
		 const idx_t kjo = N * (k * D3 + j);

#pragma simd
		 for (idx_t n = 0; n < N; ++n)
		   {
		     // C.fArray[i, j, n] += A.fArray[i, k, n] * B.fArray[k, j, n];
		     C.fArray[ijo + n] += A.fArray[iko + n] * B.fArray[kjo + n];
		   }
	       }
	   }
       }
   } // MutiplyGeneral
}// namespace
//------------------------------------------------------------------------------

#endif //  Matriplex_H
