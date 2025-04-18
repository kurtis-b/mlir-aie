#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <vec_math.h>

#include "add.cc"

float Q_rsqrt(float number) {
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y = number;
  i = *(long *)&y;
  i = 0x5f3759df - (i >> 1);
  y = *(float *)&i;
  y = y * (threehalfs - (x2 * y * y));

  return y;
}

template <typename T_in, typename T_out, const int M, const int N>
void layer_norm_scalar(T_in *in, T_out *out) {
  for (int i = 0; i < M; i++) {
    float sum = aie::to_float(0);
    float sumsq = aie::to_float(0);
    for (int j = 0; j < N; j++) {
      float val = aie::to_float(in[i * N + j]);
      sum += val;
      sumsq += val * val;
    }
    float mean = sum / aie::to_float(N);
    float var = (sumsq / aie::to_float(N)) - (mean * mean);
    float stdev =
        var * aie::to_float(Q_rsqrt(var)); // Q_rsqrt(var) is 1/sqrt(var)
    for (int j = 0; j < N; j++) {
      float val = aie::to_float(in[i * N + j]);
      float result = (val - mean) / stdev;
      out[i * N + j] = T_out(aie::to_float(result));
    }
  }
}

// LayerNorm will normalize across the embedding dimension for each token
template <typename T_in, typename T_out, const int M, const int N>
void layer_norm_vector(T_in *in, T_out *out) {
  // The commented portion below is a pass-through for the input
  //   for (int i = 0; i < M; i++) {
  //     for (int j = 0; j < N; j++) {
  //       out[i * N + j] = T_out(aie::to_float(in[i * N + j]));
  //     }
  //   }
  constexpr int vec_factor = 16;
  event0();
  const int F = N / vec_factor;
  for (int i = 0; i < M; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      T_in *__restrict pA1 = in + i * N;
      T_out *__restrict pC1 = out + i * N;
      auto sum = aie::to_float(0);
      auto sumsq = aie::to_float(0);
      for (int j = 0; j < F; j++)
        chess_prepare_for_pipelining chess_loop_range(1, ) {
          aie::vector<float, vec_factor> A0 =
              aie::to_float(aie::load_v<vec_factor>(pA1));
          pA1 += vec_factor;
          aie::vector<float, vec_factor> A0sq =
              aie::mul(A0, A0).to_vector<float>(0);
          sum += aie::reduce_add(A0);
          sumsq += aie::reduce_add(A0sq);
        }
      auto mean = aie::div(sum, aie::to_float(N));
      auto var = aie::div(sumsq, aie::to_float(N)) - (mean * mean);
      //   auto invstdev = aie::invsqrt(var);
      v16bfloat16 varVec;
      varVec[0] = var;
      aie::vector<bfloat16, 16> invstdev = getRsqrtBf16(varVec);
      pA1 = in + i * N; // reset pA1 to the start of the row
      for (int j = 0; j < F; j++)
        chess_prepare_for_pipelining chess_loop_range(1, ) {
          aie::vector<float, vec_factor> A0 =
              aie::to_float(aie::load_v<vec_factor>(pA1));
          pA1 += vec_factor;
          aie::accum<accfloat, vec_factor> result_acc;
          result_acc.from_vector(A0, 0);
          auto result = aie::sub(result_acc, mean);
          aie::accum<accfloat, vec_factor> result1 =
              aie::mul(result.to_vector<T_out>(0), invstdev[0]);
          aie::store_v(pC1, result1.to_vector<T_out>(0));
          pC1 += vec_factor;
        }
    }
  event1();
}

extern "C" {
// Commented out unused functions because it takes up space in program memory
// Need to add a build options to build only the required functions
#ifndef DIM_M
#define DIM_M 8
#endif

#ifndef DIM_N
#define DIM_N 512
#endif

void layer_norm_i16_vector(int16 *in, bfloat16 *out) {
  layer_norm_vector<int16, bfloat16, DIM_M, DIM_N>(in, out);
}

// void layer_norm_i16_scalar(int16 *in, bfloat16 *out) {
//   layer_norm_scalar<int16, bfloat16, DIM_M, DIM_N>(in, out);
// }

// void eltwise_add_i16_scalar(int16 *a_in, int16 *b_in, int16 *c_out) {
//   eltwise_add<int16, int16, DIM_M, DIM_N>(a_in, b_in, c_out);
// }

void eltwise_add_i16_vector(int16 *a_in, int16 *b_in, int16 *c_out) {
  eltwise_vadd<int16, int16, DIM_M, DIM_N>(a_in, b_in, c_out);
}

// void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16
// *c_out)
// {
//   eltwise_add<bfloat16, bfloat16, DIM_M, DIM_N>(a_in, b_in, c_out);
// }

// void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16
// *c_out)
// {
//   eltwise_vadd<bfloat16, bfloat16, DIM_M, DIM_N>(a_in, b_in, c_out);
// }
} // extern "C"
