#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <vec_math.h>

#include "add.cc"
#include "zero.cc"

// NOTE: The scalar version doesn't work
// float Q_rsqrt(float number) {
//   long i;
//   float x2, y;
//   const float threehalfs = 1.5F;

//   x2 = number * 0.5F;
//   y = number;
//   i = *(long *)&y;
//   i = 0x5f3759df - (i >> 1);
//   y = *(float *)&i;
//   y = y * (threehalfs - (x2 * y * y));

//   return y;
// }

// template <typename T_in, typename T_out, const int M, const int N>
// void layer_norm_scalar(T_in *in, T_out *out) {
//   for (int i = 0; i < M; i++) {
//     float sum = aie::to_float(0);
//     float sumsq = aie::to_float(0);
//     for (int j = 0; j < N; j++) {
//       float val = aie::to_float(in[i * N + j]);
//       sum += val;
//       sumsq += val * val;
//     }
//     float mean = sum / aie::to_float(N);
//     float var = (sumsq / aie::to_float(N)) - (mean * mean);
//     float stdev =
//         var * aie::to_float(Q_rsqrt(var)); // Q_rsqrt(var) is 1/sqrt(var)
//     for (int j = 0; j < N; j++) {
//       float val = aie::to_float(in[i * N + j]);
//       float result = (val - mean) / stdev;
//       out[i * N + j] = T_out(aie::to_float(result));
//     }
//   }
// }

// LayerNorm will normalize across the embedding dimension for each token
template <typename T_in, typename T_out, const int M, const int N>
void layer_norm_vector(T_in *__restrict in, T_out *__restrict out) {
  constexpr int vec_factor = 16;
  event0();
  const int F = N / vec_factor;
  // The commented portion below is a pass-through for the input to test that
  // the input data is correct
  //   for (int i = 0; i < M; i++) {
  //     T_in *__restrict pA1_0 = in + i * N;
  //     T_out *__restrict pC1 = out + i * N;
  //     for (int j = 0; j < F; j++) {
  //       aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1_0);
  //       pA1_0 += vec_factor;
  //       aie::accum<acc32, vec_factor> a_acc;
  //       a_acc.from_vector(A0, 0);
  //       aie::accum<accfloat, vec_factor> test1 =
  //           aie::mul(aie::to_float(A0), aie::to_float(1));
  //       aie::accum<accfloat, vec_factor> test2 =
  //           aie::mul(aie::to_float(a_acc.to_vector<int32>(0)),
  //           aie::to_float(1));
  //       aie::store_v(pC1, test2.to_vector<T_out>(0));
  //       pC1 += vec_factor;
  //     }
  //   }
  for (int i = 0; i < M; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      T_in *__restrict pA1_0 = in + i * N;
      T_in *__restrict pA1_1 = in + i * N;
      T_out *__restrict pC1 = out + i * N;
      aie::accum<accfloat, vec_factor> running_total;
      aie::accum<accfloat, vec_factor> running_sq_total;
      running_total.from_vector(aie::zeros<float, vec_factor>(), 0);
      running_sq_total.from_vector(aie::zeros<float, vec_factor>(), 0);
      float sum = 0;
      float sumsq = 0;
      for (int j = 0; j < F; j++)
        chess_prepare_for_pipelining chess_loop_range(1, ) {
          aie::vector<float, vec_factor> A0 =
              aie::to_float(aie::load_v<vec_factor>(pA1_0));
          pA1_0 += vec_factor;
          aie::accum<accfloat, vec_factor> a_acc;
          a_acc.from_vector(A0, 0);
          running_total = aie::add(running_total, a_acc.to_vector<float>(0));
          auto sumsq_v =
              aie::mul(a_acc.to_vector<float>(0), a_acc.to_vector<float>(0));
          running_sq_total =
              aie::add(running_sq_total, sumsq_v.to_vector<float>(0));
        }
      sum += aie::reduce_add(running_total.to_vector<float>(0));
      sumsq += aie::reduce_add(running_sq_total.to_vector<float>(0));
      float mean = aie::div(sum, aie::to_float(N));
      float var = aie::div(sumsq, aie::to_float(N)) - (mean * mean);
      //   aie::vector<T_out, vec_factor> test =
      //       aie::broadcast<T_out, vec_factor>((T_out)var);
      //   aie::store_v(pC1, test);
      //   pC1 += vec_factor;
      v16bfloat16 varVec;
      varVec[0] = (bfloat16)var;
      aie::vector<bfloat16, 16> invstdev = getRsqrtBf16(varVec);
      for (int j = 0; j < F; j++)
        chess_prepare_for_pipelining chess_loop_range(1, ) {
          aie::vector<float, vec_factor> A0 =
              aie::to_float(aie::load_v<vec_factor>(pA1_1));
          pA1_1 += vec_factor;
          aie::accum<accfloat, vec_factor> a_acc;
          a_acc.from_vector(A0, 0);
          auto result = aie::sub(a_acc, mean);
          aie::accum<accfloat, vec_factor> result1 =
              aie::mul(result.to_vector<T_out>(0), invstdev[0]);
          aie::store_v(pC1, result1.to_vector<T_out>(0));
          pC1 += vec_factor;
        }
    }
  event1();
}

// LayerNorm will normalize across the embedding dimension for each token
template <const int M, const int N>
void layer_norm_vector_bf16(bfloat16 *__restrict in, bfloat16 *__restrict out) {
  constexpr int vec_factor = 16;
  event0();
  const int F = N / vec_factor;
  for (int i = 0; i < M; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      bfloat16 *__restrict pA1_0 = in + i * N;
      bfloat16 *__restrict pA1_1 = in + i * N;
      bfloat16 *__restrict pC1 = out + i * N;
      aie::accum<accfloat, vec_factor> running_total;
      aie::accum<accfloat, vec_factor> running_sq_total;
      running_total.from_vector(aie::zeros<bfloat16, vec_factor>(), 0);
      running_sq_total.from_vector(aie::zeros<bfloat16, vec_factor>(), 0);
      bfloat16 sum = 0;
      bfloat16 sumsq = 0;
      for (int j = 0; j < F; j++)
        chess_prepare_for_pipelining chess_loop_range(1, ) {
          aie::vector<bfloat16, vec_factor> A0 = aie::load_v<vec_factor>(pA1_0);
          pA1_0 += vec_factor;
          aie::accum<accfloat, vec_factor> a_acc;
          a_acc.from_vector(A0, 0);
          running_total = aie::add(running_total, a_acc.to_vector<bfloat16>(0));
          auto sumsq_v = aie::mul(a_acc.to_vector<bfloat16>(0),
                                  a_acc.to_vector<bfloat16>(0));
          running_sq_total =
              aie::add(running_sq_total, sumsq_v.to_vector<bfloat16>(0));
        }
      sum += aie::reduce_add(running_total.to_vector<bfloat16>(0));
      sumsq += aie::reduce_add(running_sq_total.to_vector<bfloat16>(0));
      bfloat16 mean = aie::div(sum, aie::to_float(N));
      bfloat16 var = aie::div(sumsq, aie::to_float(N)) - (mean * mean);
      v16bfloat16 varVec;
      varVec[0] = (bfloat16)var;
      aie::vector<bfloat16, 16> invstdev = getRsqrtBf16(varVec);
      for (int j = 0; j < F; j++)
        chess_prepare_for_pipelining chess_loop_range(1, ) {
          aie::vector<bfloat16, vec_factor> A0 = aie::load_v<vec_factor>(pA1_1);
          pA1_1 += vec_factor;
          aie::accum<accfloat, vec_factor> a_acc;
          a_acc.from_vector(A0, 0);
          auto result = aie::sub(a_acc, mean);
          aie::accum<accfloat, vec_factor> result1 =
              aie::mul(result.to_vector<bfloat16>(0), invstdev[0]);
          aie::store_v(pC1, result1.to_vector<bfloat16>(0));
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

// void layer_norm_i16_vector(int16 *in, bfloat16 *out) {
//   layer_norm_vector<int16, bfloat16, DIM_M, DIM_N>(in, out);
// }

void layer_norm_bf16_vector(bfloat16 *in, bfloat16 *out) {
  layer_norm_vector_bf16<DIM_M, DIM_N>(in, out);
}

// void layer_norm_i16_scalar(int16 *in, bfloat16 *out) {
//   layer_norm_scalar<int16, bfloat16, DIM_M, DIM_N>(in, out);
// }

// void zero_i16_vector(int16 *c_out) {
//   zero_vectorized<int16, DIM_M, DIM_N>(c_out);
// }

// void zero_bf16_vector(bfloat16 *c_out) {
//   zero_vectorized<bfloat16, DIM_M, DIM_N>(c_out);
// }
} // extern "C"
