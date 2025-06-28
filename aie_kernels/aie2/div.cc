#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <const int M, const int N, const int H>
void div_v_bf16(bfloat16 *__restrict in, bfloat16 *__restrict out) {
  constexpr int vec_factor = 16;
  event0();
  const int F = N / vec_factor;
  for (int i = 0; i < M; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      for (int j = 0; j < F; j++)
        chess_prepare_for_pipelining chess_loop_range(1, ) {
          aie::vector<bfloat16, vec_factor> A0 = aie::load_v<vec_factor>(in);
          in += vec_factor;
          aie::accum<accfloat, vec_factor> result = aie::div(A0, (bfloat16)H);
          aie::store_v(out, result.template to_vector<bfloat16>());
          out += vec_factor;
        }
    }
  event1();
  return;
}

extern "C" {
#ifndef DIM_M
#define DIM_M 8
#endif

#ifndef DIM_N
#define DIM_N 512
#endif

#ifndef DIM_HEAD
#define DIM_HEAD 32
#endif

void div_2d_bf16(bfloat16 *in, bfloat16 *out) {
  div_v_bf16<DIM_M, DIM_N, DIM_HEAD>(in, out);
}

} // extern "C"
