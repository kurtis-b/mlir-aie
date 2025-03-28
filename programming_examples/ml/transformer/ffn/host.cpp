#include "test_utils.h"
#include <assert.h>
#include <ctype.h>
#include <fstream>
#include <map>
#include <math.h>
#include <stdfloat>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <tuple>
#include <unistd.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#define VALIDATE 1

#define AIEML_MAX_OFFLOAD_M 3072
#define AIEML_MAX_OFFLOAD_K 3072
#define AIE_MAX_INSTR_LEN 4096
#define TRACE_SIZE 0

#include "common.h"

// --------------------------------------------------------------------------
// AIE initialization stuff
// The following structures and functions are used to initialize multiple
// AIE kernels. We only switch the insts.txts between matmul sizes, so we
// load all inst.txt ahead of time.
// --------------------------------------------------------------------------

#define AIE_MAX_N_BOS 4

union aie_bo_map {
  char *i8;
  int *i32;
  std::bfloat16_t *bf16;
  float *f32;
};

enum aie_bo_dir { IN_ONLY, OUT_ONLY, IN_OUT };

struct aie_bo {
  xrt::bo *bo;
  int group_id;
  enum aie_bo_dir dir;
  size_t len; // in bytes
  union aie_bo_map buf;
};

struct aie_global_state {
  xrt::device *device;
};

struct aie_state {
  std::string xclbin_path;
  std::string kernel_name;
  xrt::xclbin *xclbin;
  xrt::hw_context *context;
  xrt::kernel *kernel;
  size_t n_bos;
  struct aie_bo bos[AIE_MAX_N_BOS];
  std::vector<uint32_t> *last_loaded_insts; // don't reload if they're the same
  int instr_len;
};

struct aie_offload_gemm_info {
  std::vector<uint32_t> *insts;
};

struct aie_global_state aie_global;
struct aie_state aie_gemm;
struct aie_state aie_bias;
std::vector<uint32_t> aie_gemm_64x64_insts;
std::vector<uint32_t> aie_gemm_128x128_insts;
std::vector<uint32_t> aie_gemm_288x288_insts;
std::map<std::tuple<int, int>, struct aie_offload_gemm_info> aie_offload;

std::vector<uint32_t> aie_load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

void aie_init_global() {
  // Set up device
  unsigned int device_index = 0;
  aie_global.device = new xrt::device(device_index);
}

void aie_init_design(struct aie_state *aie_state) {
  // Load xclbin
  constexpr int verbosity = 1;
  if (verbosity >= 1) {
    std::cout << "Loading xclbin: " << aie_state->xclbin_path << "\n";
  }
  aie_state->xclbin = new xrt::xclbin(aie_state->xclbin_path);
  if (verbosity >= 1) {
    std::cout << "Kernel opcode: " << aie_state->kernel_name << "\n";
  }
  auto xkernels = aie_state->xclbin->get_kernels();
  auto xkernel = *std::find_if(
      xkernels.begin(), xkernels.end(), [aie_state](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(aie_state->kernel_name, 0) == 0;
      });
  auto kernel_name = xkernel.get_name();
  if (verbosity >= 1) {
    std::cout << "Registering xclbin: " << aie_state->xclbin_path << "\n";
  }
  aie_global.device->register_xclbin(*aie_state->xclbin);
  if (verbosity >= 1) {
    std::cout << "Getting hardware context.\n";
  }
  if (verbosity >= 1) {
    std::cout << aie_state->xclbin->get_uuid().to_string() << std::endl;
  }
  aie_state->context =
      new xrt::hw_context(*aie_global.device, aie_state->xclbin->get_uuid());
  if (verbosity >= 1) {
    std::cout << "Getting handle to kernel:" << kernel_name << "\n";
  }
  aie_state->kernel = new xrt::kernel(*aie_state->context, kernel_name);

  assert(aie_state->n_bos >= 1 &&
         aie_state->n_bos <= AIE_MAX_N_BOS); // buffer 1 is insts buffer
  aie_state->bos[0].len = AIE_MAX_INSTR_LEN * sizeof(int);
  aie_state->bos[0].group_id = 1;
  aie_state->bos[0].bo = new xrt::bo(
      *aie_global.device, aie_state->bos[0].len, XCL_BO_FLAGS_CACHEABLE,
      aie_state->kernel->group_id(aie_state->bos[0].group_id));
  aie_state->bos[0].buf.i32 = aie_state->bos[0].bo->map<int *>();

  for (int i = 1; i < aie_state->n_bos; i++) {
    aie_state->bos[i].group_id =
        i + 2; // 1 is insts, 2 is insts_len, other buffers start at 3
    aie_state->bos[i].bo = new xrt::bo(
        *aie_global.device, aie_state->bos[i].len, XRT_BO_FLAGS_HOST_ONLY,
        aie_state->kernel->group_id(aie_state->bos[i].group_id));
    aie_state->bos[i].buf.i8 = aie_state->bos[i].bo->map<char *>();
  }
}

std::vector<uint32_t> load_insts(const char *insts_txt_path) {
  std::ifstream file(insts_txt_path);

  if (!file.good()) {
    throw std::runtime_error("Instruction file does not exist: " +
                             std::string(insts_txt_path));
  }
  // Load instructions
  constexpr int verbosity = 1;
  std::vector<uint32_t> instr_v = aie_load_instr_sequence(insts_txt_path);
  if (verbosity >= 1) {
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";
  }
  assert(instr_v.size() < AIE_MAX_INSTR_LEN);
  return std::move(instr_v);
}

void aie_init_insts(struct aie_state *aie_state,
                    std::vector<uint32_t> *instr_v) {
  if (instr_v == aie_state->last_loaded_insts) {
    return;
  }
  memset(aie_state->bos[0].buf.i8, 0, AIE_MAX_INSTR_LEN * sizeof(int));
  memcpy(aie_state->bos[0].buf.i8, instr_v->data(),
         instr_v->size() * sizeof(int));
  aie_state->bos[0].bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  aie_state->instr_len = instr_v->size();
}

void aie_run_design(struct aie_state *aie_state) {
  // bos[0] is synced in init function
  //   printf("syncing buffers ...");
  //   fflush(stdout);
  for (int i = 1; i < aie_state->n_bos; i++) {
    if (aie_state->bos[i].dir == OUT_ONLY) {
      continue;
    }
    aie_state->bos[i].bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  unsigned int opcode = 3;
  //   printf("finished. running design ...");
  //   fflush(stdout);
  auto run = (*aie_state->kernel)(opcode, *aie_state->bos[0].bo,
                                  aie_state->instr_len, *aie_state->bos[1].bo,
                                  *aie_state->bos[2].bo, *aie_state->bos[3].bo);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "AIE Error Status: " << r << std::endl;
    exit(1);
  }
  //   printf("finished. syncing buffers ...");
  //   fflush(stdout);
  for (int i = 1; i < aie_state->n_bos; i++) {
    if (aie_state->bos[i].dir == IN_ONLY) {
      continue;
    }
    aie_state->bos[i].bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }
}

void aie_init() {
  aie_init_global();

  // GEMM design
  aie_gemm_64x64_insts = load_insts("build/insts_64x64.txt");
  aie_gemm_128x128_insts = load_insts("build/insts_128x128.txt");
  aie_gemm_128x128_insts = load_insts("build/insts_288x288.txt");

  aie_gemm.xclbin_path = "build/final.xclbin";
  aie_gemm.kernel_name = "MLIR_AIE";
  aie_gemm.n_bos = 4;
  aie_gemm.bos[1].len =
      AIEML_MAX_OFFLOAD_M * AIEML_MAX_OFFLOAD_K * sizeof(std::bfloat16_t);
  aie_gemm.bos[1].dir = IN_ONLY;
  aie_gemm.bos[2].len = AIEML_MAX_OFFLOAD_K * sizeof(std::bfloat16_t);
  aie_gemm.bos[2].dir = IN_ONLY;
  aie_gemm.bos[3].len = (AIEML_MAX_OFFLOAD_M + TRACE_SIZE) * sizeof(float);
  aie_gemm.bos[3].dir = IN_OUT;
  aie_init_design(&aie_gemm);

  aie_offload[std::make_tuple(64, 64)] =
      (struct aie_offload_gemm_info){&aie_gemm_64x64_insts};
  aie_offload[std::make_tuple(128, 128)] =
      (struct aie_offload_gemm_info){&aie_gemm_128x128_insts};
  aie_offload[std::make_tuple(288, 288)] =
      (struct aie_offload_gemm_info){&aie_gemm_288x288_insts};
}

// --------------------------------------------------------------------------
// Main matmul implementation
// --------------------------------------------------------------------------

template <bool inp_is_col_major>
void aie_do_gemm(long M, long K, const float *__restrict inp,
                 const float *__restrict weight, const float *__restrict bias,
                 float *__restrict out) {
  auto info = aie_offload.find(std::make_tuple(M, K));

  std::bfloat16_t *aie_buf_a = aie_gemm.bos[1].buf.bf16;
  std::bfloat16_t *aie_buf_b = aie_gemm.bos[2].buf.bf16;
  float *aie_buf_c = aie_gemm.bos[3].buf.f32;
  // Copy over A
  if (inp_is_col_major) {
    // design expects inptus to be row major
    for (long i = 0; i < M; i++) {
      for (long j = 0; j < K; j++) {
        aie_buf_a[i * K + j] = (std::bfloat16_t)inp[i + j * M];
      }
    }
  } else {
    for (long i = 0; i < M; i++) {
      for (long j = 0; j < K; j++) {
        aie_buf_a[i * K + j] = (std::bfloat16_t)inp[i * K + j];
      }
    }
  }
  // Copy B
  for (long i = 0; i < K; i++) {
    aie_buf_b[i] = (std::bfloat16_t)weight[i];
  }

  // Run
  //   printf("initializing instructions for key (%d, %d) ...", M, K);
  //   fflush(stdout);
  aie_init_insts(&aie_gemm, info->second.insts);
  //   printf("finished. executing design %s ...",
  //   aie_gemm.kernel_name.c_str()); fflush(stdout);
  aie_run_design(&aie_gemm);
  //   printf("finished. ");
  //   fflush(stdout);

  // Write back results
  memcpy(out, aie_buf_c, (M + TRACE_SIZE) * sizeof(out[0]));
}

// --------------------------------------------------------------------------
// Verification
// --------------------------------------------------------------------------

// forward decl
template <bool a_is_col_major>
void matmul_reference(float *out, const float *a, const float *b,
                      const float *bias, long M, long K);

float out_ref[AIEML_MAX_OFFLOAD_M];
template <bool inp_is_col_major>
bool validate_matmul(long M, long K, const float *__restrict inp,
                     const float *__restrict weight,
                     const float *__restrict bias,
                     float *__restrict out_to_test) {
  matmul_reference<inp_is_col_major>(out_ref, inp, weight, NULL, M, K);
  std::vector<float> CRef(out_ref, out_ref + M);
  std::vector<float> C(out_to_test, out_to_test + M);
  int n_errors = 0;
  std::vector<struct matmul_common::error<float>> errors;
  float max_rel_error = (float)0.0f;
  for (long row = 0; row < M; row++) {
    std::optional<struct matmul_common::error<float>> error =
        matmul_common::verify_single(std::cout, row, 1, CRef[row], C[row], 0.5,
                                     0.05);
    if (error.has_value()) {
      if (n_errors < 10) {
        errors.push_back(*error);
      }
      float rel_error =
          std::abs(error->actual - error->expected) /
          std::max(std::abs(error->actual), std::abs(error->expected));
      if (rel_error > max_rel_error) {
        max_rel_error = rel_error;
      }
      n_errors++;
    }
  }
  if (n_errors > 0) {
    matmul_common::print_error_summary(std::cout, n_errors, errors,
                                       max_rel_error);
    std::cout << std::endl << "Reference:" << std::endl;
    matmul_common::print_matrix(CRef, 1);
    std::cout << std::endl << "Output:" << std::endl;
    matmul_common::print_matrix(C, 1);
    for (int i = 0; i < M; i++) {
      if (C[i] != CRef[i]) {
        std::cout << "Mismatch at index " << i << ": " << C[i] << " vs "
                  << CRef[i] << std::endl;
        if (C[i] == 0.0f) {
          std::cout << "Breaking at index " << i << std::endl;
          break;
        }
      }
    }
    return false;
  }
  return true;
}

template <bool a_is_col_major>
void matmul_reference(float *out, const float *a, const float *b,
                      const float *bias, long M, long K) {
  const int LOOP_UNROLL = 8;
  assert(M % LOOP_UNROLL == 0);

  for (int obt = 0; obt < M; obt += LOOP_UNROLL) {
    // we'll keep LOOP_UNROLL many results in registers
    float result[LOOP_UNROLL];
    // initialize the bias, if it exists
    for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
      result[ibt] = (bias != NULL) ? bias[obt] : 0.0f;
    }
    // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
    // the value of b[i + o * K] and reuse it.
    // we compile with -Ofast, so the compiler will turn the inner loop into
    // FMAs
    for (int i = 0; i < K; i++) {
      float w = b[i];
      for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
        int bt = obt + ibt;
        float inp = (a_is_col_major ? a[bt + i * M] : a[bt * K + i]);
        result[ibt] += inp * w;
      }
    }
    // write back results to main memory
    for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
      int bt = obt + ibt;
      out[bt] = result[ibt];
    }
  }
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

float A[AIEML_MAX_OFFLOAD_M * AIEML_MAX_OFFLOAD_K];
float B[AIEML_MAX_OFFLOAD_K];
float C[AIEML_MAX_OFFLOAD_M + TRACE_SIZE];

int main(int argc, char **argv) {
  aie_init();
  // do three iterations of switching between sizes
  for (int i = 0; i < 1; i++) {
    for (auto it = aie_offload.begin(); it != aie_offload.end(); ++it) {
      auto [M, K] = it->first;
      for (int j = 0; j < M * K; j++) {
        A[j] = matmul_common::get_random<std::bfloat16_t>();
      }
      for (int j = 0; j < K; j++) {
        B[j] = matmul_common::get_random<std::bfloat16_t>();
      }
      for (int j = 0; j < M; j++) {
        C[j] = 0.0f;
      }
      printf("Running matmul: %4dx%4d ...", M, K);
      fflush(stdout);
      auto tstart = std::chrono::system_clock::now();
      aie_do_gemm<false>(M, K, A, B, NULL, C);
      auto tstop = std::chrono::system_clock::now();
      float t =
          std::chrono::duration_cast<std::chrono::microseconds>(tstop - tstart)
              .count();
      printf(" complete after %6.0fus\n", t);

      // Write trace values if trace_size > 0
      if (TRACE_SIZE > 0 and i == 0) {
        test_utils::write_out_trace(((char *)C) +
                                        (AIEML_MAX_OFFLOAD_M * sizeof(float)),
                                    TRACE_SIZE, "trace.txt");
      }

      fflush(stdout);
#if VALIDATE
      if (validate_matmul<false>(M, K, A, B, NULL, C)) {
        printf(" - pass!\n");
      } else {
        printf("FAIL.\n");
        // exit(0);
      }
#else
      printf(" - not validated\n");
#endif
    }
  }
  printf("PASS!\n"); // We will exit in aie_do_gemm above if verification does
                     // not pass.
  return 0;
}
