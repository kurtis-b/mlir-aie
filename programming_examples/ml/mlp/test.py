#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import torch
import torch.nn as nn
import sys
import math
from aie.utils.ml import DataShaper
import time
import os
import numpy as np
import pyxrt as xrt
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
import aie.utils.test as test_utils

# torch.use_deterministic_algorithms(True)
# torch.manual_seed(0)

VERIFY_STOCHASTIC_THRESHOLD = 1024 * 1024 * 1024

def main(opts):
    verbosity = int(opts.verbosity)
    do_verify = int(opts.verify)
    n_iterations = int(opts.iters)
    n_warmup_iterations = int(opts.warmup_iters)
    trace_size = int(opts.trace_size)
    b_col_maj = int(opts.b_col_maj)
    quantize_model = True if int(opts.quantize_model) > 0 else False

    design = "mlp"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    M = int(opts.M)
    K = int(opts.K)
    N = int(opts.N)
    do_verify_stochastic_threshold = True if M * K * N > VERIFY_STOCHASTIC_THRESHOLD else False

    if verbosity >= 1:
        print("Matrix size", M, "x", K, "x", N)

    A_VOLUME = M * K
    B_VOLUME = K * N
    C_VOLUME = M * N

    cpu_time_total = 0
    cpu_time_min = 9999999
    cpu_time_max = 0
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    # There's no tracing in the kernel
    # enable_trace = False if not trace_size else True
    enable_trace = False 
    trace_file = "log/trace_" + design + ".txt"
    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int16")

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    min = -128
    max = 127
    input_fp32 = torch.randint(127, 128, (M, K)).type(torch.FloatTensor)
    weight_fp32 = torch.randint(127, 128, (N, K)).type(torch.FloatTensor)

    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    app = setup_aie(
        xclbin_path,
        insts_path,
        A_VOLUME,
        dtype_in,
        B_VOLUME,
        dtype_wts,
        C_VOLUME,
        dtype_out,
        enable_trace=enable_trace,
        trace_size=trace_size,
    )

    # ------------------------------------------------------
    # Define your golden reference
    # ------------------------------------------------------
    class simple_mlp(nn.Module):
        def __init__(self, input_size, output_size):
            super(simple_mlp, self).__init__()
            # QuantStub converts tensors from floating point to quantized
            self.quant = torch.ao.quantization.QuantStub()
            self.fc1 = nn.Linear(input_size, output_size)
            # DeQuantStub converts tensors from quantized to floating point
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self, x):
            # manually specify where tensors will be converted from floating
            # point to quantized in the quantized model
            x = self.quant(x)
            out = self.fc1(x)
            # manually specify where tensors will be converted from quantized
            # to floating point in the quantized model
            out = self.dequant(out)
            return out

    # ------------------------------------------------------
    # Pytorch baseline
    # ------------------------------------------------------
    model = simple_mlp(K, N)
    model.eval()
    model.fc1.weight.data.copy_(weight_fp32)

    if quantize_model:
        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'x86' for server inference and 'qnnpack'
        # for mobile inference. Other quantization configurations such as selecting
        # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
        # can be specified here.
        # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
        # for server inference.
        # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model.qconfig = torch.ao.quantization.get_default_qconfig('onednn')

        # Fuse the activations to preceding layers, where applicable.
        # This needs to be done manually depending on the model architecture.
        # Common fusions include `conv + relu` and `conv + batchnorm + relu`
        # model_fp32_fused = torch.ao.quantization.fuse_modules(model, [['']])

        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        # model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        model_fp32_prepared = torch.ao.quantization.prepare(model)

        # calibrate the prepared model to determine quantization parameters for activations
        # in a real world setting, the calibration would be done with a representative dataset
        model_fp32_prepared(input_fp32)

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, and replaces key operators with quantized
        # implementations.
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        # Inspect data types of each layer in the model
        print("\nData types of each layer in the model:")
        model_int8_state_dict = model_int8.state_dict()
        for key in model_int8_state_dict:
            print(key, ":", model_int8_state_dict[key])

    # run the model, relevant calculations will happen in int8 if quantized
    for i in range(n_iterations):
        input_fp32 = torch.randint(0, 1, (M, K)).type(torch.FloatTensor)
        if quantize_model:
            start = time.time_ns()
            golden_output = model_int8(input_fp32)
            stop = time.time_ns()
        else:
            start = time.time_ns()
            golden_output = model(input_fp32)
            stop = time.time_ns()
        cpu_time = stop - start
        if cpu_time < cpu_time_min:
            cpu_time_min = cpu_time
        if cpu_time > cpu_time_max:
            cpu_time_max = cpu_time
        cpu_time_total = cpu_time_total + cpu_time

    # ------------------------------------------------------
    # Quantize data for NPU
    # ------------------------------------------------------
    npu_weights = weight_fp32.squeeze().data.numpy().astype(dtype_wts)
    npu_input = input_fp32.squeeze().data.numpy().astype(dtype_in)

    # ------------------------------------------------------
    # Verify the weights
    # ------------------------------------------------------
    npu_weights_reshaped = npu_weights.reshape(K, N)
    for row in range(K):
        if not np.allclose(npu_weights_reshaped[row], weight_fp32.reshape(K,N).squeeze().data.numpy()[row], rtol=0, atol=2):
            print(f"\nFailed at row {row}.\n")
            print(f"NPU input row: {npu_weights_reshaped[row]}")
            print(f"FP32 input row: {weight_fp32.squeeze().data.numpy()[row]}")
            exit(-1)

    npu_input_reshaped = npu_input.reshape(M, K)
    for row in range(M):
        if not np.allclose(npu_input_reshaped[row], input_fp32.squeeze().data.numpy()[row], rtol=0, atol=2):
            print(f"\nFailed at row {row}.\n")
            print(f"NPU input row: {npu_input_reshaped[row]}")
            print(f"FP32 input row: {input_fp32.squeeze().data.numpy()[row]}")
            exit(-1)

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(n_iterations):
        # This commented code below times both the data transfer and the kernel execution
        # start = time.time_ns()
        # entire_buffer = execute(app, npu_input, npu_weights)
        # stop = time.time_ns()
                
        app.buffers[3].write(npu_input)
        app.buffers[4].write(npu_weights)
        app.insts_buffer.sync_to_device()
        start = time.time_ns()
        h = app.call()
        r = h.wait()
        stop = time.time_ns()
        if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise Exception(f"Kernel returned {r}")
        entire_buffer = app.buffers[5].read()

        # if enable_trace:
        #     # Separate data and trace
        #     data_buffer, trace_buffer = extract_trace(
        #         entire_buffer, C_VOLUME, dtype_out, trace_size
        #     )
        #     # Write out the trace
        #     write_out_trace(trace_buffer, trace_file)
        # else:
        #     data_buffer = entire_buffer
        #     trace_buffer = None
        data_buffer = entire_buffer
        trace_buffer = None

        npu_time = stop - start
        if npu_time < npu_time_min:
            npu_time_min = npu_time
        if npu_time > npu_time_max:
            npu_time_max = npu_time
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg CPU time: {}us.".format(int((cpu_time_total / n_iterations) / 1000)))
    print("\nMin CPU time: {}us.".format(int((cpu_time_min) / 1000)))
    print("\nMax CPU time: {}us.".format(int((cpu_time_max) / 1000)))
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / n_iterations) / 1000)))
    print("\nMin NPU time: {}us.".format(int((npu_time_min) / 1000)))
    print("\nMax NPU time: {}us.".format(int((npu_time_max) / 1000)))

    data_buffer_reshaped = data_buffer.reshape(M, N)
    for row in range(M):
        if not np.allclose(data_buffer_reshaped[row], golden_output.detach().numpy()[row], rtol=0, atol=2):
            print(f"\nFailed at row {row}.\n")
            print(f"Data buffer row: {data_buffer_reshaped[row]}")
            print(f"Golden output row: {golden_output.detach().numpy()[row]}")
            exit(-1)

    if np.allclose(
        data_buffer_reshaped,
        golden_output.detach().numpy(),
        rtol=0,
        atol=2,
    ):
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-M",
        "--M",
        dest="M",
        default=512,
        help="Matrix size M",
    )
    p.add_argument(
        "-K",
        "--K",
        dest="K",
        default=512,
        help="Matrix size K",
    )
    p.add_argument(
        "-N",
        "--N",
        dest="N",
        default=512,
        help="Matrix size N",
    )
    p.add_argument(
        "--quantize_model",
        dest="quantize_model",
        default=1,
        help="Whether to run a quantized model",
    )
    p.add_argument(
        "--b_col_maj",
        dest="b_col_maj",
        default=0,
        help="Is B matrix in colum-major format?",
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
