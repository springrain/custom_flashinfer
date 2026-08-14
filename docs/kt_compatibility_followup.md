# KT Compatibility Follow-up Checklist

This checklist tracks the remaining work after merging the upstream FlashInfer
`main` branch with the KT precision/MLA branch.

Current baseline:

- Integration commit: `8c3c9a42272ef0e216c89565ac269cc30a2e640c`
- Target machine: Ampere One 192-core ARM CPU with two NVIDIA RTX PRO 6000 GPUs
- Primary workload: KT/SGLang Q8 GGUF with LLAMAFILE CPU/GPU experts
- Scope: preserve KT behavior while retaining the current upstream FlashInfer APIs

`[x]` means implemented in the integration branch. `[ ]` means it still needs
validation or implementation.

## Completed In This Merge

- [x] Merge upstream FlashInfer while retaining the KT branch as the merge base.
- [x] Keep KT FP32 FA2 split-K partial storage separate from the SM90 `DTypeO`
      partial buffer.
- [x] Preserve the legacy `MLAPlan` call shape through a compatibility overload.
- [x] Preserve KT `bsz_tensor` behavior in the modern MLA package, including
      legacy positional call forms.
- [x] Preserve KT variable-batch norm call forms in the modern norm package.
- [x] Make upstream fixed-batch norm C++ calls compatible with the optional
      batch-size pointer.
- [x] Keep upstream MLA handling for FP8 scales, empty KV, 64-bit page indices,
      and `return_lse_base_on_e`.
- [x] Push the integration commit to the `springrain/custom_flashinfer` remote
      branches `main`, `master`, and `fix-precision-mla-merge-main`.

## P0: ARM/CUDA Build And Functional Regression

- [ ] Build and install this exact custom_flashinfer commit together with the
      matching `kt-kernel`, PyTorch, CUDA, and SGLang versions on the ARM server.
- [ ] Compile both the new upstream TVM-FFI bindings and the retained KT/JIT
      compatibility sources; do not rely on a stale wheel or JIT cache.
- [ ] Verify the imported `flashinfer` package, compiled extensions, CUDA
      toolkit, and source commit at runtime.
- [ ] Run the Q8 GGUF LLAMAFILE test with all experts on CPU.
- [ ] Run the mixed CPU/GPU expert test, including `kt-num-gpu-experts=64`.
- [ ] Run the all-GPU expert configuration as the reference baseline.
- [ ] Test tensor parallelism with TP1 and TP2.
- [ ] Test batch 1, batch 16, and a larger prefill batch.
- [ ] Test both prefill and decode paths.
- [ ] Compare output with and without `CUDA_LAUNCH_BLOCKING=1`.
- [ ] Compare first-token latency, decode throughput, and end-to-end throughput
      with and without `CUDA_LAUNCH_BLOCKING=1`.
- [ ] Test CUDA graph mode and an eager baseline with CUDA graphs disabled.
- [ ] Verify stream submission, synchronization, deferred work, and shutdown
      under repeated requests.
- [ ] Verify all outputs are finite and non-zero, with no NaN/Inf propagation.
- [ ] Verify GPU expert masks, logical-to-physical remapping, and non-contiguous
      expert placement.
- [ ] Confirm CPU experts actually execute on the ARM worker pools and that CPU
      utilization is not unexpectedly idle while GPUs are saturated.

## P1: KT Runtime Controls And Stability

- [ ] Test NUMA node selection and the default NUMA configuration.
- [ ] Test one and multiple KT thread pools.
- [ ] Verify `kt-cpuinfer` and thread-pool thread allocation, including invalid
      values and startup validation.
- [ ] Test deferred experts and `MAX_DEFER` under prefill and decode.
- [ ] Test dynamic GPU-expert updates and synchronization with active requests.
- [ ] Test expert-distribution recording and GPU-mask telemetry.
- [ ] Test EPLB with KT, including initialization and subsequent remapping.
- [ ] Test CPU/GPU expert ratio and placement strategies.
- [ ] Add an end-to-end Q8 regression script that records model/config, commit,
      extension path, hardware, latency, throughput, and checksums.
- [ ] Add a periodic ARM + Blackwell smoke test to CI or scheduled hardware CI.
- [ ] Compare the merged branch against the pre-merge KT commit for numerical
      output, errors, and performance.

## P1: MLA Compatibility

- [ ] Compile and run the existing KT MLA tests and the current upstream MLA
      tests on the target CUDA GPUs.
- [ ] Exercise changing active batch sizes with `bsz_tensor` during CUDA graph
      capture and replay.
- [ ] Test FP16, BF16, and FP8 KV-cache paths.
- [ ] Test empty-KV handling, LSE output, and `return_lse_base_on_e`.
- [ ] Compare MLA output and workspace use with the pre-merge KT branch.
- [ ] Confirm the FP32 split-K workspace allocation is correct for every FA2
      dispatch and does not regress SM90/Hopper paths.

## P1: Norm Compatibility

- [ ] Compile the norm JIT and TVM-FFI bindings on ARM/CUDA.
- [ ] Test RMSNorm, fused-add RMSNorm, and Gemma norm for fixed and variable
      batch sizes.
- [ ] Test FP16 and BF16 inputs, CUDA graph capture/replay, and changing batch
      sizes.
- [ ] Compare outputs against native PyTorch and the pre-merge KT branch.
- [ ] Decide whether the retained root-level compatibility modules
      (`flashinfer/mla.py` and `flashinfer/norm.py`) should remain as supported
      shims or be replaced by explicit compatibility modules after runtime
      validation.

## P1: Packaging And API Contract

- [ ] Build a clean wheel in the supported Torch/CUDA environment.
- [ ] Install the wheel into a clean environment and run `pip check`.
- [ ] Verify both modern imports (`flashinfer.mla`, `flashinfer.norm`) and KT
      legacy call forms.
- [ ] Verify TVM-FFI and legacy PyTorch/JIT users can coexist without selecting
      the wrong binding or stale cache.
- [ ] Record a supported version matrix for custom_flashinfer, kt-kernel,
      SGLang, PyTorch, CUDA, and the GPU architecture.
- [ ] Update the parent ktransformers submodule gitlink to
      `8c3c9a42272ef0e216c89565ac269cc30a2e640c` only after ARM server tests
      pass.

## P2: Composite KT LoRA And MXFP4 Layerwise Prefill

- [ ] Reimplement composite KT LoRA on the current SGLang/FlashInfer APIs,
      separating expert tensors from attention/linear tensors.
- [ ] Route expert LoRA tensors to the KT CPU expert wrapper and non-expert
      tensors to the native SGLang LoRA manager.
- [ ] Rework MXFP4 layerwise prefill against the current upstream MXFP4/Marlin
      APIs instead of the removed KT-specific classes.
- [ ] Add GPU weight-slot management, memory reservation, lazy loading, and
      restoration for layerwise prefill.
- [ ] Add threshold, OOM fallback, TP initialization, and decode regression
      tests.
- [ ] Keep GPU MXFP4 layerwise work separate from the independent requirement
      for an ARM CPU MXFP4 kernel.

## P2: Performance And Upstream Maintenance

- [ ] Benchmark MLA and norm before and after the merge.
- [ ] Benchmark Q8 CPU-only, mixed, and GPU-only expert execution, including
      concurrency and CPU/GPU utilization.
- [ ] Measure CUDA stream overlap and the performance impact of
      `CUDA_LAUNCH_BLOCKING=1`.
- [ ] Measure the FP32 partial-buffer memory overhead and tune workspace use.
- [ ] Add regression tests for `bsz_tensor`, FP32 partial precision, and legacy
      API compatibility.
- [ ] Keep KT compatibility changes in isolated commits for future upstream
      merges.
- [ ] Periodically rebase or merge upstream `main` and audit TVM-FFI/API changes
      before updating the public integration branches.

## Definition Of Done

- [ ] ARM build and clean-environment installation pass.
- [ ] Q8 all-CPU, mixed, and all-GPU configurations are finite and stable.
- [ ] Mixed-mode output agrees with the all-GPU reference within an agreed
      tolerance.
- [ ] TP1/TP2, prefill/decode, batch sizes, CUDA graph, and eager mode pass.
- [ ] BF16/FP16 MLA and norm compatibility tests pass.
- [ ] Removing `CUDA_LAUNCH_BLOCKING=1` does not cause hangs, corruption, or a
      material unexplained performance regression.
- [ ] The parent ktransformers submodule is updated only after the above checks
      and the exact dependency/version matrix is documented.
