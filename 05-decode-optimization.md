---
title: "Decode optimization on GB10: the bytes model"
slug: 05-decode-optimization
topic_number: 05
created: 2026-09-23
last_updated: 2026-09-23
status: active
type: topic
wave: 2
engines: [vLLM, SGLang, llama.cpp and forks, ExLlamaV3]
evidence_grades: [Measured, Community-measured, Code-verified, Historical diagnostic, Interpretation, Proposed]
---

# Decode optimization on GB10: the bytes model

Hardware for every own row: ASUS Ascent GX10 (NVIDIA GB10, SM121, compute capability 12.1, 48 SMs, 128 GB unified LPDDR5X, 273 GB/s nominal). Single-node rows ran on one box; TP2 rows on node-1 + node-2 over a direct RoCE link; TP3 rows on node-1 + node-2 + node-3 in a RoCE triangle. Community rows name the repository, thread or handle, on DGX Spark or other GB10 boxes. "tok/s" is per stream unless marked "agg". "Forced-512" means 512 output tokens with EOS ignored, temperature 0, thinking off and a fresh nonce per repetition. Code is quoted at these pins, all fetched 2026-09-23: vLLM `711fc55c10a2`, SGLang `2909f84e3b2f`, llama.cpp `6e60f35608ec`, ExLlamaV3 `6b84a21b6f1e`, antirez/ds4 `0aaea5a238fb`, Entrpi/ds4-on-spark `99fb7b170528`, Entrpi/ds4 `b0a147a7fba6`, NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark `eb11cef295c1`, Zeuss5/cuda-exl3 `6a1ffc34866e`.

**The model this page uses.** [Interpretation, assembled from the anchors in §4]

Per-rank critical-path time for one decode step:

```
T_step ≈ max_over_ranks(B_rank) / BW_eff  +  N_coll · t_coll  +  N_launch · t_launch (ungraphed only)  +  T_host

B_rank = W_dense/TP + W_replicated + E_rank(M) · b_expert + KV_read(depth, rows)
         + S_state(sequences) + B_drafter + B_lm_head/TP

tok/s per stream = a(k) / T_step        aggregate = C · a(k) / T_step
M = C · (k + 1) verify rows;  a(k) = committed tokens per step per stream
```

What changes by model family:
- **Dense.** No expert term. GQA KV shards across TP while the KV-head count is at least TP.
- **MoE top-k.** The expert term dominates. Under expert parallelism the step waits on the slowest rank. E_rank(M) under uniform independent routing is an upper bound, not an estimate (§3, §8).
- **MLA (DeepSeek V4 / V4.1).** The latent KV is small and replicated on every TP rank, so it does not shrink with TP. Sparse-attention indexer work grows with context.
- **Hybrid linear-attention.** Recurrent state is a constant per sequence and independent of depth. Only the full-attention layers carry a KV term that grows with depth.

BW_eff is a measured ruler of about 230-245 GB/s, never the 273 GB/s nominal (§3). State of the evidence in one line: the bytes model predicts single-node BF16 dense decode to within about 10%. It is only a ±25% guide for multi-node MoE, where routing correlation, L2 reuse and fixed collective and host costs dominate. No DRAM byte counter has been published for any decode kernel on GB10. [Interpretation, summarising §4 and §8]

## 1. What it is

When a language model writes, it produces one token (a word fragment) at a time. Every token costs one decode step (one forward pass per token), and that step must read from memory every weight it touches, the KV cache (stored attention context) for the text so far, and a few small fixed extras. A GB10 can do far more arithmetic per second than its memory can feed it. Its LPDDR5X memory is rated at 273 GB/s and delivers about 230-245 GB/s when measured, so the length of a decode step is set mostly by how many bytes it must read, not by how much math it does. That gives a simple yardstick, the bytes model (step time ≈ bytes ÷ bandwidth). A 27B dense model in 4-bit reads about 16 GB of weights per step and measures roughly 10-12 tokens per second on one box without speculation. Every way of making decode faster on GB10 does one of four things. It reads fewer bytes per step (smaller weight formats, a smaller draft model, compact KV). It splits the bytes across machines (tensor parallel across nodes), which adds network round-trips. It commits more tokens per read, through speculative decoding (guess ahead, verify in bulk) or by serving several users at once. Or it removes fixed costs that are not bytes at all: kernel launches, network collectives and host-side stalls. This page gives the arithmetic per model family and shows which kernels each engine picks for the small matrix shapes decode produces. It collects what has been measured and ranks the levers by what they actually delivered on GB10. [Measured and Community-measured anchors in §4; the rest Interpretation] See also [Speculative decoding](02-speculative-decoding.md), [MoE dispatch](08-moe-dispatch.md), [GEMM backends and quant formats](09-gemm-backends-and-quant-formats.md), [Inter-Spark communication](03-inter-spark-communication.md) and [SM121 hardware facts](17-sm121-hardware-facts.md).

## 2. How each engine does it

Every engine below chooses its weight kernel once per layer at load time. Decode then runs whatever was chosen, at whatever row count M the batch and draft depth produce. The only per-step small-M switches found are Marlin's half-height tile for M ≤ 8 (vLLM), MMVQ vs MMQ by batch width (llama.cpp), and the M ≤ 8 GEMV and cooperative MoE paths in ExLlamaV3. [Code-verified at the pins above] The full kernel-selection tables live in [Engine dispatch maps](16-engine-dispatch-maps.md). This section covers only what sets decode bytes and launch count.

### vLLM

Pinned at `711fc55c10a2`.

- **NVFP4 linear, W4A4 checkpoint.** The auto priority list is FlashInfer CuTe-DSL → FlashInfer CUTLASS → FlashInfer b12x → vLLM CUTLASS → CuTe-DSL W4A16 → Marlin → … (`vllm/model_executor/kernels/linear/__init__.py:551-571`). CuTe-DSL requires family 100 (`nvfp4/flashinfer.py:116-123`). FlashInfer CUTLASS requires `cutlass_fp4_supported()` and `has_device_capability(100)`, and SM121 satisfies both (`nvfp4/flashinfer.py:184-197`). `cutlass_fp4_supported()` is true for 120 ≤ cc < 130 only when the SM12x FP4 kernel was compiled in (`nvfp4_scaled_mm_entry.cu:71-90`). On a W4A4 checkpoint, SM121 therefore selects FlashInfer CUTLASS. [Code-verified]
- **NVFP4 weight-only (W4A16).** Auto selects Marlin on SM121. CuTe-DSL W4A16 is preferred only on cc 100/103 (`__init__.py:1115-1126`). [Code-verified]
- **The emulation trap.** If neither the SM12x FP4 kernel nor FlashInfer is present, selection falls down the list to emulation, which only logs a warning. The same NVFP4 weights ran about 1.1 tok/s under emulation against 77.1 with FlashInfer CUTLASS selected (NVIDIA forum 379766). Read the selected kernel in the log, not the quantization name. [Community-measured]
- **NVFP4 MoE.** The auto list is at `fused_moe/oracle/nvfp4.py:194-203`.
  - TRT-LLM and CuTe-DSL require family 100 (`experts/trtllm_nvfp4_moe.py:206-213`, `experts/flashinfer_cutedsl_moe.py:76-82`).
  - FlashInfer CUTLASS accepts family 120 (`experts/flashinfer_cutlass_moe.py:128-139`), so SM121 lands there by default.
  - FlashInfer b12x MoE supports family 120 (`experts/flashinfer_b12x_moe.py:165-171`). It is excluded from auto "until the upstream CUTLASS SM121 MMA op guard is resolved" and must be requested.
  - Marlin accepts any cc ≥ 7.5. [Code-verified]
  - Community GB10 recipes override auto and force Marlin: sfxnz's GLM NVFP4 two-node `run.sh` refuses any other MoE backend ("flashinfer_cutlass OOM'd"), and NNNtrance's TP3+EP production uses Marlin because "the FP4 paths refuse expert maps". [Community-measured] That claim is scoped to that kit, build and topology: an own TP2+EP boot receipt shows FlashInfer CUTLASS selected with EP on (see [MoE dispatch](08-moe-dispatch.md) §4), so it is not a general GB10 rule.
- **FP8 block.** The priority list is FlashInfer-DeepGEMM → DeepGEMM → CUTLASS block → b12x → Marlin → Humming → Triton → BlockWise torch (`__init__.py:455-481`). `support_deep_gemm()` includes family 120 (`vllm/platforms/cuda.py:719-725`). CUTLASS block FP8 reports support for cc ≥ 100 with CUDA ≥ 12.8 (`scaled_mm_entry.cu:161-171`). The weight-only FP8 list is Humming → Marlin (`__init__.py:483-497`). [Code-verified]
- **Marlin small-M** (`csrc/libtorch_stable/quantization/marlin/marlin.cu:295-321`).
  - Rows are split into chunks of up to 4 × 16.
  - `m_block_size_8` is set when a chunk has ≤ 8 rows and 16-bit activations, so M ≤ 8 runs a half-height MMA tile.
  - Marlin W4A8-FP8 is restricted to SM89 and SM12x.
  - Atomic-add reduction applies only when n < 2048, k ≥ 2048 and `VLLM_MARLIN_USE_ATOMIC_ADD=1` (`marlin_utils.py:687-706`). Community guidance on this variable conflicts: 0xBakeer describes it as a correctness fix on SM121, flash7777 as a no-op at batch 1.
  - Marlin is compiled for the `12.0f` family on CUDA ≥ 13.0, else `12.0a;12.1a` (`CMakeLists.txt:610-636`). [Code-verified; env guidance Community-measured and unresolved]
- **EXL3.** Not in upstream vLLM. A grep finds no `exl3` or `exllamav3` at the pin. vLLM's `ExllamaLinearKernel` is the GPTQ exllama path, not EXL3. [Code-verified] See the ExLlamaV3 subsection.
- **Speculation.** `SpeculativeConfig.create_draft_parallel_config` copies the TP size, the EP flag and the executor fields, but not `decode_context_parallel_size` (`vllm/config/speculative.py:1760-1785`). A draft running with `dcp_world_size=1` against DCP-sharded KV loses acceptance (§4, local-inference-lab/vllm PR #72). [Code-verified]
- **Known-bad or known-risky for decode:**
  - `--enforce-eager` on MoE (§4). [Community-measured]
  - On trees without vLLM #51538 (merged as `97388c44`), FULL CUDA-graph replay of padded speculative-decode rows can wedge in vLLM's DSA `persistent_topk`, which reads a padded row's negative context length as uint32 and spins. It surfaced as a sparse-MLA hang (flashinfer #5015, closed, FlashInfer exonerated). Workaround: capture every reachable row count. Fix: rebase past #51538. [Community-measured]
  - The MTP CUDA-graph memory estimate went negative on unified memory (−35.69 GiB), which inflates the KV pool (vLLM #44740). [Community-measured]
  - A hybrid GDN speculative-decode misdispatch when prompt length = 1 + k (vLLM #53051). It is engine-general, not GB10-specific. [Community-measured]
  - Raising `--max-num-batched-tokens` without re-measuring decode (§4). [Community-measured; Historical diagnostic]
- **Defaults on CC 12.1.** CUDA graphs are on unless `--enforce-eager`. The MoE backend is FlashInfer CUTLASS for W4A4 NVFP4 checkpoints, and the weight-only FP4 path is Marlin. FP8 block goes to DeepGEMM when enabled, else CUTLASS block. [Code-verified]

### SGLang

Pinned at `2909f84e3b2f`.

- **Capability helpers** (`python/sglang/srt/utils/common.py:286-336`). `is_sm120_supported` is major 12, which includes SM121. `is_sm120()` is exactly 12.0 and `is_sm121()` exactly 12.1. `get_platform().is_blackwell` covers majors 10, 11 and 12. SM121 therefore inherits every SM120 path unless a gate excludes it by name. [Code-verified]
- **FP8 block.** Any block-FP8 weight whose K-block is not 128 goes to Triton whatever the backend flag (`layers/quantization/fp8_utils.py:590-591`). The auto order is DeepGEMM (if JIT is enabled) → FlashInfer → CUTLASS (SM12x) → Triton (`fp8_utils.py:925-953`). [Code-verified]
- **NVFP4 MoE.** Auto enables the Marlin fallback only for 8.0 ≤ cc < 10.0 (`modelopt_quant.py:2317-2319`), so SM121 runs the native FP4 paths. [Code-verified]
- **The b12x adapter.** On a 32×32-block MXFP8 DeepSeek V4.1 checkpoint, dense projections went from 52 to 17 ms per step once an adapter routed M = 6 to FlashInfer's b12x warp-level MMA kernel. The SGLang backend enum does not expose that kernel (MiaAI-Lab DeepSeek V4.1 README). [Community-measured]
- **Sparse attention (QSA).** The TRT-LLM sparse decode path is admitted on SM100/SM120 but excluded on SM121: `qwen_sparse_attn_backend.py:46-58` says it "silently corrupts long-context decode on SM121/GB10" (PR #36845 found token 0 returned on 32/32 prompts at 120K-210K). SM121 is routed to a dedicated `qwen38_qsa_sm121_varlen` Triton kernel. [Code-verified; Community-measured]
- **KV.** `--kv-cache-dtype nvfp4` is admitted on SM100 or `is_sm120` (major 12, so SM121 is admitted) (`arg_groups/kv_cache_hook.py:121-127`). It measured 29% slower than FP8 KV (§4). [Code-verified; Community-measured]
- **Measured GB10 profiles.** The DeepSeek V4.1 TP3 per-step budget, Flash-Next TP2 at 24.5 ms/token, and a c1 profile with 52% of GPU time in cuBLAS BF16 GEMV (§4). [Community-measured]
- **Known-bad:**
  - `--speculative-attention-mode decode`: 62 → 21 tok/s (§4). [Measured]
  - GLM DSA has no usable backend on SM121 in v0.5.20. tilelang needs 169,984 B of shared memory, and trtllm is SM100-only (#40286). [Community-measured]
  - Overlaying cuBLAS 13.2.2.2 made decode-shaped BF16 GEMMs about 28% slower than the image's cuBLAS 13.0 (#36796). [Community-measured]

### llama.cpp and forks

Mainline pinned at `6e60f35608ec`.

- **Mainline GB10 dispatch** (`ggml/src/ggml-cuda/mmvq.cu`).
  - `ggml_cuda_should_use_mmvq` keeps MMVQ through ne11 ≤ 8 for every type except Q2_K (≤ 6) when `cc == GGML_CUDA_CC_DGX_SPARK` (1210) (`mmvq.cu:348-354`, `common.cuh:61`, `mmvq.cuh:3`).
  - An L2 prefetch (`prefetch.global.L2`) runs for Q4_0, Q5_0, Q8_0, MXFP4, Q3_K-Q6_K, IQ1_M, IQ4_NL and IQ4_XS. It is compiled only when `__CUDA_ARCH__ == 1210` (`mmvq.cu:1-36`), so a build without sm_121 in its arch list silently loses it.
  - nwarps doubles at ncols_dst == 1 only for dense weights. The `has_ids` expert path is excluded (`mmvq.cu:553-576, 1140-1156`). [Code-verified]
- **What those changes measured** (merged PRs):
  - #26079: only Q2_K crosses from MMVQ to MMQ within ne11 ≤ 8 on GB10.
  - #26705: on 27B Q4_K the branchless unpack alone lost 4.63% at ne11 = 5, and the GB10 prefetch recovered it to +8.08%. The author's explanation, paraphrased: the unpack was already hidden behind weight loads, so bandwidth, not ALU work, was the limit. The prefetch also gave +5.47% at ne11 = 1 on 27B Q4_K (14.04 → 14.80).
  - #26843: nwarps = 8 at bs = 1 gave +5.03% (Q4_K_M) and +6.28% (Q8_0).
  - (numbers in §4) [Community-measured]
- **Open GB10 issues, not measured here.** #27918: the VMM pool's 32 GB virtual-address cap blocks full offload of models over 32 GB on unified memory. #27780: a Qwen4Exp graph builder aborts under sustained load on SM121. [Community-measured]
- **antirez/ds4** (`0aaea5a238fb`). A single-model DeepSeek V4 engine with a CUDA back end.
  - Plain decode runs at 85-90% of a 231-234 GB/s read probe. Host `encode` takes about 10 ms of a 55 ms step (#773).
  - Decode-side launch removal and fusions measured flat.
  - DSpark (its speculative path) is the only lever left, and it loses on unpredictable prose.
  - Network TP on two nodes: 21.9 t/s for DeepSeek V4.1 Flash Q2 with SSD streaming (1K prompt, 64K context, 2,048 teacher-forced decode tokens, no speculation). Eight sessions reach about 28 agg, and batching starts at five ready sessions (`docs/DGX_SPARK.md`). [Community-measured]
- **Entrpi/ds4 and Entrpi/ds4-on-spark** (`b0a147a7fba6`, `99fb7b170528`).
  - Per-layer CUDA-graph capture at every depth, head-group flash-decode, continuous batching.
  - DSpark engages only at N = 1 (`DS4_DSPARK_MAX_NLIVE=1`), so concurrent requests run plain batched decode.
  - Plain decode measured 18-20 tok/s against an ~11 GB/token, 20.5 tok/s roofline.
  - Ship decode (speculation armed at every depth) is 1.33-1.47x upstream ds4. Keep that figure separate from the plain one. [Community-measured]
- **Anemll.** `Anemll/dspark-vllm-gx10` is a vLLM 0.25.1 fork, not llama.cpp. Its ds4-derived repos found (`ds4-ssd`, `ds4-qwen`) are Metal-first. No SM121 llama.cpp path from Anemll was found as of 2026-09-23. [Community-measured]
- **Defaults on CC 12.1.** MMVQ for batch ≤ 8 (≤ 6 for Q2_K), MMQ above. The GB10 prefetch and table are present only if the build targets sm_121. llama.cpp DeepSeek V4 slots are a static split of `--ctx-size` (forum 379129). [Code-verified; Community-measured]

### ExLlamaV3 in vLLM

ExLlamaV3 pinned at `6b84a21b6f1e`. EXL3 is absent from upstream vLLM and SGLang. On GB10 it runs only through forks or plugins: vcruz305/vllm-exl3 (release v0.3.1), Zeuss5/cuda-exl3, the MiaAI-Lab kits, and locally derived runtimes. [Code-verified]

- **Small-M GEMV** (`exl3_gemv_kernel.cuh:1-30`, `exl3_gemv.cu:13-74`, `exl3_gemm.cu:227-242`).
  - A QTIP-style GEMV is tried first for m ≤ 8 (`EXL3_GEMV_MAX_M 8`).
  - Warps split k and never synchronize in the main loop, with 16 k-splits (narrow, 512 threads) or 8 (wide, 256 threads).
  - Weights stream to registers with `ld.global.cs` (evict-first) behind a register prefetch ring. There is one m16n8k16 MMA pair per 16×16 tile and an m == 1 fast path.
  - The header says "Ada/Blackwell are memory-bound here and keep the regular kernel", but the arch gate is commented out (`//if (cc != CC_AMPERE) return -1;`, `exl3_gemv.cu:53`), and major ≥ 10 maps to CC_BLACKWELL (`exl3_devctx.cu:39`). The heuristic therefore runs on GB10. Its envelope was measured on RTX 3090 and Ada, never on GB10. [Code-verified]
- **INT8-activation GEMV for mul1 tensors** (`exl3_gemv_int8.cu:1-52`).
  - With `EXL3_INT8_GEMV` unset, the default is mode 2 (plain int8, documented "~0.9% output RMS deviation").
  - The maximum K is 6 on Hopper/Blackwell, else 5. Thresholds were measured on 3090/4090/H200/5090, not GB10.
  - The call-site comment (`exl3_gemm.cu:185-188`) calls the path experimental and not graph-capturable. Yet `exl3_gemv_int8.cu` (≈223-231, 353-375) records graph parameters and sizes its workspace because "the pointer is baked as a kernel argument into captured CUDA graphs". Comment and code disagree; verify at runtime. [Code-verified]
  - An own component gate rejected mode 2 at single-token decode for partition non-equivalence (~0.7-0.8% relative L2) (2026-09-16, "DS V4.1 EXL3 TP3 component gates"). [Measured]
- **Fused decode MoE ("coop").** For bsz 1..8 (`exl3_moe_coop_kernel.cuh:1-45`, `libtorch/mlp.h`), an expert "run" groups up to `ROWS = 8` slots as the rows of one m16 MMA, so each expert is read once per run. It uses k-split `WK = 16`, a prefetch ring `PF = 4`, and `MAX_BSZN = 8`. [Code-verified]
- **Community GB10 serving results** (numbers in §4).
  - vcruz305/vllm-exl3 v0.3.0 GB10 rewrite: in-register trellis dequant and fused MoE decode.
  - Zeuss5/cuda-exl3: expert and sparse-MLA kernel fractions of a 241 GB/s ruler; FP8 KV +1.5-1.9x inside its sparse-MLA kernel.
  - sfxnz DeepSeek V4.1 EXL3 on two nodes: one MUL1 + `cb=2` arm lost prose decode (median 23.52 vs 27.98 MCG, 3 runs each), and a MUL1 pack lost prose at every bit-width tested. [Community-measured]
- **Own.** The locally derived cooperative MoE beats stock by 0.107 ms/layer at 1 row and ties at 64 rows. The 1-row gap is host path, not bytes (§4). A staged kernel-lab tactic sweep (tile shapes × SM quotas, M = 1..128) has not run: "No speedup is claimed until the checked-in sweep runs on SM121." [Historical diagnostic; Measured]
- **Defaults on CC 12.1.** The GEMV heuristic, INT8 mode 2 for mul1 tensors, and coop MoE for ≤ 8 rows are all active, and none were tuned on GB10. [Code-verified]

### TensorRT-LLM (status only)

The only GB10 decode figures found are in an NVIDIA technical blog (post dated 2026-03-16).
- Table 3: Llama 3.3 70B Instruct NVFP4 on TensorRT-LLM, 32K input, 1K output, batch 1: TPOT 269 / 133 / 72 ms on one, two and four DGX Spark nodes.
- Nemotron-3-Super-120B NVFP4: 18 tok/s generation at 128K/1K.

No SM121-specific TensorRT-LLM decode-kernel selection or regression report was found. [Community-measured, vendor-published]

## 3. What is different on GB10 / SM121

- [Community-measured] **The ruler is about 230-245 GB/s, not 273.**
  - Simple read probes cluster at 227-234 GB/s: a float4 read-sum over 8 GiB (antirez/ds4 #773), Entrpi `bw_bench.cu`, and forum 363238 128 MB copy/triad.
  - NNNtrance's bf16 read with four rotating buffers at 8× L2 has a 245 GB/s median, and its fp4-crossover re-run measured the ruler three ways at 238.6 / 245.9 / 239.5 GB/s.
  - The full traceable span runs from about 215 GB/s (copy) or 220 GB/s (read minimum) to 249.5 GB/s. On eight Founders Edition nodes, GEMV probes gave 216-235 GB/s (tonyd2wild #1 comments).
  - A "218 GB/s" figure in circulation is second-hand. An own estimator's "236 GB/s" has no script behind it.
  - Divide by about 230-245 by probe, and say which probe.
  - One summary for every page: achievable bandwidth is about 215-245 GB/s depending on the probe (contested; single-probe figures per tool family: saturating read 227-235, rotating-buffer read 238.6-245.9, copy 215-229, STREAM triad 200-214 GB/s [Community-measured]); the decode ruler is 230-245 GB/s. The "achievable" figure is contested between probe families, so quote the probe with the number.
- [Community-measured] **The GPU L2 is 24 MB** (`deviceQuery`: 25,165,824 bytes; NNNtrance sizes its weight rotation at 8× L2). Chips and Cheese reports a separate 16 MB system-level cache behind the CPU clusters. No NVIDIA document stating the GPU L2 was found.
  - [Interpretation] A decode layer at 1-2 rows whose expert slice fits in L2 can show "implied bandwidth" above the ruler. Bandwidth probes must rotate weight sets larger than L2.
- [Community-measured] **Unified memory is shared with everything else.** While Qwen3.5-35B-A3B BF16 decoded on one node, an 8 MB probe's copy bandwidth fell from 190.9 to 86.8 GB/s and triad from 96.8 to 39.1 GB/s (forum 363238). The probe's working set is L2-sized. A probe run during serving cannot measure the ceiling.
- [Community-measured] **A firmware-dependent slow state.** On one mixed four-node cluster (older BIOS `0ACUM018`, drivers 580.142 / 580.159), a decode-shaped bf16 GEMV ran at 66-80 GB/s instead of 224-233, and a serving step took 94 ms instead of 63.
  - Device copy (239-242 GB/s) and the reported SM clock did not change, and throttle reasons read 0x0.
  - The state was not reproduced on eight nodes with BIOS `0ACUM027`, driver 580.173.02 and kernel build 1032 (0 slow seconds in 9 runs), nor by the reporter after a power cycle (117.6 vs 118.1 tok/s after 45 s idle).
  - Root cause is open (tonyd2wild #1). Treat it as a confounder for short tests on older firmware, not as something that affects every GB10 number.
- [Code-verified] **SM121 gets SM120 paths, never SM100 ones.**
  - vLLM gates by family 120, and SGLang by major 12 (`is_sm120_supported`).
  - FlashInfer CuTe-DSL, TRT-LLM MoE and FMHA, and CuTe-DSL NVFP4 GEMM/MoE are family-100-only and unavailable.
  - Exclusions by name exist: SGLang drops the TRT-LLM QSA path on SM121.
- [Community-measured] **About 99 KB of shared memory per block** (101,376 B available) breaks kernels tuned for other Blackwell parts. Triton MLA decode with FP8 KV needed 102,400 B (vLLM #53748, fix PR #54013 open), and tilelang DSA wants 169,984 B (SGLang #40286). See [Attention backends on SM121](06-attention-backends-sm121.md).
- [Community-measured] **FP4 tensor cores buy nothing at decode shapes.** At M = 8 / 64 / 128 / 256, even a zero-overhead FP4 GEMM measured 1.03-1.07x slower than Marlin W4A16. A custom FP4 kernel could win at most 4-9%. FP4 overtakes Marlin only from M = 1,024 with all experts local, a shape EP3 decode never runs (NNNtrance fp4-crossover sweep, pre-registered design).
- [Measured] **48 SMs, so stock launch tables oversubscribe.** A GB10-tuned sparse-attention launch table halved the kernel at 8 rows (60.7 → 33.0 µs at 32K). The stock split-K table launched about 5x more CTAs than there are SMs. The serving effect was nil at 1-5 streams, because the kernel was a small share of an ~18 ms step (2026-09-05, "FP8-KV qualification").
- [Code-verified] **llama.cpp carries a GB10-specific MMVQ table and an L2 prefetch** keyed to cc 1210 (§2). No other engine in scope has a GB10-specific GEMV table.
- [Community-measured] **Collectives are latency-bound at decode sizes.**
  - A three-node ring: 72-85 µs from 8 B to 32 KiB, 86.4 µs at 64 KiB, 172.5 µs at 128 KiB.
  - 16 MiB reaches 98.1% of a 20.8 GB/s per-pair wire, which is limited by PCIe Gen5 x4 per NIC (about 15 GB/s per card), not by the cable (NNNtrance docs/10).
  - At 88-108 collectives per step, that is 5-9 ms of pure latency per step at TP3/TP4 that no weight format removes. [Interpretation on Community-measured] See [Inter-Spark communication](03-inter-spark-communication.md).
- [Community-measured; Measured] **The GPU clock is not a decode lever on bandwidth-bound steps.** Under a 2,100-2,200 MHz cap, own TP3 C1 decode means were 5-6% lower, inside rep spread (n=5, first requests excluded); community controlled runs were within ±2%. Neither zero nor a ~6% loss is excluded. See [Clocks, thermal, and power](13-clocks-thermal-power.md) and §4.
- [Community-measured] **The host is a Grace CPU on the same memory.** Host-side graph encode was about 18% of a single-node ds4 step (#773). Worker reads of unpacked Engram tables over NFS made rank 0 wait about 11 ms per step in the MiaAI-Lab DeepSeek V4.1 TP3 kit.
- [Interpretation] **Uniform-routing expert bytes over-count on GB10 MoE.** Two independent datasets find more expert bytes per step than the ruler allows.
  - NNNtrance TP3 NVFP4: 13.1 GB at M = 8 and 49.2 GB at M = 64 against 40 and 170 ms of MoE time, i.e. 325-331 and ~289 GB/s.
  - Own EXL3 TP3 per-layer logs: 172-351 GB/s per rank.
  - Correlated routing among a sequence's draft rows (fewer distinct experts) and L2 reuse are the consistent explanations. Neither is proven without a routing histogram and a DRAM byte counter (§8).

## 4. What has been measured

Own rows cite a packet by date and title. Community rows cite a handle and link, and §9 carries the pins. Percentages of bandwidth name their denominator.

### Step anatomy and bytes anchors

| change | baseline -> result | workload | topology | engine | grade | source |
|---|---|---|---|---|---|---|
| none: step time vs accepted length, Qwen 3.8 27B dense hybrid NVFP4, DFlash2 k=7 | verify step 138 ± 3 ms (133.9-142.5) at accept lengths 1.55 to 7.95: flat, so bytes set the step, not acceptance | live decode, 27K-84K depth | 1 node | SGLang | [Measured] | 2026-08-19, "single-node uncensored DFlash2 campaign audit" |
| bytes per verify step, same lane, computed from loaded tensors | uniform NVFP4 20.09 GB (MLP 9.63, linear-attention 3.13, full attention 0.94, lm_head 2.54, drafter 3.85) → 206.5 GB/s implied (76% of nominal); FP8-block target 30.77 GB | same log | 1 node | SGLang | [Interpretation] on [Measured] step | same packet |
| KV term at depth, same lane | 43,008 B/token FP8 (target 32,768 + drafter 10,240) → 3.61 GB at 84K, +13 ms on a ~131 ms base; BF16 KV would add ~13% step time at 84K and ~40% at 262K | arithmetic on the measured allocator log | 1 node | SGLang | [Interpretation] | same packet |
| no speculation (autoregressive floor), Qwen 3.8 27B | NVFP4 ~10.1 tok/s (one 256-token request, elapsed incl. TTFT); FP8 block 7.84 tok/s (net decode, 3 reps) | short prompt | 1 node | SGLang / vLLM | [Measured] | 2026-08-19 no-spec control; 2026-08-21 "FP8 MTP sweep" |
| decode vs depth, dense 27B NVFP4 + MTP k=3 vs 3B-active MoE | dense 31 tok/s @13.5K → 18.8 @130K, 12.7 @167K, 10.6 @250K, 6.6 @376K, 5.9 @501K; Nemotron 3.5 Lightning 30B-A3B 127 @13.5K → 112 @130K | synthetic fixtures shaped from mined agent traffic | 1 node | vLLM (SGLang at 167K) | [Measured] | 2026-08-16, "fleet serving bake-off" |
| no speculation, GLM 5.3 Flash NVFP4 MoE | 14.1-14.2 tok/s on every content class (content-flat, as a bytes-bound step should be) | forced-512 prose / coding, 3 reps | TP2 | vLLM (Marlin NVFP4 MoE, FP8 KV) | [Measured] | 2026-08-27, "GLM 5.3 Flash MTP-4 versus MTP-off" |
| profiled step, GLM-5.3-Flash NVFP4, Marlin W4A16 experts, BF16 dense, DFlash2 k=7 | C1 98.9 ms: BF16 dense GEMM 49.4% (~51 ms, 5.90 GB/rank streamed), Marlin MoE 38.4-39.1%, NCCL 13.8-15.1% (108 collectives), KDA/GDN 2.5%, MLA 1.2%, sampling + spec bookkeeping 0.07 ms, CPU gap 1.9-2.4%. C8 260.4 ms: dense 17.5%, MoE 62.4%, NCCL 9.2%, KDA/GDN 7.2%, MLA 0.95-0.99% | realistic code prompts, rank spread ≤ 0.8 points; raw not published | TP3+EP (96 of 288 experts per rank) | vLLM fork | [Community-measured] | NNNtrance, [docs/11](https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark) |
| profiled step, DeepSeek V4.1 Flash, native MXFP4 experts, FP8 dense, DSpark | ~82 ms (from ~118 ms earlier): dense FP8 projections ~17 ms (52 on Triton), MoE grouped FP4 ~27, bf16 `wo_a` einsum + lm_head ~13, NCCL (104 collectives at 50-100 µs) ~13-16, ~2,000 small kernels ~10, Engram host callbacks ~1-3; MLA KV 1,670.75 B/token/rank FP8, replicated | batch 1, 6-token verify, torch profiler on 3 ranks, 42 steps | TP3 | SGLang | [Community-measured] | MiaAI-Lab, [DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) |
| profiled step, Qwen3.8-Flash-Next | 24.5 ms/token: NVFP4 GEMM 34.0%, grouped MoE 16.1%, hc_mix 11.7%, NCCL 7.2% | serving profile | TP2 | SGLang | [Community-measured] | [sglang #36796](https://github.com/sgl-project/sglang/issues/36796) |
| profiled step, c1 | 52% of GPU time in cuBLAS BF16 GEMV on non-expert weights; a 3.7x faster QSA kernel left c1 unchanged | c1 decode | TP2 | SGLang | [Community-measured] | [sglang #36558](https://github.com/sgl-project/sglang/issues/36558) |
| profiled step, Qwen3.5-35B-A3B BF16, eager | 33.7 ms: CUTLASS WMMA bf16 GEMM 55.9%, fused MoE 28.0%, GDN 5.1%, overhead 11.0%; poster's 178 GB/s is step-derived, not counted | 10 steps, eager mode | 1 node | vLLM | [Community-measured] | [NVIDIA forum 363238](https://forums.developer.nvidia.com/t/363238) |
| CUDA decode, DeepSeek V4 Flash Q2 GGUF | `execute` 45.0-45.7 ms (85-90% of a 231-234 GB/s probe at 10-13 GB/token) + host `encode` 9.8-10.6 ms; llama.cpp UD-Q2_K_XL on the same box 17.50 ± 0.10 t/s | steady-state decode | 1 node | antirez/ds4 | [Community-measured] | vincenzopalazzo, [antirez/ds4 #773](https://github.com/antirez/ds4/issues/773) |
| MLA decode vs depth, same model family | 18.05 / 15.10 / 14.43 / 13.84 t/s at 2K / 16K / 32K / 64K (−23%); llama.cpp UD-IQ2_M 19.09 → 18.15 at 16K | ds4 recorded baseline; forum run | 1 node | antirez/ds4; llama.cpp | [Community-measured] | antirez/ds4 `docs/PERFORMANCE.md`; [forum 379129](https://forums.developer.nvidia.com/t/379129) |
| dense GGUF at batch 1 | Llama-3.1-8B Q4_K 50 t/s; Qwen3.6-27B Q4_K 14 / Q2_K 24 t/s → ≈200-211 GB/s implied, 82-86% of a 245 GB/s ruler (input embedding excluded, LM head included) | `llama-bench -p 1 -n 0 -embd 1 -r 50`, --pure GGUFs, ABBA, 4 runs, 2400 MHz | 1 node | llama.cpp | [Community-measured]; GB/s [Interpretation] | [llama.cpp PR #26079](https://github.com/ggml-org/llama.cpp/pull/26079) |
| dense 27B by engine and format | llama.cpp b10423 UD-Q4_K_XL (16.68 GiB) 11.6 t/s; vLLM FP8 8.2 (author: 28.75 GB/token); vLLM NVFP4 plain 11.5 → ≈200-208 GB/s effective | tg128 / tg32 at depth 0; llama-benchy pp2048/tg128 c1 | 1 node | llama.cpp; vLLM | [Community-measured]; GB/s [Interpretation] | kubesimplify, [Qwen3.8-27B on DGX Spark](https://blog.kubesimplify.com/qwen3-8-27b-on-dgx-spark) |
| dense autoregressive floors vs bytes model (245 GB/s ruler) | Llama-3.3-70B NVFP4 4.51 tok/s (model predicts ~6.0; 75%); Gemma-3-27B BF16 4.03 (predicts 4.5; ~89%); Nemotron Super 49B NVFP4 5.79 (predicts 9.8 at the page's own 25 GB/token; ~59%) | NTTPC: `vllm:25.12-py3`, c1, 1024/1024, ignore-eos; dendro-logic: ~1.5K in / 400 out | 1 node | vLLM / NIM | [Community-measured]; predictions [Interpretation] | [NTTPC benchmark32](https://www.nttpc.co.jp/gpu/article/benchmark32.html); [dendro-logic](https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark) |
| cooperative vs stock EXL3 MoE, one real-weight layer per rank | 1 row 0.194 vs 0.301 ms; 8 rows 0.966 vs 1.069; 32 rows 3.183 vs 3.486; 64 rows 5.025 vs 4.988. Concentrated routing: stock steps at ceil(rows/16), coop at ceil(rows/8). The 1-row gap (0.107 ms/layer, ~4.5 ms/step over 42 layers) is host path, not bytes | CUDA-graph replay medians, worst of 3 ranks | TP3/EP3 | vLLM + EXL3 | [Historical diagnostic] | 2026-09-20, "TP3 frontier dossier v2" (re-reading 2026-09-16 bundle logs) |
| implied expert bandwidth per rank, same logs | 172-351 GB/s; above nominal on one rank at 1-2 rows, so the read-once byte model fails there (L2 reuse or overcounted experts) | derived | TP3/EP3 | vLLM + EXL3 | [Historical diagnostic] + [Interpretation] | same dossier |
| three-host all-reduce | 32 MiB ≈ 3.5 ms on rank 0 (12 reps) | component gate | 3 nodes | NCCL over RoCE | [Measured] | 2026-09-16, "DS V4.1 EXL3 TP3 component gates" |
| all-reduce latency | 72-85 µs from 8 B to 32 KiB; 86.4 µs at 64 KiB; 172.5 µs at 128 KiB; 16 MiB at 98.1% of 20.8 GB/s per pair. Four-node ring: p50 60 µs at 60 KB × 88 per step ≈ 5.3 ms | NCCL_ALGO=Ring, 8 channels, microbench | 3-node and 4-node rings | NCCL | [Community-measured] | NNNtrance docs/10 §4; [tonyd2wild #1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/1) |
| KDA dense GEMM standalone vs in-engine | 214 vs 150 GB/s (30% lower in-engine; the source calls it a 37% gap); alignment explains 3-5%; cause not found; raw not published | same shapes | 1 GPU | vLLM fork | [Community-measured] | NNNtrance docs/11 §7.1 |
| small-M kernels as a fraction of a measured ruler | Marlin W4A16 MoE 91-96% of 245.9 GB/s (M = 8-256, model-free); cuda-exl3 MoE 81% at M = 1 and 90-91% at M = 8-128 of 241; cuda-exl3 sparse-MLA 86-91% at B ≥ 16 and 95-96% at B = 64-512 of 241; INT8 Triton LM-head GEMV 85-88% of 273 nominal; BF16 `F.linear` LM head on vLLM 0.27.1 64% (batch 1) / 86% (batch 2-4) of 273 nominal | model-free kernel benches; no `ncu` byte counter in any | 1 GPU | vLLM kernels; cuda-exl3 | [Community-measured] | NNNtrance fp4-crossover; [Zeuss5/cuda-exl3](https://github.com/Zeuss5/cuda-exl3); [albond](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4) |

### Levers

| change | baseline -> result | workload | topology | engine | grade | source |
|---|---|---|---|---|---|---|
| MTP off → MTP-4, GLM 5.3 Flash NVFP4 | prose 14.21 → 21.82; coding 14.11 → 27.49; C4 agg 38.64 → 63.76; C8 agg 59.23 → 95.55 | forced-512, 3 reps; identical coding at C1/C4/C8 | TP2 | vLLM | [Measured] | 2026-08-27, "GLM 5.3 Flash MTP-4 versus MTP-off" |
| draft depth k=3 → k=7, GLM 5.3 Flash EXL3, DFlash2 | structured 35.97 → 59.70 (18 and the ledger read the k=7 cell as 60.86; unresolved); prose 22.11 → 18.60; coding 24.74 → 20.66; 64K 25.56 → 19.54; 120K 22.24 → 18.33; per-draft-token acceptance at k=7 0.28-0.37 on real-shaped work vs 0.47-0.59 at k=3 | forced-length fixtures, acceptance from counter deltas | TP2 | vLLM + EXL3 | [Measured] | 2026-09-03, "A1 k=3 / k=7 qualification" |
| adaptive k-set {2,4,7} / {3,5,7} / {2,4,6}, GLM EXL3 | 64K 36.6 / 31.9 / 30.9; structured 88.5 / 87.1 / 82.8 | 3 reps; confounded by a graph-capture / MoE dispatch-policy coupling (§7) | TP3 | vLLM + EXL3 | [Measured] (confounded) | 2026-09-15, "TP3 upgrade A/B" |
| DSpark k=5 → k=3, DeepSeek V4.1 Flash | code 57.92 → 50.39; prose 29.77 → 32.29; C4 agg 66.05 → 71.18; C8 agg 68.15 → 79.55 | forced-512, ignore_eos, 3 reps | TP3 | SGLang | [Measured] | 2026-09-15, "fast profile k=3 scorecard" |
| DFlash2 k=8 → k=10 (1 node) and k=8 → k=16 (TP2), Qwen 3.8 27B | 1 node: code +14.6%, prose +7.0%; TP2: code 94.5-96.6 → 111.3-111.5, prose 46.3-46.7 → 43.0 | matched-depth fixtures; net decode code / prose | 1 node; TP2 | SGLang; vLLM | [Measured] | 2026-08-28, "fastest one-GB10 recipe audit"; 2026-08-21 comparison rows |
| k=14 vs k=7 at the same batch budget, Qwen3.8-27B NVFP4 | −4.6% at c8 | distinct prompts, 1,500 output tokens | 1 node | vLLM 0.27.1 | [Community-measured] | 0xBakeer RESULTS.md |
| draft sampling probabilistic → greedy, DeepSeek V4 Flash | 8K 45.7 → 42.2; 32K 47.4 → 41.0 | recipe benchmark, single runs | TP2 | vLLM (DSpark k=5) | [Measured] | 2026-08-16, "decode tuning A/B" |
| draft vocabulary full 248K → 64K list, Qwen3.8-Flash-Next | draft head 0.59 → 0.16 GiB per rank; prose +14.1%, coding +12.8%, C1 +18.5%, C4/C5 +4-5%; acceptance 58.6% → 52.2%. Single node, same lever: +10-23%. An 8K list dropped acceptance to 36% and lost everywhere | frozen C1 3 runs; standardized suite | TP2+EP; 1 node | vLLM (MTP4, FP8 KV) | [Measured] | 2026-09-05, "FP8-KV qualification"; 2026-09-05 single-node draft-vocabulary screen |
| drafter BF16 → calibrated NVFP4, Qwen 3.8 27B | code 56.62 → 60.44 (+6.7%); prose 26.00 → 27.07 (+4.1%) | forced-512, 3 reps; screen, not repeated | 1 node | SGLang | [Measured] | 2026-09-08, "K10 screen" |
| drafter parallel config: DCP not propagated → fixed, GLM-5.2 NVFP4 MTP3 | per-position acceptance 0.72 / 0.30 / 0.07 → 0.90 / 0.79 / 0.67; hot decode 14.6 → 22.0-23.0 tok/s | fork PR, open | 4 nodes, TP4/DCP4 | vLLM fork | [Community-measured] | [local-inference-lab/vllm PR #72](https://github.com/local-inference-lab/vllm/pull/72) |
| dense weights BF16 → weight-only FP8, GLM 5.3 Flash EXL3 | decode +7-14% at short depth (+1% at 120K) (structured 61.9 → 66.2, coding 27.1 → 30.8); prefill −18 to −21% (BF16 is 22-26% faster) (1,420 → 1,142 at 100K); weights 2.4 GiB smaller | MiaAI-Lab lab-bench fixtures 5 × 400, others 3 × 512, temp 0, cold nonce | TP2 | vLLM + EXL3 (FP8 Marlin) | [Measured] | 2026-09-08, "E4 A/B" |
| dense path BF16 → 4-bit EXL3 (head 6-bit), GLM-5.3-Flash 4.05 bpw | step 88.2 → 70.3 ms; C1 62.39 → 75.91 (+21.7%); 8-way agg 175.37 → 197.20 (+12.5%); DFlash on (acceptance 61.9% vs 64.4%; tokens per step −3%); 64 heads padded to 66 | same day, MMLU equal | TP3 | vLLM + cuda-exl3 | [Community-measured] | measured by @NNNtrance in [Zeuss5/cuda-exl3](https://github.com/Zeuss5/cuda-exl3) |
| expert format EXL3 2.9 bpw vs native MXFP4/FP8, DeepSeek V4.1 Flash | decode code +13%, prose +32%; cold prefill −28 to −44%; runtime and draft depth also differ | forced-512, 3 reps | TP3 | vLLM EXL3 port vs SGLang | [Measured] (lane comparison) | 2026-09-16, "EXL3 TP3 short-context optimization" |
| weight format ladder, Qwen3-Coder-30B-A3B | INT4 W4A16 Marlin 86.9; INT4 W4A8 Marlin FP8-MMA 86.7; NVFP4 FlashInfer CUTLASS 65.0; FP8 dynamic Triton MoE 50.5; BF16 30.6 tok/s | bench.py long-generation class (~400 output tokens), context not varied | 1 node | vLLM (vllm-next, 26.01 base) | [Community-measured] | [flash7777/vllm-marlin-sm12x](https://github.com/flash7777/vllm-marlin-sm12x) |
| 4-bit vs FP8 weights, Qwen3.8-27B, by concurrency | 4-bit advantage +27% (c1) → +20% (c4) → +10% (c8) → +0.2% (c16) | distinct prompts, 1,500 output tokens, DSpark k=7 | 1 node | vLLM 0.27.1 | [Community-measured] | 0xBakeer RESULTS.md |
| kernel backend: b12x NVFP4 GEMM + FlashInfer autotune, dense 27B | decode 31 → 22 tok/s at 13.5K; fresh autotune draws later gave no repeatable C1 gain | agent-shaped fixtures | 1 node | vLLM | [Measured] | 2026-08-16, "fleet serving bake-off"; 2026-09-08, "K10 screen" |
| EXL3 GB10 rewrite (in-register trellis dequant, fused MoE decode), GLM-5.3-Flash 2-bit K2 | 40 routed-MoE layers 19.9 → 11.5 ms/token (497 → 288 µs per layer); step 59.2 → 40.6 ms; 16.9 → 24.6 tok/s average | 5-category average; speculation state not stated | 1 node | vcruz305/vllm-exl3 v0.3.0 | [Community-measured] | [@ViC305 on X](https://x.com/ViC305/status/2095559790902890679) |
| TP1 → TP2, dense Qwen 3.8 27B NVFP4 + MTP k=3 | +60-73% decode: 21.4 → 34.2 @1.8K; 19.8 → 31.9-32.3 @13.5K; 17.7 → 30.7 @83.7K (TP2 flat 30.3-34.2 from 2K to 84K) | identical battery | TP1 vs TP2 | vLLM | [Measured] | 2026-08-17, "TP1 vs TP2 identical battery" |
| TP1 → TP2 → TP4, Qwen3.6-27B-NVFP4 | 12.63 → 22.57 (1.79x) → 33.11 tok/s (2.62x); ITL 77.97 / 43.48 / 29.14 ms, i.e. +4.5 ms (TP2) and +9.6 ms (TP4) over ideal halving | CordatusAI llm-benchmark, ~128 in / 128 out, C = 1, 10 rounds, identical flags | 1 / 2 / 4 nodes | vLLM nightly | [Community-measured] | [Openzeka](https://whitepapers.openzeka.com/papers/qwen3.6-27b-dgx-spark-scaling) |
| 1 → 2 → 4 nodes, Llama 3.3 70B Instruct NVFP4 | TPOT 269 → 133 → 72 ms (2.02x, 3.74x) | 32K in / 1K out, batch 1 | 1 / 2 / 4 nodes | TensorRT-LLM | [Community-measured] (vendor-published) | [NVIDIA technical blog](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark) |
| TP2 → TP3, GLM 5.3 Flash EXL3 | coding +44.7% (30.8 → 44.58); prose +28.9%; C4 agg 54.1 → 76.9 (+42%); decode at ~1M depth 41.8 | forced-512, ignore_eos | TP2 → TP3/EP3 | vLLM + EXL3 | [Measured] | 2026-09-14, "TP3 triangle qualification" |
| TP3 → TP4, DeepSeek V4.1 Flash | prose C1 37.9 → 45.4 (+20% for +33% nodes); attributed to smaller attention GEMMs and no padded shards (rank 2's shards were all padding at TP3) | prose, 256 tokens, thinking off | TP3 → TP4 | SGLang | [Community-measured] | MiaAI-Lab DeepSeek V4.1 README |
| TP1 → TP2, small-active MoE, Qwen3-Coder-30B-A3B INT4 | 86.9 → 91.6 tok/s over RoCE (+5%); 45.3 over TCP sockets | bench.py long class; TP2 ctx = 0 | 1 → 2 nodes | vLLM | [Community-measured] | flash7777 RESULTS.md |
| decoder bundle including cooperative decode MoE, 25 captured row shapes, 8 slots and more FP8 (the fullest component list is on [08](08-moe-dispatch.md) §4; pages record the change count as four to seven, and the packet count is unresolved), GLM 5.3 Flash EXL3 | coding 41.46 → 48.88 (+17.9%); prose 32.11 → 38.80 (+20.8%); C8 agg 76.41 → 123.06 (+61%); C8 per-stream 21.58 → 17.80 (−17.5%); C8 TTFT 10.63 → 0.52 s; attribution to components impossible | exact-body pairs × 5; quick suite | TP3/EP3 | vLLM + EXL3 | [Measured] (bundle) | 2026-09-16, "combined runtime qualification" |
| fused input/route staging + shared gate/up rotation (component wins), GLM EXL3 | end-to-end coding 44.29 → 43.93 (−0.8%); C8 −5.5%; excluded | frozen core suite | TP3/EP3 | vLLM + EXL3 | [Measured] | 2026-09-20, "next-batch optimization" |
| decode-side launch removal and fusions, all bit-identical | flat; shared-memory staging −3.6% | steady-state decode | 1 node | antirez/ds4 | [Community-measured] | #773 |
| `--enforce-eager` → CUDA graphs | NVFP4 35B-A3B MoE 23.4 → 66.9 tok/s; dense Qwen3-8B 39.62 eager vs 38.59 graphed | single stream | 1 node | vLLM | [Community-measured] | ai-muninn (EN post) |
| graphs already on: residual host gap | GPU occupancy 98.1-99.2%, CPU gap 0.85-2.4%; closed as a lever | C1 profile | TP3+EP | vLLM fork | [Community-measured] | NNNtrance docs/11 |
| eager + MTP-1 vs graphs, Hy3-295B NVFP4 with Marlin | eager 21.8 tok/s vs graphs 15.5-16.3 ("measured twice"); opposite of the single-node MoE result; unexplained | single stream | TP2 | vLLM | [Community-measured] | [tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark](https://github.com/tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark) |
| LM head BF16 → INT8 Triton GEMV, Qwen3.5-122B (248320 × 3072 head, 1.53 GB BF16, 729 MB INT8) | vLLM 0.19: +33% end to end; vLLM 0.27.1: +6.8% c1, +1.1% c4, +4.8% at 16K/c1; TPOT −1.6 ms where the kernel predicts −5.6 ms (author suspects CUDA-graph batch padding) | 1024/512 c1 and c4; 16K/1024 c1 | 1 node | vLLM | [Community-measured] | albond |
| sampler and speculation bookkeeping | 0.07 ms/step at C1, 0.15 ms at C8; a fused Markov argmax (DSpark tree on vLLM 0.21.1) C1-neutral, C4 +5-14% | GLM TP3 profile; DeepSeek V4 campaign | TP3; TP2 | vLLM fork; vLLM | [Community-measured] | NNNtrance docs/11; tonyd2wild campaign (own audit of it) |
| `NCCL_MAX_NCHANNELS=8` kept vs dropped | +7.7% at C4 (99.5 → 107.2) and +9.7% at C6 (115.5 → 126.7); nil at C1 | same-session A/B | TP3+EP | vLLM fork | [Community-measured] | NNNtrance docs/10 §3 |
| fabric ConnectX-7 vs 10GbE, DeepSeek V4 Flash | 17.1 vs 9.2 tok/s | unpredictable (random) prompt | 2 nodes | vLLM | [Community-measured] | [XDA](https://www.xda-developers.com/running-284-billion-parameter-model-two-machines-matches-cloud) |
| GPU clock cap 2,100-2,200 MHz vs stock | own GLM EXL3 TP3: C1 means 36.0 stock vs 33.7 / 33.9 / 34.25 at 2,200 / 2,100 / 1,900 MHz (first requests excluded; −4.9% to −6.4%, inside the stock rep spread 33.9-38.2, n=5), C8 agg 100-111 at every cap; NNNtrance GLM TP3 lock at 3,003 MHz (never reached): C1 +1.2% (±4), C8 −1.9% (±3), 0 throttle in 1,478 samples; forum 380340 2,200 vs 2,411 MHz: N=1 −1.2%, N=8 +2.3%; own DeepSeek V4 Flash TP2 at 2,200: 37.0 → 34.5 (−6.8%, n = 3 paired calls, unreplicated) | own TP3: 5 x C1 forced-512 and 2 x C8 per arm; own TP2: 3 paired decode calls; NNNtrance: 1,478 telemetry samples, repetitions per its docs/12; forum 380340: N=1 and N=8 per its thread | TP2-TP3 | vLLM | [Measured]; [Community-measured] | 2026-09-21, "GPU clock-cap sweep"; 2026-08-16, "clock cap A/B"; NNNtrance docs/12; [forum 380340](https://forums.developer.nvidia.com/t/380340) |
| fair mixed-prefill scheduler vs skip, GLM EXL3, 4 slots (see [22](22-serving-agent-workloads.md)) | C8 agg 68.4 → 77.0; 2K newcomer TTFT 130 → 6 s; incumbents 30% slower only while the newcomer's chunks run | 3 reps | TP3/EP3 | vLLM + EXL3 | [Measured] | 2026-09-15, "TP3 upgrade A/B" |
| slots 4 → 8 (with chunk 256 and a 2 Mi pool), DeepSeek V4.1 Flash k=5 | C8 agg 68 → 89 (+30.8%); code 57.28, prose 29.96 per stream | forced-512 | TP3 | SGLang | [Measured] | 2026-09-14, "DS V4.1 TP3 quick scorecard" |
| prefill chunk size (max batched tokens) and decode | GLM TP3 at 7,168 / 4,096 / 2,048: decode unchanged (prose 39.2 / 37.4 / 39.6); Flash-Next 1 node 2,048 → 4,096 unchanged; DeepSeek V4 Flash TP2 8,192 → 10,240: 256-token decode 50.6 → 42.5 | per packet | TP3; 1 node; TP2 | vLLM | [Measured]; [Historical diagnostic] (TP2 row) | 2026-09-15 "TP3 upgrade A/B"; 2026-09-09 single-node tuning; 2026-08-30 "dual-HCA fabric enablement" |
| GB10-tuned sparse-attention launch table | kernel at 8 rows 60.7 → 33.0 µs; serving C1 +2.4%, C4 −0.2%, C5 −0.9% (nil); step ~18 ms at C1 | micro-bench, then standardized suite | TP2+EP | vLLM | [Measured] | 2026-09-05, "FP8-KV qualification" |
| KV dtype FP8 vs NVFP4 vs BF16, Qwen3.8-Flash-Next | NVFP4 KV 44.0 vs fp8_e4m3 56.8-58.6 (BF16 54-59) tok/s | NEXTN 3/1/4, n = 7-10 medians, 400-token runs, image with `NVFP4_KV_CACHE=1` | TP2 | SGLang | [Community-measured] | [sglang #36797](https://github.com/sgl-project/sglang/issues/36797) |
| KV FP8 vs BF16 inside a sparse-MLA kernel | +1.5-1.9x | kernel bench | 1 GPU | cuda-exl3 | [Community-measured] | Zeuss5/cuda-exl3 GB10 doc |
| content class through acceptance | DeepSeek V4 Flash code 69.76 vs prose 38.07 at 256-token prompts (accept length 4.4-5.2 vs 2.4-2.7); GLM EXL3 TP3 counting 101.7 at 95.9% draft acceptance vs prose 44.2 at 49.6% | forced-512, 3 reps; bench_decode 5 × 400 | TP2; TP3 | vLLM; vLLM + EXL3 | [Measured] | 2026-08-20, "speed headroom campaign"; 2026-09-16, "combined runtime qualification" |
| checkpoint with a larger vision path, DeepSeek V4 Flash | code 69.8 / 65.5 / 62.1 → 55.7 / 53.9 / 51.3; prose −17 to −21%; acceptance 28.7% at k=6 | forced-512, median of 3 | TP2 | vLLM | [Measured] | 2026-09-03, "Vision-Exp gates baseline qualification" |

### Batch scaling

| change | baseline -> result | workload | topology | engine | grade | source |
|---|---|---|---|---|---|---|
| C1 → C5, dense Qwen 3.8 27B NVFP4, DFlash2 k=8 | agg 65.2 / 117.4 / 163.1 / 207.1 / 239.0; per stream at C5 49.5-53.1 | forced-512, unique nonce, thinking off | TP2 | SGLang | [Measured] | 2026-08-23, "3.2M-token cap qualification" |
| C1 → C5, same model, single node (K10) | C1 54.6; C4 agg 77.3; C5 agg 104.1 (~20 per stream) | forced-512 | 1 node | SGLang | [Measured] | 2026-09-08, "K10 screen" |
| C4 / C8, GLM 5.3 Flash EXL3 | C4 ~77 agg; C8 76-77 on 4 slots; 123 on 8 slots at 17.8 per stream | forced-512 | TP3/EP3 | vLLM + EXL3 | [Measured] | 2026-09-14 / 2026-09-16 packets |
| C4 / C8, DeepSeek V4.1 Flash | C4 65-71; C8 68-89 by slot count | forced-512 | TP3 | SGLang | [Measured] | 2026-09-14 / 2026-09-15 scorecards |
| C4 / C8, Qwen3.8-Flash-Next MTP3 | C4 93.8, C8 146.0 agg | forced-512 | 1 node | vLLM | [Measured] | 2026-09-11, "MTP3 / MTP4 lookup screen" |
| batch path drops speculation | native ds4 fork C1 33.8 → C2 28.3 agg (tokens per step 1.0 at C2+), plateau ~47 agg from C4 to C12; EXL3 K2 C1 51.0 → C2 39.7 agg | controlled battery | 1 node | Entrpi/ds4 v0.6.3; vLLM + EXL3 | [Measured] | 2026-08-22 single-node qualifications |
| C1 → C8, GLM-5.3-Flash NVFP4, DFlash2 | per stream 59.6 / 44.5 / 29.1 / 25.2 / 22.3 at C1/2/4/6/8; C8 agg 152.8; acceptance 63-64% throughout; a repeat ran ~5% lower | 12 English code prompts | TP3+EP | vLLM fork | [Community-measured] | NNNtrance docs/07 |
| C1 → C16, DeepSeek V4.1 Flash, DSpark | TP4 per stream 45.4 / 37.9 / 26.7 / 23.2 / 22.0 at C1/2/4/8/16, agg 134.2 at C16, TTFT 2.70 s at C8 and 12.45 s at C16; TP3 37.9 / 30.5 / 24.5 / 20.9 at C1-C4 | prose, 256 tokens, thinking off | TP3; TP4 | SGLang | [Community-measured] | MiaAI-Lab DeepSeek V4.1 README |
| c1 → c16, Qwen3.8-27B NVFP4, DSpark k=7 | per stream 59.79 / 41.92 / 30.75 at c1/c4/c8, agg 246.02 at c8; c16 256.47 agg at 16.03 per stream (batch budget 16384); identical prompts overstate c8 by ~11% (277.37 vs 246.02) | distinct prompts, 1,500 output tokens | 1 node | vLLM 0.27.1 | [Community-measured] | 0xBakeer RESULTS.md |
| N = 1 → 8, DeepSeek V4 Flash, DSpark k=5 | pure-decode agg 40.1 / 59.1 / 86.5 / 122.2; fitted T_forward = 82.5 + 22.94·N ms (rates validated within +1.6 to +7.3%, split not validated); `max_num_seqs` 8 → 12: +8.1% agg, per stream −20.8% (17.56 → 13.90) | fixed 800-token outputs; agent-load cells | 2 nodes | vLLM | [Community-measured] | lingjiacong07, [forum 380340](https://forums.developer.nvidia.com/t/380340) |
| 1 → 32 streams, small models | Qwen2.5-Coder-7B AWQ per stream ~46 / 48.2 / 47.0 / 43.8 / 36.9 at 1/4/8/16/32 (1,179.7 agg at 32); Qwen3-Coder-Next AWQ 33.7 / 31.6 / 26.3 / 23.2 / 16.9 (per stream derived from aggregates) | 512 max output, max-num-seqs 128 | 1 node | vLLM 0.18 | [Community-measured] | [shamily/vllm-gb10](https://github.com/shamily/vllm-gb10) |
| c1 → c12, DeepSeek V4 Flash (DSpark at N = 1 only) | 29.9 / 15.7 / 11.7 / 7.2 / 4.9 tok/s per request at 1/2/4/8/12; ~59 agg | 192-token completions × 3 | 1 node | Entrpi/ds4-on-spark v0.5.0 | [Community-measured] | Entrpi README |
| c8 aggregate with and without speculation, Gemma-4-26B-A4B NVFP4 | 178.9 (no spec) / 210.0 (MTP×4, acceptance 45-53%) / 303.6 (MTP×2, 63-66%) | c8 | 1 node | vLLM | [Community-measured] | sergioamsilva `speculative-decoding.md` |
| rows per weight read, kernel level | MMVQ Q2_K Qwen3.6-27B 24 → 105 t/s at ne11 1 → 8 (4.4x); Llama-3.2-3B 177 → 805; Q4_K 3B 109.8 → 421.6 at 1 → 4 (3.84x) | llama-bench | 1 node | llama.cpp | [Community-measured] | PRs #26079, #26705 |
| concurrency 1 / 2 / 4, Qwen3 Coder Next FP8 | agg 38 / 47 / 53 tok/s; the vendor recipe for Nemotron-3-Super-120B-A12B-NVFP4 sets `--max-num-seqs 4` ("the per-token bandwidth tax can outweigh continuous-batching gains") | 32K / 1K | 1 node | vLLM | [Community-measured] (vendor-published) | NVIDIA technical blog; [vllm.ai blog](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark) |
| prefill batching and generation budget, Qwen3-Coder-Next INT4 | prefill 3,881 (1 request) vs 3,931 tok/s (8 requests, +1.3%); `--max-num-batched-tokens` 32,768 halved generation 74.3 → 34.9 | distinct 8K prompts | 1 node | vLLM | [Community-measured] | sergioamsilva `concurrency-and-power.md` |
| long prefill during decode (see [04](04-prefill-optimization.md)) | decode share during a 262K prefill 1.7% of undisturbed at chunk 8192, 5.0% at 2048; five concurrent ~50K prefills dropped per-stream decode to 8-10 tok/s | per thread | 2 nodes; 4 nodes | vLLM | [Community-measured] | [forum 378890](https://forums.developer.nvidia.com/t/378890); [forum 381543](https://forums.developer.nvidia.com/t/381543) |

### Regressions and negatives

| change | baseline -> result | workload | topology | engine | grade | source |
|---|---|---|---|---|---|---|
| 4-bit KV vs f16 KV, Nemotron-3-Nano-30B-A3B UD-Q4_K_XL | turbo4 (Madreag/turbo3-cuda fork, build 8793): −2.5% at depth 0, −16.4% at 16K, −23.6% at 32K; q4_0 (build 8399): −36.8% at ~110K (24 vs 38 tps); q8_0 −34%; prefill unchanged (< 1%) | tg32 by depth | 1 node | llama.cpp | [Community-measured] | [Memoriant/dgx-spark-kv-cache-benchmark](https://github.com/Memoriant/dgx-spark-kv-cache-benchmark) (v3 corrected) |
| TurboQuant KV on vLLM stacks | 122B-A10B: 39 vs 51-52 tok/s without it; 35B-A3B ladder: 56-83 vs 113-127 ("Triton attn overhead") | per source | 1 node | vLLM | [Community-measured] | albond; [forum 365639](https://forums.developer.nvidia.com/t/365639) |
| `--speculative-attention-mode decode` | coding median 62 → 20.97 tok/s | production fixtures | TP2 | SGLang | [Measured] | 2026-08-27, "production maintenance" |
| Engram row cache 4 GiB, DeepSeek V4.1 Flash EXL3 | code / prose 64.13 / 42.65 → 60.40 / 38.24 | forced-512 | TP3 | vLLM EXL3 port | [Measured] | 2026-09-16, "EXL3 TP3 prefill tuning" |
| drop expert parallelism (TP-2304 layout), GLM NVFP4 | 1.16-1.26x slower, +12.5% weight bytes | same suite | TP3 | vLLM fork | [Community-measured] | NNNtrance docs/12 |
| cuBLAS 13.2.2.2 overlay | decode-shape bf16 GEMM ~28% slower (930 vs 723 µs per pair) | kernel pair timing | TP2 | SGLang | [Community-measured] | sglang #36796 |
| NVFP4 served by emulation fallback | ~1.1 tok/s vs 77.1 with FlashInfer CUTLASS selected | same weights | 1 node | vLLM | [Community-measured] | [forum 379766](https://forums.developer.nvidia.com/t/379766) |

## 5. Levers, ranked

Ranked by the largest single-stream effect measured on GB10. For aggregate throughput, lever 10 (slots and scheduler) comes before levers 2-5. Each gain band is the measured range with its source. Nothing here adds up; levers interact through acceptance and row count. [Interpretation on §4]

1. **Speculation on, with depth tuned to the real content** — applies to: every engine × every family × every topology, except where a verify costs close to a full target step (single-node MLA on ds4, NVFP4 hybrid MTP with a BF16 head). Gain: +54% prose and +95% coding over a flat 14.1 tok/s floor (own GLM NVFP4 TP2); 1.38x suite mean (Entrpi, context-thin, §8); up to 6x on edit-heavy text (0xBakeer FP8 27B, context-thin, §8). Depth: shallow wins on prose and at depth, deep wins only on predictable text (own k=3 over k=7 on every real-shaped fixture; DSpark k=5 +15% code, −8% prose vs k=3). Losses: 0.96x on creative writing and ai-muninn MTP n = 4 at −41% (both context-thin, §8). Depth and drafter choice: [Speculative decoding](02-speculative-decoding.md). Cost: drafter memory and a tuning pass per workload. Risk: acceptance collapse at depth when rope scaling never reaches the drafter; inflated acceptance from a broken head (§7). Smallest falsifying experiment: in one boot, three forced-k windows (k = 7, 4, 2) on frozen content-split fixtures (code, prose, 64K), logging per-step accepted tokens and step time. If step time falls with k as fast as a(k) does, depth is free and the ranking inverts. [Measured; Community-measured]
2. **Fewer weight bytes: 4-bit experts, a quantized dense path, a quantized LM head** — applies to: vLLM, SGLang, llama.cpp, EXL3 × dense and MoE × all topologies, strongest at C1-C4. Gain: weight-only FP8 over BF16 dense decode +7-14% at short depth (+1% at 120K) (own GLM TP2); 4-bit EXL3 dense path +21.7% C1 and +12.5% at 8-way (NNNtrance on cuda-exl3); EXL3 2.9 bpw experts +13-32% vs native MXFP4/FP8 (own, runtime also differs); INT8 LM head +6.8% c1 on current vLLM (albond). The gain fades with concurrency: +27% at c1 to +0.2% at c16 for 4-bit over FP8 (0xBakeer). Cost: requantization, a quality gate, and sometimes prefill (FP8 dense prefill −18 to −21%; BF16 is 22-26% faster). Risk: quality loss; the wrong kernel path (NVFP4 emulation at 1.1 tok/s; b12x NVFP4 GEMM 31 → 22). Smallest falsifying experiment: one-variable A/B of the dense path format at C1 and C8 on frozen fixtures, with the selected kernel read from the log and a greedy logprob sanity check. [Measured; Community-measured]
3. **Tensor parallel across nodes for large dense or large-active MoE** — applies to: vLLM, SGLang (and ds4 network TP) × dense ≥ 27B and MoE ≥ ~10B active × TP2-TP4. Gain: dense TP2 1.6-1.8x (own +60-73% decode; Openzeka 1.79x), TP4 2.62-3.74x (Openzeka; NVIDIA TensorRT-LLM); MoE TP2 → TP3 +29-45%, confounded with recipe (slots, k, kit) (own GLM); TP3 → TP4 +20% (DeepSeek V4.1). Small-active MoE: +5% (30B-A3B). Cost: nodes and a fixed ~5-16 ms/step of collective latency at TP3/TP4. Risk: padded shards when heads do not divide by TP; socket fallback halves throughput; failure domain grows. Smallest falsifying experiment: an identical battery at TP1 and TP2, fixed depth ladder, same drafter; if the TP2 step minus half the TP1 step exceeds the per-collective budget (count × ~70 µs), look for padding or a fabric fault. [Measured; Community-measured]
4. **Fewer drafter bytes** — applies to: any engine with an MTP/EAGLE/DFlash/DSpark drafter × all families × all topologies. Gain (drafter head bytes: [Sampling and the lm_head](12-sampling-and-lm-head.md); drafters: [Speculative decoding](02-speculative-decoding.md)): a trimmed 64K draft vocabulary +13-19% C1 at TP2 and +10-23% on one node, fading to +4-5% at C4/C5 (own Flash-Next); a calibrated NVFP4 drafter +4-7% (own 27B screen). Cost: a vocabulary list or a calibrated drafter checkpoint. Risk: acceptance loss (58.6% → 52.2% for 64K; an 8K list fell to 36% and lost everywhere). Smallest falsifying experiment: the same frozen C1 fixtures with full vs trimmed draft head, logging acceptance; the lever is falsified if acceptance loss × step time exceeds the saved head bytes ÷ ruler. [Measured]
5. **Drafter placement and parallel config** — applies to: vLLM (and forks) with TP > 1 and especially DCP. Gain: 14.6 → 22.0-23.0 tok/s when DCP was propagated to the draft (fork PR #72, four nodes). Own: the drafter is already built on every TP rank, established from source, not by measurement (a per-rank shape dump is still owed), so an expected "shard the drafter" saving was already banked. Cost: a code change in draft config construction. Risk: silent acceptance loss when draft and target disagree on KV sharding. Smallest falsifying experiment: log per-position acceptance with the draft on the target's parallel config vs a single rank at fixed k. [Community-measured; Code-verified]
6. **KV at FP8, never 4-bit** — applies to: all engines × dense GQA and full-attention layers of hybrids, strongest at depth; MLA gains less (small latent KV). Gain: grows with depth, e.g. BF16 KV costs ~13% step time at 84K and ~40% at 262K on dense 27B (own arithmetic); FP8 +1.5-1.9x inside cuda-exl3's sparse-MLA kernel. The vLLM DGX Spark blog warns FP8 KV "can carry a noticeable performance cost on Spark for some workloads", without numbers; the measured side (FP8 ≈ BF16 at short context in SGLang #36797) is better evidenced. 4-bit KV is a loss (§7). KV dtype mechanics are on [KV cache and prefix caching](10-kv-cache-and-prefix-caching.md). Cost: small quality risk. Risk: some FP8-KV kernels exceed the 99 KB shared-memory limit on SM121. Smallest falsifying experiment: FP8 vs BF16 KV at 8K and 128K depth, C1, same boot pair, same fixtures. [Measured arithmetic; Community-measured]
7. **Graph coverage and the decode host path** — applies to: vLLM and SGLang × MoE (large on one node) × all topologies; ExLlamaV3 coop MoE for ≤ 8 rows. Gain: eager → graphs on single-node MoE 23.4 → 66.9 (ai-muninn); own decoder bundle +18-21% C1 in a multi-change bundle, not attributable to graphs (the fullest component list is on [MoE dispatch](08-moe-dispatch.md) §4; pages record the change count as four to seven, and the packet count is unresolved). Nil where graphs already cover the step (dense Qwen3-8B wash; NNNtrance 98-99% occupancy; ds4 flat). Cost: capture memory and startup time. Risk: Hy3 at TP2 ran faster eager (unexplained); padded-row replay wedge on vLLM trees before #51538; a k-set change that moves rows off captured shapes. Smallest falsifying experiment: profile C1 with and without the change and compare GPU-busy fraction; if the CPU gap is already < 3%, stop. See [CUDA graphs and launch overhead](01-cuda-graphs-and-launch-overhead.md). [Measured; Community-measured]
8. **Remove host stalls** — applies to: case-specific (ds4 encode, Engram table reads, per-step host offload). Gain: up to the stall itself: ~10 ms of a 55 ms ds4 step; ~11 ms/step of NFS Engram waits in the MiaAI-Lab DeepSeek V4.1 kit. Cost: engineering. Risk: low. Smallest falsifying experiment: an nsys trace at C1 showing the GPU idle gap per step before and after. [Community-measured]
9. **Collectives: fabric, channel count, fewer collectives** — applies to: TP ≥ 2 on vLLM and SGLang. Gain: RoCE over sockets ~2x (flash7777 91.6 vs 45.3; XDA 17.1 vs 9.2); `NCCL_MAX_NCHANNELS=8` +7.7% at C4 and +9.7% at C6, nil at C1 (NNNtrance). Fewer collectives: the LM-head all-gather is ≈17% of decode ring bytes at 64 rows on GLM TP3 with 16-bit logits (≤30% only if fp32; see [Sampling and the lm_head](12-sampling-and-lm-head.md)); a greedy fast path is [Proposed], no gain measured. Cost: config. Risk: low. Smallest falsifying experiment: a per-collective latency microbench on the own triangle (8 B to 16 MiB), then collectives × latency vs the measured NCCL share of the step. [Community-measured; Proposed]
10. **Slots and scheduler fairness (aggregate only)** — applies to: all serving engines × all families. Gain: slots 4 → 8 +30.8% C8 agg (own DeepSeek V4.1 TP3, with chunk and pool also changed); fair mixed-prefill +12.6% C8 and newcomer TTFT 130 → 6 s (own GLM TP3); the +61% C8 of the own decoder bundle is not attributable to slots alone. Cost: per-stream speed (−17.5% at C8 in the bundle; `max_num_seqs` 8 → 12 cost −20.8% per stream for +8.1% agg on forum 380340). Risk: memory headroom ([Memory on unified memory](11-memory-on-unified-memory.md)); the MoE row count crossing a fused-path boundary (§7). Slots and fairness are on [Serving agent workloads](22-serving-agent-workloads.md); chunk and fair prefill on [Prefill optimization](04-prefill-optimization.md). Smallest falsifying experiment: a C8 ladder at 4 and 8 slots with prefill chunk and pool fixed. [Measured; Community-measured]
11. **Sampler** — applies to: all. Gain: nil at C1 (0.07 ms/step); a fused Markov argmax +5-14% at C4 on one DSpark tree. Cost: a code port. Risk: low. Smallest falsifying experiment: C1 and C4 with and without the fused argmax on frozen fixtures. [Community-measured]
12. **GPU clock** — applies to: all. Gain: none for decode. C1 means 5-6% lower, inside rep spread (own TP3, n=5); community controlled runs ±2%; one unreplicated own −6.8% (n = 3) on a DeepSeek V4 TP2 lane, the same size as the TP3 means. Its value is thermal (see [Clocks, thermal, and power](13-clocks-thermal-power.md)). Smallest falsifying experiment: n ≥ 20 interleaved C1 decode per arm, first request after each lock discarded. [Measured; Community-measured]
13. **Micro-tuning kernels already at bandwidth** — applies to: Marlin W4A16, ds4 CUDA decode, EXL3 coop MoE. Gain: nil to negative. Own ABI3 micro-changes −0.8% C1 and −5.5% C8; ds4 fusions flat; Marlin at 91-96% of the ruler leaves at most 4-9%. Exception: kernels well below the ruler in-engine (KDA GEMM at 150 of 214 GB/s; BF16 LM head at 64% at batch 1). Cost: kernel work ([Kernel authoring on SM121](15-kernel-authoring-sm121.md)). Risk: regressions hidden by component wins. Smallest falsifying experiment: before tuning, measure the kernel's in-engine time against its bytes ÷ ruler; if it is within 10%, stop. [Measured; Community-measured]

## 6. Protocol

This is the procedure an agent follows to change one decode lever on a GB10 lane and decide whether to keep it. Bottleneck classification before the change belongs to [Profiling protocol](14-profiling-protocol.md). Quality gates are in [Quality and correctness gates](21-quality-and-correctness-gates.md).

**Preconditions.**
- All index rules apply (see [the entry page](README.md)), in particular: at least 8 GiB MemAvailable on every node before a start, a 5 GiB warm floor measured after the longest qualified prefill, and traffic stopped below 4 GiB; a greedy logprob probe after every relaunch; discard the first request and warm every serving shape before timing; a unique nonce at token 0 with zero cached tokens asserted from the usage block; every rank at the same clock cap with no latched rank; the selection or configuration line read on every rank; at least 3 repetitions with a closing baseline (A/B/A); MemAvailable sampled every 1-2 s with zero NV_ERR_NO_MEMORY or Xid lines; no reboot or power cycle without per-node permission; correctness gates from [Quality and correctness gates](21-quality-and-correctness-gates.md) before any speed number. For this page also: raise a KV pin by at most 1.5 GiB per boot.
- Fleet rules from the entry page ([GB10 Inference Wiki](README.md)):
  - no memory overcommit;
  - one variable per boot;
  - stop if `MemAvailable` falls under 4 GiB on any node;
  - a clean window with no outside traffic (poll the engine's running-request count every second, and discard any pass it overlaps);
  - if an A/B needs an engine restart, skip the image pull;
  - never restart an agent gateway;
  - never `pkill -f` a pattern that matches your own shell.
- Write the bytes model for the lane first: B_rank by component (dense, experts, KV at the test depths, drafter, LM head), collective count per step, and the predicted step at a 230-245 GB/s ruler. Keep it with the packet. If the measured step is far below the prediction (implied bandwidth above the ruler), suspect L2 reuse, correlated routing or a mixed-rank calculation before celebrating. [Historical diagnostic, own 2026-09-20 dossier]
- Read the selected kernels from the engine log for every quantized layer class (dense, MoE, LM head, attention, KV). A quantization name is not a kernel. [Community-measured, forum 379766]
- After any relaunch, run a greedy top-1 logprob sanity check. A dropped LM-head weight scale made draft and verifier agree on a broken head, which inflated acceptance and withdrew a whole arm's numbers. [Measured, 2026-08-19 audit]
- Freeze the harness: fixed fixture bodies split by content class (coding, prose, structured, depth 64K and 120K where the lane serves them), a cold nonce at the start of each prompt, and a fixed thinking mode. [Measured, own practice since 2026-09]
- Warm the lane with one discarded request. The first request after idle or after a new clock lock has read low (24.6 / 28.8 / 23.6 in one sweep). [Measured, 2026-09-21 sweep]

**The change.** One variable per boot, with every other knob fixed and written down. Examples: draft depth or k-set; draft vocabulary; one layer class's weight format; KV dtype; slot count; one kernel backend. Two exceptions:
- Speculation depth can be changed inside one boot where the engine supports a forced-k file. Use that to avoid restart confounds. [Proposed]
- A k-set change moves MoE row counts. Re-profile the MoE dispatch policy and graph-capture list for the new rows first, or the A/B measures the coupling, not k. [Measured, 2026-09-15 confound]

**Record (packet fields).**
- Engine, image and commit; recipe; topology; clocks and cap state; firmware and driver.
- For every rep and class: output tokens counted from `completion_tokens` (checked against the requested length), elapsed decode time, TTFT, per-stream tok/s, and aggregate for C > 1. Count tokens, never SSE chunks: one chunk carries a whole accepted block (2.5-3.5x under-count). An engine-native throughput counter is an aggregate, not a single stream. [Community-measured; Measured, 2026-08-29 metric audit]
- Acceptance from counter deltas per class: proposed, accepted, and mean accepted length.
- Whether speculation survives batching (tokens per step at C2 and above). [Measured, 2026-08-22]
- Selected kernels per layer class; captured graph row sizes; slots; prefill chunk; KV dtype and pool size.
- Per-rank, never mixed-rank, step time when profiling multi-node. [Historical diagnostic]
- Forced-length rows and natural-EOS rows kept separate. The effect of ignore_eos itself is unmeasured (§8).

**Pass gate.**
- Run at least 3 reps per arm on the frozen suite. Use 5 when the expected effect is under 10%: run-to-run spread on single runs reaches about 9% (Inference Atlas), and a restored control on the own frozen harness moved 43.8 ↔ 44.3. [Community-measured; Measured]
- Keep only if the gain beats the pre-registered threshold on coding and prose at C1, and C8 aggregate does not regress beyond the same threshold. A win only on counting or structured fixtures is not a pass (§7).
- No keep on a single run. A batch-budget change "kept" on one 32K run later cost 16% on 256-token decode. [Historical diagnostic]
- The quality gate passes (logprob sanity; acceptance plausible for the content, e.g. no near-1.00 acceptance on prose).
- An uncontrolled quick probe never overrides the frozen harness. A +38% probe claim was withdrawn when the frozen suite showed no change. [Measured, 2026-09-23 verification]

**Rollback.**
- The previous recipe stays launchable. Revert by relaunching it with the pull skipped, then re-run the frozen C1 pair to confirm the old numbers within spread.
- Do not restart gateways. If a client-facing change needs a gateway restart to take effect, stop and hand that step to the operator.
- File the negative result with its numbers. A rejected lever is a finding (§7).

## 7. Anti-patterns and known-bad

- [Community-measured] **Quoting a locked-attractor number as speed.** The 1,004.89 tok/s GLM-5.3-Flash figure is 2× RTX PRO 6000 (not GB10) at 3,090 MHz / 450 W, SGLang DFLASH with 64 draft tokens and "width-64 speculative chains", best valid of 6, "acceptance 64/1.00 on locked attractor". A sibling run scored 808.51 at k=28. Once output locks into a repeating period, n-gram proposals fill the chain and every draft is accepted. It measures the attractor, not the model (localmaxxing run `cmtk0maew03mrp701oyaivvka`).
- [Community-measured; Measured] **Synthetic and structured probes as coding or prose speed.** They inflate speculative decode 1.5-4x over prose on GB10:
  - "count to 200" ≈ 93 tok/s vs code 57-60 and prose ~21 (NNNtrance TP3);
  - GLM on one node, 64 structured vs 25 prose (forum 382140);
  - lorem-ipsum at acceptance 0.98 ran 33.6 vs 26-27 realistic (jeremy-newhouse);
  - DeepSeek V4.1 TP4 counting 92.2 vs code 70-77 (tonyd2wild #1);
  - an own "65.3 structured" k=7 headline came from a maximally predictable prompt.
  Exclude count-to-N, k=64 and accept-1.00 results from coding and prose tables.
- [Community-measured] **Per-chunk ITL as a regression.** A reported "~7x MTP regression" (vLLM #47297, closed) was a per-chunk measurement. TPOT was 9.72 ms, and per-chunk ITL was 30.38 ms = 3.13 × TPOT. A separate 128-token output cap despite `ignore_eos` + `min_tokens` was a server-side defect on specific nightlies, later gone, with no identified cause. Rule: check `completion_tokens` against the requested length on every run.
- [Community-measured; Measured] **Counting SSE chunks as tokens.** 2.50 tokens per non-empty chunk (forum 378890), about 3.5x on DSpark k=5 (MiaAI-Lab README). An engine counter peaking at 110 was an aggregate, while single requests had a median of 44 (own 2026-08-29 audit).
- [Community-measured] **The SSE-count error inside a scaling model.** A four-node SGLang GLM-4.7 FP8 study fitted tok/s = η·(273·TP)/(W+KV) with η ≈ 0.22. Its streaming-client numbers were under-reported about 2.2x (its own server-side test: 16.77 EAGLE on, ~14 off). Corrected, η ≈ 0.41-0.49, a ~2x over-prediction by the per-node model, not 4.5x (BTankut RESULTS.md).
- [Community-measured] **4-bit KV on GB10.** TurboQuant turbo4, q4_0 and NVFP4 KV are slower than f16 or FP8 at every published depth: −23.6% at 32K, −36.8% at ~110K, −29% for NVFP4 KV on SGLang (§4). One GB10 recipe ships turbo4 KV inside a DFlash config and never isolated the KV effect (phuongncn), so it is not evidence either way.
- [Measured] **Trusting acceptance when the head is broken.** SGLang discarded `lm_head.weight_scale` on a packed-FP4 head (~6,500x logit inflation). Draft and verifier were corrupted identically, prose acceptance p90 reached 8.0, and that arm's numbers were withdrawn (2026-08-19 audit). [Interpretation] A community A/B reporting 53.28 → 22.85 tok/s (−57%) when switching to a BF16 head shows acceptance 6.09 at k=7 on random-token input, the same signature. Treat it as unverified until a logprob check is published (liuzl JSON).
- [Measured] **Headlining an uncontrolled quick probe.** One 36-token prompt, no warm-up, no nonce: +38% claimed, then withdrawn against the frozen harness. Suspected causes: adaptive-k state after days of traffic, or first-request-after-idle (2026-09-23 verification).
- [Historical diagnostic] **Mixed-rank bandwidth curves and estimator shortcuts.** A first-pass "66-92% of nominal, 10-30% headroom" reading was built from worst-ratio ranks; per rank the same logs imply up to 351 GB/s. A community estimator claiming "98.6% of 236 GB/s" hardcoded a TP2 share, substituted accepted for verified rows (a 41% byte understatement at C1 k=7), used wall time including TTFT, and omitted drafter, KV, collectives and imbalance. Rule: no percent-of-ceiling without a DRAM byte counter, and never mix ranks (2026-09-20 dossier).
- [Historical diagnostic] **Assuming the drafter layout from a comment.** A recipe comment said the drafter ran on rank 0 only. vLLM V1 builds it under the target parallel config, padded 32/8 → 36/9 heads, at ~0.72 GiB per rank. The "shard the drafter" saving was already banked (2026-09-20 dossier). A per-rank shape dump is still owed to confirm the placement at run time.
- [Measured] **A k-set A/B without re-profiling MoE dispatch.** {2,4,6} put 5 of 25 captured rows on the stock kernel and {3,5,7} put 4 of 22, while the freeze script hardcoded the {2,4,7} grid. The k-set result measured the coupling as much as k (2026-09-15).
- [Measured] **A row count sitting on a fused-path boundary.** 8 slots × (7 + 1) verify rows = 64 = the fused-MoE temp-rows limit. One more stream at k=7 would fall onto the prefill path. Guard the boundary explicitly.
- [Measured; Community-measured] **Comparing decode across thinking modes.** Own Qwen MTP lanes decode faster with thinking on (36-38 vs 32 tok/s at TP2, identical battery). An own DeepSeek V4 TP2 lane decoded 53.7 at reasoning effort "low" vs 45.4 at "none", yet wall time was 101 vs 56 s. NNNtrance measured no change on its stack (C1 28.94 vs 28.91, C8 94.04 vs 97.11; raw lost). Both hold for their stacks. Compare decode only within one thinking mode, and judge turns on wall time.
- [Measured; Community-measured] **Quoting a concurrency ladder from a batch path that drops speculation.** Native ds4 fork C2 28.3 agg < C1 33.8; EXL3 K2 C2 39.7 < C1 51.0; Entrpi 29.9 → 15.7 per request at c2. Check tokens per step at C2 before publishing a ladder.
- [Measured] **Bench probes that production never exercises.** An MTP lane accepted 3.0/3 on probes and 0.00 on real ~245K-token traffic, because runtime rope-scaling overrides never reached the drafter. Baking YaRN into the checkpoint config restored ~2x decode at production depth (own 2026-08-18 diagnostic).
- [Measured; Community-measured] **Single-run keep decisions and sub-10% single-run wins.** A 10,240 batch budget was "kept" on one 32K run and later cost 16% at 256 tokens. Inference Atlas measured ~9% run-to-run spread on identical reruns.
- [Measured] **Micro-optimizing bandwidth-bound kernels.** Component wins on fused staging and shared gate/up rotation measured −0.8% C1 and −5.5% C8 end to end, and were excluded (2026-09-20).
- [Measured] **Speculative attention mode "decode" as a speedup.** 62 → 21 tok/s on SGLang TP2. It is a recovery crutch for builds with prefill graphs off.
- [Community-measured] **Treating an uncontrolled cross-report difference as a regression.** "vLLM v0.22 stock 9.86 vs nightly 12.63" compares two papers captured under different runtimes and flags. It is not a measured regression (Openzeka). By contrast, the cuBLAS 13.2.2.2 overlay (−28% on decode-shape GEMMs) and emulation fallback (1.1 vs 77.1) are real, same-harness regressions.
- [Community-measured] **Kernels that exceed 99 KB of shared memory.** Triton MLA decode with FP8 KV (102,400 B) and tilelang DSA (169,984 B) fail on SM121. SGLang's TRT-LLM QSA path returned token 0 on long-context decode and is now excluded by name.
- [Community-measured] **Assuming FP4 tensor cores speed up decode.** Marlin W4A16 stays 1.03-1.07x ahead of an ideal FP4 GEMM up to M = 256.
- [Measured] **Autotune boot lottery.** FlashInfer autotune picks tactics by a wall-clock race per boot. Fresh draws gave no repeatable C1 win locally. Replay a cached tactic set.
- [Measured] **Engram row cache on DeepSeek V4.1 EXL3.** −6 to −10% decode. Keep cache size 0.

## 8. Open questions

- **Any DRAM byte counter on GB10.** No `ncu --metrics dram__bytes_read.sum` has been published for Marlin, MMVQ, EXL3 or NVFP4 GEMV on SM121, so every "% of bandwidth" is a ratio to a probe. — One `ncu` pass per decode-shaped kernel, with weight sets rotating beyond 8× L2.
- **Routing correlation among verify rows.** Uniform routing over-predicts expert bytes in two datasets. — Log `topk_ids` for 1,000 steps at C1 and C8, and count distinct local experts per rank per step.
- **A per-class, per-rank step profile at TP2 for MoE and for MLA.** Only TP3 (NNNtrance, MiaAI-Lab) and single-node profiles exist; the forum 380340 fit gives rates, not bytes. — A torch-profiler or nsys trace of a TP2 MoE lane at C1 and C8, split by class and rank.
- **Why NVFP4 dense decode falls 25-40% short of the bytes model** while BF16 dense lands within ~10% (NTTPC 70B, dendro-logic 49B). — A profile of NVFP4 dense decode at C1 with the selected kernel and per-kernel time vs bytes.
- **Native NVFP4 GEMV at M = 1 on SM121.** The only NVFP4 GEMV write-ups found are for SM100 and SM120. — A model-free bandwidth bench of the vLLM FlashInfer CUTLASS FP4 path at M = 1-8 against the rotating-buffer ruler.
- **Upstream ExLlamaV3 small-M GEMV and INT8 GEMV on GB10.** Untimed by anyone; the own kernel-lab sweep is staged but unrun. — The tactic sweep (tile shapes × SM quotas, M = 1..128) on an idle node.
- **Is ExLlamaV3's INT8 GEMV graph-capturable?** The call-site comment says no; the kernel records graph parameters. — Capture a decode graph with `EXL3_INT8_GEMV` at its default and compare outputs and timings with it disabled.
- **Batch scaling past C16 with speculation on, multi-node MoE.** The largest published ladder stops at C16 (MiaAI-Lab TP4); own lanes stop at C8; the knee in the interpretation below is interpolated. — A C1 → C32 ladder with slots, chunk and pool fixed. [Interpretation] Recomputed from the cited rows, per-stream at C2 is about 75-83% of C1 and at C4 about 50-60%.
- **The effect of ignore_eos itself on tok/s.** No source compared forced-length with natural-EOS decode on a fixed prompt set. — The same frozen prompts run both ways, with acceptance logged.
- **The cause of the firmware-dependent slow state.** It has not been reproduced on newer BIOS, driver and kernel. — The reporter's GEMV probe on one older-firmware node after updating only the BIOS, then only the driver.
- **The own DeepSeek V4 TP2 clock-cap −6.8%.** The own TP3 C1 means were 5-6% lower inside rep spread; community controlled runs were within ±2%. — n ≥ 20 interleaved C1 reps per arm on that lane, first request after each lock discarded.
- **Per-collective latency on the own three-node triangle.** Community figures are 60-85 µs. — An all-reduce microbench from 8 B to 16 MiB, NCCL settings as served.
- **LM-head all-gather vs a greedy `[batch, 2]` fast path at TP3.** About 12.6 MiB/step at 64 rows with 16-bit logits (25 MiB only if fp32), ≈17% of decode ring bytes (≤30% if fp32). [Proposed] — A/B of the two paths at C1 and C8 with outputs checked.
- **The sign of a cost-aware speculation policy** (choose k to maximize a(k)/T_step(rows)). The analysis flips between chain ratios 0.70 and 0.85. — Three forced-k windows in one boot, logging per-step accepted tokens and step time.
- **Linear-attention state bytes per sequence, measured.** Only kernel time shares exist (2.5% at C1, 7.2% at C8). — Allocator accounting of recurrent state per slot plus a per-kernel byte estimate.
- **The KDA GEMM gap: 214 GB/s standalone vs 150 in-engine.** Alignment explains 3-5%. — Run the same kernel in-engine and standalone under `ncu` and compare memory transactions.
- **Eager faster than graphs on Hy3-295B TP2** (21.8 vs 15.5-16.3), the opposite of single-node MoE. — Profile both modes at C1 and compare collective and MoE kernel times.
- **A per-step host-offload ceiling on single-node Flash-Next.** Three users report 21-28 tok/s single-stream and a py-spy stall in `ple_offload.wait_d2h`, with speculation slower (13.5-13.9 at MTP=3). The mechanism is unproven, and an own single-node Flash-Next lane runs 45-53 tok/s code (MiaAI-Lab Qwen3.8-Flash-Next #59, open). — An nsys trace of that kit at C1 showing the per-step D2H wait.
- **The BF16-head −57% on Qwen3.8-27B + DFlash2** (liuzl). — The same A/B with a greedy logprob check and realistic prompts.
- **vLLM build target 12.1a vs 12.1f.** christopher_owen (forum 356651 #23) says 12.1a excludes block-scaled MMA and FP4 ldmatrix. NVIDIA's johnny_nv (#25) says "12.1a use full features … Maybe they are filtering something". eugr (#26) reads it as a FlashInfer bug. vLLM `711fc55c10a2` compiles Marlin for 12.0f on CUDA ≥ 13, else 12.0a;12.1a. — Build both targets and diff the kernels selected and decode tok/s on one NVFP4 model.
- **gpt-oss-120b MXFP4 on vLLM.** llama-benchy posted ~59-60 tok/s tg128 at shallow depth, falling to 55 at 16K (forum 356651 #28). htzl.ai measured 34.3 native on both a self-built vLLM and NGC 26.01, both stuck on TRITON_ATTN, and labels 59 community-only. A profiled "CUTLASS" run was Marlin at 29.2 (all 18 CUTLASS tactics failed for arch 120). — One run with the attention backend and MoE kernel read from the log.
- **An NVIDIA document for the 24 MB GPU L2 and the expected sustained read on GB10.** — A vendor spec or `deviceQuery` plus a vendor-sanctioned bandwidth test.
- **Context-thin community speculation rows, moved out of §4 until each has a stated baseline, workload, n and topology.** Speculation on: Gemma-4-26B-A4B NVFP4 30.3 → 54.9 (MTP×2); Qwen3.8-27B FP8 7.88 → 47.1 (DSpark k=7, edit-heavy, ~99% acceptance); DeepSeek V4 Flash DSpark 19.72 → 31.41 on code and 19.53 → 18.81 on unpredictable prose (3 × 256 tokens); Entrpi 9-workload suite mean 1.38x (1.71x stepwise math, 0.96x creative writing) (1 node; vLLM; antirez/ds4; Entrpi/ds4; sources: [sergioamsilva](https://github.com/sergioamsilva/dgx-spark-field-notes); [0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark); antirez/ds4 `docs/DGX_SPARK.md`; [Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark)) [Community-measured]. Speculation where verify bytes are large: NVFP4 hybrid 35B-A3B MTP n = 1/2/3/4: 67.0 / 56.3 / 54.6 / 39.4 vs 67.1 without; vLLM 0.27.1 MTP-2 only +2.4% ("the MTP head ships BF16 at 4.8 GB" against an INT4 target); ds4 target eval ~60 ms vs a 2-5-row verify ~83-97 ms; Entrpi: legacy MTP on DeepSeek V4 Flash was "a net throughput loss even at 100% draft acceptance" (1 node; vLLM; antirez/ds4; sources: [ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-nvfp4-w4a4-moe-cudagraph); albond; #773; Entrpi) [Community-measured]. — Re-read each source and record its workload, repetitions and topology before citing any of these numbers as a §4 row.


## 9. Sources

**Vendor docs and vendor-published pages** (fetched 2026-09-23)
- NVIDIA technical blog, "Scaling autonomous AI agents and workloads with NVIDIA DGX Spark" (post dated 2026-03-16): Table 2 (Qwen3 Coder Next FP8 on vLLM, 32K/1K), Table 3 (Llama 3.3 70B Instruct NVFP4 on TensorRT-LLM, 32K/1K, batch 1). https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark
- vllm.ai blog, "vLLM on DGX Spark" (2026-06-01). https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- NVIDIA DGX Spark user guide, clustering (no decode numbers). https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
- NVIDIA Developer Forums threads (topic ids; every post read via the Discourse JSON API): 356651 (gpt-oss MXFP4; 12.1a vs 12.1f), 362368 (397B on 4 nodes), 363238 (bandwidth; 35B-A3B profile), 365555 (4-node matrix), 365639 (122B ladder, page 7), 377334 (sparse-MLA livelock under cold prefill), 378890 (agent serving; SSE chunks; long-prefill starvation), 379129 (llama.cpp DeepSeek V4), 379766 (same-harness sweep; emulation fallback), 380340 (DeepSeek V4 Flash on two nodes; fitted model), 381543 (GLM on 4 nodes), 382140 (GLM on 1 node). https://forums.developer.nvidia.com/t/{id}

**Source code (pinned)**
- vllm-project/vllm `711fc55c10a2bbbefcb12456c43ba5eca01be3f6`:
  - `vllm/model_executor/kernels/linear/__init__.py:455-497, 551-571, 1115-1126`
  - `nvfp4/flashinfer.py:116-123, 184-197`
  - `nvfp4_scaled_mm_entry.cu:71-90`
  - `scaled_mm_entry.cu:161-171`
  - `fused_moe/oracle/nvfp4.py:194-203`
  - `experts/trtllm_nvfp4_moe.py:206-213`
  - `experts/flashinfer_cutedsl_moe.py:76-82`
  - `experts/flashinfer_cutlass_moe.py:128-139`
  - `experts/flashinfer_b12x_moe.py:165-171`
  - `vllm/platforms/cuda.py:719-725`
  - `csrc/libtorch_stable/quantization/marlin/marlin.cu:295-321`
  - `marlin_utils.py:687-706`
  - `CMakeLists.txt:610-636`
  - `vllm/config/speculative.py:1760-1785`
- sgl-project/sglang `2909f84e3b2fc627bbc9de76ac71d1c295639a8c`:
  - `srt/utils/common.py:286-336`
  - `layers/quantization/fp8_utils.py:590-591, 925-953`
  - `modelopt_quant.py:2317-2319`
  - `arg_groups/kv_cache_hook.py:121-127`
  - `qwen_sparse_attn_backend.py:46-58`
- ggml-org/llama.cpp `6e60f35608ec6918b44a9839c0c433687165f086`:
  - `ggml/src/ggml-cuda/mmvq.cu:1-36, 348-354, 553-576, 1140-1156`
  - `common.cuh:61`
  - `mmvq.cuh:3`
- turboderp-org/exllamav3 `6b84a21b6f1e5da3f291b9e1019061f0de788279`:
  - `exl3_gemv_kernel.cuh:1-30`
  - `exl3_gemv.cu:13-74` (gate at 53)
  - `exl3_gemm.cu:185-188, 227-242`
  - `exl3_gemv_int8.cu:1-52, ≈223-231, 353-375`
  - `exl3_devctx.cu:39`
  - `exl3_moe_coop_kernel.cuh:1-45`
  - `libtorch/mlp.h`
- antirez/ds4 `0aaea5a238fb41a35106a551e73c8409dfb751ac`: `docs/DGX_SPARK.md`, `docs/PERFORMANCE.md`.
- Entrpi/ds4-on-spark `99fb7b1705281a9767e18176c889d0b7dbbf6a8b`; Entrpi/ds4 `b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd`.
- Pull requests:
  - llama.cpp #26079 (head `41a70d7e6df0`, merged `2b5621094ef3`)
  - llama.cpp #26705 (head `aa17a1bcbf61`, merged `73ab7599b553`)
  - llama.cpp #26843 (head `2c1c2e5850a9`, merged `25ae3a9b331f`)
  - SGLang #36845 (head `d0ed31a40427`, merged `78c5024e9d9f`)
  - SGLang #36649 (head `b0dd2211959a`)
  - local-inference-lab/vllm #72 (head `6785ad5a789e`, open)
  - vLLM #51538 (merged as `97388c44`)
  - vLLM #54013 (open)
- Issues (fetched 2026-09-23):
  - vLLM #44740, #47297, #53051, #53748
  - SGLang #36558, #36796, #36797, #40286
  - FlashInfer #5015 (closed)
  - antirez/ds4 #773
  - tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark #1
  - MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark #59
  - llama.cpp #27780, #27918

**Papers**
- MARLIN, arXiv 2408.11743 (small-M mixed-precision GEMM design; Ampere-era context, not a GB10 measurement).

**Community repositories and write-ups** (handle, link, pin or fetch date)
- NNNtrance, https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark @ `eb11cef295c1` (docs/07 speed, docs/10 production candidate and lessons, docs/11 measured profile, docs/12 what we closed, bench/moe-kernels README, results/kernels moe-kernel-bench and fp4-crossover-sweep).
- MiaAI-Lab, https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks @ `25379c22e8ff`.
- Zeuss5, https://github.com/Zeuss5/cuda-exl3 @ `6a1ffc34866e` (README; docs/dgx-spark-glm53.md).
- vcruz305, https://github.com/vcruz305/vllm-exl3 tag v0.3.1.
- Memoriant, https://github.com/Memoriant/dgx-spark-kv-cache-benchmark @ `ecd33535c987` (v3 corrected).
- 0xBakeer, https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark @ `1515e1cccb6d` (RESULTS.md).
- flash7777, https://github.com/flash7777/vllm-marlin-sm12x @ `ae90dc66e23e` (RESULTS.md).
- albond, https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4 @ `d9672119a665`.
- sergioamsilva, https://github.com/sergioamsilva/dgx-spark-field-notes @ `b1e00bd191a7` (speculative-decoding.md, concurrency-and-power.md).
- shamily, https://github.com/shamily/vllm-gb10 @ `42fee0b482d9`.
- sfxnz, https://github.com/sfxnz/spark-recipes @ `e25f4b438f06`; https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ `45d330336f0a`.
- Anemll, https://github.com/Anemll/dspark-vllm-gx10 @ `081fda975417`.
- hasso5703, https://github.com/hasso5703/dgx-spark-qwen38 @ `81ae2bf58394`.
- tonyd2wild, https://github.com/tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark @ `285bc9177800`.
- christopherowen, https://github.com/christopherowen/spark-vllm-mxfp4-docker @ `978c330472cf`.
- liuzl, https://github.com/liuzl/qwen38-dgx-spark-lab @ `362d9e10fd9a` (head A/B JSON).
- BTankut, https://github.com/BTankut/dgx-spark-sglang-moe-configs @ `5d4f546b1ee6`.
- jeremy-newhouse, https://github.com/jeremy-newhouse/dgx-spark-nemotron-super-bench @ `00bb491b30b0`.
- phuongncn, https://github.com/phuongncn/qwen3.6-27b-speedhack-gx10-dgx-spark @ `d28e0dc2c7b5`.
- kubesimplify, "Running Qwen3.8-27B on DGX Spark" and "DGX Spark unpacked" (deviceQuery), https://blog.kubesimplify.com (fetched 2026-09-23).
- Chips and Cheese, "Inside NVIDIA GB10's memory subsystem", https://chipsandcheese.com/p/inside-nvidia-gb10s-memory-subsystem (fetched 2026-09-23).
- ai-muninn, EN post (2026-06-01) and zh-TW post (2026-04-21, updated with a retraction), https://ai-muninn.com (fetched 2026-09-23).
- NTTPC, benchmark32, https://www.nttpc.co.jp/gpu/article/benchmark32.html (fetched 2026-09-23).
- dendro-logic, DGX Spark concurrency benchmark, https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark (fetched 2026-09-23).
- Openzeka, "Qwen3.6-27B DGX Spark Cluster Scaling", https://whitepapers.openzeka.com/papers/qwen3.6-27b-dgx-spark-scaling (fetched 2026-09-23).
- htzl.ai, GB10 inference engine shootout, https://www.htzl.ai/blog/gb10-inference-engine-shootout (fetched 2026-09-23).
- XDA, "Running a 284-billion-parameter model on two machines", https://www.xda-developers.com/running-284-billion-parameter-model-two-machines-matches-cloud (fetched 2026-09-23).
- localmaxxing speed-test API for GLM-5.3-Flash (44 rows; unverified submissions), https://www.localmaxxing.com (fetched 2026-09-23).
- X posts via a Markdown proxy (fetched 2026-09-23): @ViC305 https://x.com/ViC305/status/2095559790902890679; @0xBakeer https://x.com/0xBakeer/status/2094129132858933576 (Inference Atlas run-to-run spread); @liuzl https://x.com/liuzl/status/2092597944063025222.

**Own packets** (cited by date and title; lane detail in the private overlay)
- 2026-08-16 "fleet serving bake-off"; 2026-08-16 "decode tuning A/B"; 2026-08-16 "clock cap A/B"
- 2026-08-17 "TP1 vs TP2 identical battery"; 2026-08-17 "reasoning-tax minibench"; 2026-08-18 rope-scaling drafter diagnostic
- 2026-08-19 "single-node uncensored DFlash2 campaign audit" and no-spec control; 2026-08-20 "speed headroom campaign"; 2026-08-21 "FP8 MTP sweep"
- 2026-08-22 single-node native-ds4 and EXL3 K2 qualifications; 2026-08-23 "3.2M-token cap qualification"; 2026-08-27 "GLM 5.3 Flash MTP-4 versus MTP-off"; 2026-08-27 "production maintenance"
- 2026-08-28 "fastest one-GB10 recipe audit"; 2026-08-29 metric-semantics audit; 2026-08-30 "dual-HCA fabric enablement"
- 2026-09-03 "A1 k=3 / k=7 overnight qualification"; 2026-09-03 "Vision-Exp gates baseline qualification"
- 2026-09-05 "FP8-KV qualification" and single-node draft-vocabulary screen; 2026-09-08 "E4 A/B"; 2026-09-08 "K10 screen"; 2026-09-09 single-node tuning; 2026-09-11 "MTP3 / MTP4 lookup screen"
- 2026-09-14 "TP3 triangle qualification"; 2026-09-14 "DS V4.1 TP3 quick scorecard"; 2026-09-15 "TP3 upgrade A/B"; 2026-09-15 "fast profile k=3 scorecard"
- 2026-09-16 "combined runtime qualification"; 2026-09-16 "DS V4.1 EXL3 TP3 component gates"; 2026-09-16 "EXL3 TP3 short-context optimization"; 2026-09-16 "EXL3 TP3 prefill tuning"
- 2026-09-20 "TP3 frontier dossier v2"; 2026-09-20 "next-batch optimization"; 2026-09-21 "GPU clock-cap sweep"; 2026-09-23 V3.2 verification
