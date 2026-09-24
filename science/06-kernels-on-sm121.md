---
title: The Science of GB10 Inference — chapter 06, kernels on SM121
created: 2026-09-23
status: checked
type: chapter
---

# Chapter 6 — Kernels on SM121

Extends core §6. Core §6 already holds the fusion equation, the dispatch default table, the "cannot run" table, the toolchain line and the Amdahl procedure; this chapter derives them and does not repeat their rows. Pointer keys `P01`…`P22` are the core §10.1 keys (links in 6.9). Grades as in core. Own kernel timings after 2026-09-21 run at the 2,100 MHz cap (P15 header).

---

## 6.1 Kernel time on GB10: the two ceilings

A kernel is one GPU program launch for one layer class (GEMM, attention, MoE, sampler); the engine picks it at start-up (core §6.1; 6.3 here).

### 6.1.1 Equation

```
t_k ≥ max( B_k / BW_eff ,  F_k / (Peak_dtype · f_clock/f_peak) ) + t_tail
I_k = F_k / B_k                                 arithmetic intensity, FLOP/B
I*  = Peak_dtype · (f_clock/f_peak) / BW_eff    ridge point, FLOP/B
bound(k) = bandwidth if I_k < I*, compute if I_k > I*
```
[Interpretation; roofline assembled from core facts 7 and 12]

| term | meaning | unit | source |
|---|---|---|---|
| `B_k` | DRAM bytes the kernel moves (weights + activations + KV) | B | computed; no own DRAM counter exists (core §3.3) |
| `F_k` | FLOP the kernel issues | FLOP | computed |
| `BW_eff` | achieved read ceiling; name the probe | B/s | 230-245 GB/s decode ruler [Community-measured] (P17 §3) |
| `Peak_dtype` | tensor-core ceiling for the operand dtype | FLOP/s | contested, core fact 12 |
| `f_clock/f_peak` | loaded clock over the clock the peak was taken at | — | ~0.8 under the 2,100 cap [Community-measured; Interpretation] (core fact 12) |
| `t_tail` | fixed device cost per kernel (ramp, tail, sync) | µs | 6.2.2 |

### 6.1.2 Ridge points and the row count where a GEMM turns compute-bound

For a weight-streaming GEMM of M rows over an N×K weight stored at `b_w` bytes per weight, weights-only bytes give `F = 2·M·N·K`, `B ≈ N·K·b_w`, so

```
I(M) ≈ 2·M / b_w          M* = I* · b_w / 2
```
[Interpretation; activation bytes M·(N+K)·2 ignored, they raise `B` and move `M*` up]

| operand path | `Peak` used | `I*` (FLOP/B), `BW_eff` 230-245 | `b_w` (B/weight) | `M*` rows | grade |
|---|---|---|---|---|---|
| BF16 × BF16 | 123 or 213 TFLOPS | 502-926 | 2 | 502-926 | [Interpretation] on contested peak |
| FP8 W8A8 | 214-256 TFLOPS | 873-1,113 | 1 | 437-557 | [Interpretation] |
| NVFP4 W4A4 | 356 TFLOPS achieved CUTLASS GEMM, not a peak | 1,453-1,548 | 0.5625 (4 bits + one FP8 scale per 16) | 409-435 | [Interpretation] |
| W8A16 weight-only (Marlin; math at BF16 rate) | BF16 | 502-926 | 1 | 251-463 | [Interpretation] |
| W4A16 weight-only (Marlin) | BF16 | 502-926 | 0.5625 | 141-260 | [Interpretation] |

Consequences [Interpretation]:
1. Decode rows `M = slots·(k+1)` (64 at 8 slots × k=7) sit below every `M*`: every decode GEMM at M ≤ 64 is bandwidth-bound; read slots and k from the lane card. MoE is further below: rows per expert `ρ = k_top·M/E` (core §4.4).
2. Check against evidence: an ideal FP4 GEMM is 1.03-1.07× slower than Marlin W4A16 at M = 8-256 [Community-measured] (P05 §3), and Marlin NVFP4 MoE runs at 91-96% of a 245.9 GB/s ruler for M ≤ 256 [Community-measured] (P09 §3): FP4 tensor cores cannot help a bytes-bound kernel. The measured FP4-over-Marlin crossover is M = 1,024 with all experts local (P05 §3), above the W4A16 `M*` band: read the table's `M*` as a lower bound for weight-only crossovers [Interpretation].
3. Dense and attention GEMMs in prefill chunks (M = 2,048-8,192) sit above every `M*`: compute rate and tile efficiency decide, which is why prefill is clock-sensitive (elasticity ≈ 0.4, core §4.2). Per-expert MoE GEMMs see `ρ = k_top·M/E` rows on average (56.9 for top-8 of 288 at M = 2,048), below every `M*`; concentrated routing puts some experts above 128 rows (core §4.4).

### 6.1.3 Headroom bound per kernel

```
x_max(bandwidth-bound) = 1 / φ        φ = (B_k / BW_eff) / t_k     fraction of the bandwidth wall reached
x_max(compute-bound)   = 1 / η        η = (F_k / t_k) / (Peak_dtype · f_clock/f_peak)
G_max                  = s · (1 − 1/x_max)                          best end-to-end gain; s = kernel share (6.5.1)
```
[Interpretation]
- Marlin W4A16 MoE at φ = 0.91-0.96 → `x_max` 1.04-1.10 [Interpretation on Community-measured] (P09 §3).
- A C/CUDA fork's plain decode at 18-20 tok/s against a 20.5 tok/s wall (225 GB/s ÷ ~11 GB per token), "~90%+" → `x_max` ≈ 1.03-1.14 for the whole decode [Community-measured] (P15 §4).
- `η` inherits the contested peak: state which peak with every `η`.

---

## 6.2 Launches, graphs and fusion

### 6.2.1 Derivation of core §6.1's fusion term

A chain of n kernels over an activation tensor of `A` bytes, each reading and writing it once:

```
B_unfused ≈ 2·n·A          B_fused ≈ 2·A          ΔB ≈ 2·(n − 1)·A
ΔT_fusion ≈ 2·(n − 1)·A / BW_eff + (n − 1)·c
c = t_launch + t_tail (eager)        c = t_tail (inside a CUDA graph)
A = M · H · bytes_act   [B]          H = hidden size; bytes_act = B per activation element; n = kernels in the chain
```
[Interpretation]

| term | value | grade | source |
|---|---|---|---|
| `t_launch` | NVIDIA's 1-5 µs; unmeasured on GB10 | [Community-measured] | P01 §4 |
| `t_tail` (per small kernel, in graphs) | ≈ 5 µs = ~10 ms ÷ ~2,000 small kernels in an ~82 ms step, DeepSeek V4.1 Flash on 3× GB10, SGLang | derived [Interpretation] on [Community-measured] | P01 §4 |
| Graph coverage | graphed single-node steps: host gap 0.1-2.4% (core §3.4); a graphed 3-node DS V4.1 step read ~95% GPU-busy | [Community-measured] | core §3.4; P01 §4 |

### 6.2.2 Minimum fusion that can show

Decode `A` is small: at M ≤ 64 rows and H in the thousands, `A` < 1 MB, so each removed kernel saves `2A` < 2 MB, < ~8-9 µs of traffic at 230-245 GB/s, the same order as `c`. The removed-kernel count needed to exceed a relative band δ on a step of `T_step`:

```
n_min ≈ δ · T_step / (c + 2·A / BW_eff)
```
[Interpretation]

Worked with the community step above: δ = 4%, `T_step` = 82 ms, `c + 2A/BW_eff` ≈ 5-14 µs → `n_min` ≈ 230-660 kernels removed per step [Interpretation]. Observed: −61 launches per token flat and ~300 launches moved into graphs flat on bandwidth-bound single-node decode [Community-measured] (P01 §4); a combined arm containing an own fused-staging decoder −0.8% C1, −5.5% C8, decoder share never isolated [Measured] (6.6.3). Fusion on a graphed decode step pays only when it also removes bytes the step reads (weights, KV) or removes host work (core §3.4 stop rule: CPU gap < 3%).

Prefill differs (`A` is MBs per pass): a grouped fat-expert kernel replacing ~14 launches per fat expert with 3 per MoE layer gave cold prefill +37-45% on a two-node kit [Community-measured] (P15 §4).

Launch-bound exception: batch-1 small dense decode; a megakernel took Llama-3.2-1B 72.1 → 95.9 tok/s over compiled-and-graphed [Community-measured] (P15 §4). Test: nsys kernel gaps < ~5% of GPU time → fusion cannot pay (P15 §5 lever 7).

---

## 6.3 How a kernel gets selected on CC 12.1

### 6.3.1 The four filters

A kernel runs on GB10 only if it passes all four, in this order; each has its own failure signature.

```
runs(c) = gate(c, CC=12.1) ∧ loadable(cubin(c), 12.1) ∧ fits(c, 101,376 B, 64K regs) ∧ correct(c, shapes)
selected(class) = first c in L_engine(class) with runs(c), unless a user flag names c
```
[Interpretation; filters from P16 §2, P17 §3, P15 §3]

| filter | passes on 12.1 | fails on 12.1 | failure signature | grade |
|---|---|---|---|---|
| 1 gate: "at least X" (`>= 100`, `has_device_capability`) | yes, incl. SM100-minded gates | — | a datacenter path is attempted [Interpretation] | [Code-verified] (P16 §2) |
| 1 gate: "same major" (`family(120)`, `major == 12`, `is_sm120_supported`) | yes | — | inherits 12.0 defaults never validated on 12.1 | [Code-verified] (P09 §3) |
| 1 gate: "exactly 12.0" (`== 120`, `is_sm120()`, `enable_sm120_only`) | no | excluded | silent fallback; or `trap` at launch in device code | [Code-verified] (P16 §2; P15 §2) |
| 1 gate: "family 10.x" (`family(100)`, `is_sm100_supported`) | no | excluded | removed from the auto list | [Code-verified] (P16 §3) |
| 2 loadable | `sm_121a`, `12.0f`, `121f`, `sm_120` baseline, PTX (driver JIT) | `sm_120a`; any 10.x `a`/`f` | load failure; CUTLASS 87a built `120a`: `kErrorInternal` at launch after `can_implement()` passed | [Code-verified]; [Community-measured] (P15 §3) |
| 2 loadable, feature | dense block-scaled FP4 in `12.0f`/`sm_121a` | FP4 `cvt`/block-scale on `121-real` or plain `sm_120`; sparse FP4 outside `sm_121a` | ptxas error; CUTLASS SM120 atoms compile to trap stubs | [Code-verified] (P15 §3) |
| 3 fits | smem ≤ 101,376 B per block, 100 KB per SM | tiles sized for 228 KiB parts | `OutOfResources`; autotuner tactics skipped; "Error Internal" | [Community-measured] (P17 §3) |
| 4 correct | — | exact-CC-excluded datacenter routes when widened; version-specific tiles | HTTP 200 with wrong tokens; no exception | [Measured]; [Community-measured] (P15 §7) |

Derived rules [Interpretation]: (1) PTX-only builds pass filters 1-2 but run old-arch code (ds4 without its Spark target: compute_75 PTX; native +10.8% at 12K prefill [Community-measured], P16 §4). (2) With a family cubin present, a native `sm_121` rebuild changed nothing (16.4 vs 16.3 tok/s); with no SM12x cubin in the image, shipping SM121-built cubins was the whole lever (13.3 → 48.6 tok/s) [Community-measured] (P15 §4). The lever is missing or trapping code, not the `a` suffix (P15 §5 lever 1).

### 6.3.2 Per engine: where the choice is made, logged and overridden

| engine @ pin | mechanism | proof line (every rank) | override |
|---|---|---|---|
| vLLM @ 487ecf187 (lane pin) / 711fc55c (main) | per-layer oracle; first class passing device, scheme, activation, parallel and compiled-kernel checks | `Using '<X>' ... backend out of potential backends`; `cudagraph_mode`; `use_trtllm_attention` | `--attention-backend`, `--moe-backend`, `--linear-backend` (raise if unservable); `VLLM_USE_DEEP_GEMM`; `VLLM_USE_FLASHINFER_SAMPLER` |
| SGLang @ 2909f84 (main) / da64c5cb (lane pin) | `auto` resolved in post-process hooks; `is_sm120()` excludes 12.1, `is_sm120_supported()` includes it | `server_info` backend fields | `--attention-backend` (+prefill/decode), `--moe-runner-backend`, `--fp8-gemm-backend`, `--fp4-gemm-backend`, `--sampling-backend` |
| llama.cpp @ 6e60f35 | no runtime switch; type and batch width pick cuBLAS/MMVF/MMF/MMVQ (≤ 8 columns)/MMQ | build arch; `cuobjdump --list-elf` | build flags for GEMM; runtime `-fa` and `GGML_CUDA_DISABLE_GRAPHS` for attention and graphs |
| EXL3 kit on vLLM (exllamav3 c5d9c657, kit 71962cd) | env flags; major ≥ 10 → `CC_BLACKWELL` shape heuristics; runtime GEMM autotune | no selection line recorded; log the kit variables as launched [Proposed]; MoE policy bound to the decoder `.so` hash (P15 §2) | `EXL3_*`, `GLM53_*` kit variables |

Caches to persist and warm: FlashInfer JIT and autotune, Triton, SGLang JIT, `EXLLAMAV3_TUNE_CACHE`, driver ComputeCache. All [Code-verified] (P16 §2, §5 lever 10; P15 §2-§3). Add under Inductor: a newly registered vLLM `CustomOp` runs its torch fallback unless `+OpName` is in `custom_ops`, because the default is `'none'` [Code-verified @ 34b0690] (P15 §2).

---

## 6.4 SM121 support matrix: known-bad paths at pinned commits

Rows already in core §6.2-§6.3 are not repeated. Status: **excluded** (never auto-selected), **fallback** (a slower path runs), **corrupt** (runs, wrong output), **fails** (error at build, load or launch), **fixed@** (fixed at the named commit; check the image contains it). `→` = workaround.

| path | engine @ pin | status on 12.1 | detail | grade, pointer |
|---|---|---|---|---|
| FlashInfer TRT-LLM MoE, both CuteDSL MoE, CuteDSL NVFP4 linear, DeepGEMM MXFP8, TRT-LLM FP8 MoE | vLLM @ 487ecf187, main | excluded | `family(100)` gates; auto falls to FlashInfer CUTLASS for W4A4 | [Code-verified] P16 §3 |
| `VLLM_CUTLASS` FP8 MoE | vLLM @ 487ecf187 | excluded | `cutlass_group_gemm_supported()` false for cc ≥ 110; auto takes FLASHINFER_CUTLASS (static W8A8), TRITON (dynamic), DEEPGEMM else TRITON (block) | [Code-verified] P16 §2 |
| MXFP8 MoE | vLLM pin and main | fallback | Marlin W8A16; fix PR #43911 open | [Code-verified] P16 §2 |
| GDN prefill FlashInfer backend | vLLM before #55715 (`f6326f53b`, 2026-09-08); the lane pin 487ecf187 predates it [Interpretation] | fallback | Triton/FLA ran silently; after the fix warm +5.0%, cold 6K +7.1%; GDN models only; → a build containing #55715, verified from the backend line | [Community-measured] P15 §4 |
| DFlash drafter TP | vLLM @ 487ecf187 | ignored flag | proposer never reads `draft_parallel_config`; drafter built at world TP (heads padded 36/9); no workaround at the pin | [Code-verified] P16 §2 |
| vLLM `_C`/`_moe_C` build | vLLM @ 34b0690 (P15) and 487ecf187 (P16), CUDA ≥ 13 toolkit | `12.0f` only | `TORCH_CUDA_ARCH_LIST=12.1a` still yields `12.0f`; published v0.27.1 aarch64 image: zero `sm_121` cubins (family cubins run [Interpretation]); → patch the supported-arch list (PR #38484, open) only where a feature needs `121a`; verify with `cuobjdump` | [Code-verified]; [Community-measured] P15 §2; P16 §2 |
| MXFP8 MoE runner `auto` | SGLang @ 2909f84 | probable wrong default | `flashinfer_trtllm` with no SM12x branch; untested; → one boot log decides | [Code-verified]; [Interpretation] P16 §2, §8 |
| trtllm-gen MoE, `--dsv4-attn-backend trtllm`, FlashInfer MegaMoE | SGLang @ 2909f84 | unavailable; dsv4 asserts, MegaMoE raises | SM100-only (dsv4: SM100/103) | [Code-verified] P16 §2 |
| DeepGEMM JIT | SGLang @ da64c5cb | fails | no SM120 API probe; UE8M0 scale only for `== 120` (wrong scale format on GB10); scale assertion and CUDA error 719 reproduced in a small grouped FP8 × MXFP4 GEMM; fixed@ #39482 (`3d1b9e7549a3`); → `SGLANG_ENABLE_JIT_DEEPGEMM=0` before the fix; separately, `sm121_*` names vs `sm120_*` headers fail on a cache miss | [Code-verified]; [Community-measured] P16 §2, §7 |
| stock fused MoE concurrency | exllamav3 @ c5d9c657 | bounded | `num_sms / MOE_SMS_PER_EXPERT` = 48 / 8 = 6 expert groups; changing it rebuilds the extension; upstream `58d4d732` adds `MOE_MAX_SMS_PER_EXPERT 32` and 32/64-row instances | [Code-verified]; [Interpretation] P16 §2 |
| thin-fast MoE | EXL3 kit @ 71962cd (A/B at PR #217 head 3d6ffbd) | TP2-only, bimodal | dispatches only at CC == (12, 1), K == 4, dims multiple of 256; +7.71 to +14.59% medians, ~1 run in 4 is 15-30% below stock; → ≥ 30-run histogram before keeping (P15 §5 lever 9) | [Code-verified]; [Community-measured] P16 §5 lever 5 |
| CUTLASS MoE tactics; MLA chunked prefill | TensorRT-LLM playbook images (rc5-rc21); main @ f5ca105 | fails; excluded | tactics sized for 228 KiB (filter #12704 first in `v1.3.0rc23`); chunked-prefill allowlist omits 121; → `moe_config.backend: TRITON` or an image ≥ `v1.3.0rc23` | [Community-measured]; [Code-verified] P16 §2 |
| CuTe DSL `MmaFP8Op` | CuTe DSL ≤ 4.5.2 | fails (segfault) | builds an SM89 atom; reported fixed in 4.6, unverified on GB10; → inline-PTX `kind::f8f6f4` (the only FP8 attention measured on GB10) | [Code-verified]; [Community-measured] P15 §3 |
| Inductor max-autotune GEMM templates | PyTorch @ c6959ab | excluded | `is_big_gpu` needs ≥ 68 SMs; GB10 has 48 → cuBLAS | [Code-verified] P15 §3 |
| `flex_attention` | torch 2.11.0+cu130 | corrupt | cosine 0.92-0.97 vs SDPA, no error; → SDPA or a Triton kernel | [Community-measured] P15 §3 |
| `T.gemm_v1(transpose_B=True)` | TileLang (community GB10 port) | corrupt | switch to `T.gemm_v2` | [Community-measured] P15 §3 |
| NoPE decode | FlashInfer 0.6.17 (installed on the TP3 lane) | fallback (padding) | installed build predates the native path by 278 commits; fixed head grid pads 22 → 32 heads, 512 → 576 query width; → FlashInfer ≥ v0.7.0rc1 removes decode padding, prefill stays padded | [Historical diagnostic] P16 §7; [Code-verified] P06 §3, §5 lever 3 |

---

## 6.5 When authoring is justified, and in what

### 6.5.1 The gain has to clear the comparison band

From core §6.4's Amdahl form and 6.1.3:

```
G       = s · (1 − 1/x)                     predicted end-to-end gain
s_min   = δ / (1 − 1/x)                     smallest share at which speedup x is detectable
x_need  = 1 / (1 − δ/s)                     speedup needed at share s; no x suffices if s ≤ δ
x       ≤ x_max                             6.1.3
```
[Interpretation]

`δ` is the band of the comparison design, not a constant (core §7.1):

| comparison | δ | `s_min` at x = 2 | grade |
|---|---|---|---|
| prefill, within one boot | ±2% | 4% | [Community-measured] (P04 §6) |
| own decode drift, opening vs closing baseline | −1% / +4% → use 4% | 8% | [Measured] (P15 §4) |
| boot to boot, identical config | ~8% | 16% | [Community-measured] (P14 §4) |
| prefill start to start, one stack | ±20% | 40% | [Community-measured] (P04 §6) |

A kernel swap that needs an engine restart is judged against the boot-to-boot row unless each arm gets repeated boots (core §7.1).

### 6.5.2 Resource arithmetic before code

```
blocks/SM = min( ⌊65,536 / (T · ⌈r⌉₈)⌋ ,  ⌊102,400 / s_blk⌋ ,  ⌊1,536 / T⌋ ,  B_max )
s_blk     = stages · (BM·BK·b_A + BN·BK·b_B) + s_epi   ≤ 101,376 B
```
[Code-verified limits (P17 §3); Interpretation for the form; 101,376 = 102,400 − 1,024 suggests a per-block reservation the `s_blk` term omits, unrecorded on GB10 (Gaps)]

| term | meaning | unit |
|---|---|---|
| `T` | threads per block | — |
| `r`, `⌈r⌉₈` | registers per thread; allocation rounds up in steps of 8 | registers |
| `s_blk` | shared memory per block | B |
| `B_max` | resident blocks per SM: 24 or 32, contested; the difference matters only for CTAs of ≤ 32 threads | — |
| `stages`, `BM`, `BN`, `BK`, `b_A`, `b_B` | pipeline depth, tile dims, operand bytes | —, elements, B |

Worked tiles [Interpretation, arithmetic]: BF16 128×128×64 is 32,768 B per stage, so 3 stages (98,304 B) fit and 4 (131,072 B) fail; FP8 at the same tile fits 6 stages. Tiles and stage counts that overshoot 101,376 B are the most common port failure (a Triton MoE config at 147,456 B; CUTLASS `StageCountAutoCarveout` at 102,400 B, 1 KiB over, fixed in CUTLASS 4.4.2) [Community-measured] (P15 §3-§4).

Register cliff: see 6.6.4.

### 6.5.3 Toolchain by need (at the pins in P15 header)

| need | use | why | avoid |
|---|---|---|---|
| FP8 or block-scaled FP4 GEMM | CUTLASS C++ SM120 builders (TN, 1×1×1) or CuTe DSL ≥ 4.5 | SM120 builders and GeForce examples (C++); CuTe DSL is the only DSL at the pins exposing warp `mma.sync` and SM12x block-scaled MMA with TMA plus GeForce templates [Interpretation]; 356 TFLOPS dense NVFP4 reached on one GB10 [Community-measured] | `tcgen05`/TMEM designs (core §6.3) |
| BF16 GEMM | cuBLAS, or SM80-style `mma.sync` | CUTLASS SM120 builders are F8F6F4/block-scaled only, TN, cluster 1×1×1 [Code-verified]; cuBLAS already picks `cutlass_80_*` on SM12x [Community-measured] (P09 §3) | a new BF16 or mixed FP8×BF16 collective (own dossier estimate for the mixed-input kernel: weeks) |
| fused elementwise, MoE glue, attention variants | Triton ≥ 3.7 | MMAv2, `sm_121a`, no env var [Community-measured] | Triton 3.5.x; `num_ctas > 1` |
| block-scaled FP4 in a Python DSL | TileLang `T.mma_gemm_blockscaled` or CuTe DSL | SM120 block-scaled paths registered [Code-verified] | TileLang `gemm_v1(transpose_B)` |
| SGLang-integrated kernel | SGLang JIT (`sm_121a`, content-addressed cache) | in-tree skill requires tests and a benchmark [Code-verified] | plain `sm_120` (trap stubs) |
| llama.cpp, ds4 | CUDA C++, `-gencode=arch=compute_121a,code=sm_121a` | GB10 tables need 121a code [Code-verified] | `-arch=sm_121a` (can drop the `a`) |

Not ready at the pins: ThunderKittens (`ARCH=SM120` builds `sm_120a`; SM121 needs open PR #204), cuTile (base `sm_121` only; an sm_120 study reached 53% of FA2), Mojo NVFP4 (needs PTX 9.1, a CUDA 13.1+ driver; r580 is 13.0) [Code-verified; Community-measured] (P15 §3).

### 6.5.4 Decision procedure

Stop at the first step that yields the answer. Order follows P15 §5 (ceiling and shares before adopt and tune); core §6.4 lists adopt and tune before profiling: the bound of steps 5-6 is cheap and screens adopt and tune candidates too.

1. **Lane card**: engine version, FlashInfer version, image digest, spec method and k, slots (core §10.2); confirm installed versions on the node, never public head.
2. **Dispatch audit, every rank**: 6.3.2 proof lines for attention, MoE, dense GEMM, sampler, graphs; `cuobjdump --list-elf` on every deployed `.so` and JIT cache; grep the dispatch path for `== 120`, `is_sm120()`, `family(100)`. Output one row per layer class: kernel, native or fallback, deciding filter (6.3.1).
3. **Fallback → fix dispatch, not code** (flag, env, cubin rebuild, upstream fix at a pin); one variable per boot; P16 §6 run steps.
4. **Widening an exact-CC or datacenter gate is a correctness change**: repeated-output gate before any timing (trap 7).
5. **Shares**: profile off traffic (core §7.6); `s` per kernel class for the phase that matters; never score profiled runs.
6. **Bound**: `x_max`, `G_max = s·(1 − 1/x_max)`, `s_min` for the band you can afford (6.5.1); drop candidates with `G_max < δ`.
7. **Adopt** a kit or upstream kernel through gates; search PRs for the kernel name and "sm_121"/"sm120" first (P15 §5 lever 3; §6 preconditions).
8. **Tune data** (GB10 configs via `VLLM_TUNED_CONFIG_FOLDER`; per-shape autotune with a persisted cache) (P15 §5 lever 4).
9. **Author only if** `ncu` on a component harness shows the kernel far below both ceilings (else bytes ÷ duration vs the probe ceiling) and `G_max > δ`; toolchain from 6.5.3; resource arithmetic (6.5.2) first; declare `__launch_bounds__(T, blocks_min)`.
10. **Build offline** (trap 12); record registers, spills, smem and the deployed SASS.
11. **Gate**: frozen-tolerance reference on real shapes; changed-data graph replay bitwise equal to eager; candidate-filtered compute-sanitizer; repeated-output model gate; unprofiled A/B/A serving arm (core §7.3). The serving arm decides.
12. **Record** predicted vs measured `G` per metric and the selection lines before and after (core §8 step 7).

---

## 6.6 Worked example — GLM 5.3 Flash EXL3 TP3 lane, node-1/2/3

Inputs: 2026-09-20 GLM TP3 startup and prefill optimization (also filed as "GLM 5.3 Flash V3 retained startup and prefill improvements, V2 decoder"); 2026-09-16 GLM 5.3 Flash TP3 combined-runtime qualification; 2026-09-20 GLM TP3 frontier dossier v2. Engine: vLLM @ 487ecf187 with the EXL3 kit, TP3/EP3.

### 6.6.1 Step 2 output (dispatch audit, as recorded)

| class | kernel that runs | status | grade, pointer |
|---|---|---|---|
| sparse MLA attention | `FLASHINFER_MLA_SPARSE_SM120`, KV `fp8_ds_mla` | native; FlashInfer 0.6.17 pads 22 → 32 heads, 512 → 576 (no native NoPE) | [Historical diagnostic] P16 §2, §7; P06 §3 |
| other attention group | FA2 | native | [Historical diagnostic] P16 §2 |
| decode MoE | cooperative kernel on policy rows, stock 16-row kernel elsewhere | custom | [Historical diagnostic] P16 §2 |
| thin-fast MoE | off (TP3 launcher unsets it) | excluded by recipe | [Code-verified] P16 §2 |
| prefill MoE | E3 grouped fat-expert, `EXL3_TEMP_ROWS_FUSED=64` | kit | [Historical diagnostic] P16 §2 |
| dense side weights | FP8 weight-only via Marlin W8A16 | weight-only | [Code-verified]; [Historical diagnostic] P16 §2 |
| KDA projections at TP3 | two stay BF16 (merged views violate Marlin alignment) | fallback | [Historical diagnostic] P16 §2 |
| drafter | built at world TP, heads 36/9, whatever the draft-TP setting says | flag ignored | [Code-verified] P16 §2 |

### 6.6.2 KDA input projection: Marlin → BF16 GEMM above M = 512

Measured [Measured] (2026-09-20 GLM TP3 startup and prefill optimization):
- Component (N 8,726 × K 4,096 per rank): BF16 GEMM 2.17-2.25× faster than Marlin at M = 2,048; +2.26 GiB per node.
- Group arm (the switch + persistent compute cache + indexer warm-up + lazy import): effective prefill 1,585.84 → 1,700.04 tok/s at ~7.6K, 1,695.83 → 1,830.38 at ~95K; 5 matched cold samples per size; pre-cap.

Implied share of the switched kernel [Interpretation]: time ratio at ~95K = 1,695.83 / 1,830.38 = 0.9265, so `G` = 7.35%; with x = 2.17-2.25, `1 − 1/x` = 0.539-0.556, so `s` = 13.2-13.6%. At ~7.6K: `G` = 6.72%, `s` = 12.1-12.5%. Upper side: the three start-up changes are assumed not to touch steady prefill, and x is taken at M = 2,048 only.

Cross-check [Interpretation on Measured]: node-1 prefill kernel inventory (torch profiler, first 8 cold-prefill iterations of one request, opening arm before the switch per the packet's raw traces; sums overlap, not critical-path shares) totals 10,344 ms, with Marlin ~1,697-1,812 ms = 16.4-17.5% (P15 §4). The implied 12-14% for one Marlin GEMM class sits inside the Marlin total. A pre-change prediction from the inventory alone would have been `G_max` ≤ 0.175 × 0.556 = 9.7%; measured 6.7-7.4%, inside the bound.

Same inventory, step 6 applied [Interpretation]:

| kernel class (node-1) | share of summed kernel time | `G` if 2× faster | verdict at δ = 2% (within-boot prefill; a restart-gated swap faces the boot-to-boot band, 6.5.1, unmeasured for own prefill) |
|---|---|---|---|
| NCCL all-reduce | 25.8% | — | not a kernel-authoring target (chapter 5) |
| Marlin (all) | 16.4-17.5% | ≤ 8.8% | switched in part (above) |
| grouped fat-expert gate/up + down | 13.9% | ≤ 7.0% | own tile variants lost 4-219% at 8,192 rows on the DS V4.1 lane [Measured] (P15 §4); stock is tuned |
| EXL3 fused MoE | 9.0% | ≤ 4.5% | marginal |
| two TileLang kernels | 7.9% | ≤ 4.0% | marginal |
| sparse-MLA prefill | 6.8% | ≤ 3.4% | marginal; FlashInfer ≥ v0.7.0rc1 keeps prefill padded at the head grid (P06 §5 lever 3) |
| DeepGEMM TF32 | 2.4% | ≤ 1.2% | reject: `G` ≤ 2.4% even as x → ∞, at the band edge |
| FlashKDA | 1.9% | ≤ 0.9% | reject: `s` < δ, no x suffices |

Contested boundary. The shipped threshold (512) was placed in an empty M band of the TP2 workload, not measured as a crossover, and carried to TP3 without re-derivation [Interpretation] (2026-09-20 GLM TP3 frontier dossier v2; P09 §5 lever 1). Two estimates, neither measured:
- Dossier one-point calibration [Interpretation] (P07 §5 lever 4): Marlin cost `⌈M/64⌉ · N·K / BW_eff` (weights re-read per 64-row tile; `BW_eff` ≈ 212 GB/s from the M = 7,168 point) against the BF16 floor `2·N·K / BW_eff` → BF16 overtakes from M ≈ 129; five unmeasured inputs, one contradicted by Marlin's M-dependent reduce choice.
- Roofline bound [Interpretation]: at M between the two, BF16 is bandwidth-bound (`M` < `I*`) and Marlin runs at compute efficiency `η_Marlin`, so the crossover is

```
M_c ≈ η_Marlin · I*_BF16          η_Marlin ≤ 1/x = 0.44-0.46 (from the 2,048-row ratio, if η_BF16 ≤ 1)
M_c ≲ 0.46 × (502-926) ≈ 230-430  (× ~0.8 under the cap ≈ 185-340)
```
Both put the crossover below 512. The roofline figure is an upper bound (`η_Marlin` ≤ 1/x), rests on the contested peak and ignores activation bytes, so it does not decide between ~129 and 185-430. Deciding test [Proposed]: component sweep M = 64-2,048 on all three ranks, graph replay, ≥ 200 samples near the crossover, under the lane's cap; keep the boundary above the largest captured graph shape (64 rows) so capture and replay agree (P09 §5 lever 1). The lane card files it as a maintenance-window lever, with the 128 boundary as [Proposed].

### 6.6.3 A component win that could not show: the fused-staging decoder

Measured [Measured] (2026-09-20 GLM TP3 startup and prefill optimization, arm C): agent-written decoder variant, exact and sanitizer-clean; component time ratio 0.925 at 1 row, 0.95-0.98 at 2-7 rows, 0.98-0.996 at 8-56 rows. Arm C (startup/prefill group plus the decoder pair) vs opening baseline: C1 coding 44.291 → 43.930 tok/s, prose flat, C8 aggregate 111.757 → 105.602 (−5.51%); own drift band −1% / +4%; the decoder's own effect was never isolated. Rejected at group level.

What 6.5.1 predicts [Interpretation]: the decode MoE share on this lane is unmeasured. Stand-in: the same model's community TP3+EP decomposition (NVFP4, Marlin W4A16 experts, DFlash2 k=7): MoE 38.4-39.1% of a C1 step, 62.4% at C8 [Community-measured] (P05 §4). `G_max` at 1 row ≈ 0.39 × 0.075 = 2.9%; at C8 rows (M up to 64) ≤ 0.624 × 0.02 = 1.2%. A DeepSeek V4.1 TP3 share (~27 of ~82 ms ≈ 33%, P01 §4) gives 2.5% and 0.7%. All are under the 4% band: the component result could not have shown end to end, so only the serving arm could decide. Record the lane's decode MoE share as a card constant [Proposed].

### 6.6.4 Register cliff on the cooperative decode MoE kernel

Receipt [Measured] (2026-09-16 GLM 5.3 Flash TP3 combined-runtime qualification): 63 registers, 0 spills, 48 B static + 24,576 B dynamic smem, 512 threads, only `__launch_bounds__(512)`.

6.5.2 applied [Interpretation]: registers ⌊65,536 / (512 × 64)⌋ = 2; smem ⌊102,400 / 24,624⌋ = 4; threads ⌊1,536 / 512⌋ = 3 → 2 blocks per SM, register-limited. At 65 registers ⌈65⌉₈ = 72 → ⌊65,536 / 36,864⌋ = 1 block, with "0 spills" still logged. Guard: `__launch_bounds__(512, 2)` [Proposed] (P15 §5 lever 5).

---

## 6.7 Traps

1. Taking the quant name, a trace kernel name or Marlin's "no native FP4" line as the kernel that runs; only the selection line on every rank proves it [Code-verified] (P16 §7).
2. Chasing the `a` suffix where a family cubin exists (16.4 vs 16.3 tok/s) [Community-measured] (P15 §4).
3. Targets `sm_120a` (no load), plain `sm_120` (trap stubs), `121-real` for FP4 (ptxas rejects) [Code-verified] (P15 §7).
4. `TRITON_PTXAS_PATH`: inert on Triton ≥ 3.6 (the override is `TRITON_PTXAS_BLACKWELL_PATH`); use Triton ≥ 3.7 [Code-verified] (P15 §3).
5. Measuring a new vLLM `CustomOp` under Inductor without `+OpName`: the torch fallback is what ran [Code-verified] (P15 §2).
6. Copying a MoE backend flag across quant formats or models: the same NVFP4 MoE backend that passes on one lane is reported to corrupt another build [Community-measured] (P16 §6).
7. Widening any exact-CC or datacenter guard as a speedup: the TRT-LLM sparse-decode route passed an external six-shape numeric test and still gave token-0 loops in 2 of 4 coding outputs [Measured] (P15 §4).
8. "0 spills" read as safe: +2 registers halves residency (6.6.4) [Measured] (P15 §7).
9. Component ratio read as the result: the combined arm with a clean decoder read −5.51% C8 [Measured]; 2.76× GDN kernel moved 8K TTFT ~3%; 3.91× kernel moved its workload 1.010× [Community-measured] (P15 §7). Compute `G_max` first.
10. Tile variants chosen by intuition: three clean variants lost to the stock 64-row / 4-stage tile by 4-219% at 8,192 rows [Measured] (P15 §7).
11. Stale resolvers in a layered base image: a BF16-only SM121 resolver pre-empted the FP8-capable branch and broke graph warm-up [Historical diagnostic] (P16 §7). Assert and remove the block.
12. Compiling on a serving node: cold CUTLASS JIT beside a 23 GB model was OOM-killed; `cicc` ran 1.5-6 GB per process [Community-measured] (P15 §7). Build offline; persist and warm every JIT cache; unwarmed shapes compile inside live requests and risk TP desync [Historical diagnostic] (P16 §7).
13. Component A/Bs without interleaved arms: chip state moved a large-M BF16 GEMM ~40% at the same reported clock [Community-measured] (P09 §3).
14. Citing a retracted report: the DeepGEMM "family-12 gate" framing was retracted by its reporter, while the separate K = 2560 corruption window stands [Community-measured] (P15 §7).
15. Wrong gate strictness: E3's `atomicAdd(float4)` scatter is nondeterministic by construction, so gate on tolerance, not bitwise [Interpretation] (P15 §2); a clamped `sigmoid` passed cos ≥ 0.95 at 3.36×, 1.04× at cos ≥ 0.99 [Community-measured] (P15 §7).

---

## 6.8 Where the lane's numbers live

Read from the lane card: engine version; FlashInfer version; image digest; spec method and k; slots; chunk size. Proposed card fields this chapter needs: decode kernel-class shares (C1, C8), prefill kernel-class shares, per-class selection lines, custom-kernel register/smem receipts, Triton version (the TP3 lane's receipt does not record it).

## 6.9 Evidence links

| key | page |
|---|---|
| P01 | [CUDA graphs and launch overhead](../01-cuda-graphs-and-launch-overhead.md) |
| P04 | [Prefill optimization](../04-prefill-optimization.md) |
| P05 | [Decode optimization: the bytes model](../05-decode-optimization.md) |
| P06 | [Attention backends on SM121](../06-attention-backends-sm121.md) |
| P07 | [Linear attention and hybrid layers](../07-linear-attention-and-hybrid-layers.md) |
| P08 | [MoE dispatch](../08-moe-dispatch.md) |
| P09 | [GEMM backends and quant formats](../09-gemm-backends-and-quant-formats.md) |
| P14 | [Profiling protocol](../14-profiling-protocol.md) |
| P15 | [Kernel authoring on SM121](../15-kernel-authoring-sm121.md) |
| P16 | [Engine dispatch maps](../16-engine-dispatch-maps.md) |
| P17 | [SM121 hardware facts](../17-sm121-hardware-facts.md) |
| P21 | [Quality and correctness gates](../21-quality-and-correctness-gates.md) |

---

## Gaps

1. Tensor-core peak contested (BF16 ≈123 vs ≈213 TFLOPS): every `I*`, `M*`, `η`, `M_c` is a band [Proposed: needs measurement — register-resident probe per dtype at the lane cap].
2. No own bandwidth ceiling on a node; `φ` uses community probes [Proposed: needs measurement].
3. `t_tail` derived from one community decomposition; `t_launch` unmeasured on GB10 [Proposed: needs measurement].
4. No own decode kernel-class shares on any lane; 6.6.3 borrows the same model's community TP3+EP shares (P05 §4) [Proposed: needs measurement — per-rank decode profile at C1 and C8].
5. The prefill inventory is overlapping summed kernel time under the torch profiler, not critical-path share; its arm (opening, pre-switch) is in the packet's raw traces, not on P15 §4.
6. KDA crossover: 512 shipped vs ~129 (dossier one-point) vs ≤ 185-430 (roofline bound); `η_Marlin` below M = 2,048 unmeasured; sweep not run.
7. No own `ncu` capture of a production kernel; `dram__bytes_*` availability on GB10 unknown.
8. Resident blocks per SM (24 vs 32) unresolved (CTAs ≤ 32 threads only).
9. Unverified paths: SGLang MXFP8 MoE `auto` and explicit `trtllm_mha`; DeepGEMM grouped MoE at K = 2560; vLLM main `DEEPGEMM_MXFP4`; B12X NVFP4 MoE; `VLLM_MARLIN_USE_ATOMIC_ADD` (correctness vs speed); CuTe DSL `MmaFP8Op` ≥ 4.6; cluster codegen on sm_121 (P15 §8; P16 §8).
10. Whether the own llama.cpp build carried 121a code was not recorded.
11. Per-block shared-memory reservation on GB10 not recorded (101,376 vs 102,400 B suggests 1 KiB); the 6.5.2 smem term omits it [Proposed: read it in the P17 §6 device dump].

## Sources

Wiki pages (read 2026-09-23): P15 §1-§5, §7 in full; P16 in full; P17 §3; P09 §3, §5 lever 1; P05 §3-§4 rows; P06 §3, §5 levers 2-3; P07 §5 lever 4; P01 §4 rows; core §1-§10. Lane card for the TP3 lane (private input, used only as worked-example data).

Own packets (date and title): 2026-09-16 GLM 5.3 Flash TP3 combined-runtime qualification; 2026-09-16 DeepSeek V4.1 Flash EXL3 TP3 prefill profiling and tuning; 2026-09-20 GLM TP3 startup and prefill optimization (also filed as GLM 5.3 Flash V3 retained startup and prefill improvements, V2 decoder); 2026-09-20 GLM TP3 frontier dossier v2; 2026-08-27 Flash Next upstream recipe audit.

Community and code sources are cited through their wiki rows (P15 §4, §9; P16 §4, §9) at the pins stated there.
