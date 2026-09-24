---
title: The Science of GB10 Inference — chapter 04, prefill chunk economics
created: 2026-09-23
status: checked
type: chapter
extends: core §4 (equations 4.1-4.7 are referenced by number, not restated)
---

# Chapter 04 — Prefill: chunk economics

Pointer keys as in core §10.1: P04 = [Prefill optimization](../04-prefill-optimization.md), P03 = [Inter-node communication over DAC](../03-inter-spark-communication.md), P08 = [MoE dispatch](../08-moe-dispatch.md), P09 = [GEMM backends and quant formats](../09-gemm-backends-and-quant-formats.md), P10 = [KV cache and prefix caching](../10-kv-cache-and-prefix-caching.md), P11 = [Memory on unified memory](../11-memory-on-unified-memory.md), P13 = [Clocks, thermal, power](../13-clocks-thermal-power.md), P20 = [Model architecture cards](../20-model-architecture-cards.md), P22 = [Serving agent workloads](../22-serving-agent-workloads.md). WR = the 2026-09-23 workflow review. Units: µs per token = 10^6 / (tok/s). "Worked lane" = GLM 5.3 Flash EXL3 4 bpw experts, TP3/EP3 on node-1/2/3, vLLM + EXL3 overlay (§7).

---

## 1. Step-time model, integrated

### 1.1 The chunk the engine actually runs

```
M_eff (solo cold request) = min(MNBT, LPT)   on vLLM builds without PR #57951 (merged 2026-09-22)
M_eff (solo cold request) = MNBT             on builds with it
M_eff (concurrent)        = the LPT grant, minus decode rows, per scheduler policy
```
[Code-verified] for the post-#57951 rule at the P04 vLLM pin; the pre-#57951 cap is [Interpretation] in P04 §2 ("Solo requests"), confirmed [Measured] on the worked lane: at MNBT 4,096 / LPT 2,048 the scheduler trace grants 2,048-token steps (2026-09-20 startup and prefill optimization). The concurrent line is [Interpretation]; on the worked lane the fair mixed-prefill policy also limits newcomer work per step (read from the lane card: chunk size). Consequence: an "MNBT ladder" on a pre-#57951 build with LPT fixed below MNBT is a pool experiment, not a chunk experiment (a community MNBT 8,224 → 2,048 arm with LPT 1,024 in both arms moved only memory [Community-measured], P04 §4, single-node DeepSeek V4 Flash MNBT 8,224 → 2,048 row). Hybrid alignment can shorten steps further; read the logged line (core §4.6 step 1).

### 1.2 TTFT as a sum over chunks

Summing core 4.1 over `n = P/M` chunks, chunk `j` at prefix `P_j = j·M`, and adding a per-step read of the cached prefix (`c_kv·P_j`, KV and indexer keys, independent of `M`):

```
TTFT_cold(P) = t_0 + (P/M)·t_fixed + a·P + (b/2)·P² + (c_kv/(2M))·P²              (4.8)

1/R_avg(P)   = TTFT_cold/P = t_0/P + t_fixed/M + a + (b/2 + c_kv/(2M))·P            (4.9)

τ_marg(P)    = d TTFT/dP  = t_fixed/M + a + (b + c_kv/M)·P                          (4.10)
```
[Interpretation], derived from core 4.1 (P04 §8).

| term | unit | meaning |
|---|---|---|
| `t_0` | s | per-request intercept (template, tokenization, first-step and tail work) |
| `t_fixed` | s per step | everything in a step that does not scale with `M` |
| `a` | s per token | per-token cost at zero prefix |
| `b` | s per (token × prefix token) | attention and indexer growth |
| `c_kv` | s per prefix token per step | re-read of the cached prefix each step: `KV_bpt_read / BW_eff`. On the worked lane the full-prefix read is the indexer keys, ≈363 B per token: the 11 sparse-MLA layers gather only the top 2,048 (+ tail) rows per query (P20 §4.1), a cost that scales with `M` and sits in `a`. ≈363 B gives ≈1.5 ms per step at 1M; even the full ≈7.6 KB row (7,216 B `fp8_ds_mla` + ≈363 B) gives ≈33 ms, ≈1% of a ≈3 s step there [Interpretation]; dropped below |
| `R_avg` | tok/s | whole-context effective prefill (what a cold ladder reports) |
| `τ_marg` | s per token | cost of the next token at depth `P` (warm turns, "next chunk at depth" benchmarks) |

Two consequences:
- The depth term of the **marginal** cost is twice that of the **average** (`b·P` vs `b·P/2`). A next-chunk-at-depth figure and a whole-context figure at the same `P` are different quantities; never compare them (P04 §6 record rule).
- Every `M`-dependence in (4.9) sits in `t_fixed/M`. The model has no term that makes a larger chunk slower; a ladder that runs the other way means `a` depends on `M` (§3) or the transient is unbounded (§4.2).

### 1.3 The amortization law

```
R(M) / R(∞) = M / (M + h)          h = t_fixed / a_P,  a_P = a + t_0/P + (b/2)·P   (tokens; the chunk at half the asymptotic rate)
gain of doubling M = h / (2M + h)
M_stop (doubling gain < noise floor n_f) = h·(1/n_f − 1)/2  ≈ 24.5·h at n_f = 2%
```
[Interpretation], from (4.9) at fixed `P`. `a_P` (s per token) is the per-token cost at depth `P`, which is what a per-depth rung fit returns (§7.1), so `h` depends on depth.

| h (tokens) | doubling gain at M = 1,024 / 2,048 / 4,096 | M_stop at 2% floor |
|---|---|---|
| 110 | 5.1% / 2.6% / 1.3% | ≈2.7K |
| 140 | 6.4% / 3.3% / 1.7% | ≈3.4K |
| 316 | 13.4% / 7.2% / 3.7% | ≈7.7K |

`h` is a per-lane, per-depth constant: read from the lane card: chunk size and the chunk ladder; fit it from two rungs at one depth with repetitions (§8 step 2). The ±2% floor is the within-boot whole-context prefill floor (P04 §6 pass gate); use the lane's own start-to-start spread where larger.

### 1.4 What `t_fixed` contains

```
t_fixed ≈ t_launch·N_kern + N_coll·t_eff + N_sync·t_sync + t_stream,exposed + t_sched
t_stream,exposed ≤ W_rank / BW_eff
```
[Interpretation]. Terms: `t_launch` s per kernel launch not hidden by graphs; `N_kern` launches per step; `N_coll` collectives per step; `t_eff` s per collective (latency part); `N_sync` host syncs per step; `t_sync` s per sync; `t_stream,exposed` s of weight streaming not overlapped with compute; `W_rank` weight bytes per rank (B); `BW_eff` B/s; `t_sched` s of scheduler and host work per step.

| component | worked-lane size | grade, pointer |
|---|---|---|
| collectives, latency part | ≈91 forward collectives × 130-150 µs ≈ 12-14 ms; message bytes scale with `M` and belong in `a` | decode-step counts [Code-verified] and `t_eff` [Community-measured] from core §5.1; transfer to prefill steps [Interpretation]: the 130-150 µs comes from decode all-reduces on the LL protocol, while prefill all-reduces run on Simple (P03 §2), so the prefill latency part is unmeasured |
| host syncs | upstream ExLlamaV3 tiers issue one `expert_count.tolist()` per MoE layer per prefill chunk, 42 per step on GLM (42 MoE layers, P20 §4.1); those tiers post-date the extension pin of the own GLM kit, whose grouped fat-expert path is described as host-sync-free, so the worked lane's sync count is unknown; per-sync cost unmeasured | [Code-verified] upstream at the P04 ExLlamaV3 pin (P04 §2 ExLlamaV3, "Dating"); lane count and cost [Proposed: needs measurement] |
| exposed weight stream | upper bound 54.25 GiB/rank ÷ 230-245 GB/s ≈ 0.24-0.25 s | weights [Measured] (P20 §4.2); `BW_eff` [Community-measured] (core fact 7) |

The fitted `t_fixed` on the worked lane is 67-84 ms (§7.1), about one third of the stream bound. Either most of the weight stream overlaps compute, or the MoE path re-reads weights per tile so that its cost lands in `a` (§2.3) [Interpretation]. The profile that decides: per-step GPU time at `M` = 1,024 and 4,096 with the MoE kernels isolated (P04 §8, the per-step cost-model question).

### 1.5 Compute share from clock elasticity

```
Δt/t = φ · (f_0/f − 1)          φ = share of step time that scales with SM clock
```
[Interpretation]; `f_0` = loaded stock SM clock, `f` = loaded capped clock (MHz). Worked lane, cold ~250K, one request per arm (quick A/B): the loaded medians were stock 2,392-2,411, 2100 cap 2,080-2,086, 1900 cap 1,891 MHz (P13 §4), so the true deltas are −13.2% and −21.2%, slightly larger than the nominal −12.5% and −21%. Rate −4.9% (+5.15% time) gives φ = 0.34 (0.36 on nominal clocks); −9.1% (+10.0% time) gives φ = 0.37 (0.38 nominal) [Interpretation from Measured inputs] (2026-09-21 GPU clock-cap sweep, GLM 5.3 Flash EXL3 TP3; P13). About 63-66% of the prefill step is clock-insensitive (DRAM, fabric, host). φ is a time share; core 4.2's "elasticity ≈ 0.4" is the rate change per clock change (4.9/12.5, 9.1/21), a different ratio from the same data. Use: a kernel change that cuts FLOPs alone can win at most φ of step time; a byte or sync cut targets the other 1 − φ.

### 1.6 MFU, worked

```
MFU = R_eff · F_tok,avg(P) / (N_nodes · Peak_dtype · f/f_pk)          (core 4.2)
F_tok,avg(P) ≈ F_w + F_a0 + (s_a/2)·P
```
[Interpretation]. Terms: `F_tok,avg` FLOP per prompt token averaged over a `P`-token prompt (GFLOP); `F_w` = 2 × active parameters (33.5 GFLOP on GLM); `F_a0` ≈ 2.9 GFLOP and `s_a` ≈ 2.33·10^-5 GFLOP per prefix token, the attention-core intercept and slope fitted in §7.1 from P20 §4.1; `f/f_pk` loaded clock over the clock at which `Peak_dtype` was probed (unstated for the ≈213 reading; ≈0.8 under the 2,100 MHz cap against ≈2.6 GHz, core fact 12).

Worked lane, cold 95K at 1,747 tok/s under the cap (2026-09-23 V3.2 verification, V3.2a arm) [Measured]: `F_tok,avg` ≈ 33.5 + 2.9 + 1.1 = 37.5 GFLOP → 65.5 TFLOP/s → ≈21.8 TFLOPS per node → ≈13% of 213 × 0.8, or ≈22% of 123 × 0.8 [Interpretation]. Core 4.2's ≈9% / ≈15% omits `F_attn` and the clock scaling. The widely quoted ~14% is one recipe roofline at 125 TFLOPS, arithmetic rather than measurement (P04 §7). With §1.5, if the FLOPs sit in the clock-sensitive share φ ≈ 0.34, that share runs at ≈38% (≈213 reading) or ≈65% (≈123 reading) of the scaled peak [Interpretation]: which peak holds decides whether a FLOP-side kernel has room (≈2.6×) or little. Record the peak and clock assumed with every MFU figure.

---

## 2. MoE: expert work per chunk

### 2.1 Every expert, every chunk

`touched/E = 1 − (1 − k/E)^M` (core 4.4) for uniform routing [Interpretation]:

| model (k of E) | M = 64 | M = 256 | M ≥ 1,024 |
|---|---|---|---|
| GLM 5.3 Flash (8 of 288) | 83.5% | 99.93% | 100% |
| DeepSeek V4.1 Flash (6 of 384) | 63.5% | 98.2% | 100% |

Architecture from P20 §4.1 [Code-verified]. At any prefill-sized chunk every rank reads all its experts once per MoE layer per step, whatever `M` is. The chunk decides only how many rows share that read.

### 2.2 Intensity of an expert GEMM

```
ρ   = k·M / E                              mean rows per expert
I   = 2·ρ / β                              FLOP per expert-weight byte; β = bytes per weight (0.5 at 4 bpw, 1 at FP8)
I*  = Peak_dtype·(f/f_pk) / BW_eff          ridge, FLOP/B; f_pk = clock of the peak probe
M*  = I*·β·E / (2k)                         chunk at which mean routing reaches the ridge
```
[Interpretation]. GLM at 4 bpw: `I = M/9`; at `M` = 1,024, `I` ≈ 114 FLOP/B, matching a recipe author's analysis of ≈112 FLOP/B for expert streaming at chunk 1,024 (arithmetic, graded [Interpretation] in P04 §3). Ridge with the contested peaks (core fact 12) at the probes' own clock (unstated for the ≈213 reading, ≈2.6 GHz for the FP8/FP4 probes) and `BW_eff` 230-245 GB/s: BF16 ≈123 → `I*` ≈ 500-535; BF16 ≈213 → ≈870-926; FP8 214-256 → ≈870-1,110 FLOP/B.

| expert format | `M*` for GLM (mean routing) |
|---|---|
| 4 bpw, BF16-rate MMA after dequant | 4.5K-8.3K tokens |
| FP8 E4M3 (native checkpoint) | 15.7K-20K tokens |

Mean-routed expert GEMMs at the GLM chunks in use (`M_eff` ≤ 3,584) are below the ridge under both BF16 readings [Interpretation]. At 4.5K-8.3K they cross it under the ≈123 reading and approach it under ≈213 (`I` = 910 at 8,192). Under a 2,100 MHz cap `I*` and `M*` scale by ≈0.8 (core fact 12): `M*` 3.6K-6.7K for 4 bpw, so the old 3,584 rung sits at the ridge under the ≈123 reading. Routing is concentrated: on a two-node GLM EXL3 lane at chunk 1,024, 91% (≈16K) to 98.6% (≈100K) of prefill MoE layers had an expert above 128 rows against a mean `ρ` of 28; P20 §4.1 grades this [Measured] (own profile), P04 §4 grades the same figures [Community-measured] (recipe author's notes). Fat experts run far above the mean `ρ` and thin experts far below it.

### 2.3 Why an EXL3 trellis path amortizes little

The fused EXL3 MoE kernel "dequantizes each expert's weights once per 16-row tile" [Code-verified] (ExLlamaV3 docs at the P04 pin; P04 §2 ExLlamaV3). Its cost is ∝ ⌈ρ/tile⌉ × expert bytes, which grows with `M` once ρ exceeds one tile: it lands in `a`, not in `t_fixed`. A dequant-once-per-step path (BF16 reconstruct) moves cost back into `t_fixed`, where a larger chunk amortizes it [Interpretation]. On the worked lane the fused kernel handles only thin experts; fat experts run the grouped path (profile sums: thin fused 9.0% vs grouped gate/up 9.0% + down 4.9% [Historical diagnostic], P04 §4), whose dequant schedule is not documented, so which term dominates is [Proposed: needs measurement]. Consistent with this reading, not proof of it [Interpretation]: the small `h` on the worked lane (106-139 tokens, §7.1); the larger `h` on the DeepSeek V4.1 EXL3 lane (§7.4), whose thin experts hold ρ = 16 rows (one tile) at `M` = 1,024; and the reversal of the best fused-row cap with the fat path present (a 1,024 cap lost 13-24% pre-E2; 128 → 32 gained 13-16% with E3 [Community-measured], P04 §2, §4). Decision input: where per-tile dequant dominates, a kernel lever beats a chunk lever.

---

## 3. Large-M GEMM: when `a` depends on `M`

| mechanism | condition | sign | evidence |
|---|---|---|---|
| L2 spill | blockwise-FP8 CUTLASS projection > 24 MiB on a build older than vLLM #55180 | larger `M` → lower TFLOPS | [Community-measured] (P04 §3; core 4.5) |
| M alignment | `M mod 4 ≠ 0` on a blockwise-FP8 CUTLASS linear on a build older than vLLM #52775; besides odd chunk settings, any step whose token count is not a multiple of 4 hits it: the tail chunk (`P mod M`), about three in four short prompts, mixed steps with odd decode rows [Interpretation] | small-M tile at large M | −37% 8K TTFT from padding [Community-measured] (P04 §3) |
| weight-only kernel | Marlin/EXL3 weight-only at large `M` vs a BF16 copy | BF16 wins above a boundary | own boundary M > 512 qualified [Measured] (2026-09-20 startup and prefill optimization); analytic crossover near M ≈ 128 on the worked lane's KDA shape [Proposed] (lane card lever) |
| precision | FP8 weight-only dense vs BF16 dense | FP8 prefill −18 to −21%, decode +7-14% at short depth | [Measured] (2026-09-08 E4 A/B, GLM TP2) |

Precision decision, derived from core 4.7 [Interpretation]. Let `g_p` = prefill time ratio FP8/BF16 (1.22-1.27), `g_d` = decode rate ratio FP8/BF16 (1.07-1.14). FP8 lowers `T_turn` iff

```
U / O  <  (1 − 1/g_d) / (g_p − 1) · R_prefill / R_decode
```
`R_prefill`, `R_decode` are the BF16 arm's rates. With the measured ranges the coefficient is 0.24-0.56. With the packet's BF16-arm rates (prefill 1,420 tok/s at ≈100K; decode 27.1 coding, 29.4 prose, 61.9 structured) the trade flips at ≈26K, ≈17K and ≈6K new prompt tokens per 1K output tokens [Interpretation from Measured inputs] (2026-09-08 E4 A/B, GLM TP2). The recorded "≈1K output per 27K new prompt tokens" (P04 §5 lever 6) is the coding-rate case of the same trade: the packet computes it from BF16 saving ~0.16 s per 1K prompt tokens and losing ~0.44 s per 100 output tokens at coding rates. The break-even therefore moves about 4× with output type. At 120K depth (`g_d` ≈ 1.01) the coefficient falls to ≈0.04, so FP8 wins only output-heavy turns. Test that decides for a lane: log `U`, `O` and output type per turn on real traffic and compute Δ`T_turn` for both arms (P04 lever 6 experiment).

---

## 4. The four costs of a chunk

A chunk rung is a trade among speed (§1.3), transient memory, KV pool and incumbent stall. Each has its own law.

### 4.1 Pool charge

```
κ = −ΔN_pool / ΔMNBT          pool tokens lost per MNBT token, read from two boot capacity lines at one pin
```
Worked lane, 32 GiB/rank pin: 7,168 → 4,096 gave +421,152 tokens (κ ≈ 137); 4,096 → 2,048 gave +366,335 (κ ≈ 179) [Interpretation from Measured inputs] (2026-09-15/16 upgrade A/B, GLM 5.3 Flash EXL3 TP3). LPT moved with MNBT in that ladder, so κ is charged to the pair. The packet attributes the charge to the sparse-indexer scratch KV group, which reserves page ids in proportion to the chunk (685 vs 525 block ids per 1M-token request at 7,168 vs 2,048 [Measured]; mechanism [Interpretation]). On drafter lanes the draft model's budget-scaled reservation adds to κ (GLM TP2 7,168 cost ~640K context [Measured], P04 §7). κ is known only between measured rungs; extrapolating it past the ladder is [Interpretation].

### 4.2 Transient

```
T_pre(M, P) = c_lane · M · P
c_lane = s_logit · n_mat / r_pool      s_logit = bytes per logit element, n_mat = logit rows materialized per token, r_pool = keys per pooled column
```
[Interpretation]. The SGLang GLM kpool indexer materializes fp32 logits over one pooled key per 4 tokens (`index_kpool` 4, P20 §4.1), i.e. `[M, P/4]`: `c` = 4 × 1 / 4 = 1 B [Interpretation from Code-verified parts], which reproduces the community sizing of 1.07 GB at 8,192 × 131K [Community-measured] (P04 §2 SGLang, §3). The DeepSeek V4.1 ≈14 B constant is community-stated with unpublished inputs (core 4.6); do not derive it, measure it.

Boundedness decides whether `T_pre` grows without limit:

| path | bound | grade |
|---|---|---|
| vLLM sparse indexer | `[M, N]` fp32 logits capped by `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default 512), sub-chunked on `M` | [Code-verified] at the P04 pin; the GLM-5.3 path reserves max(decode logits, prefill cap) at profile time; whether a lane's fork carries it: read from the lane card: engine version, and indexer-cap presence [Proposed field] |
| SGLang DeepSeek V4 indexer | ≤ 4,096 rows per call and free memory × 0.2 | [Code-verified] |
| SGLang GLM kpool, CUDA branch | none: whole ragged batch | [Code-verified] |
| lane-kit static indexer workspace | not a transient: sized at profile time (`max_model_len × 40` entries, 5,036 MiB at 1M on one kit) and locked | [Community-measured] (P04 §2) |

On a bounded path the envelope (core 4.6) is set by allocator growth and other scratch, not by `c_lane`; the memory-watched probe past 128K still applies (P04 §6 step 7).

### 4.3 Incumbent stall

```
stall_max ≈ t_step(M_grant) ≈ M_grant / R(M_grant)
```
[Interpretation]. Worked lane: predicted 2.13 s at 3,584 and 1.24 s at 2,048; measured longest stall 2.37 s and 1.02 s, one 30K-newcomer observation each (2026-09-15/16 upgrade A/B) [Measured]. Prediction error −10% / +22%; the fair-mixed-prefill policy on that lane splits newcomer work (P04 §4), which the formula ignores.

### 4.4 Joint objective

For a candidate rung `M` against the current `M_0`, all four are computable before the candidate's boot once `h` and κ are on the card (or fitted from two boots, §8 step 2):

| term | formula | needs |
|---|---|---|
| speed | `R(M)/R(M_0) = M(M_0 + h) / (M_0(M + h))` | `h` (lane card or two rungs) |
| pool | `ΔN = −κ·(M − M_0)` | κ (two boot lines) |
| transient | `c_lane·M·P_max` vs `H_warm − F_warm` (core 4.6) | `c_lane`, bound (4.2) |
| stall | `M / R(M)` | `R(M)` |

---

## 5. Prefix reuse economics

### 5.1 Warm-turn cost

```
T_warm(P, U) ≈ t_hit(P) + U · τ_marg(P)
U = N_new + (P_prev mod page) (+ one draft block on builds without vLLM #53388)
```
[Interpretation] (core 4.7; P04 §2 "Trailing-block drop"). `page` = hybrid state page (read from the lane card: chunk size / cache page). The page term alone costs up to `page · τ_marg`: worked lane (page 2,560) ≈ 1.5 s at 100K and ≈ 2.2 s at 1M [Interpretation]; the TP2 lane (page 3,584, ~830 tok/s) ≈ 4.3 s, against measured warm turns of 5.9-6.4 s [Measured] (P22 §4).

`t_hit(P)` is not zero. Residual of measured exact repeats after subtracting `U·τ_marg` on the worked lane (2026-09-14 triangle qualification, n = 1 per depth) [Interpretation]:

| P cached | U | measured TTFT | `U·τ_marg` | residual `t_hit` |
|---|---|---|---|---|
| 7,680 | 248 | 0.65 s | 0.14 s | 0.51 s |
| 30,720 | 1,781 | 1.50 s | 1.04 s | 0.46 s |
| 130,560 | 252 | 0.99 s | 0.15 s | 0.84 s |
| 248,320 | 1,422 | 1.98 s | 0.91 s | 1.07 s |
| 747,520 | 2,219 | 10.4 s | 1.72 s | 8.7 s (outlier) |
| 998,400 | 1,333 | 4.28 s | 1.12 s | 3.16 s |

Excluding 750K, a least-squares line gives `t_hit` ≈ 0.44 s + 2.7 µs per cached token. The intercept is about `t_0` (0.34 s) plus one short step's `t_fixed`, which `U·τ_marg` spreads over `M_eff` tokens instead of charging once; the growth term's cause is unknown [Proposed: needs measurement]. At 1M `t_hit` exceeds the recompute.

### 5.2 Hit rate against cold speed

```
E[TTFT] = H·T_warm + (1 − H)·T_cold
a cold-prefill speedup g (rate × (1+g)) ≡ ΔH ≈ (1 − H)·g/(1 + g) · T_cold/(T_cold − T_warm)  (fraction of hit rate)
```
[Interpretation]; `H` = warm fraction (the logged live hit rate stands in for it), `g` = fractional cold-rate gain, `ΔH` a fraction (× 100 for points). At the logged live hit rate `H` = 0.919 (2026-08-30 metered traffic, GLM TP2; P10 §4) a 20% cold-prefill win equals ≈1.35 points of hit rate; at `H` = 0.96 it equals ≈0.67 points. Inverted, with `T_warm` ≪ `T_cold`: one point of hits equals a cold win of ≈14% at `H` = 0.919 and ≈33% at `H` = 0.96. At `H` ≈ 0.96 only the largest P04 §5 cold levers exceed it (grouped fat experts +37-45%, engine-level MoE rewrites 2.4-3.3×, replacing an exact top-k fallback that costs up to 40%); at `H` = 0.919 the 20-26% levers (E2 fat experts, BF16 dense, the chunk on some lanes) also exceed it. The lane's `H`: read from the lane card: [Proposed field] live hit rate.

### 5.3 Client timeout depth

```
P_to : T_cold(P_to) = T_client
```
[Interpretation]. Worked lane (fit §7.1, pre-cap, `M_eff` 3,584): `T_client` = 60 s → `P_to` ≈ 101K tokens. Every cold prompt deeper than `P_to` times out and, with client retries, re-enters the queue while the first prefill still runs; this is the retry amplifier in the 2026-09-16 head-wedge incident (eight re-sends, 60 s timeouts) [Historical diagnostic] (P04 §7; core §2.6). Rule (P22 §5 lever 7, §6): the client's request and first-token timeouts exceed `T_cold` at the largest context it sends (≈708 s at 1M on the worked lane pre-cap), and the client never retries without a verified cancel; raising the timeout alone hides the symptom.

### 5.4 Cost of a changed byte

```
t_p = page · ⌊t / page⌋                          t = first changed token
C(t) ≈ (P − t_p)·(a + t_fixed/M) + (b/2)·(P² − t_p²)     re-prefill cost of the suffix
```
[Interpretation] from (4.8). A change at the head costs `T_cold(P)`; a change just before the last user turn costs one page plus that turn. Worked check on the TP2 lane (page 3,584): pre-last-user placement kept 14,336 of 16,579 tokens cached = 4 × 3,584, the page boundary below the final turn; head placement at 152K cost 92.3 s vs 2.1 s [Measured] (2026-09-11 effort-placement probe; P10 §4). The widely quoted "~84 s per toggle at median depth" is depth ÷ ~782 tok/s, derived, not timed [Measured] + [Interpretation] (P10 §4 row); it cannot validate (5.4).

### 5.5 Eviction

Occupancy alone does not predict a miss [Measured]: two ~275K sessions evicted each other at 61% of an 896K pool (P04 §7); the worked lane missed exact repeats at 524K and 900K with its pool 20-35% occupied at MNBT 7,168, while hitting at 250K, 750K and 1M; at 4,096 the 524K repeat hit (900K and 1M not re-run) (2026-09-14 triangle qualification; 2026-09-15/16 upgrade A/B); a community pair missed at 76% (P04 §4, MNBT 1,536 eviction-at-depth row). Rule: after any MNBT, pool or drafter change, re-run the exact repeat at the longest qualified depth (P04 §6 step 4); keep one long thread hot per lane (P04 §7).

---

## 6. Cache-safe template gate: pass criteria

Derivation: block hashes chain from token 0 (core 4.7), so reuse after a change at token `t` is bounded by `t_p` (5.4). The gate steps are core 4.7; the pass arithmetic is:

| probe | pass iff | grade, source |
|---|---|---|
| retention | `cached_tokens` ≥ `page·⌊t_u/page⌋` on calls 2-3, `t_u` = first token of the final user turn (or of the moved directive), on a history ≥ 46K tokens with the lane's **real** system prompt and tool list | [Interpretation]; probe design P22 §6 |
| correctness | four placements (head / tail / none / pre-last-user), scored; no arm worse than "none" | tail malformed at High in 9 of 20 or 10 of 22, unresolved [Measured] (P10 §4) |
| hybrid output | hit output equals cache-off output on the same prompt | [Community-measured] (P04 §7) |
| accounting | `cached_tokens` enabled and non-zero on the control | [Code-verified] (P10 §6) |

A retention probe without the lane's system prompt and tools passes falsely: a proxy's "95.3% retention" came from such a probe while the real shapes shared 22 characters [Measured] (P10 §7). One served model ID per lane: extra picker entries destroyed prefix caching [Historical diagnostic] (WR); that each entry acts as a head change is [Interpretation].

---

## 7. Worked example: GLM 5.3 Flash EXL3 TP3, node-1/2/3

Inputs: 4 slots through the 2026-09-15/16 packet, 8 from the 2026-09-16 runtime bundle on; vLLM + EXL3 overlay, EXL3 4 bpw experts, FP8 weight-only dense, grouped fat-expert path; stock clock unless stated. Packets: 2026-09-14 triangle qualification, GLM 5.3 Flash EXL3 TP3 (depth ladder, MNBT 7,168 / LPT 3,584); 2026-09-15/16 upgrade A/B, GLM 5.3 Flash EXL3 TP3 (chunk ladder, one arm per boot); 2026-09-20 startup and prefill optimization; 2026-09-21 GPU clock-cap sweep; 2026-09-23 V3.2 verification. All rates [Measured]; all fits [Interpretation].

### 7.1 Fit

Chunk ladder, `M_eff` = LPT (1.1):

| depth | R at M_eff 3,584 / 2,048 / 1,024 | least-squares `t_fixed` | `a` | `h` | max residual |
|---|---|---|---|---|---|
| 8K | 1,611 / 1,550 / 1,469 | 84 ms | 600 µs | 139 | 0.7% |
| 100K | 1,706 / 1,648 / 1,558 | 78 ms | 566 µs | 139 | 0.4% |
| 524K | 1,552 / 1,508 / 1,445 | 67 ms | 628 µs | 106 | 0.4% |

Pairwise slopes span 59-117 ms, and the middle rung sits above the line at every depth: mild curvature, so `h` ≈ 106-139 is a band, not a constant.

Depth ladder (fixed `M_eff` 3,584; 8K, 32K, 131K, 250K, 524K, 750K, 900K, 1M) fitted to (4.9):

```
t_0 = 0.34 s     a' = a + t_fixed/M = 575 µs/token     b = 2.65·10^-10 s per token²
```
Fitted rates fall within 0.7% of all eight measured points, with the 8K point taken from the prefill fixture (7,603 tokens, 1,613-1,619 tok/s); the ladder fixture's own 8K row (1,399 tok/s at 7,928 tokens) sits ≈13% below the fit, unexplained. The model gives TTFT(1M) = 707.9 s against 707.8 s measured. In the fit the depth term is 18.7% of the per-token time at 1M; the measured rate at 1M is 16.5% below the 131K rate (core 4.1 anchor). The marginal cost per token is +46% there (4.10).

Cross-check against architecture [Interpretation]: attention-core FLOP per token grows ≈2.9 + 2.33·10^-5·P GFLOP (from P20 §4.1: 3.1 / 5.9 / 26.6 at 8K / 128K / 1M) on a 33.5 GFLOP weight base. At one achieved rate that predicts +64% marginal cost at 1M; measured +46%. The depth work runs at ≈1.4× the weight path's achieved rate, consistent with an FP8 indexer on a faster MMA path than the dequantized experts; not profiled.

### 7.2 Prediction chain

Each factor is a measured ratio from its own packet; the chain tests that they compose multiplicatively across four images:

| step | factor | predicted 95K rate | measured |
|---|---|---|---|
| fit, `M_eff` 3,584 | — | 1,691 | ~1,695 at ~95K (2026-09-14) |
| `M_eff` 3,584 → 2,048, `h` = 123 | 0.976 | 1,650 | 1,648 at 100K (2026-09-15/16) |
| runtime bundle (+3.47% at 95K) | 1.0347 | 1,707 | 1,695.83 baseline (2026-09-20) |
| KDA BF16 reconstruct group (+7.93%) | 1.0793 | 1,842 | 1,830.38 (2026-09-20) |
| 2,100 MHz cap (−4.9%, measured at 250K) | 0.951 | 1,752 | 1,747 V3.2a (2026-09-23) |

Error ≤ 0.7% per link [Interpretation]. Link 4's measured value is the numerator of its own factor (same packet as link 3), so it adds no independent test. The cap link uses the 250K factor (n = 1); the 76.5K factor (−5.7%, n = 2-3) gives 1,737 (−0.6%). The last link assumes the other V3.2a changes are prefill-neutral, as that packet reports (V3.2a ≈ V3 × 0.955): control traffic on the direct links, compact drafter KV, and the retained cache and seed-stride overlays. V3.2a disables only the align-chunking overlay, which cost 9-10% at 95K.

### 7.3 Decisions this supports

1. **Chunk.** The measured rung above the current one (MNBT 7,168 / LPT 3,584) bought +2.9-3.9% cold prefill for −421K pool tokens (−14% of 3.0M), a 2.37 s vs 1.02 s incumbent stall and a 524K exact-repeat miss [Measured] (2026-09-15/16 upgrade A/B). A full doubling of `M_eff` to 4,096 would buy ≈2.9% (h = 123; 2.5-3.3% over the `h` band) [Interpretation] at an unmeasured pool cost. `M_eff` 2,048 is below `M_stop` (≈2.6K-3.4K), so the speed term alone still clears the 2% floor; the chunk lever is closed by the pool, stall and cache-residency terms (§4.4), not by `M_stop`. The lane keeps MNBT 4,096 / LPT 2,048 (read from the lane card: chunk size). The prefill levers left are in `a` (per-tile dequant, fat-row cap, KDA and dense GEMMs) [Interpretation].
2. **Envelope.** The 2026-09-14 campaign (15 gates, 8K-1M retrieval, 84-request soak) left minimum MemAvailable 11.6 / 13.2 / 15.2 GiB per node, node-1's at the 524K stage, with 0 NV_ERR [Measured] (2026-09-14 triangle qualification; P11 §4). Upstream vLLM bounds the sparse-indexer logits at 512 MB by default and reserves the GLM-5.3 prefill buffer at profile time [Code-verified] at the P04 pin; an unbounded kpool-style path at `c` = 1 B would need 1 B × 2,048 × 1M ≈ 2.0 GB per step [Interpretation]. Whether the lane's fork carries the cap is unverified. The envelope is re-proven on the current recipe, not inherited.
3. **Timeout.** Cold prompts beyond ≈101K (pre-cap fit; ≈105K at the capped 95K rate of 1,747) exceed a 60 s client timeout (5.3).
4. **Warm path.** A warm turn at 100K costs ≈0.7-2.3 s (`t_hit` ≈ 0.7 s plus 0 to one 2,560-token page at ≈0.6 ms per token) plus `N_new · τ_marg`, against ≈60 s cold [Interpretation]. Measured anchors: a 1.12 s warm follow-up at 131K (2026-09-16 combined runtime qualification) and a 1.3 s cached repeat at 85K (2026-09-23 V3.2 verification) [Measured]. At `H` ≈ 0.96 one point of hits outweighs a 20% prefill lever; at the logged 0.919 a 20% lever is worth ≈1.35 points (5.2).

### 7.4 Other lanes, same law

| lane (packet) | observation | reading |
|---|---|---|
| DeepSeek V4.1 Flash EXL3 TP3 (2026-09-16 optimized-1M) | 1,024 → 8,192: +3% at 8K, +26% at 32K (1,024 arm n = 1) | a single `h` fits neither; fitted per arm, the intercept is ≈0.2 s at 1,024 and ≈1.9 s at 8,192, so a large-chunk per-request cost dilutes the 8K gain [Interpretation]; from the 32K row alone `h` ≈ 316 |
| Qwen3.8 Flash Next, 1 node (2026-09-09 tuning) | 2,048 → 4,096: +7.8% at 500K | `h` ≈ 347 if `M_eff` were the setting; confounded: the 1,664-token hybrid page means a 2,048 chunk advances 1,664, and the pool changed |
| Qwen3.8 27B, 1 node, SGLang (2026-09-08 screen) | 8,192 → 2,048: TTFT −17-22% (n = 1) | wrong sign for (4.9): an `M`-dependent `a` or transient; profile before extrapolating |

---

## 8. Decision procedure: chunk and warm-path levers

Extends core 4.6's chunk procedure with numbers it can compute.
1. Read from the lane card: chunk size (MNBT, LPT, fair-prefill policy, cache page), engine version (PR #57951, #53388, #52775, #55180, indexer cap), context, hit rate, client timeout. Compute `M_eff` (1.1).
2. Take `h` and κ from the card. If absent: two boots, current rung and one rung down, cold 8K and 100K × 3 each (P04 §6); fit `h` from (4.9) at each depth, and κ from the two capacity lines. Fit `t_0` per arm; if the intercepts differ by more than the rep spread, the rungs differ in more than `t_fixed` (§7.4 row 1).
3. If `M_eff ≥ M_stop`, stop: the chunk lever is exhausted. Go to step 7.
4. For each candidate `M` compute the four terms (4.4). Drop rungs whose transient breaks the envelope or whose stall exceeds the lane's decode-gap budget.
5. MoE lanes: compute `ρ` and `I` (2.2) per candidate; if the expert path dequantizes per tile (2.3), expect `h` small and prefer a kernel lever.
6. Run the surviving rung with the P04 §6 ladder, including short decode, the concurrency probe and the memory-watched probe past 128K. Keep it only if the gain exceeds the noise floor and the predicted speed term was within its tolerance; record prediction and result.
7. Rank warm-path levers with (5.2) against cold levers using the lane's `H`. A warm-path candidate must pass the gate (§6) before it ships.
8. Check `P_to` (5.3) against the served context and the client timeout. If `P_to` < served context, fix the client path first: timeouts above `T_cold` at the largest context sent, and no retry without a verified cancel (5.3).
9. Write back `h`, κ, `t_0`, `b`, `t_hit` slope and `P_to` to the lane card (Gaps 1).

---

## 9. Traps

Each: if you believe X, you will do Y, and Z happens.

1. **"MNBT is the chunk."** You run an MNBT ladder with LPT below MNBT on a pre-#57951 build and read the pool change as a speed result. Every rung prefilled at LPT (P04 §4, MNBT 8,224 → 2,048 row; §1.1). After a rebuild past 2026-09-22 the same flags change solo ladders; re-baseline.
2. **"One depth is enough to rank rungs."** You compare two chunks at 8K and conclude "no gain", or at 32K and conclude "+26%". Per-request and per-step terms trade places with depth (§7.4 row 1). Fit `t_0` per arm, or ladder at two depths.
3. **"Bigger chunks are always faster."** You raise the chunk on a lane where `a` rises with `M` or the transient is unbounded. SGLang GLM kpool at 8,192 timed out twice at 102K; 2,048 took 65.5 s [Community-measured] (P04 §4, SGLang kpool chunk 8,192 → 2,048 row); a DeepSeek V4.1 EXL3 pair wedged at ~140K with MNBT 3,072 and no error [Community-measured] (P04 §7).
4. **"Smaller chunks are safer."** You drop 1,024 → 256 "as a memory precaution" without a measured envelope; cold prefill halved (~1.0k vs ~2.0k tok/s, not isolated) and the necessary chunk was never measured [Historical diagnostic] (WR; core 4.3). Lowering MNBT also crashed a build that locks warm-up workspaces (P04 §7).
5. **"Larger chunks amortize the experts."** True only where weights are read once per step (2.3). On the worked lane the chunk barely moves prefill (~4% per halving [Measured]), consistent with a per-tile dequant path [Interpretation], and the pool and stall costs dominate.
6. **"Prefill is compute-bound because it is clock-sensitive."** φ ≈ 0.34-0.37 on the worked lane (1.5): about two thirds of the step does not scale with clock. A FLOP-only kernel win is capped at that share.
7. **"Next-chunk and whole-context rates are the same metric."** You compare a llama-benchy next-2,048-at-depth figure with a cold whole-prompt TTFT rate and find a large "depth penalty". The marginal depth term is twice the average (4.10).
8. **"A pool much larger than the prompt guarantees a hit."** Exact repeats missed at 20-35% occupancy on the worked lane at MNBT 7,168 (5.5). Verify with an exact repeat.
9. **"The cached-prefix intercept is the warm turn."** A smaller hash block cut the intercept 485 → 265 ms, but not real-turn medians [Community-measured] (P04 §7). Measure turns, with `U` per turn.
10. **"Warm hits are free at depth."** `t_hit` reached ≈3.2 s at 1M on the worked lane (5.1; one repeat). Budget it in TTFT targets for 1M sessions.
11. **"84 s per toggle confirms the re-prefill model."** That figure is depth ÷ rate (5.4). Use the timed 152K probe.
12. **"A retention probe passed, so the template is cache-safe."** Without the real system prompt and tools the probe passes falsely (§6); tail placement keeps the cache and breaks answers (9/20 or 10/22).
13. **"Break-even is 27K prompt per 1K output" for every workload.** That figure is the coding-rate case; with prose output the trade flips near 17K and with structured output near 6K (§3). Choosing BF16 vs FP8 on the 27K figure alone mis-assigns structured-output and mixed workloads; use the lane's own `U`, `O` and output mix.
14. **"The client will wait."** A 60 s client timeout below `T_cold` at served depth turns one deep prompt into a retry storm (5.3; 2026-09-16 head-wedge incident).
15. **"Cache hits are prefill throughput."** A dashboard counting cached tokens as prefill showed 12,353 tok/s [Historical diagnostic] (WR); one engine's dashboard rate is uncached compute only [Code-verified] (P04 §7). Use prompt / TTFT with the metric named.

---

## 10. Evidence map

| chapter section | pages | packets |
|---|---|---|
| 1 Step model | P04 §2 (vLLM scheduler), §3, §8; P03 §4 | 2026-09-20 startup and prefill optimization; 2026-09-21 GPU clock-cap sweep |
| 2 MoE | P04 §2 (ExLlamaV3), §3, §5 lever 4; P08; P20 §4.1 | 2026-09-16 prefill profiling and tuning, DeepSeek V4.1 Flash EXL3 TP3 |
| 3 Large-M GEMM | P04 §3, §5 levers 5-6, 8; P09 | 2026-09-08 E4 A/B; 2026-09-20 startup and prefill optimization |
| 4 Four costs | P04 §2, §4, §5 levers 1-2, 7, §7; P11 §2, §7 | 2026-09-15/16 upgrade A/B; 2026-09-16 head-wedge incident record |
| 5 Reuse economics | P10 §3-§4, §6-§7; P22 §3-§6; P04 §4 warm-path rows | 2026-08-30 metered traffic; 2026-09-11 effort-placement probe; 2026-09-14 triangle qualification |
| 6 Gate | P10 §4, §7; P22 §5-§6; P21 | 2026-09-11 effort-placement probe |
| 7 Worked example | P04 §4; P20 §4.1-§4.2; P13 | 2026-09-14; 2026-09-15/16; 2026-09-16 combined runtime qualification; 2026-09-20; 2026-09-21; 2026-09-23 V3.2 verification |

---

## Gaps

1. Lane-card fields this chapter reads that the template lacks: `h`, κ, `t_0`, `b`, `t_hit` slope, `P_to`, live hit rate `H`, client timeout, indexer-cap presence in the fork. [Proposed]
2. No per-step GPU timing on any own lane; `t_fixed` is inferred from rate ladders, and its split (collectives, syncs, exposed stream) is unmeasured (1.4). [Proposed: needs measurement]
3. EXL3 per-MoE-layer host-sync cost on GB10 unmeasured.
4. `c_kv` (per-step prefix re-read) estimated from bytes only.
5. `t_hit(P)` growth (≈2.7 µs per cached token) rests on one repeat per depth with a 750K outlier; cause unknown (5.1).
6. The DeepSeek V4.1 EXL3 8K-vs-32K inconsistency (§7.4) rests on an n = 1 arm; no two-depth, two-arm repeat exists.
7. The depth-work efficiency ratio (≈1.4×, §7.1) is arithmetic against a FLOP inventory; indexer kernel time is not profiled.
8. The precision break-even (§3) rests on one TP2 A/B at short depth; no real-traffic `U`/`O`/output-type log exists to weight the ≈6K-26K range.
9. Eviction below 40% occupancy (§5.5) is unexplained.
10. The analytic weight-only vs BF16 crossover (M ≈ 128) is untested; only M > 512 is qualified.
11. The 91-98.6% fat-layer figure (2.2) is graded [Measured] on P20 and [Community-measured] on P04; the attribution is unresolved.
12. MFU (1.6) inherits the contested peak and the unstated probe clock of the ≈213 reading; the prefill all-reduce latency on the Simple protocol is unmeasured (1.4).

## Sources

Wiki pages (read 2026-09-23): P04 §1-§8 in full; P03 §2, §4 rows cited; P10 §3-§4, §6-§7 rows cited; P11 §4 row cited; P13 §4 rows cited; P20 §4.1-§4.3; P22 §3-§6 rows cited; core.md (checked) §1, §4, §5.1, §10. Lane card (private input, used as worked-example source): GLM 5.3 Flash EXL3 TP3.

Own packets (date and title): 2026-08-30 metered traffic; 2026-09-08 GLM 5.3 Flash EXL3 E4 five-arm A/B; 2026-09-08 cache, drafter and chunk screen, Qwen3.8-27B; 2026-09-09 tuning, Qwen3.8 Flash Next single node; 2026-09-11 GLM 5.3 Flash effort-placement probe; 2026-09-14 triangle qualification, GLM 5.3 Flash EXL3 TP3; 2026-09-15/16 upgrade A/B, GLM 5.3 Flash EXL3 TP3; 2026-09-16 combined runtime qualification; 2026-09-16 optimized-1M profile, DeepSeek V4.1 Flash EXL3 TP3; 2026-09-16 head-wedge incident record, DeepSeek V4.1 Flash SGLang TP3; 2026-09-20 startup and prefill optimization, GLM 5.3 Flash EXL3 TP3; 2026-09-21 GPU clock-cap sweep, GLM 5.3 Flash EXL3 TP3; 2026-09-23 V3.2 verification, GLM 5.3 Flash EXL3 TP3.
