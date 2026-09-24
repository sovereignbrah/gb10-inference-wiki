---
title: The Science of GB10 Inference — chapter 07, measurement
created: 2026-09-23
status: checked
type: chapter
---

# Chapter 07 — Measurement

Extends core §7. The core's tables (noise-band sources 7.1, artifact list 7.2, the 11-step protocol 7.3, the outside-claim checklist 7.5, the profiling order and commands 7.6) are not repeated here; this chapter derives them, works one lane through them, and states the decisions they feed.

Conventions: `P01`…`P22` as in core 10.1 (pathful links in Sources). Own packets cited by date and title. Topology for own rows: node-1 (head), node-2, node-3, ASUS Ascent GX10 (NVIDIA GB10, 128 GB), direct-attach RoCE triangle. Percentages are relative to the named baseline. Every derived quantity is [Interpretation] unless it is a direct reading.

---

## 7.1 Noise model and the minimum detectable effect

### 7.1.1 Equation

```
x(a, b, r) = μ_a · (1 + β_b) · (1 + ε_{b,r}) · (1 + δ(t)) · (1 + Σ_k bias_k)
```

| term | meaning | unit | how it is estimated |
|---|---|---|---|
| `x(a,b,r)` | one reading of a metric (decode tok/s per stream, prefill tok/s, TTFT) for arm `a`, boot `b`, run `r` | tok/s or s | frozen harness, usage-block tokens |
| `μ_a` | the arm's true value | same | the thing being decided |
| `β_b` | boot effect: kernel autotune draw, graph capture, host memory state, clock latch | fraction | spread of arm medians across boots of an identical config |
| `ε_{b,r}` | run-to-run effect inside one boot | fraction | spread of the `n` runs of one arm |
| `δ(t)` | slow drift between opening and closing baselines: thermal, EMA state of adaptive speculation, co-tenant memory | fraction | closing baseline A′ vs opening A |
| `bias_k` | systematic artifacts: cache hits, first request, fixture class, SSE counting, contention | fraction | never estimated from the data; removed by protocol (7.2) |

Bias terms do not shrink with repetition; `β`, `ε`, `δ` do. Protocol steps exist to set every `bias_k` to zero; repetition and boot design exist to shrink the rest [Interpretation].

### 7.1.2 Derivation of "repeat boots, not runs"

Arm median over `n` runs in each of `m` boots, with `σ_B` = SD of `β`, `σ_R` = SD of `ε`:

```
Var(arm)            = (σ_B² + σ_R²/n) / m
Var(B − A)          = 2 (σ_B² + σ_R²/n) / m                          one baseline
Var(B − (A+A′)/2)   = 1.5 (σ_B² + σ_R²/n) / m                        A/B/A, drift cancels to first order
MDE ≈ z · √c · √(σ_B² + σ_R²/n) / √m          c = 2 (A/B) or 1.5 (A/B/A); z ≈ 2 for a ~95% call
```

[Interpretation; assumes independent, roughly normal draws, untested on own data]. With `z` ≈ 2 this is the decision threshold (two-sided α ≈ 0.05): a true effect exactly at `MDE` is called only about half the time; for ~80% power use `z` ≈ 2.8 [Interpretation].

Consequence: as `n → ∞`, `MDE → z·√c·σ_B/√m`. More runs in the same boot cannot resolve an effect of the order of the boot spread; only more boots can. Inputs:
- In-boot run-to-run < 2% [Community-measured] (P14 §4, darkdatter row).
- Boot-to-boot ~8% and +11% on identical configs; C8 92–111 tok/s from a FlashInfer autotune lottery, ±1.6% with autotune disabled at ~2% throughput cost [Community-measured] (P14 §4, §5 lever 11).
- With `σ_B` = 8% the one-boot A/B/A MDE is ~20%; with `σ_B` ≈ 1.6% it is ~4-5% at `n` = 5, `σ_R` = 2% [Interpretation, treating half-ranges as SD proxies].

The core rule "a decode effect below ~8% needs repeated boots per arm" (P14 §6) takes the community boot spread (~8%) as its threshold. By the formula above, one boot per arm at `σ_B` = 8% resolves only effects of ~20%, so the 8% threshold is a lower bound on when repeated boots are needed, not an `MDE` [Interpretation]. A lane whose autotune is disabled and whose own boot spread has been measured may use its own band instead (7.1.3). Both readings stand until the lane has ≥ 3 boots of an identical config [Proposed test; P14 §8 "sample size vs effect"].

### 7.1.3 Estimating the lane band from null-predicted arms

A **null-predicted arm** is a boot whose written prediction for the metric was "no change" (identical config, or a change whose mechanism does not touch that phase). Its median is a draw of `β + δ + ε/√n`. Pool such arms:

```
s_arm = SD over null-predicted arm medians (same harness checksum, fixtures, thinking mode, class)
MDE_lane(A/B/A, m = 1) ≈ 2 · √1.5 · s_arm ≈ 2.45 · s_arm
```

Rules for pooling: only arms on the same harness and fixture manifest; only arms whose prediction for that metric was written before the run; a change of clock cap, image or engine commit between arms is allowed only if the prediction for that metric was "unchanged" and is itself under test (then the band is conservative, never tight).

### 7.1.4 Bias from the first request

```
bias_first = x_first / x_steady − 1
```
Own readings: cold 8K prefill 813 vs 1,375 tok/s (−41%); C1 decode 24.6 and 28.8 vs ~34 right after a new clock lock (−28%, −15%) [Measured] (2026-09-16 "DeepSeek V4.1 Flash EXL3 TP3 prefill profiling and tuning"; 2026-09-21 "GLM 5.3 Flash TP3 GPU clock-cap sweep"; P14 §4). A bias of −15% to −41% is as large as or larger than most single-lever effects in the core (e.g. +7.2–7.9% for a prefill group, +13–19% for a draft-vocabulary trim; core 4.5, 3.6), so one discarded warm-up request that is not the measured prompt is mandatory after every boot, clock change or idle period (core 7.3 step 6). The cause after a clock change is unexplained (P14 §8).

---

## 7.2 Why each protocol step exists

Each step of core 7.3 zeroes one term. An agent that skips a step must name which term it accepts into the reading.

| protocol step (core 7.3) | term removed | size when skipped | grade | source |
|---|---|---|---|---|
| 1 Prediction first | forking-path bias (choosing the metric after seeing it); gives the plausibility test in 7.6.1 | a +38% decode claim headlined against a mechanism that predicts 0 | [Measured] | 2026-09-23 "GLM 5.3 Flash TP3 V3.2: frozen-harness verification" |
| 2 Clean window | `bias_contention` | one pass overlapped ~90 outside requests; one outside request wiped a warm chain 66% → 0% | [Measured] | P14 §4, §7 |
| 3 Same cap, no latched rank | `β` via the slowest rank | a hidden slow state read GEMV 66–80 vs 224–233 GB/s at a constant reported clock | [Community-measured] | P14 §3 |
| 4 Restart between arms, pull skipped, one variable | confounded `μ` (two arms differing by more than the variable) | a pull on restart silently reverts a local image | [Measured] | P14 §7 |
| 5 Greedy logprob probe | invalid `μ` (a broken head inflates acceptance) | prose acceptance p90 8.0 on a dropped head scale; arm withdrawn | [Measured] | P05 §7 |
| 6 Warm every shape, discard first | `bias_first` | −15% to −41% (7.1.4) | [Measured] | P14 §4 |
| 7 Cold nonce, `cached_tokens == 0`, forced length, fixed thinking | `bias_cache`, `bias_length`, `bias_mode` | nested prompts overstated cold prefill ~50×; EOS-terminated short runs under-report 25–275% at depth and concurrency; thinking mode raised own decode 12–19% on two lanes and changed nothing on one community stack | [Community-measured]; [Measured] | P14 §4; P19 4b-40; P05 §7 |
| 8 ≥ 3 reps, A/B/A, per-run values kept | `ε`, `δ` | an n = 1 prefill "win" of +9.9% / +12.4% reversed at n = 3 | [Community-measured] | P14 §4 (jschmied row) |
| 9 MemAvailable sampling | invalid run (a run that passes near a wedge is not repeatable) | covered in core §2 | — | core 2.6 |
| 10 Correctness gates first | invalid `μ` | a build reproduced published speed and scored 0/7 on a strict agent loop | [Measured] | P19 4a-14 |
| 11 Prediction error recorded | model drift: a term missing from the lane's equations goes unnoticed | see 7.6.1 | — | — |

Fixture class is a bias, not noise: structured or counting fixtures inflate speculative decode 1.5–4× over prose (P05 §7) and a `k` = 7 setting was kept on one (core 7.2). It cannot be averaged away; report classes separately.

---

## 7.3 Per-stream versus aggregate

### 7.3.1 Identities

```
r_s   = completion_tokens / (t_last − t_first)                    per-stream decode rate        tok/s
R_agg = Σ_i completion_tokens_i / W                               aggregate over window W       tok/s
D     = R_agg / (C · r_s)                                         decode duty at concurrency C  dimensionless, 0–1
T_req(N) = TTFT + N / r_s                                         wall time of one request of N output tokens
N*    = (TTFT_0 − TTFT_1) / (1/r_s,1 − 1/r_s,0)                   output length at which two configs tie per request
```
[Interpretation; `N*` is the same break-even form as core 4.7 and P14 §4 (jschmied: 68 tokens for MTP off vs on)]

`D < 1` measures time a stream is not decoding: waiting in prefill, queued, or idle between arrivals. A change can raise `R_agg` by raising `D` while lowering `r_s`; the two numbers then move in opposite directions and both are true.

### 7.3.2 Applied to one bundle (node-1/2/3, GLM 5.3 Flash EXL3 TP3)

Source: 2026-09-16 "GLM 5.3 Flash EXL3 TP3: combined runtime qualification" [Measured] (P05 §4; P14 §4). C8 per-stream 21.58 → 17.80 tok/s (−17.5%), C8 aggregate 76.41 → 123.06 tok/s (+61.0%), C8 TTFT 10.63 → 0.52 s.

| quantity | before | after | grade |
|---|---|---|---|
| `D` = `R_agg / (8 · r_s)` | 76.41 / 172.64 = 0.44 | 123.06 / 142.40 = 0.86 | [Interpretation from Measured] |
| `N*` | — | 10.11 s / (1/17.80 − 1/21.58) ≈ 1,030 tokens | [Interpretation from Measured] |

Reading: at C8 the bundle is faster per request for outputs below ~1,030 tokens and slower above; the aggregate gain comes from duty, not from faster decode. Caveats: the C8 baseline came from the same unchanged quick script run on an earlier day with different nonces (historical, not same-day), so by the comparability rule (core 7.5) it needed a same-day control (the packet names the missing closing A1). Outputs were forced to 512 tokens and median batch completion was 53.60 → 33.29 s (packet). Since `R_agg` = 8 × 512 / `W`, `D` = (512 / `r_s`) / `W` exactly, and the non-decode share splits approximately into median TTFT 0.20 → 0.02 and admission stagger plus tail wait 0.36 → 0.12 of the window [Interpretation from Measured]. Per-stream start and finish times are not recorded, so the split is approximate (Gaps 2).

### 7.3.3 Decision

1. Assert `C` from the engine's running-request count during the pass; a "C4" soak that ran sequentially is C1 (core 7.2).
2. Report `r_s`, `R_agg`, TTFT and `D` in one row for every C > 1 cell.
3. Weight by the lane's workload: read from the lane card: Owner notes (operator weights) and the traffic profile (P22 §4). For a read-heavy agent workload with long uncached prompts, TTFT and `N*` against the lane's median output length decide; for a batch workload, `R_agg` decides.
4. A per-stream loss is acceptable only if it was declared in the prediction sheet with the aggregate or TTFT gain it buys (core §8 step 6).

---

## 7.4 Normalizing outside claims

### 7.4.1 Equation

```
x̂_own = x_claim · Π_i f_i                    f_i = x_own(condition_i of this lane) / x_own(condition_i of the author)
u(x̂)/x̂ ≈ √( (u(x_claim)/x_claim)² + Σ_i (u(f_i)/f_i)² )
```
[Interpretation] `u(·)` = one-standard-deviation uncertainty in the unit of its argument; `u(x_claim)` comes from the author's reported spread (none reported → stop, 7.4.3 step 1). An own reading departs from `x̂_own` only if the gap exceeds both `2·u(x̂)` and `MDE_lane`; `MDE_lane` is a detection threshold, not an uncertainty term [Proposed].

Each `f_i` is itself a measurement on the own lane (protocol, fixture, EOS mode, thinking mode, clock, topology, firmware). An `f_i` taken from another lane or another depth carries that lane's band plus a transfer error.

### 7.4.2 Sizes of the factors that are measured

| factor | measured ratio | grade | source |
|---|---|---|---|
| Protocol: recipe protocol (temperature 0.6, numbered-word output) vs temperature-0 essay, same lane | 45–51 vs 31–44 tok/s, so `f` ≈ 0.61–0.98 | [Measured] | 2026-08-16 "DeepSeek V4 Flash dual-GX10 decode tuning A/B" (P14 §4) |
| Fixture: counting cell vs 64K-depth decode, same lane, `k` = 7 | 60.86 vs 18.46 tok/s, `f` ≈ 0.30 | [Measured] | 2026-09-03 "A1 DFlash k=3 vs k=7" (P19 4a-15) |
| Fixture: edit-heavy vs fresh code, same draft depth | 144–151 vs 66 tok/s, `f` ≈ 0.44–0.46 | [Measured] | P19 §1, 4a-4 |
| EOS mode: EOS-terminated 128 vs forced 400 | under-report 25–275% at depth and concurrency; C1 within noise | [Community-measured] | P19 4b-40 |
| Clock: 2,100 MHz cap vs stock, cold prefill | 0.951 at ~250K; 0.943 at a 76.5K fixture | [Measured] | 2026-09-21 clock-cap sweep; 2026-09-23 V3.2 verification |
| Author's number on the author's fixture, own build | 41.43 vs 42.04 tok/s, `f` ≈ 0.99 | [Measured] | 2026-08-21 "Weschera k10 boot-gate YaRN qualification" (P19 4a-1) |

The protocol and fixture factors span 0.3–1.0, wider than any lever in the lane ledgers. A normalized outside number is therefore a pointer to a mechanism, never a prior for the expected gain, a target or a verdict [Interpretation; wording from P19 §6 step 1].

### 7.4.3 Procedure

1. Fill core 7.5 items 1–6 for the exact row being transferred; a missing item stops normalization.
2. Deduplicate lineage (same image digest or weight hash = one report) (P19 §6 step 2).
3. Correctness gates on the candidate build (P19 §6 step 3; P21).
4. Reproduce the author's number on the author's fixture, unmodified, 3–5 runs. Within about 5% of the claim means the build is the one the author measured; a large miss means the build, firmware or ruler differs and later A/Bs are not comparable to the claim (P19 §6 step 4).
5. Only then run the matched-harness A/B on the lane's own fixtures: at least two boots per arm for any decode comparison, at least five matched repeats for a claimed gain under 10%, never ranked on a counting cell (P19 §6 step 5); judge it with 7.1 and 7.5 of this chapter.
6. In any record: outside number and own number in separate sentences, each with its grade (core 7.5 item 6).

---

## 7.5 Decision procedure for one change

[Proposed procedure, derived from 7.1 and the P14 §6 pass gate; its thresholds are untested on own data.]

1. **Predict.** For each metric the change could move, write `pred ± tol`, where `tol = max(model tolerance, MDE_lane)`. Model tolerances: decode inside the bytes-model band (core 3.3), memory terms ±1.5 GiB per rank, pool tokens ±10% [Proposed] (core 2.8). Metrics the change should not move get `0 ± MDE_lane`.
2. **Size the design.** From 7.1.2, choose `m` (boots per arm) so that `MDE ≤` the smallest effect that would change the decision under the operator's weights. If that needs `m > 3` on a production lane, ask for the downtime or drop the question.
3. **Run** core 7.3 as A/B/A, each arm its own boot, pull skipped.
4. **Validity.** `d = A′/A − 1`. If `|d| > MDE_lane`, void the set: drift, contamination or a boot lottery dominates. Rerun or find the cause; never average across it.
5. **Effect.** `e_1 = B/A − 1`, `e_2 = B/A′ − 1`. The statistic that matches `c` = 1.5 in 7.1.2 is `e = B/((A + A′)/2) − 1`; requiring both `e_1` and `e_2` is the conservative form.
   - both `e` beyond `MDE_lane`, same sign → effect established at this `m`;
   - both inside → write "no change detectable at MDE = x%", never "flat" or "unchanged" without the MDE;
   - otherwise → hold; add a boot per arm.
6. **Prediction check.** `|measured − pred| ≤ tol` → the lane's equations hold for this term. A miss beyond `tol` means a term is missing or a constant is wrong: open a single-factor arm or a profile (7.6) before believing either the model or the number.
7. **Multi-metric verdict.** Correctness first; floor held (core 2.4); C1 classes by operator weight; C8 aggregate and TTFT with `D` and `N*` (7.3); keep only if the pre-registered rule holds.
8. **Record.** Band and `MDE` used, `m`, `n`, `d`, `e_1`, `e_2`, prediction and error per metric, all per-run values; update the lane card's measured table and noise band (read from the lane card: noise band [Proposed field]).

---

## 7.6 Profiling as the hypothesis source

A profile answers "where does the time go"; it never answers "did the fix work" (P14 §5 lever 1). The quantities below turn a profile into a falsifiable prediction.

### 7.6.1 Plausibility test before believing any result

```
|x_meas / x_pred − 1| > 3 · MDE_lane   and   no mechanism in the change moves that term   ⇒   treat as artifact until the frozen harness reproduces it
```
[Proposed rule; the factor 3 is a choice, not a measurement] Decode is bytes per step (core §3); a change that does not alter bytes per step, launches, collectives or tokens per step cannot move decode by tens of percent.

### 7.6.2 Exposed-time Amdahl

Core 6.4 gives `T_new/T_old = 1 − s(1 − 1/x)`. A traced region is only partly on the critical path:

```
ΔT/T = e · s · (1 − 1/x)
  s = region time / unprofiled end-to-end time       (unprofiled denominator; P14 §2 SGLang gputrc2graph)
  x = local speedup of the region                    (unprofiled component timing)
  e = exposed fraction, 0 (fully overlapped) … 1 (fully serial)
e_est = ΔT_meas / (s · (1 − 1/x) · T)
```
[Interpretation]

`e` is unknown until measured end to end; a trace shows what overlapped, not what could legally overlap (P14 §2, SGLang skill heuristics). Predict with `e` = 1 as the upper bound and let the A/B/A measure `e`.

### 7.6.3 Profiler overhead and share tables

```
o = T_profiled / T_unprofiled − 1
share_i = t_i / T_unprofiled          never t_i / T_profiled
```
Own torch capture: `o` ≈ +0.51 on TTFT (23.8 → 35.96 s) with the frontend profiler on [Measured] (2026-09-16 "DeepSeek V4.1 Flash EXL3 TP3 prefill profiling and tuning"). Community: 324 vs 893 tok/s at 16K with debug syncs also on [Community-measured] (P14 §4). Overhead is not uniform across kernels, so profiled shares are rankings, not fractions, until a second method confirms them (P14 §6 pass gate). Nsight overhead on GB10 is unmeasured (P14 §8).

### 7.6.4 Collective time is wait plus wire

```
t_coll,r = t_wire + (a_max − a_r)          a_r = arrival time of rank r at the collective
min_r t_coll,r ≈ t_wire                    the last-arriving rank waits for nobody
```
[Interpretation] The rank with the shortest collective time is the slow rank; its collective time bounds the wire cost. A long collective on some ranks is a compute imbalance question for P08 or P03, not a fabric question, until nccl-tests show the wire at the ceiling (P14 §6 table).

### 7.6.5 Clock sensitivity as a classifier

```
ε_clk = (ΔT/T) / (−Δf/f)          ε ≈ 1 compute-bound; ε ≈ 0 bandwidth-, wait- or host-bound
resolvable only if |ΔT/T| > MDE_lane for that metric
```
Prefill `ε` ≈ 0.4 on one TP3 lane [Measured] (core 4.2; 2026-09-21 clock-cap sweep). Decode on the same sweep: C1 36.0 → 33.9 tok/s (−5.8%) at −12.5% clock, but the stock arm's own range was 33.9–38.2 (n = 5), so `ε_decode` is unresolved there [Measured] (P14 §4). The frozen harness read coding 43.82–44.29 before the cap (A0, A1) and 44.34–44.64 after it on the same lane (7.7) [Measured], which suggests `ε_decode` ≈ 0 for that fixture, but other changes landed between those dates [Interpretation]. Contested; the deciding test is a decode-only stock vs capped A/B/A with at least 10 reps (P14 §8) or n ≥ 20 single-stream decodes per arm (P13 §5 lever 1).

### 7.6.6 Reading Nsight Compute on SM121

```
BW_kernel = (dram__bytes_read.sum + dram__bytes_write.sum) / gpu__time_duration.sum        B/s
ratio_bw  = BW_kernel / BW_ceiling,node                   BW_ceiling from an on-node GEMV probe, not 273 GB/s
I         = FLOPs / DRAM bytes                            FLOP/B
I*        = P_peak / BW_ceiling                           ridge point
```
| input | value | grade | source |
|---|---|---|---|
| `BW_ceiling` proxy | GEMV p50 216–235 GB/s (weight bytes only), clock-locked, 8 community nodes; no own probe | [Community-measured] | P14 §4 |
| `P_peak` BF16 | ≈123 or ≈213 TFLOPS, contested | [Community-measured] | core §1 row 12 |
| `I*` BF16 | ≈ 500–990 FLOP/B (123 TFLOPS / 245 GB/s to 213 TFLOPS / 216 GB/s, spanning the GEMV proxy and core `BW_eff`) | [Interpretation, contested inputs] | derived |
| `dram__*.pct_of_peak_*` | invalid on GB10: NVML reports no memory clock, so ncu's denominator is unknown | [Community-measured]; [Interpretation] | P14 §3 |
| tensor activity | read `sm__pipe_tensor_cycles_active`; `tc` counts UTCMMA work SM12x does not execute | [Interpretation on a vendor statement] | P14 §3 |

Classification: `ratio_bw` near 1 and `I < I*` → bandwidth-bound, route to P05; `tensor` pipe busy and `ε_clk` near 1 → compute-bound, route to P09/P06/P04; both low with few eligible warps → latency or occupancy, route to P15 (P14 §6 table). The recipe (external `-lgc` lock plus `--clock-control none`, component harness off traffic, `--list-chips`/`--query-metrics` first) is in core 7.6 step 4.

### 7.6.7 Procedure: profile to hypothesis

1. State the question as one phase and one metric (P14 §6 step 1).
2. Cheapest discriminating signal first (core 7.6 order).
3. From the capture compute `s` (unprofiled denominator), per-rank collective minima (7.6.4) and, if a kernel is named, `ratio_bw` and pipe activity (7.6.6).
4. Confirm the class by a second method: unprofiled component timing, `ε_clk`, or an on-node ceiling.
5. Write the hypothesis as a prediction: `ΔT/T ≤ s·(1 − 1/x)` with `e` = 1 as the bound, plus the metrics it must not move.
6. Test by unprofiled A/B/A (7.5). Record `e_est`. A near-zero `e` is a finding: the region is off the critical path.

### 7.6.8 Worked profile (node-1/2/3, DeepSeek V4.1 Flash EXL3 TP3)

Source: 2026-09-16 "DeepSeek V4.1 Flash EXL3 TP3 prefill profiling and tuning" [Measured] (P14 §4, §5 lever 1).

| step | reading | derived |
|---|---|---|
| host-callback lookup, CUDA events | 338.27 / 343.45 / 349.26 ms per full chunk; ~2.7–2.8 s per ~32K request | `s` = 2.75 / 23.8 ≈ 0.11–0.12 against the unprofiled TTFT [Interpretation] |
| row cache | lookup 8–30 ms at 95–99.8% hit, `x` ≈ 10–40 | predicted TTFT −10 to −11.5% if `e` = 1, i.e. prefill +11.5 to +13% [Interpretation] |
| end-to-end A/B, three prompts | +0.41%, +0.45%, −2.08% | `e_est` ≤ 0.45 / 11.5 ≈ 0.04: the lookup was off the critical path; cache reverted [Interpretation] |
| collective kernel, torch trace | ~15.5 s on two ranks vs ~2.7 s on the third, which ran longer in grouped GEMM (5.58 vs ~3.63 s) and gather (3.61 vs ~0.94 s) | wire ≤ ~2.7 s over the capture; ~12.8 s was waiting; the two named categories explain ~4.6 s of it [Interpretation] |
| unprofiled per-rank component check | 35.90 / 35.65 / 36.11 ms | max/min 1.3%: the imbalance was not persistent; the trace's skew was profiler-induced or transient [Interpretation] |

Both trace-suggested hypotheses died at step 4 or step 6 of 7.6.7.

---

## 7.7 Worked example: one lane through the protocol

Lane: GLM 5.3 Flash EXL3, TP3/EP3 on node-1 (head), node-2, node-3; DFlash2 drafter with an adaptive `k` set {2, 4, 7}; FlashInfer autotune disabled (`--no-enable-flashinfer-autotune` on the logged launch line of the 2026-09-20 arms; V3.2 changes only fabric, draft-KV and overlay settings) [Measured, launch log]; frozen harness (fixed coding and prose fixtures, C1, temperature 0, thinking off, cold nonce, forced length; fixed prefill manifests at ~7.6K and ~95K). Packets: 2026-09-20 "GLM 5.3 Flash V3: retained startup and prefill improvements" (arms A0, A1, B, C) and 2026-09-23 "GLM 5.3 Flash TP3 V3.2: frozen-harness verification" (V3.2 cutover, a single-factor prefill A/B, an ABLIT arm). All readings [Measured]; medians of 5.

### Step 1 — Band from null-predicted arms (7.1.3)

| arm | why null-predicted for C1 decode | coding tok/s | prose tok/s |
|---|---|---:|---:|
| A0 (opening baseline) | identical config | 44.291 | 36.821 |
| A1 (closing baseline) | identical config | 43.815 | 38.250 |
| V3.2 | V3 startup and large-M prefill group, fabric addresses, draft-KV compaction, overlays, and the 2,100 MHz cap (A0/A1 ran stock): no change to weight bytes, `k` set or collective count; cap effect on decode unresolved (7.6.5) | 44.339 | 37.752 |
| V3.2a | V3.2 with one prefill-path overlay removed | 44.628 | 36.967 |

- `s_arm`: coding ≈ 0.34 tok/s (0.8%); prose ≈ 0.67 tok/s (1.8%) [Interpretation, n = 4 boots].
- `MDE_lane` (A/B/A, `m` = 1) ≈ 2.45 · `s_arm`: **coding ≈ 1.9%, prose ≈ 4.4%** [Interpretation].
- Admission caveat: the packets record no written prediction for V3.2 or V3.2a, and the cap changed between A0/A1 and them, so they are retrospective arms that 7.1.3 would not pool into a production band; the pooled band here is illustrative and, across the cap change, conservative. The pool also contains the arm tested in step 2. With A0 and A1 alone (identical config, 2 boots): coding SD 0.76% → `MDE` ≈ 1.9%, prose SD 2.7% → `MDE` ≈ 6.6% [Interpretation]. Every conclusion below holds under both bands.
- Consistent with the lane's recorded run spread, prose 36.8–38.3 [Measured] (2026-09-23 V3.2 verification, ABLIT addendum), and below the community ~8% boot spread; autotune is off on this lane, but no arm isolated it as the cause (7.1.2) [Interpretation].

### Step 2 — The quick probe, tested against the prediction

- Prediction (reconstructed here; the packet records none): decode Δ = 0 ± MDE, since no term of the bytes model moved [Interpretation].
- Probe: one 36-token prompt, 400 tokens, no warm-up, no nonce, 3 reps: ~30 → ~41 tok/s, +38% [Measured, withdrawn].
- 7.6.1 test: 38% ≈ 9 × prose MDE, no mechanism → artifact until reproduced.
- Diagnosis of the probe's own baseline: ~30 vs the frozen prose value 36.8 → the "before" reading was ~18% low [Interpretation]; candidates are `bias_first` (7.1.4) and adaptive-`k` EMA state after two days of traffic (packet; unverified).
- Frozen rerun: coding +0.1% vs A0 and +1.2% vs A1; prose +2.5% vs A0 and −1.3% vs A1: both inside MDE → "no decode change detectable at MDE 1.9% / 4.4%". The +38% claim was withdrawn on the record.

### Step 3 — Prefill: a normalization that could not decide

- Reference arm B (V3 group, 2026-09-20) predates the 2,100 MHz cap. V3.2 read 1,555 (7.6K) and 1,594 (95K) tok/s vs B 1,700 and 1,830.
- Normalize B by the clock factor (7.4): ×0.943 (the −5.7% measured at a 76.5K fixture) → ~1,726 at 95K, leaving V3.2 ~8% low; normalizing by the ratio observed at 7.6K leaves ~3% [Interpretation, packet].
- The spread between the two normalizations (~5 points) is as large as the residual being sought (3–8%): normalization cannot decide. The packet correctly classified the residual as unresolved and named suspects in order.

### Step 4 — Single-factor A/B/A in one session, same cap

| prefill tok/s (median of 5) | A: V3.2 | B: V3.2a (align-chunking overlay off) | A′: V3.2 restored | `d` = A′/A − 1 | `e_1` / `e_2` |
|---|---:|---:|---:|---:|---:|
| ~7.6K | 1,555 | 1,624 | 1,554 | −0.06% | +4.4% / +4.5% |
| ~95K | 1,594 | 1,747 | 1,608 | +0.9% | +9.6% / +8.6% |

- Validity: `|d|` ≤ 0.9% across two restarts, so this pair's start-to-start prefill spread is below the ±2% in-boot floor and one start per arm suffices (P04 §6: measure start-to-start spread; use two or three starts per arm only where it is larger) → set valid [Interpretation].
- Effect: at 95K both `e` ≈ 10 × `|d|` → established with one boot per arm. At 7.6K both `e` ≈ 4.5% vs `|d|` < 0.1% → established.
- Prediction check: V3.2a ≈ V3 × 0.955, i.e. the whole residual of step 3 is the overlay once the clock factor is applied (packet) [Interpretation]. The model's missing term was a correctness overlay that changes prefill chunk boundaries.
- Decision (operator): overlay dropped; production became V3.2a.

### Step 5 — A decode arm that the band cannot call

ABLIT off vs on, V3.2a, same session [Measured]: coding 44.628 → 44.643 (0.0%); prose 36.967 → 38.301 (+3.6%); prefill +0.4% / +0.2%.
- Prose +3.6% < prose MDE 4.4%, and it sits inside the null-arm range 36.8–38.3 → "no effect detectable at 4.4%". To call a 3–4% prose cost would need `m` ≈ 2–3 boots per arm at the pooled band, 3–5 at the A0/A1 band (7.1.2) [Interpretation].
- Recorded as "at most ~3–4% prose, nothing on code or prefill"; ABLIT kept on by the operator. On a TP2 lane the same transplant cost −11% prose [Measured] (2026-09-08 GLM 5.3 Flash EXL3 E4 five-arm A/B, as cited in the 2026-09-23 V3.2 verification). The TP3 figure does not transfer; topology is the likely cause, but other lane settings also differ between the two measurements [Interpretation].

### What the example shows

1. The band came from the lane's own arms, not from a community rule; its `MDE` (1.9% / 4.4%, or 1.9% / 6.6% from A0/A1 alone) sits below the community ~8% threshold, with autotune off but the cause not isolated.
2. A written prediction (reconstructed here; none was recorded) makes the +38% claim fail the plausibility test in one line; the frozen harness killed it.
3. Normalization across a clock change produced two answers ~5 points apart; only a same-session single-factor A/B/A with a restore arm decided.
4. An effect inside the band is written with its MDE, and the number of boots needed to resolve it is stated.

---

## 7.8 Traps

Traps listed in core 7.2 (quick probes, fixtures, SSE counting, first request, cache reuse, dashboards, labels, profiled runs) are not repeated.

| # | trap | consequence | guard | grade |
|---|---|---|---|---|
| 1 | Reading "flat" or "unchanged" without an MDE | a real cost below the MDE becomes "no cost"; e.g. "decode flat at every cap" while C1 means fell ~6% inside spread | write "no change detectable at MDE x%" (7.5 step 5) | [Historical diagnostic] (core 7.2) |
| 2 | Pooling arms into a band after seeing them, or pooling arms whose prediction was not null | band inflated by a real effect, or tightened by selection | only pre-registered null arms, same harness checksum (7.1.3) | [Interpretation] |
| 3 | Using a min–max range as if it were an SD | MDE overstated by ~1.7–2.3× for n = 3–5 (expected range ≈ 1.7σ at n = 3, 2.3σ at n = 5) | compute SD from per-run or per-arm values; keep every value | [Interpretation] |
| 4 | Applying a normalization factor measured at another depth, fixture or lane | a residual that flips between 3% and 8% | measure the factor on the target fixture, or run a same-day control (7.7 step 3) | [Measured] |
| 5 | Dropping the restore arm A′ | drift and effect indistinguishable | A/B/A every time; `d` is the validity test | [Measured] (P14 §4 bundle row: all ten pairs improved, per-pair gain 1.76–32.45%, no closing baseline, no attribution) |
| 6 | A historical baseline for one cell of a same-day comparison | the C8 row of a bundle compared against the same script run on an earlier day, nonces differing | comparability rule, core 7.5 | [Measured] (7.3.2) |
| 7 | An aggregate gain reported as speed | users on long outputs get slower turns | report `r_s`, `D`, `N*` with `R_agg` (7.3) | [Measured] |
| 8 | Share tables with the profiled denominator | shares inflated in CPU-heavy regions (`o` ≈ +51%) | unprofiled denominator (7.6.3) | [Measured] |
| 9 | Collective kernel time read as network cost | fabric work on a compute imbalance | per-rank minimum bounds wire (7.6.4) | [Measured] |
| 10 | Local speedup read as an end-to-end prediction | a 10–40× local win at `e` ≈ 0.04 | exposed-time Amdahl, `e` measured (7.6.2) | [Measured] |
| 11 | Percent-of-peak DRAM from ncu, or percent of 273 GB/s | an unfalsifiable "headroom" number; a mixed-rank "66–92% of nominal" reading was withdrawn | bytes over duration against an on-node ceiling, one rank at a time | [Measured] (P14 §7) |
| 12 | Profiling and A/B in the same boot; ncu beside serving | the profiler becomes the variable; a second CUDA context cannot be created beside a serving rank; replay save area competes with the warm floor | dedicated profiling boot; ncu on a component harness only | [Code-verified]; [Community-measured] (P14 §3, §6) |
| 13 | A clean profile read as health | a graph-mode hang was hidden by Nsight profiling and `CUDA_LAUNCH_BLOCKING=1` | health from progress counters and an unprofiled soak | [Community-measured] (P14 §3) |
| 14 | A bimodal lever judged on 3–5 runs | one run in four 15–30% below stock goes unseen | ≥ 30 runs and a histogram when a mode is reported | [Community-measured] (P19 4b-35) |

---

## Gaps

1. No own boot-variance study: `σ_B` and `σ_R` for any lane come from four null arms on one lane (7.7) or from community sources; the MDE formula's normality and independence assumptions are untested [Proposed: needs measurement] (≥ 3 boots of an identical config, per-run values).
2. The C8 quick script records forced 512-token outputs, median TTFT and batch completion, but not per-stream start and finish times, so the non-decode share of `D` splits only approximately into TTFT and tail wait (7.3.2).
3. No own Nsight Systems or Nsight Compute capture exists; overhead `o` per mode on GB10, which metrics return values on SM121, and the chip identifier are unknown (P14 §8).
4. No own on-node GEMV or STREAM ceiling; `BW_ceiling` and `I*` rest on community probes and a contested tensor peak.
5. Decode clock elasticity on TP lanes unresolved (7.6.5).
6. Protocol and fixture factors `f_i` exist for few lanes and classes; EOS-mode and thinking-mode factors are unmeasured on most own lanes.
7. The exposed fraction `e` has one own measurement (7.6.8).
8. The lane-card template has no noise-band, MDE or null-arm field (core Gaps 12).
9. The adaptive-`k` EMA state as a source of `δ(t)` after long traffic is a named suspect, not measured.
10. Requirement items not carried here: a single metric glossary with the flag that enables each metric (requirements gap 24; `last_gen_throughput` averaging has no wiki source) and honest external bands by model × topology × workload class (gap 25). Both remain core Gap 15 items.

## Sources

Wiki pages (fetched 2026-09-23):
- [Profiling protocol](../14-profiling-protocol.md) (P14) §2–§8, read in full
- [Decode optimization: the bytes model](../05-decode-optimization.md) (P05) §4, §6, §7
- [Case study: community recipe authors](../19-case-study-community-recipe-authors.md) (P19) §1, §4, §6
- [Clocks, thermal, power](../13-clocks-thermal-power.md) (P13) §5 lever 1, via P14 and core
- [Quality and correctness gates](../21-quality-and-correctness-gates.md) (P21), by reference
- [Serving agent workloads](../22-serving-agent-workloads.md) (P22) §4, by reference
- [GB10 Inference Wiki](../README.md) measurement rules
- Core: "The Science of GB10 Inference — core" §1, §2.4, §3, §4.2, §4.7, §6.4, §7, §8

Own packets (date and title):
- 2026-08-16 "DeepSeek V4 Flash dual-GX10 decode tuning A/B"
- 2026-08-21 "Weschera k10 boot-gate YaRN qualification"
- 2026-09-03 "A1 DFlash k=3 vs k=7"
- 2026-09-08 GLM 5.3 Flash EXL3 E4 five-arm A/B
- 2026-09-16 "DeepSeek V4.1 Flash EXL3 TP3 prefill profiling and tuning"
- 2026-09-16 "GLM 5.3 Flash EXL3 TP3: combined runtime qualification"
- 2026-09-20 "GLM 5.3 Flash V3: retained startup and prefill improvements"
- 2026-09-21 "GLM 5.3 Flash TP3 GPU clock-cap sweep"
- 2026-09-23 "GLM 5.3 Flash TP3 V3.2: frozen-harness verification" (incl. the single-factor prefill and ABLIT addenda)

Workflow review: the 2026-09-23 reviews of agent sessions and 161 changelogs, and the requirements synthesis (§2 #2, #13, #23, #24, #25), restated export-safe.
