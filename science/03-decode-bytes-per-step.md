---
title: The Science of GB10 Inference — chapter 03, decode = bytes per step
created: 2026-09-23
status: checked
type: chapter
core_section: 3
---

# Chapter 3 — Decode = bytes per step

Extends core section 3. It does not restate it: the step equation, the term table, the family table, the speculation formulas and the selection procedure are in core 3.1-3.6. This chapter derives the per-class form of the floor, the expert-union bounds, the collective count, the depth rule and the head share. It then works one real lane end to end and gives the decision procedure and the traps.

Pointer key (as core 10.1): P01 [CUDA graphs and launch overhead](../01-cuda-graphs-and-launch-overhead.md); P02 [Speculative decoding](../02-speculative-decoding.md); P03 [Inter-node communication over DAC](../03-inter-spark-communication.md); P05 [Decode optimization: the bytes model](../05-decode-optimization.md); P12 [Sampling and the lm_head](../12-sampling-and-lm-head.md); P13 [Clocks, thermal, power](../13-clocks-thermal-power.md); P14 [Profiling protocol](../14-profiling-protocol.md); P20 [Model architecture cards](../20-model-architecture-cards.md); ledger [Levers ledger](../ledger.md).

Units: GB = 10^9 B, GiB = 2^30 B, ms per step, tok/s per stream unless marked agg. GB/s divided into GB gives seconds. Every derived number here is [Interpretation] unless graded otherwise.

---

## 3.1 Per-rank bytes, split by kernel class

### 3.1.1 Why one ruler is not enough

Core 3.3 divides all bytes by one `BW_eff`. In-engine kernels do not all reach the ruler. The floor is therefore a sum over kernel classes:

```
T_mem = Σ_c  B_c,r / BW_c                    c ∈ {experts, quantized dense, BF16 dense and head, drafter, KV, state}
T_step ≈ T_mem + N_coll · t_eff + T_small + T_host        (graphed step; T_small = device time of tiny kernels)
```

| term | meaning | unit |
|---|---|---|
| `B_c,r` | bytes of class c that rank r streams per step at the step's row count M | GB |
| `BW_c` | achieved read rate of the kernel that serves class c, at that M, read from the boot log's selected kernel | GB/s |
| `T_small` | device time of kernels too small to be bandwidth-bound (norms, index ops, rope) | ms |
| `T_host` | host path inside the step (graph encode, callbacks, per-layer host work) | ms |

Anchors for `BW_c` (all measured against a named probe; none has a DRAM byte counter, core Gap 4):

| class | kernel | `BW_c` | grade | pointer |
|---|---|---|---|---|
| W4A16 experts | Marlin MoE, M = 8-256 | 91-96% of 245.9 → 224-236 GB/s | [Community-measured] | P05 §4 |
| EXL3 experts | cuda-exl3 MoE | 81% of 241 at M = 1 (≈195); 90-91% at M = 8-128 (≈217-219) | [Community-measured] | P05 §4 |
| BF16 dense, in-engine, small M | vLLM fork, NVFP4 TP3 profile | 5.90 GB/rank in ~51 ms → ≈116 GB/s | [Interpretation] on [Community-measured] | P05 §4 |
| BF16 dense (KDA projection) | same GEMM in-engine vs standalone | 150 vs 214 GB/s | [Community-measured] | P05 §4 |
| BF16 LM head | `F.linear`, vLLM | 64% of 273 at batch 1 (≈175), 86% at batch 2-4 (≈235) | [Community-measured] | P05 §4 |
| FP8 weight-only (Marlin FP8) | — | no GB10 figure; use the W4A16 Marlin band as a proxy and mark it | [Proposed: needs measurement] | — |

Rule [Interpretation]: two BF16 kernels measured in-engine at small M ran at 116-175 GB/s (one fork's dense GEMMs in a TP3 profile; a BF16 `F.linear` head at batch 1), under the 230-245 GB/s ruler. On a lane whose non-expert weights stay BF16 (the GLM NVFP4 export keeps them BF16, P20 §4.1), a single-ruler floor over-predicts tok/s; the calibration below is consistent with that. It is not a general BF16 rate: BF16 dense models land within ~10% of the bytes model (Gemma-3-27B BF16 at ~89%) [Community-measured], and why NVFP4 dense models land 25-40% below it (Llama-3.3-70B at 75%, Nemotron Super 49B at ~59%) is open (P05 §4, §8; core 3.3).

Calibration on a measured no-speculation point (2026-08-27 GLM 5.3 Flash MTP-4 versus MTP-off, TP2 on node-1 + node-2, vLLM, Marlin NVFP4 MoE) [Measured]: 14.1-14.2 tok/s on every content class.

| term | bytes per rank | rate | ms |
|---|---|---|---|
| BF16 non-expert (15.9 GB non-expert minus ≈0.25 GB NVFP4 dense MLP, ÷ 2) | ≈7.8 GB | 116-175 GB/s | 44.7-67.5 |
| NVFP4 experts at M = 1 (4.8 GB ÷ 2) plus NVFP4 dense MLP | ≈2.5 GB | 224-236 GB/s | 10.7-11.3 |
| KV and state | ≈0.1 GB | ≈230 GB/s | ≈0.5 |
| collectives: 91 per forward (3.1.4) × 41-150 µs (TP2 floor to TP3 in-engine; TP2 in-engine unmeasured) | — | — | 3.7-13.7 |
| **T_step** | | | **59.6-93.0 → 10.8-16.8 tok/s** |

Byte inputs are from P20 §4.1 [Interpretation from Code-verified inventories]. The measured 14.1 falls inside the bracket. A single 230-245 GB/s ruler predicts 17-22 tok/s, which over-predicts by 20-55%. The bracket is ±22% wide: the floor is a bracket, not a point.

### 3.1.2 The expert term: two bounds

For an MoE layer with `E` experts, top-`k_top` routing and `EP` expert-parallel ranks, the distinct local experts that rank r touches at M rows are bounded:

```
E_lo(M) = (E/EP) · k_top/E  = k_top/EP                 all M rows route like one row (fully correlated)
E_hi(M) = (E/EP) · [1 − (1 − k_top/E)^M]              independent uniform routing (core 3.2)
B_exp,r ∈ [E_lo, E_hi] · b_exp · L_moe                 b_exp = bytes of one expert in one layer; L_moe = MoE layers
```

`E_lo` is a floor only on average. `E_hi` is the upper bound. Two datasets put the truth below `E_hi`. Own EXL3 TP3 per-layer logs imply 172-351 GB/s per rank under uniform counting, above the ruler on one rank at 1-2 rows [Historical diagnostic]. The same model's NVFP4 TP3 profile implies 325-331 GB/s at M = 8 and ~289 GB/s at M = 64 [Interpretation on Community-measured] (P05 §3, §4). If that expert kernel runs at its 224-236 GB/s band, real expert bytes there are ≈0.7 × `E_hi` at M = 8 and ≈0.8 × `E_hi` at M = 64 [Interpretation]. Correlated routing and L2 reuse are the candidate causes; telling them apart needs a `topk_ids` histogram and a DRAM byte counter (P05 §8). Until then, carry `E_lo`, `E_hi` and the ≈0.7-0.8 × `E_hi` point as a bracket.

GLM 5.3 Flash at EP3 (288 experts, top-8, 42 MoE layers, 304.4B routed parameters; P20 §4.1 [Code-verified inventory]). At 4.0 bpw, `b_exp · L_moe` = 304.4e9 / 288 × 0.5 B ≈ 0.528 GB per unit of `E_r` (EXL3 scale overhead not in the evidence):

| M (C1, k) | `E_hi` of 96 local | `B_exp,r` upper | `B_exp,r` lower |
|---|---|---|---|
| 1 (no spec) | 2.67 | 1.41 GB | 1.41 GB |
| 3 (k = 2) | 7.78 | 4.11 GB | 1.41 GB |
| 5 (k = 4) | 12.61 | 6.66 GB | 1.41 GB |
| 8 (k = 7) | 19.37 | 10.24 GB | 1.41 GB |
| 64 (C8, k = 7) | 80.18 | 42.4 GB | 1.41 GB |

Consequences:
- Verify is not near free on sparse MoE. Under `E_hi`, going from k = 3 to k = 7 multiplies expert bytes ≈1.9× at EP3. Dense targets have no such term (core 3.5, the 138 ms flat step).
- At C8, k = 7 the rank reads ≈84% of its local experts under `E_hi`. The bytes then saturate, so aggregate throughput can keep rising with C while per-stream falls [Interpretation]. The one own C8 datapoint is a multi-change bundle, not a test of this: C8 agg 76.41 → 123.06 and per-stream 21.58 → 17.80 with slots raised 4 → 8, which the packet reads as more simultaneous streams and less admission waiting [Measured] (2026-09-16 combined runtime qualification).
- EP divides both bounds, so adding a node cuts the expert term most when M is large (3.7.3; core 5.2).

### 3.1.3 Replication, padding and depth

Per-rank bytes by term and topology (extends core 3.2):

| term | per-rank rule | example |
|---|---|---|
| sharded weights | `W_shard / TP · (H_pad / H)` when heads are padded to divide TP | GLM 64 → 66 heads (+3.1% on attention bytes); DS V4 Flash 64 → 72 (+12.5%) [Community-measured; Measured] (P20 §4.3; core 5.2) |
| replicated weights | full on every rank (routers, some norms; small) | GLM indexer + routers + HC ≈ 0.17B params [Interpretation] (P20 §4.1) |
| GQA KV | `KV_total / TP` while `n_kv ≥ TP`; stock vLLM refuses TP that does not divide `n_kv` | Qwen 3.8 27B (4 KV heads) runs TP1/TP2, not TP3 [Code-verified] (P20 §4.3) |
| MLA latent KV | full on every rank; more ranks add bandwidth, not KV tokens | DS V4.1 1,670.75 B/token/rank [Measured] (core 2.1) |
| sparse attention (DSA, capped top-k) | `min(depth, top_k_attn) · KV_bpt · rows` + `indexer_bpt · depth` | GLM: 2,048 selected × 7,216 B ≈ 14.8 MB per row; indexer keys ≈363 B × depth [Interpretation] (P20 §4.1) |
| hybrid recurrent state | constant in depth, ÷ TP by heads; read and written each step | GLM KDA 1.44 MB/layer/rank per sequence (22 padded heads × 128 × 128 × 4 B) × 34 layers ≈ 49 MB [Interpretation from Code-verified shapes] (P20 §4.1) |

Depth at which the KV read equals the weight read, for a dense GQA target:

```
D* = B_weights,r / KV_bpt,read,r          share_KV(D) = D / (D + D*)
```
`B_weights,r` = per-rank weight bytes per step (B); `KV_bpt,read,r` = KV bytes read per token of depth per rank, drafter KV included (B/token); `D`, `D*` in tokens.

- Qwen 3.8 27B NVFP4 + DFlash2, one node: 20.09 GB / 43,008 B (FP8 target + drafter KV) ≈ 467K tokens. At 84K, KV ≈ 15% of the read. Measured arithmetic from the allocator log: +13 ms on a ~131 ms step; BF16 KV ≈ +40% step time at 262K [Interpretation on Measured] (P05 §4).
- GLM (capped sparse attention): the indexer term alone would need ≈40M tokens to match the per-rank verify weights. Decode is near depth-flat by construction [Interpretation]. Measured in one qualification: depth fixtures 30.95 tok/s at 64K and 32.84 at 120K against legacy prose 31.84 shallow; a retrieval fixture decoded 41.8 at ~1M, a different content class from the 44.58 shallow coding figure [Measured] (2026-09-14 TP3 triangle qualification).
- MLA with an indexer whose scoring grows with depth (DS V4 Flash on one single-node engine): decode fell 23% from 2K to 64K [Community-measured] (P05 §4), while the ≈3.85 KB/token KV read adds ≈0.25 GB to a ≈10-13 GB step at 64K (≈2%). Bytes alone do not explain it; indexer compute is the consistent explanation [Interpretation] (P20 §4.1). Do not model MLA depth cost with bytes only.

### 3.1.4 Collective count from the layer list

With row-parallel outputs, each attention or mixer block and each MLP or MoE block ends in one all-reduce. The vocabulary head adds one gather:

```
N_coll,target ≈ L_attn + L_ffn + 1          (EP with a summed partial: one all-reduce per MoE layer, no all-to-all)
N_coll,step   = N_coll,target + N_coll,draft (a TP-sharded drafter adds its own)
```

GLM TP3: 45 + (3 dense + 42 MoE) + 1 = 91 per target forward, plus ≈11 for the drafter = 102 per decode step, and 91 per prefill chunk [Code-verified] (P03 §2). Fusions that would cut the count are disabled on GB10 multi-node (core 5.1). Community-measured in-engine counts on other lanes (104 DS V4.1 TP3, 108 GLM NVFP4 TP3) sit in the same band [Community-measured] (core 5.1). The count does not depend on TP width: every TP ≥ 2 topology pays about 2L + 1 collectives per target forward [Interpretation].

### 3.1.5 Topology table with the tax

Core 5.2 gives the width equation. Per-family ceilings with the tax added. Byte inputs are from P20 §4.1, no speculation, M = 1, `BW_eff` 240 GB/s, `t_eff` 130-150 µs per collective. TP2's in-engine `t_eff` is unmeasured, so TP2 uses 41-150 µs:

| family and export | TP | `W/TP` (GB) | `T_mem` (ms) | `N_coll · t_eff` (ms) | ceiling with tax (tok/s) | ceiling without tax, P20 (tok/s) |
|---|---|---|---|---|---|---|
| Qwen 27B NVFP4 17.6 GB | 1 | 17.6 | 73.3 | 0 | 13.6 | ≈13.6 |
| GLM NVFP4 20.7 GB | 2 | 10.35 | 43.1 | 3.7-13.7 | 17.5-21.4 | ≈23 |
| GLM NVFP4 20.7 GB | 3 | 6.9 | 28.8 | 11.8-13.7 | 23.5-24.6 | ≈35 |
| DS V4.1 native 13.0 GB | 3 | 4.33 | 18.1 | 13.5-15.6 (104) | 29.7-31.6 | ≈55 |
| DS V4.1 EXL3 8.07 GB | 3 | 2.69 | 11.2 | 13.5-15.6 | 37.3-40.5 | ≈89 |
| Qwen 27B NVFP4 17.6 GB | 2 | 8.8 | 36.7 | 5.3-19.4 (≈129 = 2 × 64 layers + 1, layer count from P20 §4.1 arithmetic; not code-verified) | 17.8-23.8 | ≈27 |

[Interpretation]. Every GLM and DS row assumes the uniform single ruler; apply 3.1.1 where non-expert weights are BF16. Tax share = `N_coll · t_eff / T_step`. The share grows as `W/TP` shrinks: 8-24% for GLM NVFP4 at TP2, 29-32% at TP3, 43-46% for DS V4.1 native at TP3, 55-58% for its EXL3 export. The small-weight, wide-topology lanes gain most from dividing the tax by the acceptance count `a` (3.4). The Qwen TP2 row's collective count is not in the evidence: [Proposed: needs measurement]. The 1-node row pays no tax; its measured NVFP4 no-spec point, ≈10.1 tok/s (one 256-token request, elapsed incl. TTFT) [Measured] (P05 §4), is ≈74% of that floor, inside the unexplained NVFP4 dense shortfall (3.1.1).

---

## 3.2 The bandwidth floor as a test, not a forecast

```
R = T_step,measured − T_step,predicted        ρ = R / T_step,measured
```

| observation | reading | next action |
|---|---|---|
| implied BW (bytes ÷ measured step) above the probe ruler | the byte count is too high: correlated routing, L2 reuse, or accepted rows counted instead of verified rows | recount verified rows; bracket with `E_lo`; never report a percent of ceiling [Historical diagnostic] (P05 §7) |
| ρ within the bracket | the model explains the step | rank levers by byte share (3.7.4) |
| ρ more than ~25% above the upper bracket (threshold [Interpretation], from P05's ±25% multi-node MoE guide) | an unmodelled term: host path, tiny kernels, a slow rank, eager shapes | profile before any lever (P14 §6; core 7.6) |

Worked residual on the record. One single-node DFlash2 packet stated ~206 GB/s. Its own byte table gives 20.09 GB over the measured 138 ms step, which is ≈146 GB/s. The 206 figure is not reproducible from the packet (P02 §3) [Interpretation]. Only the flat 138 ± 3 ms step is load-bearing.

---

## 3.3 Launch and small-kernel term

```
T_launch = N_eager · t_launch (host, 1-5 µs each, unmeasured on GB10)       only for steps outside captured shapes
T_small  ≈ N_small · t_small                                                 inside graphs; device time
```

- `t_small` ≈ 5 µs: ~2,000 small kernels took ~10 ms of an ~82 ms DS V4.1 TP3 step [Interpretation on Community-measured] (P01 §4). Fusing them can remove at most their device time; the serving gain on GB10 is unknown, and launch removal plus fusions on an already bandwidth-bound single-node step measured flat [Community-measured] (P01 §5 lever 6).
- Host path inside a graphed step: a stock EXL3 MoE layer costs 0.107 ms more than the cooperative kernel at 1 row. That is ≈4.5 ms per step over 42 layers, and it is not bytes: P05 §4 calls it host path, P01 §4 the stock path's extra allocations and small kernels [Historical diagnostic; Interpretation]. ds4 host encode ≈10 ms of 55 ms [Community-measured] (P05 §2).
- Coverage set. Every shape the scheduler can emit must be captured:

```
S_req = { n · (k + 1) : n ∈ 1..slots, k ∈ k-set }       |S_req| ≤ captured sizes;  max(S_req) ≤ fused-row cap
```

For slots 8 and k-set {2, 4, 7}: multiples of 3 to 24, of 5 to 40 and of 8 to 64, minus {15, 24, 40} counted twice = 21 required shapes, ceiling 64 [Interpretation]. The lane in 3.7 captures 25 shapes through 64 rows; the evidence gives the count and the ceiling, not the list, so confirm the 21 are among them from the boot log (read from the lane card: slots).
- Fused-row boundary: `slots · (k_max + 1)` must not exceed the fused-MoE temp-rows limit. 8 × 8 = 64 sits exactly on it [Measured] (P05 §7). Any change that lowers the limit (for example 64 → 32) or raises slots or `k_max` moves the top shapes onto a slower path.

---

## 3.4 Collectives term per committed token

```
τ_tok = N_coll,step · t_eff / a          (tax per committed token)
```

- Speculation divides the tax because it is charged per step [Interpretation] (P03 §5 lever 3). A TP-sharded drafter adds ≈11 collectives per step to buy its `a` [Code-verified] (P03 §2).
- Worked: GLM TP3, 102 × 130-150 µs = 13.3-15.3 ms per step. At `a` = 3 that is 4.4-5.1 ms per token; at `a` = 7.7 (counting) 1.7-2.0 ms [Interpretation].
- Over a slow link the same law holds: an accept length of ~2.6 moved a TP3 lane on 1 GbE only 6 → 7.5 tok/s [Community-measured] (P02 §3).
- Acceptance did not change from TP2 to TP3 on the same weights and drafter (64K 0.39-0.51, 120K 0.39-0.46) [Measured] (2026-09-14 TP3 triangle qualification). Model a topology change as a change in `T_step` at fixed `a`.

---

## 3.5 Speculative decoding: acceptance and depth

### 3.5.1 Convention map

| logged quantity | engine | convert to `a` (committed per step per stream) |
|---|---|---|
| draft acceptance rate `α_pd` = accepted ÷ proposed | vLLM and forks | `a = 1 + k · α_pd`; with adaptive k, use the k actually run, not `k_max` |
| accept length `L` (bonus included) | SGLang | `a = L` |
| per-position rates `q_i` | vLLM detailed metrics | `a = 1 + Σ q_i` |
| DSpark "k" (block γ) | SGLang | draft tokens N = γ + 1; rows M = C · (γ + 1) [Code-verified] (P02 §2) |
| tok/s × measured step | any | `a = tok/s · T_step`: the only route when counters are missing |

### 3.5.2 The chain model over-predicts deep acceptance

Core 3.5's i.i.d. form `a = (1 − p^{k+1}) / (1 − p)` fits one depth, not two. GLM EXL3 TP2 prose, measured `α_pd` 0.645 at k = 3 and 0.367 at k = 7 [Measured] (2026-09-03 A1 k=3 / k=7 qualification):
- k = 3: `a` = 2.935. Solving `p + p² + p³ = 1.935` gives `p` ≈ 0.80.
- The chain model then predicts `a(7)` = 4.13. Measured `a(7)` = 1 + 7 × 0.367 = 3.57. The model over-predicts by 16%.
- Implied `p` falls from ≈0.80 to ≈0.75 with depth on this lane. That is not general: the MTP-4 per-position profile 84.8 / 70.5 / 59.3 / 51.0% on another lane [Measured] (core 3.5) is near geometric (conditional rates 0.83-0.86 after position 1), and one `p` fitted at position 1 over-predicts its `Σq` by only ~1.4% [Interpretation]. The two drafters differ in kind: a block drafter rebuilt for each k (P02 §2) against chained MTP heads.

Rule [Interpretation]: extrapolate `a` to a deeper k only from measured per-position `q_i` on the same drafter. Never extrapolate from one fitted `p`.

### 3.5.3 Marginal rule and break-even acceptance

Committed throughput per stream is `a(k) / T(k)`. Adding draft position k + 1 adds `q_{k+1}` to `a`. It pays when:

```
q_{k+1} / a(k)  >  [T(k+1) − T(k)] / T(k)
```

For a jump from `k_0` to `k_1`, with `R_T = T(k_1) / T(k_0)`:

```
α*(k_1) = (R_T · a(k_0) − 1) / k_1          deeper k wins only if measured α_pd(k_1) > α*
```

| step-cost regime | `R_T` | consequence | evidence |
|---|---|---|---|
| dense, 1 node | ≈1 [Interpretation: the 138 ± 3 ms step is flat across accepted length at fixed k; not measured across k] | deeper wins while `a` rises, up to the drafter's trained block | k = 10 over k = 8, code +14.6%, prose +7.0% [Measured] (2026-08-21 k=10 boot-gated qualification); abliterated variant peaked at its trained block [Measured] (P02 §5 lever 2) |
| MoE, EP2, k 3 → 7 | 1.44-1.48 (derived from prose and code) | prose `α*` = (1.45 × 2.935 − 1)/7 ≈ 0.47, measured 0.367 (loss); code `α*` ≈ 0.40 (`a(3)` = 2.61), measured 0.319 (loss); structured `α*` ≈ 0.67 (`a(3)` = 3.93), measured 0.890 (win) | [Interpretation on Measured] (2026-09-03 A1 k=3 / k=7 qualification) |
| MoE, EP3, k 4 → 7 (3.7 model, uniform routing) | 1.17-1.27 | `α*` ≈ 0.36-0.40 on prose (`a(4)` = 2.985); the TP2 prose rate at k = 7 was 0.367, inside that band: no predicted gain | [Interpretation] |
| batched path that drops speculation | `a` → 1 at C ≥ 2 | any k loses at concurrency | ds4 fork 1.0 tokens/step at C2+ [Measured] (P02 §7) |
| verify ≈ 2 plain steps (ds4 DSpark, 1 + 4 rows) | ≈2.05 against plain decode | break-even `a` = 2.05. The author's "~51% acceptance" is a chain rate `p` (core 3.5); in this table's convention `α*(4)` = (2.05 − 1)/4 ≈ 0.26 per draft token [Interpretation] | [Community-measured] (P02 §3) |

Concurrency moves `R_T`. `M = C · (k + 1)`, so at higher C the verify rows reach the expert-saturation zone (3.1.2) and the dense compute region. At C5 the dense lane's k = 10 advantage vanished (163.2 vs 163.7 agg) [Measured] (2026-08-21 k=10 boot-gated qualification).

### 3.5.4 Depth by workload

Per-draft-token or engine acceptance by content, as measured. Structured and counting rows are listed only to be excluded from any decision.

| workload class | acceptance seen | depth consequence | source |
|---|---|---|---|
| counting / structured | 0.89-0.99 | deep always wins; excluded from k choice | [Measured] (P02 §4; core 7.2) |
| copyable edit | 91.7% at k = 16 (15.67 tokens per pass) | deep wins on dense (144-151 tok/s) | [Measured] (2026-08-20 vLLM DFlash2 k=16 qualification) |
| fresh code | GLM 0.32 (k = 7) / 0.54 (k = 3); dense 32.4% (k = 16); DS V4 DSpark 68-84% | shallow on MoE; deep on dense | [Measured] (P02 §4) |
| prose | GLM 0.37 / 0.65; DS V4 DSpark 25-34%; abliterated dense 12-25% at K8 | shallowest; check `α*` | [Measured] (P02 §4) |
| depth 64K-120K | GLM 0.28 (k = 7) → 0.59 (k = 3) at 64K | shallow: k = 3 beat k = 7 by +31% at 64K | [Measured] (2026-09-03 A1) |
| live agent traffic | 37.4-54.1% on four lanes; one lane 0.00 until YaRN was baked into the drafter | the class to optimise; read `/metrics` deltas | [Measured] (P02 §4, §7) |

Selection by operator weight (code vs prose) is lane state: read from the lane card: Owner notes. Adaptive k does not remove the choice:
- It trims target verify rows only. The drafter still runs one forward per step at its load-time block.
- The batch-minimum hook pins the whole batch at `k_max` while one structured request is live [Code-verified] (P02 §2).

---

## 3.6 Sampler and lm_head share

```
share_head = n_vocab_passes · (V · H · bytes_w / TP) / B_step,r          n_vocab_passes = 1 (target) + full-vocabulary draft passes
Δ_trim ≈ (B_head − B_head,trim) / BW_c  −  (Δa / a) · T_step              trim pays if Δ_trim > 0
```
`V` vocabulary rows, `H` hidden size, `bytes_w` bytes per weight, `B_step,r` per-rank bytes per step (GB); `B_head − B_head,trim` saved head bytes per step summed over draft passes (GB); `BW_c` the head kernel's rate (GB/s); `Δa / a` relative loss in committed tokens per step; `T_step`, `Δ_trim` in ms.

| lane | head bytes per pass per rank | share of the step | measured lever |
|---|---|---|---|
| Qwen 27B, one node, BF16 head + DFlash2 | 2.54 GB | 2.54 / 24.22 = 10.5% | head kept BF16 for correctness (a quantized head broke DFlash2 on SGLang) [Measured] (P20 §5 lever 10) |
| Flash-Next TP2, MTP-4, full 248K draft vocabulary | draft head 0.59 GiB → 0.16 GiB | own share not measured; a community byte budget for a dual-node MTP3 lane puts head reads (1 verify + 3 draft passes) at ≈26% of step bytes [Interpretation] | 64K list: C1 +13-19% while acceptance fell 58.6% → 52.2% [Measured] (P12 §3, §4) |
| GLM TP3 (3.7) | 0.42 GB if BF16 (precision not recorded) | 3-6% of step bytes (1-2 passes); 2-9% of step time at the 116-235 GB/s BF16 band | none tried; a head lever's ceiling is that share [Interpretation] |

- Sampler plus speculation bookkeeping: 0.07 ms per step at C1 (≈0.07% of a 98.9 ms step), 0.15 ms at C8 [Community-measured] (P05 §4). Not a lever at C1. A fused argmax measured +5-14% at C4 on one DSpark tree [Community-measured] (P05 §5 lever 11).
- TP head gather: ≈309,888 B per row with 16-bit logits (padded vocabulary 154,944); at 64 rows ≈17% of per-rank ring bytes. At C1 (≤ 8 rows) it is one collective of 102 and latency-bound [Interpretation] (P12 §3).
- Trim falsifier: acceptance loss × step time against saved bytes ÷ rate. An 8K list fell to 36% acceptance and lost everywhere [Measured] (P12 §4).

---

## 3.7 Worked example: GLM 5.3 Flash EXL3, TP3/EP3 on node-1/2/3

Lane: the production GLM 5.3 Flash EXL3 lane, vLLM with an EXL3 fork, DFlash2 drafter, adaptive k-set {2, 4, 7}, 8 slots, FP8 `fp8_ds_mla` KV.

Anchors:
- The decode benchmark in the 2026-09-16 combined runtime qualification (5 reps, 400-token ceiling, temp 0; clock state not recorded in the packet, which predates the 2100 MHz cap).
- The frozen-harness numbers in the 2026-09-23 GLM TP3 V3.2 verification (2100 MHz cap; at that cap the C1 decode mean was 5.8% below stock, inside the stock rep spread, n = 5 [Measured], 2026-09-21 GPU clock-cap sweep).
- The TP2 arm of the 2026-09-23 two-node evaluation, on node-1 + node-2.

### 3.7.1 Inputs

| input | value | source |
|---|---|---|
| experts | EXL3 4 bpw; 288 × top-8 × 42 layers; 96 local per rank | read from the lane card: model / weights; P20 §4.1 [Code-verified inventory] |
| non-expert | FP8 weight-only on dense, KDA, shared, MLA: 7.48B params → 7.48 GB, ÷ 3 = 2.49 GB | lane card: model / weights; P20 §4.1 |
| small BF16 (indexer, routers, HC) | 0.17B → ≤ 0.34 GB, replicated worst case | P20 §4.1 [Interpretation] |
| lm_head | 0.63B; BF16 assumed → 0.42 GB per pass per rank; 1-2 passes (verify, draft) | precision not on the lane card (Gaps) |
| drafter | ≈0.72 GiB (0.77 GB) per rank, TP-sharded through padded 36/9 heads; one forward per step | [Code-verified; per-rank dump owed] (P02 §2, §7) |
| KV and state at 64K, M = 8 | ≤ 0.12 GB selected KV + 0.02 GB indexer keys + ≈0.1 GB KDA state read and write | 3.1.3 |
| collectives | 102 per step × 130-150 µs | 3.1.4; core 5.1 |
| class rates | experts 195-219 (cuda-exl3 proxy: 195 at M = 1, 217-219 at M = 8-128; the lane's cooperative kernel has no rate measurement); FP8 224-236 (Marlin W4A16 proxy); BF16 116-235 GB/s | 3.1.1 [Proposed: proxies] |
| graphs | 21 required shapes; 25 captured through 64 rows (the list is not in the evidence) → `T_launch` ≈ 0 if the 21 are among them | 3.3 |

### 3.7.2 Predicted verify step at C1, uniform routing (upper bound on expert bytes)

| k (M) | experts (ms) | FP8 non-expert (ms) | BF16 small + head + drafter, 1.53-1.95 GB (ms) | KV / state (ms) | collectives (ms) | `T_verify` (ms) |
|---|---|---|---|---|---|---|
| 2 (3) | 18.8-21.1 | 10.6-11.1 | 6.5-16.8 | ≈1.0 | 13.3-15.3 | 50.2-65.3 |
| 4 (5) | 30.4-34.2 | 10.6-11.1 | 6.5-16.8 | ≈1.0 | 13.3-15.3 | 61.8-78.4 |
| 7 (8) | 46.8-47.2 | 10.6-11.1 | 6.5-16.8 | ≈1.0 | 13.3-15.3 | 78.2-91.4 |
| no spec (1), drafter off, 91 collectives | ≈7.2 | 10.6-11.1 | 3.2-6.6 (0.76 GB) | ≈0.6 | 11.8-13.7 | 33.4-39.2 → 25.5-29.9 tok/s |

Expert rates: 195-219 GB/s at M = 3 and 5 (between the proxy's M = 1 and M = 8 anchors), 217-219 at M = 8, 195 at M = 1. With correlated routing (`E_lo`), the expert column at every k falls to 6.4-7.2 ms and `T_verify(7)` to ≈38-51 ms. At the ≈0.7 × `E_hi` point that the NVFP4 profile implies at M = 8 (3.1.2), the k = 7 expert column is ≈33 ms and `T_verify(7)` ≈64-77 ms. All rows [Interpretation].

### 3.7.3 Measured against predicted

Measured anchor [Measured] (2026-09-16 combined runtime qualification): the same setting gave counting 101.655 tok/s at 95.88% acceptance and hash-map prose 44.151 tok/s at 49.62%.

| case | `a` and implied step | where it falls |
|---|---|---|
| counting, k saturated at 7 (EMA "saturate max") | `a` = 1 + 7 × 0.9588 = 7.71 → `T` = 75.9 ms | ≈3% below the uniform k = 7 band (78.2-91.4); inside the ≈0.7 × `E_hi` band (64-77); far above `E_lo` (38-51) |
| prose if k = 7 | `a` = 4.47 → `T` = 101 ms | above every band: rejected |
| prose if k = 4 | `a` = 2.985 → `T` = 67.6 ms | inside the uniform k = 4 band (61.8-78.4) |
| prose if k = 2 | `a` = 1.99 → `T` = 45.1 ms | below the uniform k = 2 band (50.2-65.3); needs expert bytes below `E_hi` |

[Interpretation]: prose is consistent with the EMA mostly at k = 4 under uniform routing, or with a k = 2-4 mix if its routing is as correlated as the 3.1.2 NVFP4 point; the mix is not logged (Gaps). The counting step sits near the uniform bound, not near `E_lo`. That is unexpected for repetitive text and is not explained without a routing histogram (Gaps).

Topology check, the same model at TP2/EP2 (144 local experts; drafter ≈1.16 GB per rank; the TP2 in-engine `t_eff` unknown, 41-150 µs):

| quantity | value |
|---|---|
| predicted `T_verify(7)` at TP2 | 100.2-127.5 ms |
| predicted TP2 ÷ TP3 speed at fixed `a` (`a` is TP-invariant, 3.4) | 0.61-0.91 → −9% to −39% |
| measured, same frozen prompts | code 44.3 → 35.6 (−20%), prose 37.8 → 30.7 (−19%) [Measured] (2026-09-23 two-node evaluation) |

Inside the band. At C1, k = 7, the width decision follows the expert term (÷ EP), not the collective count.

Speculation gain on this lane: predicted no-spec 25.5-29.9 tok/s against frozen-harness code 44.3 and prose 37.8 [Measured] (2026-09-23 V3.2 verification) → `S` ≈ 1.5-1.7 code, 1.3-1.5 prose [Interpretation]. The lane has no spec-off arm: [Proposed].

### 3.7.4 What the numbers rank

Shares at k = 7, uniform bound [Interpretation]:
- experts ≈ 52-60% of `T_verify`;
- collectives ≈ 17% (the NVFP4 TP3 community profile measured 13.8-15.1% [Community-measured], core 5.1);
- FP8 non-expert ≈ 12-14%;
- BF16 small + head + drafter ≈ 8-18%;
- sampler < 0.1%.

The implied order of levers:
1. The k policy and routing correlation: the expert term moves ≈1.9× between k = 3 and 7. The lane card's first untried lever (per-step instrumentation, then forced-k windows [7], [4], [2] in one boot) is the discriminating measurement. Read from the lane card: Levers not yet tried.
2. Expert and non-expert bytes: format, and full-scope 4-bit dense paths [Community-measured] (P05 §5 lever 2).
3. The collective tax, divided by `a`.
4. The head: 3-6% of step bytes, 2-9% of step time (3.6).

A change that lowers the fused-row limit to 32 pushes k = 7 at C ≥ 5 and k = 4 at C ≥ 7 off the fused path (3.3). The lane card's 5-7% decode-cost estimate for that lever [Proposed] is consistent with that.

What would falsify this example:
- A per-step log showing `T_verify(7)` < 60 ms. The expert term is then near `E_lo`, and deeper k gets cheaper than modelled.
- Prose EMA time mostly at k = 7. The model then misses ≈10 ms per step or more (101 ms implied against the 91.4 ms upper edge), which is a host or small-kernel term to profile.

---

## 3.8 Decision procedure

1. Read the lane card: model / weights, spec method and k, slots, engine version, NCCL env, clock cap. Missing fields become Gaps, not guesses.
2. Build the per-rank byte table by kernel class (3.1.1) at every reachable `M = C · (k + 1)`. Carry `E_lo` and `E_hi` for experts (3.1.2), and apply padding, replication and depth rules (3.1.3).
3. Read each class's selected kernel from the boot log. Assign `BW_c` from 3.1.1, and mark any proxy [Proposed].
4. Count collectives from the layer list (3.1.4). Multiply by `t_eff` 130-150 µs (TP3 in-engine) or bracket 41-150 µs (TP2). Never use the standalone floor alone (core 5.1).
5. Check the coverage set and fused-row boundaries (3.3). Any uncaptured reachable shape is fixed before measuring.
6. Take `a` per workload class from measured per-position rates or live `/metrics` deltas (3.5.1), never from structured fixtures. Predict tok/s = `a / T_verify` per class with its bracket. Write the prediction sheet before booting (core 7.3 item 1).
7. Measure on the frozen harness. Compute `ρ` (3.2). If the implied bandwidth exceeds the ruler, recount rows. If `ρ` exceeds the upper bracket by > 25%, profile before any lever (P14 §6).
8. Rank levers by class share of `T_verify` (3.7.4). Apply the Amdahl bound: gain ≤ share × class speedup.
9. For a depth change, compute `α*(k_1)` (3.5.3) from `R_T` (predicted, or measured from a forced-k window). Test only when the class's measured or per-position-extrapolated `α_pd(k_1)` clears `α*`. The decision class is real-shaped code, prose and depth, weighted per Owner notes.
10. For a width change, predict `T_step` at each TP with `a` fixed (3.4, 3.7.3). Run the width procedure in core 5.2.
11. Record predicted against measured per term, and the prediction error, in the packet. Update lane card fields and settled experiments.

---

## 3.9 Traps

| # | if you believe | you will | what happens | evidence |
|---|---|---|---|---|
| 1 | one ruler fits every kernel | over-predict tok/s on a lane whose non-expert weights stay BF16 (17-22 predicted against 14.1 measured on one NVFP4 export) and hunt a phantom "overhead" | the residual is consistent with in-engine BF16 GEMMs at 116-175 GB/s [Interpretation]; BF16 dense models still land within ~10%, so this is not a general BF16 rate | 3.1.1; P05 §4, §8 |
| 2 | implied BW above the ruler means a fast kernel | celebrate an impossible number | it is correlated routing, L2 reuse, accepted rows counted instead of verified, or a mixed-rank ratio (a 41% byte understatement at C1, k = 7 on one estimator) | [Historical diagnostic] (P05 §7) |
| 3 | verify is near free | pick deep k on MoE | `R_T` ≈ 1.45 for k 3 → 7 at EP2, consistent with the expert union [Interpretation on Measured]; k = 7 lost every real-shaped cell | 3.5.3; [Measured] (2026-09-03 A1) |
| 4 | one fitted `p` predicts deeper acceptance | over-predict `a(7)` by ~16% on one GLM DFlash2 lane | there, per-draft-token rates fell faster than i.i.d.; an MTP-4 profile was near geometric, so the decay is drafter-specific | 3.5.2 |
| 5 | adaptive k removes drafter cost at low k | model the drafter as k-scaled | the drafter runs its load-time block every step; one structured request pins the batch at `k_max` | [Code-verified] (P02 §2) |
| 6 | a k-set A/B isolates k | attribute the result to depth | the k-set changes which rows are captured and which fall to stock MoE kernels; confounded | [Measured] (P02 §7; P05 §7) |
| 7 | "k" means the same across engines | compute rows wrong | SGLang DSpark k = 5 runs 6 draft tokens; `L` includes the bonus token | 3.5.1; [Code-verified] (P02 §2) |
| 8 | the collective floor × count is the tax | predict half the real tax | in-engine `t_eff` includes rank waits: 130-150 µs, not 40-86 | [Interpretation] (P03 §4) |
| 9 | a TP-sharded drafter is free | forget its collectives | ≈11 more collectives per step on the GLM TP3 lane | [Code-verified] (P03 §2) |
| 10 | a recipe comment tells you drafter placement | propose "shard the drafter" | it was already sharded on every rank | [Code-verified] (P02 §7) |
| 11 | graphs make launch cost zero | skip the tiny-kernel term | ~2,000 small kernels ≈10 ms of an 82 ms graphed step | 3.3; [Community-measured] (P01 §4) |
| 12 | lowering the fused-row limit is a prefill-only change | cut decode at high C | 8 × 8 = 64 sits on the limit; a lower limit moves top verify shapes off the fused path | 3.3; [Measured] (P05 §7) |
| 13 | concurrency ladders show speculation scaling | publish a ladder from a batched path at `a` = 1 | C2 below C1 on two engines | [Measured] (P05 §7) |
| 14 | a quantized or scale-dropped head is a free byte cut | trust the acceptance it produces | draft and verifier agree on a broken head (p90 accept 8.0); a quantized head broke DFlash2 correctness | [Measured] (P02 §7; P12 §7) |
| 15 | a smaller draft vocabulary always pays | ship an 8K list | acceptance 36%, lost everywhere | [Measured] (P12 §4) |
| 16 | MLA depth cost is bytes | under-predict long-context decode | the byte term adds ≈2% at 64K where decode fell 23%; indexer compute is the consistent explanation [Interpretation] | 3.1.3; P05 §4 |
| 17 | an old packet's GB/s figure is data | propagate the "206 GB/s" | not reproducible from the packet's own table | 3.2; P02 §3 |

---

## Gaps

1. No per-step instrumentation on the worked lane. `T_verify(k)` is implied from tok/s and acceptance, not measured. The adaptive-k mix per class is not logged. [Proposed: needs measurement]
2. lm_head precision and drafter precision on the worked lane are not on the lane card. 3.7 assumes BF16 for both.
3. FP8 weight-only (Marlin FP8) read rate on GB10 is unmeasured. The W4A16 band is a proxy.
4. TP2 in-engine `t_eff` is unmeasured. Only the 41-43 µs standalone floor exists.
5. No `topk_ids` histogram. `B_exp` is carried as the `[E_lo, E_hi]` bracket on every MoE lane.
6. No spec-off arm on the worked lane. `S` is predicted, not measured.
7. The drafter's per-rank bytes (0.72 GiB) come from a code read. The per-rank shape dump is owed.
8. EXL3 4 bpw scale overhead is not in the evidence. Expert bytes assume exactly 0.5 B per parameter.
9. The collective count for dense Qwen 27B TP2 is not in the evidence (3.1.5 row).
10. `t_small` (≈5 µs) comes from one profile. Tiny-kernel count per step on the worked lane is unknown.
11. The acceptance-rate convention inside the fork's adaptive-k counters (proposed at `k_max` or at the EMA k) is not stated in the evidence. 3.7.3 assumes the k actually run.
12. The EXL3 expert kernel's rate at M = 3-5 and the worked lane's cooperative MoE rate are unmeasured. 3.7 brackets them with the cuda-exl3 proxy (195 GB/s at M = 1, 217-219 at M = 8-128).
13. The ≈0.7-0.8 × `E_hi` point comes from one NVFP4 TP3 profile of the same model (Marlin experts). Its transfer to the EXL3 lane is untested.

## Sources

Wiki pages (fetched 2026-09-23): P05 in full; P02 §2-§7; P12 §3-§5; P20 §4.1-§4.3; P03 §2, §4, §5; P01 §4-§5 by row; core.md sections 2-5 and 7 (not restated).

Own packets (date and title):
- 2026-08-19 single-node DFlash2 campaign audit
- 2026-08-20 vLLM DFlash2 k=16 qualification
- 2026-08-21 k=10 boot-gated qualification
- 2026-08-27 GLM 5.3 Flash MTP-4 versus MTP-off
- 2026-09-03 A1 k=3 / k=7 qualification
- 2026-09-05 FP8-KV qualification
- 2026-09-14 TP3 triangle qualification
- 2026-09-15 TP3 upgrade A/B
- 2026-09-16 combined runtime qualification
- 2026-09-21 GPU clock-cap sweep, GLM 5.3 Flash EXL3 TP3
- 2026-09-23 GLM TP3 V3.2 verification
- 2026-09-23 two-node evaluation

Lane card (private input, cited export-safe): the GLM 5.3 Flash EXL3 TP3 card, fields model / weights, spec method and k, slots, chunk size, clock cap, Levers not yet tried, Owner notes.

Community inputs are reached only through the wiki pages above (P05 §4 kernel-rate rows, P03 §4 in-engine NCCL rows, P01 §4 small-kernel row).
