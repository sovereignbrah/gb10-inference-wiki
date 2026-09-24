---
title: The Science of GB10 Inference — chapter 02, the memory budget
created: 2026-09-23
status: checked
type: chapter
extends: core §2 (budget equation, knob table, floors, runbook, worksheet) and core §4.6 (request envelope)
---

# Chapter 02 — The memory budget

Scope. Core §2.1-§2.8 hold the term table, the knob table, the floors, the one-knob rule, the exhaustion mechanism, the runbook and the pre-boot worksheet. This chapter does not repeat them. It derives why the budget has the shape it has, turns each engine knob into an equation for the headroom it leaves, refines the request envelope, works one real lane end to end, and states the decision procedures and traps that follow. Symbols are the core's (`A_0`, `W_rank`, `KV_pool`, `G`, `WS`, `P_warm`, `T_pre`, `ΔH`, `F_warm`, `H_warm`, `c_lane`). Units: GiB = 2^30 B; SGLang "GB" lines are GiB. Pointers `P01`…`P22` are the core's §10.1 key.

---

## 1. Derivation

### 1.1 Reserved, unreserved, drift

Per rank r and time t while serving:

```
A_r(t) = A_0,r − R_r − U_r(t) − ΔH_r(t)
R_r    = W_rank + W_draft + KV_pool + KV_draft + G + WS + B_nccl          (reserved at boot, constant after)
U_r(t) = T_pre(t) + P_warm(t) + allocator growth(t)                        (unreserved transients)
H_warm,r = A_0,r − R_r                                                     (warm-idle headroom)
Serving is safe iff  min_r [ H_warm,r − max_t U_r(t) − ΔH_r ] ≥ F_warm
```

| symbol | meaning | unit |
|---|---|---|
| `A_r(t)` | `MemAvailable` on rank r at time t | GiB |
| `R_r` | bytes the engine allocates at boot and holds for the process lifetime | GiB |
| `U_r(t)` | bytes allocated per request or per warm-up and released (or not) afterwards | GiB |
| `ΔH_r(t)` | host drift since boot (page cache, daemons, builds) | GiB |

[Interpretation, assembled from the P11 §2 accounting and the P11 §4 ladder rows]

Consequences, each testable:
1. Pool occupancy is not a term. The pool is allocated in full at boot, so a 25-28%-full pool and a full one leave the same `A_r` [Interpretation]; the 2026-09-16 wedge at 25-28% occupancy [Historical diagnostic] (P11 §7) is what the equation predicts. A pool cap changes `R_r` once; it never bounds `U_r(t)`.
2. The failing term is always unreserved [Interpretation]. Everything in `R_r` was proven to fit when boot finished; what can exhaust the node later is `U_r(t)` or `ΔH_r`.
3. The tightest rank governs. TP pools and weights are near-symmetric; `A_0,r` is not. Anchors: campaign minima 11.6 / 13.2 / 15.2 GiB on the three ranks of one lane [Measured] (2026-09-14 GLM 5.3 EXL3 TP3 triangle qualification); 97.96 vs 110.93 GiB available on a rank carrying `vm.min_free_kbytes` 4 GiB + `watermark_scale_factor` 100 against ranks at 45,155 kB / 10 [Measured] (2026-09-13 DS V4.1 Flash TP3 qualification; P17 §4); a cross-rank comparison, not a before/after, so other host differences are not excluded. The head usually carries extra host processes [Interpretation].
4. Two design choices exist for a transient: reserve its worst case at boot (moves it into `R`, costs pool, cannot wedge) or allocate on demand (keeps pool, can wedge). Stock vLLM's GLM indexer workspace (5,036 MiB at 1M, locked) is the first [Code-verified]; SGLang's indexer logits are the second [Code-verified] (P11 §2). A bounded workspace is safe only if its bound is a true maximum [Interpretation]; a workspace sized and locked at warm-up under MNBT 2048 later asserted on a warm-prefix turn (required 1,949.52 MB, held 1,157.00 MB) [Community-measured] (P11 §7).

### 1.2 Which terms each knob reserves

| term | vLLM `gpu_memory_utilization` | vLLM `--kv-cache-memory-bytes` | SGLang `mem_fraction_static` (+ `--max-total-tokens`) | ds4 |
|---|---|---|---|---|
| `W_rank`, `W_draft` | measured at load | measured | measured (`Load weight end avail mem=`) | resident model |
| `KV_pool` | residual | fixed = pin | residual `rest`, or `min(rest, cap)` | planned; managed when ≥ 8 GiB |
| `G` | subtracted as `G_est` | profiled, not enforced | not reserved: allocated after the pool, out of headroom | — |
| `T_pre`, chunk-linear part | profiled at MNBT (reserved) | profiled, not enforced | not reserved | `--prefill-chunk` sizes it |
| `T_pre`, chunk × prefix part | partly: profiled peak grows with `max_model_len` | not enforced | not reserved | — |
| `P_warm` (multimodal) | not budgeted when `--skip-mm-profiling` is set (required on the own lanes) | not budgeted | `mm_reservation` | — |
| built-in margin | `(1 − u) · Total` minus host | none | `(1 − f) · A_before` | 8 GiB in the expert-cache budget |

[Code-verified] for the accounting (P11 §2), except the growth of vLLM's profiled peak with `max_model_len`, which is [Community-measured] (P11 §2); [Measured] for vLLM's unbudgeted vision warm-up, ~4-5 GiB (2026-08-28 GLM 5.3 Flash EXL3 TP2 500K stabilization) and for SGLang graphs landing after the pool (the 5.66 → 3.78 GiB step in 2026-09-16 DS V4.1 TP3 memory-ceiling record).

### 1.3 Each knob as a headroom equation

vLLM, utilization `u`:
```
H_warm ≈ (1 − u) · Total − H_host − U_unprofiled          ∂H_warm/∂u = −Total ≈ −1.2 GiB per 0.01
```
`Total` = CUDA-visible total (GiB); `H_host` = memory used outside engine accounting (GiB); `U_unprofiled` = allocations vLLM did not see in `profile_run` that are still held at warm idle, e.g. allocator residue after warm-up (GiB); per-request transients stay in `U` (1.1). Check [Interpretation on Measured]: u = 0.84, Total 121.63, `H_host` ~12 GiB → 7.46 GiB predicted; ~7.2 GiB steady available measured (2026-08-28 GLM 5.3 Flash EXL3 TP2 500K stabilization). The prediction holds only if `G_est` is right; vLLM's pre-capture estimator went negative on GB10 (−35.69 GiB, −20.50 GiB) and oversized the pool [Community-measured] (P01 §7).

vLLM, explicit pin `K`:
```
H_warm = A_0 − nonKV_profiled − K − G           (no check that H_warm − P_warm ≥ 0)
```
`nonKV_profiled` = the free-memory drop from engine creation to the end of profiling, weights included, plus (torch peak − torch allocated) (GiB) [Code-verified] (P11 §2); `K` = the pin (GiB); `G` = graph pools (GiB). The equation is [Interpretation]. The pin removes the only computed margin. With `P_warm` unbudgeted, any configuration that leaves `H_warm < P_warm` dies at multimodal warm-up, pinned or not: five failed launches on one TP2 lane were pins of 15 and 14 GiB at 900K (kernel OOM at the multimodal warm-up), 12.5 GiB at 800K (KV validation) and one auto-KV launch at utilization 0.87 (kernel OOM at warm-up) [Measured] (2026-08-28 GLM 5.3 Flash EXL3 TP2 500K stabilization; P11 §4 labels all five as pins).

SGLang, fraction `f`, cap bytes `B_cap` (pool bytes at `--max-total-tokens`):
```
rest    = A_after_load − (1 − f) · A_before − mm
KV_pool = min(rest, B_cap)                          boot fails if rest ≤ 0
H_pool  = A_after_load − KV_pool = max( (1 − f) · A_before + mm , A_after_load − B_cap )
H_warm  = H_pool − G − O_post
```
`A_before`, `A_after_load` = the `Load weight begin` / `end` avail lines (GiB); `mm` = multimodal reservation (0 for text-only) (GiB); the DS V4.1 fork the own lane ran subtracts a fixed 0.1 GB in its place (P11 §2); `O_post` = everything allocated after graphs until warm idle (workspaces, autotune, NCCL) (GiB). [Code-verified] form (P11 §2); `H_pool` and its two regimes are [Interpretation].

Two regimes:
- Fraction binds (`rest < B_cap`): `∂H_warm/∂f = −A_before` (≈ −1.09 GiB per 0.01 at `A_before` 108.9). Lowering `f` raises headroom until `rest` ≤ 0.
- Cap binds (`rest ≥ B_cap`): `∂H_warm/∂f = 0`. `f` is then only a boot-feasibility gate; headroom moves with `B_cap` (and weights) alone.

Check [Interpretation on Measured]: `A_after_load` 7.41, pool 1.51-1.59 GiB (749,824 × 1,670.75 B = 1.17 GiB + 0.34 GiB SWA; P11 records 1.59) → `H_pool` 5.82-5.90 predicted; 5.66 logged (2026-09-16 DS V4.1 TP3 memory-ceiling record).

ds4:
```
expert_cache = (MemAvailable − CmaFree) − 8 GiB          KV → managed when ≥ 8 GiB or crowding
```
[Code-verified] (P11 §2). The 8 GiB is a built-in floor, not headroom the operator can spend [Interpretation].

### 1.4 The prefill working set

```
T_pre(M, P) ≈ c · M · P + d · M + e
```

| term | meaning | unit | anchors |
|---|---|---|---|
| `M` | tokens in the prefill step (chunk, MNBT or LPT grant) | tokens | logged scheduler config |
| `P` | prefix already in context at that step | tokens | — |
| `c` | bytes held per (step token × prefix token) across live buffers: indexer score blocks, sparse top-k scratch | B | see contested table below |
| `d` | bytes per scheduled token independent of prefix: activations, MoE scratch | B/token | ≈ 0.51-0.65 MiB/token, vLLM DS V4 Flash, one node: 3.91 GiB over 6,176 tokens (MNBT 8224 → 2048 at 262K, profiled peak 5.30 → 1.39 GiB) gives 0.65 if the length term below is a fixed buffer, 0.51 if it scales as `c · M · P` (c ≈ 0.56 B accounts for 0.84 GiB) [Community-measured inputs; Interpretation] (P11 §2) |
| `e` | step intercept | GiB | ≈ 0.1 GiB from the same two points [Interpretation] |

[Interpretation]: the `c` term exists because an indexer or dense-logit path scores every query row in the step against every prior key; the core's `T_pre ≈ c_lane · chunk · P` is this equation with `d = e = 0`.

`c` by stack (never port between stacks):

| stack | c (B) | grade | source |
|---|---|---|---|
| SGLang DS V4 low-ratio indexer, analysis note | ≈ 14 | [Community-measured, unpublished constants] | P11 §2 |
| Same lane class, back-solved from an upstream recipe (4.5-5.3 GB post-boot headroom, 1.5 GB guard, 208K pass, 256K trip at chunk 1024) | ≈ 11-19 (11.2-17.4 if the headroom is GB, 12.0-18.7 if GiB) | [Interpretation on Community-measured] | 2026-09-16 DS V4.1 TP3 incident versus upstream recipe audit |
| Own DS V4.1 SGLang TP3 head, back-solved (§2 step 2) | ≈ 8.7-12.0 (a lower bound: 10 s sampling) | [Interpretation on Measured] | 2026-09-16 audit; 2026-09-15 k5 scorecard |
| SGLang GLM kpool | ≈ 1 | [Code-verified] | core §4.6; P11 §2 |
| GLM-5.2 | ≈ 9.32 | [Community-measured] | P04 §3 |
| vLLM DS V4 Flash, profiled length term (1.39 → 2.23 GiB, 262K → 1M, MNBT 2048) | ≈ 0.56 | [Interpretation on Community-measured] | P11 §2 |

Deciding test for a lane's `c` and `d` [Proposed]: one boot, chunk fixed, cold unique prompts at P ∈ {32K, 64K, 131K} with 1 s per-rank sampling; slope of the minimum `MemAvailable` against P gives `c · M`; a second boot at M/2 separates `d`. Stop the ladder at the traffic-stop floor.

Refined envelope and the largest safe chunk:
```
SGLang (T_pre unreserved):   P_max,safe = (H_warm − F_warm − d·M − e) / (c · M)
vLLM   (d·M profiled):       P_max,safe = (H_warm − F_warm − e) / (c · M)
M*  = (H_warm − F_warm − e) / (c · P_adv + d)       largest chunk that keeps the advertised context P_adv safe (SGLang form)
```
[Interpretation]. Two corollaries:
- Iso-transient rule: the `c` term depends on `M · P` only. Chunk 512 at 262K and chunk 1024 at 131K have the same `c` term; they differ only in `d · M` and in prefill rate.
- If `H_warm ≤ F_warm`, `M*` < 0: no chunk makes the lane safe. Chunk and context only move where it fails.

### 1.5 The floors as margins

```
F_warm ≥ F_onset + ΔH_max + σ_boot
```

| term | meaning | evidence | grade |
|---|---|---|---|
| `F_onset` | `MemAvailable` at which driver allocations start failing (the driver needs free pages, not reclaimable ones) | refusals near `MemFree` ~3 GiB, MemFree gate applied while `MemAvailable` < 10 GiB [Community-measured] (P11 §3); own head, chunk 1024, 10 s sampling: 6 NV_ERR in a 32K cold prefill at a sampled minimum of 2,154 MiB (2.10 GiB) on one boot; on the next boot 22 NV_ERR fell inside the 32K cold-prefill window by raw journal timestamps while samples read 2.2-2.8 GiB, and 82 inside the 131K window down to 1,359 MiB (1.33 GiB) [Measured] (2026-09-15 k3 and k5 scorecards, raw) | onset ≈ 2-3 GiB `MemAvailable` [Interpretation] |
| `ΔH_max` | host drift over a serving period | node-1 4.7 → 3-4 GiB (0.7-1.7), node-2 6.2 → 3-4 GiB (2.2-3.2), six hours after a 1M activation [Measured] (2026-08-30 GLM TP2 composite 1M activation) | lane-specific |
| `σ_boot` | boot-to-boot spread of `H_warm` at identical knobs | ±0.6 GiB on one TP2 lane; 6.60 vs 4.19 / 4.07 GiB after KV on one TP3 lane, with swap state differing [Measured] (P11 §4, §6) | lane-specific |

Arithmetic [Interpretation]: 2-3 + 0.7-3.2 ≈ 2.7-6.2 GiB before `σ_boot`, which is why the fleet's 5 GiB warm floor is a minimum, not a target; a lane with high drift can need more. The 4 GiB traffic stop sits between `F_onset` and `F_warm`: it halts load while drift can still be absorbed. The 8 GiB pre-start floor matches the 8 GiB reserves in ds4 and in an open vLLM cap PR [Code-verified; Proposed] (P11 §2); its derivation from startup-burst depth is open (Gaps).

---

## 2. Worked example: DS V4.1 Flash, SGLang fork, TP3/EP3, head node-3

Question: which restart configuration can serve a 262K-token context within the floors? (The lane has been down since the 2026-09-16 wedge.)

Inputs, head rank (tightest):

| input | value | grade | source |
|---|---|---|---|
| `A_before` | 108.90 GiB | [Measured] | 2026-09-16 DS V4.1 TP3 memory-ceiling record |
| `A_after_load` (weights + draft) | 7.41 GiB; draft weights 2.41 GiB | [Measured] | same |
| `f`, cap | 0.95; 749,824 tokens at 1,670.75 B/token + 0.34 GiB SWA | [Measured] | same; P11 §6 |
| `H_warm` | 2.41 GiB first post-start sample; 2.49 end of scorecard; 2.83 post-run | [Measured] | 2026-09-16 DS V4.1 TP3 incident versus upstream recipe audit |
| 131K cold prefill (130,797 tokens), chunk 1024 | head 10 s-sampled minimum 1,359 MiB = 1.33 GiB (P11 and core print 1.36); 104 serving NV_ERR on this boot, 22 inside the 32K cold-prefill window and 82 inside the 131K window by raw journal timestamps (the packet text puts all 104 at 131K); 32K on the previous boot: 6 NV_ERR at a 2.10 GiB sampled minimum | [Measured] | 2026-09-15 DS V4.1 Flash TP3 k3 and k5 scorecards |
| cold prefill rate | 1,976 / 1,924 / 1,044 tok/s at 8K / 32K / 131K | [Measured] | same |
| chunk 256 profile | 8-32K prefill ~2.0k → ~1.0k tok/s | [Measured] | 2026-09-14 DS V4.1 TP3 optimization |

Step 1 — regime. `rest` = 7.41 − 0.05 × 108.90 = 1.97 GiB (1.87 with the fork's 0.1 GB term) ≥ `B_cap` 1.51-1.59, so the cap binds and `∂H_warm/∂f` = 0 (1.3). Lowering `f` to 0.932 would leave `rest` 0 and fail boot; raising it frees nothing. The fraction is not a headroom lever on this lane [Interpretation].

Step 2 — `c` for this lane. `T_pre`(1024, 130,797) = `H_warm` − minimum = (2.41 to 2.83) − 1.33 = 1.08-1.50 GiB → `c` = 1.16-1.61 × 10^9 B / (1,024 × 130,797) ≈ 8.7-12.0 B [Interpretation on Measured]. `d · M` is folded into this figure (not separable from these records). Samples are 10 s apart and moved by up to 1.4 GiB between consecutive samples during the prefill (1,359 → 2,785 MiB), so the true minimum is at or below 1.33 GiB and the range is a lower bound on `c`.

Step 3 — floor check before any prefill. `H_warm` 2.41-2.83 < `F_warm` 5 GiB, so `P_max,safe` < 0 at every chunk (1.4). The warm floor is already breached at idle.

Step 4 — predict the restart options (c = 8.7-14 B, `H_warm` 2.41-2.83 GiB) [Interpretation]:

| option | `c·M·P` at the advertised context | predicted head minimum | predicted NV_ERR | verdict |
|---|---|---|---|---|
| chunk 1024, context 131K | 1.09-1.75 GiB | 0.66-1.74 GiB | yes; measured 82 in the 131K window at this exact point | fails zero-NV_ERR gate |
| chunk 512, context 262K | 1.09-1.75 GiB (iso-transient with the row above) | 0.66-1.74 GiB, plus a smaller `d·M` | yes | fails |
| chunk 256, context 262K | 0.54-0.88 GiB | 1.53-2.29 GiB | contested: 32K at chunk 1024 (0.27-0.44 GiB of `c` term) logged NV_ERR on both boots (6; 22), but a chunk-256 boot with a 2Mi pool passed a 519,712-token cold prompt (2× this `M · P`) at a 2.02 GiB head minimum with startup-only NV_ERR [Measured] (2026-09-14 DS V4.1 TP3 optimization; Gap 5) | fails the 5 GiB warm floor either way; prefill ~halved |
| chunk 1024, context 200K (the incident) | 1.66-2.67 GiB | ≤ 1.17 GiB, down to below zero | yes | wedged [Historical diagnostic] |

Step 5 — what freeing is available, one variable per boot [Interpretation; costs as graded]:

| lever | bytes freed on the head | cost | source |
|---|---|---|---|
| speculation off | ≥ 2.41 GiB (draft weights; drafter buffers unmeasured) | k=5 decode gain lost (no spec-off arm exists on this lane) | memory-ceiling record; P02 |
| pool cap 749,824 → 262,144 tokens | 0.76 GiB (1.17 → 0.41 GiB) | sum of live contexts ≤ 262K across 4 slots | P11 §6 bytes/token |
| lower `f` | 0 (cap binds) | boot fails below ~0.932 | Step 1 |
| NCCL small buffers | 0 (already applied on this lane) | — | P11 §5 lever 7 |
| 2.9 bpw EXL3 port, same three nodes | ~36 GiB/rank (64.6 vs ~101 GiB weights); measured minimum available 26.1 GiB at chunk 8192 | separate engine and quant; long-context unqualified | [Measured] 2026-09-16 DS V4.1 EXL3 TP3 optimized-1M; P20 §4.3 |
| fourth node, TP4 | headroom ~3 → ~40 GB/rank at idle (native ≈77 GiB/rank) | hardware | [Community-measured] (P20 §4.3) |

In-lane best case: spec off + cap → `H_warm` 5.58-6.00 GiB, margin 0.58-1.00 GiB over `F_warm`. At chunk 256 and 262K the `c` term needs 0.54-0.88 GiB: inside the envelope for `c` ≤ 9.3 B at the low end of `H_warm` and `c` ≤ 16 B at the high end, before `d · M`. At chunk 512 it needs 1.09-1.75 GiB: outside.

Output of the procedure [Interpretation; decision is the operator's]:
- The restart gate written after the incident allows chunk 256 or 512 at 262K, or context 131K at chunk 1024. The chunk-512 option is iso-transient with a measured failure and the 131K-at-1024 option is that failure (82 NV_ERR in its window); both fail their own zero-NV_ERR gate. The chunk-256 option fails the 5 GiB warm floor at idle (Step 3), and its NV_ERR outcome is contested (Step 4).
- Within this engine, format and node count, the only path to a 262K context within the floor is three boots (spec off; cap; chunk 256), with the envelope met marginally and prefill halved.
- The EXL3 port (26.1 GiB minimum over short-context tests) and TP4 (~40 GB free at idle, community) clear the floor by more than 20 GiB where measured; neither is qualified at 262K on the own fleet.
- Falsifier: a chunk-512, 262K boot showing zero NV_ERR and a head minimum ≥ 2 GiB through a cold 262K prompt refutes the `c` range and the onset estimate; record it as a prediction error.

Contrast on the same nodes [Measured]: the vLLM GLM 5.3 Flash EXL3 TP3 lane (weights 54.25 GiB/rank, 32 GiB/rank KV pin) ran 15 gates, 8K-1M retrievals and an 84-request soak with campaign minima 11.6 / 13.2 / 15.2 GiB per rank and 0 NV_ERR (2026-09-14 GLM 5.3 EXL3 TP3 triangle qualification); its graphs are 2.34 GiB (2026-09-17 GLM TP3 boot sample). Its weights are 46.75 GiB/rank lighter, which pays for a 32 GiB pool and still leaves ≥ 11.6 GiB: the difference is weight bytes, not tuning [Interpretation].

---

## 3. Decision procedures

### 3.1 Choosing the memory knob

1. Identify the engine's residual (1.2). The knob that sets the residual moves headroom; the others do not.
2. SGLang: compute `rest` and `B_cap` from the last boot receipt. If the cap binds, move the cap or the weights, never `f`. If `f` binds, each 0.01 moves `A_before` / 100 GiB, and `f_min = 1 − A_after_load / A_before` is a hard boot floor.
3. vLLM: prefer utilization with auto KV; predict `H_warm` with 1.3 and compare with the boot receipt. Pin only after arm A of the core 2.3 test, and only if `H_warm − P_warm ≥ F_warm` with `P_warm` measured.
4. Any engine: if the freed bytes you need exceed every in-lane lever (as in §2), change format, drafter or node count; do not iterate chunk or context.

### 3.2 Chunk and advertised context

1. Read `H_warm` on the tightest rank from a warm, idle, post-warm-up sample of this boot.
2. If `H_warm ≤ F_warm`: stop; go to 3.1 step 4.
3. Read `c` and `d` from the lane card; if absent, run the 1.4 deciding test (or back-solve from one measured cold prefill minimum, as §2 step 2, and label it [Interpretation]).
4. Compute `M*` for the advertised context; take the largest rung ≤ `M*` that is page-aligned and passes the core §4.6 rows-per-expert and ladder checks.
5. If the rung costs more prefill than the operator accepts, lower the advertised context instead: `P_adv = P_max,safe(M)`.
6. Client timeout: require `timeout > t_0 + P_adv / R_prefill(P_adv)`. At 1,044 tok/s measured at 131K, a 200K cold prompt needs ≥ 192 s [Interpretation on Measured], so a 60 s client timeout turns such a turn into repeated sends [Interpretation]. In the wedge the ~200K prefix was mostly cached; chunks at that depth slowed to 2-3 min each under memory pressure, and the client resent 8 times on 60 s timeouts [Historical diagnostic] (P22 §4).
7. Gate: cold unique prompt at `P_adv`, 1 s sampling, zero NV_ERR, ≥ `F_warm` on every rank; the advertised context never exceeds the gated length.

### 3.3 Memory-change discipline, quantified

```
s_max   = min( 1.5 GiB , H_warm,prev − F_warm − σ_boot )         allowed pin raise this boot
attrib  : |ΔH_obs| > σ_boot  else repeat the boot before attributing
Δt_samp ≤ min( 2 s , M / R_prefill(P_max) )                       sampling interval
```
[Interpretation]; the 1.5 GiB cap and the 1-2 s interval are fleet rules (core §2.4; index). At chunk 1024 and ~1,000 tok/s, `M / R` ≈ 1 s.

Why freeing and spending are two boots, in numbers [Interpretation on Measured]: the 2026-09-03 change combined an indexer right-size with a +4.0 GiB pin at 4.4 GiB host free (P11 §7 does not say whether this was `MemAvailable` or `MemFree`). For the floor to hold, the right-size had to free ≥ 4.0 + 5 − 4.4 = 4.6 GiB, against a stock workspace of at most 4.92 GiB (5,036 MiB at 1M) [Code-verified]. The plan needed ≥ 93.5% recovery of the largest possible workspace with zero margin for `σ_boot`. The node was unreachable ~7 h (P11 §7; duration WR).

### 3.4 Incident triage (before core §2.7)

| observation | class | first action | grade / source |
|---|---|---|---|
| NV_ERR in `journalctl -k`, node answers SSH, `MemAvailable` < 4 GiB | pre-wedge | stop admissions and client retries; keep the engine; plan one change that shrinks `U` | fleet rule; P11 §6 |
| ping and TCP 22 answer, banner or login stalls | NVRM exhaustion wedge | core §2.7 steps 2-6 | [Community-measured; Measured] P11 §3 |
| tokens frozen, requests held, GPU power well below a healthy long prefill | livelock or collective hang, not an overrun | capture logs, remove every rank of the group; no memory change indicated | [Community-measured] P11 §6 |
| node powered off rather than frozen, with no serving NV_ERR and no falling `MemAvailable` before it | power or thermal event | clock and thermal checks (P13); not a memory fix. A shutdown after a memory wedge stays a memory incident: in the 2026-09-16 wedge the operator reported the head wedged, then shut down and rebooted | [Community-measured] P11 §3; [Historical diagnostic] (WR) |
| one rank dead, peers "Up" holding memory | partial group death | remove every rank before relaunch | [Measured] P11 §7 |
| `/health` 200, no progress counters moving | not liveness | judge by prompt and generation counters | [Community-measured] P11 §3 |

---

## 4. Traps

1. Treating SGLang `f` as the headroom knob when the cap binds; it changes nothing but boot feasibility (1.3).
2. Choosing between restart options that have the same `M · P`; they carry the same indexer transient (1.4).
3. Reading `H_warm` at boot or before multimodal warm-up; `P_warm` and the first long prefill have not happened yet.
4. Comparing `A_0` across ranks or boots with different `vm.min_free_kbytes` / `watermark_scale_factor` or swap state: a 13 GiB cross-rank gap coincided with a 4 GiB `min_free_kbytes` + `watermark_scale_factor` 100 on one rank [Measured], consistent with the 11-15 GiB formula estimate [Interpretation]; a 2.4-2.5 GiB boot gap coincided with ~5 GiB of old pages on swap, cause open (P11 §8) (1.1, 1.5). Raising watermarks lowers reported `MemAvailable` ~11-15 GiB for the same physical state [Interpretation] (P11 §5 lever 8); re-derive every floor after such a change.
5. Budgeting the lane from its best rank; the head, with extra host processes, is usually the tightest [Interpretation].
6. Trusting vLLM's `G_est` on GB10; negative estimates oversize the pool [Community-measured] (P01 §7).
7. Porting `c` between stacks: 0.56-19 B across six anchors (1.4).
8. Treating a gated pass with any NV_ERR as headroom; on the §2 head, driver errors fired during 32K prefills while 10 s samples read 2.1-2.8 GiB, well above zero [Measured] (1.5).
9. Lowering MNBT to win pool on a build whose workspace is locked after warm-up; it boots, then asserts on a warm prefix [Community-measured] (P11 §7).
10. Assuming a right-sized workspace is bounded; its bound must be a true maximum for every request shape [Interpretation] (1.1 point 4).
11. Iterating chunk or context on a lane whose `H_warm ≤ F_warm`; the envelope is empty (§2 step 3).

---

## 5. Evidence links

- [Memory on unified memory](../11-memory-on-unified-memory.md): §2 accounting per engine, §3 GB10 differences, §4 measured rows, §5 levers, §6 protocol, §7 anti-patterns.
- [Prefill optimization](../04-prefill-optimization.md): §3 transient constants, §4 chunk ladders, §7 envelope incident.
- [CUDA graphs and launch overhead](../01-cuda-graphs-and-launch-overhead.md): §6-§7 graph memory, post-capture rule, estimator failures.
- [KV cache and prefix caching](../10-kv-cache-and-prefix-caching.md): §3 bytes per token, logical vs shareable pool.
- [Model architecture cards](../20-model-architecture-cards.md): §4.3 weights per rank, TP fit, TP4 headroom.
- [Speculative decoding](../02-speculative-decoding.md): drafter cost against its decode gain (the spec-off lever in §2).
- [Serving agent workloads](../22-serving-agent-workloads.md): §3-§4 per-request working set, retries.
- [SM121 hardware facts](../17-sm121-hardware-facts.md) and [Clocks, thermal, power](../13-clocks-thermal-power.md): pinned allocations, power-off vs freeze.
- [GB10 Inference Wiki](../README.md): memory rules.
- Lane card fields: model / weights; KV pin; mem fraction; chunk size; context; slots; spec method and k; host; Last incident. Fields this chapter reads that the template lacks: `H_warm` per rank, `c_lane`, `d_lane`, `σ_boot`, sysctl state.

## Gaps

1. `c` and `d` are not separated on any own lane; the §2 back-solve folds `d · M` into `c` [Proposed: needs measurement] (1.4 deciding test).
2. `F_onset` is inferred from two own data points and one maintainer's MemFree figure; no own 100 ms `MemFree` trace across an NV_ERR onset exists [Proposed: needs measurement].
3. The 8 GiB pre-start floor has no derivation from startup-burst depth (72-172 startup NV_ERR per rank on own lanes, cause open, P11 §8).
4. Drafter buffer bytes beyond draft weights are unmeasured, so "speculation off" in §2 frees an unknown amount above 2.41 GiB.
5. At equal `M · P` (~1.33 × 10^8 token²), the chunk-256 profile left a 2.02 GiB head minimum while the chunk-1024 profile fell to 1.33 GiB; pool (2Mi vs 750K tokens), slots, context and boot state all differed, so the gap cannot be assigned to `d · M`.
6. P11 records the 750K pool as 1.59 GiB. 749,824 × 1,670.75 B = 1.253 × 10^9 B, so P11's "about 1.25" is decimal GB and 1.25 + 0.34 adds GB to GiB; in GiB the pool is 1.17 + 0.34 = 1.51 [Interpretation]. The ladder's 1.75 GiB step (7.41 → 5.66) exceeds either figure (core ladder rule).
7. Whether the eight resends in the 2026-09-16 wedge overlapped (stacking transients) or serialized (extending exposure) is not recorded.
8. Upstream post-boot headroom (4.5-5.3 "GB") has ambiguous units, which widens the upstream `c` back-solve to 11-19 B.
9. The `O_post` term (5.66 → ~2.9 GiB on the §2 head, of which 0.52 GiB is graphs) is not itemized.
10. On the 131K scorecard boot, raw journal timestamps put 22 of the 104 serving NV_ERR inside the 32K cold-prefill window and 82 inside the 131K window; the packet text attributes all 104 to 131K and reports none at 32K. Core §2.6 and P11 §4 carry the packet text.
11. Every DS V4.1 minimum used here is a 10 s sample; no 1 s trace of a 131K or 200K prefill exists on this lane.

## Sources

Wiki pages (read 2026-09-23): P11 in full; P01 §6-§7 (by grep); P04 section index; index memory rules; core.md §1-§10 and its check report. Added at check: P04 §3, §7; P17 §4; P20 §4.3; P22 §4.

Own packets (date and title): 2026-08-28 GLM 5.3 Flash EXL3 TP2 500K stabilization; 2026-08-30 GLM TP2 composite 1M activation; 2026-09-03 GLM pair overnight wedge record (via P11 §7); 2026-09-13 DS V4.1 Flash TP3 qualification; 2026-09-14 DS V4.1 TP3 optimization; 2026-09-14 GLM 5.3 EXL3 TP3 triangle qualification; 2026-09-15 DS V4.1 Flash TP3 k3 and k5 scorecards; 2026-09-16 DS V4.1 TP3 memory-ceiling and head-wedge records; 2026-09-16 DS V4.1 TP3 incident versus upstream recipe audit; 2026-09-16 DS V4.1 EXL3 TP3 optimized-1M; 2026-09-17 GLM TP3 boot sample (via P11 §4).

Lane cards (private inputs, used for the worked example only): the DS V4.1 SGLang TP3 card and the GLM 5.3 Flash EXL3 TP3 card, dated 2026-09-23. Workflow review (WR): 2026-09-23 requirements synthesis, gaps #1, #4, #5, #6.
