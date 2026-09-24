---
title: The Science of GB10 Inference — chapter 05, multi-node = a per-step tax
created: 2026-09-23
status: checked
type: chapter
---

# Chapter 05 — Multi-node = a per-step tax

Extends core §5. Core §5.1 has the tax anchors, §5.2 the width table and procedure, §5.3 the preflight check table; they are not repeated here. This chapter derives them, works one lane, and adds the decision rules. Pointer keys follow core §10.1; the pages used here:

| key | page |
|---|---|
| P01 | [CUDA graphs and launch overhead](../01-cuda-graphs-and-launch-overhead.md) |
| P02 | [Speculative decoding](../02-speculative-decoding.md) |
| P03 | [Inter-node communication over DAC](../03-inter-spark-communication.md) |
| P05 | [Decode optimization: the bytes model](../05-decode-optimization.md) |
| P08 | [MoE dispatch](../08-moe-dispatch.md) |
| P11 | [Memory on unified memory](../11-memory-on-unified-memory.md) |
| P12 | [Sampling and the lm_head](../12-sampling-and-lm-head.md) |
| P13 | [Clocks, thermal, power](../13-clocks-thermal-power.md) |
| P14 | [Profiling protocol](../14-profiling-protocol.md) |
| P20 | [Model architecture cards](../20-model-architecture-cards.md) |

Units: KiB/MiB/GiB binary; GB decimal; `busbw` is the nccl-tests normalisation (per-rank ring bytes ÷ time), not wire rate.

---

## 1. Derivation

### 1.1 Collectives per step

```
N_coll = N_fwd + N_draft
N_fwd  = Σ_layers c_l + g            (target forward, one engine step)
c_l    = 2   TP, row-parallel attention output + row-parallel MLP/MoE output (one all-reduce each)
       = 4   sequence parallelism forced (each all-reduce → reduce-scatter + all-gather)
       = 3   MoE layer under all-to-all EP (attention all-reduce + dispatch + combine)
g      = 1   full-vocabulary logits all-gather (vLLM default, use_all_gather)
N_draft = collectives issued by a TP-sharded drafter per step (0 if replicated or on one rank)
```
[Code-verified for the GLM decomposition; general form Interpretation] (P03 §2 vLLM and ExLlamaV3 bullets; P03 §2 SP bullet)

| term | meaning | unit |
|---|---|---|
| `N_coll` | collectives on the critical path of one decode step | count |
| `c_l` | collectives layer l issues | count |
| `g` | logits gathers per forward | count |
| `N_draft` | drafter collectives per step; read from source or a per-rank shape dump, never from a recipe comment | count |

Why `N_coll` does not fall with width: each row-parallel matmul leaves partial sums on every rank, so every layer needs its reductions at any TP ≥ 2; width changes bytes per collective, not count [Interpretation] (P03 §1). Community counts 88 (TP4) to 108 (TP3) agree [Community-measured] (P03 §4).

GLM 5.3 Flash EXL3 TP3/EP3 check (P03 §2 ExLlamaV3; P20 §4.1):

| component | count | note |
|---|---|---|
| attention output all-reduce | 45 | 34 KDA + 11 sparse-MLA layers |
| dense-MLP all-reduce | 3 | layers 0-2 |
| MoE all-reduce | 42 | EP3 under EXL3: each rank runs its 96 experts on the whole batch; routed + shared partials summed in one all-reduce; no all-to-all |
| lm_head all-gather | 1 | vocabulary padded 154,880 → 154,944 at TP3 |
| `N_fwd` | 91 | = 2·45 + 1 |
| drafter (DFlash2, TP-sharded on every rank) | ≈11 | the lane registry's single-rank drafter setting is a no-op at the pinned fork |
| `N_coll` | ≈102 | [Code-verified] (2026-09-20 GLM 5.3 Flash TP3 frontier dossier v2) |

### 1.2 Time per collective and the LL floor

```
t_coll(n, S) ≈ t_base + h(n) · t_hop + β_p · R(n, S) / BW_link + t_skew
h(n) = 2(n − 1)            Ring all-reduce hops
h(n) = 2 · ⌊log2 n⌋        Tree all-reduce hops
R(n, S) = 2(n − 1)/n · S   ring bytes per rank, all-reduce
R(n, S) = (n − 1)/n · S    ring bytes per rank, all-gather
S = M · H · b              payload: M rows, H hidden, b bytes per element
```
[Interpretation, from the NCCL latency model at `v2.30.7-1`] (P03 §2 "Latency model", "Protocols")

| term | meaning | unit | GB10 value |
|---|---|---|---|
| `n` | ranks (nodes; one GPU each) | — | 2, 3 |
| `t_base` | per-algorithm, per-protocol base latency in the NCCL tuner model | µs | LL Ring ≈6.6, with ≈5.0 per hop, implied by the 16.6 / 26.6 µs model floors at n = 2 / 3 [Interpretation from Code-verified model] (P03 §2) |
| `t_hop` | per-hop latency: NIC + proxy thread + host staging (no GPUDirect RDMA) | µs | not isolated; `ib_write_lat` 2-byte floor 2.01 µs [Community-measured] (P03 §4) |
| `β_p` | protocol wire overhead; LL sends 4 B data + 4 B flag per 8 B word | — | LL ≥ 2; literature LL 25-50% of peak bandwidth [Community-measured, not GB10] (P03 §2) |
| `BW_link` | per-hop bandwidth | GB/s | own busbw 12.9 at 32 MiB, one PCIe half per hop [Interpretation on Measured] (P03 §4) |
| `t_skew` | residual: wait for the slowest rank plus effects the standalone floor lacks | µs | 3 nodes: in-engine 125-154 − standalone 72-86 ≈ 40-80 [Interpretation] |

Model vs measurement:

| quantity | 2 nodes | 3 nodes | grade | source |
|---|---|---|---|---|
| NCCL model LL floor, Ring | 16.6 µs | 26.6 µs | [Code-verified] | P03 §2 |
| NCCL model LL floor, Tree | 16.8 µs | 16.8 µs | [Code-verified] | P03 §2 |
| standalone small all-reduce | 40.79 µs at 16 KB (one half) | 72-86 µs flat from 8 B to 64 KiB | [Community-measured] | P03 §4 |
| in-engine per collective, rank waits included | — | ≈125-154 µs | [Community-measured] | P03 §4 |

Consequences [Interpretation]:
1. Measured floors run 2.5-3.2× the Ring model (40.79-43.42 vs 16.6 µs at n = 2; 72-86 vs 26.6 µs at n = 3) and 4.3-5.1× the 3-node Tree model; P03 §2 summarises this as 2-5×. The flat region from 8 B to 64 KiB means the bandwidth term is invisible at decode payloads of 64 KiB or less. The decode tax is hop latency × count, not bytes.
2. Ring on 3 nodes has 4 hops, Tree 2. If `t_hop` dominates, Tree could approach half the Ring floor. Tree has never run on the own triangle (the ring graph file was needed to pass init) [Proposed] (P03 §5 lever 8).
3. The residual `t_skew` is a third to a half of the in-engine cost on 3 nodes. It is a cross-deployment difference (an SGLang DeepSeek V4.1 profile against a standalone sweep on a plugin-patched fabric), so it also carries payload, eager-versus-graph and profiler effects; it has not been decomposed. The part that is true rank skew belongs to the slowest rank: clocks, a latched or hidden-slow rank (P13 §5 lever 2), memory pressure (P03 §6). The one own unprofiled per-rank re-measure found no persistent skew (35.90 / 35.65 / 36.11 ms) [Measured] (P03 §4). Whether skew is the largest reducible share needs an own per-rank timeline (Gap 2).

### 1.3 Step tax and per-token tax

```
T_tax  ≈ N_coll · t_coll(n, S_decode)                        (serial on the critical path: no overlap path on GB10, §3)
τ_tok  = T_tax / a                                             a = committed tokens per step per stream (core §3.5)
T_comm,chunk ≈ N_fwd · R(n, S_chunk) / busbw                  prefill chunk, bandwidth-dominated
```
[Interpretation] (P03 §4 "How to read the TP tax numbers together"; P03 §5 lever 3)

Terms: `T_tax` step tax (ms); `τ_tok` tax per committed token (ms); `S_decode`, `S_chunk` payload of one all-reduce at the decode rows and at the chunk rows (bytes); `busbw` measured bus bandwidth (GB/s); `T_comm,chunk` communication time per prefill chunk (ms).

Payload regimes for H = 4,096, b = 2 (GLM; 90 all-reduces per forward plus one logits gather) [Interpretation from Code-verified shapes] (P03 §2 ExLlamaV3):

| regime | rows M | S per all-reduce | ring bytes/rank per step, n = 3 | regime of `t_coll` |
|---|---|---|---|---|
| C1 decode, k = 7 | 8 | 64 KiB | 7.5 MiB + 1.58 MiB logits gather | flat floor (latency) |
| C8 decode, k = 7 | 64 | 512 KiB | 60 MiB + 12.6 MiB logits gather | past the flat region; bandwidth term appears |
| prefill chunk | 2,048 | 16 MiB | ≈1.9 GiB | bandwidth |

MoE all-reduce dtype, contested: the dossier states that the 42 routed-MoE all-reduces run in fp32 (`exl3.py:1443`) [Code-verified] (2026-09-20 GLM 5.3 Flash TP3 frontier dossier v2, F3.9), while its own ring arithmetic and P03 §2 and P12 §3 use 2 bytes for all 90. If fp32: the C1 MoE payload is 128 KiB (past the measured flat region), ring bytes per step ≈11 MiB at 8 rows and ≈2.75 GiB per 2,048-row chunk; in the prefill check that follows, the TP3 chunk share becomes ≈20% instead of ≈13% and the TP2 share ≈10% instead of 6.8%. The own TP2 profile (7.2%) sits nearer the 2-byte figure [Interpretation]. Deciding test: the MoE all-reduce dtype in a per-rank shape and dtype dump at the pinned fork (Gap 11).

Prefill check against evidence [Interpretation]:
- TP3, 2,048-row chunk: 1.875 GiB ÷ 12.9 GB/s ≈ 156 ms per chunk. A 2,048-token chunk at 1,747 tok/s takes ≈1.17 s, so communication ≈13% of the chunk. Community NCCL share of GLM TP3 prefill: 11.9-13.9% [Community-measured] (P03 §4).
- TP2, 1,024-row chunk: R = S (factor 1 at n = 2), 90 × 8 MiB = 720 MiB ÷ 10.2 GB/s (nccl-tests all-reduce, one half, two nodes [Community-measured], P03 §4) ≈ 74 ms of a 1.086 s step = 6.8%. The own profile read 7.2% with the profiler on [Measured] (P20 §4.1, own 2026-08-29 TP2 profile).
- Both PCIe halves per hop raised community busbw 12.0 → 20.8 GB/s at 64 MB (1.7×, patched plugin) [Community-measured] (P03 §4). Applied here as 12.9 → 20.8 GB/s, the TP3 chunk term falls to ≈97 ms, saving ≈59 ms ≈ 5% of the chunk [Interpretation]. Community triangle result: prefill +5.5%, decode flat [Community-measured] (P03 §5 lever 5).
- Ring bytes per rank scale as 2(n − 1)/n (1.0, 1.33, 1.5 at n = 2, 3, 4) while compute per rank scales as 1/n. The prefill communication share therefore grows with width [Interpretation].

---

## 2. Worked example: GLM 5.3 Flash EXL3, TP3/EP3, node-1 head, node-2 and node-3 workers

Inputs (read from the lane card; values as of 2026-09-23): one PCIe half per hop, per-rank ring graph, `NCCL_ALGO=Ring`, 8 channels; 2,100 MHz cap on all ranks; adaptive k ∈ {2, 4, 7}, so M ∈ {3, 5, 8} at C1; C1 decode 44.3 code / 37.8 prose tok/s; prefill 95K 1,747 tok/s [Measured] (2026-09-23 GLM 5.3 Flash EXL3 TP3 V3.2 fabric and compact-KV verification). Not recorded: `a` on this lane (the per-step instrumentation tuple was never built; lane card: Levers not yet tried #1).

### 2.1 Step tax

`T_tax` = 102 × 130-150 µs = 13.3-15.3 ms, transferring the community in-engine band; lower bound 102 × 72-86 µs = 7.3-8.8 ms [Interpretation]. The own dossier's 4-10 ms band sits below both in-engine profiles and is a lower bound (core §5.1).

Because `a` is unknown, the share is a function of it. `T_step = a / 44.3 tok/s` (code, C1) [Interpretation on Measured]:

| a | T_step | tax share, in-engine band | tax share, floor band | τ_tok (in-engine) |
|---|---|---|---|---|
| 2 | 45.1 ms | 29-34% | 16-19% | 6.6-7.7 ms |
| 3 | 67.7 ms | 20-23% | 11-13% | 4.4-5.1 ms |
| 4 | 90.3 ms | 15-17% | 8-10% | 3.3-3.8 ms |

Reading [Interpretation]: speculation is the largest tax lever on this lane, because `τ_tok` falls as 1/a while the drafter's 11 collectives add about 12% to the 91 of the target forward (11 of the 102 per step). Deciding measurement: a per-rank timeline at forced k = 2 and k = 7 in one boot. NCCL time per step should stay flat while tokens per step rise (P03 §5 lever 3). A community C1 profile of the same model at TP3 (NVFP4 target, 98.9 ms step) put NCCL at 13.8-15.1% [Community-measured] (P20 §4.1).

### 2.2 Width: TP2 versus TP3 on the same model

Measured (2026-09-23 GLM 5.3 Flash EXL3 two-node TP2 evaluation, byte-identical frozen prompts, best safe TP2 arm with cooperative MoE and FP8 on all groups):

| metric | TP3 | TP2 | TP3 ÷ TP2 |
|---|---|---|---|
| C1 code decode | 44.3 | 35.6 tok/s | 1.24 |
| C1 prose decode | 37.8 | 30.7 tok/s | 1.23 |
| prefill 7.6K / 95K | 1,624 / 1,747 | 1,481 / 1,526 tok/s | 1.10 / 1.14 |
| weights per rank (engine load line) | 55.8 GiB (54.25 in the 2026-09-14 qualification, before the V3 large-M KDA BF16 copy) | 81.9 GiB | — |
| KV | 3.94× at 1M, 8 slots | 1.09× at 600K, 4 slots | — |

Confounds [Measured] (packet recipes): different launchers and cooperative-MoE builds for 2 and 3 nodes, EXL3 row settings, slots (4 vs 8), served context (600K vs 1M) and pool size; the TP3 decode and prefill figures come from two arms of one session (V3.2 and V3.2a, which the packet states do not differ on decode). Fabric is matched: one controller per hop in both arms. The kit author's own lab ratio was 0.82-0.84 TP2/TP3 [Community-measured]; own 0.80-0.81.

Apply the width equation (core §5.2) as a difference, assuming the same `a` in both arms [Interpretation]:
```
T_step(TP2) − T_step(TP3) = B_s / BW_eff · (1/2 − 1/3) − N_coll · (t(3) − t(2))
measured left side      = a · (1/35.6 − 1/44.3) s = a · 5.5 ms   (code);  a · 6.1 ms (prose)
```
- Tax term: 102 × (72-86 − 41-43) µs = 3.0-4.6 ms more at TP3, floor band [Interpretation].
- Bytes term: `B_s` = weight bytes read per step, summed over ranks, that sharding divides (GB); `t(n)` = per-collective cost at n nodes (µs). `B_s` per step at M = 3-8. Non-expert weights ≈8.3 GB (8.29B params at FP8, head precision not recorded) plus routed experts. The expert union bound is 23.3-58.1 experts per layer × 42 layers × 12.58 MB (3 × 4,096 × 2,048 at 4 bits) = 12.3-30.7 GB. Together `B_s` ≈ 20.6-39 GB, an upper bound because routing is correlated (core §3.2). Saving at TP3 = `B_s` / (6 × 230-245 GB/s) ≈ 14-28 ms [Interpretation]. Omitted: drafter reads (≈0.72 GiB per rank at TP3 [Interpretation], dossier F3.3), which would add ≈1.6 ms.
- Predicted net: ≈9-25 ms per step in favour of TP3. Measured: 11 ms at a = 2 to 22 ms at a = 4 (code). This is consistent, not proof [Interpretation].
- Verdict carried by the lane (operator ruling): TP2 rejected. Decode −19 to −20%, prefill −9 to −13% [Measured]. Pool: the packet's summary says "about a quarter"; its own table gives 1.09 × 600K ≈ 0.65M against 3.94M tokens, about one sixth [Interpretation on Measured].

Rule the example supports [Interpretation]: for a weight-bound MoE, the bytes saved per rank by one more node (`B_s/6` from 2 → 3) exceed the ≈3-5 ms added tax. For a small-active MoE that fits on one node, the bytes term is small and the tax wins: 35B-A3B TP2 → TP4 fell from 99.0 to 89.2 tok/s [Community-measured] (P03 §4).

### 2.3 Padding on this lane

GLM heads 64 → 66, 22 per rank against 21.33 ideal (+3.1% on the attention critical path). The attention adapter pads again, 22 → 32 heads and 512 → 576 query width on every call, because 22-head prefill crashed [Code-verified]. The drafter pads 32/8 → 36/9 heads (+12.5%). The shared expert pads 2,048 → 2,112 (+3.1%). Vocabulary pads 154,880 → 154,944 (+0.04%) [Code-verified] (P20 §3; P05 §7). The pad cost has never been timed (Gap 4).

---

## 3. What is disabled on GB10, and why

Each fusion or hiding path needs something GB10 lacks. Every cross-node collective is a plain NCCL call over RoCE [Code-verified] (P03 §2, §3, §7); with no overlap path it sits serially on the critical path [Interpretation].

| path | would remove | requirement GB10 lacks | if forced |
|---|---|---|---|
| all-reduce + RMSNorm fusion (vLLM) | a norm pass and a launch per collective | size table keyed on capability 9.0/10.x; FlashInfer world sizes {2, 4, 8, 16} exclude TP3; multi-node needs the `mnnvl` backend (NVSwitch multicast) | explicit size bypasses the table, then fails on world size or backend |
| SGLang all-reduce fusion | same | SM90/SM100 only; SM12x enable PR closed unmerged | graph-capture crashes on SM120 prompted the restriction |
| custom all-reduce (both engines) | NCCL latency | one node or NVLink multicast | disabled across nodes |
| torch / NCCL symmetric memory, NVLS | lower latency | multicast pointer | unavailable |
| async TP (GEMM-collective overlap) | serial comm | symmetric memory, so multicast; SP as prerequisite | unavailable |
| sequence parallelism | nothing alone ("does not directly improve performance") | thresholds for capability 90/100 only | `c_l` 2 → 4: GLM `N_fwd` 91 → ≈181 at flat bytes, so ≈+6-14 ms per step at the 3-node floor-to-in-engine band [Interpretation] |
| GPUDirect RDMA | host staging | peer-memory, dma-buf, GDRCopy all fail; init log `GDR 0` [Measured] | — |
| DeepEP internode all-to-all | EP dispatch over RDMA | GDR + IBGDA | no GB10 run of any internode all-to-all backend was found |
| LL128 | mid-size efficiency | inter-node path type ≤ PXN; a `typeinter="PHB"` graph already excludes it; NCCL warns of corruption on unsupported platforms | correctness gate before any speed arm |

Transfer rule [Interpretation]: an overlap or fusion speedup published on NVLink hardware does not transfer. TokenWeave reports ~20% TP overhead even on NVLink; an H100 fusion win of 5.49 → 9.41 req/s is NVLink-only (P03 §2, §7).

---

## 4. What reduces the tax

Ranked by which term of §1 each lever attacks and by evidence grade.

| # | lever | term | evidence | cost / risk |
|---|---|---|---|---|
| 1 | Raise `a` (depth, drafter, content) | τ_tok = T_tax / a | cost fixed per engine step [Community-measured quote; Interpretation]; no isolated GB10 comm-vs-acceptance measurement | drafter collectives if TP-sharded (+≈11); see core §3.5 |
| 2 | Remove rank skew: equal caps, per-rank health probe, no latched rank, memory floor on every rank | `t_skew` (≈half of in-engine) | latched ranks 631-949 MHz; hidden slow state 63 vs 94 ms per TP4 step [Community-measured] (P13 §5 lever 2) | under a minute per rank; a latch clear is a power cycle (operator go) |
| 3 | Keep collectives inside CUDA graphs | host part of `t_hop` | 60 µs captured (4 nodes, switch) vs 80 µs eager (own, 3 nodes): suggestive only [Interpretation] (P03 §5 lever 4) | capture coverage (core §3.4) |
| 4 | Drafter placement: replicated or single-rank instead of TP-sharded | `N_draft` | ≈11 of 102 [Code-verified]; placement settings can be no-ops (§1.1) | memory per rank; broadcast of drafts |
| 5 | Tree instead of Ring on 3 nodes | `h(n)`: 4 → 2 | model 16.8 vs 26.6 µs [Code-verified]; [Proposed] | the ring graph was needed to pass init |
| 6 | `NCCL_MAX_NCHANNELS=8` | bandwidth term at C4-C8 | +7.7% C4, +9.7% C6, C1 flat, raw unpublished [Community-measured]; own lanes already pin 8 (P03 §5 lever 6; lane card: NCCL env) | noise: ABBA, ≥ 10 reps; spent where 8 is already pinned |
| 7 | Both PCIe halves (twins) | bandwidth term (prefill) | prefill +5-9%; decode flat to −9%; aggregate −11% on one switched TP4 [Community-measured] (P03 §5 lever 5) | new addresses and HCA list; not A/B-measured on own fleet |
| 8 | Local-argmax logits reduction | logits gather bytes (≈17% of ring bytes at 64 rows) | [Proposed]; upstream path covers greedy non-tree drafts only | fork work; sampling correctness (P12) |
| 9 | Narrower TP, or independent pairs for aggregate | `h(n)`, `t_skew`, count of groups | pairs beat one wide group 2.2× aggregate [Community-measured] (P03 §5 lever 2) | per-stream speed and pool |

Not levers on GB10: fusion, custom all-reduce, async TP (§3); a second cable to the same peer (no gain; can cap at 100 Gb/s); `NCCL_PROTO=LL` forced for decode (no isolated gain, and it forces LL onto prefill-size messages) [Community-measured] (P03 §5 levers 10, §7).

---

## 5. Padded TP shapes

```
D_pad = TP · u · ⌈D / (TP · u)⌉          φ = (D_pad / TP) / (D / TP) − 1 = D_pad / D − 1
ΔT_pad ≈ φ · T_term,rank                  (step = max over ranks, so the padded share is on the critical path)
```
[Interpretation] (P20 §3; P05 §4, §7)

| term | meaning | unit |
|---|---|---|
| `D` | dimension split across ranks: query heads, output groups, KV heads, expert or shared intermediate, vocabulary | count |
| `u` | per-rank alignment unit the kit or kernel imposes | count |
| `φ` | fractional inflation of the per-rank share | — |
| `T_term,rank` | time of the padded term (attention, shared MLP, head) on one rank | ms |

| dimension (model, kit) | D → D_pad at TP3 | u (inferred) | φ | grade |
|---|---|---|---|---|
| query heads (GLM, own fork) | 64 → 66 | 1 | +3.1% | [Code-verified] |
| query heads (DeepSeek V4 Flash) | 64 → 72, 8 → 9 output groups | 8 (one group; padding inside a group breaks the DeepGEMM output contract) | +12.5% | [Community-measured] |
| query heads (DeepSeek V4.1, one SGLang kit) | 64 → 96, 8 → 12 groups; rank 2 all padding | 32 per rank | +50% | [Community-measured] |
| query heads (DeepSeek V4.1, one vLLM kit) | 64 → 72 | 8 | +12.5% | [Community-measured] |
| drafter heads (GLM DFlash2) | 32/8 → 36/9 | — | +12.5% | [Code-verified] |
| shared expert (GLM) | 2,048 → 2,112 | 64 | +3.1% | [Code-verified] |
| vocabulary (GLM; DeepSeek) | 154,880 → 154,944; 129,280 → 129,408 | 64 | +0.04%; +0.1% | [Code-verified] |
| KV heads (Qwen 3.8 27B: 4; Flash-Next: 2) | not paddable in stock engines | — | TP3 invalid | [Code-verified] |
| expert intermediate (Flash-Next NVFP4, TP4) | 640/4 = 160, below kernel alignment | — | TP4 needs EP to load | [Community-measured] |

Consequences [Interpretation]:
- For 64-head models, TP4 is exact (`φ` = 0), and TP3 costs between 3% and 50% of the attention term, depending on the kit's `u`. DeepSeek V4.1 TP3 → TP4 gained +20% prose, attributed partly to losing the all-padding rank [Community-measured] (P05 §4). Pick the kit with the smallest `u` before comparing widths.
- A pad on a small dimension (vocabulary) costs nothing. A pad on the attention of a model whose C1 step is dominated by dense and attention bytes costs `φ` × that share. No source has timed a pad (Gap 4).
- An adapter that pads again inside the kernel (GLM 22 → 32 heads at the attention call) is a second, independent `φ` for that kernel only.

---

## 6. Fabric preflight: why each control matters

Core §5.3 is the check table. This section gives the mechanism and the order.

| control | mechanism | evidence | failure signature |
|---|---|---|---|
| Engine control plane on the fabric; Wi-Fi power save off | The engine's master address and message-queue connect address default to the host's primary IP, which can be Wi-Fi [Historical diagnostic] (WR). Scheduler metadata and multimodal tensors cross that path; NCCL does not [Interpretation; image tensors per the V3.2 packet's external note] | RTT over Wi-Fi 156 ms average with 800 ms spikes and power save on; 10 ms with power save off; 0.6 ms over the cable [Historical diagnostic] (WR). C1 code decode 44.29 (V2 arm, before the clock cap) → 44.34 (V3.2, capped) across a bundle of fabric, compact-KV and overlay changes; image cold TTFT 16.6 → 5.7 s in single uncontrolled probes [Measured] (2026-09-23 V3.2 verification) | invisible to NCCL logs; shows as multimodal latency and jitter |
| GID index verified after every reboot | A reboot renumbers the RoCE GID table. A pinned `NCCL_IB_GID_INDEX` can point at an all-zero entry. Auto (`-1`) picks an IPv4 RoCE v2 GID per device [Code-verified] | an old shared index went all-zero on one worker after reboot, which would kill that rank ≈60 s into load [Measured] (P03 §7). The two functions of one node can hold their IPv4 GID at different indices (5 and 6 on node-2), so a launcher that takes one index per rank cannot use both [Measured] (2026-09-23 two-node TP2 evaluation) | rank death during load; or a twin silently unused |
| Twin controllers named | Each QSFP port is two PCIe Gen5 x4 functions in different PCIe domains. NCCL merges only same-domain functions, so the domain-0/domain-2 twins are never merged by default [Code-verified] (P03 §2) | one function listed per port ran the pair at half for weeks; the other had only IPv6 link-local GIDs [Measured] (P03 §7) | nothing warns; counters show traffic on one function |
| Triangle routing block copied whole | Each node's two cages face different peers on different subnets, so default device choice and merging are wrong | `ibv_modify_qp` error 110 at init without subnet-aware routing or cuMem [Measured] (P03 §4) | init failure; any error 110 is a stop |
| Equal state on every rank (clock cap, sysctls, library, image) | Lockstep: the slowest rank sets `t_skew` and liveness | a ~200K turn stalled a TP3 group on memory exhaustion. P03 §6 records the workers running out while the head waited in collectives; P11 §4 records the head at a 1.36 GiB minimum on the preceding 131K gate, and whether only the head or all three nodes wedged is unresolved (core Gap 14) [Measured; Historical diagnostic for the wedge] (2026-09-16 DeepSeek V4.1 TP3 wedge) | group stall with containers "Up" |

Why the twins are a prefill lever and not a decode lever: §1.3. At C1 decode the payload sits on the flat latency floor, so doubling link bandwidth cannot move `t_coll` [Interpretation]. An own two-node run that moved NCCL onto both controllers recorded "doubling the pipe doesn't double the speed" [Historical diagnostic] (WR).

Order:
1. Once per fabric change: link layer (P03 §6 steps 1-3), then one `NCCL_DEBUG=INFO` boot. Warn-level logging hides the transport line.
2. After every reboot: GID table per device on every rank, a clock-cap read, and a check that sysctls are equal.
3. Before every TP boot: the control-plane address on the fabric; the client → head path on the intended network (core §5.3 row 1); the kit build that matches the node count (§7.2 step 5); the same image, library and disk on every rank; all ranks launched from one command; a per-rank health probe.
4. After the first run: per-cable RDMA counters (tx on each function equals rx on its peer).

---

## 7. Decision procedures

### 7.1 Predict the tax for a new model and topology

1. Count `N_fwd` from the model card: `c_l` per layer type (§1.1), `g` = 1 unless a local-argmax path is enabled. Read `N_draft` from the drafter's parallel config in source at the pinned commit. Confirm with a per-rank shape dump.
2. Payload `S = M · H · b` at C1 (M = 1 + k) and at the served concurrency. If `S` ≤ 64 KiB, use the floor band for `n`. Otherwise add `R/busbw`.
3. `T_tax` = `N_coll` × `t_coll`, stated twice: with the floor band (lower bound) and with the in-engine band.
4. `T_step` from core §3.1 with `T_tax` included; `τ_tok` = `T_tax/a`. Write the tax share into the prediction sheet with both bands.
5. Prefill: `T_comm,chunk` = `N_fwd` × `R(n, S_chunk)` ÷ busbw from the lane's measured busbw (read from the lane card: NCCL env and fabric state; if absent, measure per P03 §6 step 5).
6. Falsify: a per-rank timeline with collective start and end times at C1 and C8, with unprofiled wall time alongside. Profiled runs are attribution only: the profiler moved one TTFT from 23.8 to 35.96 s, and a skew seen in the trace did not persist [Measured] (P03 §4).

### 7.2 Choose a width

1. Fit: per-rank budget (core §2.8) at the target context for each candidate `n`. A width that fails fit is out.
2. Divisibility and `φ` for each dimension (§5). Reject widths where a KV-head count does not divide. Prefer the kit with the smallest `u`.
3. For each adjacent pair of widths, compute `ΔT_bytes = B_s/BW_eff · (1/n − 1/(n+1))`, `ΔT_tax = N_coll · (t(n+1) − t(n))` and `ΔT_pad`. Widen only if `ΔT_bytes > ΔT_tax + ΔT_pad`, or if capacity is the binding need.
4. If aggregate is the goal, compare one wide group against independent pairs (core §5.2).
5. Kit build per node count, twins, clocks and slots must match across arms. Otherwise label the comparison confounded (the first own TP2 arm lacked the cooperative-MoE kit and FP8 on all groups and read −25%/−32%; the corrected arm read −20%/−19%) [Historical diagnostic; Measured] (WR; 2026-09-23 two-node TP2 evaluation).
6. Record the prediction next to the result, and write the width verdict to the lane card with its retest trigger (new kit, new fabric state, new drafter).

### 7.3 Choose a tax lever on a running lane

1. If `a` is unknown, instrument first (per-step committed tokens and per-rank step time). Without `a`, the tax share cannot be computed (§2.1).
2. Probe skew before any fabric change: the P13 §5 lever 2 checks on every rank (≥ 30 s GEMM preflight, healthy at a mean clock ≥ 1,400 MHz and ≥ 40 W; a GEMV bandwidth probe) plus a per-rank timeline. Fix a failing rank first. No source gives a threshold on in-engine `t_coll` ÷ floor (healthy community stacks already read ≈1.5-2.1×); record the lane's own ratio as its baseline [Proposed].
3. Workload weighted to decode at C1: depth and drafter (core §3.5), then drafter placement. Leave twins and channels alone.
4. Weighted to C4-C8: channel count A/B, only where the lane does not already pin 8 (lane card: NCCL env).
5. Weighted to prefill: twins, measured with nccl-tests at 16 and 64 MiB before any serving A/B. Stop if busbw does not rise well above ≈13 GB/s (P03 §5 lever 5).
6. Tree on 3 nodes only in a maintenance window, after a `TUNING` dump, with the correctness gate first.

---

## 8. Traps

| # | belief | what happens | grade | source |
|---|---|---|---|---|
| 1 | floor × count = step tax | about half the in-engine cost; skew and waits are missing | [Interpretation] | P03 §4, §7 |
| 2 | 0.3% link use means the fabric is irrelevant | the tax is hop latency, not bytes; utilisation cannot see it | [Community-measured; Interpretation] | P03 §4 |
| 3 | twins speed decode | decode flat to −9%; prefill +5-9% | [Community-measured] | P03 §5 lever 5 |
| 4 | a width A/B across kits is a width result | the first own TP2 arm read −25%/−32% without the cooperative-MoE kernel and FP8 on all groups; with both it read −20%/−19%; both arms ran one controller per hop | [Historical diagnostic; Measured] | WR; 2026-09-23 two-node TP2 evaluation |
| 5 | a recipe comment or registry setting gives drafter placement | the drafter was TP-sharded on every rank (+≈11 collectives) regardless | [Code-verified] | P03 §7; P05 §7 |
| 6 | an NVLink fusion or overlap result transfers | every such path is disabled here; forcing SP doubles `N_fwd` | [Code-verified] | §3 |
| 7 | an env variable took effect | variables absent from the loaded NCCL build are silently ignored; only the INFO `set by environment` line proves it | [Interpretation] | P03 §2 |
| 8 | a pip install leaves NCCL alone | a nightly silently downgraded 2.30.7 → 2.29.7 and init failed on RoCE; mismatched libraries fail with `Message truncated` | [Community-measured] | P03 §6 |
| 9 | the container sees the RDMA devices | without `/dev/infiniband`, NCCL falls back to sockets: 1.88× slower serving | [Community-measured] | P03 §5 lever 1, §7 |
| 10 | wider is faster for any MoE | small-active MoE TP2 → TP4 99.0 → 89.2 tok/s | [Community-measured] | P03 §4 |
| 11 | memory on a worker is the worker's problem | one rank out of memory stalls the whole synchronous group | [Measured] | P03 §6; P11 |
| 12 | NCCL buffers are free | default 9.19 MiB per connection; 512 connections = 4.7 GiB pinned per node, shrinkable to 139 MB | [Code-verified] default; [Community-measured] 4.7 GiB and 139 MB | P03 §2, §3, §5 lever 7 |
| 13 | a PCIe rescan is a safe probe | `echo 1 > /sys/bus/pci/rescan` hard-crashed a GB10 | [Community-measured] | P03 §6 |

---

## Gaps

1. `a` per step is unrecorded on every own TP lane, so the own tax share exists only as a function of `a` (§2.1) [Proposed: needs measurement].
2. No own per-rank collective timeline. The in-engine `t_coll` on the own triangle is unknown; the own 4-10 ms band is a lower bound against the community 13-16 ms.
3. `t_hop` has not been decomposed into NIC, proxy and host-staging parts. Tree vs Ring, and LL vs LL128 vs Simple, are untried on GB10.
4. No source has timed a pad cost. The DeepSeek V4.1 `u` = 32 and `u` = 8 kits have never been compared on the same fabric.
5. No same-kit TP2 vs TP3 A/B exists on the own fleet. The 2026-09-23 comparison differs in kit, slots and pool, and `B_s` is an upper bound because no routing histogram exists.
6. The control-plane effect on decode has never been isolated (the V3.2 bundle). The image TTFT figures are single uncontrolled probes.
7. Own NCCL pinned memory (`Shmem`) has not been measured per lane. The buffer-size latency effect is unmeasured.
8. Twins on triangle legs through NCCL's own IB path are [Proposed]. Twins on own pairs have never been A/B-measured.
9. No GB10 measurement of multi-node pipeline parallelism for decode, or of any internode all-to-all EP backend.
10. The per-rank shape dump that confirms drafter placement at run time is still owed.
11. The MoE all-reduce dtype is unconfirmed at run time (fp32 per the dossier, 2 bytes in the wiki arithmetic); the §1.3 decode and prefill byte figures and the prefill share depend on it [Proposed: needs measurement].

## Sources

Wiki pages (fetched 2026-09-23): P03 in full; P20 §3, §4.1, §4.3, §5; P05 §4, §5 lever 3, §7; P13 §5 lever 2; P01, P02, P08, P11, P12, P14 by cross-reference; core §3, §5, §8-§10.

Own packets (date and title): 2026-08-29 GLM TP2 prefill profile; 2026-09-13 DeepSeek V4.1 Flash TP3 qualification; 2026-09-14 GLM-5.3-Flash EXL3 TP3 triangle qualification; 2026-09-16 DeepSeek V4.1 Flash EXL3 TP3 port component gates; 2026-09-16 DeepSeek V4.1 Flash EXL3 TP3 prefill profiling and tuning; 2026-09-16 DeepSeek V4.1 TP3 wedge records; 2026-09-20 GLM 5.3 Flash TP3 frontier dossier v2; 2026-09-21 GLM 5.3 Flash EXL3 TP3 GPU clock-cap sweep; 2026-09-23 GLM 5.3 Flash EXL3 TP3 V3.2 fabric and compact-KV verification; 2026-09-23 GLM 5.3 Flash EXL3 two-node TP2 evaluation.

Lane card (private input, cited here by packet): GLM 5.3 Flash EXL3 TP3 (node-1 head, node-2 and node-3 workers); fields used: NCCL env, drafter placement, spec method and k, clock cap, Levers not yet tried #1. Weights per rank from the engine load lines in the 2026-09-23 V3.2 verification (cutover log) and the 2026-09-20 GLM TP3 startup and prefill optimization (arm B).

Workflow review (WR; private, restated export-safe): the 2026-09-23 review of agent sessions and changelogs, and the requirements synthesis.
