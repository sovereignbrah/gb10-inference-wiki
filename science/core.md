---
title: The Science of GB10 Inference — core
created: 2026-09-23
status: checked
type: core
---

# The Science of GB10 Inference — core

Rules of use:
1. Read the lane card first (section 10). A lane constant is never taken from this file; where the card lacks it, the term is marked "read from the lane card: <field>".
2. Every number carries a grade: [Measured] own fleet, [Community-measured], [Code-verified], [Historical diagnostic], [Interpretation], [Proposed]. Anchor values show a term's size on one lane; they are never defaults.
3. Pointers: `P01`…`P22` = the wiki pages keyed in 10.1, `§n` = page section. Own packets are cited by date and title. "WR" = the 2026-09-23 workflow review of agent sessions and 161 changelogs.

Units: GiB = 2^30 B; GB = 10^9 B. SGLang logs "GB" but divides by 2^30, so its figures are GiB (P11 header). tok/s is per stream unless marked agg.

---

## 1. The machine

Own hardware: three ASUS Ascent GX10 (NVIDIA GB10, 128 GB), driver 580.173.02, firmware v0104 (P17 header).

| # | fact | value | grade | pointer |
|---|---|---|---|---|
| 1 | Compute capability | 12.1 (SM121), SM120 family. Loadable cubins: `sm_121a`, `12.0f`, `sm_120` baseline; `sm_120a` does not load [Interpretation from the suffix rules]. CUDA ≥ 12.9 required; own driver r580 = CUDA 13.0 class. | [Code-verified]; [Interpretation] | P17 §3 |
| 2 | Tensor instructions | Warp-level `mma.sync` with FP8 and block-scaled FP4 (`kind::mxf4nvf4` legal in `12.0f`). Absent: `tcgen05`, TMEM, `wgmma`, 2-SM MMA, `tile::scatter4`, gather4 into `.shared::cluster`. Sparse FP4 needs native `sm_121a`. | [Code-verified] | P17 §3 |
| 3 | SMs | 48 (6,144 CUDA cores); an audit stating 84 is wrong. | [Community-measured] | P17 §3; P12 §3 |
| 4 | Shared memory | 101,376 B per block opt-in; 100 KB (102,400 B) per SM; static 48 KB per CTA (100 KB on arch-specific `sm_121a` targets), more needs dynamic opt-in. Tiles above 101,376 B fail or fall back. | [Code-verified]; [Community-measured] | P17 §3 |
| 5 | Residency | 64K registers per SM, 255 per thread; 48 warps, 1,536 threads per SM; blocks per SM 24 vs 32 unresolved (matters only for CTAs ≤ 32 threads). | [Code-verified], contested | P17 §3, §8 |
| 6 | L2 | 24 MB. Bandwidth probes must rotate weight sets larger than L2 (one probe rotates 8× L2) or read "bandwidth" above the DRAM ceiling. | [Community-measured]; rotation rule [Interpretation] | P05 §3 |
| 7 | DRAM | LPDDR5X, 273 GB/s nominal, never achieved. Idle GPU probes: saturating read 227-235, rotating-buffer read 238.6-245.9, copy 215-229, STREAM triad 200-214 GB/s. Decode ruler `BW_eff` = 230-245 GB/s. Under a running model residual copy bandwidth is ~40-50% of idle. | [Community-measured], contested per probe | P17 §3, §4; P05 §3 |
| 8 | Unified memory | One pool shared with the OS. `MemTotal` 121.63 GiB own (119-121.7 by SKU/firmware); idle `MemAvailable` 114-116 GiB. vLLM, SGLang, llama.cpp read `MemAvailable` as GPU free. ~2 GiB display reserve, CUDA-invisible (up to 2,032 MiB reclaimable headless, ~69% read bandwidth). | [Measured]; [Code-verified]; [Community-measured] | P17 §3, §4; P11 §3 |
| 9 | Pinned allocations | Invisible to the OOM killer, cgroups, `docker --memory` and swap. Failure is `NV_ERR_NO_MEMORY` below the CUDA runtime; exhaustion shows as a node freeze, not a CUDA error. | [Measured]; [Community-measured]; [Interpretation] | P17 §3; P11 §3 |
| 10 | Fabric | ConnectX-7; each QSFP port = two PCIe Gen5 x4 functions (two RoCE devices, ~126 Gb/s raw each). One function 108.8-112 Gb/s RDMA; both halves ~196 Gb/s. RoCE only. No GPUDirect RDMA (`GDR 0`): NCCL stages through pinned host buffers from the same pool. | [Measured]; [Community-measured]; [Code-verified] | P03 §3, §4 |
| 11 | Clocks and heat | Loaded stock clock 2.39-2.46 GHz (3,003 MHz is a label). `nvidia-smi -lgc 0,<cap>` is the only host lever; `-pl` is N/A. A 2100 request lands at 2,080-2,086 MHz. `nvidia-smi` temperature reads 7-17 °C below the hottest ACPI zone; platform shutdown near 95 °C ACPI. No cross-node clock coupling: the slowest rank paces a TP group. A cap without a unit is lost at reboot. | [Measured]; [Community-measured]; [Historical diagnostic] | P13 §3, §5 |
| 12 | Tensor-core ceiling | Contested: BF16 ≈123 or ≈213 TFLOPS; FP8 ≈214-256; dense NVFP4 register-resident 486-511 at ~2.6 GHz; achieved CUTLASS NVFP4 GEMM 356 at 4096×14336×4096. Assume neither "FP8 = BF16 rate" nor "FP8 = 2× BF16". Scale ~0.8 under a 2100 cap. | [Community-measured], contested | P17 §3, §4; P04 §3 |

Clock-cap policy: cap for heat and throttling, not speed. One TP3 lane: cold ~250K prefill −3.0% at 2200, −4.9% at 2100; C1 decode means −4.9% to −6.4%, inside rep spread (n=5); peak GPU 88 → 78 °C at 2100 [Measured] (2026-09-21 GPU clock-cap sweep, GLM 5.3 Flash EXL3 TP3). One TP2 lane: −6.8% decode at 2200 (n=3, confounded by a firmware change between runs) [Measured] (2026-08-16 clock-cap A/B, DeepSeek V4 Flash TP2). Community controlled bandwidth-bound single-stream runs: ±2% [Community-measured]. Transfer is unsettled; decide per lane with P13 §5 lever 1's A/B/A rule (TTFT-weighted cost ≤ 3%, peak down ≥ 5 °C, throttle seconds down). Current state: read from the lane card: clock cap.

---

## 2. The memory budget

### 2.1 Budget equation (per rank)

```
A_0 − [ W_rank + W_draft + KV_pool + KV_draft + G + WS + B_nccl ] − max(P_warm, T_pre(chunk, P_max)) − ΔH  ≥  F_warm
KV_pool = KV_bpt × N_pool
```
[Interpretation: terms from P11 §2-§6; `max` because the warm-up peak and the longest prefill do not coincide]

| term | meaning | unit | read it from | knob | anchor (example only) |
|---|---|---|---|---|---|
| `A_0` | `MemAvailable` before engine start (MemTotal − host baseline − unreclaimable cache) | GiB | `/proc/meminfo`, every rank | drop page cache; align `vm.min_free_kbytes` / `watermark_scale_factor` across ranks | ~12 GiB of 121.63 GiB outside engine accounting on a TP2 head [Measured] (2026-08-28 GLM 5.3 Flash EXL3 TP2 500K stabilization); one rank with `min_free_kbytes` 4 GiB + `watermark_scale_factor` 100 reported 97.96 GiB available vs 110.93 on the others and exited before weights until aligned [Measured] (2026-09-13 DS V4.1 Flash TP3 qualification). Never stop the display manager to reclaim memory (it took a headless remote-desktop session down); drop the page cache instead [Historical diagnostic] (P17 §7). Lane: read from the lane card: host |
| `W_rank`, `W_draft` | target and drafter weights per rank | GiB | load lines; SGLang `Load weight end avail mem=` | quant; TP/EP; head padding | 54.25 GiB/rank GLM EXL3 TP3 (2026-09-14 value; the 2026-09-23 V3.2a load line reads 55.8); ~101 GiB/rank DS V4.1 native TP3 [Measured] (P20 §4.3). Lane: read from the lane card: model / weights |
| `KV_bpt` | KV bytes per token per rank as stored on SM12x; MLA latent replicated on every rank, GQA ÷ TP while KV heads ≥ TP | B/token | allocated bytes ÷ allocated tokens (config arithmetic under-counts ~5% [Interpretation]) | KV dtype; record format | GLM TP2 ~0.72 GiB/100K tokens; DS V4.1 SGLang 1,670.75 B/token/rank + ~0.34 GiB SWA pool [Measured] (P10 §3; P11 §6). Lane: read from the lane card: KV pin |
| `N_pool` | allocated KV tokens | tokens | vLLM `Available KV cache memory`; SGLang `max_total_num_tokens`, `DSV4 memory calculation` | 2.2 | — |
| `KV_draft` | drafter KV reserve incl. padding | GiB | boot line | compact draft KV; MNBT | padded DFlash2 ~36.7 KB/token reserved vs ~2 KB used: 3.3 GiB at MNBT 2048, 7.4 GiB at 7168 [Measured] (2026-09-08 GLM 5.3 Flash EXL3 E4 five-arm A/B) |
| `G` | CUDA-graph pools | GiB | SGLang per-runner `mem usage`; vLLM "took X GiB" is a free-memory difference (0.16 / 2.30 / 3.48 GiB on three ranks of one boot), not graph cost | capture list | 0.52 (DS V4.1), 0.7 (GLM TP2), 2.34 GiB (GLM TP3, 25 sizes) [Measured] (P11 §4) |
| `WS` | fixed workspaces (indexer, fused-row temps, autotune) | GiB | engine log; `torch.cuda.memory_stats` | right-size patches | stock vLLM GLM indexer 5,036 MiB at 1M, locked [Code-verified]; E3 scratch 560 MiB [Community-measured] (P11 §2) |
| `B_nccl` | NCCL buffers in pinned `Shmem` | GiB | `Shmem`, `Mlocked` before/after init | small-buffer env | ~4.6 → 0.14 GiB per node [Community-measured, single source] (P11 §5) |
| `P_warm` | multimodal warm-up peak | GiB | 1-2 s sampler | none (skipping defers the OOM to the first image) | ~4-5 GiB, unbudgeted by vLLM [Measured] (2026-08-28 GLM TP2 500K stabilization) |
| `T_pre` | per-request prefill working set | GiB | 1-2 s minimum through the longest prompt | chunk; context cap; logits cap (4.6) | ≈2.9 GB at chunk 1024 × 200K, DS V4.1 SGLang [Community-measured constant; Interpretation] (P04 §7) |
| `ΔH` | host drift while serving | GiB | minimum over hours | `MAX_JOBS` on builds; no co-tenants | 4.7 / 6.2 → 3-4 GiB six hours after a 1M activation [Measured] (2026-08-30 GLM TP2 composite 1M activation) |
| `F_warm` | warm floor | GiB | — | never lowered | 5 GiB (2.4) |

Ladder rule [Interpretation]: differences between consecutive `avail mem=` lines are not term sizes (other allocations land between lines: 7.41 → 5.66 is 1.75 GiB against a 1.59 GiB pool, 5.66 → 3.78 is 1.88 against 0.52 GiB of graphs). Example, TP3 head at fraction 0.95: 108.90 → 10.00 → 7.41 → 5.66 → 3.78 → ~2.9 GiB idle [Measured] (2026-09-16 DS V4.1 TP3 memory-ceiling record).

Pairing rule: all terms draw on one pool. Freeing one (`WS` right-size, FP8 dense, compact draft KV) and spending it (`N_pool`, chunk) are two boots (2.5).

### 2.2 What each engine's knob reserves

| engine | equation [grade] | failure shape | read |
|---|---|---|---|
| vLLM | `B = ceil(util × Total)`; `KV = B − nonKV_profiled − G_est`; outside-engine headroom ≈ `(1 − util) × Total`; each 0.01 ≈ 1.2 GiB at 121.7 GiB [Code-verified; Interpretation] | ValueError if available < B; `profile_run` is not capped by `util` (MemAvailable 14,228 → 7,352 MiB at 0.70, hard hangs) [Community-measured] | `GPU KV cache size: N tokens` = concurrency × `max_model_len`, not prefix capacity [Code-verified] (P11 §2) |
| vLLM pin | `--kv-cache-memory-bytes` returns the pin after profiling, logs "skipped memory profiling"; margin gone [Code-verified] | 5 of 5 own launches in one series failed (four 12.5-15 GiB pins and one auto-KV launch at utilization 0.87, the kit's stock configuration): kernel OOM at multimodal warm-up (no traceback) or KV validation [Measured]; community ladder 4.14 GiB 3/3, 5.5-7.5 GiB 0/5 [Community-measured] | kernel journal (P11 §4, §7) |
| vLLM MNBT | MNBT is charged to the profiled activation peak: 8224 → 2048 cut it 5.30 → 1.39 GiB, pool 290,632 → 1,262,567 [Community-measured] | builds that lock warm-up workspaces boot smaller MNBT then assert on a warm-prefix turn [Community-measured] | profiled peak (P11 §2, §7) |
| SGLang | `rest = A_after_load − A_before_load × (1 − f) − mm_res`; boot fails at `rest ≤ 0`; `f_min = 1 − A_after_load / A_before_load` [Code-verified] | weight-bound lane: 0.90 gives 7.41 − 10.89 < 0; `f_min` ≈ 0.932, ≈0.947 for a 750K pin; "≥ ~0.945 at these inputs", not "0.95 is the only bootable value" [Measured inputs; Interpretation] | `Load weight begin/end`, `Memory pool end avail mem=` (P11 §2) |
| SGLang cap | `--max-total-tokens` caps the pool; upstream main logs a warning above profiled capacity and uses the profiled value [Code-verified]; one fork's README says it clamps silently [Community-measured] | autotune and graph capture allocate outside the pool: 0.76 left 21.30 GB, 0.80 crashed at 3 requests [Community-measured] | warning line (P11 §2) |
| ds4 | streaming expert-cache budget = `(MemAvailable − CmaFree) − 8 GiB`; KV → `cudaMallocManaged` when ≥ 8 GiB or when buffers crowd free memory (decided on `cudaMemGetInfo`); `oom_score_adj` 1000 [Code-verified] | — | `memory: KV … planned` (P11 §2) |
| llama.cpp | `--fit` 1024 MiB margin; `--cache-ram` 8192 MiB host prompt cache from the same pool [Code-verified] | — | (P11 §2) |
| Ray launcher | host-memory monitor kills at 95% host use, which on UMA includes GPU memory | 4 boots died missing by 0.22-0.65 GB; fixed by the multiprocessing backend [Historical diagnostic] (WR) | Ray log |

### 2.3 KV pin versus fraction: contested

| position | evidence | grade |
|---|---|---|
| Fractions are brittle | `util` 0.94 needed 114.33 GiB vs 113.77-114.07 seen; a threshold missed by 0.11 GiB | [Historical diagnostic] (WR) |
| Pins remove the margin and move the OOM | 4 of 4 own pinned launches failed, but so did the one auto-KV launch (utilization 0.87) in the same series, which weakens this side; community 0/5 above the engine suggestion | [Measured]; [Community-measured] (P11 §4, §7) |

Deciding test per lane [Proposed]: same pull, three boots per arm; arm A = utilization U, auto KV; arm B = pin equal to arm A's logged KV bytes. Record per-rank `MemAvailable` minimum through warm-up and the longest qualified prefill, NV_ERR count, boot success. Keep the arm with zero NV_ERR and the higher minimum; on a tie keep auto KV.

### 2.4 Floors

| rule | value | grade | pointer |
|---|---|---|---|
| Pre-start | `MemAvailable` ≥ 8 GiB every rank | fleet rule | index; P11 §6 |
| Warm floor | ≥ 5 GiB every rank, after multimodal warm-up and after the longest qualified prefill, never at boot | fleet rule | P11 §6 |
| Traffic stop | < 4 GiB on any rank: stop benchmark traffic, keep engines running | fleet rule | P11 §6 |
| Pin step | raise ≤ 1.5 GiB per boot; stop at the first boot under 5 GiB warm | fleet rule | P11 §6 |
| Single-node cushion | 10-15 GiB or more | [Proposed] | P11 §6 |
| Post-capture | reject < 15 GiB `MemAvailable` after capture (adopted after a single-node SGLang userspace wedge at ~6.9 GiB post-capture); P01 states it for any configuration, the originating changelog for single-node lanes: scope contested | [Measured] rule | P01 §6, §7 |
| MemFree gate | ~2-3 GiB, only while `MemAvailable` < 10 GiB | [Proposed] | P11 §6 |
| Portability | never copy another fleet's fraction, pin or floor; only the delta a change produces ports | [Community-measured] | P11 §6 |

A sampled minimum is not a budget: 10 s samples missed the real minimum and the next boot ran at ~3 GiB with NV_ERR [Historical diagnostic] (WR). Sample every 1-2 s.

### 2.5 One knob per boot

1. Change exactly one memory variable per boot: pin, fraction/utilization, chunk/MNBT, context, slots, drafter layout, allocator flag, loader, NCCL buffers, weight format.
2. Freeing and spending are two boots. Confirm the warm floor rose by ≥ 80% of the predicted freed bytes before spending them [Proposed] (P11 §5 lever 3).
3. Compute the worksheet (2.8) first; abort if predicted warm floor < 5 GiB or `A_0` < 8 GiB.
4. Never relaunch a configuration that wedged a node; check the next-launch recipe on disk before any reboot [Historical diagnostic] (WR; P11 §6).
5. `expandable_segments`: one-variable A/B with a greedy logit-sum check at 64 / 65 / 512 / 4000 tokens; it fixed one indexer ratchet (32.26 → 1.49 GiB) and cost 59% of the pool on another stack [Community-measured] (P11 §5 lever 6).

Origin: an indexer right-size plus a +4 GiB pin in one boot with 4.4 GiB free left a node unreachable ~7 h until a physical power cycle; the host swapped out `sshd` [Measured] (P11 §7; duration WR).

### 2.6 Exhaustion failure mode

Mechanism [Interpretation on Measured and Community-measured incidents] (P11 §3, §7; P17 §3):
- The driver needs free pages; `MemAvailable` counts page cache that reclaim must free first, so NVRM allocations fail while gigabytes read "available".
- Pinned NVRM memory is unswappable and unscored; the OOM killer takes small host processes (desktop daemons twice, never the engine, in the 2026-09-16 wedge) [Historical diagnostic].
- `sshd` and network daemons starve; a synchronous TP group stalls with its worst rank.

Early warnings: serving-time `NV_ERR_NO_MEMORY`/Xid in `journalctl -k` (startup-only bursts of 72-172 per rank with clean serving are not failures [Measured]); `MemAvailable` under 5 GiB. Any NV_ERR during a gated prefill fails the gate: 6 at 32K on the k=3 boot and 104 on the k=5 boot (packet text: none at 32K, 104 at 131K; raw timestamps place 22 inside the 32K window and 82 inside the 131K window; head minimum 1,359 MiB = 1.33 GiB) preceded the ~200K wedge [Measured] (2026-09-15 DS V4.1 Flash TP3 k3 and k5 scorecards).

Amplifier: client retries; a ~200K prefix stalled 2-3 min per 1,024-token chunk and was resent 8 times on 60 s timeouts [Historical diagnostic] (P22 §4).

Wedge signature: ping and L2 answer, TCP 22 accepts, no banner or no login; `/health` may return 200; containers may read "Up" with the engine dead [Community-measured; Measured] (P11 §3, §7).

Not protection: swap (0 MiB used during 104 NV_ERR at swappiness 0) [Measured]; `docker --memory` (7.15 GiB container RSS at fraction 0.76 under a 100g cgroup) [Community-measured]; a KV cap (pool 25-28% full at the wedge) [Historical diagnostic].

### 2.7 Incident runbook

1. Stop client retries and admissions to the lane.
2. Probe the fabric path: ping, then SSH over the direct-cable alias per rank.
3. On reachable ranks read `journalctl -k` (not `dmesg`, which rotates within ~an hour under these errors) for NV_ERR, Xid, `oom-kill`.
4. Wait for the head's scheduler abort; if a rank died, remove every rank of the group before relaunch (survivors keep their allocations) [Community-measured] (P11 §6).
5. Never infer "no reboot happened" from telemetry; a telemetry-based account was contradicted by the operator's direct observation [Historical diagnostic] (WR). Record the operator's account.
6. Power cycle only with the operator's explicit permission for that node.
7. Before relaunch: confirm the next-launch recipe is not the wedging one; change one variable that shrinks `T_pre` or the context (4.6); gate on a cold prefill at the longest advertised context with zero NV_ERR and ≥ 5 GiB on every rank.
8. File the incident (configuration, per-rank minimum, NV_ERR counts, retries, rule that followed); update the lane card: Last incident.

### 2.8 Pre-boot worksheet

1. Read `A_0` per rank now, not from a packet.
2. Predict `W_rank`, `W_draft` (checkpoint bytes ÷ ranks + padding), `G`, `WS`, `B_nccl`, `P_warm` (last boot receipt or lane card).
3. Predict `N_pool` from 2.2 and `KV_bpt`.
4. Predict warm floor = `A_0 − Σ fixed − KV_pool − KV_draft − max(P_warm, T_pre(chunk, P_max)) − ΔH`, with Σ fixed = `W_rank + W_draft + G + WS + B_nccl`.
5. Abort if < 5 GiB on any rank.
6. After boot compare receipts with predictions: tolerance ±1.5 GiB per term, ±10% pool tokens [Proposed] (requirements §6). A miss beyond tolerance is a finding to explain before any speed run.

---

## 3. Decode = bytes per step

### 3.1 Step equation

```
T_step ≈ max_r(B_r) / BW_eff + N_coll · t_coll + N_launch · t_launch (ungraphed only) + T_host
B_r    = W_dense/TP + W_repl + E_r(M) · b_exp + KV_read(depth, rows) + S_state + B_draft + B_head/TP
tok/s per stream = a / T_step        agg = C · a / T_step        M = C · (k + 1)
```
[Interpretation, assembled from anchors] (P05 opening section)

| term | meaning | unit | TP behaviour |
|---|---|---|---|
| `B_r` | bytes rank r reads per step | GB | — |
| `BW_eff` | achieved DRAM read (fact 7; name the probe) | GB/s | — |
| `W_dense`, `W_repl` | sharded dense/attention weights; replicated weights | GB | ÷ TP; none |
| `E_r(M)`, `b_exp` | distinct experts rank r touches at M rows; bytes per expert | count, GB | ÷ EP |
| `KV_read` | `Σ_seq depth · KV_bpt_read` over full-attention layers | GB | ÷ TP (GQA, KV heads ≥ TP); none (MLA latent) |
| `S_state` | recurrent state (KDA/GDN/Mamba), depth-independent | GB | ÷ TP |
| `B_draft`, `B_head` | drafter reads; vocabulary-head reads (target + each full-vocab draft pass) | GB | per placement; ÷ TP |
| `N_coll`, `t_coll` | collectives per step; effective in-engine cost | count, µs | 5.1 |
| `N_launch`, `t_launch`, `T_host` | host-issued launches, cost each; host stalls | count, µs, ms | — |
| `C`, `k`, `a` | concurrent streams; draft tokens per step; committed tokens per step per stream | — | read from the lane card: slots; spec method and k |

### 3.2 Bytes by family

| family | dominant term | TP behaviour | notes |
|---|---|---|---|
| Dense | `W_dense` | ÷ TP | stock vLLM requires TP to divide query and KV heads: Qwen 3.8 27B (4 KV heads) and Flash-Next (2) cannot run TP3 [Code-verified] (P20 §2, §4.3) |
| MoE top-k | `E_r(M) · b_exp` | ÷ EP; slowest rank sets the step | `E_r(M) ≤ (E/EP) · [1 − (1 − k_top/E)^M]` under uniform independent routing, an upper bound [Interpretation]; measured per-rank implied expert bandwidth 172-351 GB/s at 1-2 rows refutes read-once accounting (correlated routing, L2 reuse) [Historical diagnostic] (P05 §3) |
| MLA | small latent KV, replicated | adding ranks adds bandwidth, not KV tokens (unless DCP) | sparse-indexer read grows with depth: −23% from 2K to 64K on one engine [Community-measured] (P05 §4) |
| Hybrid linear-attention | `S_state` constant; only full-attention layers grow with depth | state ÷ TP | GLM-class decode near depth-flat (34 of 45 layers KDA) [Community-measured] (P20 §4.1) |

### 3.3 The bandwidth floor

```
T_floor = max_r(B_r) / BW_eff        ceiling(no spec) = 1 / T_floor        ceiling(spec) = a / T_verify
```
Accuracy: within ~10% for single-node BF16 dense; a ±25% guide for multi-node MoE [Interpretation] (P05 opening). Anchors:
- Qwen 3.8 27B FP8, 1 node, no speculation: 7.84 tok/s [Measured] (2026-08-21 FP8 MTP sweep); at 26-31 GB/token that is 204-243 GB/s, at the ruler [Interpretation] (P20 §4.1).
- GLM 5.3 Flash NVFP4 TP2, no speculation: 14.1-14.2 tok/s on every content class, content-flat as a bytes-bound step should be [Measured] (2026-08-27 GLM 5.3 Flash MTP-4 versus MTP-off).
- NVFP4 dense lands 25-40% below the model (70B at 75%, 49B at ~59%) while BF16 dense lands within ~10% [Community-measured; Interpretation] (P05 §4, §8). Cause open.
- No DRAM byte counter exists for any GB10 decode kernel; never compute bandwidth across mixed ranks (a mixed-rank "66-92% of nominal" was withdrawn) [Historical diagnostic] (P05 §7).

Per-lane ceiling: read from the lane card: model / weights, then P20 §4.1's bytes-bound column (e.g. ≈35 tok/s NVFP4 GLM at 3 nodes, ≈27 NVFP4 Qwen 27B at 2 nodes, no speculation) [Interpretation].

### 3.4 Launch term

- Applies only to eagerly issued steps (uncaptured shapes, eager mode). Graphed steps run ~98-99.9% GPU-busy on one or two nodes (a separate count puts the host gap at 0.1-2.4%) and ~95% on three nodes [Community-measured] (P01 §3, §4; P05 §4).
- `t_launch`: NVIDIA's 1-5 µs estimate; unmeasured on GB10 (non-GB10: 2.7-3.0 µs launch, 0.003-0.006 µs per kernel in replay) [Community-measured] (P01 §4).
- Small kernels still cost device time inside graphs: ~2,000 per step ≈ 10 ms of an ~82 ms DS V4.1 TP3 step [Community-measured] (P01 §4).
- Stop rule: C1 profile CPU gap < 3% → graph/launch work exhausted (P05 §5 lever 7).
- Coverage rule: capture list ⊇ `{n·(1+k) : n ∈ 1..slots, k ∈ k-set}`, ceiling rounded up to a multiple of 8; prove N/N captured and no `Truncating max_cudagraph_capture_size` line. An uncaptured top shape cost −12% C6 agg on one pair [Community-measured] (P01 §5 lever 1).

### 3.5 Speculative decoding arithmetic

```
a = 1 + Σ_{i=1..k} q_i                         q_i = P(draft position i committed)          exact accounting
a = 1 + k · α_pd                               α_pd = accepted / proposed (vLLM rate)
a = L                                          L = SGLang accept length incl. bonus; rate = (L − 1)/(N − 1), N = num draft tokens (P02 §2)
a = (1 − p^{k+1}) / (1 − p)                    i.i.d. conditional chain rate p              [Interpretation]
r(k, C) = T_verify(k, C) / T_plain(C)          speedup S = a / r            break-even a = r
```
Accounting check: MTP-4 per-position 84.8 / 70.5 / 59.3 / 51.0% gives `Σq/4` = 66.4%, matching the logged 66.39% [Measured] (2026-08-31 vLLM MTP depth sweep, Qwen 3.8 Flash Next TP2).

| regime | r | grade | source |
|---|---|---|---|
| Dense, 1 node, SGLang DFlash2 | `T_verify` flat: 138 ± 3 ms whether 1.55 or 7.95 tokens committed [Measured]; `r` ≈ 1 + drafter bytes / target bytes ≈ 1.2 (3.85 of 20.09 GB) [Interpretation]; no matched-depth spec-off arm | [Measured]; [Interpretation] | 2026-08-19 single-node DFlash2 campaign audit; P05 §4 |
| MoE, TP2 EXL3, k=3 → 7 | `T_verify(7) / T_verify(3)` ≈ 1.44-1.48, not `r` (no spec-off arm) (derived: prose a 2.94 → 3.57 at 22.11 → 18.60 tok/s; code a 2.61 → 3.23 at 24.74 → 20.66) | [Interpretation] on [Measured] | 2026-09-03 A1 k=3 / k=7 qualification |
| ds4 fork, DSpark 1+4 rows | 2.03-2.08 at 2K-64K, ~2.4 at 240K-515K; stated break-even ~51% acceptance = `p` ≈ 0.51-0.53 above | [Community-measured] | P02 §3, §4 |

Why speculation is the lever: a dense step reads the same target weights for 1 or k+1 rows (verify step flat in committed tokens [Measured]), so `r` ≈ 1 + drafter bytes / target bytes and `S` ≈ `a / r` [Interpretation]. It stops paying as `r` grows: MoE expert union, per-step collectives, a BF16 draft head on a 4-bit target, a batched path that drops speculation (ds4 fork: 1.0 tokens/step at C ≥ 2) [Measured; Community-measured] (P02 §3, §7).

Acceptance is a property of content: counting 95.88% vs prose 49.62% → 101.7 vs 44.2 tok/s, same drafter and setting [Measured] (2026-09-16 combined runtime qualification). Live acceptance ran 37.4-54.1% on four lanes; one MTP lane read 3.00/3 on probes and 0.00 on ~245K live traffic because runtime YaRN never reached the drafter [Measured] (P02 §4, §7).

Depth by workload:
- Real-shaped code, prose, depth on MoE TP2: k=3 beat k=7 by +19-31% on every cell; k=7 won only count-to-200 [Measured] (2026-09-03 A1 k=3 / k=7 qualification).
- Code on a dense single node: k=10 over k=8, code +14.6%, prose +7.0% [Measured] (2026-08-21 k=10 boot-gated qualification).
- DS V4.1 DSpark k=5 vs k=3 trades code +12% (56.62 vs 50.39, P02 §4) or +15% (57.92, P05 §4; two k=5 records) and prose −3 to −8% for C8 agg −14% [Measured] (2026-09-15 k3 and k5 scorecards); the winner is an operator weight (lane card: Owner notes).

Selection procedure:
1. Bake context extension (YaRN/rope) into target and drafter configs; never pass it at runtime (P02 §6).
2. Check alignment: drafter trained for this target and quant family; abliterated targets lose prose acceptance [Historical diagnostic] (WR); a cross-quant drafter crashed at 15.8% [Measured] (P02 §5).
3. Choose k per workload class and depth on real-shaped fixtures; maximize `a / T_verify`, not acceptance %.
4. Check `M = slots · (k+1)` against the capture list and kernel row caps (EXL3: `EXL3_TEMP_ROWS_FUSED` ≥ `MAX_NUM_SEQS · (k+1)`; 8 × 8 = 64 sat on the fused limit) [Measured] (P05 §7). A k-set change moves rows across dispatch boundaries: re-profile or label the A/B confounded [Measured] (P02 §7).
5. A k result holds only for its drafter, scheduler build and topology (lane card: spec method and k).
6. After any change, read live acceptance from idle-gated `/metrics` deltas for ≥ 30 min; live far below bench is a failure (P02 §6).

### 3.6 Sampler and lm_head share

```
T_head = V · H · bytes_w / BW_eff    per full-vocabulary read; V = vocabulary rows, H = hidden size, bytes_w = bytes per weight
```
- Sampler plus speculation bookkeeping: 0.07 ms/step at C1, 0.15 at C8 [Community-measured] (P05 §4). Not a C1 lever.
- Qwen 3.8 27B BF16 head: 248,320 × 5,120 × 2 B = 2.54 GB ≈ 10.9-11.0 ms per read; packed NVFP4 ≈ 0.72 GB ≈ 3.1 ms [Interpretation] (P12 §3). The head was 2.54 of 24.22 GB per verify step on one lane [Measured bytes] (2026-08-19 single-node DFlash2 campaign audit).
- Draft-vocabulary trim 248K → 64K: head 0.59 → 0.16 GiB/rank, C1 +13-19% while acceptance fell 58.6% → 52.2%; an 8K list fell to 36% and lost everywhere [Measured] (2026-09-05 FP8-KV qualification). Falsifier (P05 §5 lever 4) [Interpretation]: relative loss in committed tokens per step (Δa / a) × `T_step` > saved head bytes ÷ `BW_eff`.
- TP gather at 64 rows: ≈17% of per-rank ring bytes with 16-bit logits on GLM TP3 (≤ 30% if fp32) [Interpretation] (P12 §3).
- Integrity gate: a dropped `lm_head.weight_scale` let draft and verifier agree on a broken head (prose acceptance p90 8.0); run a greedy top-1 logprob probe after every relaunch [Measured] (P05 §7).

---

## 4. Prefill = compute and chunk economics

### 4.1 Per-chunk cost model

```
t_step(M, P) ≈ t_fixed + a(M, W, L2, M mod 4) · M + b · M · P + c_sync
TTFT_cold ≈ t_0 + Σ_chunks t_step(M_j, P_j)          R_eff = P_prompt / TTFT
```
[Interpretation] (P04 §8)

| term | meaning | unit | evidence |
|---|---|---|---|
| `M` | tokens scheduled in the step (MNBT, chunk, LPT grant) | tokens | the logged scheduler config, never derived from the hybrid page (P04 §7) |
| `P` | prefix already processed | tokens | — |
| `t_fixed`, `t_0` | per-step fixed cost; request intercept | ms, s | intercept ~0.2-0.4 s on one stack; ~40 ms host + template on another, the rest short-M GPU work [Community-measured] (P04 §4) |
| `a(·)` | per-token GEMM cost; worse when weights exceed L2 or M is misaligned | ms/token | 4.5 |
| `b·M·P` | attention and indexer growth with prefix | ms | single-digit % through ~300K on GLM lanes, −16.5% near 1M; DS V4 ≈ −19% per depth doubling 297K → 992K [Measured; Community-measured] (P04 §8) |
| `c_sync` | host syncs (`.tolist()`, `nonzero`, D2H) | ms | 1.45 s CPU in `nonzero` over a 3-step 16K capture on one kit [Community-measured] (P04 §5 lever 9) |

Effective prefill includes the intercept; one engine's dashboard "prefill rate" is uncached compute throughput, not cold full-prompt rate [Code-verified] (P04 §7).

### 4.2 MFU

```
MFU = R_eff · F_tok / (N_nodes · Peak_dtype · f_clock / f_peak)          F_tok ≈ 2 · N_active + F_attn(P)
```
Terms: `R_eff` prompt tok/s (4.1); `F_tok` FLOP per token; `N_active` active parameters; `F_attn(P)` attention FLOP per token at prefix P; `Peak_dtype` TFLOPS per node at `f_peak`; `f_clock` loaded clock (MHz).
- State the assumed peak and clock with every MFU figure; the peak is contested (fact 12).
- Worked [Interpretation], omitting `F_attn(P)` and the clock factor: GLM 5.3 Flash `F_tok` ≈ 33.5 GFLOP (16.74B active, P20 §4.1); 1,700 tok/s at ~95K on 3 nodes ≈ 19 TFLOPS/node = ~9% of 213 or ~15% of 123. With both terms applied (chapter 04 §1.6): V3.2a at 95K under the 2100 cap ≈ 21.8 TFLOPS/node → ≈13% of 213 × 0.8 or ≈22% of 123 × 0.8. Recorded estimates: own TP3 ~9-10% at 95K; a two-node recipe ~11-14% per node; a recipe roofline at 125 TFLOPS put one lane at ~14%, ~8% against 213 (P04 §7, §8). Neither is a verdict.
- Clock elasticity of prefill ≈ 0.4 (−12.5% clock → −4.9%; −21% → −9.1%) on one TP3 lane [Measured] (2026-09-21 GPU clock-cap sweep): large-chunk prefill is clock-sensitive, partly compute-bound; "prefill on a 273 GB/s bus is bandwidth-bound" is wrong for large chunks [Historical diagnostic] (WR).

### 4.3 Chunk: speed versus transient versus interleave

| raising the chunk… | effect | anchors |
|---|---|---|
| GEMM efficiency per token | + for MoE (short steps re-stream experts); ~0 for dense | llama.cpp 512-token rate 45-55% of 4,096 for four MoE models; dense 7B faster at 512 [Community-measured] (P04 §3) |
| transient `T_pre` | ∝ chunk × prefix | DS V4.1 EXL3 TP3 1024 → 8192: ~4 GiB more at the minimum [Measured] (2026-09-16 DS V4.1 EXL3 TP3 optimized-1M) |
| KV pool | − (MNBT profile charge; drafter reservation scales with budget) | GLM TP3 7,168 / 4,096 / 2,048 → 2.58M / 3.0M / 3.37M from one pin; GLM TP2 7,168 cost ~640K context [Measured] (P04 §4) |
| incumbent stall | + | longest stall at a 30K newcomer 2.37 s (7,168) vs 1.02 s (4,096) [Measured]; decode share during a 262K prefill 1.7% (8,192) vs 5.0% (2,048) [Community-measured] (P04 §4) |

Direction is lane-specific [Measured] (P04 §4; WR): GLM TP3 ~4% per halving; Flash-Next 1 node 2048 → 4096 +7.8% at 500K; DS V4.1 EXL3 1024 → 8192 +26% at 32K; Qwen 27B 1 node 8192 → 2048 TTFT −17-22% [Historical diagnostic] (WR); DS V4.1 SGLang chunk 256 ~1.0k vs ~2.0k tok/s at 1024 (not isolated: pool, slots and context also changed). Declare every lowering of a performance setting; a silent 1024 → 256 "memory precaution" halved prefill and the needed chunk was never measured [Historical diagnostic] (WR).

Interleave: decode tokens produced during a long prefill ≈ (prefill tokens / chunk) × committed tokens per decode step [Interpretation] (P04 §5 lever 2). LPT, not the budget, lets a newcomer in. Fair mixed prefill: 2K newcomer TTFT 130 → 5.9 s, C8 agg +12.6%, incumbents −30% only while newcomer chunks run [Measured] (2026-09-15/16 GLM TP3 upgrade A/B). Concurrent cold prefills do not batch (8 requests +1.3% aggregate) [Community-measured] (P22 §3).

### 4.4 MoE: all experts per chunk

```
ρ = k_top · M / E                                   rows per expert
touched ≈ E · [1 − (1 − k_top/E)^M]  → E at chunk sizes          [Interpretation]
expert bytes per prefill token ≈ W_experts_rank / M
```
- Every expert is read once per layer per chunk; a larger chunk amortizes expert bytes, and below GEMM saturation the step re-streams experts [Interpretation] (P04 §3).
- GLM top-8 of 288 at M = 2,048: ρ = 56.9 against a 64-row fat cutoff [Interpretation] (requirements §2 #4). Routing is concentrated: 91% (16K) to 98.6% (100K) of prefill MoE layers had an expert above 128 rows [Measured per P20 §4.1; Community-measured per P04 §4: grade contested between pages].
- Grouped fat-expert kernels: E2 +20-21% (reproduced), E3 +37-45% over E2 [Community-measured]; own geometry variants lost 4-219% [Measured] (P04 §5 lever 4).

### 4.5 Large-M GEMM

- Weights > 24 MiB L2: blockwise-FP8 CUTLASS 173 / 90 / 64 TFLOPS at M = 4K / 8K / 16K; a scheduler swizzle (vLLM PR #55180) restores 148-168 [Community-measured] (P04 §3).
- M not a multiple of 4 ran the small-M tile before vLLM PR #52775; padding cut 8K TTFT −37% [Community-measured] (P04 §3).
- Weight-only kernels lose at large M: a BF16 reconstruction of the KDA input projection, used above M = 512, was 2.17-2.25× faster than the weight-only Marlin kernel at M = 2,048 (component) and gave +7.2-7.9% prefill in a group with three startup fixes, at +2.26 GiB/rank [Measured] (2026-09-20 GLM TP3 startup and prefill optimization).
- Dense precision trade: FP8 weight-only decode +7-14% at short depth, prefill −18 to −21% [Measured] (2026-09-08 E4 A/B); recorded break-even ≈ 1K output tokens per 27K new prompt tokens at coding rates; the same trade gives ≈17K for prose and ≈6K for structured output (P04 §5 lever 6; chapter 04 §3).

### 4.6 Request envelope

```
T_pre(chunk, P) ≈ c_lane · chunk · P
P_max,safe ≈ (H_warm − F_warm) / (c_lane · chunk)          H_warm = warm-idle MemAvailable on the tightest rank
```
`c_lane`: transient bytes per (chunk token × prefix token); read from the lane card: c_lane [Proposed field]. Anchors: DS V4.1 low-ratio indexer ≈14 B, T × L below ~2.0e8 token² (analysis-only note, unpublished constants) [Community-measured]; SGLang GLM kpool ≈1 B (chunk rows × total tokens bytes; 1.07 GB at 8,192 × 131K) [Code-verified]; GLM-5.2 ≈9.32 B (11.45 GiB at 600K × 2,048) [Community-measured] (P04 §3; P11 §2).

Worked [Interpretation]: DS V4.1 SGLang TP3, chunk 1024, head ~1.4-2.9 GiB warm (already under `F_warm`, so `P_max,safe` < 0): 14 × 1024 × 200,000 ≈ 2.9 GB (2.7 GiB), at or above that headroom before allocator growth → the observed ~200K wedge with the pool 25-28% full [Historical diagnostic] (P04 §7).

Rules: the context advertised to clients must not exceed the tested envelope; cap the request or the chunk, not just the pool (P11 §7; P22 §3).

Chunk decision procedure:
1. Read the logged scheduler block, hash block and hybrid page; never derive them (a predicted 2,048/512 alternation did not exist) [Measured] (P04 §7).
2. Compute ρ per candidate against the fat cutoff (4.4).
3. Compute `T_pre` at `P_max`; drop candidates that break the envelope.
4. Compute the pool change (MNBT charge, drafter reservation).
5. Ladder: current value and one rung each side; cold 8K and 100K × 3, one memory-watched probe past 128K at 1 s, short decode (P04 §6). Keep a rung only if it beats the noise floor without losing capacity or decode.

### 4.7 Prefix reuse economics

```
T_turn ≈ t_0 + U / R_prefill + O / R_decode (+ tool time)          U = uncached prompt tokens, O = output tokens
warm U ≈ new tokens + trailing partial page (+ one speculative block dropped per hit on some builds)
```
[Interpretation] (requirements §3 #15; P04 §4)
- The hash chains from token 0; anything varying near the head re-prefills the prompt. One head-placed effort toggle cost ~84 s at median depth (66.8K), ~157 s at p90 (124.1K) on real traffic [Measured depths; Interpretation: depth ÷ ~782 tok/s, not timed] (2026-08-30 metered traffic; P10 §4).
- Hybrid pages are thousands of tokens (2,560 on a TP3 lane, 3,584 on a TP2 lane, 1,664 on a single-node lane): each warm turn recomputes a trailing page, ~5.9-6.4 s vs ~0.6 s for an exact hit on one lane [Measured] (P10 §3; P04 §4). Lane page: read from the lane card: chunk size.
- The logged pool is not shareable capacity: 501,152 tokens held ~6,960 shareable until a drafter block-id fix (then 907,024; follow-up hits 0 → 85.8%) [Measured] (P04 §7).
- Two ~275K sessions on an 896K pool evicted each other: 97.7% → 0% cached [Measured] (P04 §7).
- Read `cached_tokens` per request after enabling it (vLLM `--enable-prompt-tokens-details`, SGLang `--enable-cache-report`); without them clients show zero cache reads while caching works [Code-verified] (P10 §6; P22 §6).

Cache-safe template gate (every template, shim, picker or alias change):
1. Bytes before the last user turn stay stable: no timestamps, per-request headers, hashes, effort directives or tool-list changes at the head (P22 §5 lever 1); one model ID per lane (extra picker IDs destroyed caching) [Historical diagnostic] (WR).
2. Retention probe: low → high → low tiers on a ≥ 46K-token history; pass if calls 2-3 reuse everything up to the final user turn, to the page grain.
3. Correctness probe: four-arm placement (head / tail / none / pre-last-user) with scored answers; tail placement produced malformed high-tier answers (9 of 20 or 10 of 22, count unresolved) [Measured] (P10 §5 lever 1).
4. On hybrids compare a cache hit's output with cache-off (hits restored zero state on some builds) [Community-measured] (P04 §7).
5. Ship only when both probes pass; a revert shipped without the cache check left the canary red two days [Historical diagnostic] (WR).

---

## 5. Multi-node = a per-step tax

### 5.1 Tax equation

```
T_tax ≈ N_coll · t_eff          per-token tax = T_tax / a
t_eff (in-engine, incl. rank waits) ≈ 130-150 µs          t_floor (standalone, small message) ≈ 40-86 µs
```

| quantity | value | grade | source |
|---|---|---|---|
| `N_coll` per decode step | ~90-120 at any TP: 102 GLM TP3 (≈91 forward + ≈11 drafter) [Code-verified]; 104 DS V4.1 TP3; 108 GLM NVFP4 TP3; 88 at TP4 [Community-measured] | mixed | P03 §1, §4; P12 §3 |
| `t_floor` | 2 nodes 41-43 µs; 3-node ring 72-86 µs; 4 nodes 60 µs captured; own eager 80 µs at 4 KiB (upper bound) | [Community-measured]; [Measured] | P03 §4 |
| In-engine NCCL per C1 step | 13-16 ms, ~14-20% of the step | [Community-measured] | P03 §4 |
| `t_floor × N_coll` | about half the in-engine cost; the own 4-10 ms dossier band is a lower bound | [Interpretation] | P03 §4 |
| LL model floor, 3 nodes | Tree 16.8 µs vs Ring 26.6 µs; Tree never tried on the own triangle | [Code-verified] | P03 §5 lever 8 |
| Link use during decode | 0.3% of 185 Gb/s on one TP2 pair | [Community-measured] | P03 §4 |

What reduces it: more committed tokens per step (the cost is per step; a TP-sharded drafter adds ~11 collectives) [Interpretation; Code-verified] (P03 §5 lever 3); graph-captured collectives (suggestive only) (lever 4); `NCCL_MAX_NCHANNELS=8` (+7.7% C4, +9.7% C6, nil C1, raw unpublished; own lanes already pin 8, so it is not a remaining lever there) [Community-measured] (lever 6); both PCIe halves (prefill +5-9%, decode flat to −9%; not A/B-measured on own fleet) [Community-measured] (lever 5).

Disabled on GB10 multi-node: all-reduce + RMSNorm fusion, FlashInfer and custom all-reduce, symmetric memory, NCCL symmetric kernels, NVLS/multicast, async TP, GPUDirect RDMA; sequence parallelism is not auto-applied and doubles op count if forced [Code-verified] (P03 §3, §7; P01 §3).

### 5.2 TP width by model need

```
T_step(TP) ≈ (B_shardable / TP + B_replicated) / BW_eff + T_tax(TP) + T_fixed
```
[Interpretation] (P05 §5 lever 3)

| model need | TP effect | anchors |
|---|---|---|
| Dense ≥ 27B that fits one node | TP2 speeds decode | +60-73% TP1 → TP2 on a dense 27B [Measured] (2026-08-17 TP1 vs TP2 identical battery); ITL +4.5 ms (TP2), +9.6 ms (TP4) over ideal halving [Community-measured] (P05 §4) |
| Small-active MoE that fits | little or negative | 30B-A3B TP1 → TP2 +5%; 35B-A3B TP2 → TP4 99.0 → 89.2 [Community-measured] (P03 §4) |
| Large or weight-bound MoE | capacity and per-rank bytes | TP3 → TP4 +20% prose on DS V4.1 [Community-measured]; own GLM TP2 → TP3 +29-45% confounded with kit, slots, k [Measured] (P03 §5 lever 2); the controlled 2026-09-23 two-node TP2 evaluation gives TP3/TP2 = 1.23-1.24 decode, 1.10-1.14 prefill [Measured] |
| Aggregate throughput | independent pairs beat one wide group, 2.2× | [Community-measured] (P03 §5 lever 2) |

Width procedure:
1. Divisibility of query heads, KV heads, output groups, expert intermediate, vocabulary by TP [Code-verified] (P20 §4.3).
2. Padding if not divisible: DS V4 Flash TP3 64 → 72 heads (8 → 9 groups); DS V4.1 EXL3 TP3 2,304 = 3 × 768 with 96 heads; GLM EXL3 TP3 64 → 66 heads [Community-measured; Measured] (P03 §7; P20 §4.3). Pad cost untimed; a rank of all-padding shards wastes a node [Community-measured] (P05 §4).
3. Per-rank fit (section 2) at the target context; native DS V4.1 TP3 at ~101 GiB/rank is short-context only [Measured] (P20 §4.3).
4. If `T_step(TP2) − T_step(TP1)/2 > N_coll × ~70 µs`, look for padding or a fabric fault (P05 §5 lever 3).
5. Kit build must match node count (2-node and 3-node cooperative-MoE builds differ) [Historical diagnostic] (WR).
6. Compare widths with every other knob fixed; an arm missing its kit and one controller per port was withdrawn as unfair [Historical diagnostic] (WR).

### 5.3 Fabric preflight (before any TP ≥ 2 boot)

| # | check | how | pass |
|---|---|---|---|
| 1 | Engine control plane on the fabric; Wi-Fi power save off; client → head path on the intended network | engine master / `mq_connect_ip` address; ping RTT | fabric address (a control plane on Wi-Fi ran 156 ms average, spikes 800 ms, undetected) [Historical diagnostic] (WR) |
| 2 | Link | `ethtool` 200000Mb/s; `ibv_devinfo` `PORT_ACTIVE`, `active_mtu` | as expected (P03 §6) |
| 3 | GID after every reboot | GID table per device | IPv4 RoCE v2 entry at the index used on every rank; a reboot left the old index all-zero (kills a rank ~60 s into load) [Measured] (P03 §7) |
| 4 | Both functions per port named, each with an IPv4 RoCE v2 GID | `NCCL_IB_HCA`; GID table | a function with only IPv6 link-local GIDs left the link at half for weeks [Measured] (P03 §7) |
| 5 | MTU | `ping -M do -s 8972` every leg | no fragmentation |
| 6 | Triangle block copied whole (a subset fails init) | per-rank ring graph, `NCCL_ALGO=Ring`, `NCCL_CROSS_NIC=1`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_IB_MERGE_NICS=0`, `NCCL_NET_MERGE_LEVEL=LOC`, `NCCL_CUMEM_ENABLE=1` | init passes; `ibv_modify_qp … 110` is a stop [Measured] (P03 §6, §7) |
| 7 | Transport | one boot with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING,ENV` | `NET/IB`, `GDR 0`, expected channels, env echoed; sockets cost 1.88× serving [Community-measured] (P03 §5 lever 1) |
| 8 | Traffic on cabled functions | per-cable RDMA counters after the run | tx on each function = rx on its peer |
| 9 | Same NCCL library on every rank | logged version | match ("Message truncated" on mismatch) |
| 10 | Host sysctls equal | `vm.min_free_kbytes`, `vm.watermark_scale_factor` | equal (2.1) |
| 11 | Same cap, no slow rank | median `clocks.gr` under load, power, GEMV probe | latched ranks read 631-949 MHz; a hidden slow state reads GEMV 66-80 GB/s at normal clocks [Community-measured] (P13 §5 lever 2) |
| 12 | Image and disk on every rank; ranks launched from one command | `docker images`; `df` | sequential launches produced false fabric failures [Measured] (P03 §6) |

---

## 6. Kernels

### 6.1 Definitions and fusion arithmetic

A kernel is one GPU program for one layer class (GEMM, attention, MoE, sampler); the engine takes the first candidate whose gates pass on the GPU (P16 §1).

```
ΔT_fusion ≈ ΔN_launch · t_launch (ungraphed only) + ΔB_intermediate / BW_eff + Δ(kernel tails, idle)
```
[Interpretation]
- On a graphed, bandwidth-bound step fusion is near zero: −61 launches/token flat; ~300 launches moved into graphs flat; own combined arm C (startup/prefill group plus the fused-staging decoder pair; decoder not isolated) −0.8% C1, −5.5% C8 [Community-measured; Measured] (P01 §4; P05 §4).
- Stop rule: in-engine time within 10% of `bytes / BW_eff` → do not tune (P05 §5 lever 13). Marlin W4A16 MoE already runs at 91-96% of a 245.9 GB/s ruler [Community-measured].

### 6.2 How engines pick kernels on CC 12.1

GB10 passes every "≥ 10.0" gate and fails every "family 10.x" and "== 12.0" gate [Code-verified] (P16 §3).

| class | engine | default on 12.1 | switch | known-bad |
|---|---|---|---|---|
| Non-MLA attention | vLLM | FA2 | `--attention-backend` | FlashInfer without XQA downgrades full decode graphs; +16% from keeping full graphs [Community-measured] (P16 §5 lever 1) |
| Sparse MLA | vLLM | `FLASHINFER_MLA_SPARSE_SM120` (only candidate) | — | padded FULL-graph replay hang before vLLM #51538 [Community-measured] (P02 §3) |
| NVFP4 W4A4 linear | vLLM | FlashInfer CUTLASS | `--linear-backend` | emulation fallback without the SM12x FP4 kernel: 1.1 vs 77.1 tok/s [Community-measured] (P05 §2) |
| NVFP4 W4A16 | vLLM | Marlin | `--linear-backend` | Marlin's "no native FP4" line is a code-path message, not a hardware fact [Interpretation] (P16 §7) |
| NVFP4 MoE | vLLM | FlashInfer CUTLASS; b12x excluded from auto | `--moe-backend` | open b12x SM121 TP reports [Community-measured] (P16 §5 lever 7) |
| Block FP8 | vLLM | DeepGEMM when enabled (family 120 admitted), else CUTLASS | `VLLM_USE_DEEP_GEMM` | silent corruption at K = 2560 in an N/M window [Community-measured] (P16 §5 lever 6) |
| Capability helpers | SGLang | `is_sm120_supported` = major 12; `is_sm120()` = exactly 12.0 | — | guards on `is_sm120()` or `== 120` skip GB10 [Code-verified] (P17 §7) |
| QSA sparse decode | SGLang | SM121 Triton varlen kernel; TRT-LLM route excluded by name | — | widening the guard: token-0 output (2 of 4 own; 32/32 community at 120K-210K) [Measured; Community-measured] (P16 §7) |
| MMVQ decode | llama.cpp | GB10 table and L2 prefetch, compiled only when the build targets `121a` | build arch | lost silently without 121 [Code-verified] (P05 §2) |
| EXL3 decode | vLLM + EXL3 kits | GEMV m ≤ 8; coop MoE ≤ 8 rows; INT8 GEMV mode 2 (untuned on GB10) | kit env | mode 2 failed partition equivalence (~0.7-0.8% rel L2) [Measured] (P05 §2) |

No-op env vars on current vLLM: `VLLM_NVFP4_GEMM_BACKEND`, `VLLM_USE_FLASHINFER_MOE_FP4`, `VLLM_FLASHINFER_MOE_BACKEND`, `VLLM_ATTENTION_BACKEND`, `VLLM_MXFP4_BACKEND`, `VLLM_TORCH_PROFILER_DIR` [Code-verified] (P16 §7; P14 §7). Proof of dispatch is the engine's selection line on every rank, never the quantization name or a trace kernel name.

### 6.3 Designs SM121 cannot run

| design | reason / failure | grade |
|---|---|---|
| `tcgen05`/TMEM/`wgmma`/2-SM kernels (FA3, FA4, CuTe-DSL NVFP4, TRT-LLM MoE/FMHA) | absent instructions; family-100 gates, never auto-selected | [Code-verified] (P05 §3) |
| Tiles > 101,376 B | SGLang extend attention 106,496 B; Triton MoE 147,456 B; tilelang DSA 169,984 B; Triton MLA FP8-KV decode 102,400 B; FilteredTopK needs ≥ 131,072 B per SM | [Community-measured] (P17 §7; P12 §3) |
| Cluster-launch cooperative top-k | rejected on SM12x, 51/51 cases fail | [Community-measured] (P12 §3) |
| GDR collectives, NVLink fusions | 5.1 | [Code-verified] |
| Sparse FP4 in a `12.0f` build | `sm_121a`-only | [Code-verified] (P17 §3) |
| FP4 tensor cores as a decode speedup | an ideal FP4 GEMM is 1.03-1.07× slower than Marlin W4A16 up to M = 256 | [Community-measured] (P05 §3) |

Toolchain: CUDA ≥ 12.9; `-gencode=arch=compute_121a,code=sm_121a` (`-arch=sm_121a` can drop the `a`); a vLLM CUDA 13.x `TORCH_CUDA_ARCH_LIST=12.1a` request yields `12.0f`; Triton at current pins compiles `sm_121a` through `ptxas-blackwell`, which `TRITON_PTXAS_PATH` does not govern; verify with `cuobjdump --list-elf` on every deployed `.so` and JIT cache [Code-verified; Community-measured] (P17 §3, §7).

### 6.4 When authoring is justified

```
T_new / T_old = 1 − s · (1 − 1/x)          s = kernel share of end-to-end time, x = kernel speedup     (Amdahl)
```
A 2.76× GDN prefill kernel moved 8K TTFT ~3%; 20% on a family at 61.9% of GPU time is ~12% end to end [Community-measured] (P15 §5 lever 6).

Procedure (stop at the first step that answers):
1. Audit what runs: `cuobjdump --list-elf`, selection lines, exact-CC gates. Missing cubins or a trapping gate: fix dispatch (13.3 → 48.6 tok/s from shipping SM121 cubins on one request) [Community-measured] (P15 §5 lever 1).
2. Measure the ceiling and classify (7.6 steps 1-2; 6.1 stop rule): in-engine time within 10% of `bytes / BW_eff` ends the search (lever 2).
3. Adopt a proven upstream or kit kernel through gates (lever 3).
4. Tune data, not code (GB10 config files for Triton/MoE shapes) (lever 4).
5. Profile: `ncu` on a component harness, off traffic, showing the kernel far below both bandwidth and compute ceilings, with `s` large enough that `x = 2` moves the metric beyond the lane's noise band (lever 6).
6. Author only then; bandwidth-bound MoE decode at the wall is not a target (lever 9).
7. Language: CUDA C++/CUTLASS (SM120 atoms, cluster 1×1×1), CuTe DSL, Triton, TileLang (P15 §3). Add `__launch_bounds__`: at 512 threads, 63 registers is exactly two blocks per SM; +2 registers halves residency with "0 spills" still logged [Measured] (P15 §5 lever 5).
8. Gate: reference test on real shapes, compute-sanitizer, then unprofiled end-to-end A/B/A; an agent-written decoder won on components while its combined arm (startup/prefill group plus the decoder pair; decoder not isolated) lost end to end [Measured] (P15 §5 lever 8).

---

## 7. Measurement

### 7.1 Noise bands

| band | value | grade | source |
|---|---|---|---|
| Own decode drift, opening vs closing baseline | coding −1%, prose +4% | [Measured] | 2026-09-20 GLM 5.3 Flash V3 retained startup and prefill improvements |
| In-boot run-to-run | < 2% | [Community-measured] | P14 §4 |
| Boot-to-boot, identical config | ~8%; +11%; C8 92-111 tok/s from an autotune lottery (±1.6% with autotune off at ~2% cost) | [Community-measured] | P14 §4 |
| Single run | ~6.5-9% | [Community-measured] | P02 §7; P05 §6 |
| Prefill | ±2% within a boot; ±20% start-to-start on one stack | [Community-measured] | P04 §6 |
| This lane | read from the lane card: noise band [Proposed field] | — | — |

Decision rule: a gain inside the band is no gain; a decode effect below ~8% needs repeated boots per arm (P14 §6).

### 7.2 Artifacts that fooled past sessions

| artifact | effect | guard |
|---|---|---|
| Quick probe (one short prompt, no warm-up, no nonce) | +38% claimed, withdrawn [Measured] (2026-09-23 verification) | frozen harness only |
| Structured/counting/filler fixtures | inflate speculative decode 1.5-4×; a k=7 setting rested on one; filler measured a drafter's worst case (0.23) [Measured; Historical diagnostic] (P05 §7; WR) | real-shaped fixtures per class; structured reported apart |
| SSE chunks as tokens; one reasoning field parsed | 2.2-5× under-count by engine and drafter; 97 vs 19.9 tok/s [Community-measured] (P02 §7; P05 §7; P14 §4, §7) | `usage.completion_tokens`; parse `reasoning` and `reasoning_content` |
| First request after boot, clock change or idle | 813 vs 1,375 tok/s cold 8K; 24.6 vs ~34 after a new lock [Measured] (P14 §4) | discard one warm-up that is not the measured prompt |
| Reused or nested prompts; outside traffic | cold prefill overstated ~50×; a pass overlapped ~90 outside requests [Community-measured; Measured] (P14 §4, §7) | nonce at token 0, `cached_tokens == 0`; poll running requests |
| Dashboards and engine counters | 353 shown vs 1,490 actual; cache hits counted as prefill; a counter peak of 110 was aggregate vs single median 44 [Historical diagnostic; Measured] (WR; P05 §7) | one direct API request; label per-stream vs aggregate |
| Labels and summaries | a "C4" soak ran sequentially; "decode flat at every cap" while C1 means fell ~6% [Historical diagnostic] (WR) | assert concurrency from the engine; state number and band |
| Profiled runs; old baselines | +51% TTFT under the torch profiler [Measured] (P14 §4); "~47% above historical" on another digest [Historical diagnostic] (WR) | never score; same-day baseline, same digest |

### 7.3 Protocol

1. Prediction first: predicted value and tolerance for each metric the change should move (2.8, 3.3, 4.6).
2. Clean window: poll running requests and shim in-flight before and during each pass; discard overlapped passes.
3. Same cap on every rank, median `clocks.gr` under load, no latched rank (5.3 #11).
4. Restart between arms with the pull skipped; one variable (or one declared axis group, section 8) per boot; one compile cache per arm.
5. Greedy top-1 logprob probe after every relaunch; failure voids the boot.
6. Warm every serving shape; discard the first request.
7. Cold cells: nonce at token 0, `cached_tokens == 0` from usage; forced length checked against `completion_tokens`; thinking mode fixed.
8. Frozen harness: ≥ 3 reps (index), ≥ 5 matched repeats when the expected effect < 10% (P05 §6 pass gate; P19 §6 step 5), closing baseline (A/B/A), per-run values kept; a decode effect below ~8% needs repeated boots per arm (P14 §6).
9. `MemAvailable` every 1-2 s per rank, NV_ERR/Xid from the journal; stop traffic at 4 GiB.
10. Correctness gates for the change class before any speed number (P21).
11. Record the prediction error per metric.

### 7.4 Per-stream versus aggregate

```
per-stream = completion_tokens / (t_last_token − t_first_token)          agg = Σ completion_tokens / wall window
effective completion = completion_tokens / (TTFT + decode time)
```
- C-labels mean concurrent requests; confirm from the engine's running count.
- Report side by side: a bundle raised C8 agg 76.4 → 123.1 while C8 per-stream fell 21.6 → 17.8 and TTFT fell 10.63 → 0.52 s [Measured] (P05 §4).
- vLLM credits prompt tokens only when a prefill completes; flat counters during a long prefill are not a hang [Community-measured] (P04 §7).

### 7.5 Normalizing outside claims

Checklist before placing an outside number beside an own one:
1. Hardware class (GB10 vs SM120 vs datacenter), node count, fabric.
2. Engine, commit, kit, image; quant and which layers stay BF16.
3. Slots, concurrency, output cap, `ignore_eos`, temperature, thinking mode.
4. Fixture class, prompt depth, cache state.
5. Metric (per-stream vs aggregate; SSE vs usage; wall vs decode).
6. Reported vs reproduced; n, spread, grade on the number. Never put an unverified outside number in a sentence with an own [Measured] number without both grades (it drew the operator's rebuke) [Historical diagnostic] (WR).
7. Re-baseline on the author's protocol on the own lane before tuning toward the number (own fixtures read ~5 tok/s below one author's protocol on the same lane) [Historical diagnostic] (WR).

Comparability rule [Proposed, from requirements §2 #2]:

| same image digest | same harness checksum | same fixture and sampling | same clock cap | verdict |
|---|---|---|---|---|
| yes | yes | yes | yes | comparable; no control rerun |
| any no in the first three | | | | run a same-day control arm |
| yes | yes | yes | no | decode comparable only within the P13 band; control for prefill |

### 7.6 Profiling as the hypothesis source

Order, cheapest first (P14 §6):
1. Zero-overhead signals: counters, selection lines, scheduled tokens per step, acceptance, graph bucket, cache hits.
2. Unprofiled component timing per rank (CUDA events/graph replay; ≥ 25 samples, ≥ 200 near a dispatch threshold; never mix ranks).
3. Nsight Systems in a dedicated profiling boot (attach does not work):
   `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --trace-fork-before-exec=true --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=repeat -o <file per rank> <engine command>` with the engine's window (vLLM `--profiler-config.profiler cuda` + `delay_iterations`/`max_iterations`; SGLang `/start_profile` with `CUDA_PROFILER`, `num_steps` 5 prefill / 25 decode). Containers: `SYS_PTRACE`, `ipc: host`, unlimited memlock, `nsys status --environment` first. UM tracing is unsupported on GB10 [Community-measured]; overhead on GB10 unmeasured.
4. Nsight Compute only on a component harness, off traffic: external `-lgc` lock plus `ncu --clock-control none`; first `--list-chips`, `--query-metrics`, `--list-sets`, recording `n/a` names; then SOL, memory, occupancy, warp-state, `dram__bytes_read.sum`, `sm__pipe_tensor_cycles_active` (the warp-level MMA pipe; the `tc` pipe counts UTCMMA work, which SM12x does not execute [Interpretation on a vendor statement]). The permission module option needs a reboot and so operator permission; else use sudo. NCCL kernels need `--communicator tcp --lockstep-kernel-launch` on every rank [Code-verified; Community-measured].
5. SASS/resources of the deployed binary: `cuobjdump -lelf`, `-res-usage`, STL/LDL counts.
6. Ceilings in a maintenance window: GEMV/STREAM probe per node, nccl-tests 8 KiB-32 MiB, `ib_write_bw`/`ib_write_lat`.
7. Classify with the P14 §6 table (bandwidth / compute / latency / launch / host-sync / rank-wait / comm); confirm by a second method (clock sensitivity, component timing, ceiling); decide only by unprofiled A/B/A.

Record so far: 0 nsys/ncu captures in 161 changelogs [Historical diagnostic] (WR); a profile-suggested own fix vanished end to end (lookup cache 10-40× faster locally, +0.41 / +0.45 / −2.08% end to end on three prompts) [Measured], and community trace-suggested fixes lost 13-61% [Community-measured] (P14 §5 lever 1, §7).

---

## 8. The loop

1. **Entry.**
   a. Read the lane card: Current recipe and contract; What has been measured on this lane; Levers not yet tried; Last incident; Owner notes; settled experiments (read from the lane card: settled experiments [Proposed section]; until it exists, the measured table and the private ledger status). Settled-row schema [Proposed]: lever; conditions measured under (drafter layout, build, topology, clock); result with baseline; packet; retest trigger. A result holds only for its build.
   b. Verify the card against the live lane registry (`current`, `active`, contract); the registry wins. A stale memory note once named the wrong production topology [Historical diagnostic] (WR).
   c. Read the build on the node: image digest, engine commit, kernel-library versions, clock cap.
   d. Delegating? Paste the guardrails below into the worker brief.
2. **Ceiling and bound.** Decode floor (3.3), prefill model (4.1), per-rank budget (2.8), request envelope (4.6), turn-time share per phase for the workload (4.7). Rank candidates by Amdahl bound (6.4) under the operator's weights (read from the lane card: Owner notes).
3. **Hypothesis.** From a profile or the cheapest discriminating measurement (7.6), not from an upstream recipe. Read the maintainer's issues and PRs at a pinned SHA before theorizing (skipping this cost ~4 h of wrong cache theories; the upstream fix took reuse 0 → 85.8%) [Historical diagnostic] (WR). Re-justify every inherited flag. Screen against settled experiments, 6.2-6.3, and the fit table. Each candidate carries a falsifiable numeric prediction with tolerance.
4. **Change.**
   - Memory knobs: one per boot (2.5).
   - Other axes: batch by axis (startup, prefill, decode, capacity) with a control arm; keep or revert by group; isolate a component only when a group regresses and the decision depends on it [Interpretation; operator practice] (requirements §3 #20). A bundle is recorded as a bundle; a lever's share comes only from its own arm (index).
   - Agree the downtime budget and boot-cost table first (read from the lane card: boot time [Proposed field]).
   - Lifecycle checklist: pull skipped; pinned digest; frozen-recipe hash; drain marker cleared and verified; staging symlinks resolve inside the container; stop patterns by exact name; no `pkill -f` self-match; long cutovers detached; registry `active` updated after a manual start; image and disk on every rank; liveness by progress counters (requirements §2 #10).
   - After any model or alias swap: verify served name → shim alias → client ID with a real POST (requirements §3 #28).
   - Reasoning-path changes: prove the reasoning state from the rendered prompt and output for every effort word the client sends; check `reasoning`/`reasoning_content` survive the shim; check small `max_tokens` is not consumed by reasoning (requirements §2 #9).
5. **Measure.** 7.3; boot receipts against the prediction sheet first (2.8 step 6).
6. **Decide.** Correctness first; floor held; envelope and cache gates re-checked; keep only a gain beyond the noise band on the operator-weighted classes, with C8 aggregate not worse unless the trade was accepted in advance.
7. **Record.** Packet with prediction, measurement and error per metric, cost fields (wall time, boots, outages), negatives; update the lane card (measured table, settled experiments, constants) and the topic page §4 and ledger row. Never report a write that did not execute [Historical diagnostic] (WR).

Delegation guardrails (paste into every worker brief):
- An unattended or overnight worker never changes memory knobs, stops at any floor breach, and reports.
- No auto-teardown supervisors (one killed baseline boots twice) [Historical diagnostic] (WR).
- No production interruption, deletion, gateway restart, reboot or private-data export without the operator's explicit go; an availability statement is not a go.
- Never send private configuration, topology or results to outside services.

---

## 9. Failure catalog (short form; full catalog in stage 2)

If you believe X, you will do Y, and Z happens.

| # | belief X | action Y | outcome Z | grade | source |
|---|---|---|---|---|---|
| 1 | Freed memory can be spent in the same boot | right-size + 4 GiB pin, 4.4 GiB free | node unreachable ~7 h, physical reboot | [Measured] | P11 §7 |
| 2 | Lowering SGLang's static fraction frees RAM | 0.90 or 0.85 on a weight-bound lane | 7.41 − 10.89 < 0: no pool; advice retracted | [Measured]; [Interpretation] | P11 §7 |
| 3 | A KV-pool cap bounds a request | 262K at chunk 1024 with ~1.4-2.9 GiB warm | ~200K turn wedged the TP3 group at 25-28% pool use | [Historical diagnostic] | P04 §7 |
| 4 | NV_ERR in a completed prefill is harmless | pass a 131K gate with 104 NV_ERR, keep serving | next ~200K turn wedged | [Measured] | P11 §7 |
| 5 | Swap is headroom | enable swap as protection | 0 MiB used during 104 NV_ERR; host swapped out `sshd` in another wedge | [Measured] | P11 §5 lever 10 |
| 6 | An explicit vLLM pin forces a context length | pin 12.5-15 GiB for 800K-900K | 4 of 4 pins failed, as did the series' one auto-KV launch (kernel OOM at multimodal warm-up, or KV validation) | [Measured] | P11 §4, §7 |
| 7 | SGLang default 0.90 is safe on one node | 1.64M-token pool for a 1-slot 262K lane | ~6.9 GiB post-capture, userspace wedge; a token cap raised free 6.7 → 29 GiB | [Measured] | P11 §4; P01 §7 |
| 8 | Runtime YaRN reaches the drafter | pass overrides at launch | 0.00 acceptance on ~245K live traffic, 3.00/3 on probes | [Measured] | P02 §7 |
| 9 | Deeper k is faster | keep k=7 from a structured fixture | k=3 +19-31% on every real-shaped cell | [Measured] | P02 §4 |
| 10 | A quick probe is a result | headline a +38% single-prompt probe | withdrawn against the frozen harness | [Measured] | P05 §7 |
| 11 | Smaller chunks are a free precaution | drop 1024 → 256 silently | cold prefill halved; needed chunk never measured | [Historical diagnostic] | WR |
| 12 | One `NCCL_IB_HCA` entry is the whole port | name one function per port | link at half for weeks, no warning | [Measured] | P03 §7 |
| 13 | A GID index survives reboot | one shared GID index | index all-zero on one rank after reboot | [Measured] | P03 §7 |
| 14 | The control plane runs on the fabric | skip the master-address check | control plane on Wi-Fi, 156 ms average, power save on | [Historical diagnostic] | WR |
| 15 | Containers "Up" means healthy | judge liveness by container status | dead TP group with both containers Up | [Measured] | P11 §7 |
| 16 | Telemetry proves no reboot happened | report "no reboot" | contradicted by the operator's direct account | [Historical diagnostic] | WR |
| 17 | Extra model IDs in the client picker are harmless | wire several IDs to one lane | prefix caching destroyed; rejected by the operator | [Historical diagnostic] | WR |
| 18 | Widening the SM121 TRT-LLM QSA guard is a speedup | enable it | token-0 loops in 2 of 4 coding outputs | [Measured] | P16 §7 |

---

## 10. Where to look

### 10.1 Pointer key

| key | page |
|---|---|
| P01 | [CUDA graphs and launch overhead](../01-cuda-graphs-and-launch-overhead.md) |
| P02 | [Speculative decoding](../02-speculative-decoding.md) |
| P03 | [Inter-node communication over DAC](../03-inter-spark-communication.md) |
| P04 | [Prefill optimization](../04-prefill-optimization.md) |
| P05 | [Decode optimization: the bytes model](../05-decode-optimization.md) |
| P06 | [Attention backends on SM121](../06-attention-backends-sm121.md) |
| P07 | [Linear attention and hybrid layers](../07-linear-attention-and-hybrid-layers.md) |
| P08 | [MoE dispatch](../08-moe-dispatch.md) |
| P09 | [GEMM backends and quant formats](../09-gemm-backends-and-quant-formats.md) |
| P10 | [KV cache and prefix caching](../10-kv-cache-and-prefix-caching.md) |
| P11 | [Memory on unified memory](../11-memory-on-unified-memory.md) |
| P12 | [Sampling and the lm_head](../12-sampling-and-lm-head.md) |
| P13 | [Clocks, thermal, power](../13-clocks-thermal-power.md) |
| P14 | [Profiling protocol](../14-profiling-protocol.md) |
| P15 | [Kernel authoring on SM121](../15-kernel-authoring-sm121.md) |
| P16 | [Engine dispatch maps](../16-engine-dispatch-maps.md) |
| P17 | [SM121 hardware facts](../17-sm121-hardware-facts.md) |
| P18 | [Case study: MiaAI-Lab recipes](../18-case-study-miaai-lab-recipes.md) |
| P19 | [Case study: community recipe authors](../19-case-study-community-recipe-authors.md) |
| P20 | [Model architecture cards](../20-model-architecture-cards.md) |
| P21 | [Quality and correctness gates](../21-quality-and-correctness-gates.md) |
| P22 | [Serving agent workloads](../22-serving-agent-workloads.md) |
| index, ledger | [GB10 Inference Wiki](../README.md) (entry rules, symptom router); [Levers ledger](../ledger.md) |

### 10.2 Section → evidence → lane card fields

| core section | evidence pages | lane card fields (existing) |
|---|---|---|
| 1 Machine | P17, P13, P03 §3, P05 §3 | host; clock cap |
| 2 Memory | P11, P10 §3, P01 §3-§4, P04 §3, P20 §4.3, §6 | model / weights; KV pin; mem fraction; chunk size; context; slots; spec method and k; drafter placement; NCCL env; host; Last incident |
| 3 Decode | P05, P02, P12, P08, P01, P20 §4.1 | model / weights; spec method and k; slots; engine version |
| 4 Prefill | P04, P10, P22, P07, P09 | chunk size (cache page, LPT, fair prefill); context |
| 5 Multi-node | P03, P20 §4.3, P13 §3, P08 | NCCL env; host; clock cap |
| 6 Kernels | P16, P17, P15, P06, P09, P08 | engine version; FlashInfer version; image digest |
| 7 Measurement | P14, P05 §6-§7, P02 §6-§7, P19, P21 | What has been measured on this lane |
| 8 Loop | index, P14 §6, P20 §6, P21, P22 | all sections; Owner notes; Levers not yet tried |
| 9 Failures | P11 §7, P04 §7, P02 §7, P03 §7, P16 §7, WR | Last incident; Owner notes |

Fields this core reads that the template lacks: Gaps #12.

---

## Gaps

Where the core uses one of these terms it carries a contested grade, an anchor labelled as one lane's value, or a [Proposed] field; treat each as [Proposed: needs measurement].
1. No own device dump: blocks per SM (24 vs 32), copy engines, empty-kernel launch latency, graph replay cost (P17 §8; 3.4).
2. No own bandwidth ceiling (`bandwidthTest`, STREAM, GEMV); `BW_eff` is community-measured and probe-dependent.
3. Tensor-core peak contested (BF16 ≈123 vs ≈213); every MFU figure inherits it (4.2).
4. No DRAM byte counter for any GB10 decode kernel; no routing histogram, so `E_r(M)` is an upper bound (3.2-3.3); NVFP4 dense shortfall unexplained.
5. No own per-collective latency on the triangle; Tree vs Ring untried; no own per-rank NCCL timeline (5.1).
6. `c_lane` known only as one unpublished community constant (≈14 B); no own peak trace over chunk × prefix (4.6).
7. KV pin vs fraction unresolved; the 2.3 test has never run.
8. `r(k, C)` for multi-node MoE only derived from tok/s and acceptance; no forced-k windows in one boot (3.5).
9. Clock-cap decode cost on TP lanes unsettled (−6.8% n=3 vs −4.9% to −6.4% n=5 inside spread).
10. Nsight overhead on GB10 and which `ncu` metrics return values on SM121 unknown (7.6).
11. Unmeasured memory terms: speculative verify-buffer peaks at k = 3/5/7; NCCL pinned bytes on own nodes; a pool filled to ≥ 95% (P11 §8).
12. Lane-card template lacks fields the core reads: host baseline, `KV_bpt`, `c_lane`, graph size, warm floor, noise band, bytes-bound ceiling, boot time, settled experiments, per-node GID/rail/control-plane state, reasoning wire contract, served-name → alias → client-ID map.
13. Operator weights and the live workload profile are operator state; the turn-time break-even (≈1K output per 27K new prompt tokens at coding rates; ≈17K prose, ≈6K structured) exists for one lane only.
14. Records disagree whether only the head or all three nodes wedged on 2026-09-16 (P11 §4); the core states only that the TP group stalled.
15. Requirements left to chapters or stage 2 (no wiki source, or too large for the core): token-ID submission bypassing vision; `last_gen_throughput` averaging and a full metric glossary; the handoff template and refresh-loop health signal; `origin`/`rigor` enums and retract-in-place; the abliteration and quant cost table; honest external bands by workload class; the contradiction register as one table.

## Sources

Wiki pages (fetched 2026-09-23): P01-P05, P07, P08, P10-P17, P20-P22, index, contributing read in full or by section; P06, P09, P18, P19 through cross-references.

Own packets (date and title): 2026-08-16 clock-cap A/B, DeepSeek V4 Flash TP2; 2026-08-17 TP1 vs TP2 identical battery; 2026-08-19 single-node DFlash2 campaign audit; 2026-08-21 FP8 MTP sweep; 2026-08-21 k=10 boot-gated qualification; 2026-08-27 GLM 5.3 Flash MTP-4 versus MTP-off; 2026-08-28 GLM 5.3 Flash EXL3 TP2 500K stabilization; 2026-08-30 metered traffic; 2026-08-30 GLM TP2 composite 1M activation; 2026-08-31 vLLM MTP depth sweep; 2026-09-03 A1 k=3 / k=7 qualification; 2026-09-05 FP8-KV qualification; 2026-09-08 GLM 5.3 Flash EXL3 E4 five-arm A/B; 2026-09-13 DS V4.1 Flash TP3 qualification; 2026-09-15 DS V4.1 Flash TP3 k3 and k5 scorecards; 2026-09-15/16 GLM TP3 upgrade A/B; 2026-09-16 DS V4.1 TP3 memory-ceiling and head-wedge records; 2026-09-16 combined runtime qualification; 2026-09-16 DS V4.1 EXL3 TP3 optimized-1M; 2026-09-20 GLM TP3 startup and prefill optimization; 2026-09-20 GLM 5.3 Flash V3 retained improvements; 2026-09-21 GPU clock-cap sweep, GLM 5.3 Flash EXL3 TP3; 2026-09-23 verification (quick-probe withdrawal).

Workflow review (WR; private inputs, restated export-safe): the 2026-09-23 reviews of two agent CLIs' sessions and of 161 changelogs, and the requirements synthesis (28 gaps, 29 requirements).
