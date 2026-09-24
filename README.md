# GB10 Inference Wiki

A source-checked knowledge base for running open-weight LLMs fast and correctly on NVIDIA GB10 hardware (DGX Spark, ASUS Ascent GX10 and other 128 GB unified-memory boxes), on one node or on two or three nodes joined by ConnectX-7 RoCE cables. It is written for the engineers and coding agents who tune serving lanes, not as a tutorial. **Start with the symptom router below**, then read the rules before you touch a lane.

**What is here.** 22 topic pages, each in two parts: the knowledge (mechanism, how each engine does it, what differs on SM121, what has been measured) and the protocol (how to change it on a serving lane one variable at a time, with gates and rollback). Plus a [levers ledger](ledger.md) (one row per lever, with gain band, evidence grade, cost and risk), a [graded registry](registry.md) of about 1,400 public GB10 resources, and [how pages are written](CONTRIBUTING.md).

**How to read a number.** Every claim carries an evidence grade in square brackets: `[Measured]` on the maintainer's fleet of three ASUS Ascent GX10 nodes over direct RoCE cables; `[Community-measured]` by a named public author; `[Code-verified]` read from pinned source; `[Historical diagnostic]` recovered from the maintainer's own incident records; `[Interpretation]` derived, not measured; `[Proposed]` untested. Nothing here is a vendor spec presented as achieved. Nodes are written node-1, node-2, node-3; machine-specific details (hosts, paths, recipe names) are omitted.

**How it was built.** The pages were researched and written by AI agents (Claude Opus) in a pipeline: mine the maintainer's own measurement packets, scout public sources, research, adversarially verify every claim against its source, then write. About 976 claims were confirmed, 371 corrected and 13 refuted in verification, and a whole-wiki consistency review and a pre-publication review followed. The maintainer set the rules and ran or approved every measured change on the fleet; the pages themselves are agent-written and agent-reviewed, not line-edited by a human. Mistakes remain possible; corrections with a source are welcome (see the end of this page).

---

## Before you touch a lane

These rules appear in the preconditions of the topic protocols. A protocol may add stricter lane-specific rules, but it never relaxes these.

Memory
- Need at least 8 GiB MemAvailable on every node before a start, and keep a 5 GiB warm floor on every node, measured after the longest qualified prefill, not at boot.
- Raise a KV pin by at most 1.5 GiB per boot and change only one memory variable per boot. Memory that one change frees is never spent in the same boot. Swap is not headroom.
- Derive memory settings on this fleet. Never copy another operator's memory fraction, KV pin or absolute floor, because only the delta a change produces carries over between fleets.
- Stop benchmark traffic when any node's MemAvailable drops below 4 GiB. Keep the engines running; do not kill them.
- Sample MemAvailable every 1-2 s on every node through the longest prompt, and count NV_ERR_NO_MEMORY and Xid lines from the kernel journal. Any such line during the gated workload fails the arm.

Experiment design
- One variable per boot. Record a bundle as a bundle, and claim a lever's share only from its own one-variable arm.
- A restart-based A/B skips the image pull and gives each arm its own compile and Triton cache, so the arms differ only in the variable.
- Measure in a clean window. Poll running requests before each pass, and discard any pass that overlaps outside traffic; do not average it in.
- Prove the change ran by reading the engine's selection or configuration line on every rank. A boot without that line is a boot where the change did not run.
- Read the build on the node, never from an old packet or a public branch head: image digest, engine commit, kernel-library versions and clock-cap state.
- Keep every rank of a TP group at the same clock cap, and check for a rank latched at a low clock before any multi-node run.

Measurement
- After every relaunch, run a greedy logprob sanity probe before recording any other number. A failed probe voids the boot.
- Discard the first request after a boot, a clock change or an idle gap, and warm every serving shape before timing.
- Start every cold prompt with a unique nonce at token 0 and assert zero cached tokens. Count tokens from the usage block, never from stream chunks.
- Decide on the frozen harness: at least 3 repetitions, with a closing baseline (A/B/A). A quick probe or a single run never promotes a change, and a gain inside run-to-run spread is no gain.
- Correctness comes before speed. Pass the gates that [Quality and correctness gates](21-quality-and-correctness-gates.md) requires for the change class before you quote any speed.
- Profiled runs locate a bottleneck but are never scored.
- Record every measured change as a packet: baseline, workload, scope, image digest, config hashes, and MemAvailable before and after. Keep negative results.

Operations
- Never restart an agent gateway to test an engine change. Make the change, verify it, and leave the restart decision to the operator.
- Never reboot or power-cycle a node without the operator's explicit permission for that node.
- Never `pkill -f` a pattern that appears in your own command line. Use pid files.
- Keep the previous recipe launchable. Promote and roll back through the lane's promotion mechanism, never by editing a running container.

## Symptom to topic

Each row points at pages whose levers (§5) or protocol (§6) address the symptom. Open the first page listed first.

Prefill, TTFT and serving
| Symptom | Start here |
|---|---|
| A cold long prompt takes minutes to reach the first token | [Prefill optimization](04-prefill-optimization.md), then [Profiling protocol](14-profiling-protocol.md) |
| A follow-up turn re-reads the whole prompt (prefix-cache miss) | [KV cache and prefix caching](10-kv-cache-and-prefix-caching.md), [Serving agent workloads](22-serving-agent-workloads.md) |
| On a hybrid model, cache hits vanish at some depths or once speculation is on | [Linear attention and hybrid layers](07-linear-attention-and-hybrid-layers.md), [KV cache and prefix caching](10-kv-cache-and-prefix-caching.md) |
| A newcomer's long prompt freezes other users' decode, or newcomers queue behind it | [Serving agent workloads](22-serving-agent-workloads.md), [Prefill optimization](04-prefill-optimization.md) |
| Many agents on one lane: TTFT under load, slot count, timeouts and retries | [Serving agent workloads](22-serving-agent-workloads.md) |

Decode
| Symptom | Start here |
|---|---|
| Single-stream decode stays flat no matter what you change | [Decode optimization](05-decode-optimization.md), [Speculative decoding](02-speculative-decoding.md) |
| Speculative decoding is not paying off, or acceptance is low | [Speculative decoding](02-speculative-decoding.md), [Sampling and the lm_head](12-sampling-and-lm-head.md) |
| Acceptance or tok/s jumped suspiciously after a head, quant or drafter change | [Quality and correctness gates](21-quality-and-correctness-gates.md), [Sampling and the lm_head](12-sampling-and-lm-head.md) |
| Graph capture fails, or decode steps fall back to eager after a config change | [CUDA graphs and launch overhead](01-cuda-graphs-and-launch-overhead.md) |
| The MoE layer is slow, or the engine picked a fallback MoE kernel | [MoE dispatch](08-moe-dispatch.md), [GEMM backends and quant formats](09-gemm-backends-and-quant-formats.md) |
| Faster in a micro-benchmark, unchanged end to end | [Profiling protocol](14-profiling-protocol.md), [Kernel authoring on SM121](15-kernel-authoring-sm121.md) |

Multi-node
| Symptom | Start here |
|---|---|
| Aggregate does not scale across nodes, and per-step time is dominated by waits | [Inter-node communication](03-inter-spark-communication.md) |
| Choosing a TP width: one, two or three nodes | [Inter-node communication](03-inter-spark-communication.md), [Model architecture cards](20-model-architecture-cards.md) |
| NCCL init fails, falls back to sockets, or throws a size-mismatch error at boot | [Inter-node communication](03-inter-spark-communication.md) |
| One rank is slower than the others and paces the whole TP group | [Clocks, thermal, power](13-clocks-thermal-power.md), [Profiling protocol](14-profiling-protocol.md) |

Memory, boot and stability
| Symptom | Start here |
|---|---|
| A model will not fit, or it fits with no room left for the KV cache | [Model architecture cards](20-model-architecture-cards.md), [Memory on unified memory](11-memory-on-unified-memory.md), [MoE dispatch](08-moe-dispatch.md) |
| KV capacity varies between identical boots | [Memory on unified memory](11-memory-on-unified-memory.md), [Profiling protocol](14-profiling-protocol.md) (lever 11, boot identity as a variable) |
| OOM, driver allocation errors, or a node wedge during a long prefill | [Memory on unified memory](11-memory-on-unified-memory.md), [Prefill optimization](04-prefill-optimization.md) |
| High temperatures, throttling, power clamps, or a node that powers off without a log | [Clocks, thermal, power](13-clocks-thermal-power.md) |
| Deciding whether to cap the GPU clock | [Clocks, thermal, power](13-clocks-thermal-power.md) |
| Boot is slow: minutes to API ready, or first-use compiles | [CUDA graphs and launch overhead](01-cuda-graphs-and-launch-overhead.md) (compile and caches), [MiaAI-Lab recipes](18-case-study-miaai-lab-recipes.md) (lever 12, weight loader), [Engine dispatch maps](16-engine-dispatch-maps.md), [Attention backends](06-attention-backends-sm121.md) (JIT) |

Correctness and quality
| Symptom | Start here |
|---|---|
| Garbage output, token-0 loops or NaN appear only past some prompt length | [Attention backends](06-attention-backends-sm121.md), [Linear attention and hybrid layers](07-linear-attention-and-hybrid-layers.md), [Quality and correctness gates](21-quality-and-correctness-gates.md) (Step 0 checks NaN/inf in logprobs) |
| Quality regressed after a speedup (quant, speculative decoding, FP8 KV, a kernel) | [Quality and correctness gates](21-quality-and-correctness-gates.md) |
| Answers differ at C8 from serial runs, or vary between runs at temperature 0 | [Quality and correctness gates](21-quality-and-correctness-gates.md) (lever 4), [Speculative decoding](02-speculative-decoding.md), [Sampling and the lm_head](12-sampling-and-lm-head.md) (sampling defaults, indexer determinism) |
| Strict JSON or tool calls fail on a reasoning model | [Sampling and the lm_head](12-sampling-and-lm-head.md), [Quality and correctness gates](21-quality-and-correctness-gates.md) |
| Choosing a quantization format | [GEMM backends and quant formats](09-gemm-backends-and-quant-formats.md), [Model architecture cards](20-model-architecture-cards.md), [Quality and correctness gates](21-quality-and-correctness-gates.md) |

Build and dispatch
| Symptom | Start here |
|---|---|
| A regression after an engine or library upgrade | [Engine dispatch maps](16-engine-dispatch-maps.md), [Attention backends](06-attention-backends-sm121.md) |
| You need to know which kernel is actually running | [Engine dispatch maps](16-engine-dispatch-maps.md) |
| A kernel refuses to load, or a feature is silently missing (wrong arch in the build) | [SM121 hardware facts](17-sm121-hardware-facts.md), [Kernel authoring on SM121](15-kernel-authoring-sm121.md), [Engine dispatch maps](16-engine-dispatch-maps.md) |
| You are considering a custom kernel | [Profiling protocol](14-profiling-protocol.md) first, then [Kernel authoring on SM121](15-kernel-authoring-sm121.md) |

Measurement and outside claims
| Symptom | Start here |
|---|---|
| The profiler shows nothing, or the trace is empty or partial | [Profiling protocol](14-profiling-protocol.md), [CUDA graphs and launch overhead](01-cuda-graphs-and-launch-overhead.md) |
| Numbers look too good, or two tools disagree (stream chunks, first request, cache hits) | [Profiling protocol](14-profiling-protocol.md), [Decode optimization](05-decode-optimization.md) |
| Decode speed varies from boot to boot at a fixed config and clock | [Profiling protocol](14-profiling-protocol.md) (lever 11), [Clocks, thermal, power](13-clocks-thermal-power.md) (lever 2, per-rank health), [Community recipe authors](19-case-study-community-recipe-authors.md) (lever 11) |
| A published recipe's number does not reproduce here | [Community recipe authors](19-case-study-community-recipe-authors.md), [MiaAI-Lab recipes](18-case-study-miaai-lab-recipes.md) |
| Bringing up a new model family or a new quant of a known family | [Model architecture cards](20-model-architecture-cards.md) |

## Pages

Foundations
- [SM121 hardware facts](17-sm121-hardware-facts.md): what a GB10 actually is. Compute capability 12.1 with no datacenter tensor-memory instructions, 99 KB of shared memory per block, 48 SMs and shared LPDDR5X that falls well short of its nominal bandwidth in practice.
- [Engine dispatch maps](16-engine-dispatch-maps.md): which kernel each engine actually picks on 12.1 by default, the flag that changes each choice, and the known-bad defaults.
- [Profiling protocol](14-profiling-protocol.md): how to classify a slow step (bandwidth, compute, latency, launch or communication bound) with nsys, ncu and the torch profiler. An unprofiled A/B/A decides every fix.

Core levers
- [CUDA graphs and launch overhead](01-cuda-graphs-and-launch-overhead.md): recorded launch lists remove host cost only for the shapes captured at boot. The graphs cost boot time and memory, and they have their own failure modes.
- [Speculative decoding](02-speculative-decoding.md): a drafter guesses tokens and the target verifies them in one pass. Usually the biggest single decode lever on GB10, but acceptance depends on the workload.
- [Inter-node communication over DAC](03-inter-spark-communication.md): the fixed per-collective TP tax, the two PCIe halves behind each cable, and why fewer, larger collectives beat more bandwidth.
- [Prefill optimization](04-prefill-optimization.md): chunk size, mixed prefill and decode scheduling, and the warm path. The fastest chunk on paper can stall other users or freeze the node.
- [Memory on unified memory](11-memory-on-unified-memory.md): one 128 GB pool shared with the OS. MemAvailable is not allocatable memory, driver-pinned memory is invisible to the OOM killer, and every speed lever is paid from the same headroom.

Deep dives
- [Decode optimization](05-decode-optimization.md): the bytes model (step time is roughly bytes read divided by bandwidth) and the four ways to beat it: fewer bytes, split bytes, more tokens per read, fewer fixed costs.
- [MoE dispatch](08-moe-dispatch.md): route, dispatch and the fused expert GEMM, and how often each expert's bytes are read. Expert parallelism without NVLink buys KV capacity, not speed.
- [GEMM backends and quant formats](09-gemm-backends-and-quant-formats.md): the weight format decides which GEMM runs. Weight-only kernels win at decode sizes, native or BF16 wins at prefill sizes, and format is also a quality choice.
- [KV cache and prefix caching](10-kv-cache-and-prefix-caching.md): KV formats and pool sizing, and hash-chained prefix reuse, where one changed early token turns a one-second turn into minutes.
- [Attention backends on SM121](06-attention-backends-sm121.md): which attention kernels run on 12.1. Correctness at long context comes first, CUDA-graph compatibility second and raw speed last.
- [Linear attention and hybrid layers](07-linear-attention-and-hybrid-layers.md): KDA, GDN and Mamba layers, their SM121 kernel traps, and how recurrent state changes prefix caching.
- [Sampling and the lm_head](12-sampling-and-lm-head.md): the sampler is cheap and the vocabulary head is expensive. The drafter's head bytes and the pinned sampling regime are the levers.
- [Clocks, thermal, power](13-clocks-thermal-power.md): what a clock cap costs prefill and saves in heat, the throttle and power-off bands, and why the usual telemetry cannot see them.
- [Kernel authoring on SM121](15-kernel-authoring-sm121.md): when to write a kernel (rarely first), which toolchains target 12.1, how to prove a kernel correct, and how to run an AI-assisted loop without being fooled.

Case studies and gates
- [Case study: MiaAI-Lab's recipes, mined from GitHub history](18-case-study-miaai-lab-recipes.md): one lab's recipe history read as a dated lab notebook, including its claimed and corrected numbers, the order in which the maintainer attacks a new model, and which levers transfer.
- [Case study: the other recipe authors](19-case-study-community-recipe-authors.md): about forty authors, the lever each pushes, how to read each one's numbers, and which reproduced here, held only on the author's fixture, or reversed.
- [Model architecture cards](20-model-architecture-cards.md): per model family, the attention mix, the expert layout, bytes per token and KV bytes, plus measured recipes by node count and a fit table.
- [Quality and correctness gates](21-quality-and-correctness-gates.md): how optimizations silently change output, the narrow gates that caught real GB10 failures, and the tier battery a lane must pass before promotion.
- [Serving agent workloads](22-serving-agent-workloads.md): concurrency, scheduling, prefix reuse and TTFT budgets for long, mostly identical agent prompts that arrive every few seconds.

Cross-cutting
- [Levers ledger](ledger.md): one row per lever across all pages, with applicability, gain band, evidence grade, cost and risk.
- [Resource registry](registry.md): a graded catalog of public GB10 inference repos, recipes, tools and threads, with maintenance state, a verdict, and the pages each one feeds.
- [How pages are written](CONTRIBUTING.md): the nine-section skeleton, the evidence vocabulary and the export-safety rules.

## How to add a finding

Run the protocol in §6 of the page that owns the change, under the rules above, and file the packet. Then add a dated finding to that page's §4, with its evidence grade, baseline, workload and scope, and name what it supersedes. Negative results stay on the page. Update the matching row in the [Levers ledger](ledger.md). Anything specific to a named lane (paths, hosts, recipe names) goes in the private overlay, not on the public page. The page format, the grades and the export rules are in [How pages are written](CONTRIBUTING.md).

---

## What is coming

A second layer, *the science of GB10 inference*, is in progress: a compact model of the machine (memory budget, decode as bytes per step, prefill and chunk economics, the multi-node tax, kernels on SM121, measurement) plus a method for reasoning from a new model's architecture card to a predicted recipe. It will be added under `science/` once it is reviewed.

## Contributing

Open an issue with the measurement template (baseline, workload, scope, image digest, a pin for every source), or the correction template for a claim that is wrong or stale. Pull requests follow [CONTRIBUTING.md](CONTRIBUTING.md): the nine-section skeleton, the evidence vocabulary and the export-safety rules. `python3 tools/lint_page.py <page.md>` checks a page's skeleton, frontmatter and generic scrub patterns before you submit.

This repo is exported from the maintainer's working notes, which hold the canonical copy; links were converted from Obsidian wikilinks to relative markdown links on export.

## License

Documentation (all `.md` files) is licensed under [CC BY 4.0](LICENSE). Attribute as: *GB10 Inference Wiki, github.com/sovereignbrah/gb10-inference-wiki, CC BY 4.0*. Tooling under `tools/` is licensed under [Apache-2.0](tools/LICENSE). Community work cited in these pages remains its authors' and is credited by handle and link.
