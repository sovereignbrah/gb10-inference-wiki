# {Lane name} — lane record

A lane is one served model on one topology. The science core reads these fields; keep one record per lane and update it after every change.

## Current recipe and contract
| item | value | where recorded |
|---|---|---|
| topology (1x / TP2 / TP3) | | |
| image digest, engine commit, kernel-library versions | | |
| model, checkpoint, quant format, layers kept in BF16 | | |
| context served, slots (max concurrent sequences) | | |
| spec method and k, drafter placement | | |
| chunk size (MNBT / prefill chunk), cache page size | | |
| KV pin or memory fraction | | |
| NCCL env (multi-node) | | |
| clock cap | | |
| host baseline: MemAvailable before engine start, per rank | | |

## Constants the core reads
| term | value | measured how, when |
|---|---|---|
| KV bytes per token per rank (`KV_bpt`) | | |
| graph pool size (`G`) | | |
| warm floor (`F_warm` observed after the longest qualified prefill) | | |
| transient prefill constant (`c_lane`) | | |
| noise band (in-boot and boot-to-boot) | | |
| bytes-bound decode ceiling | | |
| boot time to API ready | | |

## What has been measured on this lane
| date | change | baseline -> result | workload | packet |
|---|---|---|---|---|

## Settled experiments
| lever | conditions (drafter layout, build, topology, clock) | result vs baseline | packet | retest trigger |
|---|---|---|---|---|

## Levers not yet tried
(ranked by the Amdahl bound under the operator's weights)

## Last incident
(configuration, per-rank MemAvailable minimum, NV_ERR counts, retries, the rule that followed)

## Owner notes
(operator weights across workload classes; accepted trades; anything the numbers cannot say)
