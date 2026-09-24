---
name: Submit a measurement
about: Add a measured finding to a topic page or the ledger
title: "[measurement] <page>: <lever or symptom>"
labels: measurement
---

A finding is usable only with all of the following. Leave a field as "not recorded" rather than guessing.

**Page and lever:** (e.g. 02-speculative-decoding, lever 3)

**Hardware:** GB10 box model · node count · fabric (single node / RoCE cables / switch) · driver · firmware

**Build:** engine and commit or tag · image digest · kernel libraries (FlashInfer, Triton, kit) with versions · GPU clock cap state

**Model:** family · checkpoint · quant format · which layers stay BF16 · context served

**Configuration:** the exact flags of the arm that changed, and the baseline arm. One variable per boot; a bundle is reported as a bundle.

**Workload:** fixture class (real-shaped code / prose / structured / counting) · prompt depth · concurrency (confirmed from the engine, not the label) · output cap · thinking mode

**Method:** repetitions per arm (≥ 3) · closing baseline (A/B/A) · cold-prompt nonce and `cached_tokens == 0` asserted · first request discarded · MemAvailable sampled every 1-2 s on every rank · NV_ERR / Xid count from the kernel journal

**Result:** baseline value · changed value · per-run values or spread · metric definition (per-stream vs aggregate; tokens from the usage block, not stream chunks)

**Correctness gate passed:** which gate from 21-quality-and-correctness-gates, and the result

**Evidence grade you claim:** [Measured] / [Community-measured] / [Code-verified]

**Sources:** pin every one (commit SHA or tag for code, dated fetch for pages, arXiv id for papers)
