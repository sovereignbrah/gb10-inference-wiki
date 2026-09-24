# The Science of GB10 Inference

This folder is the understanding the topic pages rest on: how a GB10 behaves under an LLM, written so that an agent can reason from a model's architecture to the fastest recipe instead of copying one.

**Read [core.md](core.md) first**, in full (about 15K tokens). It carries the machine facts, one equation per phase, the procedures, the optimization loop and the short failure catalog, with a pointer key from `P01`…`P22` to the evidence pages one directory up.

Then the chapter for the phase you are working on. Each derives its equation, works it on a real lane with graded numbers, gives a numbered decision procedure and lists traps.

| chapter | question it answers |
|---|---|
| [02, the memory budget](02-memory-budget.md) | Will it fit, with what headroom, and what happens at the edge on unified memory |
| [03, decode = bytes per step](03-decode-bytes-per-step.md) | How fast decode can possibly go, by model family and topology, and why speculation is the lever |
| [04, prefill chunk economics](04-prefill-chunk-economics.md) | Why prefill is compute-bound, what the chunk trades against, and the prefix-reuse arithmetic |
| [05, multi-node = a per-step tax](05-multi-node-tax.md) | What two or three nodes cost per step, when TP pays, and the fabric preflight |
| [06, kernels on SM121](06-kernels-on-sm121.md) | Which kernels can exist on this chip, how engines pick them, and when authoring one is justified |
| [07, measurement](07-measurement.md) | Noise bands, the artifacts that fooled past sessions, and the protocol that survives them |

Rules of use, from the core: a lane constant is never taken from these files (read it from your own lane record; [lane-card-template.md](lane-card-template.md) lists the fields); every number carries a grade; anchor values show a term's size on one lane and are never defaults; a result holds only for its build, drafter and topology.

The `WR` pointer in these files is a review of the maintainer's own optimization sessions, quoted where a lesson came from a real mistake rather than a measurement.
