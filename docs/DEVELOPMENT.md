# How the series is produced

The patches are **not hand-cut from a monolithic diff**. They are generated from verified tree
states:

```
t0  clean target tree
t1  t0 + coalesced restore     -> 01-kv-restore-coalesce.patch      = diff(t0,t1)
t2  t1 + pool placement        -> 02-kv-pool-placement.patch        = diff(t1,t2)
t3  t2 + decode priority       -> 03-scheduler-decode-priority.patch = diff(t2,t3)
t4  fully patched              -> 04-slot-management.patch          = diff(t3,t4)
```

Verification is threefold:

1. **every step compiles on its own** (CPU build, `--target llama-server`)
2. **the cascade applies to a clean tree** with zero failed hunks
3. **the end state is bit-identical** to the fully patched reference tree

Done for two targets:

| Target | Version | Patches | Result |
|---|---|---|---|
| `ggml-org/llama.cpp` master | `c32d1dabe` (2026-09-10) | 02, 03, 04 | all three checks pass |
| `unslothai/llama.cpp` | `b10840-mix-d5c17a0` (2026-09-08) | 01, 02, 03, 04 | all three checks pass |

Upstream has the run coalescing itself now (`for (const auto & r : runs)`), so 01 drops out there.

## For a different llama.cpp version

`scripts/apply.sh` first replays the series against a copy of the tree and does not touch the real
one if any patch fails. Against a distant version it therefore aborts cleanly. Then:

1. Pick the closer variant (both are from September 2026; if in doubt, try both).
2. If only a few hunks fail, it is usually context lines that moved upstream. `patch -p1 --merge`
   leaves conflict markers and you fix them by hand.
3. Afterwards **regenerate the series** rather than patching the patch: build the tree states, diff
   them, run the three checks. `scripts/hunks.py` helps with selecting individual hunks.

## Why 04 is one patch and not five

The five parts in 04 are intertwined in the code: the GC writes snapshots, the turn counter feeds
the snapshot rule, and telemetry reads fields from all of them. Splitting them at patch level would
mean surgically removing those call sites for each intermediate state and putting them back — four
extra tree states that all have to compile, for a benefit that already exists: **every part has its
own runtime switch**, no rebuild needed. See RUNBOOK.md.
