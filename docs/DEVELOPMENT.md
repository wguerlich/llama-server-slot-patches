# Development

## Layout

```
patches/upstream-3cf03257f/    the series, one file per patch
scripts/apply.sh               apply, with a dry run against a copy first
scripts/verify.sh              report which parts are present in a tree
scripts/hunks.py               split a diff, drop hunks, write it back
docs/MEASUREMENTS.md           what was measured, on what
RUNBOOK.md                     every runtime flag
```

The patches touch nine files:

```
common/arg.cpp                 the flag surface
common/chat.h                  message spans, first_user_message_end()
common/common.h                the parameters
src/llama-kv-cache.cpp         pool placement (patch 01)
tools/server/server-common.h   shared declarations
tools/server/server-common.cpp
tools/server/server-context.cpp   the bulk of it
tools/server/server-task.h
tools/server/server-queue.cpp
```

plus two additions: `tools/server/server-analyzer.h` (the hint channel) and
`tools/server/tests-snap/` (the test bed).

## Applying and checking

```bash
scripts/apply.sh upstream-3cf03257f /path/to/llama.cpp        # all three
scripts/apply.sh upstream-3cf03257f /path/to/llama.cpp 02     # only 01 and 02
scripts/verify.sh /path/to/llama.cpp
```

`apply.sh` replays the whole series against a copy of the tree first and only touches the
real tree if every requested patch applies there. **A half-patched tree cannot happen.**

`verify.sh` greps for a marker of each part and prints a count:

```
  pool filled from bottom            present (3)
  decode priority                    present (7)
  slot eviction and GC               present (33)
  prefix sharing                     present (9)
  prefix index                       present (13)
  disk snapshots                     present (21)
  divergence probe                   present (19)
  hint channel                       present (4)
  request telemetry                  present (8)
  test bed                           present
```

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # plus your backend flags
cmake --build build --target llama-server -j
```

On gfx1151 (Strix Halo) two flags are not optional:

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
      -DGGML_HIP_ROCWMMA_FATTN=OFF \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DCMAKE_BUILD_TYPE=Release
```

`ROCWMMA_FATTN=OFF` because the rocWMMA flash-attention path does not work there.
`FA_ALL_QUANTS=ON` because anything but a matched K/V type — `q8_0`/`q4_0`, f16-K/`q8_0`-V
— needs the full kernel set compiled in.

## Rebasing onto a newer llama.cpp

The patch base is named in the directory: `patches/upstream-<commit>/`. Onto a nearby
commit the series usually applies unchanged; further out, expect fuzz in `arg.cpp` and
`server-context.cpp`.

The working method is to carry the tree, not the diff:

```bash
git clone https://github.com/ggml-org/llama.cpp work && cd work
git checkout 3cf03257f
../scripts/apply.sh upstream-3cf03257f .
git add -A && git commit -m "patched base"
git rebase <newer-commit>                    # resolve there, where you have context
```

Then regenerate the series against the new base:

```bash
git diff <newer-commit> -- src/llama-kv-cache.cpp > 01-kv-pool-placement.patch
# 02 and 03 are per-hunk selections of the server diff - see hunks.py below
```

Verify the regenerated series reproduces the tree bit for bit:

```bash
scripts/apply.sh upstream-<newer> /tmp/fresh-checkout
diff -r /tmp/fresh-checkout work        # must be empty
```

**Then compile the fresh checkout, and treat that as the check that matters.** Bit-identical
reproduction cannot catch the one mistake that actually breaks the series for everyone else:
a hunk that quietly depends on something the declared base does not have. If the tree you
developed in carries a fork's extra symbol, or an upstream commit newer than the base, the
diff carries it too — `apply.sh` succeeds, `verify.sh` reports every part present, and the
compiler is the first thing to object.

The failure looks like this, and it is silent up to the last step:

```
apply.sh   -> 3 patches ok
verify.sh  -> 10 parts present
cmake      -> error: use of undeclared identifier 'LLAMA_LAZY_MODE_DIRECT'
```

So: a pristine checkout of the **declared** base, the series, a build, and only then is the
variant real. Developing against a fork is fine — regenerating the series against it and
leaving the old base in the directory name is not.

## Splitting the series

Patches 02 and 03 both touch `server-context.cpp`, so they are hunk selections out of one
diff rather than separate file diffs. `hunks.py` does that mechanically:

```bash
python3 scripts/hunks.py list combined.patch
#   tools/server/server-context.cpp  hunk 7  @@ -1240,6 +1240,9 @@  bool prefill_defer...

python3 scripts/hunks.py cut combined.patch "tools/server/server-context.cpp:3,7-9" out.patch
```

`cut` writes everything **except** the listed hunks, so the two patches are complementary
selections of the same diff. After splitting, check that each stage still compiles on its
own — patch 01 alone, then 01+02, then all three. The published series is verified that
way.

## The test bed

```bash
cd tools/server/tests-snap
MODEL=/path/to/model.gguf ./run.sh                          # the model's own template
MODEL=/path/to/model.gguf ./run.sh templates/last.jinja tl  # a bundled one
```

`LLAMA_SERVER`, `PORT`, `CTX`, `CASES` and `PRE` (a file sourced for backend environment)
override the defaults.

**The LLM is simulated** — assistant replies are made up by the driver rather than
generated, so a run is deterministic and the history exactly controlled. What is under test
is the template and the tokenizer, not the model. A run needs a real GGUF only for its
tokenizer and its chat template.

**A fresh server per case is deliberate.** The probe's learned state (`probe_miss`,
`div_offset`, the committed `probe_kind`) is per slot and would otherwise carry from one
case into the next, which is exactly the kind of contamination that makes a suite pass for
the wrong reason.

The bed computes the real divergence **independently of the server**, via `/apply-template`
and `/tokenize`, and holds it against what the probe predicted. Two levels: the text level
checks that markers vanish without a trace and that a harness with the hint channel renders
byte-identically to one without; the token level checks where two consecutive prompts
actually part.

Adding a harness shape means one builder function in `suite.py` and a name in `HARNESSES`;
the criteria apply to it automatically. Adding a template means a `.jinja` file in
`templates/` and a line in whatever drives the matrix.

`index-test.py` covers the prefix index separately — it needs a running server and a
snapshot directory:

```bash
BASE=http://127.0.0.1:8099 python3 index-test.py <telemetry.jsonl> <snapshot-dir>
```

## Things to know before changing the core

**Never convert a character offset into a token position.** Both sides get tokenised and
the common token prefix is taken. A character offset falling inside a token forces a
rounding decision, and every such decision is wrong somewhere. It also misses the
re-tokenisation seam: two strings can share a character prefix and still tokenise
differently across it — the `seam` harness in the bed exists for exactly that case, and it
costs a whole prompt when it is not handled.

**The probe's futures are absolute, not relative.** `user_think` / `user_nothink` /
`user_drop` / `tool` name what the *next prompt will contain*. An earlier formulation
("same policy" / "other policy") is relative to what was observed, and on the first turn
there is no assistant message to observe, so the same name meant different things at
different points in a conversation.

**The snapshot hash is seeded with the model's identity** — size, parameter count,
vocabulary size, `llama_model_desc`. Both `snapshot_hash()` and `snapshot_lookup()` must
use the seeded basis; seeding one and not the other makes every written file invisible to
the lookup, silently.

**Markers are stripped unconditionally**, including one with a wrong secret. Stripping only
on a valid secret means a mismatch leaks a marker into the prompt the model sees, which is
both a correctness and a prompt-injection problem.

**The echo policy is read from the *last* assistant message**, not the first — a harness
may switch policy mid-conversation, and the `mixed` shape in the bed does.

**`--kv-unified` is the precondition for two parts.** Prefix sharing and the GC of dead
slots do nothing on separate streams, where each stream's scan depth is private. The code
detects this and disables both, sharing with a warning. Do not remove that check.

## Reporting a problem

The one thing that makes a report actionable is a telemetry line:

```
--telemetry-file /tmp/requests.jsonl
```

One JSON line per request with the phase timeline, the reuse path, what the probe predicted
and whether it held. For a specific prompt that behaves unexpectedly, `--telemetry-prompt-dir`
adds the full prompt as text and token ids — **that writes conversation content in clear on
disk**, so use it for one question and then remove the flag.
