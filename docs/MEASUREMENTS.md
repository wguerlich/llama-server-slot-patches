# Measurements

All numbers come from **one machine**. The mechanisms are general; the magnitudes are not
transferable.

## The machine

| | |
|---|---|
| SoC | AMD Ryzen AI MAX+ 395 (Strix Halo), Radeon 8060S iGPU, gfx1151 |
| Memory | 91 GB usable, unified — CPU and GPU share it |
| Backend | ROCm / HIP, `GGML_HIP_ROCWMMA_FATTN=OFF`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| llama.cpp | `b78a39a2f` (build 10916) + this patch series |
| Storage | NVMe SSD |

Two servers run at the same time, each at 262 144 context with a unified KV cache, four
slots, two checkpoints per slot, sharing one snapshot directory.

| | Qwen 3.8 27B | Gemma 4 31B |
|---|---|---|
| Weights | `Qwen3.8-27B-UD-Q4_K_XL` 17.9 GB | `gemma-4-31B-it-qat-UD-Q4_K_XL` 17.3 GB |
| Attention | hybrid: 48 Gated DeltaNet + 16 full attention, interval 4 | iSWA, window 1024, every 6th layer full |
| KV cache | K `q8_0` / V `q8_0` | K `q8_0` / V `q4_0` |
| Speculation | in-model MTP head, `--spec-draft-n-max 8` | MTP sidecar `Q4_0`, `--spec-draft-n-max 4` |
| Vision | `mmproj-F16` | `mmproj-BF16` |

The flags each server runs with are in the [RUNBOOK](../RUNBOOK.md).

## Snapshot load vs. prefill

Straight from the production telemetry (`--telemetry-file`), one JSON line per request.
`ms_prefill` is the whole prefill phase, `ms_snap_load` the file load inside it.

**Gemma 4, a 1787-token preamble:**

| | `n_new` | `ms_snap_load` | `ms_prefill` |
|---|---|---|---|
| first arrival, prefilled from scratch | 1810 | — | **6076** |
| next session with the same preamble | 27 | 244.5 | **398.7** |
| again | 11 | 246.0 | 368.4 |
| after a service restart | 35 | 120.3 | **395.8** |

**Qwen 3.8, a 1711-token preamble:**

| | `n_new` | `ms_snap_load` | `ms_prefill` |
|---|---|---|---|
| first arrival | 1737 | — | **5485.8** |
| next session, restored and extended | 778 | 46.1 | 2830.7 |
| next session, restored | 29 | 52.0 | **505.1** |
| a 2339-token preamble, restored | 37 | 169.3 | 602.6 |

The load itself is well under a second in every case. What remains in `ms_prefill` is the
handful of tokens after the shared prefix plus the batch overhead.

## Follow-up turns

The probe places checkpoints where the next prompt will part. In the production telemetry a
follow-up turn of a conversation reads:

```
hit          n_prompt  n_cached  n_new  ms_prefill  probe_kind  probe_miss
checkpoint   1776      1748      28     409.3       0           0
checkpoint   1801      1774      27     434.2       0           0
```

`n_new` is the new turn itself — **nothing of the history is recomputed**. `probe_kind 0`
is the future the probe committed to after the first turn; `probe_miss 0` means every
prediction since held.

Where the probe is wrong it says so and learns. The Gemma run below is a harness whose
render the template cannot predict — three misses in a row, then the probe stops placing:

```
hit    n_prompt  n_cached  n_new  probe_miss
live   1821      1798      23     1
live   1847      1821      26     2
live   1873      1847      26     3
```

The recomputation there is bounded by `--ctx-checkpoints` falling back to the live slot;
what it costs on a harness that rewrites more aggressively is the next section.

## The hint channel

The case the channel exists for: a recursive harness that wraps its whole history into one
message. Measured in the test bed, same harness, three follow-up turns:

| | probe candidates | tokens recomputed |
|---|---|---|
| `wrapped` — no hint | [174], [315], [464] | **168** |
| `wrapped-hint` — one `ckpt` marker | [174], [315], [464] | **0** |

The candidates are identical, which is the point: the probe computes the same thing either
way and is wrong either way — the real divergences are 168, 309, 458 — and the hint is what
carries the truth the template cannot.

The two render **byte-identically**: the bed compares the template output and the token
stream with and without the markers, and a difference is a finding.

## Probe cost

Extra template renders per request, and a tokenisation per render. No model runs.

The first version compared whole token streams and therefore tokenised the entire prompt once
per future — linear in the conversation, and impossible with media in the prompt, which is
why such prompts were excluded from positioning. Since 2026-09-15 the probe finds the seam in
characters and tokenises only an 8192-character window around it, the same window in both
renders. The error the tokeniser makes at the window's ragged start cancels, because the
window's length and the common prefix within it are measured in the same tokenisation.

A/B on one warm request, same prompt, five runs each, `llama-server` on CPU:

| | median | min–max |
|---|---|---|
| whole-stream, 19 325 tokens | 77 ms | 66–86 |
| windowed, 19 325 tokens | **42 ms** | 31–45 |

The gap widens with context, because the window does not grow. For scale, one full
tokenisation of a real production prompt measured 20 ms at 6 k tokens, 67 ms at 41 k and
190 ms at 159 k — the old form paid that once per future.

## Snapshot size

Size is linear in position, `fixed + per_token × n`. Fitted on the production files:

| | fixed | per token | 1787 tokens | 2339 tokens |
|---|---|---|---|---|
| Qwen 3.8 (hybrid recurrent), `q8_0`/`q8_0` | 157 MB | 38.9 KB | 224 MB | 248 MB |
| Gemma 4 (iSWA), `q8_0`/`q4_0` | 340 MB | 33.9 KB | 400 MB | — |

Measured files, for the fit:

```
df8d306aebdc646d-1711.lsnp   223.6 MB   Qwen,  1711 tokens
c87cc684912d59e0-2339.lsnp   248.0 MB   Qwen,  2339 tokens
9ed2a98ffa12955b-3961.lsnp   311.2 MB   Qwen,  3961 tokens
9e77846898890d22-1787.lsnp   400.3 MB   Gemma, 1787 tokens
b4477aeca203ddb1-1846.lsnp   402.3 MB   Gemma, 1846 tokens
```

**The fixed part is what matters, and the two architectures differ in what it is made of.**

On the hybrid model it is the recurrent state: 48 Gated-DeltaNet layers × 6144 × 128 × 4 B
in f32 = **157 MB, independent of `--cache-type-k/v`** — the recurrent state is not
quantised. On the iSWA model it is the sliding-window caches, which *do* scale with the KV
type: 839 MB at f16 KV, 340 MB at `q8_0`/`q4_0`.

Either way the fixed part does not shrink with position. A 600-token snapshot on Gemma
still costs 340 MB. That is what `--snapshot-min-tokens` and `--snapshot-min-gap` are for.

Earlier configurations, for the KV-type dependency:

| | fixed | per token |
|---|---|---|
| hybrid recurrent, f16-K / `q8_0`-V | 157 MB | 54.3 KB |
| iSWA, f16 KV | 839 MB | 81.9 KB |

## Checkpoint size

Checkpoints are `PARTIAL_ONLY` — they hold only the part that cannot be sliced by prefix,
which is exactly the fixed part above and therefore **independent of position**:

| | per checkpoint |
|---|---|
| Qwen 3.8, hybrid recurrent | **155 MiB** |
| Gemma 4, iSWA | **800 MiB** |

Host RAM, times `--ctx-checkpoints`, times the slot count. At four slots and two
checkpoints that is 1.2 GB for Qwen and 6.3 GB for Gemma — which is why the production
configuration runs two, not the default 32.

## Speculative decoding

Both models use MTP drafting. The optimum was measured with **eight distinct prompts, each
sent once** — repeating a prompt lets `ngram-mod` replay the previous answer and moves the
optimum by several steps.

| | optimum `--spec-draft-n-max` |
|---|---|
| Qwen 3.8, in-model MTP head (`blk.64` nextn tensors) | **8** |
| Gemma 4, MTP sidecar | **4** |

For Gemma the sidecar variant matters too — `Q4_0` was fastest of the four measured,
`BF16` slowest, which is what you would expect for a draft model on a
memory-bandwidth-bound machine.

## Test bed

`tools/server/tests-snap/` — the LLM is simulated, so runs are deterministic and what is
under test is the template and the tokenizer. Eleven harness shapes × templates, fresh
server per case, **two slots** (with one, a rejected continuation lands in the same slot
anyway and looks like a hit; a follow-up served from the empty second slot shows `hit full`,
`lcp 0`, and its recomputation is counted as the whole divergence).

Criteria per follow-up turn: **A** the turn resumes without recomputing (`restore_at ==
lcp`), **B** the real divergence is among the probe's candidates, **C** from the third
follow-up exactly one checkpoint is placed, **D** an unpredictable harness silences the
probe after two misses, **E** render and tokens are identical with and without the hint
channel.

Harness shapes: `preserve` · `strip` · `last-only` · `wrapped` · `tools` · `tools-think` ·
`seam` · `rewrite` · `mixed` · `wrapped-hint` · `toolloop`. Templates: `native` plus `preserve`,
`strip`, `last`, `plain`.

The native template, all ten shapes (`divergence` is the truth computed independently of
the server, `probe` what it predicted per turn):

```
preserve      divergence [167, 305, 451]  probe [[166],[304,166],[450,304],[589,450]]  recomputed   0  ok
strip         divergence [166, 296, 427]  probe [[166],[296],[427],[556]]              recomputed   0  ok
last-only     divergence [167, 166, 296]  probe [[166],[304,166],[442,296],[566,427]]  recomputed   0  ok
wrapped       divergence [168, 309, 458]  probe [[174],[315],[464],[606]]              recomputed 168  ok
tools         divergence [416, 591, 767]  probe [[416],[591],[767],[941]]              recomputed   0  ok
tools-think   divergence [417, 608, 814]  probe [[416],[607,469],[813,667],[1007,868]] recomputed   0  FINDING (closed, below)
seam          divergence [167, 305, 451]  probe [[166],[304,166],[450,304],[589,450]]  recomputed   0  ok
rewrite       divergence [167,  52,  73]  probe [[166],[304,166],[205],[212]]          recomputed 125  ok
mixed         divergence [167, 166, 427]  probe [[166],[304,166],[427],[556]]          recomputed   0  ok
wrapped-hint  divergence [168, 309, 458]  probe [[174],[315],[464],[606]]              recomputed   0  ok
```

`wrapped` and `rewrite` are the two that recompute, and they pass: what is asked of the
probe there is not accuracy but **giving up** — two misses and it stops placing, which is
criterion D. `wrapped-hint` is the same harness as `wrapped` plus one marker.

The other four templates are clean on all ten shapes.

Result on Qwen 3.8 27B: **50 combinations, 49 clean, 1 finding** — the finding is analysed
and closed further down.

Re-run on 2026-09-15 against a plain global-attention model (Qwen 2.5 0.5B, which needs no
checkpoints of its own and so exercises the gate rather than the state): **50 combinations,
0 findings**. On Gemma 4 12B the findings are byte-identical to the state before that day's
changes — no regression; what it reports is 5 tokens recomputed per turn, from a divergence
sitting exactly at the prompt end, where nothing can be placed during the prefill.

### The `tools-think` finding, and what closed it

It is on the native (Qwen) template, harness `tools-think` — a tool loop that
keeps its reasoning in history — and it is criterion **C**, convergence:

```
tools-think   divergence [417, 608, 814]   probe [[416], [607, 469], [813, 667], [1007, 868]]
              recomputed 0
              ! T2: still placing 2 checkpoints [813, 667] after 2 follow-up turns
```

The probe's predicted position is a *lower bound* — it is the common prefix against a dummy
continuation, and here the dummy happens to agree with reality for one more token (416
against a true 417). A checkpoint one token early is safe and costs nothing, but the commit
rule requires the follow-up to land **exactly** on a candidate, so it never fires and both
futures keep getting a checkpoint.

Cost: one checkpoint slot. **Not time** — every turn of that harness shows `hit: live` with
`restore_at == lcp`, so nothing is recomputed; the live slot still holds the prefix and no
checkpoint is consulted at all.

Loosening the commit to "landed at or past a candidate" would close it, at the price of a
rule that can commit to the wrong future on a harness the probe did not model — which costs
real re-prefills, not a checkpoint slot. **The exact rule stays.**

What closed it instead is reading the same turns from the other side. Every follow-up here
is a **pure append**: `restore_at == lcp`, the live slot covers everything, no candidate is
consulted. Measured over four turns on the 0.5B bed, native template:

```
n_prompt  260  lcp    0  restore_at    0   considered []      placed []
n_prompt  427  lcp  260  restore_at  260   considered []      placed []
n_prompt  595  lcp  427  restore_at  427   considered []      placed []      <- brake
n_prompt  761  lcp  595  restore_at  595   considered []      placed []
```

and, on the `preserve` template where candidates do exist, the brake is visible directly:

```
n_prompt  268  lcp  128  restore_at  128   considered [128]   placed [128]
n_prompt  416  lcp  268  restore_at  268   considered [268]   placed []      <- brake
n_prompt  557  lcp  416  restore_at  416   considered [416]   placed []
```

So appending became an outcome the probe can learn, next to "miss": two consecutive
follow-ups that are a pure append and use no candidate, and it stops placing for that slot.
Anything else resets the counter, so a harness that changes shape pays one re-prefill and
gets its candidates back. The commit rule is untouched, and nothing in the bed recomputes a
token that did not before.

Re-run after the change, all five templates × all ten shapes on Qwen 2.5 0.5B:
**50 combinations, 0 findings, `recomputed 0` throughout.**

### What the test bed hid, second time (2026-09-17)

Every "tool loop" above was a *history shape*: the messages carried tool calls and results,
but no request ever asked the server to continue after a tool result. So no request ever had
a probe candidate **behind** its prefill — and that is where the far-back positions of the
last-query family of templates (Qwen3/3.5/3.6, gpt-oss, MiniMax, GLM: reasoning kept inside a
tool loop, stripped once a user turn follows) come from. The production template renders
reasoning for every assistant turn regardless and so never produced one; the 44 of 44
follow-ups that hit exactly `L-1` in production were all the same, easy case.

Reproduced with a real loop (`toolloop`, nine tool-result requests, then a user turn) under
`templates/lastquery.jinja`, Qwen 2.5 0.5B, two slots:

```
ctx-checkpoints 3, before the fix
req  n_prompt  lcp  restore  hit   probe_all / checkpoints
  0     294      0      0    full  [291, 293] / [291, 293]
  1     416    294    294    live  [291] / [291, 293, 412]
  2     538    416    416    live  [291] / [293, 412, 534]        <- 291 evicted
  ...
  6    1026    904    904    live  [291] / [778, 900, 1022]
  7     939      0      0    full  [936]                         <- user turn, full prefill
picker: slot 0: lcp 291 restore_at 0 ckpts [778,900,1022] -> rejected: no usable checkpoint
journal: "[probe] this session takes future 0 (landed on 291)"   <- the prediction was right

ctx-checkpoints 3, after: 291 pinned on every iteration, req 7 restore_at 291 = lcp, 0 recomputed.
ctx-checkpoints 8, nine iterations: same picture, evicted at the seventh before, kept after.
```

The prediction was right six times over; the retention was wrong. A candidate behind the
prefill is now listed as a computed position when a checkpoint already sits there, which
pins it against the cap — and the listing happens *before* the append rule, because an
append-only loop is exactly the session that needs it.

Three more things the same session of measurements turned up, all on production telemetry
and journal of 2026-09-16:

- **A dump that wrote nothing.** A 13-turn, 103k-token session's last request was deferred
  eight minutes behind a decode, then cancelled with `n_tokens` exactly at its probe stop —
  the checkpoint there is written at the start of the next iteration, which never came. The
  eviction dump found no checkpoint at the probed position, wrote nothing and logged nothing;
  the next two prompts fell back to a 68 779-token snapshot and re-prefilled **34 776 and
  39 757 tokens**. The dump now falls back to the nearest checkpoint below, then to the end
  of generation, and says so; a cancel writes the checkpoint at a reached stop, drops the
  candidates of a prefill that never finished, and leaves a telemetry line (`aborted`).
- **A restart that took a session with it.** 108k tokens, four turns, gone at
  `systemctl restart`. Chat slots now go through the same dump on shutdown, after the loop
  ends and before the backend is freed.
- **`probe_kind` was `-1` in 97 of 97 requests.** A harness that never echoes reasoning had
  no "drop" future to render, and that was counted as an incomplete candidate set, so it
  could never commit. A moot future is not a gap. Separately, the reasoning-tag derivation
  returned "not derivable" on the production template, which emits `<think>\n\n</think>`
  around empty reasoning; the tags now come from the chat format instead.

Re-run after the changes, eleven shapes on `native`, `last` and `lastquery`:
**33 combinations, 0 findings**; `recomputed 0` everywhere except `rewrite` (14, the
divergence lies below every checkpoint and the fresh slot is the right call).

### The hint channel, checked end to end (2026-09-17)

`hints-test.py`, Qwen 2.5 0.5B, `--ctx-checkpoints 3`, two slots. Before and after:

| case | before | after |
|---|---|---|
| `ckpt at=user-1`, 4000 tokens before the end | "stop planned at 40", hint counted, checkpoints `[4022]` — **none at 40** | `[40, 4022]` |
| `ckpt at=user-1` behind the prefill, continuation | rolled back 58 → 52, checkpoint at 52 | unchanged |
| a `ckpt` marker carried in history, four turns, cap 3 | every old position re-pinned, cap evicts PINNED, turn 4 **rolls back 116 → 34 (82 tokens)** for a position nobody resumes at | historic positions soft, evicted first; `restore_at == lcp`, 0 re-prefill, one hint counted per turn |
| `ckpt at=agent`, nothing generated | a 1084-token **file** written, kind ignored, hint not counted | no file, hint counted; the shutdown/eviction dump keeps the end of generation |
| `snap at=user-1` mid-prompt | file at 41 | unchanged |

The first row is the mid-prompt gate — checkpoints only at user-message starts or within a
ubatch of the end — evaluated before the hint was consulted. The same gate would have
withheld a probe candidate more than a batch from the end on a fresh prefill.

`mid-system` in the bed — a system message before every user turn — is clean on `lastquery` and
on ChatML (0 recomputed, the probe's candidates unchanged); on the Flash-Next template the
request itself fails at the template, before any of this runs.

The prefix index is covered separately by `index-test.py`: a single session cannot inflate
the index or fork against itself, identical preambles trigger nothing, one fork earns a
file, and a later session loads it.

## What "pure append" has to be measured against (2026-09-17)

The probe calls a follow-up a **pure append** when the new prompt reproduces the previous one
and carries on past it. The yardstick for that was the slot's token buffer:

```c
const bool appended = lcp_now >= slot.prompt.tokens.size();
```

Against real traffic that is never true. The slot buffer holds the prompt **and the answer
that was generated into it**, and the answer is not part of any prompt. Measured over the
follow-ups of one production day, comparing the common prefix against the slot length from
the picker's own trace:

```
slot_len  13365   lcp  13285   ->  80 tokens short
slot_len  19984   lcp  19905   ->  79 tokens short
...
lcp >= slot_len (appended TRUE) :  0
lcp <  slot_len (appended FALSE): 42
```

**0 of 42.** Always short by the 79-80 tokens of the generated answer plus its turn-end
markup. The yardstick is now the length of the previous *prompt* - the same quantity the
turn detection two lines above already uses, with the same slack for the re-tokenisation
seam:

```c
const bool appended = slot.snap_prompt_last > 0 &&
                      lcp_now + next_turn_slack >= (size_t) slot.snap_prompt_last;
```

### What the test bed hid

The bed's simulated LLM writes a short fixed answer and the template re-renders it byte for
byte on the next turn, so the common prefix *does* cover the whole slot there. The condition
held every turn in simulation and never once in production: **the bed agreed with a
requirement the real thing cannot meet.** Any finding about append behaviour has to be
confirmed against real traffic, not against the bed alone.

### And the probe was right all along

Chasing this turned up the opposite of what the counters suggested. `probe_append` reads 0 in
production, and the reason is not the dead branch - it is that a candidate **is** used. Five
consecutive turns of one conversation, each candidate checked against the next turn's truth:

| candidate placed | next turn's divergence | |
|---|---|---|
| 5565 | 5565 | exact |
| 11085 | 11085 | exact |
| 16605 | 16605 | exact |
| 22125 | 22125 | exact |

The computed position is the divergence point to the token, four turns running, and the
roll-back lands on it (`hit checkpoint` each time): **50 %, 67 %, 75 %, 80 %** of the prompt
reused. The brake is silent because it should be. Nothing appears in the log either, and that
is consistent too: a hit is only announced when it *establishes* the session's future, which
needs a complete candidate set; an incomplete one hits quietly.

Over a full production day, 53 requests: **2 229 884 prompt tokens, 2 020 517 of them reused
(91 %)**, 29 checkpoint hits all at rank 0, 8 snapshot loads from SSD, prefill median 623 t/s
and decode median 28.2 t/s.

## Reproducing any of this

Every number above comes from `--telemetry-file`. Turn it on, run your own traffic, and
read the fields listed in the [RUNBOOK](../RUNBOOK.md#reading-the-telemetry). That is the
point of it being there — none of these magnitudes should be trusted on a different
machine.
