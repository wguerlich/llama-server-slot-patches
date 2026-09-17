# Runbook

A server built with all three patches and started **without any of the flags below behaves
exactly like upstream**. Every part is switched on individually, at runtime, and can be
switched off again by dropping one flag — no rebuild.

## A starting configuration

```bash
llama-server -m model.gguf \
    --host 0.0.0.0 --port 8080 \
    -c 262144 --parallel 4 --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    \
    --prefill-defer-above 2048 \
    \
    --ctx-checkpoints 2 \
    --ctx-checkpoint-probe on \
    \
    --snapshot-path /var/lib/llama/snapshots \
    --snapshot-max-disk-mb 409600 \
    --snapshot-min-tokens 500 \
    --snapshot-evict-turns 2 \
    \
    --slot-evict-policy cost \
    --kv-share-min 4096 \
    \
    --telemetry-file /var/log/llama/requests.jsonl
```

Then read the telemetry for a day and decide which of it pays on your traffic.

## Turning individual parts on and off

| Part | On | Off |
|---|---|---|
| Pool filled from the bottom | always (patch 01) | — |
| Decode priority | `--prefill-defer-above 2048` | omit, or `0` |
| Checkpoint placement by probe | `--ctx-checkpoint-probe on` (default) | `off` → upstream's batch-boundary placement |
| Checkpoints at all | `--ctx-checkpoints N` | `0` |
| Disk snapshots + prefix index | `--snapshot-path DIR` | omit → both silent |
| Eviction dump | `--snapshot-evict-turns 2` | `0` |
| In-band hint channel | an activation marker in the system prompt | no marker → nothing is parsed |
| Hint roll-back | `--snap-hint-max-reprefill N` | `0` → hints only ever look forward |
| Cost-based slot picking | `--slot-evict-policy cost` | `lru` (default) → upstream behaviour |
| Prefix sharing between slots | `--kv-share-min 4096` | `0` (default) |
| Telemetry | `--telemetry-file F` | omit |
| Full prompt dumps | `--telemetry-prompt-dir D` | omit — **writes conversation text in clear** |

## All flags

### Scheduler

| Flag | Default | |
|---|---|---|
| `--prefill-defer-above N` | `0` (off) | While any slot is generating, or a smaller prefill is waiting, a slot with more than `N` prompt tokens left contributes nothing to the batch. The batch size itself is untouched. Start at `2048`. |
| `--slot-max-active N` | `0` (unlimited) | Cap on slots decoding at the same time; further tasks stay queued. Decouples the slot count — which you want high for warm-prefix affinity — from concurrency, which costs per-request throughput. |

### Slot management

| Flag | Default | |
|---|---|---|
| `--slot-evict-policy {lru,cost}` | `lru` | `cost` picks the slot cheapest to rebuild, weighted by how likely it is continued, and frees only as many KV cells as the incoming prompt needs. `lru` is upstream. |
| `--slot-evict-grace SECONDS` | `30` | How long a fresh one-shot keeps its protection window. It outranks every other one-shot in that window and never a chat. |
| `--slot-evict-ttl SECONDS` | `600` | Slots idle longer than this are evicted first. `cost` policy only. |
| `--kv-share-min N` | `0` (off) | Fold a prefix shared between two slots onto one physical copy once it reaches `N` tokens. Needs `--kv-unified` — there `seq_cp` copies no data, it only updates the cell bitmap. Off by itself on separate streams. |

### Checkpoints and the probe

| Flag | Default | |
|---|---|---|
| `--ctx-checkpoints N` | `32` | Checkpoints per slot. **This is the memory knob**: 155 MiB each on a recurrent model, 800 MiB on an iSWA one, times the slot count. A settled session holds **2** (measured: one automatic, one computed or hinted), but until the probe has seen which future a harness takes it offers four, so **do not set it below 3**. `32` with the probe on will exhaust host RAM. The cap evicts the oldest *unpinned* checkpoint first and warns when every one of them is a declared or computed position — that warning is the signal it is too small. |
| `--ctx-checkpoint-probe on\|off` | `on` | Compute the next prompt's divergence point by rendering the chat template twice more per request. No model runs. `off` falls back to upstream's batch-boundary placement. |

### Prefix index and snapshots

| Flag | Default | |
|---|---|---|
| `--snapshot-path DIR` | off | Directory for disk snapshots. Setting it enables the prefix index too. |
| `--snapshot-max-disk-mb N` | `16384` | LRU disk budget. `0` = unlimited. Covers the whole directory, so several servers sharing one directory share one budget. |
| `--snapshot-min-tokens N` | `512` | Shortest prefix worth a file. A preamble below it is not indexed and an eviction dump skips positions below it. **Check this against your real prompts** — set above them, nothing is ever written. |
| `--prefix-min-forks N` | `1` | How many *other* conversations must reach a depth before it earns a file. One behind the **same** preamble reaches its full end; one behind a **different** preamble reaches where the two part. At `1` the file appears on the second conversation. What counts as another conversation is that it arrived *fresh* — no slot took it as a continuation — so a session extending itself never counts itself up, and a forked chat counts as two. |
| `--prefix-index-size N` | `1000` | Distinct preambles the RAM index remembers, LRU. Worst-case RAM is this × `--prefix-max-compare` × 4 bytes. |
| `--prefix-max-compare N` | `32768` | Tokens of a preamble kept for the fork search. A longer one is still identified in full by hash. If two preambles agree to the end of a **truncated** entry, that pair contributes nothing: where they really part lies beyond what was kept, and the cap is never reported as a fork. |
| `--snapshot-min-gap N` | `100` | Never place a snapshot within `N` tokens *behind* an existing one. In front is always allowed. |
| `--snapshot-evict-turns N` | `0` (off) | When a slot that served ≥ `N` slot-matched turns with growing prompts loses its content, snapshot it where the probe says the next prompt resumes — or, if no checkpoint sits there, at the nearest one below, else at the end of generation; it never leaves silently. The same dump runs for every idle chat slot on shutdown. `2` is a good starting point. |
| `--snapshot-at N` | `0` (off) | Test override: snapshot every from-scratch prompt at position `N`, bypassing the index. For experiments only. |

A file name carries a hash seeded with the model's identity — size, parameter count,
vocabulary size, description — so **several servers can share one directory** without
colliding.

### Hint channel

| Flag | Default | |
|---|---|---|
| `--snap-hint-max-reprefill N` | `-1` (= `n_ubatch`) | How many tokens of re-prefill a hint may cost when its position has already been passed. The server rolls back to a checkpoint at or before the hinted position and prefills forward again; a costlier hint is dropped, with the reason in the telemetry. `0` = never roll back. Only a **live** hint rolls back — one standing in a message after the last assistant turn. A marker carried along in history keeps an existing checkpoint softly pinned and does nothing else. |

The channel itself needs no flag. It stays closed until an activation marker appears in a
system or developer message:

```
<<llama-snap enable secret=S>>                  unlocks, for this request
<<llama-snap snap secret=S [at=A] [off=N]>>     persist this position to disk
<<llama-snap ckpt secret=S [at=A] [off=N]>>     pin a checkpoint here
```

`A` is `user`, `agent` or `tool`, optionally with `-N` for the N-th previous one of that
kind. Without `at=`, the marker addresses its own position. `off=N` shifts the position
`N` tokens back, for a harness that knows its own suffix is volatile.

Markers are **stripped before the template renders — but only once they have proved
themselves.** The activation in the system prompt is cut (the harness wrote it, and it
carries the secret). A marker in a user message is cut only when the request is activated
*and* the secret matches. Anything else stays in the text exactly as it arrived.

That asymmetry is a security property, not an oversight. Stripping every marker hands an
attacker a text transformation that runs **after** any inspection upstream:

```
ignore<<llama-snap a>> all<<llama-snap b>> previous instructions
```

A filter looking for that sentence does not see it; the model would receive it assembled.
Leaving an unproven marker verbatim means the model reads exactly the bytes the user sent,
which is the only thing that makes an upstream check worth running. The price is that a bogus
marker is visible to the model — as the user's own text, which is what it is.

A marker that *is* cut and sat on its own line takes the newline with it, so the render is
byte-identical to the same conversation without the channel.

`ckpt` on a position already passed needs a roll-back. On a model that needs checkpoints
that means a checkpoint at or before it, and without `--ctx-checkpoints` there is nothing to
roll back to, so such a hint is dropped. On a plain global-attention model the position is
reached by truncating instead, and no checkpoint is required — but the re-prefill budget
applies either way.

A hint names a position in **characters**; a checkpoint lives at a token boundary, and the
token spanning the seam depends on what follows it, which is the part not yet written. So a
hint is treated as an upper bound and backs off one token.

### Telemetry

| Flag | Default | |
|---|---|---|
| `--telemetry-file F` | off | One JSON line per request, appended. |
| `--telemetry-prompt-dir D` | off | Three files per request including the full prompt as text and token ids. **Conversation content in clear on disk.** For diagnosing one question, then remove the flag. |
| `--telemetry-prompt-max-gb N` | `8` | Cap for that directory, seeded from what it already holds, so it survives a restart. `0` = no cap. |

## Reading the telemetry

The fields worth watching, out of a line that carries the full picture:

| Field | |
|---|---|
| `hit` | `full` · `live` · `checkpoint` · `snapshot` — which reuse path served the request. A `full` on a follow-up turn is what you are looking for. |
| `n_cached` / `n_new` | tokens reused vs. prefilled. `n_new` near 0 on a follow-up is the goal. |
| `lcp` / `restore_at` | common prefix with what the slot held, and the position actually resumed from. `restore_at == lcp` means nothing was recomputed. |
| `probe_done` / `probe_kind` / `probe_miss` | whether the probe placed anything, which future it committed to, and how often it was wrong. `probe_miss` reaching 2 means the probe has given up on that slot — the hint channel is the answer. |
| `div_offset` | the distance from the prompt end where divergence was actually found, after a miss. Dropped again after two follow-ups that did not land on it. |
| `aborted` | the client cancelled before the request finished. Such a line carries the state the slot was left in; a `full` on the *next* request of that session is the cost of the cancel. |
| `snap_saved` / `snap_loaded` / `ms_snap_load` | files written, loaded, and what the load cost. |
| `hints` / `hints_dropped` / `hint_reprefill` | hints seen, dropped, and the re-prefill a roll-back cost. |
| `ms_queue` / `ms_prepare` / `ms_restore` / `ms_prefill` / `ms_gen` | the phase timeline. `ms_prepare` is where `--prefill-defer-above` shows up as waiting. |
| `cls` / `n_grow` / `cont` | slot class (1 = one-shot, 2 = chat), growing turns served, whether this was a continuation. |
| `idx_size` / `idx_lcp2` / `idx_lcp3` | prefix index occupancy and the fork depths found. |
| `merged` / `kv_depth` | cells folded by prefix sharing, and the pool scan depth at the time. |

Quick looks:

```bash
# how many requests fell back to a full prefill
jq -r .hit requests.jsonl | sort | uniq -c

# tokens recomputed per follow-up turn
jq -r 'select(.n_grow>0) | "\(.n_new)\t\(.hit)\t\(.probe_kind)"' requests.jsonl

# slots where the probe gave up
jq -r 'select(.probe_miss>=2) | .slot' requests.jsonl | sort -u
```

## Which model classes this works on

All of them, through the same gate — which is the point, not a detail.

| Class | Checkpoints | How a position is reached |
|---|---|---|
| hybrid / recurrent | carry state (155 MiB each) | load the checkpoint |
| iSWA | carry state (800 MiB each) | load the checkpoint |
| plain global attention | **position only**, no state | `seq_rm` truncates to it |

The positions are created on every model. On a global-attention model that looks like waste —
it supports partial removal, so any position is reachable anyway — but the position is what
decides whether a slot may be taken as a **continuation**. Leave it out and a slot matches on
the raw common prefix, which is non-zero for any prompt sharing three tokens: the smallest
common prefix captures a chat slot, and because a takeover is a continuation and not an
eviction, that session is overwritten without its snapshot ever being written.

**Prompts with media**: both work, with one condition. Probe and hint alike measure a distance
from the *end* and let the server supply the token count, so neither has to reproduce a stream
that interleaves image chunks. The condition is that **no media sits between the hinted
position and the end of the prompt** — those tokens would be counted as their marker text
rather than their real length. Media *before* the position, which is where images actually
are, changes nothing. A hint that fails the condition is refused with a reason in the
telemetry, not silently misplaced.

## Two models on one machine

Both may point `--snapshot-path` at the same directory; the model is folded into the file
name hash. Give the second unit an `After=` on the first — two models loading at once race
for the same memory, and on a unified-memory machine that ends in an OOM restart loop
rather than a slow start.

## What needs `--kv-unified`

Prefix sharing and the garbage collection of dead slots only mean anything under a unified
KV cache; with separate streams the patches switch both off by themselves, sharing with a
warning. Everything else works either way. See the table in the README.

## Sizing

Host RAM, on top of the KV cache:

```
checkpoints  = --ctx-checkpoints × slots × (155 MiB recurrent | 800 MiB iSWA)
prefix index = --prefix-index-size × --prefix-max-compare × 4 B   (worst case, 131 MB at defaults)
```

Disk:

```
snapshot     = fixed + per-token × position
               hybrid recurrent, f16-K/q8_0-V : 157 MB + 54.3 KB/token
               iSWA, f16 KV                   : 839 MB + 81.9 KB/token
```

The fixed part does not shrink with position — a 600-token snapshot on an iSWA model still
costs 839 MB. That is what `--snapshot-min-tokens` and `--snapshot-min-gap` exist for.
