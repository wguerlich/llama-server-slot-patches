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
| `--ctx-checkpoints N` | `32` | Checkpoints per slot. **This is the memory knob**: 155 MiB each on a recurrent model, 800 MiB on an iSWA one, times the slot count. With the probe placing them deliberately, `2` is usually enough; `32` with the probe on will exhaust host RAM. |
| `--ctx-checkpoint-probe on\|off` | `on` | Compute the next prompt's divergence point by rendering the chat template twice more per request. No model runs. `off` falls back to upstream's batch-boundary placement. |

### Prefix index and snapshots

| Flag | Default | |
|---|---|---|
| `--snapshot-path DIR` | off | Directory for disk snapshots. Setting it enables the prefix index too. |
| `--snapshot-max-disk-mb N` | `16384` | LRU disk budget. `0` = unlimited. Covers the whole directory, so several servers sharing one directory share one budget. |
| `--snapshot-min-tokens N` | `512` | Shortest prefix worth a file. A preamble below it is not indexed and an eviction dump skips positions below it. **Check this against your real prompts** — set above them, nothing is ever written. |
| `--prefix-min-forks N` | `1` | How many *other* preambles must agree down to a position and then part before it earns a file. One is the rule; see the README. |
| `--prefix-index-size N` | `1000` | Distinct preambles the RAM index remembers, LRU. Worst-case RAM is this × `--prefix-max-compare` × 4 bytes. |
| `--prefix-max-compare N` | `32768` | Tokens of a preamble kept for comparison. A longer preamble is still identified in full by hash; only the fork search is capped. |
| `--snapshot-min-gap N` | `100` | Never place a snapshot within `N` tokens *behind* an existing one. In front is always allowed. |
| `--snapshot-evict-turns N` | `0` (off) | When a slot that served ≥ `N` slot-matched turns with growing prompts loses its content, snapshot it at its decoded length and, rolling back, at every checkpoint. `2` is a good starting point. |
| `--snapshot-at N` | `0` (off) | Test override: snapshot every from-scratch prompt at position `N`, bypassing the index. For experiments only. |

A file name carries a hash seeded with the model's identity — size, parameter count,
vocabulary size, description — so **several servers can share one directory** without
colliding.

### Hint channel

| Flag | Default | |
|---|---|---|
| `--snap-hint-max-reprefill N` | `-1` (= `n_ubatch`) | How many tokens of re-prefill a hint may cost when its position has already been passed. The server rolls back to a checkpoint at or before the hinted position and prefills forward again; a costlier hint is dropped, with the reason in the telemetry. `0` = never roll back. |

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

Markers are **stripped before the template renders**, unconditionally — including one
carrying a wrong secret, so a mismatch never leaks a marker to the model. A marker on its
own line takes the newline with it. The render is byte-identical to the same conversation
without markers.

`ckpt` on a position already passed needs a roll-back, and a roll-back needs a checkpoint
at or before it. Without `--ctx-checkpoints` there is nothing to roll back to and such a
hint is dropped.

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
| `div_offset` | the distance from the prompt end where divergence was actually found, after a miss. |
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
