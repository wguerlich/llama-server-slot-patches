# Runbook

**Every part here is individually switchable**, and each one starts out matching upstream
behaviour. That is deliberate: turn one thing on, measure it, keep what holds — nothing changes
until you ask for it.

## At a glance: how to exclude each part

| Part | Patch | Turn on with | Off (default) | Rebuild needed? |
|---|---|---|---|---|
| Coalesced restore | 01 | always active | leave the patch out | yes |
| Fill pool from bottom | 02 | always active | leave the patch out | yes |
| Decode priority | 03 | `--prefill-defer-above N` | `0` | no |
| Turn-boundary checkpoints | 04 | `--ctx-checkpoint-tail N` | `0` | no |
| Slot eviction + GC | 04 | `--slot-evict-policy cost` | `lru` | no |
| Prefix sharing | 04 | `--kv-share-min N` | `0` | no |
| Prefix snapshots on disk | 04 | `--snapshot-path DIR` | empty | no |
| Request telemetry | 04 | `--telemetry-file FILE` | empty | no |
| Plain-text prompt dump | 04 | `--telemetry-prompt-dir DIR` | empty | no |

**01 and 02 are the only parts without a runtime switch** — they change KV cache behaviour
unconditionally. To exclude them, leave the patch out: `apply.sh <variant> <tree> 03` applies
01–03 and skips 04; to skip 02, remove that file from the variant directory.

---

## 03 — decode priority

**What it does.** Two rules. The slot loop runs in order of *remaining* prompt tokens, ascending,
instead of slot index order. And while any slot is generating **or a smaller prefill is waiting**,
prefills with more than `N` remaining tokens contribute **nothing** to the batch — paused, not
slowed. The batch size is left alone.

**Why.** `update_slots()` builds one batch per iteration and issues one `llama_decode`; a prefill
chunk fills the budget and each generating slot advances by one token. And a slot waiting for its
*own* prefill gets nothing at all in slot index order, because the first slot with pending prompt
tokens takes the whole budget in every iteration.

```
--prefill-defer-above 2048     # 0 = off
```

**Choosing the threshold.** It decides two things: how long a newcomer blocks the decoders while
its own prompt is prefilled (2048 tokens ≈ 7 s at 300 t/s), and from which size a prefill has to
yield. Too small means ordinary requests get classified as "large" and wait: with 2048, requests
carrying 2 226 / 3 110 / 7 177 fresh tokens were measured waiting **50 to 113 s**. If your traffic
has many medium prompts, go higher.

**Remaining, not original.** The classification computes `task->n_tokens() - prompt.n_tokens()`. A
large prefill therefore becomes "small" itself at some point and gets priority rather than being
throttled just short of the finish line; and a warm follow-up turn (large prompt, few fresh tokens)
counts as small from the start.

**Do not confuse this with `--batch-size`.** Lowering the batch size helps *decode* (its tokens are
added first anyway), but a waiting *prefill* stays out as long as the same slot takes the whole
budget every iteration — independent of the batch size. Capping the prompt tokens per iteration was
measured and it **plateaus**: 64/32/16 all land at tg 4.7 t/s, because ~0.21 s is the fixed cost of
an iteration at that depth, and 256 costs 17 % of weighted total throughput. That is why this
approach is not in the patch.

**Risk.** A long prefill yields to *every* decode. Under continuous traffic it can wait a long
time; a safety valve is **not** implemented. Watch `ms_prepare` in the telemetry.

**Verify.** `grep -c prefill_defer_above tools/server/server-context.cpp` (expect ≥ 5), and
`prefill-defer-above = N` in the journal at startup.

---

## 04 — turn-boundary checkpoints

**What it does.** Checkpoints are created at turn boundaries instead of batch boundaries. The
server remembers the last `N` tokens of each prompt and places a checkpoint where that sequence
reappears in a later prompt.

```
--ctx-checkpoint-tail 32       # 0 = off
--ctx-checkpoints 2            # upstream flag; 2 is enough with turn boundaries
--checkpoint-min-step 0        # upstream flag; a minimum spacing is pointless now
```

**Why two checkpoints are enough.** Exactly two are created per prefill (the tail match and L−4).
Measured identical to three, saving 0.8–2.5 GB of host RAM. Checkpoint size is context dependent,
measured linear over 7 points: **112.6 MiB + 2.023 KiB per token** — 630 MiB at a full 262k
context, **per slot**. Upstream's default of 32 checkpoints would be 80 GB across 4 slots.

**A limit worth knowing.** If the rendered prompt end gets re-tokenized as the conversation grows
(see the seam in the README), the tail match **never** fires. Over 19 measured turns the checkpoint
list was then always just `[L−4]`. That is not a failure of the mechanism, but it means L−4 is what
you end up relying on.

---

## 04 — slot eviction, classes and GC

**What it does.** Instead of LRU, a class ordering decides which slot gets recycled, and a GC
reclaims dead slots as soon as work resumes.

```
--slot-evict-policy cost       # lru = upstream behaviour (default)
--slot-evict-ttl 600           # seconds of idleness after which a slot is reclaimed
--slot-evict-grace 30          # how long a fresh single-shot slot counts as a young chat
--slot-max-active 0            # 0 = no cap on concurrently computing slots
```

**Classes.** 0 = empty slot, 1 = single-shot (`n_grow == 0`), 2 = young chat, 3 = chat. Within a
class, the one idle longest. A slot past the TTL is reclaimed directly regardless of class — and
gets its snapshots on the way out if snapshots are on.

**Why reclaim at all.** `n_kv` is global under a unified KV cache. A 500-token request next to an
**idle** 36k session decodes at 16.6 instead of 22.0 t/s. The GC does **not** run while everything
is idle — the depth costs nobody anything there, and reclaiming would discard contexts that are
still wanted.

**Continuation.** A slot is taken as a continuation if anything in it is reachable at all (a
checkpoint at or before the common prefix), and among the candidates the cheapest prefill wins. A
snapshot on disk that reaches further takes precedence. Strangers stay out because checkpoints sit
at prompt ends: a stranger sharing the same document diverges where the *question* begins and finds
no checkpoint (measured: `lcp 2446` against checkpoints `[2478,2501]` → rejected).

**`--slot-max-active`** caps how many slots compute at once. **Careful:** it also caps pure
decoders, and batched decode is nearly free per additional sequence — with a cap of 2 the aggregate
was 1.49 t/s, without a cap 42.78 t/s across three decoders. Only set it for a concrete reason.

---

## 04 — prefix sharing between slots

**What it does.** After the prompt is decoded, the attention cells of a prefix shared by two slots
are folded onto **one** physical copy via the cell bitmap.

```
--kv-share-min 2048            # 0 = off; requires --kv-unified
```

**Why it is cheap.** Under a unified KV cache `seq_cp` copies no data, it only sets bits. The
recurrent state is saved around the operation with `PARTIAL_ONLY`, because `seq_cp` overwrites it.

**Measured.** Three concurrent sessions holding **9624 logical tokens in a 12288-cell pool**, merge
23–53 ms.

**Only helps with concurrent sessions.** Without a unified KV cache it is off automatically (with a
warning in the journal), because each stream is private there.

---

## 04 — prefix snapshots on disk

**What it does.** The complete sequence state is written to disk at a position in the middle of a
prefill, and loaded again for later prompts that start with the same prefix.

```
--snapshot-path DIR            # empty = off
--snapshot-max-disk-mb 16384   # LRU budget
--snapshot-min-hits 3          # divergence = a node with this many DISTINCT children
--snapshot-min-branch 8        # shorter arms are stubs and do not count
--snapshot-min-prefix 512      # shorter prefixes are not indexed
--snapshot-max-depth 32768     # caps the index AND the creation position
--snapshot-index-size 1000     # RAM: index-size x max-depth x 4 bytes
--snapshot-min-gap 100         # no new snapshot this close BEHIND an existing one
--snapshot-evict-turns 3       # a slot with this many turns is saved when it loses its content
--snapshot-evict-points 7      # bitmask: 1 end of generation, 2 newest, 4 second-newest checkpoint
--snapshot-evict-learn         # keep only the points this session actually resumed from (off by default)
--snapshot-at 0                # testing only: fixed position, no index
```

**Measured.** 12.5k tokens load in **1.7 s instead of 32 s**, bit-identical to the producing run,
and it works for the prefix *before* an image.

**The divergence rule is strict on purpose.** A snapshot is only created at a node of the prompt
tree with `min-hits` **distinct children**. A single session rewriting its own history produces
two-armed forks only and can never fill the cache.

**Its limit, measured.** At template boundaries that many prompts pass through there are often only
two possible continuations. A node with **16 prompts and 2 children** is unreachable for
`min-hits = 3` — and that is precisely the most valuable node. Lowering `min-hits` to 2 opens the
door for a single session to flood the cache.

**RAM and disk.** The index is volatile and rebuilt at startup; the *files* on disk are read at
startup and stay usable. `index-size 2000 x max-depth 32768 x 4 B` = 262 MB. File size is about
84 MB fixed plus ~41 KB per token, measured from the pair 3956 → 248 MB and 6075 → 336 MB. A 159k
snapshot is therefore ~6.7 GB, and `evict-points 7` writes three of them.

**`--snapshot-evict-learn` — the same idea applied to the client.** A client renders the history
the same way every turn, so a state that no turn of a session ever resumed from will not be needed
after eviction either. With this on, an eviction dump keeps only the points the session actually
used. Measured on one machine over 166 requests: **23 of 23 checkpoint hits were the newest one and
the second-newest never fired** — two of the three snapshots would have been written for nothing.
Verified end to end on a five-turn chat:

```
[gc] releasing slot: class 3, n = 3279, grow = 4, idle 54s (ttl)
[snapshot] learned access pattern [live 0, newest 4, 2nd 0]: points 0x7 -> 0x2
[snapshot] slot leaves after 4 growing turns: keeping 1 of 3 states
[snapshot] saved 3176 tokens, 232.1 MB in 89 ms
```

One file instead of three. The mask is only ever **narrowed**, never widened — a bit outside
`--snapshot-evict-points` can never appear — and a session with no informative turn yet (a single
turn, or one that only ever hit disk snapshots) narrows nothing and gets every configured point.
The telemetry carries `evict_hits[3]` and `evict_points_learned` so you can see what a session
learned before it is evicted.

**Privacy.** These files contain the raw KV state of user prompts.

---

## 04 — request telemetry

**What it does.** One flat JSON line per finished request, ~1 kB, **without conversation content**.

```
--telemetry-file FILE          # empty = off
--telemetry-prompt-dir DIR     # empty = off; writes PLAIN TEXT prompts
--telemetry-prompt-max-gb 0    # 0 = no cap
```

Fields: absolute timestamps per phase (`t_arrive_us`, `t_start_us`, `t_prefill_us`, `t_gen_us`,
`t_end_us`), durations (`ms_queue`, `ms_prepare`, `ms_restore`, `ms_prefill`, `ms_gen`, `ms_total`,
`ms_snap_load`, `ms_merge`), which reuse variant was hit (`hit` = full/live/checkpoint/snapshot,
plus `ckpt_rank`, `ckpt_pos`, `var_hits[4]`), every input of the decision logic (`cls`, `n_grow`,
`cont`, `next_turn`, `lcp`, `slot_len`, `prompt_last`, `restore_at`, `snapshot_at`,
`reprefill_expected`, `f_keep`, `merged`, `kv_depth`, `slots_used`, `checkpoints[]`, `idx_lcp2`,
`idx_lcp3`, `idx_size`), and the `picker` trace with one line per candidate slot.

`-1` means "this phase did not happen". `kv_depth = -1` on a fresh slot is **correct** —
`pos_max` returns −1 for an empty sequence.

**Without the other parts, fields stay empty.** Telemetry observes what is there: without
snapshots `snap_loaded` stays 0, without eviction `cls` is always 0.

**`--telemetry-prompt-dir` writes conversation text in the clear**, plus the token ids. The token
ids are the reason it exists: only with them is a seam visible whose plain text is identical.
Growth is about 8 bytes per prompt token, so ~0.8 MB per 100k-token request.

---

## A starting configuration

For a server with mixed traffic — long, growing sessions plus short requests — and a unified KV
cache:

```
--kv-unified --parallel 6
--ctx-checkpoints 2 --checkpoint-min-step 0 --ctx-checkpoint-tail 32
--slot-evict-policy cost --slot-evict-ttl 600 --slot-evict-grace 30
--prefill-defer-above 2048
--kv-share-min 2048
--telemetry-file /var/lib/llama/requests.jsonl
```

Turn snapshots on **later**, on purpose: collect telemetry first, then check whether the divergence
rule fires at all on your prompts (`idx_lcp2` and `idx_lcp3` in the telemetry show it), then set
`--snapshot-path`.
