# llama-server slot patches

**Persistent prefix caching, checkpoints placed where the next prompt will actually need
them, and a scheduler that stays responsive under load — for `llama-server`.** ⚡

Three patches. **Every part is individually switchable at runtime** — turn one thing on,
measure it, keep what holds. A server built with all three and started without the new
flags behaves exactly like upstream.

**Nothing changes on the client side.** No API additions, no configuration in your harness.
Existing clients keep sending the same requests; the server works out on its own which
prefixes are worth keeping and where a conversation will resume. A harness that *wants* to
say more can — see the hint channel below — but nothing requires it.

## 🧩 Features

**♻️ Persistent prefix caching.** The full sequence state — attention KV, recurrent state,
speculative state — is written to disk for a prefix that two different conversations both
start with, and loaded again by any later prompt that starts the same way. No configuration
per prompt, no manual save/restore calls, and it survives a restart. **A 1787-token
preamble loads in 300 ms where prefilling it costs 6.1 s.**

**🎯 Checkpoints where the next prompt diverges — computed, not guessed.** Where a
follow-up turn parts from the current one depends entirely on how the client renders the
history, and nothing in a request says which way it does it. So the server renders the
template twice more per request with a dummy continuation appended and takes the first
token where the streams differ. That *is* the divergence point of the next request. No
model runs; the cost is **4.4 ms on a 3.4 KB render, 10.9 ms on 64 KB**, against seconds of
prefill it saves. **0 tokens recomputed on every follow-up turn** on eight of the ten
harness shapes in the test bed — the two exceptions rewrite their own history and are
unpredictable from a template by construction; that is what the next section is for.

**📣 An in-band channel for harnesses that know better.** A recursive harness that wraps
its whole history in one message, or rewrites it between turns, produces a next prompt no
template render can predict. Such a harness can say where it will resume, in-band, with no
API change:

```
<<llama-snap enable secret=S>>            in the system prompt: unlocks the channel
<<llama-snap ckpt secret=S>>              pin a checkpoint here
<<llama-snap snap secret=S at=user-1>>    persist this position to disk
```

Markers are stripped before the template ever sees them, so **the model never receives one
and a harness with the channel renders byte-identically to one without**. Measured on the
harness this exists for — the whole history wrapped into one message: **168 tokens
recomputed over three follow-up turns without it, 0 with a single `ckpt` marker.**

**⚡ Decode priority under load.** A long prefill does not starve everything else. While
any slot is generating — or a smaller prefill is waiting — prefills above a threshold
contribute nothing to the batch. They are paused, not slowed, and the batch size is left
alone.

**💰 Cost-aware slot management.** Slots are picked by what it costs to rebuild them, not
by who waited longest. Two classes: a one-shot and a chat, the latter from its second turn
on. A fresh one-shot gets a soft protection window in which it outranks every other
one-shot but never a chat. Dead slots are collected, so their KV depth stops slowing
everyone else down.

**🔗 Prefix sharing between slots.** Two live sessions with a common prefix keep one
physical copy of it. Under a unified KV cache this copies no data at all — it only updates
the cell bitmap.

**📈 Per-request telemetry.** One JSON line per request with the phase timeline, which
reuse path was hit, what the probe predicted and whether it held — so you can check whether
any of this fires on *your* traffic instead of trusting these numbers.

## 🏗️ Built on what was already there

Almost none of the machinery here is new. The llama.cpp community built all of it:

- **context checkpoints** (`--ctx-checkpoints`) — snapshots of a sequence at a position,
  already handling recurrent and hybrid memory
- **sequence state serialisation** (`llama_state_seq_get_data` / `_set_data`) — a complete,
  versioned on-disk format for a single sequence, including draft and speculative state
- **`PARTIAL_ONLY`** — a flag that isolates the recurrent part of a hybrid cache, and the
  SWA part of an iSWA cache, which is exactly what you need to save one without disturbing
  the other
- **`seq_cp` on a unified cache** — copies no data at all, only updates the cell bitmap
- **slot reuse with longest-common-prefix matching**, continuous batching, the whole slot
  abstraction

What was missing was a **policy**: where to put a checkpoint, which prefix deserves a file,
which slot to recycle, and who goes first when a long prefill meets a short request. The
defaults answer those conservatively — sprinkle checkpoints at batch boundaries, evict the
least-recently-used slot, fill the batch in slot order — and those answers cost real time
and memory once prompts get long.

So these patches add almost no capability. They mostly decide *when* to use the
capabilities that were already there.

## ♻️ About the prefix index

The idea is borrowed: [SGLang](https://docs.sglang.ai/)'s RadixAttention keeps a radix tree
over the prompts it has seen and finds the longest matching prefix by itself. What is
different here follows from running on **one box** instead of a cluster: a cached prefix is
gigabytes and seconds, not kilobytes, so the question is not *how* to cache but *what
deserves it*.

The unit is the **preamble**: a prompt's tokens up to the end of its first user message,
taken from the parsed message spans. That boundary is the whole design.

Before it lies what several sessions genuinely have in common — the system prompt, the tool
definitions, the document the first question carries. After it lies one conversation: its
own answers, its own follow-ups. Where *that* diverges is computed per request by the probe
or declared by a hint; it is not something an index across sessions can know, and entries
reaching into it are entries nothing will ever match.

So the index is a list of **distinct preambles**, and **one fork is the whole rule**: if
another preamble agrees with this one down to some depth and then parts, that shared depth
is a prefix two different conversations both start with, and it has earned a file.

One fork, not several. A higher threshold exists elsewhere to tell a genuine fork from a
chat extending itself — a chat's next prompt always parts from its previous one after a few
tokens, which looks like a fork and is not. Truncating at the first user message removes
that case at the root: every turn of a session has the same preamble, so a session cannot
fork against itself at all. **Identical preambles are uninteresting**: there is nothing to
discover in them, no position where anything parts.

Two more rules keep the SSD writes rare: `--snapshot-min-tokens` refuses prefixes too short
to be worth a file, and `--snapshot-min-gap` forbids a second file right behind an existing
one, so `"Who…?"` and `"What…?"` do not each get their own multi-GB copy. Plus an LRU disk
budget you set.

**Several servers can share one snapshot directory.** The file name carries a hash seeded
with the model's identity — size, parameter count, vocabulary size, description — so two
models cannot collide, and one LRU budget covers them both.

## 🎯 How the probe decides

Two renders per request, on the string the server was going to build anyway:

| probe | continuation appended |
|---|---|
| `user_think` | the next prompt carries this turn's reasoning |
| `user_nothink` | it does not |
| `user_drop` | it carries this one but has dropped the previous turn's |
| `tool` | a tool result follows instead of a user turn |

The futures are named **absolutely**, by what the next prompt will contain — never relative
to what was observed, because the observed policy differs on the first turn, where there is
no assistant message at all. Every distinct answer gets a checkpoint while it is still
possible; once a follow-up lands exactly on one of them, that future is established and the
others stop being placed.

**A prediction is verified against the truth.** The common prefix of the next prompt is
computed anyway, so checking costs nothing. An append predicted as an append is a hit; a
divergence an existing checkpoint already reaches is a hit too, because the roll-back lands
on it and nothing is recomputed. Only an **unreachable** divergence is a miss. On a real
miss the server learns the distance from the prompt end at which the divergence sat and
anchors there from then on — the distance is the stable quantity, since it is a property of
the template and of how the harness re-renders a finished turn, while the absolute position
moves with the conversation. After two misses the probe stops placing anything for that
slot: a harness that rewrites its own history cannot be predicted from a template, and a
wrong checkpoint displaces a right one.

Positions are **exact**. A character offset is never converted into a token position —
that would mean deciding what to do when the offset falls inside a token, and every such
decision is a rounding. Instead both sides are tokenised and the common token prefix is
taken. That is also the only formulation that catches the re-tokenisation seam: two strings
can share a character prefix and still tokenise differently across it, which is the case
that costs a whole prompt.

## 📊 What you get

Measured on one machine, both models running at 262 144 context with a unified KV cache.
Full setups in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

| | |
|---|---|
| Prefilling a 1787-token preamble | 6.1 s |
| Loading the same state from disk | **0.3 s** |
| Same, after a service restart | **0.1 s** |
| Follow-up turn of a conversation | **0 tokens recomputed** |
| Follow-up turn, harness the template cannot predict | 168 tokens → **0** with one hint |
| Probe cost per request | 4.4 ms at 3.4 KB render, 10.9 ms at 64 KB |
| Test bed, 10 harness shapes × 5 templates | **50 combinations, 1 finding** — costing 0 tokens |

Snapshot size is linear and worth knowing before you set a budget:

| | fixed | per token | at 16k | what the fixed part is |
|---|---|---|---|---|
| hybrid recurrent, f16-K / q8_0-V | 157 MB | 54.3 KB | 1.0 GB | 48 Gated-DeltaNet states in f32 |
| iSWA, f16 KV | 839 MB | 81.9 KB | 1.5 GB | the sliding-window caches |

The fixed part is the reason this class of model needs snapshots at all: a recurrent state
cannot be sliced by prefix, and an SWA window rotates. There is no seeking back into either
— a checkpoint or a file is the only way to return. The same asymmetry shows up in
checkpoints, which are `PARTIAL_ONLY` and therefore carry only that fixed part: **155 MiB
on the recurrent model, 800 MiB on the iSWA one**, regardless of position. Budget
`--ctx-checkpoints` accordingly; it is per slot.

## 🔌 What needs `--kv-unified`

| Part | Unified KV | Separate streams |
|---|---|---|
| Decode priority | yes | yes |
| Checkpoint placement, probe, hints | yes | yes |
| Prefix index and disk snapshots | yes | yes |
| Slot classes, continuation picking, eviction dump | yes | yes |
| Garbage collection of dead slots | yes | **off** — pointless, each stream is private |
| Prefix sharing | yes | **off** — enforced, with a warning |
| Pool filled from the bottom | the point of it | harmless but pointless |

With separate streams the `n_kv` scan depth is per stream, so a slot's own occupancy only
ever costs itself — reclaiming it frees nothing for anyone else. The patches detect this
and switch those two parts off by themselves. The price of separate streams is that
`n_ctx_slot` is divided statically.

## 🚀 Quick start

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b78a39a2f            # or a newer commit
../llama-server-slot-patches/scripts/apply.sh upstream-b78a39a2f .
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j
```

Nothing changes until you set a flag. A starting configuration and the switch for every
individual part are in the [RUNBOOK](RUNBOOK.md).

Verified against `ggml-org/llama.cpp` at `b78a39a2f` (build 10916): the cascade applies to
a clean tree with zero failed hunks, every step compiles on its own, and the end state is
bit-identical to the reference tree.

## 📦 What the three patches do

| Patch | Lines | Applies on its own |
|---|---|---|
| `01-kv-pool-placement` | 67 | yes |
| `02-scheduler-decode-priority` | 143 | yes |
| `03-slot-management` | 4958 | needs 02 |

**01 — fill the pool from the bottom.** `find_slot` resets the head to 0 as soon as the
pool has holes, and the restore path fills from the front instead of insisting on a
contiguous block high up. The reason is the global `n_kv`: appending on top raises the scan
depth for everyone, and `used_max_p1()` only drops when the topmost cells are freed.
Measured performance-neutral.

**02 — absolute priority for decode.** Two rules. The slot loop runs in order of
*remaining* prompt tokens, ascending. And while any slot is generating or a smaller prefill
is waiting, prefills with more than `--prefill-defer-above` tokens left contribute nothing
to the batch.

**03 — slot management.** Checkpoint placement and the probe, the prefix index and disk
snapshots, the in-band hint channel, cost-aware slot eviction with a class ordering and
garbage collection, prefix sharing between slots, and full per-request telemetry. These are
intertwined in the code — the probe decides where checkpoints go, the eviction dump writes
the position the probe computed, the hint channel overrides both, and telemetry reads all
of it — and therefore ship as one patch. **Each part still has its own runtime switch**, no
rebuild needed. Also carries the test bed under `tools/server/tests-snap/`.

## 🧪 Tested with

| Model | Architecture | Snapshots |
|---|---|---|
| Qwen 3.8 27B | hybrid: 48 Gated DeltaNet (recurrent) + 16 attention layers | **bit-exact** |
| Gemma 4 31B | iSWA (sliding-window attention, no recurrent state) | works, output equivalent but not token-identical |

**This helps most on hybrid and recurrent models.** There is no KV cache you can simply
seek back into: without a checkpoint, a change at the tail of the prompt costs the *whole*
prompt. On attention-only models the gain is smaller but real — checkpoints and snapshots
still save the re-prefill — and the scheduler, slot eviction and prefix sharing are
independent of the architecture.

The test bed walks the space the design actually depends on: ten harness shapes — history
preserved, stripped, only the most recent kept, wrapped in one message, tool loops with and
without reasoning, a harness whose prompt end gets re-tokenized as it grows, one that
summarises its own history, one that switches policy mid-conversation, one declaring its
resume point through the hint channel — across five chat templates.
`tools/server/tests-snap/` — the LLM is simulated there, so a run is deterministic and
tests the template and the tokenizer rather than the model.

Of those 50 combinations, 49 are clean and one carries a finding: on a tool loop that keeps
its reasoning in history, the probe never narrows from two candidate futures to one. It
spends a checkpoint slot it does not need; it recomputes nothing, because the live slot
still holds the prefix. Details in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md#test-bed).

## ⚠️ Before you turn any of this on

**Snapshot files contain the raw KV state of user prompts.** That is conversation content
in reconstructible form on disk, unencrypted. `--telemetry-prompt-dir` writes prompts as
plain text. Both are off by default.

**On SWA models everything works, but not bit-identically.** All of it runs on iSWA models
such as Gemma 4 — probe, checkpoints, eviction with rollback, restore, GC, classes,
sharing. Restored output is equivalent in substance but not token-identical, because the
SWA cache rotates during prefill and is restored linearly. In practice that is fine for
chat and agent work. If you need byte-identical replay of a session, either keep snapshots
off for that model or use a non-SWA one.

**Checkpoints are the memory cost to watch**, and it differs by an order of magnitude
between architectures: 155 MiB each on a recurrent model, 800 MiB on an iSWA one, times
`--ctx-checkpoints`, times the number of slots. That is host RAM, on top of the KV cache.

**A known risk in 02.** A long prefill yields to *every* decode. Under continuous traffic
it can wait a long time; a safety valve is not implemented. Watch `ms_prepare` in the
telemetry.

**All numbers come from one machine.** An APU with unified memory, measured with the two
models above. The mechanisms are general, the magnitudes are not transferable.

## 📄 License and thanks

MIT, like llama.cpp. These patches are derived work on MIT-licensed code.

Thanks to everyone who built `llama-server` and the KV cache machinery underneath it. The
interesting parts of this repository are decisions about code somebody else wrote well.
