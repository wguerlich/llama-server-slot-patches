# llama-server slot patches

**Automatic persistent prefix caching, automatic long session restore, and a scheduler that stays
responsive under load — for `llama-server`.** ⚡

Four patches. **Every part is individually switchable at runtime** — turn one thing on, measure it,
keep what holds. Nothing is forced on you: a server built with all four and started without the
new flags behaves exactly like upstream.

**Nothing changes on the client side.** No API additions, no special configuration in your harness,
no per-request hints, no "cache this" markers. Existing clients keep sending the same requests;
the server works out on its own which prefixes are worth keeping and where the turn boundaries are.
Any chat template, any framework, any agent loop.

## 🧩 Features

**♻️ Automatic persistent prefix caching.** Prompts that share a prefix stop paying for it twice.
The full sequence state — attention KV, recurrent state, speculative state — is written to disk at
positions where prompts *demonstrably diverge*, and loaded again for any later prompt that starts
the same way. No configuration per prompt, no manual save/restore calls, and it survives a restart.
Bit-exact on non-SWA models. **12.5k tokens load in 1.7 s instead of 32 s of prefill.**

**⚡ Decode priority under load.** A long prefill does not starve everything else. Requests that
only need to generate keep running at full speed while a 130k-token prompt is being processed, and
a small request that arrives gets its first token in seconds instead of minutes. **Aggregate
decode 1.49 → 42.78 t/s, time to first token 25.3 s → 1.4 s.**

**🎯 Turn-boundary checkpoints — and less RAM than the default.** Checkpoints land where the
conversation actually branches instead of at arbitrary batch boundaries, and the boundary is
*learned* from the prompt itself, so it works on any chat template without parsing it. Because they
are placed rather than sprinkled, **two per slot are enough**, where upstream's default is 32 —
which at full context is 630 MiB each, 80 GB of host RAM across four slots.

**🔄 Automatic long session restore — for agents and chats.** The server tells a conversation
apart from a one-off request, structurally, by whether a prompt is a genuine follow-up turn — not
by anything the client declares. Two things follow. A one-shot request never evicts a live chat.
And when a chat's slot *is* needed for something else, its state goes to disk first — at the
position that conversation will actually resume from, which the server works out by itself. So a
long conversation stays resumable: hours later, after other traffic has cycled through every slot,
or after a server restart. **Returning to an evicted chat: 47.5 s → 0.3 s.**

Which position that is, is where a naive save-and-restore breaks. How far back a follow-up prompt
diverges depends entirely on how the client renders the reasoning of previous turns — three
possibilities, and nothing in the request says which one you are talking to:

| How the client renders history | Where the next prompt diverges | Which state has to be saved |
|---|---|---|
| **thinking preserved** | nowhere — it is a pure append | the state at the end of generation |
| **thinking stripped** | at the last turn boundary | the newest checkpoint |
| **most-recent thinking kept** | one turn boundary *earlier* | the second-newest checkpoint |

**So the server detects it.** Every slot records which of the three positions actually carried a
follow-up turn — that is the *harness type*, and it holds for the rest of the conversation, because
a chat talks to one client throughout. On eviction only the detected position is written: **one
file instead of three.** Measured over 166 requests, 23 of 23 checkpoint hits were the newest one
and the second-newest never fired once — two thirds of those writes would have been for nothing,
~13 GB of 20 GB for a single 159k-token session.

Detection only ever narrows, never widens. A session that has not yet shown its type still gets
all three, so nothing is lost on a first eviction and even a **client switch mid-conversation**
finds a state that fits. Measured: coming back as a pure append loaded the end-of-generation state
(`cached 2996`, 46 ms); coming back with only the last thinking kept loaded the second-newest
checkpoint (`cached 1774`). Detection is a switch (`--snapshot-evict-learn`), and
`--snapshot-evict-points` remains a plain bitmask if you already know your client and would rather
pin it by hand.

**And the file records the type it came from.** Every snapshot carries a length-prefixed *harness
info block* in its header, holding the harness type its writing session detected. It belongs in the
file rather than in a side table: how a client renders history is a property of the harness, and
the file is the only thing sessions of the same harness share.

```
"LSNP" u32 version | u32 info_bytes, info_bytes of harness info | u64 model_size ...
```

A reader takes the fields it knows and seeks past the rest, so **the block grows without breaking
any file that is already on disk**.

Note the division of labour: **every** prompt benefits from the shared prefix cache, one-time
requests included — that is where the system prompt, the tool definitions and the shared document
come from. Only *conversations* additionally get their own state persisted. That is deliberate:
it is what keeps the number of SSD writes low.

Upstream already saves idle slots — `--cache-idle-slots` is on by default and puts them in the
host-RAM prompt cache (`--cache-ram`, 8 GB by default). This works differently on purpose:

| | `--cache-idle-slots` + `--cache-ram` | this |
|---|---|---|
| Where the state goes | host RAM | disk |
| Survives a restart | no | **yes** |
| Which slots | every idle one | **only those that demonstrably were a conversation** |
| Which position | the state as it stands | **the three a follow-up prompt can land on** |
| Bounded by | MiB of RAM | LRU disk budget you set |

Being selective is the whole point. Saving every idle slot to disk would mean gigabytes of writes
for one-off requests that will never come back — a 159k-token session is ~6.7 GB. Restricting it
to long-running sessions is what makes disk persistence affordable at all. And the RAM variant has
a limit worth knowing: nothing gives that memory back short of a restart, so on a
unified-memory machine — where host RAM is the same pool the model lives in — we run with
`--cache-ram 0` and rely on the disk path instead.

**💰 Cost-aware slot management.** Slots are picked by what it actually costs to rebuild them, not
by who waited longest. A returning chat finds its context; a one-shot request does not evict one.
Dead slots get collected, so their KV depth stops slowing everyone else down. **Follow-up turn
after eviction: 47.5 s → 0.3 s.**

**🔗 Prefix sharing between slots.** Two live sessions with a common prefix keep one physical copy
of it. Under a unified KV cache this copies no data at all — it only updates the cell bitmap.
**Three concurrent sessions holding 9624 logical tokens in a 12288-cell pool.**

**📈 Per-request telemetry.** One JSON line per request with the full phase timeline, which reuse
path was hit, and every input of every decision — so you can check whether any of this fires on
*your* traffic instead of trusting our numbers.

## 🏗️ Built on what was already there

Almost none of the machinery here is new. The llama.cpp community built all of it, and it has been
sitting in the tree for a long time:

- **context checkpoints** (`--ctx-checkpoints`) — snapshots of a sequence at a position, already
  handling recurrent and hybrid memory
- **sequence state serialisation** (`llama_state_seq_get_data` / `_set_data`, `--slot-save-path`) —
  a complete, versioned on-disk format for a single sequence, including draft and speculative state
- **`PARTIAL_ONLY`** — a flag that isolates the recurrent part of a hybrid cache, and the SWA part
  of an iSWA cache, which is exactly what you need to save one without disturbing the other
- **`seq_cp` on a unified cache** — copies no data at all, only updates the cell bitmap; upstream
  even marks the spot `[TAG_KV_CACHE_SHARE_CELLS]`
- **the host-RAM prompt cache** (`--cache-ram`, `--cache-idle-slots`) — already saves idle slots
  automatically, which is the same instinct one level up
- **slot reuse with longest-common-prefix matching**, continuous batching, the whole slot
  abstraction

What was missing was a **policy**: where to put a checkpoint, which prefix deserves a file on disk,
which slot to recycle, and who goes first when a long prefill meets a short request. The defaults
answer those questions conservatively — sprinkle checkpoints at batch boundaries, evict the
least-recently-used slot, fill the batch in slot order — and those answers cost real time and
memory once prompts get long.

So these patches add almost no capability. They mostly decide *when* to use the capabilities that
were already there. Credit for the hard parts — the state format, the hybrid memory handling, the
cell bitmap, the batching — belongs upstream.

## ♻️ About the prefix caching

The idea is borrowed: [SGLang](https://docs.sglang.ai/)'s RadixAttention keeps a radix tree over
the prompts it has seen and finds the longest matching prefix by itself. That is what automatic
prefix caching should feel like, and `llama-server` had no equivalent — its prompt cache is per
slot, and `--slot-save-path` needs an explicit save and restore call for every state you want.

What is different here follows from running on **one box** instead of a cluster:

**🌳 Selection by divergence instead of caching everything.** A radix tree with LRU works when a
cached prefix costs kilobytes of GPU memory. Here a snapshot is gigabytes and seconds, so the
question is not *how* to cache but *what deserves it*. The answer is structural: **divergence is a
node of the prompt tree with N distinct children.** A single session rewriting its own history
produces two-armed forks only and can never flood the cache, however often it runs. Three
different conversations branching at the same position do — and that is exactly the prefix worth
keeping. `--snapshot-min-hits` sets N.

**💾 Less RAM than the usual setup, not more.** The common approach is to sprinkle checkpoints and
hope one of them lands usefully. Checkpoints are not cheap: measured linearly over 7 points,
**112.6 MiB + 2.023 KiB per token** — 630 MiB per checkpoint at full context, *per slot*.
Upstream's default of 32 would be 80 GB of host RAM across four slots. Because the boundaries are
learned rather than guessed, **two per slot are enough**: one to restore from, one that the next
turn will need. Fewer, better-placed checkpoints beat many arbitrary ones.

**🔒 Deliberately few SSD writes.** A snapshot of a long session is large (~84 MB fixed plus
~41 KB per token, so ~6.7 GB at 159k tokens). Writing those carelessly would wear the drive for
nothing, so three rules keep it rare: a snapshot needs a divergence node with N distinct children;
`--snapshot-min-gap` forbids a second one right behind an existing one, so `"Who…?"` and `"What…?"`
do not each get their own multi-GB file; and a slot losing its content is only saved if it
demonstrably *was* a conversation. Plus an LRU disk budget you set. The design goal was never
"cache as much as possible" — it was "write only what pays for itself".

**👂 Turn boundaries observed, not parsed.** The server remembers the last tokens of each prompt
and places state where that sequence reappears later. No delimiter tokens, no template knowledge,
no per-model tuning — and nothing for the client to declare.

**🧬 The whole sequence state, not just KV.** On a hybrid model — 36 of 48 layers recurrent in our
test — a per-token KV slice is not enough to resume: the recurrent state is not indexed by token
and cannot be sliced by prefix. The snapshots carry it, which is what makes them usable there at
all.

**📏 Everything measured.** Every number in this README comes out of the telemetry the patches
emit, on real traffic. `idx_lcp2` and `idx_lcp3` in each telemetry line tell you whether the
divergence rule fires on your prompts, before you turn snapshots on.

Where this does *not* compete: SGLang and vLLM are built for many concurrent requests and scale
accordingly. This is for a server with a handful of slots, where a single long prefill can block
everything and one persisted prefix can be worth minutes.

## 📊 What you get

| | Upstream defaults | With these patches |
|---|---|---|
| Decode rate while a long prefill runs | 1.49 t/s | **42.78 t/s** aggregate |
| Time to first token for a small request | 25.3 s | **1.4 s** |
| Weighted throughput `U` (prefill + decode) | 1.035 | **2.103** |
| Follow-up turn whose slot was recycled | 47.5 s | **0.3 s** |
| Restoring a 12.5k-token context from disk | 32 s prefill | **1.7 s** |
| Tail change on a recurrent model | 31 869 tokens, 97.0 s | **2 055 tokens, 8.0 s** |
| Three concurrent sessions in a 12288-cell pool | 3× full copies | **9624 logical tokens** |

Cost of the throughput gain: **10 % prefill**. Everything else is free or saves memory.

## 🔌 What needs `--kv-unified`

| Part | Unified KV | Separate streams |
|---|---|---|
| Decode priority (03) | yes | yes |
| Turn-boundary checkpoints (04) | yes | yes |
| Prefix snapshots on disk (04) | yes | yes |
| Slot classes, continuation picking, eviction dump (04) | yes | yes |
| Garbage collection of dead slots (04) | yes | **off** — pointless, each stream is private |
| Prefix sharing (04) | yes | **off** — enforced, with a warning |
| Pool filled from the bottom (02) | the point of it | harmless but pointless |

With separate streams the `n_kv` scan depth is per stream, so a slot's own occupancy only ever
costs itself — reclaiming it frees nothing for anyone else. The patches detect this and switch
those two parts off by themselves. The price of separate streams is that `n_ctx_slot` is divided
statically (measured: 43 648 instead of 261 888 tokens per slot at 6 slots).

## 🚀 Quick start

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout c32d1dabe            # or a newer commit
../llama-server-slot-patches/scripts/apply.sh upstream-c32d1dabe .
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j
```

Nothing changes until you set a flag. A starting configuration and the switch for every individual
part are in the [RUNBOOK](RUNBOOK.md); all numbers with their setups are in
[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

Verified against:

| Target | Version | Date |
|---|---|---|
| `ggml-org/llama.cpp` master | `c32d1dabe` | 2026-09-10 |
| `unslothai/llama.cpp` | `b10840-mix-d5c17a0` | 2026-09-08 |

Every step of the cascade compiles on its own, and the end state is bit-identical to the fully
patched reference tree.

---

## 📦 What the four patches do

| Patch | Lines | Files | Applies on its own |
|---|---|---|---|
| `01-kv-restore-coalesce` | 63 | 1 | yes (only for trees without the `runs` loops) |
| `02-kv-pool-placement` | 52–67 | 1 | yes |
| `03-scheduler-decode-priority` | 143 | 3 | yes |
| `04-slot-management` | 2606 | 6 | needs 03 |

**01 — coalesce runs on restore.** `state_read_data` issues one `read_tensor` per *run* of
contiguous destination cells rather than one read per cell, so the cost scales with the number of
gaps rather than the number of cells (1 run 245 ms, 24576 runs 10.6 s). **Trees from September 2026
on already have this** — the patch is for older ones.

**02 — fill the pool from the bottom.** `find_slot` resets the head to 0 as soon as the pool has
holes, and the restore path fills from the front instead of insisting on a contiguous block high
up. The reason is the global `n_kv`: appending on top raises the scan depth for everyone, and
`used_max_p1()` only drops when the topmost cells are freed. Measured performance-neutral.

**03 — absolute priority for decode.** Two rules. The slot loop runs in order of *remaining*
prompt tokens, ascending. And while any slot is generating **or a smaller prefill is waiting**,
prefills with more than `--prefill-defer-above` tokens left contribute nothing to the batch — they
are paused, not slowed. The batch size itself is left alone.

**04 — slot management.** Turn-boundary checkpoints, cost-aware slot eviction with a class
ordering and garbage collection, prefix sharing between slots, prefix snapshots on disk, and full
per-request telemetry. These five are intertwined in the code (the GC writes snapshots, the turn
counter feeds the snapshot rule, telemetry reads all of it) and therefore ship as *one* patch —
but **each part has its own runtime switch**, no rebuild needed. See [RUNBOOK.md](RUNBOOK.md).

---

## 🔍 Where the numbers come from

Four findings drove the design. Short versions; the full measurements are in
[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).

**Prefill and decode share one loop.** `update_slots()` assembles one batch per iteration and
issues one `llama_decode`, so a prefill chunk fills the budget and each generating slot advances by
exactly one token. Next to large prefills that means 0.18–0.97 t/s against a median of 17.1 t/s
alone. And a slot waiting for its *own* prefill got nothing at all: 669 s and 766 s measured
between slot assignment and the start of processing, with an empty queue.

**`n_kv` is global.** With a unified KV cache, the depth every attention kernel walks is the
maximum over all sequences. A 500-token request next to an **idle** 36k session decodes at 16.6
instead of 22.0 t/s — exactly the rate of that other session.

**One token can cost fifteen turns.** A session with a monotonically growing prompt re-prefilled
fifteen turns in a row — 1 064 874 tokens, 64 minutes. At the character level each follow-up was an
exact append; at the token level it was not, because the tokenizer merges two adjacent newlines
into one and the previous prompt's last token ceases to exist. Any client whose rendered prompt end
gets re-tokenized as the conversation grows hits this, in every single turn.

**Snapshots work if you put them in the right place.** 12.5k tokens load in 1.7 s instead of 32 s
of prefill, bit-identical to the run that produced them. The hard part is not the I/O, it is
deciding *where* a snapshot pays for itself.

## 🧪 Tested with

| Model | Architecture | Snapshots |
|---|---|---|
| Qwen3.8-Flash-Next 125B-A6B (`qwen4exp`) | hybrid: 36 Gated DeltaNet (recurrent) + 12 attention layers, MoE | **bit-exact** |
| Gemma 4 12B | iSWA (sliding-window attention, no recurrent state) | works, output equivalent but not token-identical |

**This helps most on hybrid and recurrent models.** There is no KV cache you can simply seek back
into: without a checkpoint, a change at the tail of the prompt costs the *whole* prompt (measured
31 869 instead of 2 055 tokens to recompute, 97.0 s instead of 8.0 s). On attention-only models the
gain is smaller but real — checkpoints and snapshots still save the re-prefill — and the scheduler,
slot eviction and prefix sharing are independent of the architecture.

On iSWA (Gemma 4) turn detection, checkpoints, all eviction points with rollback, restore, GC,
classes and sharing all work unchanged; `PARTIAL_ONLY` isolates the SWA cache there instead of the
recurrent state. The one caveat is token-level reproducibility, see below.

---

## ⚠️ Before you turn any of this on

**Snapshot files contain the raw KV state of user prompts.** That is conversation content in
reconstructible form on disk, unencrypted. `--telemetry-prompt-dir` writes prompts as plain text.
Both are off by default.

**On SWA models everything works, but not bit-identically.** All of it runs on iSWA models such as
Gemma 4 — turn detection, checkpoints, eviction with rollback, restore, GC, classes, sharing.
Restored output is equivalent in substance but not token-identical: 0.55 ΔLogprob for a snapshot
restore and 0.32 for a prefix merge, against a 0.0000 noise floor, because the SWA cache rotates
during prefill and is restored linearly. In practice that is fine for chat and agent work. If you
need byte-identical replay of a session, either keep snapshots off for that model or use a
non-SWA one — on `qwen4exp` it is exact.

**All numbers come from one machine.** An APU with unified memory, measured with the two models
above. The mechanisms are general, the magnitudes are not transferable.

**One open question I cannot answer.** The decode rate next to a prefill is 1.49 t/s where the
model "one token per iteration" predicts 0.08 t/s. Something already limits the prompt tokens per
iteration to roughly 200. What, I do not know. The measurement series stands regardless.

**A known risk in 03.** A long prefill yields to *every* decode. Under continuous traffic it can
wait a long time; a safety valve (priority after X seconds of waiting) is not implemented. The
telemetry shows it in `ms_prepare`.

---

## 📄 License and thanks

MIT, like llama.cpp. These patches are derived work on MIT-licensed code.

Thanks to everyone who built `llama-server` and the KV cache machinery underneath it. The
interesting parts of this repository are decisions about code somebody else wrote well.
