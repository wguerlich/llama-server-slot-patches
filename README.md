# llama-server slot patches

**Automatic persistent prefix caching for `llama-server`, plus a scheduler that stays responsive
under load.** ⚡

Four patches. Everything is off by default and switchable at runtime — a server built with all
four and started without the new flags behaves exactly like upstream.

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

**⚡ Decode priority under load.** A long prefill no longer starves everything else. Requests that
only need to generate keep running at full speed while a 130k-token prompt is being processed, and
a small request that arrives gets its first token in seconds instead of minutes. **Aggregate
decode 1.49 → 42.78 t/s, time to first token 25.3 s → 1.4 s.**

**🎯 Turn-boundary checkpoints — and less RAM than before.** Checkpoints land where the
conversation actually branches instead of at arbitrary batch boundaries, and the boundary is
*learned* from the prompt itself, so it works on any chat template without parsing it. Because they
are placed rather than sprinkled, **two per slot replace upstream's default of 32** — which at full
context would be 80 GB of host RAM across four slots. **Measured identical in effect to three, and
0.8–2.5 GB lighter.**

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
learned rather than guessed, **two per slot are enough** — measured identical in effect to three,
and 0.8–2.5 GB lighter. Fewer, better-placed checkpoints beat many arbitrary ones.

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

**📏 Everything measured.** Two of the numbers in this README exist because the telemetry
contradicted a hypothesis we were confident about. `idx_lcp2` and `idx_lcp3` in each telemetry line
tell you whether the divergence rule fires on your prompts, before you turn snapshots on.

Where this does *not* compete: SGLang and vLLM are built for many concurrent requests and scale
accordingly. This is for a server with a handful of slots, where a single long prefill can block
everything and one persisted prefix can be worth minutes.

## 📊 What you get

| | Before | After |
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
| `04-slot-management` | 2496 | 6 | needs 03 |

**01 — coalesce runs on restore.** `state_read_data` issued one read per cell when the destination
cells were not contiguous. Now it is one `read_tensor` per *run*, so the cost scales with the
number of gaps rather than the number of cells (1 run 245 ms, 24576 runs 10.6 s). **Upstream has
this since September 2026** — the patch is only for older trees.

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
| Gemma 4 12B | iSWA (sliding-window attention, no recurrent state) | functionally complete, **not bit-exact** |

**This helps most on hybrid and recurrent models.** There is no KV cache you can simply seek back
into: without a checkpoint, a change at the tail of the prompt costs the *whole* prompt (measured
31 869 instead of 2 055 tokens to recompute, 97.0 s instead of 8.0 s). On attention-only models the
gain is smaller but real — checkpoints and snapshots still save the re-prefill — and the scheduler,
slot eviction and prefix sharing are independent of the architecture.

On iSWA (Gemma 4) turn detection, checkpoints, all eviction points with rollback, restore, GC,
classes and sharing all work unchanged; `PARTIAL_ONLY` isolates the SWA cache there instead of the
recurrent state. But see bit-exactness below.

---

## ⚠️ Before you turn any of this on

**Snapshot files contain the raw KV state of user prompts.** That is conversation content in
reconstructible form on disk, unencrypted. `--telemetry-prompt-dir` writes prompts as plain text.
Both are off by default.

**SWA models are not bit-exact.** Noise floor 0.0000, snapshot restore 0.55 ΔLogprob, prefix merge
0.32. No gibberish and equivalent in substance, but the full text diverges after ~70 characters,
because the SWA cache rotates during prefill and is restored linearly. Harmless for chat,
disqualifying for reproducibility. On `qwen4exp` it is exact.

**All numbers come from one machine.** An APU with unified memory, measured with the two models
above. The mechanisms are general, the magnitudes are not transferable.

**One open question I cannot answer.** The decode rate next to a prefill is 1.49 t/s where the
model "one token per iteration" predicts 0.08 t/s. Something already limits the prompt tokens per
iteration to roughly 200. What, I do not know. The measurement series stands regardless.

**A known risk in 03.** A long prefill yields to *every* decode. Under continuous traffic it can
wait a long time; a safety valve (priority after X seconds of waiting) is not implemented. The
telemetry shows it in `ms_prepare`.

---

## 📄 License

MIT, like llama.cpp. These patches are derived work on MIT-licensed code.
