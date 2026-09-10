# Measurements

All numbers from one machine: AMD Strix Halo APU (gfx1151, RDNA 3.5), 128 GB unified memory,
~215 GB/s in practice, ROCm 7.14, HIP build.

Primary model: Qwen3.8-Flash-Next (`qwen4exp`), 125B total / 6B active (MoE), 51B n-gram table
mmap'd from NVMe, 4B MTP head, 36 of 48 layers Gated DeltaNet (recurrent), 12 QSA layers with
`compress_ratio 4` and `head_dim 256`. UD-Q4_K_XL, `ubatch 2048`, `batch 4096`, unified KV,
6 slots.

Cross-checked with **Gemma 4 12B (iSWA)** — sliding-window attention, no recurrent state. Turn
detection, checkpoints, all three eviction points with rollback, restore, GC, classes and prefix
sharing all work unchanged there; `PARTIAL_ONLY` isolates the SWA cache instead of the recurrent
state. **But not bit-exact:**

| Comparison | ΔLogprob |
|---|---|
| noise floor (two identical runs) | 0.0000 |
| snapshot restore vs full prefill | 0.55 |
| prefix merge vs no merge | 0.32 |

No gibberish and equivalent in substance, but the full text diverges after ~70 characters, because
the SWA cache rotates during prefill and is restored linearly. Harmless for chat, disqualifying for
reproducibility. On `qwen4exp` it is exact.

One note for all comparison measurements, learned the hard way: **a batch boundary shifted by a
single token changes the logprobs 10× more than GPU noise.** Only compare snapshot and full runs
with identical boundaries.

## How reliable each axis is

Before any number: **prefill is reliably measurable, decode is not without repetition.** From four
runs of the same 48k prompt:

| Axis | Values | Spread |
|---|---|---|
| prefill (48k, solo) | 309.1 / 308.6 / 307.7 / 316.2 t/s | **±0.2 %** (excluding the outlier) |
| decode (short prompt) | 35.51 / 37.96 / 35.07 / 34.81 t/s | **±6.9 %** |
| decode, 6 repetitions | 33.57 31.13 35.08 31.94 31.91 35.53 | median 32.76, range 31.1–35.5 |

A single run cannot resolve a 5 % effect in decode. Where decode numbers below come from single
runs, it is stated.

---

## Scheduler (patch 03)

Setup: one 48 084-token prompt (unique prefix, so no cache hit) plus one to three small requests
(92 tokens, 150 generated tokens) alongside. Solo references in the same setup: **pp 309.1 t/s**,
**tg 35.51 t/s**. Weighted as `U = pp/pp_solo + tg_aggregate/tg_solo`.

| Configuration | pp | tg aggregate | U | TTFT small | wall small |
|---|---|---|---|---|---|
| upstream (max-active 2, 1 decoder) | 306.8 | 1.49 | 1.035 | 25.3 s | 129.4 s |
| batch cap 256 | 249.6 | 1.64 | 0.854 | 25.5 s | 120.8 s |
| batch cap 64 | 272.1 | 4.64 | 1.011 | 25.5 s | 61.9 s |
| batch cap 32 | 271.2 | 4.71 | 1.010 | 25.5 s | — |
| batch cap 16 | 271.8 | 4.79 | 1.014 | 25.5 s | — |
| defer 2048, 3 decoders | 276.9 | 38.4 | 1.977 | 25.4 s | 41.4 s |
| **defer 2048 + small pending** | **277.6** | **42.78** | **2.103** | **1.4 s** | **16.2 s** |

Three readings from this:

**The batch cap plateaus.** 64, 32 and 16 all land at tg 4.7 t/s, because ~0.21 s is the fixed cost
of an iteration at this depth. 256 is worse than 64 on *both* axes, because the small request lives
longer and the large prefill therefore has company for longer.

**The second half of the condition is the lever for TTFT.** Without "a smaller prefill is waiting",
nobody is generating at the moment the newcomer arrives, so it waits out the running iteration:
25.4 s, reproduced three times, for 96 tokens. The sort puts it first, but its logits only arrive
at the end of an iteration that also carries 4 000 foreign prefill tokens.

**Starved prefills, measured directly.** `ms_prepare` — the time between slot assignment and the
start of prompt processing — was **669 647 ms** and **766 021 ms** for requests of 312 and 300
tokens next to a 130k prefill, with an empty queue. Afterwards they needed 4.6 s.

---

## Scan depth: `n_kv` is global (patch 02)

`get_n_kv()` returns `min(cells.size(), GGML_PAD(cells.used_max_p1(), n_pad))` — the highest
occupied cell across **all** sequences. Every attention kernel walks 0..n_kv.

| Situation | Decode |
|---|---|
| 500-token request, 3k pool | 21.95 t/s |
| the same, next to an **idle** 36k session | **16.61 t/s** |
| that 36k session itself | 16.63 t/s |

The small request gets exactly the rate of the foreign session: **−24 %**, paid on every token.
Hence filling from the bottom: appending on top raises `used_max_p1()` for everyone, and it only
drops when the topmost cells are freed.

From production telemetry over 176 requests, by occupied slots:

| slots_used | n | draft_acc median | tg median |
|---|---|---|---|
| 0 | 20 | 0.669 | 27.66 |
| 1 | 81 | 0.730 | 17.07 |
| 2 | 45 | 0.620 | 15.82 |
| 3+ | 30 | 0.620 | 16.05 |

Side finding: MTP draft acceptance drops from 0.73 to 0.62 under concurrency — a moderate decline,
not a collapse. The common claim that the speculative-decoding advantage disappears at 2–4 streams
was not confirmed here.

---

## Restore path (patch 01)

Cost scales with the number of *runs*, not cells:

| Runs | Restore time |
|---|---|
| 1 (contiguous) | 245 ms |
| 24576 (every cell separate) | 10.6 s |

The worst case can be forced with `LLAMA_SNAPSHOT_FORCE_SCATTER`. **Upstream has the coalescing
since September 2026** — this patch is only for older trees.

---

## Slot eviction (patch 04)

| Situation | upstream LRU | class ordering |
|---|---|---|
| follow-up turn after eviction | 47.5 s | **0.3 s** |

And the case that shows why capping concurrent slots is expensive:

| Configuration | tg per slot | aggregate |
|---|---|---|
| `slot-max-active 2`, 1 decoder next to a large prefill | 1.49 | 1.49 |
| no cap, 3 decoders | 14.2 / 14.3 / 14.3 | **42.78** |

42.78 t/s aggregate is **above** the solo rate of a single decoder (35.51) — exactly the continuous
batching effect.

---

## Turn-boundary checkpoints (patch 04)

Checkpoint size, linear over 7 points from 42 to 66 629 tokens:
**112.6 MiB + 2.023 KiB per token**. So 174 MiB at 30k, **630 MiB** at a full 262k context — and
that is **per slot**. Upstream's default of 32 checkpoints would be 80 GB of host RAM across 4
slots.

With turn boundaries two checkpoints per slot are enough; measured identical to three, saving
0.8–2.5 GB.

Why it is indispensable on recurrent models: a change at the tail of the prompt costs 2 055 tokens
to recompute with a checkpoint instead of 31 869 without — **8.0 s instead of 97.0 s**.

---

## Prefix snapshots (patch 04)

| Measurement | Value |
|---|---|
| loading 12.5k tokens | **1.7 s instead of 32 s** of prefill |
| pure load time | 139 ms |
| bit-identity with the producing run (`qwen4exp`) | exact |
| size | ~84 MB fixed + ~41 KB per token |
| writing (6 075 tokens, 336 MB) | 97 ms |

Creation observed in production: three chats with different opening tokens produced a node with
three distinct children, the snapshot at 6 075 was written and used by the next chat immediately.

**Limit of the divergence rule**, measured: at one node **16 prompts** shared the same prefix but
there were only **2 distinct children** — at that position the template can only continue two ways.
`min-hits = 3` is structurally unreachable there.

---

## Prefix sharing (patch 04)

Three concurrent sessions hold **9624 logical tokens in a 12288-cell pool**, merge duration
23–53 ms, no defer events.

---

## The seam (the reason for all of this)

A session whose prompt grew monotonically from 87 159 to 115 276 tokens re-prefilled **15 turns in
a row** from scratch:

| | Turns | fresh tokens (median) | prefill (median) |
|---|---|---|---|
| before | 8 (with phase data) | 81 142 | **488.5 s** (416–518) |
| after | 19 | 1 986 | **18.0 s** (3.6–37.8) |

64 minutes against 6. `cached` was **exactly 32 768 every time** before — the same static snapshot,
loaded 20 times, because the continuation was never recognised.

Cause: one token. The prompt ended on a newline, the follow-up continued with another, and the
tokenizer merges two adjacent newlines into one. At the **character** level the follow-up was an
exact append (verified with `cmp`: the first 380 936 bytes identical), at the **token** level
`lcp = prompt length − 1`. In **19 of 19** continuations after the fix the same seam appeared again
— it forms at the same template boundary every time and is therefore not a special case.

Only the token ids make this visible. Anyone checking the same suspicion needs both sides: the
plain text (which is identical) and the token sequence (which is not).

---

## NVMe / n-gram table

Because it gets asked: the 28.8 GB table mmap'd from NVMe is **not** a bottleneck.

| Phase | Tokens | read | per token | rate |
|---|---|---|---|---|
| decode | 400 | 9.8 MB | 24 KB | 0.9 MB/s |
| prefill | 30 000 | 432.7 MB | 14 KB | 0.8 MB/s |

At 24 KB per token that is ~6 pages, so ~0.5 ms against 28 ms of token time — eliminating it
entirely would gain ~2 %. More page cache does not help; a counter-test with 3.4 GB more free
memory showed no significant decode improvement.

---

## Things that did not help

So nobody repeats them; each measured in this setup:

| Change | Result |
|---|---|
| `ROCBLAS_USE_HIPBLASLT=1` | pp 316.2 vs 309.1 — within noise |
| HIP graphs enabled (dropping `GGML_CUDA_DISABLE_GRAPHS`) | pp 307.7, tg 35.07 — no effect |
| `ubatch 1024` instead of 2048 | pp **−5.4 %** at 48k, **−6.8 %** at 7k; decode +4.7 % not significant (t ≈ 1.2) |
| capping prompt tokens per iteration | plateaus at tg 4.7 t/s; 256 costs 17 % of `U` |
| paged KV cache (upstream discussion) | FCFS without decode priority; worse at low concurrency (TTFT 791 vs 159 ms) |
