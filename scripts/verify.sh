#!/usr/bin/env bash
# Check which patches are present in a tree - and whether a built binary knows the flags.
#   verify.sh <tree> [path/to/llama-server]
set -uo pipefail
TREE=${1:-.}
BIN=${2:-}
KV="$TREE/src/llama-kv-cache.cpp"
SC="$TREE/tools/server/server-context.cpp"

count() { local n; n=$(grep -c "$1" "$2" 2>/dev/null | head -1); echo "${n:-0}"; }
row()   { local n; n=$(count "$2" "$3"); printf "  %-34s %s\n" "$1" "$([ "$n" -gt 0 ] && echo "present ($n)" || echo "missing")"; }

echo "tree: $TREE"
row "01 coalesced restore"      'coalesce\|const auto & r : runs'   "$KV"
row "02 pool filled from bottom" 'head_prev'                        "$KV"
row "03 decode priority"        'prefill_defer_above'               "$SC"
row "04 slot eviction"          'SLOT_EVICT'                        "$SC"
row "04 turn checkpoints"       'ctx_tail_seen_in'                  "$SC"
row "04 prefix sharing"         'kv_share_merge'                    "$SC"
row "04 disk snapshots"         'snapshot_dump_slot'                "$SC"
row "04 request telemetry"      'telemetry_write'                   "$SC"

if [ -n "$BIN" ] && [ -x "$BIN" ]; then
    echo; echo "binary: $BIN"
    H=$("$BIN" --help 2>&1)
    for f in --prefill-defer-above --slot-evict-policy --slot-evict-ttl --ctx-checkpoint-tail \
             --kv-share-min --snapshot-path --telemetry-file; do
        printf "  %-26s %s\n" "$f" "$(printf '%s' "$H" | grep -qF -- "$f" && echo "known" || echo "missing")"
    done
fi
