#!/usr/bin/env bash
# Report which parts are present in a patched tree.
#   verify.sh <tree>
set -euo pipefail
TREE=$(cd "${1:?usage: verify.sh <llama.cpp tree>}" && pwd)
C="$TREE/tools/server/server-context.cpp"
[ -f "$C" ] || { echo "not a llama.cpp tree: $TREE"; exit 1; }

echo "tree: $TREE"
check() {
    local name=$1 file=$2 pat=$3 min=$4
    local n; n=$(grep -c "$pat" "$file" 2>/dev/null || true)
    if [ "${n:-0}" -ge "$min" ]; then printf "  %-34s present (%s)\n" "$name" "$n"
    else                              printf "  %-34s MISSING\n" "$name"; fi
}
check "pool filled from bottom"  "$TREE/src/llama-kv-cache.cpp" 'head_cur = 0'            2
check "decode priority"          "$C" 'prefill_defer_above'                               5
check "slot eviction and GC"     "$C" 'SLOT_EVICT'                                       20
check "prefix sharing"           "$C" 'kv_share_min'                                      2
check "prefix index"             "$C" 'prefix_index'                                      5
check "disk snapshots"           "$C" 'snapshot_path'                                     6
check "divergence probe"         "$C" 'probe_at\|ckpt_probe'                             5
check "hint channel"             "$TREE/tools/server/server-analyzer.h" 'llama-snap'      3
check "request telemetry"        "$C" 'telemetry_file'                                     4
[ -f "$TREE/tools/server/tests-snap/suite.py" ] \
    && printf "  %-34s present\n" "test bed" \
    || printf "  %-34s MISSING\n" "test bed"
