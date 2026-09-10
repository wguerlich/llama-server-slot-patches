#!/usr/bin/env bash
# Apply a patch series to a llama.cpp tree.
#
#   apply.sh <variant> <tree> [up-to-number]
#
#   variant       : directory name under patches/, e.g. upstream-c32d1dabe
#   tree          : path to a llama.cpp source tree
#   up-to-number  : optional, e.g. 03 -> apply only 01..03
#
# The series is first replayed against a copy of the tree. Only if ALL requested patches apply
# there does the real tree get touched - a half-patched tree can never happen.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $# -lt 2 ]; then
    echo "Variants:"
    for d in "$HERE"/patches/*/; do [ -d "$d" ] && echo "  $(basename "$d")"; done
    echo; echo "Usage: apply.sh <variant> <tree> [up-to-number]"
    exit 1
fi

VAR=$1; TREE=$(cd "$2" && pwd); UPTO=${3:-99}
SRC="$HERE/patches/$VAR"
[ -d "$SRC" ] || { echo "unknown variant: $VAR"; exit 1; }
[ -f "$TREE/tools/server/server-context.cpp" ] || { echo "not a llama.cpp tree: $TREE"; exit 1; }

FILES="common/common.h common/arg.cpp src/llama-kv-cache.cpp tools/server/server-common.h
       tools/server/server-context.cpp tools/server/server-task.h tools/server/server-queue.cpp"

wanted() { local n; n=$(basename "$1" | cut -d- -f1); [ "$n" -le "$UPTO" ] 2>/dev/null; }

echo "variant $VAR -> $TREE"
echo

# --- dry run against a copy, cascading ---
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
for f in $FILES; do
    [ -f "$TREE/$f" ] || { echo "  missing file in tree: $f"; exit 1; }
    mkdir -p "$TMP/$(dirname "$f")"; cp "$TREE/$f" "$TMP/$f"
done
for p in "$SRC"/*.patch; do
    wanted "$p" || continue
    if (cd "$TMP" && patch -p1 --forward < "$p" >/dev/null 2>&1); then
        printf "  check  %-42s ok\n" "$(basename "$p")"
    else
        printf "  check  %-42s FAILED\n" "$(basename "$p")"
        (cd "$TMP" && patch -p1 --dry-run --forward < "$p" 2>&1 | grep -E "^(Hunk .* FAILED|patching)" | sed 's/^/      /')
        echo; echo "Aborted: the tree was NOT touched."
        echo "Your llama.cpp version is probably too far from the patch base - see docs/DEVELOPMENT.md."
        exit 2
    fi
done

# --- real application ---
echo
for p in "$SRC"/*.patch; do
    wanted "$p" || continue
    (cd "$TREE" && patch -p1 --forward < "$p" >/dev/null)
    printf "  apply  %-42s ok\n" "$(basename "$p")"
done
echo
echo "Done. Check with: $HERE/scripts/verify.sh $TREE"
