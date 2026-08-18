#!/usr/bin/env bash
#
# download_dataset.sh — fetch and distribute the AIC25 dataset via aria2c.
#
# The link list lives in scripts/download_links.txt — edit THAT file to
# add/remove/change URLs. This script never needs to change for that.
#
# Category and extraction target are inferred from the filename:
#   Keyframes_*.zip                 -> data/raw/keyframes/
#   Videos_*.zip                    -> data/raw/videos/
#   clip-features-*.zip             -> data/features/clip/
#   map-keyframes-*.zip             -> data/metadata/map_keyframes/
#   media-info-*.zip                -> data/metadata/media_info/
#   objects-*.zip                   -> data/metadata/objects/
#   (anything else)                 -> data/_downloads/misc/  (extracted there,
#                                       so new asset types still land somewhere
#                                       sane without editing this script)
#
# Usage:
#   ./scripts/download_dataset.sh                    # download + extract everything
#   ./scripts/download_dataset.sh --only keyframes    # one category only
#   ./scripts/download_dataset.sh --only videos,clip  # multiple categories
#   ./scripts/download_dataset.sh --list-only         # just print the plan, do nothing
#   ./scripts/download_dataset.sh --keep-zips         # don't delete archives after extraction
#   ./scripts/download_dataset.sh --jobs 4 --conns 8  # tune aria2c parallelism
#   ./scripts/download_dataset.sh --links other.txt   # use a different link file
#
# Requires: aria2c, unzip. (sudo apt install aria2 unzip)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINKS_FILE="${ROOT_DIR}/scripts/download_links.txt"
DOWNLOAD_DIR="${ROOT_DIR}/data/_downloads"
RAW_DIR="${ROOT_DIR}/data/raw"
FEAT_DIR="${ROOT_DIR}/data/features"
META_DIR="${ROOT_DIR}/data/metadata"

JOBS=2          # aria2c -j : number of parallel downloads (separate files)
CONNS=8         # aria2c -x/-s : connections per file (splits single file for speed)
KEEP_ZIPS=0
LIST_ONLY=0
ONLY_CATEGORIES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-zips)  KEEP_ZIPS=1; shift ;;
    --list-only)  LIST_ONLY=1; shift ;;
    --jobs)       JOBS="$2"; shift 2 ;;
    --conns)      CONNS="$2"; shift 2 ;;
    --only)       ONLY_CATEGORIES="$2"; shift 2 ;;
    --links)      LINKS_FILE="$2"; shift 2 ;;
    -h|--help)    grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if ! command -v aria2c &>/dev/null; then
  echo "ERROR: aria2c not found. Install it with: sudo apt install aria2" >&2
  exit 1
fi
if ! command -v unzip &>/dev/null; then
  echo "ERROR: unzip not found. Install it with: sudo apt install unzip" >&2
  exit 1
fi
if [[ ! -f "$LINKS_FILE" ]]; then
  echo "ERROR: links file not found: $LINKS_FILE" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR" "$RAW_DIR/videos" "$RAW_DIR/keyframes" \
         "$FEAT_DIR/clip" "$META_DIR/map_keyframes" "$META_DIR/media_info" \
         "$META_DIR/objects" "$DOWNLOAD_DIR/misc"

# ---- classify a filename into (category, extract_target) ------------------
classify() {
  local fname="$1"
  case "$fname" in
    Keyframes_*)          echo "keyframes|${RAW_DIR}/keyframes" ;;
    Videos_*)             echo "videos|${RAW_DIR}/videos" ;;
    clip-features-*)      echo "clip|${FEAT_DIR}/clip" ;;
    map-keyframes-*)      echo "map|${META_DIR}/map_keyframes" ;;
    media-info-*)         echo "media_info|${META_DIR}/media_info" ;;
    objects-*)            echo "objects|${META_DIR}/objects" ;;
    *)                    echo "misc|${DOWNLOAD_DIR}/misc" ;;
  esac
}

# ---- read + filter link list ------------------------------------------
mapfile -t ALL_URLS < <(grep -vE '^\s*(#|$)' "$LINKS_FILE" | sed 's/[[:space:]]*$//')

SELECTED_URLS=()
SELECTED_CATS=()
SELECTED_TARGETS=()
SELECTED_NAMES=()

IFS=',' read -ra ONLY_ARR <<< "$ONLY_CATEGORIES"

for url in "${ALL_URLS[@]}"; do
  fname="$(basename "$url")"
  IFS='|' read -r cat target <<< "$(classify "$fname")"

  if [[ -n "$ONLY_CATEGORIES" ]]; then
    keep=0
    for c in "${ONLY_ARR[@]}"; do
      [[ "$c" == "$cat" ]] && keep=1
    done
    [[ "$keep" -eq 0 ]] && continue
  fi

  SELECTED_URLS+=("$url")
  SELECTED_CATS+=("$cat")
  SELECTED_TARGETS+=("$target")
  SELECTED_NAMES+=("$fname")
done

echo "Plan: ${#SELECTED_URLS[@]} archive(s) selected from $(basename "$LINKS_FILE")"
printf '  %-10s %s\n' "CATEGORY" "FILE"
for i in "${!SELECTED_URLS[@]}"; do
  printf '  %-10s %s\n' "${SELECTED_CATS[$i]}" "${SELECTED_NAMES[$i]}"
done

if [[ "$LIST_ONLY" -eq 1 ]]; then
  exit 0
fi
if [[ "${#SELECTED_URLS[@]}" -eq 0 ]]; then
  echo "Nothing to do (empty selection)."
  exit 0
fi

# ---- write an aria2c input file for a single batched, resumable run -------
# aria2c's -i flag downloads a whole list in one process with shared
# connection pooling, and -c makes every entry resumable individually.
ARIA_INPUT="${DOWNLOAD_DIR}/.aria2_input_$$.txt"
: > "$ARIA_INPUT"
for i in "${!SELECTED_URLS[@]}"; do
  echo "${SELECTED_URLS[$i]}" >> "$ARIA_INPUT"
  echo "  dir=${DOWNLOAD_DIR}" >> "$ARIA_INPUT"
  echo "  out=${SELECTED_NAMES[$i]}" >> "$ARIA_INPUT"
done

echo ""
echo "Downloading with aria2c (jobs=${JOBS} parallel files, conns=${CONNS} per file)..."
aria2c \
  --input-file="$ARIA_INPUT" \
  --continue=true \
  --max-concurrent-downloads="$JOBS" \
  --split="$CONNS" \
  --max-connection-per-server="$CONNS" \
  --min-split-size=1M \
  --retry-wait=5 \
  --max-tries=0 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --summary-interval=15 \
  --console-log-level=warn \
  --show-console-readout=true

rm -f "$ARIA_INPUT"

# ---- extract each archive to its target, then optionally clean up ---------
echo ""
echo "Extracting..."
for i in "${!SELECTED_URLS[@]}"; do
  fname="${SELECTED_NAMES[$i]}"
  target="${SELECTED_TARGETS[$i]}"
  archive="${DOWNLOAD_DIR}/${fname}"

  if [[ ! -f "$archive" ]]; then
    echo "  [WARN] expected archive missing, skipping: $archive"
    continue
  fi

  echo "  [${fname}] -> ${target}"
  mkdir -p "$target"
  unzip -q -o "$archive" -d "$target"

  if [[ "$KEEP_ZIPS" -eq 0 ]]; then
    rm -f "$archive"
  fi
done

echo ""
echo "Done."
echo "  Videos:        $RAW_DIR/videos"
echo "  Keyframes:     $RAW_DIR/keyframes"
echo "  CLIP features: $FEAT_DIR/clip"
echo "  Map-keyframes: $META_DIR/map_keyframes"
echo "  Media info:    $META_DIR/media_info"
echo "  Objects:       $META_DIR/objects"
[[ "$KEEP_ZIPS" -eq 1 ]] && echo "  (zips kept in: $DOWNLOAD_DIR)"
