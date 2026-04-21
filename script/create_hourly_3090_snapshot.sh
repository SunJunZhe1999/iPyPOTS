#!/bin/bash
set -u
SOURCE="/Users/sun/Library/UniFormTSV/backups/energy_benchmark_3090_publication"
SNAP_ROOT="/Users/sun/Library/UniFormTSV/hourly_snapshots"
LOG_FILE="/Users/sun/Library/UniFormTSV/hourly_snapshots.log"
mkdir -p "$SNAP_ROOT"
stamp=$(date '+%Y%m%dT%H%M%S')
snapshot="$SNAP_ROOT/energy_benchmark_3090_publication_$stamp"
latest="$SNAP_ROOT/latest"
{
  printf '%s creating hourly snapshot %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$snapshot"
  if [ ! -d "$SOURCE" ]; then
    printf '%s source missing: %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$SOURCE"
    exit 1
  fi
  mkdir -p "$snapshot"
  rsync -az --partial "$SOURCE/" "$snapshot/"
  status=$?
  if [ "$status" -eq 0 ]; then
    rm -f "$latest"
    ln -s "$snapshot" "$latest"
    count=$(find "$SNAP_ROOT" -maxdepth 1 -type d -name 'energy_benchmark_3090_publication_*' | wc -l | tr -d ' ')
    remove_count=$(( count - 24 ))
    if [ "$remove_count" -gt 0 ]; then
      find "$SNAP_ROOT" -maxdepth 1 -type d -name 'energy_benchmark_3090_publication_*' | sort | sed -n "1,${remove_count}p" | while IFS= read -r old_snapshot; do
        rm -rf "$old_snapshot"
      done
    fi
  fi
  printf '%s snapshot exit=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$status"
  exit "$status"
} >> "$LOG_FILE" 2>&1
