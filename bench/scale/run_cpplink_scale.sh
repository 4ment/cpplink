#!/bin/bash
# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
#
# Time one cpplink pipeline end to end: estimate -> predict -> cluster.
# Usage: run_cpplink_scale.sh <workdir> <tag> [probability]
set -euo pipefail
W="$1"; TAG="$2"; PROB="${3:-0.99}"
CP="$(cd "$(dirname "$0")/../.." && pwd)/build/cpplink"
S="$W/parity_schema.json"; P="$W/$TAG.parquet"

run() {
    local stage="$1"; shift
    /usr/bin/time -l "$@" > "$W/$TAG.$stage.log" 2> "$W/$TAG.$stage.time"
    local real user sys rss
    real=$(awk '/real/{print $1}' "$W/$TAG.$stage.time")
    user=$(awk '/real/{print $3}' "$W/$TAG.$stage.time")
    sys=$(awk '/real/{print $5}' "$W/$TAG.$stage.time")
    rss=$(awk '/maximum resident/{print $1}' "$W/$TAG.$stage.time")
    echo "$TAG $stage real=$real user=$user sys=$sys rss_mb=$((rss / 1048576))"
}

rm -rf "$W/$TAG.edges"; mkdir -p "$W/$TAG.edges"
run estimate "$CP" estimate --schema "$S" --out "$W/$TAG.model.json" "$P"
run predict "$CP" predict --schema "$S" --model "$W/$TAG.model.json" \
    --out "$W/$TAG.edges" --probability "$PROB" "$P"
run cluster "$CP" cluster --schema "$S" --edges "$W/$TAG.edges" \
    --out "$W/$TAG.clusters.csv" --truth "$W/$TAG.truth.csv" \
    --probability "$PROB" "$P"
