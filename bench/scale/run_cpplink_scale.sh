#!/bin/bash
# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
#
# Time one cpplink pipeline end to end: blocking price -> estimate -> predict ->
# cluster. Usage: run_cpplink_scale.sh <workdir> <tag> [probability] [threads] [schema]
# With no thread count the tool picks one per core; the scaling sweep passes 1.
set -euo pipefail
W="$1"; TAG="$2"; PROB="${3:-0.99}"; THREADS="${4:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CP="$ROOT/build/cpplink"
S="${5:-$W/parity_schema.json}"; P="$W/$TAG.parquet"
T=()
[ -n "$THREADS" ] && T=(--threads "$THREADS")

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
run blocking "$CP" explain-blocking --schema "$S" "$P"
run estimate "$CP" estimate --schema "$S" "${T[@]}" --out "$W/$TAG.model.json" "$P"
run predict "$CP" predict --schema "$S" --model "$W/$TAG.model.json" "${T[@]}" \
    --out "$W/$TAG.edges" --probability "$PROB" "$P"
run cluster "$CP" cluster --schema "$S" --edges "$W/$TAG.edges" \
    --out "$W/$TAG.clusters.csv" --truth "$W/$TAG.truth.csv" \
    --probability "$PROB" "$P"
