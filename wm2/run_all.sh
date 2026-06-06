#!/usr/bin/env bash
# Reproduce the 2026-frontier world model (wm2) from scratch, end to end.
# Usage:  bash wm2/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (so `python -m wm2.*` resolves)
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
PY=${PYTHON:-python3}

echo "[1/7] collecting multi-body rollouts ..."
$PY -u -m wm2.train collect

echo "[2/7] training the discrete-latent RSSM world model (V+M) ..."
$PY -u -m wm2.train wm

echo "[3/7] training the actor-critic in imagination (C, no probe) ..."
$PY -u -m wm2.train ac

echo "[4/7] training the latent action model (Genie, label-free) ..."
$PY -u -m wm2.train lam

echo "[5/7] training the decoder-free JEPA track (V-JEPA-style) ..."
$PY -u -m wm2.train jepa

echo "[6/7] head-to-head evaluation vs the wm/ baseline ..."
$PY -u -m wm2.train eval

echo "[7/7] rendering dreams / drift / latent maps / code-action map ..."
$PY -u -m wm2.train viz

echo "done -- see wm2_artifacts/ for GIFs, montages, plots and eval_results.json."
