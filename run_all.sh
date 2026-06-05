#!/usr/bin/env bash
# Reproduce the entire world model from scratch, end to end.
# Usage:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
PY=${PYTHON:-python3}

echo "[1/6] collecting rollouts from the environment ..."
$PY -u -m wm.collect

echo "[2/6] training the VAE (V) ..."
$PY -u -m wm.train_vae --epochs 26

echo "[3/6] training the latent->position probe ..."
$PY -u -m wm.probe

echo "[4/6] training the MDN-RNN dynamics model (M) ..."
$PY -u -m wm.train_mdnrnn --epochs 24

echo "[5/6] evolving the controller (C) inside the dream ..."
$PY -u -m wm.controller

echo "[6/6] deploying the dream-trained controller in the real env ..."
$PY -u -m wm.evaluate

echo "done -- see artifacts/ for GIFs, montages and plots."
