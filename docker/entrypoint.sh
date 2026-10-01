#!/usr/bin/env bash
# Default: serve (image + video) over an OpenAI-compatible API, speculating with z-lab's DFlash2
# drafter (SPEC_DRAFT=dflash2, the default; SPEC_DRAFT=none serves without a drafter).
# `serve-dspark`: download the DSpark drafter and speculate with it instead.
# `bench [workload]`: run the DSpark speculative benchmark instead.
set -euo pipefail

fetch() {  # repo dir label
  if [ ! -f "$2/config.json" ]; then
    # SPARKINFER_NO_DOWNLOAD=1: the weights are pre-staged and the container has no route out
    # (#1090). Say what is missing instead of failing inside a download that cannot work.
    if [ "${SPARKINFER_NO_DOWNLOAD:-0}" = "1" ]; then
      echo "[sparkinfer] SPARKINFER_NO_DOWNLOAD=1 and $2 holds no config.json: mount the $3 weights there, or point the matching *_DIR at them." >&2
      exit 1
    fi
    echo "[sparkinfer] downloading $3 ($1) — first run only, cached in /models"
    hf download "$1" --local-dir "$2"
  fi
}

if [ "${1:-}" = "bench" ]; then
  shift
  WORKLOAD="${1:-code}"
  fetch "$MODEL_REPO" "$MODEL_DIR" "target"
  fetch "$DRAFT_REPO" "$DRAFT_DIR" "DSpark drafter"
  echo "[sparkinfer] DSpark bench · workload=$WORKLOAD · ${BENCH_TOKENS:-256} tokens"
  MODEL_DIR="$MODEL_DIR" python3 /opt/sparkinfer/mkids.py "$WORKLOAD" > /tmp/ids.txt
  exec /opt/sparkinfer/bin/qwen38_hf_dflash_bench \
       "$MODEL_DIR" "$DRAFT_DIR" "${BENCH_TOKENS:-256}" $(cat /tmp/ids.txt)
fi

if [ "${1:-}" = "serve-dspark" ]; then
  shift
  fetch "$DRAFT_REPO" "$DRAFT_DIR" "DSpark drafter"
  export SPARKINFER_DRAFT_MODEL="$DRAFT_DIR"
  # The drafter needs device memory the full 262,144-token KV pool leaves no room for on a 32 GB
  # card (#1086), so DSpark defaults to half the context. -e CTX=... (or --ctx) still overrides.
  CTX="${CTX:-131072}"
fi
# Autoregressive serving defaults to half the model's context too. The full 262,144-token pool
# does load on a 32 GB card, but leaves ~3 GB for everything else, and concurrent serving needs
# more: the packed decode graphs and batched-prefill scratch then fail to allocate, every request
# decodes on its own, and 16 concurrent 8K-token prompts stalled the server outright (RTX 5090,
# 2026-09-28). At 131,072 the same load leaves ~7 GB and none of that happens. A single long
# conversation can still ask for the full context with -e CTX=262144 (or --ctx).
CTX="${CTX:-131072}"

# The default drafter. With it loaded, concurrent requests speculate together (up to eight):
# 1.2-2.2x the throughput at 1-4 concurrent requests and the same at 16-32, where the drafter's
# device memory steps aside while it cannot be used (AIPerf, RTX 5090, v0.6.1). Skipped where it
# does not fit: past 131,072 tokens of context a 32 GB card has no room for it beside the KV pool.
# An explicit SPARKINFER_DRAFT_MODEL (or `serve-dspark`, or --draft-model) is always kept.
if [ -z "${SPARKINFER_DRAFT_MODEL:-}" ] && [ "${SPEC_DRAFT:-dflash2}" = "dflash2" ]; then
  ARG_CTX=""
  prev=""
  for a in "$@"; do
    [ "$prev" = "--ctx" ] && ARG_CTX="$a"
    case "$a" in --ctx=*) ARG_CTX="${a#--ctx=}" ;; --draft-model|--draft-model=*) ARG_CTX="skip" ;; esac
    prev="$a"
  done
  EFF_CTX="${ARG_CTX:-$CTX}"
  if [ "$EFF_CTX" = "skip" ]; then
    :   # a --draft-model argument chooses the drafter
  elif [ "$EFF_CTX" -gt 131072 ] 2>/dev/null; then
    echo "[sparkinfer] ctx $EFF_CTX: serving without the DFlash2 drafter (no room for it beside a pool this size on a 32 GB card)"
  elif [ "${SPARKINFER_NO_DOWNLOAD:-0}" = "1" ] && [ ! -f "$DFLASH2_DIR/config.json" ]; then
    echo "[sparkinfer] SPARKINFER_NO_DOWNLOAD=1 and $DFLASH2_DIR holds no drafter: serving without speculation"
  else
    fetch "$DFLASH2_REPO" "$DFLASH2_DIR" "DFlash2 drafter"
    export SPARKINFER_DRAFT_MODEL="$DFLASH2_DIR"
  fi
fi

fetch "$MODEL_REPO" "$MODEL_DIR" "target"
MODE="autoregressive"
[ -n "${SPARKINFER_DRAFT_MODEL:-}" ] && MODE="speculative, drafter $SPARKINFER_DRAFT_MODEL"
echo "[sparkinfer] serving $MODEL_DIR as '$MODEL_NAME' on $HOST:$PORT (ctx $CTX, max output $SPARKINFER_MAX_OUTPUT_TOKENS, $MODE)"
exec /opt/sparkinfer/bin/sparkinfer_server \
  -m "$MODEL_DIR" --tokenizer "$MODEL_DIR/tokenizer.json" \
  --model-name "$MODEL_NAME" --ctx "$CTX" --host "$HOST" --port "$PORT" "$@"
