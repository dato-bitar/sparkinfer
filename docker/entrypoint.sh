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
SPEC_DRAFT="${SPEC_DRAFT:-dflash2}"
case "$SPEC_DRAFT" in
  dflash2|none) ;;
  *) echo "[sparkinfer] SPEC_DRAFT=$SPEC_DRAFT is not dflash2 or none: serving without a drafter"; SPEC_DRAFT=none ;;
esac
if [ -z "${SPARKINFER_DRAFT_MODEL:-}" ] && [ "$SPEC_DRAFT" = "dflash2" ]; then
  # The server reads `--ctx N` and `--draft-model DIR` (space-separated, the last one wins).
  ARG_CTX=""
  DRAFT_ARG=0
  prev=""
  for a in "$@"; do
    [ "$prev" = "--ctx" ] && ARG_CTX="$a"
    [ "$a" = "--draft-model" ] && DRAFT_ARG=1
    prev="$a"
  done
  EFF_CTX="${ARG_CTX:-$CTX}"
  # `hf download` moves each file into place only once it is whole, so the weights and config
  # being there means a finished download, staged, copied or ours.
  HAVE_DF2=0
  [ -f "$DFLASH2_DIR/model.safetensors" ] && [ -f "$DFLASH2_DIR/config.json" ] && HAVE_DF2=1
  if [ "$DRAFT_ARG" = "1" ]; then
    :   # a --draft-model argument chooses the drafter
  elif ! [[ "$EFF_CTX" =~ ^[1-9][0-9]*$ ]]; then
    # The server reads it as 0, the model's full context: no room for a drafter beside that pool.
    echo "[sparkinfer] ctx '$EFF_CTX' is not a token count: serving without the DFlash2 drafter"
  elif [ "$EFF_CTX" -gt 131072 ]; then
    echo "[sparkinfer] ctx $EFF_CTX: serving without the DFlash2 drafter (no room for it beside a pool this size on a 32 GB card)"
  elif [ "$HAVE_DF2" = "1" ]; then
    export SPARKINFER_DRAFT_MODEL="$DFLASH2_DIR"
  elif [ "${SPARKINFER_NO_DOWNLOAD:-0}" = "1" ]; then
    echo "[sparkinfer] SPARKINFER_NO_DOWNLOAD=1 and $DFLASH2_DIR holds no drafter: serving without speculation"
  else
    echo "[sparkinfer] downloading DFlash2 drafter ($DFLASH2_REPO) — first run only, cached in /models"
    if hf download "$DFLASH2_REPO" --local-dir "$DFLASH2_DIR" >/dev/null &&
       [ -f "$DFLASH2_DIR/model.safetensors" ]; then
      export SPARKINFER_DRAFT_MODEL="$DFLASH2_DIR"
    else
      # The default drafter is optional: no route to the Hub, a rate limit or a read-only /models
      # must not stop a server that would otherwise start.
      echo "[sparkinfer] the DFlash2 drafter could not be downloaded: serving without speculation"
    fi
  fi
fi

fetch "$MODEL_REPO" "$MODEL_DIR" "target"
MODE="autoregressive"
[ -n "${SPARKINFER_DRAFT_MODEL:-}" ] && MODE="speculative, drafter $SPARKINFER_DRAFT_MODEL"
echo "[sparkinfer] serving $MODEL_DIR as '$MODEL_NAME' on $HOST:$PORT (ctx $CTX, max output $SPARKINFER_MAX_OUTPUT_TOKENS, $MODE)"
exec /opt/sparkinfer/bin/sparkinfer_server \
  -m "$MODEL_DIR" --tokenizer "$MODEL_DIR/tokenizer.json" \
  --model-name "$MODEL_NAME" --ctx "$CTX" --host "$HOST" --port "$PORT" "$@"
