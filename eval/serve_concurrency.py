#!/usr/bin/env python3
"""Server-level concurrency check: does a sampled request still batch?

Every other concurrency number the bots take comes from qwen3_gguf_cb_bench, which drives the
engine directly with greedy requests. A server regression can hide from all of them. On
2026-09-15 fbce2e4 (#1088) made a request that sets no sampler take the checkpoint's
generation_config (temperature 1.0 on Qwen3.8). Packed decode then accepted greedy rows only, so
from that commit every default request decoded on its own: about 90 tok/s aggregate at any
concurrency, for two weeks, while every bot reported healthy cb-decode (fixed in #1201).

This starts sparkinfer_server, sends the same chat request alone and then N at a time, and
reports aggregate output tok/s for both and their ratio. Two sampling modes are measured:

  default    no sampler fields at all, which is what most clients send;
  sampled    an explicit temperature, top_p and a seed per request, so a checkpoint whose
             defaults are greedy (a GGUF) still exercises the sampled path.

The check FAILS when a mode's ratio is below --min-scaling. Measured on an RTX 5090 at c16 with
Qwen3.8-27B NVFP4: about 6-10x after #1201, 1.0x before it, so the default of 3.0 separates the
two with room for noise. It is a floor, not a benchmark: it says nothing about a 5% change.

Output lines (parsed by the bots):
  SERVECONC <mode> c<N> <agg_tok_s> <completed>/<sent>
  SERVECONC_SCALING <mode> <ratio>
  SERVECONC_FAILED <reason>        (once per failure; exit status 1 if any)
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

PROMPT = ("Write a detailed, step-by-step explanation of how a hash table handles collisions, "
          "with an example in Python.")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _chat(port, body, out, idx):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out[idx] = int(json.load(r).get("usage", {}).get("completion_tokens") or 0)
    except Exception as e:                                        # noqa: BLE001 - counted, not raised
        out[idx] = None
        print(f"  request {idx} failed: {e}", file=sys.stderr)


def _wave(port, n, conc, body_for):
    """n requests, at most `conc` in flight. Returns (output tokens, seconds, completed)."""
    out = [None] * n
    sem = threading.Semaphore(conc)
    threads = []

    def run(i):
        with sem:
            _chat(port, body_for(i), out, i)

    t0 = time.monotonic()
    for i in range(n):
        th = threading.Thread(target=run, args=(i,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    dt = time.monotonic() - t0
    done = [x for x in out if x]
    return sum(done), dt, len(done)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="sparkinfer_server binary")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="", help="tokenizer.json (default: <model>/tokenizer.json)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--min-scaling", type=float, default=3.0)
    ap.add_argument("--modes", default="default,sampled")
    ap.add_argument("--startup-timeout", type=int, default=900)
    a = ap.parse_args()
    tok = a.tokenizer or os.path.join(a.model, "tokenizer.json")

    port = _free_port()
    srv = subprocess.Popen([a.server, "-m", a.model, "--tokenizer", tok, "--ctx", str(a.ctx),
                            "--host", "127.0.0.1", "--port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    failures = []

    def stop():
        if srv.poll() is None:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
                srv.wait(timeout=60)
            except Exception:                                     # noqa: BLE001
                os.killpg(srv.pid, signal.SIGKILL)

    signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(1)))
    try:
        deadline = time.monotonic() + a.startup_timeout
        up = False
        while time.monotonic() < deadline and srv.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                up = True
                break
            except Exception:                                     # noqa: BLE001 - still loading
                time.sleep(3)
        if not up:
            print("SERVECONC_FAILED server never became healthy")
            return 1

        for mode in [m for m in a.modes.split(",") if m]:
            def body_for(i, mode=mode):
                b = {"model": "m", "max_tokens": a.max_tokens,
                     "messages": [{"role": "user", "content": PROMPT}]}
                if mode == "sampled":
                    b.update(temperature=0.8, top_p=0.95, seed=1000 + i)
                return b

            _wave(port, 2, 2, body_for)                           # warm-up: graphs, arenas
            single_tok, single_s, single_done = _wave(port, 3, 1, body_for)
            wide_tok, wide_s, wide_done = _wave(port, 2 * a.conc, a.conc, body_for)
            single = single_tok / single_s if single_s > 0 else 0.0
            wide = wide_tok / wide_s if wide_s > 0 else 0.0
            print(f"SERVECONC {mode} c1 {single:.1f} {single_done}/3")
            print(f"SERVECONC {mode} c{a.conc} {wide:.1f} {wide_done}/{2 * a.conc}")
            if single_done < 3 or wide_done < 2 * a.conc:
                failures.append(f"{mode}: {3 - single_done + 2 * a.conc - wide_done} request(s) failed")
                continue
            ratio = wide / single if single > 0 else 0.0
            print(f"SERVECONC_SCALING {mode} {ratio:.2f}")
            if ratio < a.min_scaling:
                failures.append(f"{mode}: c{a.conc} aggregate is {ratio:.2f}x a single stream "
                                f"(floor {a.min_scaling:.1f}x) -- requests are not batching")
    finally:
        stop()
    for f in failures:
        print(f"SERVECONC_FAILED {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
