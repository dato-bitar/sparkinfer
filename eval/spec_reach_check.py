#!/usr/bin/env python3
"""A request speculates up to the end of the draft's context, then decodes on, losslessly.

1. Draft context shrunk to 1,024 positions (SPARKINFER_DSPARK_MAX_CTX), ~200-token prompts asking
   for 1,500 tokens: each request speculates, reaches the draft's context and hands off. Completions
   must equal a SPARKINFER_SPECULATIVE=0 launch, and the trace must show the hand-off.
2. Default draft context, server cap 16,384 (the release container's), a request WITHOUT
   max_tokens: it must speculate (it used to be barred: prompt + 16,384 > the draft's context).

usage: spec_reach_check.py <server-bin> <model-dir> <draft-dir>
"""
import json, os, signal, subprocess, sys, tempfile, time, urllib.request

PORT = 18141
TMP = tempfile.mkdtemp(prefix="spec_reach_")
PROMPTS = [
    "Write a detailed, multi-section essay on the history of the printing press and its effects on science, religion and politics.",
    "Explain, step by step and with examples, how a modern optimizing compiler turns C source into machine code.",
]


def post(body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metric(name):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith(name + " "):
                return float(line.split()[1])
    return 0.0


def launch(binary, model, draft, log, extra):
    env = dict(os.environ, SPARKINFER_DETERMINISTIC="1", SPARKINFER_PREFIX_CACHE="0", **extra)
    p = subprocess.Popen([binary, "-m", model, "--tokenizer", f"{model}/tokenizer.json", "--model-name", "q",
                          "--ctx", "131072", "--draft-model", draft, "--host", "127.0.0.1", "--port", str(PORT)],
                         stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    for _ in range(720):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            return p
        except Exception:
            if p.poll() is not None:
                sys.exit(f"server exited: see {log}")
            time.sleep(5)
    sys.exit("server did not come up")


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    time.sleep(5)


def run_set():
    out = {}
    for t in (0.0, 0.7):
        for i, pr in enumerate(PROMPTS):
            r = post({"model": "q", "messages": [{"role": "user", "content": pr}], "max_tokens": 1500,
                      "temperature": t, "top_p": 0.95, "top_k": 20, "seed": 77})
            out[(i, t)] = (r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"])
    return out


def main():
    binary, model, draft = sys.argv[1:4]
    small = {"SPARKINFER_DSPARK_MAX_CTX": "1024"}
    log = os.path.join(TMP, "spec.log")
    p = launch(binary, model, draft, log, dict(small, SPARKINFER_SPEC_GROUP_TRACE="1"))
    spec = run_set()
    stop(p)
    text = open(log).read()
    handoffs = text.count("reached the draft's context")
    p = launch(binary, model, draft, os.path.join(TMP, "ref.log"), dict(small, SPARKINFER_SPECULATIVE="0"))
    ref = run_set()
    stop(p)
    same = all(spec[k][0] == ref[k][0] for k in ref)
    lens = [spec[k][1] for k in sorted(spec)]
    print(f"part 1: identical={same} handoffs={handoffs} completion_tokens={lens}")
    # Part 2: no max_tokens, the release container's cap, the draft's default context.
    p = launch(binary, model, draft, os.path.join(TMP, "nomax.log"), {"SPARKINFER_MAX_OUTPUT_TOKENS": "16384"})
    before = metric("sparkinfer_speculative_runs_total")
    r = post({"model": "q", "messages": [{"role": "user", "content": "Name three primary colours, briefly."}],
              "temperature": 0.7, "seed": 5})
    after = metric("sparkinfer_speculative_runs_total")
    stop(p)
    spec_nomax = after > before
    print(f"part 2: a request without max_tokens speculated={spec_nomax} "
          f"({r['usage']['completion_tokens']} tokens)")
    ok = same and handoffs >= 1 and spec_nomax
    print("SPEC_REACH_CHECK", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
