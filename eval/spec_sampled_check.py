#!/usr/bin/env python3
"""End-to-end check for sampled speculative decoding through sparkinfer_server.

Runs the same seeded requests, one at a time, against two server launches from the same binary:
  spec   -- DSpark draft loaded, sampled requests speculate (the branch default)
  plain  -- DSpark draft loaded, SPARKINFER_SPEC_SAMPLED=0: sampled requests decode token by token
Greedy requests speculate in both. Reports, per temperature, whether every completion matches
between the two launches, the decode speed of each, and the speculative counters.

Both launches run with SPARKINFER_DETERMINISTIC=1 (batched prefill is otherwise not bit-reproducible
across launches, so even greedy completions can fork) and SPARKINFER_PREFIX_CACHE=0 (a prefix-cache
hit is not speculated). Prints SPEC_SAMPLED_CHECK PASS when every completion is identical.

usage: spec_sampled_check.py <server-bin> <model-dir> <draft-dir> <out.json>
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORT = 18131
PROMPTS = [
    "Write a Python function that parses an ISO-8601 date string and returns a datetime, with error handling and three unit tests.",
    "Explain how a bloom filter works, when to use one, and how to choose its size and number of hash functions.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "List the planets of the solar system in order, with one interesting fact about each.",
    "Translate into French and then back into English: 'The committee postponed the decision until the budget review next spring.'",
    "Give a step-by-step recipe for a simple tomato pasta sauce for four people.",
]
TEMPS = [0.0, 0.7, 1.0]
MAX_TOKENS = 256


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metrics():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30) as r:
        out = {}
        for line in r.read().decode().splitlines():
            if line.startswith("sparkinfer_speculative_"):
                k, v = line.split()
                out[k] = float(v)
        return out


def run(server, model, draft, env_extra, label, log_dir):
    env = dict(os.environ, SPARKINFER_PREFIX_CACHE="0", SPARKINFER_DETERMINISTIC="1", **env_extra)
    log = open(os.path.join(log_dir, f"server_{label}.log"), "w")
    p = subprocess.Popen([server, "-m", model, "--tokenizer", os.path.join(model, "tokenizer.json"),
                          "--model-name", "q", "--ctx", "32768", "--draft-model", draft,
                          "--host", "127.0.0.1", "--port", str(PORT)],
                         stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        for _ in range(360):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
                break
            except Exception:
                if p.poll() is not None:
                    raise RuntimeError(f"{label}: server exited")
                time.sleep(5)
        results = {}
        for t in TEMPS:
            for i, prompt in enumerate(PROMPTS):
                body = {"model": "q", "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": MAX_TOKENS, "temperature": t, "seed": 1000 + i,
                        "top_p": 0.95, "top_k": 20}
                t0 = time.time()
                r = post("/v1/chat/completions", body)
                dt = time.time() - t0
                msg = r["choices"][0]["message"]
                text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
                results[f"{t}:{i}"] = {"text": text, "tokens": r["usage"]["completion_tokens"], "s": dt}
        return results, metrics()
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
        time.sleep(5)


def main():
    server, model, draft, out = sys.argv[1:5]
    log_dir = os.path.dirname(os.path.abspath(out))
    spec, m_spec = run(server, model, draft, {}, "spec", log_dir)
    plain, m_plain = run(server, model, draft, {"SPARKINFER_SPEC_SAMPLED": "0"}, "plain", log_dir)
    report = {"metrics_spec": m_spec, "metrics_plain": m_plain, "by_temp": {}}
    all_ok = True
    for t in TEMPS:
        keys = [f"{t}:{i}" for i in range(len(PROMPTS))]
        same = sum(spec[k]["text"] == plain[k]["text"] for k in keys)
        tok_s = sum(spec[k]["tokens"] for k in keys) / sum(spec[k]["s"] for k in keys)
        tok_p = sum(plain[k]["tokens"] for k in keys) / sum(plain[k]["s"] for k in keys)
        report["by_temp"][str(t)] = {"identical": f"{same}/{len(keys)}", "spec_tok_s": round(tok_s, 1),
                                     "plain_tok_s": round(tok_p, 1), "speedup": round(tok_s / tok_p, 3)}
        print(f"T={t}: identical {same}/{len(keys)}  spec {tok_s:.1f} tok/s  plain {tok_p:.1f} tok/s  "
              f"x{tok_s / tok_p:.2f}")
        if same != len(keys):
            all_ok = False
            for k in keys:
                if spec[k]["text"] != plain[k]["text"]:
                    a, b = spec[k]["text"], plain[k]["text"]
                    n = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                    print(f"  differs {k} at char {n}: spec ...{a[max(0,n-40):n+40]!r}")
                    print(f"  {' ' * len('differs ' + k)}    plain ...{b[max(0,n-40):n+40]!r}")
    print("metrics spec:", m_spec)
    print("metrics plain:", m_plain)
    report["results"] = {"spec": spec, "plain": plain}
    json.dump(report, open(out, "w"), indent=1)
    print("SPEC_SAMPLED_CHECK", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
