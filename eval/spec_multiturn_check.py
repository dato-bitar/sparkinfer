#!/usr/bin/env python3
"""End-to-end check for speculative decoding on multi-turn chats through sparkinfer_server.

Every turn of a conversation re-sends the conversation so far, so from the second turn on its
prompt starts with the previous turn's prompt and answer: a prefix-cache hit. Three launches from
the same binary run the same conversations, one request at a time, prefix cache on:
  new    draft loaded: a hit speculates, prefilling past the cached prefix, and a speculated
         prompt takes its prefix-cache checkpoints (the branch default)
  old    draft loaded, SPARKINFER_SPEC_PREFIX_HIT=0: a speculated prompt takes no checkpoints, so
         the next turn misses and re-prefills everything (and a hit, when there is one, decodes
         token by token)
  ref    draft loaded, SPARKINFER_SPECULATIVE=0: every request decodes token by token (the same
         device memory as the other two, so the same prefill and decode paths)
All run with SPARKINFER_DETERMINISTIC=1 and SPARKINFER_PREFIX_CACHE=1 (which keeps the cache on
under it). `new` must reproduce `ref` exactly -- the same prefill
splits (the checkpoints and the cached prefix are the same) and lossless speculation; `old`
re-prefills differently, so it is reported for speed only. Prints SPEC_MULTITURN_CHECK PASS when
every `new` turn matches `ref`.

usage: spec_multiturn_check.py <server-bin> <model-dir> <draft-dir> <out.json> [system-repeat]
  system-repeat: how many times the system prompt's paragraph repeats (default 7, ~1.6K-token
  prompts); 30 gives ~7K-token prompts, where a cache hit saves most of the prefill.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORT = 18133
SYSTEM = (
    "You are a senior software engineer helping a colleague. Answer precisely and concretely. "
    "When you write code, use Python 3.11 with type hints, keep functions short, and explain any "
    "non-obvious decision in one sentence after the code. Prefer the standard library. When a "
    "question is ambiguous, state the assumption you make and continue. Keep answers under 300 "
    "words unless the colleague asks for more. The project you both work on is a log-processing "
    "service: it reads newline-delimited JSON records from files and sockets, validates them "
    "against a schema, enriches them with geo-IP and user-agent data, aggregates counters per "
    "minute, and writes results to a time-series store. It runs as a single asyncio process per "
    "host, handles about 40,000 records per second at peak, and must never drop a record "
    "silently: malformed input goes to a dead-letter file with the reason. Configuration comes "
    "from a TOML file and environment variables, environment winning. Tests use pytest with "
    "fixtures under tests/fixtures, and the CI runs mypy --strict and ruff. "
) * (int(sys.argv[5]) if len(sys.argv) > 5 else 7)
CONVERSATIONS = [
    ["Write the function that parses one input line into a record dict, with the dead-letter rule.",
     "Now add a unit test for a line that is valid JSON but misses the timestamp field.",
     "How would you make the parser 2x faster without changing its behaviour?",
     "Summarize the three changes we made in this conversation as a commit message."],
    ["Design the per-minute aggregation: which data structure, and how do late records get handled?",
     "Write the class for it.",
     "What happens at a daylight-saving change? Fix it if needed.",
     "Give me a test plan in five bullet points."],
    ["The service sometimes uses 3 GB of memory after a day. List likely causes in this design.",
     "Write a small tracemalloc-based helper to find the biggest allocation sites at runtime.",
     "Suppose it is the geo-IP cache. Propose a bounded cache with an eviction policy and code it.",
     "What metrics should we export to confirm the fix in production?"],
]
SAMPLING = [(0.0, 0), (0.7, 11)]   # (temperature, seed offset)
MAX_TOKENS = 256


def stream_chat(body, timeout=900):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    reasoning, content, usage = [], [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece_r, piece_c = d.get("reasoning_content") or "", d.get("content") or ""
                if (piece_r or piece_c) and ttft is None:
                    ttft = time.time() - t0
                reasoning.append(piece_r)
                content.append(piece_c)
    total = time.time() - t0
    return "".join(reasoning), "".join(content), usage or {}, ttft or total, total


def metrics():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30) as r:
        out = {}
        for line in r.read().decode().splitlines():
            if line.startswith(("sparkinfer_speculative_", "sparkinfer_prefix_cache_")) and " " in line:
                k, v = line.rsplit(" ", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
        return out


def run(server, model, draft, env_extra, label, log_dir):
    # SPARKINFER_PREFIX_CACHE=1 keeps the cache on under SPARKINFER_DETERMINISTIC=1: every launch
    # sends the same requests in the same order, so it hits the same prefixes.
    env = dict(os.environ, SPARKINFER_DETERMINISTIC="1", SPARKINFER_PREFIX_CACHE="1", **env_extra)
    cmd = [server, "-m", model, "--tokenizer", os.path.join(model, "tokenizer.json"),
           "--model-name", "q", "--ctx", "32768", "--host", "127.0.0.1", "--port", str(PORT)]
    if draft:
        cmd += ["--draft-model", draft]
    log = open(os.path.join(log_dir, f"server_{label}.log"), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
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
        for temp, seed_off in SAMPLING:
            for c, turns in enumerate(CONVERSATIONS):
                messages = [{"role": "system", "content": SYSTEM}]
                for t, user in enumerate(turns):
                    messages.append({"role": "user", "content": user})
                    body = {"model": "q", "messages": messages, "max_tokens": MAX_TOKENS,
                            "temperature": temp, "seed": 500 + 17 * c + seed_off, "top_p": 0.95,
                            "top_k": 20}
                    rsn, cnt, usage, ttft, total = stream_chat(body)
                    results[f"{temp}:{c}:{t}"] = {"text": rsn + "\x00" + cnt,
                                                  "tokens": usage.get("completion_tokens", 0),
                                                  "prompt_tokens": usage.get("prompt_tokens", 0),
                                                  "ttft": ttft, "total": total}
                    # The next turn continues from what this launch generated: identical launches
                    # therefore send identical conversations.
                    messages.append({"role": "assistant", "content": cnt})
        return results, metrics()
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
        time.sleep(5)


def summary(res, keys):
    ttft = sorted(res[k]["ttft"] for k in keys)
    dec = sum(res[k]["tokens"] for k in keys) / max(1e-9, sum(res[k]["total"] - res[k]["ttft"] for k in keys))
    return {"ttft_p50_ms": round(1000 * ttft[len(ttft) // 2], 1), "decode_tok_s": round(dec, 1)}


def main():
    server, model, draft, out = sys.argv[1:5]
    log_dir = os.path.dirname(os.path.abspath(out))
    new, m_new = run(server, model, draft, {}, "new", log_dir)
    old, m_old = run(server, model, draft, {"SPARKINFER_SPEC_PREFIX_HIT": "0"}, "old", log_dir)
    ref, m_ref = run(server, model, draft, {"SPARKINFER_SPECULATIVE": "0"}, "ref", log_dir)
    report = {"metrics": {"new": m_new, "old": m_old, "ref": m_ref}, "by": {}}
    all_ok = True
    for temp, _ in SAMPLING:
        for first in (True, False):
            keys = [f"{temp}:{c}:{t}" for c in range(len(CONVERSATIONS)) for t in range(4)
                    if (t == 0) == first]
            same = sum(new[k]["text"] == ref[k]["text"] for k in keys)
            row = {"identical_new_ref": f"{same}/{len(keys)}", "new": summary(new, keys),
                   "old": summary(old, keys), "ref": summary(ref, keys)}
            label = f"T={temp} {'turn 1  ' if first else 'turns 2-4'}"
            report["by"][label] = row
            print(f"{label}: identical {same}/{len(keys)}  TTFT p50 new {row['new']['ttft_p50_ms']} / "
                  f"old {row['old']['ttft_p50_ms']} / ref {row['ref']['ttft_p50_ms']} ms  decode new "
                  f"{row['new']['decode_tok_s']} / old {row['old']['decode_tok_s']} / ref "
                  f"{row['ref']['decode_tok_s']} tok/s")
            if same != len(keys):
                all_ok = False
                for k in keys:
                    a, b = new[k]["text"], ref[k]["text"]
                    if a != b:
                        n = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                        print(f"  differs {k} at char {n}: new ...{a[max(0, n - 40):n + 40]!r}")
                        print(f"  {' ' * len('differs ' + k)}    ref ...{b[max(0, n - 40):n + 40]!r}")
    for lbl, m in (("new", m_new), ("old", m_old), ("ref", m_ref)):
        print(f"metrics {lbl}:", {k: v for k, v in m.items() if v})
    report["results"] = {"new": new, "old": old, "ref": ref}
    json.dump(report, open(out, "w"), indent=1)
    print("SPEC_MULTITURN_CHECK", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
