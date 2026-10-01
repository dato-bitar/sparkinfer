#!/usr/bin/env python3
"""The draft steps off the device under load and comes back intact.

One server (draft loaded, SPARKINFER_DRAFT_OFFLOAD_MS=200): seeded prompts one at a time (they
speculate), then a 16-request burst (the draft goes to the host), then the same prompts again (it
comes back and they speculate again). A second launch with SPARKINFER_SPECULATIVE=0 is the
reference. PASS when all three sets of completions are identical and the log shows both moves.

usage: offload_check.py <server-bin> <model-dir> <draft-dir>
"""
import json, os, signal, subprocess, sys, tempfile, threading, time, urllib.request

PORT = 18137
TMP = tempfile.mkdtemp(prefix="offload_check_")
PROMPTS = [
    "Write a Python function that parses an ISO-8601 date string and returns a datetime, with error handling and three unit tests.",
    "Explain how a bloom filter works, when to use one, and how to choose its size and number of hash functions.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "List the planets of the solar system in order, with one interesting fact about each.",
]
TEMPS = [0.0, 0.7]


def post(body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(prompt, temp, n=256):
    r = post({"model": "q", "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
              "temperature": temp, "top_p": 0.95, "top_k": 20, "seed": 1234})
    return r["choices"][0]["message"]["content"]


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
    return {(i, t): ask(pr, t) for t in TEMPS for i, pr in enumerate(PROMPTS)}


def main():
    binary, model, draft = sys.argv[1:4]
    log = os.path.join(TMP, "offload_check_spec.log")
    p = launch(binary, model, draft, log, {"SPARKINFER_DRAFT_OFFLOAD_MS": "200"})
    first = run_set()
    burst = [threading.Thread(target=ask, args=(PROMPTS[i % 4] + f" (variant {i})", 0.7, 384)) for i in range(16)]
    for t in burst:
        t.start()
    for t in burst:
        t.join()
    second = run_set()
    stop(p)
    text = open(log).read()
    off, back = "draft off the device" in text, "draft back on the device" in text
    print("log:", [l for l in text.splitlines() if "[spec] draft" in l or "step off the device" in l][:6])
    p = launch(binary, model, draft, os.path.join(TMP, "offload_check_ref.log"), {"SPARKINFER_SPECULATIVE": "0"})
    ref = run_set()
    stop(p)
    bad = [k for k in ref if not (first[k] == second[k] == ref[k])]
    for k in bad:
        print("MISMATCH", k, "first==ref", first[k] == ref[k], "second==ref", second[k] == ref[k])
    ok = not bad and off and back
    print(f"offloaded={off} restored={back} identical={not bad} ({len(ref) - len(bad)}/{len(ref)})")
    print("OFFLOAD_CHECK", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
