#!/usr/bin/env python3
"""A speculation group adopts a request already decoding, and that request's tokens do not change.

With SPARKINFER_SPEC_GROUP=2:
1. three requests arrive together (more than a group takes, so they decode ordinarily);
2. the two short ones finish;
3. a fresh fourth arrives while the long first one is still decoding. The group forms and adopts
   the first one as a member without a draft.

Every completion must equal a SPARKINFER_SPECULATIVE=0 launch, and the trace must show the adoption.

usage: spec_adopt_check.py <server-bin> <model-dir> <draft-dir>
"""
import json, os, signal, subprocess, sys, tempfile, threading, time, urllib.request

PORT = 18153
TMP = tempfile.mkdtemp(prefix="spec_adopt_")
REQS = [  # (prompt, max_tokens, temperature, delay before sending)
    ("Write a long, detailed essay on the history of cartography, from clay tablets to satellites.", 900, 0.7, 0.0),
    ("Say hello.", 12, 0.0, 0.0),
    ("Name a colour.", 12, 0.7, 0.0),
    ("Explain how a hash map handles collisions, with a short code example.", 400, 0.7, 3.0),
]


def post(body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def launch(binary, model, draft, log, extra):
    env = dict(os.environ, SPARKINFER_DETERMINISTIC="1", SPARKINFER_PREFIX_CACHE="0",
               SPARKINFER_SPEC_GROUP="2", **extra)
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
    out = [None] * len(REQS)

    def one(i, prompt, n, t, delay):
        time.sleep(delay)
        r = post({"model": "q", "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
                  "temperature": t, "top_p": 0.95, "top_k": 20, "seed": 100 + i})
        out[i] = r["choices"][0]["message"]["content"]

    ts = [threading.Thread(target=one, args=(i, *r)) for i, r in enumerate(REQS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def main():
    binary, model, draft = sys.argv[1:4]
    log = os.path.join(TMP, "spec.log")
    p = launch(binary, model, draft, log, {"SPARKINFER_SPEC_GROUP_TRACE": "1"})
    spec = run_set()
    stop(p)
    text = open(log).read()
    adopted = sum(int(l.split(",")[-1].split()[0]) for l in text.splitlines()
                  if "[spec-group] start" in l and "adopted" in l)
    p = launch(binary, model, draft, os.path.join(TMP, "ref.log"), {"SPARKINFER_SPECULATIVE": "0"})
    ref = run_set()
    stop(p)
    same = [a == b for a, b in zip(spec, ref)]
    print(f"identical={same} adopted={adopted}")
    print("SPEC_ADOPT_CHECK", "PASS" if all(same) and adopted >= 1 else "FAIL")


if __name__ == "__main__":
    main()
