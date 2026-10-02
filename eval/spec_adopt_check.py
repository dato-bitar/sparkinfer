#!/usr/bin/env python3
"""A speculation group adopts a request already decoding, and that request's tokens do not change.

With SPARKINFER_SPEC_GROUP=2:
1. three requests arrive together (more than a group takes, so they decode ordinarily);
2. the two short ones finish;
3. a fresh fourth arrives while the long first one is still decoding. The group forms and adopts
   the first one as a member without a draft.

The two requests that never meet a group must equal a SPARKINFER_SPECULATIVE=0 launch. The adopted
request and its group partner must complete. A group verifies with batch arithmetic, and the adopted
request's recurrent state goes back to fp32 (exactly widened from the bf16 packed decode kept), so
they need not match token for token; their similarity is printed. The trace must show the adoption.

usage: spec_adopt_check.py <server-bin> <model-dir> <draft-dir>
"""
import difflib, json, os, signal, subprocess, sys, tempfile, threading, time, urllib.request

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
        out[i] = (r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"])

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
    ok_runs = all(o is not None and o[1] in ("stop", "length") for o in spec)
    same = [spec[i] is not None and ref[i] is not None and spec[i][0] == ref[i][0] for i in range(len(REQS))]
    sim = [round(difflib.SequenceMatcher(None, spec[i][0], ref[i][0]).ratio(), 3)
           if spec[i] and ref[i] else None for i in (0, 3)]
    print(f"completed={ok_runs} identical={same} adopted={adopted} similarity(adopted, partner)={sim}")
    ok = ok_runs and same[1] and same[2] and adopted >= 1
    print("SPEC_ADOPT_CHECK", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
