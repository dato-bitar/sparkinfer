#!/usr/bin/env python3
"""The prefix cache's host KV tier returns exactly the KV it took.

Two launches from the same binary, both deterministic with the prefix cache on, run the same
multi-turn conversations, interleaved (every conversation's first turn, then every second turn,
...), so later turns hit the cache:
  host     the cache may hold only ~2% of the KV pool on the device
           (SPARKINFER_PREFIX_CACHE_KV_PCT=2): entries move to the host tier, and hits copy them back
  device   the default share: the hits are served from the device
A hit resumes at the same position either way and reads the same recurrent-state snapshot, so the
only difference is where the KV bytes came from: every answer must be identical, and the host
launch must have served hits from the host tier.

usage: prefix_host_tier_check.py <server-bin> <model-dir> [ENV=VAL ...]   (extra env for both)
"""
import json, os, signal, subprocess, sys, time, urllib.request

server, model = sys.argv[1:3]
extra = dict(kv.split("=", 1) for kv in sys.argv[3:])
PORT = 18143
WORDS = ("amber basin cobalt delta ember fjord granite harbor island juniper kestrel lagoon meadow "
         "nickel orchard prairie quartz ridge summit tundra").split()
SYSTEM = "You are a meticulous archivist. Reference list: " + " ".join(
    f"{WORDS[i % len(WORDS)]}-{i}" for i in range(700))
CONVS, TURNS = 6, 3


def launch(host):
    env = dict(os.environ, **extra, SPARKINFER_DETERMINISTIC="1", SPARKINFER_PREFIX_CACHE="1")
    if host:
        env["SPARKINFER_PREFIX_CACHE_KV_PCT"] = "2"
    p = subprocess.Popen([server, "-m", model, "--tokenizer", os.path.join(model, "tokenizer.json"),
                          "--model-name", "q", "--ctx", "32768", "--host", "127.0.0.1", "--port", str(PORT)],
                         stdout=open(f"/tmp/phtc_{int(host)}.log", "w"), stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    for _ in range(360):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            return p
        except Exception:
            if p.poll() is not None:
                sys.exit("server exited")
            time.sleep(5)
    sys.exit("server did not come up")


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


def chat(messages):
    body = {"model": "q", "messages": messages, "max_tokens": 64, "temperature": 0.0}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def metrics():
    text = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()
    out = {}
    for l in text.splitlines():
        if l.startswith("sparkinfer_prefix_cache_"):
            k, v = l.rsplit(" ", 1)
            out[k] = float(v)
    return out


def run(host):
    p = launch(host)
    try:
        convs = [[{"role": "system", "content": SYSTEM}] for _ in range(CONVS)]
        answers = []
        for t in range(TURNS):
            for c in range(CONVS):
                convs[c].append({"role": "user", "content":
                                 f"Conversation {c}, turn {t}: name three entries from the list that "
                                 f"start with {WORDS[(c * 3 + t) % len(WORDS)]}, then one more."})
                a = chat(convs[c])
                convs[c].append({"role": "assistant", "content": a})
                answers.append(a)
        return answers, metrics()
    finally:
        stop(p)


host_ans, host_m = run(True)
dev_ans, dev_m = run(False)
same = sum(a == b for a, b in zip(host_ans, dev_ans))
g = lambda m, k: int(m.get(f"sparkinfer_prefix_cache_{k}", 0))
for name, m in (("host", host_m), ("device", dev_m)):
    print(f"{name}: hits {g(m, 'hits_total')}/{g(m, 'lookups_total')}, from the host tier "
          f"{g(m, 'host_hits_total')}, demotions {g(m, 'demotions_total')}, "
          f"host KV {g(m, 'host_kv_bytes') / 1e6:.0f} MB")
print(f"identical answers: {same}/{len(host_ans)}")
for i, (a, b) in enumerate(zip(host_ans, dev_ans)):
    if a != b:
        print(f"  differs at answer {i}:\n    host:   {a[:120]!r}\n    device: {b[:120]!r}")
ok = same == len(host_ans) and g(host_m, "host_hits_total") > 0 and g(host_m, "demotions_total") > 0 \
    and g(dev_m, "hits_total") > 0
print("PREFIX_HOST_TIER_CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
