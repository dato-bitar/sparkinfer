#!/usr/bin/env python3
"""Throughput and latency under open arrivals: requests arrive as a Poisson process rather than in a
closed loop, so requests are decoding when others arrive (what speculation-group adoption is for).

usage: spec_open_arrivals.py <port> <label> [rate_per_s] [n] [seed]
Prints total completion tokens / makespan, mean and p90 request latency, and per-class counts.
"""
import json, random, sys, threading, time, urllib.request

PROMPTS = [
    "Write a Python function that parses an ISO-8601 date string and returns a datetime, with error handling and three unit tests.",
    "Explain how a bloom filter works, when to use one, and how to choose its size and number of hash functions.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "List the planets of the solar system in order, with one interesting fact about each.",
    "Compare TCP and UDP and say when each is the right choice, with examples.",
    "Summarise the causes and consequences of the 2008 financial crisis.",
    "Give a step-by-step recipe for sourdough bread, including the starter.",
]


def main():
    port, label = sys.argv[1], sys.argv[2]
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 7)
    plan, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rate)
        short = rng.random() < 0.2
        plan.append((t, PROMPTS[i % len(PROMPTS)], 32 if short else rng.randint(128, 768), short, 1000 + i))
    res = [None] * n

    def one(i, at, prompt, mt, short, seed, t0):
        time.sleep(max(0.0, t0 + at - time.time()))
        body = {"model": "q", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt,
                "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": seed}
        s = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        res[i] = (r["usage"]["completion_tokens"], time.time() - s, short, time.time())

    t0 = time.time() + 1.0
    ts = [threading.Thread(target=one, args=(i, *p, t0)) for i, p in enumerate(plan)]
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    toks = sum(r[0] for r in res)
    span = max(r[3] for r in res) - t0
    lat = sorted(r[1] for r in res)
    print(f"{label}: {toks} tokens in {span:.1f} s = {toks / span:.1f} tok/s; latency mean {sum(lat)/len(lat):.2f} s "
          f"p90 {lat[int(0.9 * len(lat))]:.2f} s; short {sum(r[2] for r in res)}/{n}")


if __name__ == "__main__":
    main()
