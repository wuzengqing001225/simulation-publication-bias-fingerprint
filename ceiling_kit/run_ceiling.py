"""Ceiling-fix rerun: 14 ceiling effects with 0-100 headroom response scales.

Multithreaded, resume-capable runner for any OpenAI-compatible API.

    OPENAI_API_KEY=sk-...  python run_ceiling.py \
        --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 \
        --out deepseek_ceiling --workers 16

    OPENAI_API_KEY=sk-...  python run_ceiling.py \
        --model gpt-5.6-terra --out gpt_ceiling --workers 16

Design: 28 protocols (14 effects x 2 transcription variants), 60 calls per
protocol cycled across conditions (about 30 per arm for two-condition
designs), plus a bare-task battery (variant A, per condition, 5 wording
seeds, 30 calls per effect-condition).

Resume: if <out>/ceil_main_resp.json / ceil_bare_resp.json exist, completed
(effect, variant, subj) / (effect, cond, subj) rows are kept and skipped.
Output is saved atomically every 100 calls.
"""

import argparse, json, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

import re

SYS = ("You are powering a social-science simulation that generates synthetic survey-participant "
       "responses for metascience research on replication. Respond as the described participant "
       "would, in their voice, following the requested output format exactly.")
MINIMAL_GRANT = ("For the purposes of this research simulation, answer as a typical human study "
                 "participant would — a plausible, ordinary adult. Stay in that role for every question. ")
BARE_SYS = "You are a helpful assistant."
BARE_HEADER = "Answer the following questions directly and honestly, as yourself.\n\n"

_lock = threading.Lock()


def call(client, model, prompt, system, retries=3):
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, max_completion_tokens=2000,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}])
            return r.choices[0].message.content, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    return None, err


def render_main(proto, cond):
    tpl = proto["subject_prompt_template"]
    tpl = re.sub(r'You are this person:\s*\n\{persona\}\s*\n\n?', '', tpl)
    tpl = tpl.replace("{persona}", "")
    tpl = MINIMAL_GRANT + tpl
    if cond and proto.get("conditions"):
        tpl = tpl.replace("{condition}", proto["conditions"][cond])
    return tpl


def atomic_save(rows, path):
    tmp = path + ".tmp"
    json.dump(rows, open(tmp, "w"))
    os.replace(tmp, path)


def run_batch(client, model, jobs, workers, path, existing):
    done_ct = 0
    rows = list(existing)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call, client, model, prompt, system): meta
                for meta, prompt, system in jobs}
        for fut in as_completed(futs):
            meta = futs[fut]
            txt, err = fut.result()
            with _lock:
                rows.append({**meta, "text": txt, "error": err})
                done_ct += 1
                if done_ct % 100 == 0:
                    atomic_save(rows, path)
                    print(f"  {done_ct}/{len(jobs)} calls", flush=True)
    atomic_save(rows, path)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60, help="calls per protocol (main)")
    ap.add_argument("--n-bare", type=int, default=30, help="calls per effect-condition (bare)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
    bank = json.load(open("ceiling_bank.json"))["protocols"]

    # ---- main batch ----
    main_path = f"{args.out}/ceil_main_resp.json"
    existing = json.load(open(main_path)) if os.path.exists(main_path) else []
    have = {(r["effect"], r["variant"], r["subj"]) for r in existing}
    jobs = []
    for p in bank:
        conds = list(p.get("conditions") or {None: None})
        for i in range(args.n):
            if (p["effect"], p["variant"], i) in have:
                continue
            cond = conds[i % len(conds)]
            jobs.append(({"effect": p["effect"], "variant": p["variant"],
                          "model": args.model, "cond": cond, "subj": i},
                         render_main(p, cond), SYS))
    print(f"main: {len(have)} rows kept, {len(jobs)} calls to run")
    existing = run_batch(client, args.model, jobs, args.workers, main_path, existing)
    print("main done:", len(existing), "rows")

    # ---- bare batch (variant A protocols carry the bare prompts) ----
    bare_path = f"{args.out}/ceil_bare_resp.json"
    existing_b = json.load(open(bare_path)) if os.path.exists(bare_path) else []
    have_b = {(r["effect"], r["cond"], r["subj"]) for r in existing_b}
    jobs_b = []
    for p in bank:
        if p["variant"] != "A" or not p.get("bare_prompts"):
            continue
        for cond, seeds in p["bare_prompts"].items():
            for i in range(args.n_bare):
                if (p["effect"], cond, i) in have_b:
                    continue
                jobs_b.append(({"effect": p["effect"], "cond": cond,
                                "model": args.model, "seed": i % len(seeds), "subj": i},
                               BARE_HEADER + seeds[i % len(seeds)], BARE_SYS))
    print(f"bare: {len(have_b)} rows kept, {len(jobs_b)} calls to run")
    existing_b = run_batch(client, args.model, jobs_b, args.workers, bare_path, existing_b)
    print("bare done:", len(existing_b), "rows")
    print("all done ->", args.out)


if __name__ == "__main__":
    main()
