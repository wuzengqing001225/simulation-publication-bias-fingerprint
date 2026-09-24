"""Multithreaded cross-model runner with resume support.

Drop this file into the same xmodel_kit folder and run:

    OPENAI_API_KEY=sk-...  python run_xmodel_mt.py \
        --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 \
        --n 30 --out deepseek_full --workers 16

Resume behavior: if <out>/xm_main_resp.json or <out>/xm_bare_resp.json already
exist (from a previous run of run_xmodel.py or this script), every (effect,
variant) cell that is already complete (all rows present with text or a
recorded refusal) is skipped, so you keep the ~20 protocols you already paid
for. Output schema is identical to run_xmodel.py.
"""
import argparse, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI

SYS = ("You are powering a social-science simulation that generates synthetic survey-participant "
       "responses for metascience research on replication. Respond as the described participant "
       "would, in their voice, following the requested output format exactly.")
MINIMAL_GRANT = ("For the purposes of this research simulation, answer as a typical human study "
                 "participant would — a plausible, ordinary adult. Stay in that role for every question. ")
BARE_HEADER = "Answer the following questions directly and honestly, as yourself.\n\n"

STRIP_PATTERNS = [
    r'(?i)you are taking part in[^.\n]*\.\s*', r'(?i)you are (a )?participant[^.\n]*\.\s*',
    r'(?i)you have given consent[^.\n]*\.\s*', r'(?i)you are alone at a desk[^.\n]*\.\s*',
    r'(?i)a research assistant[^.\n]*\.\s*', r'(?i)the experimenter[^.\n]*\.\s*',
    r'(?i)welcome to (the|this) (study|experiment)[^.\n]*\.\s*',
]

def render_main(proto, cond_key):
    tpl = proto["subject_prompt_template"]
    tpl = re.sub(r'You are this person:\s*\n\{persona\}\s*\n\n?', '', tpl)
    tpl = tpl.replace("{persona}", "")
    tpl = MINIMAL_GRANT + tpl
    if cond_key and proto.get("conditions"):
        tpl = tpl.replace("{condition}", proto["conditions"][cond_key])
    return tpl

def render_bare(proto, cond_key):
    tpl = proto["subject_prompt_template"]
    tpl = re.sub(r'You are this person:\s*\n\{persona\}\s*\n\n?', '', tpl)
    tpl = tpl.replace("{persona}", "")
    for p in STRIP_PATTERNS:
        tpl = re.sub(p, '', tpl)
    if cond_key and proto.get("conditions"):
        tpl = tpl.replace("{condition}", proto["conditions"][cond_key])
    return BARE_HEADER + tpl

def call(client, model, prompt, system, max_completion_tokens=2000, retries=3):
    for a in range(retries):
        try:
            r = client.chat.completions.create(model=model, max_completion_tokens=max_completion_tokens,
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
            return r.choices[0].message.content, None
        except Exception as e:
            if a == retries-1: return None, str(e)[:150]
            time.sleep(3*(a+1))

def run_batch(client, model, jobs, workers, save_path, existing, save_every=100):
    """jobs: list of (meta_dict, prompt, system). Appends to existing, saves periodically."""
    out = list(existing)
    lock = Lock()
    done_ct = [0]
    def work(job):
        meta, prompt, system = job
        txt, err = call(client, model, prompt, system)
        row = {**meta, "text": txt, "error": err}
        with lock:
            out.append(row)
            done_ct[0] += 1
            if done_ct[0] % save_every == 0:
                tmp = save_path + ".tmp"
                json.dump(out, open(tmp, "w")); os.replace(tmp, save_path)
                print(f"  {done_ct[0]}/{len(jobs)} calls done", flush=True)
        return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs): f.result()
    tmp = save_path + ".tmp"
    json.dump(out, open(tmp, "w")); os.replace(tmp, save_path)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-bare", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--effects", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
    bank = json.load(open("P2_master_protocol_bank.json"))["protocols"]
    keep = set(args.effects.split(",")) if args.effects else None
    protos = [p for p in bank if p["variant"] in ("A","B") and (keep is None or p["effect"] in keep)]

    # ---- main simulation ----
    main_path = f"{args.out}/xm_main_resp.json"
    existing = json.load(open(main_path)) if os.path.exists(main_path) else []
    have = {}
    for r in existing:
        have.setdefault((r["effect"], r["variant"]), set()).add(r["subj"])
    jobs = []
    for p in protos:
        conds = list(p.get("conditions") or {None: None})
        done = have.get((p["effect"], p["variant"]), set())
        for i in range(args.n):
            if i in done: continue
            cond = conds[i % len(conds)]
            jobs.append(({"effect": p["effect"], "variant": p["variant"], "model": args.model,
                          "persona": "none", "cond": cond, "subj": i}, render_main(p, cond), SYS))
    print(f"main: resuming with {len(existing)} rows kept, {len(jobs)} calls to run, {args.workers} workers")
    existing = run_batch(client, args.model, jobs, args.workers, main_path, existing)
    print("main done:", len(existing), "rows total")

    # ---- bare battery ----
    import csv
    pc = {r["effect"]: r.get("bare_eligible","True") for r in csv.DictReader(open("P2_paradigm_class.csv"))}
    bare_path = f"{args.out}/xm_bare_resp.json"
    existing_b = json.load(open(bare_path)) if os.path.exists(bare_path) else []
    have_b = {}
    for r in existing_b:
        have_b.setdefault(r["effect"], set()).add(r["subj"])
    jobs_b = []
    for p in [x for x in protos if x["variant"]=="A" and pc.get(x["effect"],"True") in ("True","1","true")]:
        conds = list(p.get("conditions") or {None: None})
        done = have_b.get(p["effect"], set())
        for i in range(args.n_bare):
            if i in done: continue
            cond = conds[i % len(conds)]
            jobs_b.append(({"effect": p["effect"], "seed": i % 5, "model": args.model,
                            "cond": cond, "subj": i}, render_bare(p, cond), "You are a helpful assistant."))
    print(f"bare: resuming with {len(existing_b)} rows kept, {len(jobs_b)} calls to run")
    existing_b = run_batch(client, args.model, jobs_b, args.workers, bare_path, existing_b)
    print("bare done:", len(existing_b), "rows total")

if __name__ == "__main__":
    main()
