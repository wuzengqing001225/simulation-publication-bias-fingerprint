"""Cross-model replication runner (OpenAI-compatible API).

Reruns the main simulation (variant A + B, minimal participant framing) and the
bare-task battery on any OpenAI-compatible model, producing response files in the
exact same JSON schema as the Claude waves, so the existing parsers apply unchanged.

Usage:
  OPENAI_API_KEY=sk-...  python run_xmodel.py --model gpt-5.2 --n 40 --out gpt52
  # optional: --base-url http://localhost:8000/v1  (any OpenAI-compatible server)
  # optional: --effects Knobe2003_side_effect,Tversky1973_availability_letters  (subset)
  # a full run at --n 40 is ~61 effects x 2 variants x 2 conds x 40 = ~9,800 calls

Outputs (in --out directory):
  xm_main_resp.json  [{effect, variant, model, persona, cond, subj, text, error}]
  xm_bare_resp.json  [{effect, seed, model, cond, subj, text, error}]
Score them with code/xm_analyze.py (see the repository README).
"""
import argparse, json, os, random, re, time
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-bare", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--effects", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
    bank = json.load(open("P2_master_protocol_bank.json"))["protocols"]
    keep = set(args.effects.split(",")) if args.effects else None
    protos = [p for p in bank if p["variant"] in ("A","B") and (keep is None or p["effect"] in keep)]

    # main simulation
    out_main, done = [], 0
    for p in protos:
        conds = list(p.get("conditions") or {None: None})
        for i in range(args.n):
            cond = conds[i % len(conds)]
            txt, err = call(client, args.model, render_main(p, cond), SYS)
            out_main.append({"effect": p["effect"], "variant": p["variant"], "model": args.model,
                             "persona": "none", "cond": cond, "subj": i, "text": txt, "error": err})
        done += 1
        if done % 5 == 0:
            print(f"main: {done}/{len(protos)} protocols"); 
            json.dump(out_main, open(f"{args.out}/xm_main_resp.json","w"))
    json.dump(out_main, open(f"{args.out}/xm_main_resp.json","w"))

    # bare battery (variant A only, judgment paradigms; 5 seeds handled by subj index)
    pc = {}
    import csv
    for r in csv.DictReader(open("P2_paradigm_class.csv")):
        pc[r["effect"]] = r.get("bare_eligible","True")
    out_bare = []
    for p in [x for x in protos if x["variant"]=="A" and pc.get(x["effect"],"True") in ("True","1","true")]:
        conds = list(p.get("conditions") or {None: None})
        for i in range(args.n_bare):
            cond = conds[i % len(conds)]
            txt, err = call(client, args.model, render_bare(p, cond), "You are a helpful assistant.")
            out_bare.append({"effect": p["effect"], "seed": i % 5, "model": args.model,
                             "cond": cond, "subj": i, "text": txt, "error": err})
    json.dump(out_bare, open(f"{args.out}/xm_bare_resp.json","w"))
    print("done:", len(out_main), "main +", len(out_bare), "bare ->", args.out)

if __name__ == "__main__":
    main()
