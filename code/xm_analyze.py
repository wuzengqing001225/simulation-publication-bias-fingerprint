"""Cross-model analysis pipeline: raw response JSONs -> cell effect sizes -> analysis units.

End-to-end scorer for a cross-model run produced by xmodel_kit/run_xmodel.py
(or run_xmodel_mt.py). Reproduces the paper's GPT/DeepSeek scoring exactly.

Usage (from the repository root):

    python code/xm_analyze.py --main xm_main_resp.json --bare xm_bare_resp.json \
        --prefix XM [--fit]

Outputs: <prefix>_cells_new.csv, <prefix>_cells_old.csv, <prefix>_bare_new.csv,
<prefix>_bare_old.csv, <prefix>_units.csv; with --fit also the three-coefficient
measurement-error model on this model family's units (requires pymc, arviz).

Model-family note: the group parsers were written against Claude responses and
validated to reproduce the Claude reference cells; for a NEW model family, check
the per-effect parse rates printed at the end and add format fallbacks in the
style of xm_parsers_patch_new.py / ds_parsers_patch_new.py if any cell parses
below 0.60.
"""
import argparse, json, os, re, sys, collections, math
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True, help="xm_main_resp.json from the runner")
ap.add_argument("--bare", required=True, help="xm_bare_resp.json from the runner")
ap.add_argument("--prefix", default="XM", help="output file prefix")
ap.add_argument("--units", default="analysis_tables/P2_analysis_units.csv",
                help="units CSV carrying the human anchors (z_o, z_r, var_o, var_r)")
ap.add_argument("--bank", default="protocols/P2_master_protocol_bank.json")
ap.add_argument("--code-dir", default=None, help="directory with the parser modules (default: this script's dir)")
ap.add_argument("--fit", action="store_true", help="fit the measurement-error model on the resulting units")
args = ap.parse_args()
PREFIX = args.prefix
CODE_DIR = args.code_dir or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

bank = json.load(open(args.bank))["protocols"]
OLD19 = set("Asch1946 Bargh2012 Banerjee2012 Wakslak2006 Tamir2012 Klink2000 Valdesolo2007 Cooney2016 VanBoven2007 Slepian2012 Husnu2010 Tversky1973 Eyal2008 Rottenstreich2001 Oppenheimer2009 Knobe2003 Miyamoto2002 Genschow2017 Chao2017".split())
NEW42 = sorted({p["effect"] for p in bank if p["effect"].split("_")[0] not in OLD19})
assert len(NEW42) == 42

# exec the three parser modules into separate namespaces
G = {}
for g in range(3):
    ns = {"__name__": f"p2g{g}"}
    exec(open(os.path.join(CODE_DIR, f"p2_parsers_g{g}.py")).read(), ns)
    G[g] = ns
G0_EFF = sorted({k[0] for k in G[0]["SPEC"]})
G1_EFF = sorted(G[1]["PARSERS"])
G2_EFF = sorted(G[2]["PARSERS"])
assert sorted(G0_EFF + G1_EFF + G2_EFF) == NEW42
GROUP = {e: 0 for e in G0_EFF} | {e: 1 for e in G1_EFF} | {e: 2 for e in G2_EFF}

main_rows = json.load(open(args.main))
bare_rows = json.load(open(args.bare))
main_new = [r for r in main_rows if r["effect"] in set(NEW42)]
bare_new = [r for r in bare_rows if r["effect"] in set(NEW42)]
bare_by_eff = collections.defaultdict(list)
for r in bare_new:
    bare_by_eff[r["effect"]].append(r)

# ---- Patch for group 0 parser ----
def _mixed_line_values(text, spec, g0):
    if text is None:
        return None
    n = spec.get("n", len(spec.get("names", [])))
    vre = g0["_VALUE_RE"][spec["value"]]
    lines = [ln.strip() for ln in str(text).strip().split("\n") if ln.strip()]
    if len(lines) != n:
        return None
    lab = re.compile(rf"^\s*[A-Za-z][A-Za-z0-9 _\-]*\s*[:=：]\s*({vre})\s*$")
    bare = re.compile(rf"^\s*({vre})\s*$")
    vals = {}
    for i, ln in enumerate(lines, start=1):
        m = lab.match(ln) or bare.match(ln)
        if not m:
            return None
        try:
            v = g0["_cast"](spec, m.group(1))
        except ValueError:
            return None
        if not g0["_ok"](spec, i, v):
            return None
        vals[i] = v
    return vals if len(vals) == n else None


def parse_response_patched(text, spec, g0, tier=2):
    if text is None:
        return None, None
    n = spec.get("n", len(spec.get("names", [])))
    for t in (1, 2):
        if t > tier:
            break
        v = g0["_parse_tier"](text, spec, n, t)
        if v is not None:
            return v, t
    v = _mixed_line_values(text, spec, g0)
    if v is not None:
        return v, 2.5
    if tier >= 3:
        v = g0["_parse_tier"](text, spec, n, 3)
        if v is not None:
            return v, 3
    return None, None


def install(g0):
    if g0.get("_XM_PATCHED"):
        return g0
    g0["_xm_orig_parse_response"] = g0["parse_response"]
    g0["parse_response"] = lambda text, spec, tier=2: parse_response_patched(
        text, spec, g0, tier=tier)
    g0["_XM_PATCHED"] = True
    return g0

install(G[0])

# ---- Parser functions ----
def parse_g0(eff, var, text, tier):
    spec = G[0]["SPEC"][(eff, var)]
    vals, t = G[0]["parse_response"](text, spec, tier=tier)
    if vals is None: return None, None
    return G[0]["score_response"](vals, spec), t

def parse_g1(eff, var, text, lenient):
    return G[1]["parse"](eff, var, text, lenient=lenient)

def parse_g2(eff, var, cond, text, lenient):
    return G[2]["parse_row"](eff, text, var, cond, lenient=lenient)

# ---- G2 estimator specification ----
S2 = G[2]

def _z(a):
    a = np.asarray(a, float); s = a.std(ddof=1)
    return (a - a.mean()) / s if s > 0 else np.zeros_like(a)

SPEC2 = {
 "Kupor2015_god_risk":            dict(kind="two_group", dv="risk", hi="god", lo="neutral"),
 "McCullough2002_gratitude_agreeableness": dict(kind="pearson", x="gratitude", y="agreeableness"),
 "Mellers2001_conjunction_frequency": dict(kind="one_sample", dv="violation", mu=0.0, arm="and_are"),
 "Park2019_scarcity_price_quality": dict(kind="two_group", dv="pq", hi="control", lo="scarcity"),
 "Paunonen2003_attractiveness_extraversion": dict(kind="pearson", x="attractiveness", y="extraversion"),
 "Peters2006_numeracy_framing":    dict(kind="peters"),
 "Reinhard2013_unconscious_lie_detection": dict(kind="two_group", dv="accuracy", hi="unconscious_thought", lo="conscious_thought"),
 "Scopelliti2015_bragging_emotion": dict(kind="two_group", dv="pos", hi="self_promoter", lo="recipient"),
 "Shnabel2008_needs_reconciliation": dict(kind="inter2x2", dv="reconcile",
        plus=("perpetrator_acceptance","victim_empowerment"), minus=("perpetrator_empowerment","victim_acceptance")),
 "Stevenson2000_web_brand_attitude": dict(kind="pearson", x="aw", y="ab"),
 "Thoresen2003_commitment_negative_affect": dict(kind="pearson", x="commitment", y="neg_affect"),
 "Weinstein1984_comparative_optimism": dict(kind="one_sample", dv="risk", mu=4.0),
 "WilliamsBargh2008_distance_priming": dict(kind="two_group", dv="attachment", hi="close", lo="distant"),
 "Zarkadi2013_visual_contrast":    dict(kind="anova", dv="dev", top="black_white",
        levels=("black_white","gray","blue_yellow")),
}

def g2_point(eff, recs):
    sp = SPEC2[eff]; k = sp["kind"]
    if k == "two_group":
        A = [d[sp["dv"]] for c, d in recs if c == sp["hi"]]
        B = [d[sp["dv"]] for c, d in recs if c == sp["lo"]]
        fn = lambda bl: S2["r_two_group"](bl[0], bl[1])
        return fn([A, B]), [A, B], fn
    if k == "pearson":
        P = [(d[sp["x"]], d[sp["y"]]) for c, d in recs]
        fn = lambda bl: S2["r_pearson"]([p[0] for p in bl[0]], [p[1] for p in bl[0]])
        return fn([P]), [P], fn
    if k == "one_sample":
        V = [d[sp["dv"]] for c, d in recs if ("arm" not in sp or c == sp["arm"])]
        mu = sp["mu"]
        fn = lambda bl: S2["r_one_sample"](bl[0], mu)
        return fn([V]), [V], fn
    if k == "anova":
        lv = sp["levels"]; dv = sp["dv"]
        blocks = [[d[dv] for c, d in recs if c == L] for L in lv]
        fn = lambda bl: S2["r_anova_oneway"]({L: bl[i] for i, L in enumerate(lv)}, sp["top"])
        return fn(blocks), blocks, fn
    if k == "inter2x2":
        keys = list(sp["plus"]) + list(sp["minus"]); dv = sp["dv"]
        blocks = [[d[dv] for c, d in recs if c == K] for K in keys]
        fn = lambda bl: S2["r_interaction_2x2"]({K: bl[i] for i, K in enumerate(keys)}, sp["plus"], sp["minus"])
        return fn(blocks), blocks, fn
    if k == "peters":
        obj = np.array([d["obj"] for c, d in recs], float)
        sub = np.array([d["subj"] for c, d in recs], float)
        num = (_z(obj) + _z(sub)) / 2.0
        rat = [d["rating"] for c, d in recs]
        fr = [1.0 if c == "gain_frame" else 0.0 for c, d in recs]
        trip = list(zip(rat, fr, num))
        fn = lambda bl: S2["r_interaction"]([t[0] for t in bl[0]], [t[1] for t in bl[0]], [t[2] for t in bl[0]])
        return fn([trip]), [trip], fn
    raise ValueError(k)

# ---- Cell computation functions ----
def cell_g0(rows, eff, var, boot=0, seed=0):
    spec = G[0]["SPEC"][(eff, var)]
    d = G[0]["collect"](rows, spec, tier=2); mode = 2
    if d["n_denom"] and d["parse_rate"] is not None and d["parse_rate"] < 0.60:
        d = G[0]["collect"](rows, spec, tier=3); mode = 3
    r = G[0]["effect_size"](spec, d)
    lo = hi = np.nan
    if boot and r is not None:
        lo, hi = G[0]["bootstrap_ci"](spec, d, n_boot=boot, seed=seed)
    return dict(r=r, lo=lo, hi=hi, n_ok=d["n_ok"], n_ref=d["n_refused"],
                den=d["n_denom"], mode=mode, stat=spec["stat"], abs_primary=spec.get("abs_primary", False))

S1 = G[1]

def cell_g1(rows, eff, var, boot=0, seed=0):
    recs, n_ref = [], 0
    parsed = []
    for r in rows:
        if r["text"] is None or not str(r["text"]).strip():
            n_ref += 1; continue
        parsed.append((r, parse_g1(eff, var, r["text"], False)))
    den = len(parsed); ok = sum(1 for _, p in parsed if p)
    mode = "strict"
    if den and ok/den < 0.60:
        parsed = [(r, parse_g1(eff, var, r["text"], True) or p) for r, p in parsed]
        ok = sum(1 for _, p in parsed if p); mode = "lenient"
    recs = [{"cond": r["cond"], "dv": p} for r, p in parsed if p]
    r_pt, lo, hi, n_used, flag = S1["estimate"](recs, eff, seed, n_boot=boot or 1)
    if not boot: lo = hi = np.nan
    return dict(r=r_pt, lo=lo, hi=hi, n_ok=n_used, n_parsed=ok, n_ref=n_ref, den=den, mode=mode,
                stat=S1["SPECS"][eff]["kind"], claim_sign=S1["SPECS"][eff].get("claim_sign", 1), flag=flag)

def cell_g2(rows, eff, var, boot=0, seed=0):
    recs, n_ref, n_fail = [], 0, 0
    mode = "strict"
    parsed = []
    for r in rows:
        if r["text"] is None or not str(r["text"]).strip():
            n_ref += 1; continue
        p = parse_g2(eff, var, r["cond"], r["text"], False)
        parsed.append((r, p))
    den = len(parsed)
    ok = sum(1 for _, p in parsed if p is not None)
    if den and ok/den < 0.60:
        mode = "lenient"
        parsed = [(r, parse_g2(eff, var, r["cond"], r["text"], True) or p) for r, p in parsed]
        ok = sum(1 for _, p in parsed if p is not None)
    recs = [(r["cond"], p) for r, p in parsed if p is not None]
    n_fail = den - ok
    r_pt, blocks, fn = g2_point(eff, recs)
    lo = hi = np.nan
    if boot and np.isfinite(r_pt if r_pt is not None else np.nan):
        lo, hi = S2["boot_ci"](fn, blocks, n_boot=boot, seed=seed)
    n_used = sum(len(b) for b in blocks) if SPEC2[eff]["kind"] not in ("pearson", "one_sample", "peters") else len(blocks[0])
    return dict(r=r_pt, lo=lo, hi=hi, n_ok=n_used, n_ref=n_ref, n_fail=n_fail, den=den, mode=mode)

# ---- Degeneracy detection ----
def degen_main(eff, var):
    rs = [r for r in main_new if r["effect"] == eff and r["variant"] == var]; g = GROUP[eff]
    if g == 0:
        spec = G[0]["SPEC"][(eff, var)]
        d = G[0]["collect"](rs, spec, tier=2)
        if spec["stat"] == "between":
            a, b = np.array(d["pos"]), np.array(d["neg"])
            if len(a) < 2 or len(b) < 2: return "insufficient_n"
            if a.std() == 0 and b.std() == 0: return "zero_var"
        elif spec["stat"] == "paired_zero":
            dd = np.array(d["d"]); return "zero_var" if len(dd) > 1 and dd.std() == 0 else ""
        else:
            x, y = np.array(d["x"]), np.array(d["y"])
            if len(x) < 3: return "insufficient_n"
            if x.std() == 0 or y.std() == 0: return "no_variance"
        return ""
    if g == 1:
        o = cell_g1(rs, eff, var); return o["flag"] or ""
    # g2
    parsed = [(r, parse_g2(eff, var, r["cond"], r["text"], False)) for r in rs if r["text"]]
    den = len(parsed); ok = sum(1 for _, p in parsed if p)
    if den and ok/den < 0.60:
        parsed = [(r, parse_g2(eff, var, r["cond"], r["text"], True) or p) for r, p in parsed]
    recs = [(r["cond"], p) for r, p in parsed if p]
    sp = SPEC2[eff]; k = sp["kind"]
    if k == "two_group":
        a = [d[sp["dv"]] for c, d in recs if c == sp["hi"]]; b = [d[sp["dv"]] for c, d in recs if c == sp["lo"]]
        if len(a) < 2 or len(b) < 2: return "insufficient_n"
        return "zero_var" if S2["is_const"](a) and S2["is_const"](b) else ""
    if k == "pearson":
        x = [d[sp["x"]] for _, d in recs]; y = [d[sp["y"]] for _, d in recs]
        return "no_variance" if (S2["is_const"](x) or S2["is_const"](y)) else ""
    if k == "one_sample":
        v = [d[sp["dv"]] for c, d in recs if ("arm" not in sp or c == sp["arm"])]
        return "zero_var" if S2["is_const"](v) else ""
    if k == "peters":
        obj = [d["obj"] for _, d in recs]; sub = [d["subj"] for _, d in recs]
        if S2["is_const"](obj) and S2["is_const"](sub): return "rank_deficient"
        return "obj_const" if S2["is_const"](obj) else ""
    if k == "anova":
        gs = {L: [d[sp["dv"]] for c, d in recs if c == L] for L in sp["levels"]}
        return "zero_var" if all(S2["is_const"](v) for v in gs.values() if v) else ""
    if k == "inter2x2":
        keys = list(sp["plus"]) + list(sp["minus"])
        gs = {K: [d[sp["dv"]] for c, d in recs if c == K] for K in keys}
        return "zero_var" if all(S2["is_const"](v) for v in gs.values() if v) else ""
    return ""

# ---- Compute GPT main cells ----
NB = 2000
out = []
for (eff, var) in sorted({(r["effect"], r["variant"]) for r in main_new}):
    rs = [r for r in main_new if r["effect"] == eff and r["variant"] == var]
    g = GROUP[eff]
    if g == 0:
        o = cell_g0(rs, eff, var, boot=NB, seed=0)
        n_ok, den, nref = o["n_ok"], o["den"], o["n_ref"]
        r, lo, hi = o["r"], o["lo"], o["hi"]
        stat, absp = o["stat"], o["abs_primary"]
        n_parsed = n_ok
    elif g == 1:
        o = cell_g1(rs, eff, var, boot=NB, seed=0)
        n_ok, den, nref = o["n_ok"], o["den"], o["n_ref"]
        r, lo, hi = o["r"], o["lo"], o["hi"]
        stat, absp = o["stat"], False
        n_parsed = o["n_parsed"]
    else:
        o = cell_g2(rs, eff, var, boot=NB, seed=0)
        n_ok, den, nref = o["n_ok"], o["den"], o["n_ref"]
        r, lo, hi = o["r"], o["lo"], o["hi"]
        stat, absp = SPEC2[eff]["kind"], False
        n_parsed = den - o["n_fail"]
    out.append(dict(effect=eff, variant=var, r_sim=r, ci_lo=lo, ci_hi=hi,
                    n_ok=n_ok, parse_rate=round(n_parsed/den, 4) if den else None,
                    n_refusal=nref, group=g, stat=stat, n_parsed=n_parsed, n_denom=den,
                    parse_mode=o.get("mode"), abs_primary=absp))

cells = pd.DataFrame(out)
cells["r_sim"] = pd.to_numeric(cells.r_sim, errors="coerce")

cells["degen"] = [degen_main(e, v) for e, v in zip(cells.effect, cells.variant)]
cells["flag"] = [";".join([f for f in [("low_parse" if pr is not None and pr < 0.80 else ""), dg] if f])
                 for pr, dg in zip(cells.parse_rate, cells.degen)]

CELLS_COLS = ["effect", "variant", "r_sim", "ci_lo", "ci_hi", "n_ok", "parse_rate", "n_refusal",
              "group", "stat", "n_parsed", "n_denom", "parse_mode", "flag"]
cells_out = cells.copy()
for c in ("r_sim", "ci_lo", "ci_hi"):
    cells_out[c] = pd.to_numeric(cells_out[c], errors="coerce").round(6)
cells_out = cells_out[CELLS_COLS].sort_values(["effect", "variant"])
cells_out.to_csv(f"{PREFIX}_cells_new.csv", index=False)
print(f"{PREFIX}_cells_new.csv", cells_out.shape)

def g0_bare3(rows, eff, var):
    spec = G[0]["SPEC"][(eff, var)]
    def scan(tier):
        n_ref = 0; scored = []
        for r in rows:
            if r["text"] is None or not str(r["text"]).strip(): n_ref += 1; continue
            v, _ = G[0]["parse_response"](r["text"], spec, tier=tier)
            if v is None: continue
            scored.append((r["seed"], r["cond"], G[0]["score_response"](v, spec)))
        return scored, n_ref
    scored, n_ref = scan(2); mode = 2
    den = len(rows) - n_ref
    if den and len(scored) / den < 0.60:
        scored, n_ref = scan(3); mode = 3
    buck = collections.defaultdict(list)
    for sd, c, s in scored: buck[(sd, c)].append(s)
    st = spec["stat"]; d = dict(pos=[], neg=[], d=[], x=[], y=[])
    if st == "between":
        for sd in sorted({k[0] for k in buck}):
            a = buck.get((sd, spec["pos_cond"])); b = buck.get((sd, spec["neg_cond"]))
            if not a or not b: continue
            d["pos"].append(float(np.mean(a))); d["neg"].append(float(np.mean(b)))
        nunit = len(d["pos"])
    elif st == "paired_zero":
        for k in sorted(buck): d["d"].append(float(np.mean(buck[k])))
        nunit = len(d["d"])
    else:
        for k in sorted(buck):
            d["x"].append(float(np.mean([s[0] for s in buck[k]])))
            d["y"].append(float(np.mean([s[1] for s in buck[k]])))
        nunit = len(d["x"])
    return dict(r=G[0]["effect_size"](spec, d), n_parsed=len(scored), n_units=nunit,
                n_ref=n_ref, den=den, stat=st, mode=mode, data=d,
                abs_primary=spec.get("abs_primary", False))

# --------------------------------------------------------------------------
# g1 bare
# --------------------------------------------------------------------------

def g1_bare(rows, eff, boot=0, seed=0):
    n_ref = 0; recs = []
    for r in rows:
        if r["text"] is None or not str(r["text"]).strip(): n_ref += 1; continue
        p = parse_g1(eff, "A", r["text"], False) or parse_g1(eff, "A", r["text"], True)
        if p is None: continue
        recs.append({"seed": r["seed"], "cond": r["cond"], "dv": p})
    den = len(rows) - n_ref
    r, lo, hi, nu, flag = S1["estimate_bare"](recs, eff, seed, n_boot=boot or 1)
    if not boot: lo = hi = np.nan
    ri, _, _, _, _ = S1["estimate"]([{"cond": x["cond"], "dv": x["dv"]} for x in recs], eff, seed, n_boot=1)
    return dict(r=r, r_indiv=ri, lo=lo, hi=hi, n_parsed=len(recs), n_units=nu, n_ref=n_ref, den=den,
                stat=S1["SPECS"][eff]["kind"])

# --------------------------------------------------------------------------
# g2 bare
# --------------------------------------------------------------------------

def g2_bare(rows, eff):
    n_ref = 0; parsed = []
    for r in rows:
        if r["text"] is None or not str(r["text"]).strip(): n_ref += 1; continue
        parsed.append((r, parse_g2(eff, "A", r["cond"], r["text"], False)))
    den = len(parsed); ok = sum(1 for _, p in parsed if p)
    if den and ok / den < 0.60:
        parsed = [(r, parse_g2(eff, "A", r["cond"], r["text"], True) or p) for r, p in parsed]
        ok = sum(1 for _, p in parsed if p)
    good = [(r, p) for r, p in parsed if p]
    recs_i = [(r["cond"], p) for r, p in good]
    r_i, bl_i, fn_i = g2_point(eff, recs_i)
    buck = collections.defaultdict(list)
    for r, p in good: buck[(r["seed"], r["cond"])].append(p)
    keys = set().union(*[set(p) for _, p in good]) if good else set()
    recs_s = []
    for (sd, c), ps in sorted(buck.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        recs_s.append((c, {k: float(np.mean([p[k] for p in ps])) for k in keys}))
    r_s = g2_point(eff, recs_s)[0] if recs_s else np.nan
    return dict(r_indiv=r_i, r_seed=r_s, n_parsed=ok, n_ref=n_ref, den=den,
                n_units_i=sum(len(b) for b in bl_i) if isinstance(bl_i[0][0], (int, float)) else len(bl_i[0]),
                n_seed_units=len(recs_s), blocks=bl_i, fn=fn_i)

# --------------------------------------------------------------------------
# few_levels helper
# --------------------------------------------------------------------------

def few_levels(eff):
    g = GROUP[eff]; rs = bare_by_eff[eff]
    if g == 0:
        spec = G[0]["SPEC"][(eff, "A")]
        if spec["stat"] != "pearson": return False
        o = g0_bare3(rs, eff, "A"); d = o["data"]
        return len(set(zip(d["x"], d["y"]))) <= 2
    if g == 2 and SPEC2.get(eff, {}).get("kind") == "pearson":
        o = g2_bare(rs, eff); sp = SPEC2[eff]
        parsed = [parse_g2(eff, "A", r["cond"], r["text"], False) for r in rs]
        good = [p for p in parsed if p]
        return len({(p[sp["x"]], p[sp["y"]]) for p in good}) <= 2
    return False

# --------------------------------------------------------------------------
# Build bare_df
# --------------------------------------------------------------------------

brec = []
for eff in sorted(bare_by_eff):
    rs = bare_by_eff[eff]; g = GROUP[eff]
    if g == 0:
        o = g0_bare3(rs, eff, "A")
        r = o["r"]; n_units = o["n_units"]; npar = o["n_parsed"]; nref = o["n_ref"]; den = o["den"]; stat = o["stat"]
        dd = o["data"]
        if stat == "between":
            dg = "insufficient_n" if len(dd["pos"]) < 2 or len(dd["neg"]) < 2 else ("zero_var" if np.std(dd["pos"]) == 0 and np.std(dd["neg"]) == 0 else "")
        elif stat == "paired_zero":
            dg = "zero_var" if len(dd["d"]) > 1 and np.std(dd["d"]) == 0 else ""
        else:
            dg = "no_variance" if (len(dd["x"]) < 3 or np.std(dd["x"]) == 0 or np.std(dd["y"]) == 0) else ""
        n_ok = n_units
    elif g == 1:
        o = g1_bare(rs, eff); r = o["r"]; n_ok = o["n_units"]; npar = o["n_parsed"]; nref = o["n_ref"]; den = o["den"]
        stat = o["stat"]; dg = "" if (r is not None and np.isfinite(r)) else "no_variance"
    else:
        o = g2_bare(rs, eff); r = o["r_indiv"]; n_ok = o["n_parsed"]; npar = o["n_parsed"]; nref = o["n_ref"]; den = o["den"]
        stat = SPEC2[eff]["kind"]; dg = "" if (r is not None and np.isfinite(r)) else "no_variance"
    pr = round(npar / den, 4) if den else None
    brec.append(dict(effect=eff, r_bare=r, n_ok=n_ok, group=g, stat=stat,
                     n_parsed=npar, n_denom=den, parse_rate=pr, n_refusal=nref,
                     flag=";".join([f for f in [("low_parse" if pr is not None and pr < 0.80 else ""), dg] if f])))

bare_df = pd.DataFrame(brec)
bare_df["r_bare"] = pd.to_numeric(bare_df.r_bare, errors="coerce")

bare_df["flag"] = [";".join([f for f in [fl, ("few_levels" if few_levels(e) else "")] if f])
                   for e, fl in zip(bare_df.effect, bare_df.flag)]

BARE_COLS = ["effect", "r_bare", "n_ok", "group", "stat", "n_parsed", "n_denom", "parse_rate", "n_refusal", "flag"]
bare_out = bare_df.copy()
bare_out["r_bare"] = bare_out.r_bare.round(6)
bare_out = bare_out[BARE_COLS].sort_values("effect")
bare_out.to_csv(f"{PREFIX}_bare_new.csv", index=False)

print(f"{PREFIX}_bare_new.csv", bare_out.shape)

# ==========================================================================
# old-19 effects (pilot waves): xm_parsers_patch_old is self-contained
# ==========================================================================
import xm_parsers_patch_old as XPO
for k, v in XPO.bind_keys(bank).items():
    pass  # keys derived from the protocol bank itself
main_old = [r for r in main_rows if r["effect"] in set(XPO.EFFECTS_OLD19)]
bare_old = [r for r in bare_rows if r["effect"] in set(XPO.EFFECTS_OLD19)]
rec_o = []
for effect in XPO.EFFECTS_OLD19:
    for variant in ("A", "B"):
        rs = [r for r in main_old if r["effect"] == effect and r["variant"] == variant]
        if rs:
            rec_o.append(XPO.score_cell(rs, effect, variant, n_boot=2000))
cells_old = pd.DataFrame(rec_o)
cells_old.to_csv(f"{PREFIX}_cells_old.csv", index=False)
print(f"{PREFIX}_cells_old.csv", cells_old.shape)
rec_b = []
for effect in XPO.EFFECTS_OLD19:
    rs = [r for r in bare_old if r["effect"] == effect]
    if rs:
        rec_b.append(XPO.score_bare(rs, effect, n_boot=2000))
bare_old_df = pd.DataFrame(rec_b)
bare_old_df.to_csv(f"{PREFIX}_bare_old.csv", index=False)
print(f"{PREFIX}_bare_old.csv", bare_old_df.shape)

# ==========================================================================
# analysis units: effect-level aggregation + human anchors
# ==========================================================================
zc = lambda r: np.arctanh(np.clip(np.asarray(r, float), -0.99, 0.99))
cn = cells_out.dropna(subset=["r_sim"])
eff_r = cn.groupby("effect").r_sim.median()
co = cells_old.dropna(subset=["r_sim"])
eff_r = pd.concat([eff_r, co.groupby("effect").r_sim.median()])
bare_map = {}
for df_, col in [(bare_out, "r_bare"), (bare_old_df, "r_bare")]:
    for _, rw in df_.dropna(subset=[col]).iterrows():
        bare_map[rw.effect] = rw[col]
anchors = pd.read_csv(args.units).drop_duplicates("effect").set_index("effect")[
    ["z_o", "z_r", "var_o", "var_r", "n_o", "n_r"]]
units = []
for eff, rsim in eff_r.items():
    if eff not in anchors.index:
        continue
    a = anchors.loc[eff]
    units.append(dict(effect=eff, r_sim=rsim, z_sim=float(zc(rsim)),
                      r_bare=bare_map.get(eff, np.nan),
                      z_bare=float(zc(bare_map[eff])) if eff in bare_map else np.nan,
                      z_o=a.z_o, z_r=a.z_r, var_o=a.var_o, var_r=a.var_r))
units = pd.DataFrame(units)
units.to_csv(f"{PREFIX}_units.csv", index=False)
print(f"{PREFIX}_units.csv", units.shape)

# ==========================================================================
# optional: three-coefficient measurement-error model on this family
# ==========================================================================
if args.fit:
    import pymc as pm, arviz as az
    conf = units.dropna(subset=["z_sim", "z_bare", "z_o", "z_r"]).reset_index(drop=True)
    print(f"fitting on {len(conf)} effects")
    with pm.Model() as m:
        z_o_true = pm.Normal("z_o_true", mu=conf.z_o.values, sigma=np.sqrt(conf.var_o.values), shape=len(conf))
        z_r_true = pm.Normal("z_r_true", mu=conf.z_r.values, sigma=np.sqrt(conf.var_r.values), shape=len(conf))
        D_pub = z_o_true - z_r_true
        b0 = pm.Normal("b0", 0, 1); b_bare = pm.Normal("b_bare", 0, 1)
        b_rep = pm.Normal("b_rep", 0, 1); b_pub = pm.Normal("b_pub", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        mu = b0 + b_bare * conf.z_bare.values + b_rep * z_r_true + b_pub * D_pub
        pm.Normal("y", mu, sigma, observed=conf.z_sim.values)
        idata = pm.sample(1500, tune=1000, chains=4, cores=1, target_accept=0.95, random_seed=11)
    print(az.summary(idata, var_names=["b_bare", "b_rep", "b_pub"], hdi_prob=0.95).round(3).to_string())
    idata.to_netcdf(f"{PREFIX}_model_idata.nc")
