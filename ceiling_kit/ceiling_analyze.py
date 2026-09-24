"""Score a ceiling-fix run: numeric 0-100 responses -> per-cell effect sizes.

Usage (from the folder containing ceiling_bank.json):

    python ceiling_analyze.py --main <out>/ceil_main_resp.json \
        --bare <out>/ceil_bare_resp.json --prefix CEIL_<model>

Outputs <prefix>_cells.csv (effect x variant: r_sim, ceiling share, n) and
<prefix>_bare.csv (effect: r_bare, n). One generic parser: every response
ends with `RATING: <int>` or `ITEM <k>: <int>` lines (0-100).

Scoring follows each protocol's `scoring` spec:
  two_group_conditions  r from mean difference between group_hi and group_lo
                        condition arms (point-biserial via t)
  paired_items          r from within-response difference of group_hi vs
                        group_lo item ratings (paired t -> r)
  pearson_items         r = Pearson correlation between item ratings and the
                        items' anchor values, averaged over responses
                        (Fisher-z mean)
"""

import argparse, json, re
import numpy as np
import pandas as pd

RATING = re.compile(r"RATING\s*[:=]\s*(\d{1,3})", re.I)
ITEM = re.compile(r"ITEM\s*(\d+)\s*[:=]\s*(\d{1,3})", re.I)


def parse(text):
    if not text:
        return None
    items = {int(k): int(v) for k, v in ITEM.findall(text) if 0 <= int(v) <= 100}
    if items:
        return items
    m = RATING.findall(text)
    if m and 0 <= int(m[-1]) <= 100:
        return {1: int(m[-1])}
    return None


def two_group_r(hi, lo):
    hi, lo = np.asarray(hi, float), np.asarray(lo, float)
    if len(hi) < 3 or len(lo) < 3:
        return np.nan, len(hi) + len(lo)
    n1, n2 = len(hi), len(lo)
    sp = np.sqrt(((n1 - 1) * hi.var(ddof=1) + (n2 - 1) * lo.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return (0.0 if hi.mean() == lo.mean() else np.sign(hi.mean() - lo.mean()) * 0.99), n1 + n2
    t = (hi.mean() - lo.mean()) / (sp * np.sqrt(1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    return float(t / np.sqrt(t ** 2 + df)), n1 + n2


def paired_r(diffs):
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return np.nan, len(d)
    if d.std(ddof=1) == 0:
        return (0.0 if d.mean() == 0 else np.sign(d.mean()) * 0.99), len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(t / np.sqrt(t ** 2 + len(d) - 1)), len(d)


def score_protocol(proto, rows):
    sc = proto["scoring"]
    parsed = [(r, parse(r["text"])) for r in rows]
    ok = [(r, p) for r, p in parsed if p]
    n_total, n_ok = len(rows), len(ok)
    if sc["type"] == "two_group_conditions":
        gh = sc["group_hi"] if isinstance(sc["group_hi"], list) else [sc["group_hi"]]
        gl = sc["group_lo"] if isinstance(sc["group_lo"], list) else [sc["group_lo"]]
        hi = [np.mean(list(p.values())) for r, p in ok if r["cond"] in gh]
        lo = [np.mean(list(p.values())) for r, p in ok if r["cond"] in gl]
        r_, n_ = two_group_r(hi, lo)
    elif sc["type"] == "paired_items":
        hi_ids = set(sc["group_hi"]) if isinstance(sc["group_hi"], list) else {sc["group_hi"]}
        lo_ids = set(sc["group_lo"]) if isinstance(sc["group_lo"], list) else {sc["group_lo"]}
        diffs = []
        for r, p in ok:
            h = [v for k, v in p.items() if k in hi_ids]
            l = [v for k, v in p.items() if k in lo_ids]
            if h and l:
                diffs.append(np.mean(h) - np.mean(l))
        r_, n_ = paired_r(diffs)
    elif sc["type"] == "pearson_items":
        anchors = {it["id"]: it["anchor"] for it in proto["items"] if it.get("anchor") is not None}
        zs = []
        for r, p in ok:
            common = sorted(set(p) & set(anchors))
            if len(common) >= 4:
                x = np.array([p[k] for k in common], float)
                y = np.array([anchors[k] for k in common], float)
                if x.std() > 0 and y.std() > 0:
                    zs.append(np.arctanh(np.clip(np.corrcoef(x, y)[0, 1], -0.99, 0.99)))
        r_, n_ = (float(np.tanh(np.mean(zs))), len(zs)) if len(zs) >= 3 else (np.nan, len(zs))
    else:
        raise ValueError(sc["type"])
    return r_, n_, n_ok, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--bare", required=True)
    ap.add_argument("--prefix", default="CEIL")
    args = ap.parse_args()

    bank = json.load(open("ceiling_bank.json"))["protocols"]
    by_ev = {(p["effect"], p["variant"]): p for p in bank}
    main_rows = json.load(open(args.main))
    bare_rows = json.load(open(args.bare))

    cells = []
    for (eff, var), proto in sorted(by_ev.items()):
        rows = [r for r in main_rows if r["effect"] == eff and r["variant"] == var]
        if not rows:
            continue
        r_, n_, n_ok, n_total = score_protocol(proto, rows)
        cells.append({"effect": eff, "variant": var, "model": rows[0]["model"],
                      "r_sim": r_, "n_scored": n_, "parse_rate": n_ok / max(n_total, 1),
                      "at_ceiling": (abs(r_) > 0.95) if r_ == r_ else None})
    cells = pd.DataFrame(cells)
    cells.to_csv(f"{args.prefix}_cells.csv", index=False)
    print(cells.to_string(index=False))

    bare = []
    for (eff, var), proto in sorted(by_ev.items()):
        if var != "A":
            continue
        rows = [r for r in bare_rows if r["effect"] == eff]
        if not rows:
            continue
        r_, n_, n_ok, n_total = score_protocol(proto, rows)
        bare.append({"effect": eff, "model": rows[0]["model"], "r_bare": r_,
                     "n_scored": n_, "parse_rate": n_ok / max(n_total, 1)})
    bare = pd.DataFrame(bare)
    bare.to_csv(f"{args.prefix}_bare.csv", index=False)
    print(bare.to_string(index=False))
    print(f"\nsaved {args.prefix}_cells.csv, {args.prefix}_bare.csv")


if __name__ == "__main__":
    main()
