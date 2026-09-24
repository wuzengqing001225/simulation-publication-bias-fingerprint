"""A1 per-effect parsers + cell-level r estimators (Fisher-z verdict handled downstream)."""
import re
import numpy as np
from a0_runner import two_group_r, pearson_r, proportion_diff_r, bootstrap_ci

def _lines_int(text, n, lo, hi):
    vals = {}
    for m in re.finditer(r'(\d+)\s*[:.)-]\s*(-?\d+)', text or ""):
        k, v = int(m.group(1)), int(m.group(2))
        if 1 <= k <= n and lo <= v <= hi and k not in vals: vals[k] = v
    return vals if len(vals) == n else None

def parse_husnu(t):
    v = _lines_int(t, 4, 1, 9)
    return {"intent": float(np.mean(list(v.values())))} if v else None

def parse_tversky(t):
    # 5 lines 'N: P, R' P in {1,3}, R = ratio estimate
    rows = re.findall(r'(\d+)\s*[:.)-]\s*([13])\s*[, ]\s*(\d+(?:\.\d+)?)', t or "")
    if len({int(a) for a,_,_ in rows}) < 5: return None
    first = sum(1 for _,p,_ in rows if p=="1")
    return {"n_first": first, "n_items": 5}

def parse_eyal(t):
    v = _lines_int(t, 3, -5, 5)
    return {"offense": float(np.mean(list(v.values())))} if v else None

def parse_rottenstreich(t):
    m = re.search(r'CHOICE\s*[:=]\s*([AB])', t or "", re.I)
    if not m: return None
    return {"affective": 1 if m.group(1).upper()=="A" else 0}

def parse_oppenheimer(t):
    m = re.search(r'ROLLS\s*[:=]\s*(\d+)', t or "", re.I)
    if not m: return None
    v = float(m.group(1))
    if v > 10000: return None
    return {"rolls": v}

def parse_knobe(t):
    m = re.search(r'1\s*[:.)-]\s*([AB])', t or "")
    if not m: return None
    return {"intentional": 1 if m.group(1).upper()=="A" else 0}

def parse_miyamoto(t):
    v = _lines_int(t, 2, 1, 9)
    return {"attitude": v[1], "confidence": v[2]} if v else None

def parse_genschow(t):
    # 27 labeled lines S1.a.. ; average FW subscale & attribution subscale defined in measure
    nums = re.findall(r'S(\d+)\.([a-z])\s*[:=]\s*(-?\d+)', t or "")
    if len(nums) < 20: return None
    d = {(int(a), b): int(c) for a,b,c in nums}
    return {"raw": {f"{k[0]}{k[1]}": v for k,v in d.items()}}

def parse_chao(t):
    m1 = re.search(r'1\s*[:.)-]\s*([AB])', t or "")
    m2 = re.search(r'2\s*[:.)-]\s*(\d+)', t or "")
    if not (m1 and m2): return None
    return {"accept": 1 if m1.group(1).upper()=="A" else 0, "effort": int(m2.group(1))}

PARSERS1 = {
    "Husnu2010_imagined_contact": parse_husnu,
    "Tversky1973_availability_letters": parse_tversky,
    "Eyal2008_moral_distance": parse_eyal,
    "Rottenstreich2001_affective_lottery": parse_rottenstreich,
    "Oppenheimer2009_retrospective_gambler": parse_oppenheimer,
    "Knobe2003_side_effect": parse_knobe,
    "Miyamoto2002_correspondence_bias": parse_miyamoto,
    "Genschow2017_free_will_attribution": parse_genschow,
    "Chao2017_thankyou_gift": parse_chao,
}
