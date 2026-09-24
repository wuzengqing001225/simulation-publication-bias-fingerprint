"""Per-effect response parsers and cell-level effect-size (r) estimators for A0.

Each parser: text -> individual score dict or None (unparseable).
Each estimator: list of {cond, score...} rows -> r estimate on the FReD scale
(sign positive = direction of the original claim).
"""
import re
import numpy as np

LINE_AB = re.compile(r'(\d+)\s*[:.)-]\s*([AB])\b', re.I)

def _ab_lines(text, n_items):
    pairs = dict()
    for m in LINE_AB.finditer(text or ""):
        k = int(m.group(1))
        if 1 <= k <= n_items and k not in pairs:
            pairs[k] = m.group(2).upper()
    return pairs if len(pairs) == n_items else None

def _num_after(label, text, lo, hi, integer=False):
    m = re.search(label + r'\s*[:=]?\s*(-?\d+(?:\.\d+)?)', text or "", re.I)
    if not m: return None
    v = float(m.group(1))
    if not (lo <= v <= hi): return None
    return int(v) if integer else v

# ---------------- parsers ----------------

def parse_asch(t):
    p = _ab_lines(t, 18)
    return {"generous": 1 if p[1]=="A" else 0} if p else None

def parse_banerjee(t):
    v = _num_after(r'BRIGHTNESS', t, 1, 7, True)
    return {"bright": v} if v is not None else None

def parse_bargh(t):
    # 17 numeric lines: S1.1..S1.7 then S2.1..S2.10
    nums = re.findall(r'S([12])\.(\d+)\s*[:=]\s*(-?\d+(?:\.\d+)?)', t or "")
    d = {(int(a), int(b)): float(c) for a,b,c in nums}
    if len([k for k in d if k[0]==1]) < 7 or len([k for k in d if k[0]==2]) < 10:
        # fallback: bare numbers, 17 lines
        vals = re.findall(r'^\s*(?:\d+\s*[:.)-])?\s*(-?\d+(?:\.\d+)?)\s*$', t or "", re.M)
        if len(vals) >= 17:
            vals = [float(v) for v in vals[:17]]
            d = {(1,i+1): vals[i] for i in range(7)}
            d.update({(2,i+1): vals[7+i] for i in range(10)})
        else:
            return None
    warmth_items = [d.get((1,5)), d.get((1,6)), d.get((1,7))]
    lonely_items = [d.get((2,i)) for i in range(1,11)]
    if any(v is None for v in warmth_items+lonely_items): return None
    return {"warm_raw": warmth_items, "lonely": float(np.mean(lonely_items))}

def parse_cooney(t):
    # happiness / satisfaction rating 1-9 or similar; look for labeled number
    for lab in [r'HAPPINESS', r'FEELING', r'RATING', r'Q1']:
        v = _num_after(lab, t, 1, 9, False)
        if v is not None: return {"score": v}
    m = re.search(r'\b([1-9])\b', t or "")
    return {"score": float(m.group(1))} if m else None

def parse_klink(t):
    p = _ab_lines(t, 12)
    return {"choices": p} if p else None

def parse_slepian(t):
    s = _num_after(r'SLANT', t, 0, 90); dct = _num_after(r'DISTANCE', t, 0, 500)
    if s is None or dct is None: return None
    return {"slant": s, "dist": dct}

def parse_tamir(t):
    p = _ab_lines(t, 10)
    return {"choices": p} if p else None

def parse_valdesolo(t):
    v = _num_after(r'FAIRNESS', t, 1, 7, True)
    return {"fair": v} if v is not None else None

def parse_vanboven(t):
    q1 = _num_after(r'Q1', t, 1, 9); q2 = _num_after(r'Q2', t, 1, 9)
    if q1 is None or q2 is None: return None
    return {"q1": q1, "q2": q2}

def parse_wakslak(t):
    # BIF 25 items, A/B choice; abstract choice count.
    p = _ab_lines(t, 25)
    if not p: return None
    return {"choices": p}

PARSERS = {
    "Asch1946_warm_cold": parse_asch,
    "Banerjee2012_brightness": parse_banerjee,
    "Bargh2012_warmth": parse_bargh,
    "Cooney2016_fairness": parse_cooney,
    "Klink2000_soundsymbolism": parse_klink,
    "Slepian2012_secrets": parse_slepian,
    "Tamir2012_selfdisclosure": parse_tamir,
    "Valdesolo2007_hypocrisy": parse_valdesolo,
    "VanBoven2007_anticipation": parse_vanboven,
    "Wakslak2006_construal": parse_wakslak,
}
