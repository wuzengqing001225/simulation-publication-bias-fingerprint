"""B2 runner: personas, prompt assembly, response parsing, construct scoring.

Model configs are identical to A0 (a0_runner.MODELS):
  sonnet5  -> claude-sonnet-5, NO temperature parameter passed
  sonnet45 -> claude-sonnet-4-5-20250929, temperature=1.0

System framing is the verbatim A0 string.

Persona note (deviation from A0, declared in the memo): A0's make_persona built a
WEIRD-undergrad frame (age 18-24, major, year) because A0 replicated
student-sample lab effects. B2's human comparison target is Twin-2K-500, a
US-representative adult panel, and the task specifies the fields
age / gender / occupation / region / one-sentence self-description. So the same
structural logic (fixed-seed random.Random, rng.choice over fixed lists, one
templated sentence, identical closing instruction) is applied to a general US
adult frame. An undergraduate-only frame would truncate variance on political
liberalism, religiosity and purchase intent and make the panel comparison
uninterpretable.
"""
import random
import re

SYS = ("You are powering a social-science simulation that generates synthetic survey-participant "
       "responses for metascience research on replication. Respond as the described participant "
       "would, in their voice, following the requested output format exactly.")

MODELS = {
    "sonnet5": {"model": "claude-sonnet-5"},                        # no temperature
    "sonnet45": {"model": "claude-sonnet-4-5-20250929", "temperature": 1.0},
}

GENDERS = ["woman", "man", "woman", "man", "non-binary person"]

OCCUPATIONS = [
    "registered nurse", "high-school teacher", "retail store manager", "truck driver",
    "software developer", "restaurant server", "accountant", "electrician",
    "administrative assistant", "warehouse worker", "small-business owner", "social worker",
    "police officer", "farmer", "graphic designer", "home health aide",
    "college student working part time", "retired factory worker", "insurance agent",
    "construction supervisor", "pharmacist", "call-centre representative",
    "self-employed contractor", "stay-at-home parent",
]

REGIONS = [
    "a suburb outside Atlanta, Georgia", "rural central Ohio", "Phoenix, Arizona",
    "a small town in eastern Kentucky", "Seattle, Washington", "the Bronx, New York",
    "suburban Dallas, Texas", "Des Moines, Iowa", "a coastal town in Maine",
    "Los Angeles, California", "Milwaukee, Wisconsin", "rural western Montana",
    "Charlotte, North Carolina", "a suburb of Philadelphia, Pennsylvania",
    "Denver, Colorado", "New Orleans, Louisiana",
]

SELF_DESCR = [
    "I keep to myself mostly and spend my free time with family.",
    "I follow the news closely and have strong opinions about where the country is headed.",
    "I am busy with work and do not think much about politics.",
    "I go to church most weeks and it shapes how I see things.",
    "I like trying new things and travelling when I can afford it.",
    "Money is tight and that is what I think about most.",
    "I read a lot and like arguing about ideas.",
    "I am practical and prefer to stick with what I know works.",
    "I volunteer in my community and care about local issues.",
    "I am pretty easy-going and try not to worry about much.",
    "I am ambitious and focused on getting ahead in my career.",
    "I am cautious about anything the government or big companies tell me.",
]


def make_persona(rng):
    """A0-style persona, general US adult frame. Returns (text, fields)."""
    age = rng.randint(18, 80)
    gender = rng.choice(GENDERS)
    occ = rng.choice(OCCUPATIONS)
    region = rng.choice(REGIONS)
    descr = rng.choice(SELF_DESCR)
    art = "an" if occ[0] in "aeiou" else "a"
    text = (f"You are a {age}-year-old {gender} working as {art} {occ}, living in {region}. "
            f"In your own words: \"{descr}\" "
            f"You have agreed to take part in a national survey. "
            f"Answer as this person would, naturally and honestly.")
    return text, dict(age=age, gender=gender, occupation=occ, region=region,
                      self_description=descr)


def make_personas(n=120, seed=20260827):
    rng = random.Random(seed)
    return [dict(subj=i, persona=(p := make_persona(rng))[0], **p[1]) for i in range(n)]


HEADER = (
    "Below is a survey with {n} numbered questions. Different questions use different "
    "response scales; each question states its own scale.\n\n"
    "Answer EVERY question. Do not skip any. Do not explain, justify or comment.\n\n"
    "Output format — exactly {n} lines, nothing before or after them:\n"
    "1: <answer>\n2: <answer>\n...\n{n}: <answer>\n\n"
    "Each <answer> is a bare number or a single letter, as the question specifies.\n\n"
    "=== SURVEY ===\n"
)


def render_prompt(persona_text, bank):
    lines = [persona_text, "", HEADER.format(n=len(bank))]
    for b in bank:
        a = f"  [{b['anchors']}]" if b["anchors"] else ""
        lines.append(f"{b['iid']}. {b['text']}{a}")
    lines.append("\n=== END SURVEY ===\nGive your {n} answers now, one per line, in the format "
                 "shown above.".format(n=len(bank)))
    return "\n".join(lines)


def build_requests(personas, bank, models=("sonnet5", "sonnet45"), max_tokens=2500):
    reqs, index = [], []
    for mk in models:
        for p in personas:
            reqs.append({"prompt": render_prompt(p["persona"], bank), "system": SYS,
                         "max_tokens": max_tokens, **MODELS[mk]})
            index.append({"model": mk, "subj": p["subj"]})
    return reqs, index


LINE_RE = re.compile(r'^\s*(\d{1,3})\s*[:.\)]\s*(.+?)\s*$')
NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def parse_response(text, bank):
    """-> {iid: raw_string}."""
    if not text:
        return {}
    valid = {b["iid"] for b in bank}
    out = {}
    for ln in str(text).splitlines():
        m = LINE_RE.match(ln)
        if not m:
            continue
        iid = int(m.group(1))
        if iid in valid and iid not in out:
            out[iid] = m.group(2).strip()
    return out


def score_item(raw, b):
    """-> numeric item score (already reverse-scored / correctness-scored) or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    f = b["fmt"]
    if f == "likert7":
        m = NUM_RE.search(s)
        if not m:
            return None
        v = float(m.group())
        if not (1 <= v <= 7):
            return None
        return 8 - v if b["reverse"] else v
    if f == "pct":
        m = NUM_RE.search(s)
        if not m:
            return None
        v = float(m.group())
        if not (0 <= v <= 100):
            return None
        return 100 - v if b["reverse"] else v
    if f == "num":
        m = NUM_RE.search(s.replace(",", ""))
        if not m:
            # allow spelled-out zero / none
            if re.search(r'\b(none|zero|no dirt)\b', s, re.I):
                return 1.0 if abs(b["key"]) < 1e-9 else 0.0
            return None
        return 1.0 if abs(float(m.group()) - b["key"]) <= b["tol"] else 0.0
    if f in ("mc", "choice"):
        m = re.match(r'^[\(\[]?([A-Za-z])\b', s)
        if not m:
            return None
        return 1.0 if m.group(1).upper() == b["key"] else 0.0
    raise ValueError(f)


def score_subject(raw_map, bank, min_frac=0.5):
    """-> {construct: score}, item-mean with reverse/correctness applied."""
    from collections import defaultdict
    acc = defaultdict(list)
    tot = defaultdict(int)
    for b in bank:
        tot[b["construct"]] += 1
        v = score_item(raw_map.get(b["iid"]), b)
        if v is not None:
            acc[b["construct"]].append(v)
    out = {}
    for c, k in tot.items():
        vs = acc.get(c, [])
        out[c] = sum(vs) / len(vs) if len(vs) >= max(1, int(round(min_frac * k))) else None
    return out
