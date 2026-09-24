"""Narrative tagging round for GPT (run locally with your OpenAI API key).

Usage:  OPENAI_API_KEY=sk-...  python gpt_narrative_tagging.py
Input:  gpt_tagging_input.csv (same folder)
Output: narrative_gpt_round.csv
Model:  gpt-5.2 by default; change MODEL below if you prefer another.
"""
import csv, json, os, time
from openai import OpenAI

MODEL = "gpt-5.2"
client = OpenAI()

SYSTEM = (
    "You judge what the DOMINANT narrative about a psychology finding is in today's broad "
    "scientific and popular corpus (textbooks, popular science, applied literature). "
    "Categories: original_dominates (the original finding is still told as true); "
    "failure_dominates (the replication failure is now the better-known story); "
    "contested (both narratives circulate); obscure (the effect is rarely discussed). "
    'Reply ONLY with JSON: {"narrative": "<category>", "reason": "<one sentence>"}'
)

rows = list(csv.DictReader(open("gpt_tagging_retry.csv", encoding="utf-8")))
out = []
for i, r in enumerate(rows):
    prompt = (f"Finding: {r['description'][:300]}\n"
              f"Original study: {r['ref_o'][:150]}\n"
              f"What is the dominant narrative today?")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_completion_tokens=2000,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}])
            txt = resp.choices[0].message.content.strip().strip("`")
            if txt.startswith("json"): txt = txt[4:]
            j = json.loads(txt)
            out.append({"effect_id": r["effect_id"], "doi_o": r["doi_o"],
                        "narr_gpt": j["narrative"], "gpt_reason": j["reason"]})
            break
        except Exception as e:
            if attempt == 2:
                out.append({"effect_id": r["effect_id"], "doi_o": r["doi_o"],
                            "narr_gpt": "ERROR", "gpt_reason": str(e)[:100]})
            time.sleep(2)
    if (i + 1) % 20 == 0: print(f"{i+1}/{len(rows)}")

with open("narrative_gpt_retry.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["effect_id", "doi_o", "narr_gpt", "gpt_reason"])
    w.writeheader(); w.writerows(out)
print("done:", len(out), "rows -> narrative_gpt_round.csv")
