"""R15 - trace every number in the revised manuscript and the response letter to an output file.

Collects all numeric values from reports/revision/*.json|csv, the frozen reports/ and
models/final/ files, and the Approach 1 reports, then checks each number printed in the
documents against that pool at the printed precision. Numbers that are not results
(years, section numbers, reference details, grid values) are listed separately for reading.
    python reports/revision/r15_number_trace.py <doc.docx> [<doc.docx> ...]
"""
import glob, json, re, sys
from pathlib import Path
import pandas as pd
from docx import Document

pool = set()
def add(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        if isinstance(v, str):
            for m in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", v): add(m)
        return
    if x == x: pool.add(abs(x))
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items(): add(k) if re.fullmatch(r"-?\d+\.?\d*", str(k)) else None; walk(v)
    elif isinstance(o, (list, tuple)):
        for v in o: walk(v)
    else: add(o)
files = (glob.glob("reports/revision/*.json") + glob.glob("reports/*.json") + glob.glob("models/final/*.json"))
for f in files:
    if "xgboost_final" in f: continue
    walk(json.load(open(f)))
for f in glob.glob("reports/revision/*.csv") + glob.glob("reports/*.csv"):
    if "predictions" in f: continue
    d = pd.read_csv(f)
    for c in d.columns: [add(v) for v in d[c].tolist()]
for f in glob.glob("../approach1/reports/*.md") + glob.glob("../approach1/processed/*.json") + glob.glob("../approach1/artifacts/*.json"):
    add(Path(f).read_text(encoding="utf8", errors="ignore"))
# derived values that the text states as differences / percentages of pooled values
b = pd.read_csv("reports/revision/r03_benchmarks.csv").set_index("model")
pool |= {abs(b.loc["xgboost", "mae"] - b.loc["lasso", "mae"])}
pe = json.load(open("reports/revision/r11_pooling_evidence.json")); pool.add(100 * (pe["where_pooled_differs"]["ratio_of_means"] - 1))
import os
if os.path.exists("reports/revision/r16_overlap_with_24434.csv"):
    ov = pd.read_csv("reports/revision/r16_overlap_with_24434.csv"); pool |= {abs(100 * (v - 1)) for v in ov.ratio_of_means.dropna()}
    inv = pd.read_csv("reports/revision/r16_sensor_inventory.csv"); pool.add(float((inv.parameter == "pm25").sum())); pool.add(float(((inv.parameter == "pm25") & (inv.isMonitor == False) & (inv.pm25_hours > 0)).sum()))
ex = json.load(open("reports/revision/r07_extremes.json")); pool.add(100 * ex["share_of_sq_error_reduction_from_obs_gt_150"])
q = json.load(open("reports/revision/r02_qc985_sensitivity.json")); pool.add(q["original"]["xgboost"]["rmse"] - q["masked"]["xgboost"]["rmse"])
ds = json.load(open("reports/revision/r01_dataset.json")); pool |= {ds["pairs_both_valid"] - ds["pairs_complete_45"]}
r6 = pd.read_csv("reports/revision/r06_per_fold.csv"); mrows = json.load(open("reports/revision/r14_extra_checks.json"))["model_rows_by_month"]
pool.add(sum(mrows[k] for k in ("10", "11", "12")))
for v in list(pool): pool.add(v * 100)          # shares printed as percentages

def ok(tok):
    t = tok.replace(",", "").replace("−", "-")
    dec = len(t.split(".")[1]) if "." in t else 0
    x = abs(float(t))
    return any(abs(round(p, dec) - x) < 10 ** (-dec) / 2 + 1e-12 for p in pool)

for path in sys.argv[1:]:
    doc = Document(path)
    texts = [p.text for p in doc.paragraphs] + [c.text for t in doc.tables for r in t.rows for c in r.cells]
    cut = next((i for i, s in enumerate(texts) if s.strip() == "References"), None)
    end = next((i for i, s in enumerate(texts) if s.strip() in ("Declarations", "Statements and Declarations")), None)
    if cut is not None: texts = texts[:cut] + (texts[end:] if end else [])
    toks, miss = [], {}
    for s in texts:
        s2 = re.sub(r"\b(?:Sect|Fig|Table|Tables)\.? ?S?\d+(?:\.\d+)?[a-c]?(?:[–-]\d+(?:\.\d+)?)?", " ", s)
        s2 = re.sub(r"\(?(?:19|20)\d\d[a-z]?\)?", " ", s2)                      # years
        s2 = re.sub(r"\d{4}-\d{4}-\d{4}-\d{4}|AIRQ-D-26-01116|v\d\.\d|R[12]\.\w+|PM2\.5|PM2\.2|PM10|\(2,0,1\)|t[+−-]\d", " ", s2)
        for m in re.findall(r"(?<![\w.])[−-]?\d[\d,]*\.?\d*(?<![.,])", s2):
            toks.append(m)
            if not ok(m): miss.setdefault(m, s[:90])
    dec = [m for m in miss if "." in m or len(m.replace(",", "")) >= 4]
    print(f"\n{Path(path).name}: {len(toks)} numeric tokens, {len(toks) - sum(1 for t in toks if t in miss)} matched to output files")
    print(" unmatched decimals / large integers (read each):")
    for m in dec: print(f"   {m:>10}  ...{miss[m]}")
    print(" unmatched small integers:", sorted({m for m in miss if m not in dec}, key=lambda z: float(z.replace(',', '').replace('−', '-'))))
