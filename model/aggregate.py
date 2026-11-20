import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True)
DATASETS = ("PEMS08", "PEMS04")


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def pm(v, d=3):
    if len(v) == 0:
        return "n/a"
    m, s = ms(v)
    return f"{m:.{d}f} ± {s:.{d}f}"


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


# ---------------------- RQ1: trained on all sensors ---------------------- #
rows1 = [["dataset", "variant", "n", "mae", "mae_std", "rmse", "rmse_std",
          "mape", "mape_std", "dmae_pct_vs_original", "emb_params", "sec_per_epoch"]]
for ds in DATASETS:
    runs = load(f"{ROOT}/results/runs/{ds}_*.json")
    if not runs:
        continue
    base = np.mean([r["test"]["mae"] for r in runs if r["variant"] == "original"])
    print(f"\n=== RQ1 {ds}: trained on all sensors, mean ± std over seeds ===")
    print(f"{'variant':11s}{'n':>3s}  {'MAE':>15s}  {'RMSE':>15s}  {'MAPE %':>15s}"
          f"  {'dMAE %':>7s}  {'emb params':>10s}  {'s/epoch':>7s}")
    for v in ("original", "spatial", "factorised", "generated", "noadp"):
        rr = [r for r in runs if r["variant"] == v]
        if not rr:
            continue
        mae = [r["test"]["mae"] for r in rr]
        rmse = [r["test"]["rmse"] for r in rr]
        mape = [r["test"]["mape"] for r in rr]
        d = 100 * (np.mean(mae) - base) / base
        emb = rr[0]["params_embedding"]
        spe = float(np.mean([r["sec_per_epoch"] for r in rr]))
        print(f"{v:11s}{len(rr):3d}  {pm(mae):>15s}  {pm(rmse):>15s}  {pm(mape, 2):>15s}"
              f"  {d:7.2f}  {emb:10,d}  {spe:7.1f}")
        rows1.append([ds, v, len(rr), *ms(mae), *ms(rmse), *ms(mape), d, emb, spe])
csv.writer(open(f"{OUT}/rq1.csv", "w")).writerows(rows1)


# --------------------- RQ2: sensors unseen in training --------------------- #
def ind_split(ds, v, key, metric):
    out = {}
    for r in load(f"{ROOT}/results/ind/{ds}_{v}_split*.json"):
        if key in r:
            out[r["split_seed"]] = r[key][metric]
    return out


def ref_split(ds, v, metric):
    files = load(f"{ROOT}/results/ind/ref_{ds}_{v}_s*.json")
    out = {}
    for k in ("0", "1", "2"):
        vals = [f["splits"][k][metric] for f in files if k in f["splits"]]
        if vals:
            out[int(k)] = float(np.mean(vals))
    return out


def vals(d):
    return [d[k] for k in sorted(d)]


LABELS = {"hi": "Historical Inertia", "noadp": "no E_a",
          "original": "E_a + NN fallback", "factorised": "factorised + NN fallback",
          "generated": "generated (G)"}
rows2 = [["dataset", "group", "method", "n", "mae", "mae_std", "rmse", "rmse_std",
          "mape", "mape_std", "mae_new_only", "gap_vs_transductive_original"]]
for ds in DATASETS:
    ref_orig = ref_split(ds, "original", "mae")
    if not ref_orig:
        continue
    print(f"\n=== RQ2 {ds}: held-out sensors, mean ± std over splits ===")
    print(f"{'method':28s}{'n':>3s}  {'MAE':>15s}  {'RMSE':>15s}  {'MAPE %':>15s}"
          f"  {'MAE new-only':>15s}  {'gap MAE':>15s}")
    for v in ("original", "factorised", "generated", "noadp"):
        m = {k: ref_split(ds, v, k) for k in ("mae", "rmse", "mape")}
        name = f"transductive {v}"
        print(f"{name:28s}{len(m['mae']):3d}  {pm(vals(m['mae'])):>15s}  "
              f"{pm(vals(m['rmse'])):>15s}  {pm(vals(m['mape']), 2):>15s}")
        rows2.append([ds, "transductive", v, len(m["mae"]), *ms(vals(m["mae"])),
                      *ms(vals(m["rmse"])), *ms(vals(m["mape"])), "", ""])
    got = {}
    for v in ("hi", "noadp", "original", "factorised", "generated"):
        m = {k: ind_split(ds, v, "new_in_all", k) for k in ("mae", "rmse", "mape")}
        only = ind_split(ds, v, "new_only", "mae")
        gap = [m["mae"][s] - ref_orig[s] for s in m["mae"] if s in ref_orig]
        got[v] = m["mae"]
        if not m["mae"]:
            print(f"{LABELS[v]:28s}  0  n/a")
            continue
        print(f"{LABELS[v]:28s}{len(m['mae']):3d}  {pm(vals(m['mae'])):>15s}  "
              f"{pm(vals(m['rmse'])):>15s}  {pm(vals(m['mape']), 2):>15s}  "
              f"{pm(vals(only)):>15s}  {pm(gap):>15s}")
        rows2.append([ds, "inductive", v, len(m["mae"]), *ms(vals(m["mae"])),
                      *ms(vals(m["rmse"])), *ms(vals(m["mape"])),
                      ms(vals(only))[0] if only else "", ms(gap)[0] if gap else ""])
    for other in ("noadp", "original"):
        diff = [got["generated"][s] - got[other][s]
                for s in got.get("generated", {}) if s in got.get(other, {})]
        if diff:
            print(f"paired MAE, generated minus {LABELS[other]}: {pm(diff)}")
csv.writer(open(f"{OUT}/rq2.csv", "w")).writerows(rows2)
print("\nsaved CSVs in", OUT)
