# ruff: noqa
# Committed as it ran on the VM. The marker regexes below are the
# definition of the published cohort groups; tidying this file would
# silently redefine what the paper's table measures. Only this header
# was added. See scripts/cohort/README.md.
"""Cohort sanity check on the MIMIC held-out alert rows: who is scored, and does it matter?"""
import glob, json, sys
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score

shards = sorted(glob.glob(sys.argv[1] + "/*.parquet"))
dump = pl.read_parquet(sys.argv[2])
ev = pl.concat([pl.scan_parquet(p).select("subject_id", "code", "time") for p in shards]).collect()
FLAGS = {
    "esrd_dialysis": r"TREATMENT//ENTERED//renal\|dialysis\|.*chronic renal failure|DIAGNOSIS//ICD//(10//(N186|Z992)|9//(5856|V4511))",
    "palliative_hospice": r"Comfort measures only|palliative care consultation|comfort goal identified",
}
subj = ev.select("subject_id").unique()
for name, pat in FLAGS.items():
    hit = ev.filter(pl.col("code").str.contains(pat)).select("subject_id").unique().with_columns(pl.lit(True).alias(name))
    subj = subj.join(hit, on="subject_id", how="left").with_columns(pl.col(name).fill_null(False))
subj = subj.with_columns(pl.col("subject_id").cast(pl.Float64))
n_subj = subj.height
out = {"n_held_out_subjects": n_subj, "flag_prevalence_subjects": {k: int(subj[k].sum()) for k in FLAGS}}
d = dump.join(subj, on="subject_id", how="left")
d = d.with_columns([pl.col(k).fill_null(False) for k in FLAGS])
d = d.with_columns(pl.col("ctx.age_years").alias("age"), pl.col("ctx.in_icu").alias("in_icu"), pl.col("ctx.n_events_visit").alias("nev"))
out["age_years_quantiles_rows"] = {q: float(d["age"].quantile(q)) for q in (0.0, 0.01, 0.5, 0.99, 1.0)}
out["rows_age_lt_18"] = int((d["age"] < 18).sum())
out["rows_by_event"] = {}
for evn in d["event"].unique().to_list():
    de = d.filter(pl.col("event") == evn)
    rec = {"n_rows": de.height}
    y = de["y@24h"].to_numpy(); p = de["hazard@24h"].to_numpy()
    ok = ~np.isnan(y) & ~np.isnan(p)
    rec["n_at_risk_24h"] = int(ok.sum()); rec["positives_24h"] = int(np.nansum(y[ok]))
    rec["auroc_24h_all"] = float(roc_auc_score(y[ok], p[ok]))
    if evn == "icu_admission":
        rec["at_risk_rows_scored_while_in_icu"] = int(de.filter(pl.col("in_icu") == 1)["y@24h"].is_not_null().sum())
    for k in FLAGS:
        f = de[k].to_numpy()
        m = ok & ~f
        rec[k] = {
            "at_risk_rows_flagged": int((ok & f).sum()),
            "positives_flagged": int(np.nansum(y[ok & f])),
            "auroc_24h_excluding_flagged": float(roc_auc_score(y[m], p[m])),
            "auroc_24h_flagged_only": float(roc_auc_score(y[ok & f], p[ok & f])) if len(set(y[ok & f])) > 1 else None,
        }
    small = ok & (de["nev"].to_numpy() < 10)
    rec["at_risk_rows_visit_lt_10_events"] = int(small.sum())
    out["rows_by_event"][evn] = rec
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps(out, indent=1))
