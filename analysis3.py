"""Additional analyses for the rewrite: pooled per-class results over seeds, confusion matrices,
prior reweighting (EM-DAT 2000-2019 frequencies), deterministic calibration on the analysis split,
baseline parameter counts, and figures."""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent  # repository root (all files in one folder)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_revision_experiments as R
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder

OUT = ROOT / "analysis3"; OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "events_original_with_regions.csv", converters={"event_id": str, "region": str})
_cl = pd.read_csv(ROOT / "events_clean_with_regions.csv", converters={"event_id": str})
df = df.merge(_cl[["event_id"] + [c for c in _cl.columns if c not in df.columns]], on="event_id")
F15 = "rainfall_mm temperature_c humidity_percent wind_speed_kmh atmospheric_pressure_hpa river_level_m soil_moisture_percent slope_degrees vegetation_dryness seismic_activity ground_displacement_mm satellite_water_index population_density road_accessibility historical_risk_score".split()
LEAK = ["seismic_activity", "ground_displacement_mm", "wind_speed_kmh", "historical_risk_score"]
F11 = "precip_mean_mm_day precip_antecedent_30d_mm temperature_mean_c rel_humidity_mean_pct wind_max_mean_kmh surface_pressure_mean_hpa et0_mean_mm_day soil_moisture_0_7cm_m3m3 elevation_m slope_deg".split()
le = LabelEncoder(); y = le.fit_transform(df.disaster_type); names = list(le.classes_); C = len(names)

class A: seeds = 10
EMDAT = {"Cyclone": 2043, "Earthquake": 552, "Flood": 3254, "Landslide": 376, "Wildfire": 238}  # UNDRR/CRED 2020, Fig. 4 (storm used for cyclone)
pri = np.array([EMDAT[n] for n in names], float); pri /= pri.sum()

summary = {}
for tag, feats in [("B", F11), ("A", F15)]:
    X = df[feats].to_numpy(float)
    yt, yp, P = [], [], []
    for s, tr, va, te in R.split_generator("repeated_stratified", df, y, A):
        pre = R.Pipeline(R.make_pre()).fit(X[tr])
        clf = R.TorchRadiant(seed=s, n_classes=C).fit(pre.transform(X[tr]), y[tr], pre.transform(X[va]), y[va])
        p = clf.predict_proba(pre.transform(X[te]))
        yt.append(y[te]); yp.append(p.argmax(1)); P.append(p)
        if s == 0:  # deterministic metrics on the analysis split (seed 0)
            summary[f"{tag}_seed0"] = dict(R.metrics(y[te], p, C), mean_entropy=float(R.entropy(p).mean()),
                                         error_detection_auroc=float(roc_auc_score(p.argmax(1) != y[te], R.entropy(p))))
    yt, yp, P = np.concatenate(yt), np.concatenate(yp), np.concatenate(P)
    cm = confusion_matrix(yt, yp, labels=range(C))
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, labels=range(C), zero_division=0)
    pc = pd.DataFrame(dict(cls=names, precision=pr, recall=rc, f1=f1, support=cm.sum(1)))
    pc.to_csv(OUT / f"per_class_{tag}.csv", index=False)
    pd.DataFrame(cm, index=names, columns=names).to_csv(OUT / f"confusion_{tag}.csv")
    rates = cm / cm.sum(1, keepdims=True)
    out = {}
    for lab, pi in [("balanced", np.full(C, 1 / C)), ("EM-DAT 2000-2019", pri)]:
        joint = pi[:, None] * rates
        prec = np.diag(joint) / np.clip(joint.sum(0), 1e-12, None); rec = np.diag(rates)
        f = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        out[lab] = dict(accuracy=float(np.trace(joint)), macro_f1=float(f.mean()),
                        precision={n: float(prec[k]) for k, n in enumerate(names)})
    summary[f"{tag}_priors"] = out
    summary[f"{tag}_pooled_acc"] = float((yt == yp).mean())

# parameter counts of deep models
for tag, feats in [("B", F11), ("A", F15)]:
    Xs = np.nan_to_num(df[feats].to_numpy(float)[:64]); Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    counts = {}
    for nm, spec in {"RADIANT-Net v1": {}, **R.DEEP_BASELINES}.items():
        m = R.TorchRadiant(max_epochs=1, **spec).fit(Xs, y[:64]); counts[nm] = m.n_params()
    summary[f"{tag}_params"] = counts
summary["priors_used"] = dict(zip(names, pri.round(4).tolist()))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

# soil-moisture zeros and other artefacts
art = df.assign(sm0=df.soil_moisture_percent == 0, rf0=df.rainfall_mm == 0).groupby("disaster_type")[["sm0", "rf0"]].sum()
art.to_csv(OUT / "artefacts.csv")

# ---------------- figures ----------------
col = {"Cyclone": "#1f77b4", "Earthquake": "#d62728", "Flood": "#2ca02c", "Landslide": "#9467bd", "Wildfire": "#ff7f0e"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
for n in names:
    d = df[df.disaster_type == n]
    ax[0].scatter(d.longitude, d.latitude, s=10, c=col[n], label=f"{n} ({len(d)})", alpha=.8)
ax[0].set_xlim(-180, 180); ax[0].set_ylim(-70, 75); ax[0].grid(alpha=.3)
ax[0].set_xlabel("Longitude (°)"); ax[0].set_ylabel("Latitude (°)"); ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_title("(a) Event locations", fontsize=10)
dt = pd.to_datetime(df.event_date)
for i, n in enumerate(names):
    d = dt[df.disaster_type == n]
    ax[1].scatter(d, np.full(len(d), i) + np.random.default_rng(i).uniform(-.25, .25, len(d)), s=8, c=col[n])
ax[1].set_yticks(range(C)); ax[1].set_yticklabels(names); ax[1].grid(alpha=.3, axis="x")
ax[1].set_title("(b) Event dates (each hazard type has its own period)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_events.png", dpi=250); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for a, tag, title in [(axes[0], "B", "(a) 10 re-extracted features (main analysis)"), (axes[1], "A", "(b) original 15 features")]:
    cm = pd.read_csv(OUT / f"confusion_{tag}.csv", index_col=0).values
    rn = cm / cm.sum(1, keepdims=True)
    a.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(C):
        for j in range(C):
            a.text(j, i, f"{cm[i, j]}\n({rn[i, j]:.0%})", ha="center", va="center", fontsize=7.5,
                   color="white" if rn[i, j] > .55 else "black")
    a.set_xticks(range(C)); a.set_xticklabels(names, rotation=30, fontsize=8); a.set_yticks(range(C)); a.set_yticklabels(names, fontsize=8)
    a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(title, fontsize=10)
fig.suptitle("RADIANT-Net v1, test predictions pooled over 10 seeds (750 events)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_confusion.png", dpi=250); plt.close(fig)

# model comparison figure (main analysis B)
r = pd.read_csv(ROOT / "results_clean" / "results_per_split.csv"); r = r[r.feature_set == "all"]
models = ["RADIANT-Net v1 (full)", "Random forest", "Gradient boosting", "Logistic regression", "SVM (RBF)",
          "MLP (param-matched)", "CNN-LSTM", "Transformer (feature tokens, ViT-style)", "GNN (kNN GraphSAGE, inductive)"]
short = ["RADIANT-Net", "Random forest", "Grad. boosting", "Logistic reg.", "SVM", "MLP", "CNN-LSTM", "Transformer", "GraphSAGE"]
prots = [("repeated_stratified", "Random splits"), ("event_grouped", "Event-grouped"), ("leave_region_out", "Leave-region-out")]
fig, ax = plt.subplots(figsize=(11, 4)); w = .27
for k, (p, lab) in enumerate(prots):
    g = r[r.protocol == p].groupby("model").accuracy.agg(["mean", "std"]).reindex(models)
    ax.bar(np.arange(len(models)) + (k - 1) * w, g["mean"], w, yerr=g["std"], capsize=2, label=lab,
           color=["#2b5c9e", "#7fa6d6", "#c7d7ee"][k], edgecolor="#333", linewidth=.4)
ax.axhline(.2, ls=":", c="gray", lw=1); ax.text(len(models) - .5, .21, "chance (0.20)", fontsize=8, ha="right", color="gray")
ax.axhline(.714, ls="--", c="#c0392b", lw=1); ax.text(len(models) - .5, .725, "event date only, random forest (0.714)", fontsize=8, ha="right", color="#c0392b")
ax.set_xticks(range(len(models))); ax.set_xticklabels(short, fontsize=9); ax.set_ylim(0, 1); ax.set_ylabel("Test accuracy (mean ± SD)")
ax.legend(fontsize=8, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.13)); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig_models.png", dpi=250); plt.close(fig)

# SHAP side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for a, d, t in [(axes[0], "results_clean", "(a) 10 re-extracted features (main analysis)"), (axes[1], "results15_orig", "(b) original 15 features")]:
    s = pd.read_csv(ROOT / d / "shap_mean_abs.csv", index_col=0).sort_values("overall")
    c = ["#c0392b" if i in LEAK + ["slope_degrees", "population_density", "road_accessibility"] else "#2b5c9e" for i in s.index]
    a.barh(s.index, s.overall, color=c); a.set_title(t, fontsize=10); a.set_xlabel("mean |SHAP value|"); a.tick_params(labelsize=8)
axes[1].text(.98, .02, "red: catalogue-derived or generated features", transform=axes[1].transAxes, ha="right", fontsize=8, color="#c0392b")
fig.tight_layout(); fig.savefig(OUT / "fig_shap.png", dpi=250); plt.close(fig)

# stress tests
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for d, lab, c in [("results_clean", "10 re-extracted features", "#2b5c9e"), ("results15_orig", "original 15 features", "#888888")]:
    s = pd.read_csv(ROOT / d / "stress_tests.csv")
    for a, key, xl in [(axes[0], "random missing values", "Fraction of values missing"),
                       (axes[1], "Gaussian sensor noise (x feature SD)", "Noise SD (× feature SD)")]:
        q = s[s.test == key]; x = q.level.astype(float)
        a.errorbar(x, q.accuracy_mean, yerr=q.accuracy_sd, marker="o", c=c, label=lab, capsize=2)
        a.set_xlabel(xl); a.set_ylabel("Accuracy"); a.set_ylim(.3, 1); a.grid(alpha=.3)
axes[0].set_title("(a) Random missing values", fontsize=10); axes[1].set_title("(b) Gaussian sensor noise", fontsize=10); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_stress.png", dpi=250); plt.close(fig)

# Supplementary Table S1 (event list)
s1 = df[["event_id", "source", "source_id", "disaster_type", "event_date", "latitude", "longitude", "region"]]
s1.to_csv(OUT / "Supplementary_Table_S1_events.csv", index=False)
print(json.dumps(summary, indent=1)[:3000]); print(art)
