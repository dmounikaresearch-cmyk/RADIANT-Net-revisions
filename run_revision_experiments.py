#!/usr/bin/env python3
"""
run_revision_experiments.py
===========================
Experiments requested by Reviewer 1 for the RADIANT-Net manuscript (Environmental Sciences Europe).

What it does
------------
* Leak-free preprocessing: median imputation + StandardScaler are fitted on the TRAINING part of every
  split only (scikit-learn Pipeline), then applied to validation/test.
* Four evaluation protocols
    1. repeated stratified train/validation/test (70/15/15) over --seeds seeds
    2. event-grouped 5-fold CV         (needs --event-col)
    3. leave-region-out CV             (needs --region-col)
    4. temporal holdout                (needs --date-col and --temporal-cutoff)
  Early stopping / hyper-parameter selection uses the validation partition (or inner CV) only;
  the test partition is scored exactly once per split.
* Models: RADIANT-Net v1 (encoder + GRUCell + decision layer) and its ablations, plus tuned tabular
  baselines (logistic regression, RBF-SVM, random forest, gradient boosting, parameter-matched MLP).
* Metrics: accuracy, macro P/R/F1, macro one-vs-rest ROC-AUC, macro average precision, Brier score,
  expected calibration error (10 bins), confusion matrices; Wilson CI for single test sets.
* Statistics: Nadeau-Bengio corrected resampled t-test and Wilcoxon signed-rank test on identical splits.
* Feature analyses: input-group ablations, exclusion of potential label proxies, permutation importance.
* Reviewer 5: stress tests (graded random missingness, Gaussian sensor noise, conflicting sources) and
  repeated timing (forward pass only vs preprocessing + forward + decision, 20 repetitions).
* Reviewer 4: missing-modality robustness (one modality removed at inference), leave-one-hazard-out
  out-of-distribution detection, performance under operational class priors (--priors), and
  per-class feature distributions (Supplementary Fig. S1).
* Reviewer 3: per-feature missing-data rates, descriptive statistics and the fitted imputer/scaler
  parameters (Supplementary Table S2), and a class-by-region table.
* Reviewer 2: CNN-LSTM, ViT-style feature-token Transformer and inductive kNN-GraphSAGE baselines;
  KernelSHAP explanations (pip install shap); MC-dropout and deep-ensemble uncertainty with
  error-detection AUROC; inference throughput vs batch size; learning curve vs training-set size.
* Writes per-split CSV, summary CSV and Markdown tables ready to paste into Tables 7-9 of the manuscript.

Usage
-----
    pip install numpy pandas scipy scikit-learn torch matplotlib shap   # xgboost optional
    python run_revision_experiments.py --data events.csv --label disaster_type \
        --event-col event_id --region-col region --date-col event_date --temporal-cutoff 2022-01-01 \
        --seeds 10 --out results/

The CSV must have one row per event, the label column, and the numeric feature columns.
Columns used for grouping (event id, region, date) are excluded from the features automatically.
EDIT `RadiantNetV1` below so that it matches your original architecture exactly (the script prints the
parameter count; the paper reports 45,189).
"""
import argparse, os, json, math, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None

# --------------------------------------------------------------------------------------------------
# Feature groups (edit names to match your CSV columns). Matching is case-insensitive substring.
# --------------------------------------------------------------------------------------------------
DEFAULT_GROUPS = {
    "meteorological": ["rain", "precip", "temp", "humid", "wind", "pressure", "soil_moist"],
    "satellite": ["vv", "vh", "ndvi", "ndwi"],
    "proxies": ["seismic", "displacement", "risk_score", "historical_risk"],
}


def match_cols(cols, patterns):
    out = []
    for c in cols:
        lc = c.lower()
        if any(p.lower() == lc or p.lower() in lc for p in patterns):
            out.append(c)
    return out


# --------------------------------------------------------------------------------------------------
# RADIANT-Net v1 (PyTorch) with ablation switches, wrapped as a scikit-learn classifier
# --------------------------------------------------------------------------------------------------
if torch is not None:
    class RadiantNetV1(nn.Module):
        """The authors' original RADIANT-Net prototype (src/model.py of the project), unchanged:
        MultimodalFeatureEncoder: Linear(F->64)+BN+ReLU+Dropout -> Linear(64->64)+ReLU
        NeuralTwinMemory: GRUCell(64,64) for one step from a zero state; refinement
            Linear(128->64)+ReLU+Dropout -> Linear(64->64) on [encoded, twin state]; residual add
        AgenticDecisionLayer: Linear(64->32)+ReLU+Dropout -> Linear(32->5)
        45,189 parameters for 19 inputs (as reported in the original submission).
        Ablation flags: use_gru=False drops the GRUCell (refinement sees the encoded features only);
        use_encoder=False removes the whole neural-twin module (encoder -> decision layer)."""

        def __init__(self, n_in, n_cls, hidden=64, enc_layers=None, use_bn=True, dropout=0.2,
                     use_gru=True, use_encoder=True):
            super().__init__()
            enc = [nn.Linear(n_in, hidden)]
            if use_bn:
                enc += [nn.BatchNorm1d(hidden)]
            enc += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU()]
            self.encoder = nn.Sequential(*enc)
            self.use_twin, self.use_gru = use_encoder, use_gru
            if self.use_twin:
                self.gru = nn.GRUCell(hidden, hidden) if use_gru else None
                self.refine = nn.Sequential(nn.Linear(hidden * (2 if use_gru else 1), hidden), nn.ReLU(),
                                            nn.Dropout(dropout), nn.Linear(hidden, hidden))
            self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden // 2, n_cls))

        def forward(self, x):
            h = self.encoder(x)
            if self.use_twin:
                if self.use_gru:
                    z = self.gru(h, torch.zeros_like(h))
                    r = self.refine(torch.cat([h, z], dim=1))
                else:
                    r = self.refine(h)
                h = r + h
            return self.head(h)


if torch is not None:
    class CNNLSTM(nn.Module):
        """CNN-LSTM baseline: the feature vector is read as a 1-D sequence (feature order is arbitrary)."""
        def __init__(self, n_in, n_cls, channels=16, hidden=32, dropout=0.2):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(1, channels, 3, padding=1), nn.ReLU())
            self.lstm = nn.LSTM(channels, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, n_cls)

        def forward(self, x):
            h = self.conv(x.unsqueeze(1)).transpose(1, 2)      # (B, F, C)
            _, (hn, _) = self.lstm(h)
            return self.head(self.drop(hn[-1]))

    class TabTransformer(nn.Module):
        """ViT-style Transformer for tabular input: each feature becomes a token (FT-Transformer-like)."""
        def __init__(self, n_in, n_cls, d=32, heads=4, layers=2, dropout=0.2):
            super().__init__()
            self.w = nn.Parameter(torch.randn(n_in, d) * 0.1)
            self.b = nn.Parameter(torch.zeros(n_in, d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(d, n_cls)

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.w + self.b             # (B, F, d)
            tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)
            return self.head(self.enc(tok)[:, 0])

    class KNNSage(nn.Module):
        """Inductive 2-layer GraphSAGE (mean aggregator) on a k-nearest-neighbour graph whose nodes are
        TRAINING events only. Test events are attached to their k nearest training events, so no test
        information enters training."""
        def __init__(self, n_in, n_cls, hidden=64, dropout=0.2):
            super().__init__()
            self.s1, self.n1 = nn.Linear(n_in, hidden), nn.Linear(n_in, hidden, bias=False)
            self.s2, self.n2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden, bias=False)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, n_cls)

        def layer1(self, x, nbr_mean):
            return torch.relu(self.s1(x) + self.n1(nbr_mean))

        def forward(self, x, nbr, Xtr, nbr_tr):
            # layer 1 for all training nodes (needed as neighbour messages for layer 2)
            h_tr = self.layer1(Xtr, Xtr[nbr_tr].mean(1))
            h = self.layer1(x, Xtr[nbr].mean(1))
            h = torch.relu(self.s2(self.drop(h)) + self.n2(h_tr[nbr].mean(1)))
            return self.head(self.drop(h))


class TorchRadiant(BaseEstimator, ClassifierMixin):
    """sklearn wrapper. Early stopping on a validation set passed via fit(..., X_val, y_val)."""

    def __init__(self, hidden=64, enc_layers=(128, 64), use_bn=True, dropout=0.2, use_gru=True,
                 use_encoder=True, lr=1e-3, batch_size=64, max_epochs=200, patience=20, seed=0,
                 fixed_epochs=None, arch="radiant", knn=10, n_classes=None):
        self.hidden, self.enc_layers, self.use_bn, self.dropout = hidden, enc_layers, use_bn, dropout
        self.use_gru, self.use_encoder, self.lr, self.batch_size = use_gru, use_encoder, lr, batch_size
        self.max_epochs, self.patience, self.seed, self.fixed_epochs = max_epochs, patience, seed, fixed_epochs
        self.arch, self.knn, self.n_classes = arch, knn, n_classes

    def _build(self, n_in, n_cls):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.arch == "cnnlstm":
            return CNNLSTM(n_in, n_cls, dropout=self.dropout)
        if self.arch == "transformer":
            return TabTransformer(n_in, n_cls, dropout=self.dropout)
        if self.arch == "gnn":
            return KNNSage(n_in, n_cls, hidden=self.hidden, dropout=self.dropout)
        return RadiantNetV1(n_in, n_cls, self.hidden, self.enc_layers, self.use_bn, self.dropout,
                            self.use_gru, self.use_encoder)

    # --- graph helpers (arch == "gnn") ---
    def _query_nbrs(self, X):
        from sklearn.neighbors import NearestNeighbors
        return torch.tensor(self.nn_.kneighbors(np.asarray(X), n_neighbors=self.knn, return_distance=False))

    def _logits(self, Xb, idx=None):
        if self.arch != "gnn":
            return self.model_(Xb)
        nbr = self.nbr_tr_[idx] if idx is not None else self._query_nbrs(Xb.numpy())
        return self.model_(Xb, nbr, self.Xtr_, self.nbr_tr_)

    def fit(self, X, y, X_val=None, y_val=None):
        if torch is None:
            raise ImportError("PyTorch is required for RADIANT-Net")
        torch.set_num_threads(1)
        # all classes are kept as outputs even if one is absent from this training split
        self.classes_ = np.arange(self.n_classes) if self.n_classes else np.unique(y)
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.long)
        self.model_ = self._build(Xt.shape[1], len(self.classes_))
        if self.arch == "gnn":
            from sklearn.neighbors import NearestNeighbors
            self.Xtr_ = Xt
            self.nn_ = NearestNeighbors().fit(np.asarray(X))
            # neighbours of training nodes among training nodes, excluding the node itself
            nb = self.nn_.kneighbors(np.asarray(X), n_neighbors=self.knn + 1, return_distance=False)[:, 1:]
            self.nbr_tr_ = torch.tensor(nb)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, wait = np.inf, None, 0
        epochs = self.fixed_epochs or self.max_epochs
        for ep in range(epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2 and self.use_bn:
                    continue
                opt.zero_grad()
                loss = lossf(self._logits(Xt[idx], idx), yt[idx])
                loss.backward()
                opt.step()
            if X_val is not None and self.fixed_epochs is None:
                self.model_.eval()
                with torch.no_grad():
                    vl = lossf(self._logits(torch.tensor(np.asarray(X_val), dtype=torch.float32)),
                               torch.tensor(np.asarray(y_val), dtype=torch.long)).item()
                if vl < best - 1e-5:
                    best, wait = vl, 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_epochs_ = ep + 1
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self._logits(torch.tensor(np.asarray(X), dtype=torch.float32))
        return torch.softmax(logits, 1).numpy()

    def predict_proba_mc(self, X, T=50):
        """Monte-Carlo dropout: dropout active, batch-norm in inference mode."""
        self.model_.eval()
        for m in self.model_.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            P = np.stack([torch.softmax(self._logits(Xt), 1).numpy() for _ in range(T)])
        self.model_.eval()
        return P

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def n_params(self):
        return sum(p.numel() for p in self.model_.parameters() if p.requires_grad)


def make_pre():
    return [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------
RADIANT_VARIANTS = {
    "RADIANT-Net v1 (full)": dict(),
    "w/o GRUCell (refinement kept)": dict(use_gru=False),
    "w/o batch normalisation": dict(use_bn=False),
    "w/o dropout": dict(dropout=0.0),
    "w/o neural-twin module": dict(use_encoder=False, use_gru=False),
}


DEEP_BASELINES = {
    "CNN-LSTM": dict(arch="cnnlstm"),
    "Transformer (feature tokens, ViT-style)": dict(arch="transformer"),
    "GNN (kNN GraphSAGE, inductive)": dict(arch="gnn"),
}


def baseline_grids():
    grids = {
        "Logistic regression": (LogisticRegression(max_iter=5000), {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "SVM (RBF)": (CalibratedClassifierCV(SVC(), ensemble=False),
                      {"clf__estimator__C": [0.1, 1, 10, 100], "clf__estimator__gamma": ["scale", 0.01, 0.1]}),
        "Random forest": (RandomForestClassifier(n_jobs=-1),
                          {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 5, 10],
                           "clf__min_samples_leaf": [1, 3]}),
        "Gradient boosting": (HistGradientBoostingClassifier(),
                              {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [3, None],
                               "clf__max_iter": [200, 500]}),
        "MLP (param-matched)": (MLPClassifier(max_iter=2000, early_stopping=True),
                                {"clf__hidden_layer_sizes": [(128, 64), (64, 64)], "clf__alpha": [1e-4, 1e-3]}),
    }
    try:
        from xgboost import XGBClassifier
        grids["XGBoost"] = (XGBClassifier(eval_metric="mlogloss", n_jobs=1),
                            {"clf__n_estimators": [200, 500], "clf__max_depth": [3, 6],
                             "clf__learning_rate": [0.03, 0.1]})
    except ImportError:
        pass
    return grids


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------
def ece_score(y, proba, n_bins=10):
    conf = proba.max(1)
    pred = proba.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def metrics(y, proba, n_cls):
    pred = proba.argmax(1)
    Y = np.eye(n_cls)[y]
    out = dict(
        accuracy=accuracy_score(y, pred),
        macro_precision=precision_score(y, pred, average="macro", zero_division=0),
        macro_recall=recall_score(y, pred, average="macro", zero_division=0),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        ece=ece_score(y, proba),
        brier=np.mean(np.sum((proba - Y) ** 2, 1)),
    )
    try:
        out["macro_auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_cls)))
        out["macro_ap"] = average_precision_score(Y, proba, average="macro")
    except ValueError:  # a class missing from this test fold
        out["macro_auc"] = np.nan
        out["macro_ap"] = np.nan
    return out


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def corrected_t(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired scores a, b over J splits."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    if J < 2 or np.var(d, ddof=1) == 0:
        return np.nan
    var = (1 / J + n_test / n_train) * np.var(d, ddof=1)
    t = d.mean() / math.sqrt(var)
    return 2 * stats.t.sf(abs(t), J - 1)


def wilcoxon_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0) or len(d) < 5:
        return np.nan
    return stats.wilcoxon(a, b).pvalue


# --------------------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------------------
def inner_val(idx_train, y, seed, frac=0.15 / 0.85):
    tr, va = train_test_split(idx_train, test_size=frac, stratify=y[idx_train], random_state=seed)
    return tr, va


def split_generator(protocol, df, y, args):
    n = len(y)
    idx = np.arange(n)
    if protocol == "repeated_stratified":
        for s in range(args.seeds):
            tr_va, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=s)
            tr, va = inner_val(tr_va, y, s)
            yield s, tr, va, te
    elif protocol == "event_grouped":
        sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for k, (tr_va, te) in enumerate(sgk.split(idx, y, df[args.event_col])):
            tr, va = inner_val(tr_va, y, k)
            yield k, tr, va, te
    elif protocol == "leave_region_out":
        groups = df[args.region_col].values
        n_reg = len(np.unique(groups))
        gkf = GroupKFold(n_splits=min(n_reg, args.region_folds))
        for k, (tr_va, te) in enumerate(gkf.split(idx, y, groups)):
            tr, va = inner_val(tr_va, y, k)
            yield k, tr, va, te
    elif protocol == "temporal_holdout":
        dates = pd.to_datetime(df[args.date_col])
        cut = pd.Timestamp(args.temporal_cutoff)
        tr_va, te = idx[dates < cut], idx[dates >= cut]
        # validation = the most recent 15% of the training period (keeps time order)
        order = tr_va[np.argsort(dates.values[tr_va])]
        nv = max(1, int(0.15 * len(order)))
        yield 0, order[:-nv], order[-nv:], te


# --------------------------------------------------------------------------------------------------
# Fitting one model on one split
# --------------------------------------------------------------------------------------------------
def fit_eval(name, kind, spec, X, y, tr, va, te, seed, n_cls, inner_cv_seed=0):
    t0 = time.time()
    if kind == "radiant":
        pre = Pipeline(make_pre())
        Xtr = pre.fit_transform(X[tr])                 # fitted on training only
        Xva, Xte = pre.transform(X[va]), pre.transform(X[te])
        clf = TorchRadiant(seed=seed, n_classes=n_cls, **spec).fit(Xtr, y[tr], Xva, y[va])
        proba = clf.predict_proba(Xte)
        extra = dict(epochs=clf.n_epochs_, n_params=clf.n_params())
        model = (pre, clf)
    else:
        est, grid = spec
        pipe = Pipeline(make_pre() + [("clf", clone(est))])
        if "clf__random_state" in pipe.get_params():
            pipe.set_params(clf__random_state=seed)
        trva = np.concatenate([tr, va])                # inner CV on train+val only; test untouched
        k_inner = int(max(2, min(5, np.bincount(y[trva])[np.bincount(y[trva]) > 0].min())))
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(k_inner, shuffle=True, random_state=inner_cv_seed),
                          scoring="f1_macro", n_jobs=-1)
        gs.fit(X[trva], y[trva])
        p_raw = gs.predict_proba(X[te])
        proba = np.zeros((len(te), n_cls))
        proba[:, gs.classes_] = p_raw            # expand if a class was absent from training
        extra = dict(best_params=json.dumps(gs.best_params_))
        model = gs
    m = metrics(y[te], proba, n_cls)
    m.update(extra, model=name, fit_seconds=time.time() - t0, n_test=len(te), n_train=len(tr) + len(va))
    return m, proba, model



# --------------------------------------------------------------------------------------------------
# Reviewer 2: explainability, uncertainty and scalability analyses (reference split)
# --------------------------------------------------------------------------------------------------
def entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)


def run_extras(out, feats, X, y, n_cls, tr, va, te, pre, clf, split_id):
    Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])

    # ---- SHAP (model-agnostic KernelSHAP) ----
    try:
        import shap
        bg = shap.kmeans(Xtr, 20)
        expl = shap.KernelExplainer(clf.predict_proba, bg)
        sv = expl.shap_values(Xte, nsamples=200, silent=True)
        sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)   # (n, F, C)
        mean_abs = np.abs(sv).mean(0)                                        # (F, C)
        tab = pd.DataFrame(mean_abs, index=feats, columns=[f"class_{c}" for c in range(n_cls)])
        tab["overall"] = mean_abs.mean(1)
        tab.sort_values("overall", ascending=False).to_csv(out / "shap_mean_abs.csv")
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            t = tab.sort_values("overall")
            fig, ax = plt.subplots(figsize=(6, 0.3 * len(feats) + 1))
            ax.barh(t.index, t["overall"], color="#2b5c9e")
            ax.set_xlabel("mean |SHAP value| (averaged over classes)")
            fig.tight_layout(); fig.savefig(out / "shap_importance.png", dpi=300); plt.close(fig)
        except Exception as e:  # plotting is optional
            print("  SHAP plot skipped:", e)
        print("  SHAP: done")
    except ImportError:
        print("  SHAP skipped: pip install shap")

    # ---- Uncertainty: MC dropout and a 5-member deep ensemble ----
    rows = []
    err = None
    P = clf.predict_proba_mc(Xte, T=50)
    pm = P.mean(0)
    err = pm.argmax(1) != y[te]
    H = entropy(pm)
    rows.append(dict(method="MC dropout (T=50)", **metrics(y[te], pm, n_cls),
                     mean_entropy=H.mean(),
                     error_detection_auroc=roc_auc_score(err, H) if 0 < err.sum() < len(err) else np.nan))
    Xva = pre.transform(X[va])
    members = [TorchRadiant(seed=100 + k).fit(Xtr, y[tr], Xva, y[va]) for k in range(5)]
    pe = np.mean([m.predict_proba(Xte) for m in members], 0)
    err = pe.argmax(1) != y[te]
    H = entropy(pe)
    rows.append(dict(method="Deep ensemble (5 seeds)", **metrics(y[te], pe, n_cls),
                     mean_entropy=H.mean(),
                     error_detection_auroc=roc_auc_score(err, H) if 0 < err.sum() < len(err) else np.nan))
    pd.DataFrame(rows).to_csv(out / "uncertainty.csv", index=False)
    print("  uncertainty: done")

    # ---- Scalability: inference throughput vs batch size; learning curve vs training-set size ----
    sc = []
    clf.model_.eval()
    for bs in [1, 64, 1024, 8192]:
        xb = torch.tensor(np.resize(Xte, (bs, Xte.shape[1])), dtype=torch.float32)
        with torch.no_grad():
            for _ in range(5): clf.model_(xb)
            R = max(5, int(2000 / bs))
            t0 = time.perf_counter()
            for _ in range(R): clf.model_(xb)
        dt = (time.perf_counter() - t0) / R
        sc.append(dict(batch_size=bs, ms_per_batch=dt * 1000, ms_per_sample=dt * 1000 / bs, samples_per_s=bs / dt))
    pd.DataFrame(sc).to_csv(out / "scalability_inference.csv", index=False)
    lc = []
    rng = np.random.default_rng(0)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        for rep in range(3):
            sub = tr if frac == 1.0 else rng.choice(tr, int(frac * len(tr)), replace=False)
            pre_s = Pipeline(make_pre()).fit(X[sub])
            t0 = time.perf_counter()
            m = TorchRadiant(seed=rep).fit(pre_s.transform(X[sub]), y[sub], pre_s.transform(X[va]), y[va])
            ttrain = time.perf_counter() - t0
            pr = m.predict_proba(pre_s.transform(X[te]))
            lc.append(dict(train_fraction=frac, n_train=len(sub), rep=rep, train_seconds=ttrain,
                           **{k: v for k, v in metrics(y[te], pr, n_cls).items() if k in ("accuracy", "macro_f1")}))
    lcdf = pd.DataFrame(lc)
    lcdf.to_csv(out / "learning_curve.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        g = lcdf.groupby("n_train")[["accuracy"]].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.errorbar(g.index, g[("accuracy", "mean")], yerr=g[("accuracy", "std")], marker="o", color="#2b5c9e")
        ax.set_xlabel("training events"); ax.set_ylabel("test accuracy")
        fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=300); plt.close(fig)
    except Exception as e:
        print("  learning-curve plot skipped:", e)
    print("  scalability: done")


# --------------------------------------------------------------------------------------------------
# Reviewer 4: missing-modality robustness, out-of-distribution detection, class-prior reweighting,
# feature distributions
# --------------------------------------------------------------------------------------------------
MODALITY_GROUPS = {
    "Meteorological": ["rain", "precip", "temp", "humid", "wind", "pressure", "soil_moist"],
    "Hydrological": ["river"],
    "Terrain / land surface": ["slope", "dryness"],
    "Geophysical": ["seismic", "displacement"],
    "Exposure": ["population", "road", "risk_score", "historical_risk"],
    "Satellite water index": ["water_index"],
    "Sentinel-1 SAR": ["vv", "vh"],
    "Sentinel-2 optical": ["ndvi", "ndwi"],
}


def run_r4(out, df, feats, X, y, n_cls, class_names, tr, va, te, pre, clf, priors_path):
    # ---- (a) missing-modality robustness: remove one modality at inference (set to training median) ----
    base = metrics(y[te], clf.predict_proba(pre.transform(X[te])), n_cls)
    rows = [dict(modality_removed="none (all modalities)", accuracy=base["accuracy"], macro_f1=base["macro_f1"])]
    for name, pats in MODALITY_GROUPS.items():
        cols = match_cols(feats, pats)
        if not cols:
            continue
        Xm = X[te].copy()
        Xm[:, [feats.index(c) for c in cols]] = np.nan       # imputer replaces with training medians
        m = metrics(y[te], clf.predict_proba(pre.transform(Xm)), n_cls)
        rows.append(dict(modality_removed=f"{name} ({', '.join(cols)})", accuracy=m["accuracy"],
                         macro_f1=m["macro_f1"], accuracy_drop=base["accuracy"] - m["accuracy"]))
    pd.DataFrame(rows).to_csv(out / "missing_modality_robustness.csv", index=False)
    print("  missing-modality robustness: done")

    # ---- (b) out-of-distribution detection: leave one hazard class out of training ----
    ood = []
    for c in range(n_cls):
        tr_c, va_c = tr[y[tr] != c], va[y[va] != c]
        keep = np.array([k for k in range(n_cls) if k != c])
        remap = {k: i for i, k in enumerate(keep)}
        pre_c = Pipeline(make_pre()).fit(X[tr_c])
        m_c = TorchRadiant(seed=0).fit(pre_c.transform(X[tr_c]), np.vectorize(remap.get)(y[tr_c]),
                                       pre_c.transform(X[va_c]), np.vectorize(remap.get)(y[va_c]))
        P = m_c.predict_proba(pre_c.transform(X[te]))
        is_ood = (y[te] == c).astype(int)
        ood.append(dict(held_out_class=class_names[c],
                        auroc_entropy=roc_auc_score(is_ood, entropy(P)),
                        auroc_max_softmax=roc_auc_score(is_ood, -P.max(1)),
                        mean_max_softmax_ood=P[is_ood == 1].max(1).mean(),
                        mean_max_softmax_in=P[is_ood == 0].max(1).mean()))
    pd.DataFrame(ood).to_csv(out / "ood_leave_one_class_out.csv", index=False)
    print("  OOD detection: done")

    # ---- (c) performance under non-uniform (operational) class priors ----
    pred = clf.predict(pre.transform(X[te]))
    cm = confusion_matrix(y[te], pred, labels=list(range(n_cls))).astype(float)
    rates = cm / cm.sum(1, keepdims=True)                    # P(pred = j | true = i)
    pri = []
    prior_sets = {"balanced (test set)": np.full(n_cls, 1 / n_cls)}
    if priors_path:
        user = json.load(open(priors_path))
        v = np.array([float(user[name]) for name in class_names])
        prior_sets["user-supplied priors"] = v / v.sum()
    for label, pi in prior_sets.items():
        joint = pi[:, None] * rates                          # P(true = i, pred = j)
        recall = np.diag(rates)
        precision = np.diag(joint) / np.clip(joint.sum(0), 1e-12, None)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        row = dict(prior=label, accuracy=float(np.trace(joint)), macro_f1=float(f1.mean()))
        row.update({f"precision_{class_names[k]}": precision[k] for k in range(n_cls)})
        row.update({f"prior_{class_names[k]}": pi[k] for k in range(n_cls)})
        pri.append(row)
    pd.DataFrame(pri).to_csv(out / "prior_reweighted_performance.csv", index=False)
    print("  prior reweighting: done" + ("" if priors_path else " (add --priors priors.json for operational priors)"))

    # ---- (d) feature distributions by class (Supplementary Fig. S1) ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ncol = 4; nrow = int(math.ceil(len(feats) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
        for ax, f in zip(axes.ravel(), feats):
            data = [df.loc[y == k, f].dropna().values for k in range(n_cls)]
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, n_cls + 1)); ax.set_xticklabels([c[:5] for c in class_names], fontsize=7)
            ax.set_title(f, fontsize=9)
        for ax in axes.ravel()[len(feats):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(out / "supplementary_feature_distributions.png", dpi=200); plt.close(fig)
        print("  feature distributions: done")
    except Exception as e:
        print("  feature-distribution plot skipped:", e)


# --------------------------------------------------------------------------------------------------
# Reviewer 5: systematic stress tests and repeated timing
# --------------------------------------------------------------------------------------------------
def run_r5(out, feats, X, y, n_cls, te, pre, clf, n_rep=10):
    rng = np.random.default_rng(0)
    Xte = X[te]
    sd = np.nanstd(X, 0)
    rows = []
    # (a) random missingness: a fraction of feature values set missing (imputed with training medians)
    for frac in [0.0, 0.1, 0.2, 0.3, 0.5]:
        accs = []
        for r in range(n_rep):
            Xm = Xte.copy()
            Xm[rng.random(Xm.shape) < frac] = np.nan
            accs.append(accuracy_score(y[te], clf.predict(pre.transform(Xm))))
        rows.append(dict(test="random missing values", level=frac, accuracy_mean=np.mean(accs), accuracy_sd=np.std(accs, ddof=1)))
    # (b) sensor noise: Gaussian noise with SD = level x feature SD
    for lev in [0.0, 0.1, 0.25, 0.5, 1.0]:
        accs = []
        for r in range(n_rep):
            Xn = Xte + rng.normal(0, 1, Xte.shape) * sd * lev
            accs.append(accuracy_score(y[te], clf.predict(pre.transform(Xn))))
        rows.append(dict(test="Gaussian sensor noise (x feature SD)", level=lev, accuracy_mean=np.mean(accs), accuracy_sd=np.std(accs, ddof=1)))
    # (c) conflicting sources: one modality group replaced by values from an event of a DIFFERENT class
    for name, pats in MODALITY_GROUPS.items():
        cols = [feats.index(c) for c in match_cols(feats, pats)]
        if not cols:
            continue
        accs = []
        for r in range(n_rep):
            Xc = Xte.copy()
            for i in range(len(te)):
                donors = te[y[te] != y[te][i]]
                Xc[i, cols] = X[rng.choice(donors), cols]
            accs.append(accuracy_score(y[te], clf.predict(pre.transform(Xc))))
        rows.append(dict(test="conflicting source (modality from another hazard class)", level=name,
                         accuracy_mean=np.mean(accs), accuracy_sd=np.std(accs, ddof=1)))
    pd.DataFrame(rows).to_csv(out / "stress_tests.csv", index=False)
    print("  stress tests: done")

    # (d) repeated timing under identical conditions (batch size 1, forward pass only)
    xb = torch.tensor(pre.transform(Xte[:1]), dtype=torch.float32)
    clf.model_.eval()
    reps = []
    with torch.no_grad():
        for _ in range(50): clf.model_(xb)
        for r in range(20):
            t0 = time.perf_counter()
            for _ in range(500): clf.model_(xb)
            reps.append((time.perf_counter() - t0) / 500 * 1000)
    # end-to-end per-event time: preprocessing (imputation + scaling) + forward pass + argmax
    e2e = []
    for r in range(20):
        t0 = time.perf_counter()
        for _ in range(200):
            clf.predict(pre.transform(Xte[:1]))
        e2e.append((time.perf_counter() - t0) / 200 * 1000)
    pd.DataFrame([dict(measure="forward pass only (ms/event)", mean=np.mean(reps), sd=np.std(reps, ddof=1), repetitions=20),
                  dict(measure="preprocessing + forward + decision (ms/event)", mean=np.mean(e2e), sd=np.std(e2e, ddof=1), repetitions=20)]
                 ).to_csv(out / "timing_repeated.csv", index=False)
    print("  repeated timing: done")

# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def to_md(df, **kw):
    try:
        return df.to_markdown(index=False, **kw)
    except ImportError:  # 'tabulate' not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--event-col")
    ap.add_argument("--region-col")
    ap.add_argument("--date-col")
    ap.add_argument("--temporal-cutoff")
    ap.add_argument("--features", nargs="*", help="explicit feature columns (default: all numeric)")
    ap.add_argument("--groups", help="JSON file mapping group name -> list of column-name patterns")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--region-folds", type=int, default=10)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-extras", action="store_true", help="skip SHAP / uncertainty / scalability / R4 analyses")
    ap.add_argument("--priors", help="JSON file {class name: relative frequency} for operational-prior evaluation")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # read grouping columns as raw strings so codes like "NA" (North America) are not parsed as missing
    gcols = [c for c in [args.event_col, args.region_col, args.date_col, args.label] if c]
    df = pd.read_csv(args.data, converters={c: str for c in gcols})
    meta = {c for c in [args.label, args.event_col, args.region_col, args.date_col] if c}
    feats = args.features or [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    groups = json.load(open(args.groups)) if args.groups else DEFAULT_GROUPS
    if "modality" in groups:
        MODALITY_GROUPS.clear(); MODALITY_GROUPS.update(groups["modality"])
    le = LabelEncoder()
    y = le.fit_transform(df[args.label].astype(str))
    X = df[feats].to_numpy(dtype=float)
    n_cls = len(le.classes_)
    print(f"{len(df)} samples, {len(feats)} features, classes: {list(le.classes_)}")
    print("features:", feats)

    # ---- Reviewer 3: dataset description (Table 3 missing rates; Supplementary Table S2) ----
    desc = pd.DataFrame({
        "feature": feats,
        "missing_pct": [100 * df[c].isna().mean() for c in feats],
        "mean": [df[c].mean() for c in feats], "sd": [df[c].std() for c in feats],
        "min": [df[c].min() for c in feats], "max": [df[c].max() for c in feats],
    })
    _, tr0, _, _ = next(split_generator("repeated_stratified", df, y, args))
    pre0 = Pipeline(make_pre()).fit(X[tr0])
    desc["train_median_imputer"] = pre0.named_steps["impute"].statistics_
    desc["train_mean_scaler"] = pre0.named_steps["scale"].mean_
    desc["train_sd_scaler"] = pre0.named_steps["scale"].scale_
    desc.to_csv(out / "supplementary_feature_description.csv", index=False)
    pd.crosstab(df[args.label], df[args.region_col] if args.region_col else "all").to_csv(out / "class_by_region.csv")
    print("feature description written (missing %, stats, fitted preprocessing parameters)")

    if torch is not None:
        probe = TorchRadiant().fit(StandardScaler().fit_transform(np.nan_to_num(X[:64])), y[:64])
        print(f"RADIANT-Net v1 trainable parameters: {probe.n_params():,} (paper: 45,189 -> edit RadiantNetV1 if different)")

    protocols = ["repeated_stratified"]
    if args.event_col: protocols.append("event_grouped")
    if args.region_col: protocols.append("leave_region_out")
    if args.date_col and args.temporal_cutoff: protocols.append("temporal_holdout")
    if os.environ.get("ONLY_PROT"):
        protocols = [p for p in protocols if p in os.environ["ONLY_PROT"].split(",")]

    models = [(k, "radiant", v) for k, v in RADIANT_VARIANTS.items()]
    if not args.skip_baselines:
        models += [(k, "radiant", v) for k, v in DEEP_BASELINES.items()]
        models += [(k, "sk", v) for k, v in baseline_grids().items()]

    ckpt = out / "results_per_split.csv"
    rows = pd.read_csv(ckpt).to_dict("records") if ckpt.exists() else []
    done = {(r["protocol"], int(r["split"]), r["model"], r["feature_set"]) for r in rows}
    if rows:
        print(f"resuming: {len(rows)} results already saved")

    def save(m):
        rows.append(m)
        pd.DataFrame(rows).to_csv(ckpt, index=False)

    for prot in protocols:
        print(f"\n=== protocol: {prot} ===")
        for split_id, tr, va, te in split_generator(prot, df, y, args):
            for name, kind, spec in models:
                if (prot, split_id, name, "all") in done:
                    continue
                m, _, _ = fit_eval(name, kind, spec, X, y, tr, va, te, seed=split_id, n_cls=n_cls,
                                   inner_cv_seed=split_id)
                m.update(protocol=prot, split=split_id, feature_set="all")
                save(m)
                print(f"  split {split_id:>2} | {name:<36} acc={m['accuracy']:.3f} F1={m['macro_f1']:.3f}")

    if os.environ.get("LOOP_ONLY"):
        print("LOOP_ONLY: done"); return
    # ----- input-group ablations (repeated stratified only, full RADIANT-Net) -----
    if "subsets" in groups:
        subsets = {}
        for gname, pats in groups["subsets"].items():
            cols = match_cols(feats, pats)
            subsets[f"{gname} only"] = cols
            subsets[f"Without {gname.lower()}"] = [f for f in feats if f not in cols]
        print("\nsubsets:", subsets)
    else:
        met = match_cols(feats, groups["meteorological"])
        sat = match_cols(feats, groups["satellite"])
        prox = match_cols(feats, groups["proxies"])
        print("\nmeteorological:", met, "\nsatellite:", sat, "\npotential proxies:", prox)
        subsets = {
            "Meteorological features only": met,
            "Satellite-derived features only": sat,
            "Without satellite-derived features": [f for f in feats if f not in sat],
            "Without potential label proxies": [f for f in feats if f not in prox],
        }
    for sname, cols in subsets.items():
        if not cols:
            print(f"  skipped '{sname}' (no matching columns; edit --groups)")
            continue
        ci = [feats.index(c) for c in cols]
        for split_id, tr, va, te in split_generator("repeated_stratified", df, y, args):
            if ("repeated_stratified", split_id, "RADIANT-Net v1 (full)", sname) in done:
                continue
            m, _, _ = fit_eval("RADIANT-Net v1 (full)", "radiant", {}, X[:, ci], y, tr, va, te, split_id, n_cls)
            m.update(protocol="repeated_stratified", split=split_id, feature_set=sname)
            save(m)
        print(f"  {sname}: done")

    res = pd.DataFrame(rows)

    # ----- summaries -----
    def summarise(g):
        n = len(g)
        d = {}
        for k in ["accuracy", "macro_f1", "macro_auc", "ece", "brier"]:
            v = g[k].dropna()
            mu, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
            half = stats.t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v)) if len(v) > 1 else np.nan
            d[k] = f"{mu:.3f} ± {sd:.3f}" + (f" [{mu - half:.3f}, {mu + half:.3f}]" if not np.isnan(half) else "")
        d["n_splits"] = n
        return pd.Series(d)

    summ = res.groupby(["protocol", "feature_set", "model"], sort=False).apply(summarise).reset_index()
    summ.to_csv(out / "summary.csv", index=False)

    # paired tests vs full RADIANT-Net (same protocol, same splits)
    tests = []
    for prot in res.protocol.unique():
        base = res[(res.protocol == prot) & (res.feature_set == "all") & (res.model == "RADIANT-Net v1 (full)")]
        base = base.sort_values("split")
        for mname in res[(res.protocol == prot) & (res.feature_set == "all")].model.unique():
            if mname == "RADIANT-Net v1 (full)":
                continue
            other = res[(res.protocol == prot) & (res.feature_set == "all") & (res.model == mname)].sort_values("split")
            a, b = other.macro_f1.values, base.macro_f1.values
            if len(a) != len(b):
                continue
            ntr, nte = base.n_train.mean(), base.n_test.mean()
            tests.append(dict(protocol=prot, model=mname, delta_macro_f1=float(np.mean(a - b)),
                              p_corrected_t=corrected_t(a, b, ntr, nte), p_wilcoxon=wilcoxon_p(a, b)))
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "paired_tests.csv", index=False)

    # permutation importance on one representative split (full model)
    if torch is not None:
        split_id, tr, va, te = next(split_generator("repeated_stratified", df, y, args))
        _, _, (pre, clf) = fit_eval("RADIANT-Net v1 (full)", "radiant", {}, X, y, tr, va, te, split_id, n_cls)
        pi = permutation_importance(clf, pre.transform(X[te]), y[te], scoring="f1_macro", n_repeats=30,
                                    random_state=0)
        imp = pd.DataFrame({"feature": feats, "importance_mean": pi.importances_mean,
                            "importance_sd": pi.importances_std}).sort_values("importance_mean", ascending=False)
        imp.to_csv(out / "permutation_importance.csv", index=False)

        # inference latency (CPU, batch 1, after warm-up)
        xb = torch.tensor(pre.transform(X[te[:1]]), dtype=torch.float32)
        clf.model_.eval()
        with torch.no_grad():
            for _ in range(50): clf.model_(xb)
            t0 = time.perf_counter(); R = 2000
            for _ in range(R): clf.model_(xb)
        lat = (time.perf_counter() - t0) / R * 1000
        (out / "latency.txt").write_text(f"mean inference latency, batch size 1, CPU: {lat:.4f} ms over {R} runs\n")

        if not args.skip_extras:
            run_extras(out, feats, X, y, n_cls, tr, va, te, pre, clf, split_id)
            run_r4(out, df, feats, X, y, n_cls, list(le.classes_), tr, va, te, pre, clf, args.priors)
            run_r5(out, feats, X, y, n_cls, te, pre, clf)

    # Markdown tables for the manuscript
    md = ["# Results for the revised manuscript\n"]
    for prot in summ.protocol.unique():
        md.append(f"\n## Protocol: {prot}\n")
        sub = summ[(summ.protocol == prot)]
        md.append(to_md(sub.drop(columns="protocol")))
    if len(tests):
        md.append("\n## Paired tests vs RADIANT-Net v1 (macro-F1)\n")
        md.append(to_md(tests, floatfmt=".4f"))
    (out / "tables_for_manuscript.md").write_text("\n".join(md))
    print(f"\nDone. See {out}/tables_for_manuscript.md")


if __name__ == "__main__":
    main()
