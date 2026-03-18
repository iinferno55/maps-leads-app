"""
Diagnostic: evaluate the review preference model via LOO cross-validation.
Tests multiple approaches including TF-IDF + combined features.
Safe to run anytime — read-only, no training data is modified.
"""
import csv, json, math, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

from app import (
    _extract_meta_features,
    _extract_numeric_features,
    _extract_per_review_features,
    _clean_trainer_review_text,
    _TRAINER_FEATURE_NAMES,
    _meta_features_to_vector,
)

# ── Helpers ──────────────────────────────────────────────────────────

def get_reviews_raw(r):
    try:
        review_list = json.loads(r.get("reviews_json") or "[]")
        if isinstance(review_list, list):
            clean = [_clean_trainer_review_text(str(x)) for x in review_list]
            return " ".join(c for c in clean if c)
    except Exception:
        pass
    return _clean_trainer_review_text(str(r.get("reviews_json") or ""))


# ── Load & deduplicate labels ────────────────────────────────────────

rows = []
with open("review_training_labels.csv", "r", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        label = str(r.get("would_call") or "").strip().lower()
        if label in ("yes", "no"):
            rows.append(r)

seen = {}
for r in rows:
    bname = str(r.get("business_name") or "").strip().lower()
    addr = str(r.get("address") or "").strip().lower()
    phone = str(r.get("phone") or "").strip()
    key = f"{bname}|{addr}" if addr else f"{bname}|{phone}"
    seen[key] = r
rows = list(seen.values())

yes_n = sum(1 for r in rows if r["would_call"].strip().lower() == "yes")
no_n = sum(1 for r in rows if r["would_call"].strip().lower() == "no")

print("=" * 70)
print("REVIEW TRAINER DIAGNOSTIC  (comprehensive model search)")
print("=" * 70)
print(f"Total labeled examples : {len(rows)}")
print(f"  YES (would call)     : {yes_n}")
print(f"  NO  (would skip)     : {no_n}")

# ── Build feature representations ────────────────────────────────────

reviews_texts = []
y = []
biz_names = []

for r in rows:
    reviews_raw = get_reviews_raw(r)
    reviews_texts.append(reviews_raw)
    label = 1 if r["would_call"].strip().lower() == "yes" else 0
    y.append(label)
    biz_names.append(r.get("business_name", "???"))

y = np.array(y)

# Meta+numeric+per-review features
binary_names = [f for f in _TRAINER_FEATURE_NAMES if not f.startswith("_NUM_") and not f.startswith("_PRV_")]
X_meta = []
for i, text in enumerate(reviews_texts):
    feats = _extract_meta_features(text)
    nums = _extract_numeric_features(text)
    prv = _extract_per_review_features(rows[i].get("reviews_json") or "[]")
    nums.update(prv)
    vec = _meta_features_to_vector(feats, nums)
    X_meta.append(vec)
X_meta = np.array(X_meta)


# ── LOO evaluation helper ────────────────────────────────────────────

def run_loo_simple(X, y, make_clf, name):
    """LOO for numpy array features."""
    n = len(y)
    y_pred = np.zeros(n, dtype=int)
    y_proba = np.zeros(n)
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        clf = make_clf()
        clf.fit(X[train_mask], y[train_mask])
        y_pred[i] = clf.predict(X[i:i+1])[0]
        y_proba[i] = clf.predict_proba(X[i:i+1])[0][1]
    acc = (y_pred == y).sum() / n
    best_acc, best_thresh = 0, 0.5
    for t in [i/100 for i in range(20, 81)]:
        a = ((y_proba >= t).astype(int) == y).sum() / n
        if a > best_acc:
            best_acc, best_thresh = a, t
    fn = ((y_pred == 0) & (y == 1)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    return {"name": name, "acc": acc, "fn": fn, "fp": fp,
            "best_acc": best_acc, "best_thresh": best_thresh,
            "y_pred": y_pred, "y_proba": y_proba}


def run_loo_tfidf(texts, y, meta_X, tfidf_params, clf_factory, name):
    """LOO with TF-IDF + meta features combined."""
    n = len(y)
    y_pred = np.zeros(n, dtype=int)
    y_proba = np.zeros(n)
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        train_texts = [texts[j] for j in range(n) if train_mask[j]]
        test_text = [texts[i]]

        tfidf = TfidfVectorizer(**tfidf_params)
        X_train_tfidf = tfidf.fit_transform(train_texts)
        X_test_tfidf = tfidf.transform(test_text)

        # Combine TF-IDF with meta features
        X_train_combined = hstack([X_train_tfidf, csr_matrix(meta_X[train_mask])])
        X_test_combined = hstack([X_test_tfidf, csr_matrix(meta_X[i:i+1])])

        clf = clf_factory()
        clf.fit(X_train_combined, y[train_mask])
        y_pred[i] = clf.predict(X_test_combined)[0]
        y_proba[i] = clf.predict_proba(X_test_combined)[0][1]

    acc = (y_pred == y).sum() / n
    best_acc, best_thresh = 0, 0.5
    for t in [i/100 for i in range(20, 81)]:
        a = ((y_proba >= t).astype(int) == y).sum() / n
        if a > best_acc:
            best_acc, best_thresh = a, t
    fn = ((y_pred == 0) & (y == 1)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    return {"name": name, "acc": acc, "fn": fn, "fp": fp,
            "best_acc": best_acc, "best_thresh": best_thresh,
            "y_pred": y_pred, "y_proba": y_proba}


# ── Run evaluations ──────────────────────────────────────────────────

print("\nRunning LOO cross-validation for all configurations...")
print("(This may take a few minutes)\n")

results = []

# 1. Meta features only
for make_clf, cname in [
    (lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42), "LogReg C=0.1"),
    (lambda: LogisticRegression(C=0.5, max_iter=1000, random_state=42), "LogReg C=0.5"),
    (lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42), "LogReg C=1.0"),
    (lambda: RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42), "RF d=3"),
]:
    res = run_loo_simple(X_meta, y, make_clf, f"{cname} (meta+num)")
    results.append(res)

# 2. TF-IDF alone + TF-IDF combined with meta features
tfidf_configs = [
    {"max_features": 100, "min_df": 2, "max_df": 0.9, "ngram_range": (1, 1), "sublinear_tf": True},
    {"max_features": 200, "min_df": 2, "max_df": 0.9, "ngram_range": (1, 2), "sublinear_tf": True},
    {"max_features": 300, "min_df": 2, "max_df": 0.85, "ngram_range": (1, 2), "sublinear_tf": True},
    {"max_features": 50, "min_df": 2, "max_df": 0.9, "ngram_range": (1, 1), "sublinear_tf": True},
]

clf_configs = [
    (lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42, penalty="l1", solver="liblinear"), "LogReg L1 C=0.1"),
    (lambda: LogisticRegression(C=0.5, max_iter=1000, random_state=42, penalty="l1", solver="liblinear"), "LogReg L1 C=0.5"),
    (lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42, penalty="l1", solver="liblinear"), "LogReg L1 C=1.0"),
    (lambda: LogisticRegression(C=0.05, max_iter=1000, random_state=42, penalty="l1", solver="liblinear"), "LogReg L1 C=0.05"),
]

for ti, tparams in enumerate(tfidf_configs):
    desc = f"tfidf{tparams['max_features']}{'_bg' if tparams['ngram_range'][1]==2 else ''}"
    for clf_factory, cname in clf_configs:
        name = f"{cname} + {desc} + meta"
        res = run_loo_tfidf(reviews_texts, y, X_meta, tparams, clf_factory, name)
        results.append(res)
        # Also without meta
        zero_meta = np.zeros_like(X_meta)
        name2 = f"{cname} + {desc} (no meta)"
        res2 = run_loo_tfidf(reviews_texts, y, zero_meta, tparams, clf_factory, name2)
        results.append(res2)

# Sort by best threshold accuracy
results.sort(key=lambda r: -r["best_acc"])

print(f"\n{'='*70}")
print("MODEL COMPARISON (LOO Cross-Validation)")
print(f"{'='*70}")
print(f"\n{'Model':<55} {'LOO@0.5':>8} {'Best':>8} {'Thresh':>7} {'FN':>4} {'FP':>4}")
print("-" * 90)
for r in results[:25]:  # Show top 25
    print(f"  {r['name']:<53} {r['acc']*100:>7.1f}% {r['best_acc']*100:>7.1f}% {r['best_thresh']:>6.2f} {r['fn']:>4} {r['fp']:>4}")


# ── Detailed analysis of best model ──────────────────────────────────

best = results[0]
print(f"\n{'='*70}")
print(f"BEST MODEL: {best['name']}")
print(f"LOO accuracy: {best['acc']*100:.1f}% @ threshold=0.50")
print(f"Best accuracy: {best['best_acc']*100:.1f}% @ threshold={best['best_thresh']:.2f}")
print(f"{'='*70}")

y_proba_best = best["y_proba"]
y_pred_tuned = (y_proba_best >= best["best_thresh"]).astype(int)

print(f"\nConfusion Matrix (threshold={best['best_thresh']:.2f}):")
cm = confusion_matrix(y, y_pred_tuned)
tn, fp, fn, tp = cm.ravel()
print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"  False Negatives (MISSED LEADS): {fn}")
print(f"  False Positives (wasted calls): {fp}")

# Misclassified
wrong = []
for i in range(len(y)):
    if y_pred_tuned[i] != y[i]:
        truth = "YES" if y[i] == 1 else "NO"
        pred = "YES" if y_pred_tuned[i] == 1 else "NO"
        wrong.append((biz_names[i], truth, pred, y_proba_best[i], i))

fn_cases = [w for w in wrong if w[1] == "YES"]
fp_cases = [w for w in wrong if w[1] == "NO"]

if fn_cases:
    print(f"\nFALSE NEGATIVES — Missed Leads ({len(fn_cases)}):")
    for name, truth, pred, prob, idx in sorted(fn_cases, key=lambda x: x[3]):
        feats = _extract_meta_features(reviews_texts[idx])
        print(f"  [{truth}→{pred}] {name[:55]:<55} prob={prob:.2f}  feats={feats}")

if fp_cases:
    print(f"\nFALSE POSITIVES — Wasted Calls ({len(fp_cases)}):")
    for name, truth, pred, prob, idx in sorted(fp_cases, key=lambda x: -x[3]):
        feats = _extract_meta_features(reviews_texts[idx])
        print(f"  [{truth}→{pred}] {name[:55]:<55} prob={prob:.2f}  feats={feats}")


# ── Train full best model and show top TF-IDF features ───────────────

print(f"\n{'='*70}")
print("TOP TF-IDF FEATURES (full model, for understanding)")
print(f"{'='*70}")

# Train on all data with the TF-IDF config that appears most in top results
tfidf_full = TfidfVectorizer(max_features=200, min_df=2, max_df=0.9, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf_full = tfidf_full.fit_transform(reviews_texts)
X_combined_full = hstack([X_tfidf_full, csr_matrix(X_meta)])

lr_full = LogisticRegression(C=0.1, max_iter=1000, random_state=42, penalty="l1", solver="liblinear")
lr_full.fit(X_combined_full, y)

feature_names = tfidf_full.get_feature_names_out().tolist() + _TRAINER_FEATURE_NAMES
coefs = lr_full.coef_[0]
sorted_idx = np.argsort(np.abs(coefs))[::-1]

print(f"\n{'Feature':<40} {'Coef':>8} {'Dir':<6}")
print("-" * 56)
shown = 0
for idx in sorted_idx:
    if abs(coefs[idx]) > 0.01 and shown < 40:
        d = "YES" if coefs[idx] > 0 else "NO"
        print(f"  {feature_names[idx]:<38} {coefs[idx]:>8.3f} → {d}")
        shown += 1

print(f"\nDone.")
