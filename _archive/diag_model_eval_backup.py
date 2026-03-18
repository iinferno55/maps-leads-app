"""
Diagnostic: evaluate the review preference model via LOO cross-validation.
Safe to run anytime - read-only, no data is modified.
"""
import csv, json, math, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")

from app import _tokenize_for_training as tokenize, _extract_meta_features as meta_features

# Niche detection (mirrors app.py training logic)
_NICHE_KEYWORDS = {
    "detailing": "detailing", "detailer": "detailing", "detail": "detailing",
    "auto detail": "detailing", "car wash": "detailing", "car care": "detailing",
    "plumb": "plumbing", "plumber": "plumbing", "plumbing": "plumbing",
    "sewer": "plumbing", "drain": "plumbing",
    "floor": "flooring", "flooring": "flooring", "carpet": "flooring", "tile": "flooring",
    "pressure wash": "pressure_washing", "power wash": "pressure_washing",
    "roof": "roofing", "roofing": "roofing",
    "hvac": "hvac", "heating": "hvac", "air condition": "hvac",
    "electric": "electrical", "electrician": "electrical",
    "paint": "painting", "painter": "painting",
    "landscap": "landscaping", "lawn": "landscaping", "mow": "landscaping",
    "clean": "cleaning", "maid": "cleaning", "janitorial": "cleaning",
}

def _detect_niche(biz_name: str) -> str:
    low = biz_name.lower()
    for keyword, niche in _NICHE_KEYWORDS.items():
        if keyword in low:
            return niche
    return "other"

def get_reviews_raw(r):
    try:
        review_list = json.loads(r.get("reviews_json") or "[]")
        return " ".join(str(x) for x in review_list) if isinstance(review_list, list) else str(review_list)
    except Exception:
        return str(r.get("reviews_json") or "")

def clamp(val, lo=0.05, hi=0.95):
    return max(lo, min(hi, val))

def build_model(training_rows):
    from collections import Counter

    # Niche-aware weighting
    n_raw = len(training_rows)
    niche_counts = {}
    row_niches = []
    for r in training_rows:
        niche = _detect_niche(str(r.get("business_name") or ""))
        row_niches.append(niche)
        niche_counts[niche] = niche_counts.get(niche, 0) + 1
    n_niches = max(1, len(niche_counts))
    ideal_per_niche = n_raw / n_niches
    niche_weights = {niche: min(1.0, ideal_per_niche / cnt) for niche, cnt in niche_counts.items()}

    pos_docs, neg_docs = [], []
    for idx, r in enumerate(training_rows):
        label = r["would_call"].strip().lower()
        reviews_raw = get_reviews_raw(r)
        niche_w = niche_weights.get(row_niches[idx], 1.0)

        # Feature selection: meta-only for small datasets, full for large
        meta_toks = meta_features(reviews_raw)
        if n_raw >= 100:
            raw_toks = tokenize(reviews_raw)
            doc_toks_raw = raw_toks + meta_toks
        else:
            doc_toks_raw = meta_toks
        if not doc_toks_raw:
            continue

        tok_counts = Counter(doc_toks_raw)
        doc_weights = {t: (1.0 + math.log(cnt)) * niche_w for t, cnt in tok_counts.items()}

        # Highlight tokens: meta-only for small datasets, full for large
        hl_tok_list = []
        try:
            hl_list = json.loads(r.get("highlighted_evidence_json") or "[]")
            if isinstance(hl_list, list):
                hl_text = " ".join(str(h.get("text", "")) for h in hl_list if isinstance(h, dict))
                if hl_text.strip():
                    if n_raw >= 100:
                        hl_tok_list.extend(tokenize(hl_text))
                    hl_tok_list.extend(meta_features(hl_text))
        except Exception:
            pass

        # Name tokens: only for large datasets
        name_tok_list = []
        owner_guess = str(r.get("owner_name_guess") or "").strip()
        if owner_guess and n_raw >= 100:
            name_tok_list.extend(tokenize(owner_guess))

        hl_counts = Counter(hl_tok_list)
        hl_weights = {t: 1.0 + math.log(cnt) for t, cnt in hl_counts.items()} if hl_counts else {}
        name_counts_local = Counter(name_tok_list)
        name_weights = {t: 1.0 + math.log(cnt) for t, cnt in name_counts_local.items()} if name_counts_local else {}

        if label == "yes":
            pos_docs.append(doc_weights)
            if hl_weights:
                pos_docs.append(hl_weights)
                pos_docs.append(hl_weights)
            if name_weights:
                for _ in range(4):
                    pos_docs.append(name_weights)
        else:
            neg_docs.append(doc_weights)
            if hl_weights:
                neg_docs.append(hl_weights)
                neg_docs.append(hl_weights)
            if name_weights:
                for _ in range(4):
                    neg_docs.append(name_weights)

    if not pos_docs or not neg_docs:
        return None

    pos_counts, neg_counts = {}, {}
    for doc in pos_docs:
        for t, w in doc.items():
            pos_counts[t] = pos_counts.get(t, 0.0) + w
    for doc in neg_docs:
        for t, w in doc.items():
            neg_counts[t] = neg_counts.get(t, 0.0) + w
    vocab = sorted(set(pos_counts) | set(neg_counts))
    v = max(1, len(vocab))
    alpha = 1.0
    pos_total = sum(pos_counts.values())
    neg_total = sum(neg_counts.values())
    MAX_TOKEN_LOG_ODDS = min(1.5, 0.5 + n_raw / 100.0)
    tlo = {}
    for t in vocab:
        p_pos = (pos_counts.get(t, 0) + alpha) / (pos_total + alpha * v)
        p_neg = (neg_counts.get(t, 0) + alpha) / (neg_total + alpha * v)
        raw_lo = math.log(p_pos / p_neg)
        tlo[t] = max(-MAX_TOKEN_LOG_ODDS, min(MAX_TOKEN_LOG_ODDS, raw_lo))
    prior = 0.0
    return {"tlo": tlo, "prior": prior, "n_raw_labels": n_raw}

def score(r, m):
    reviews_raw = get_reviews_raw(r)
    # Match training feature set based on dataset size
    n_labels = m.get("n_raw_labels", 0)
    if n_labels >= 100:
        toks = set(tokenize(reviews_raw))
        toks.update(meta_features(reviews_raw))
    else:
        toks = set(meta_features(reviews_raw))
    logit = m["prior"]
    for t in toks:
        if t in m["tlo"]:
            logit += m["tlo"][t]
    logit = max(-20.0, min(20.0, logit))
    return clamp(1.0 / (1.0 + math.exp(-logit)))

# ---- Load labels ----
rows = []
with open("review_training_labels.csv", "r", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        label = str(r.get("would_call") or "").strip().lower()
        if label in ("yes", "no"):
            rows.append(r)

yes_n = sum(1 for r in rows if r["would_call"].strip().lower() == "yes")
no_n = sum(1 for r in rows if r["would_call"].strip().lower() == "no")

print("=" * 60)
print("REVIEW TRAINER DIAGNOSTIC")
print("=" * 60)
print(f"Total labeled examples : {len(rows)}")
print(f"  YES (would call)     : {yes_n}")
print(f"  NO  (would skip)     : {no_n}")

# Niche distribution
niche_dist = {}
for r in rows:
    n = _detect_niche(str(r.get("business_name") or ""))
    niche_dist[n] = niche_dist.get(n, 0) + 1
print(f"\nNiche distribution:")
for niche, cnt in sorted(niche_dist.items(), key=lambda x: -x[1]):
    print(f"  {niche:<20} : {cnt}")

# ---- Load saved model ----
with open("review_trainer_model.json", "r", encoding="utf-8") as f:
    saved = json.load(f)
sm = {"tlo": saved["token_log_odds"], "prior": saved["prior_log_odds"]}
print(f"\nSaved model trained    : {saved['trained_at_utc']}")
print(f"  n_yes={saved['n_yes']}, n_no={saved['n_no']}, vocab={saved['vocab_size']}")

# ---- In-sample accuracy ----
print("\n--- In-sample accuracy (inflated - model saw its own data) ---")
print(f"{'Business':<40} {'Truth':>5} {'Prob':>6} {'OK':>3}")
print("-" * 60)
correct_in = 0
for r in rows:
    prob = score(r, sm)
    truth = r["would_call"].strip().lower()
    pred = "yes" if prob >= 0.5 else "no"
    ok = "Y" if pred == truth else "N"
    if pred == truth:
        correct_in += 1
    print(f"{r.get('business_name','')[:39]:<40} {truth:>5} {prob:>6.2f}  {ok}")
print(f"\nIn-sample: {correct_in}/{len(rows)} = {correct_in/len(rows)*100:.0f}%")

# ---- Leave-One-Out Cross Validation ----
print("\n--- Leave-One-Out Cross-Validation (real generalization estimate) ---")
print(f"{'Business':<40} {'Truth':>5} {'Prob':>6} {'OK':>3}")
print("-" * 60)
correct_loo = 0
loo_total = 0
wrong_cases = []
for i, test_row in enumerate(rows):
    train_rows = [r for j, r in enumerate(rows) if j != i]
    m = build_model(train_rows)
    if m is None:
        continue
    prob = score(test_row, m)
    truth = test_row["would_call"].strip().lower()
    pred = "yes" if prob >= 0.5 else "no"
    ok = "Y" if pred == truth else "N"
    if pred == truth:
        correct_loo += 1
    else:
        wrong_cases.append((test_row.get("business_name", ""), truth, prob))
    loo_total += 1
    print(f"{test_row.get('business_name','')[:39]:<40} {truth:>5} {prob:>6.2f}  {ok}")

loo_acc = correct_loo / loo_total if loo_total else 0
print(f"\nLOO accuracy: {correct_loo}/{loo_total} = {loo_acc*100:.0f}%")

if wrong_cases:
    print("\nMisclassified (model got these wrong when not trained on them):")
    for name, truth, prob in wrong_cases:
        print(f"  [{truth.upper()}] {name[:50]} -> predicted {'YES' if prob>=0.5 else 'NO'} ({prob:.2f})")

# ---- Training scale estimate ----
print("\n" + "=" * 60)
print("TRAINING SCALE ESTIMATE")
print("=" * 60)
print(f"Current LOO accuracy: {loo_acc*100:.0f}%")
print()

if loo_acc >= 0.85:
    status = "GOOD - model generalizes well"
    target_per_class = 35
elif loo_acc >= 0.70:
    status = "FAIR - decent but noisy"
    target_per_class = 40
else:
    status = "WEAK - model is mostly guessing"
    target_per_class = 60

print(f"Status: {status}")
print()
print(f"For reliable 85%+ generalization on unseen listings,")
print(f"Naive Bayes text classifiers typically need ~{target_per_class} examples per class.")
more_yes = max(0, target_per_class - yes_n)
more_no = max(0, target_per_class - no_n)
print(f"  Need {target_per_class} YES: have {yes_n}, need {more_yes} more")
print(f"  Need {target_per_class} NO:  have {no_n}, need {more_no} more")
print(f"  Total labels to add: ~{more_yes + more_no}")
print()

# Labeling strategy advice
print("=" * 60)
print("LABELING STRATEGY FOR MAXIMUM IMPACT")
print("=" * 60)
underrep_niches = [n for n, c in niche_dist.items() if c <= 3]
if underrep_niches:
    print(f"Underrepresented niches (≤3 labels): {', '.join(underrep_niches)}")
    print("  → Labels from NEW niches teach the model to generalize!")
overrep_niches = [f"{n} ({c})" for n, c in sorted(niche_dist.items(), key=lambda x: -x[1]) if c >= 10]
if overrep_niches:
    print(f"Overrepresented niches: {', '.join(overrep_niches)}")
    print("  → Additional labels in these niches have diminishing returns.")
print()
print("Priority order for new labels:")
print("  1. NEW niches you haven't labeled yet (electricians, roofers, painters)")
print("  2. Niches with <5 labels (currently underrepresented)")
print("  3. Hard cases the model gets wrong (borderline businesses)")
print("  4. More of existing niches (lowest priority)")
