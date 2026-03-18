"""
Diagnostic: evaluate Ollama LLM accuracy on labeled businesses.
Tests the rewritten "would I call?" prompt directly against all labels.
Uses local Ollama (free, no DataForSEO credits). Runs 6 concurrent requests.
"""
import csv, json, os, re, sys, time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding="utf-8")

from app import _clean_trainer_review_text

PARALLEL = 6  # concurrent Ollama requests

# ── Load labels ──────────────────────────────────────────────────────

rows = []
with open("review_training_labels.csv", "r", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        label = str(r.get("would_call") or "").strip().lower()
        if label in ("yes", "no"):
            rows.append(r)

# Deduplicate
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

print(f"Loaded {len(rows)} labeled businesses ({yes_n} YES, {no_n} NO)")

# ── Ollama prompt ────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are qualifying small businesses for cold-calling. Question: if I call "{business_name}", will I reach the actual OWNER — the person who built the business and makes all decisions?

Think about it like this: is there ONE specific identifiable person who IS this business?

STRONG YES signals (need at least one):
- Business is named after a person AND that person appears in reviews doing the work (e.g. "Casey's Plumbing" and reviews mention Casey fixing things)
- The same person's name appears across many reviews as THE person customers deal with directly — not just one mention
- Reviewers say things like "the owner [name]", "he/she runs the whole operation", or treat one person as THE face of everything
- Solo or micro operation where one person clearly does all the work

STRONG NO signals (any of these = NO):
- Different employees/technicians show up each time (multiple different names, rotating staff)
- Franchise or chain brand (e.g. "Mr. Electric", "1-800", regional chain names)
- Business has a call center, dispatcher, or coordinator — you'd never reach the owner directly
- Reviewers say "they sent a technician", "the crew showed up", "the team was great"
- Name appears but only as one of many employees, not as THE owner
- Company-style name with no personal connection (e.g. "Atlanta HVAC Pros", "Top Locksmith")

EDGE CASES:
- A person is named in reviews but different people rotate → NO (employees, not the owner)
- Business has 2-3 helpers but ONE owner clearly runs everything → YES
- Nice personal service but no specific owner emerges → NO

REVIEWS:
{reviews}

Output ONLY a JSON object: {{"would_call": true or false, "person_name": "Name" or null, "reason": "one sentence"}}"""


def call_ollama(prompt: str) -> dict | None:
    """Call Ollama and parse JSON response."""
    import urllib.request
    data = json.dumps({
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("response", "").strip()
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```\s*$", "", content)
            m = re.search(r"\{[^{}]*\}", content)
            if m:
                return json.loads(m.group())
            return json.loads(content)
    except Exception as e:
        print(f"    Ollama error: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None


def get_reviews_text(r, max_chars=4000) -> str:
    try:
        review_list = json.loads(r.get("reviews_json") or "[]")
        if isinstance(review_list, list):
            clean = [_clean_trainer_review_text(str(x)) for x in review_list]
            text = "\n---\n".join(c for c in clean if c)
            return text[:max_chars]
    except Exception:
        pass
    return _clean_trainer_review_text(str(r.get("reviews_json") or ""))[:max_chars]


def process_row(args):
    idx, r = args
    biz_name = r.get("business_name", "???")
    truth = r["would_call"].strip().lower()
    reviews = get_reviews_text(r)
    prompt = PROMPT_TEMPLATE.format(business_name=biz_name, reviews=reviews)
    result = call_ollama(prompt)
    return idx, biz_name, truth, result


# ── Evaluate ─────────────────────────────────────────────────────────

print(f"\nRunning Ollama evaluation on {len(rows)} businesses ({PARALLEL} parallel)...\n")

correct = 0
total = 0
fn_cases = []
fp_cases = []
errors = 0
print_lock = threading.Lock()

start = time.time()

results = {}
with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
    futures = {executor.submit(process_row, (i, r)): i for i, r in enumerate(rows)}
    done = 0
    for future in as_completed(futures):
        idx, biz_name, truth, result = future.result()
        results[idx] = (biz_name, truth, result)
        done += 1
        print(f"  [{done:>3}/{len(rows)} done] {biz_name[:40]}", flush=True)

print("\n--- Final results in order ---\n", flush=True)
for idx in sorted(results):
    biz_name, truth, result = results[idx]
    if result is None:
        errors += 1
        print(f"  [{idx+1:>3}/{len(rows)}] [ERROR] {biz_name[:40]}", flush=True)
        continue

    would_call = bool(result.get("would_call", False))
    confidence = float(result.get("confidence", 0.5))
    person = result.get("person_name")
    reason = str(result.get("reason", ""))[:80]

    pred = "yes" if would_call else "no"
    ok = pred == truth
    if ok:
        correct += 1
    total += 1

    if not ok:
        if truth == "yes":
            fn_cases.append((biz_name, confidence, person, reason))
        else:
            fp_cases.append((biz_name, confidence, person, reason))

    status = f"[WRONG]" if not ok else f"[  OK ]"
    print(f"  [{idx+1:>3}/{len(rows)}] {status} {biz_name[:40]:<40} truth={truth} pred={pred} name={person}", flush=True)

elapsed = time.time() - start

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"QWEN2.5:7B EVALUATION RESULTS")
print(f"{'='*70}")
acc = correct / total if total else 0
print(f"Accuracy: {correct}/{total} = {acc*100:.1f}%")
print(f"Time: {elapsed:.0f}s ({elapsed/len(rows):.1f}s/business)")
print(f"Errors (failed to parse): {errors}")
print(f"False Negatives (missed leads): {len(fn_cases)}")
print(f"False Positives (wasted calls): {len(fp_cases)}")

if fn_cases:
    print(f"\nFALSE NEGATIVES — Missed Leads (truth=YES, pred=NO):")
    for name, conf, person, reason in fn_cases:
        print(f"  {name[:50]:<50} name={person} reason={reason}")

if fp_cases:
    print(f"\nFALSE POSITIVES — Wasted Calls (truth=NO, pred=YES):")
    for name, conf, person, reason in fp_cases:
        print(f"  {name[:50]:<50} name={person} reason={reason}")

print(f"\nDone.")
