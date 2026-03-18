"""
Fast prompt iteration on the hardest cases only (~30 businesses).
Cost: ~$0.02/run instead of $0.09 for the full eval.
Once prompt is good, run diag_gemini_eval.py for the full test.
"""
import csv, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from app import _clean_trainer_review_text

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.5-flash-lite"
PARALLEL = 10

# ── Hard cases — consistently wrong across prompt iterations ──────────
# These are the businesses the model keeps getting wrong.
# Fix these = fix the whole dataset.
HARD_CASES = {
    # FN — model says NO, truth=YES
    "Pioneer Plumbing & Sewer",           # Aaron personal calls, but 4 staff names
    "West Seattle Sewer & Drain Inc",      # Melissa runs it + Bill does work
    "Knockout Plumbing & Mechanical LLC",  # no clear name
    "Miami Mobile Car Wash",               # Norbert (new owner) + employees
    "Denver Flooring Collective",          # Andrew owner-operated flooring
    "Las Vegas Flooring",                  # Mike owner-operated flooring
    "Atlanta Floor Store",                 # Josh runs it
    "Atlanta Flooring Solutions",          # owner-operated flooring
    "Southern Roofing Company",            # Joey mentioned, user says YES
    "Modern Roofing",                      # no name, user says YES
    "JB Wholesale Flooring",               # wholesale flooring, user says YES
    "Elite Mobile Detailing Inc.",         # James with team
    # FP — model says YES, truth=NO
    "7Mondays Locksmith LLC",              # Bek — locksmith dispatcher
    "MO Locksmith Services",               # Mo — locksmith dispatcher
    "Top Locksmith",                       # Adam
    "GM Locksmith inc",                    # Amos
    "CallOrange Locksmith of Nashville Tennessee",
    "Krueger's Mobile Detailing - Seattle",# Lawrence+Diego — multiple workers
    "Perfection Mobile Detailing",         # Travis
    "Air Masters HVAC Inc",                # Jacob
    "INDOOR EXPERTS HEATING & AIR CONDITIONING",
    "Revived Rides Mobile Detailing",      # Ayden and Ben
    "Affordable Flooring & More",          # Mike owner-claimed but user NO
    "Floor Nashville",                     # Greg
    # Easy YES (make sure we don't break these)
    "Plunge It Sound LLC",                 # Casey — clear solo operator
    "Roto-Drain",                          # clear solo
    "Cynthia Mobile Detail",               # Cynthia — clear owner
    # Easy NO (make sure we don't break these)
    "Seattle Super Mobile Detailing",
    "L&M Mobile Ceramic Coating – Lake City WA",
}

# ── Load matching rows ────────────────────────────────────────────────
all_rows = []
with open("review_training_labels.csv", "r", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if str(r.get("would_call") or "").strip().lower() in ("yes", "no"):
            all_rows.append(r)

seen = {}
for r in all_rows:
    bname = str(r.get("business_name") or "").strip().lower()
    addr = str(r.get("address") or "").strip().lower()
    phone = str(r.get("phone") or "").strip()
    key = f"{bname}|{addr}" if addr else f"{bname}|{phone}"
    seen[key] = r
all_rows = list(seen.values())

hard_lower = {h.lower() for h in HARD_CASES}
rows = []
for r in all_rows:
    bname = r.get("business_name","").strip()
    if bname.lower() in hard_lower or any(h in bname.lower() or bname.lower() in h for h in hard_lower):
        rows.append(r)
print(f"Testing {len(rows)} hard cases (of {len(HARD_CASES)} targeted)\n")

# ── Prompt (edit this to iterate) ────────────────────────────────────

SYSTEM_INSTRUCTION = """Extract information from business reviews. Do NOT decide yes/no — just extract facts.

For each person's name mentioned in the reviews, report:
- name: the person's first name
- mentions: how many reviews mention this name
- role: "owner" if reviews say they own/run the business, "worker" if they do the hands-on work, "office" if they handle calls/scheduling/reception, "unknown" if unclear

Also report:
- plural_pronouns: count of reviews that use "they", "the team", "their crew", "these guys" to refer to the business
- singular_pronouns: count of reviews that use "he", "she", "him", "her" to refer to one person
- owner_mentioned: true if any review says "the owner", "owner-operated", "owns the business"
- gatekeeper: true if reviews mention "dispatcher", "receptionist", "front desk", "office staff", "answering service", "in the office"
- total_reviews: total number of reviews provided

Output ONLY JSON:
{"names": [{"name": "X", "mentions": N, "role": "owner|worker|office|unknown"}], "plural_pronouns": N, "singular_pronouns": N, "owner_mentioned": true/false, "gatekeeper": true/false, "total_reviews": N}"""

PROMPT_TEMPLATE = """Business: "{business_name}"

REVIEWS:
{reviews}

Extract facts as JSON. Do NOT decide would_call — just extract."""

# ── Run ───────────────────────────────────────────────────────────────

def call_gemini(prompt):
    try:
        full_prompt = SYSTEM_INSTRUCTION + "\n\n" + prompt
        r = client.models.generate_content(
            model=MODEL, contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=1024,
            ),
        )
        if not r.text:
            if r.candidates:
                c = r.candidates[0]
                print(f"  DEBUG: finish={c.finish_reason}, safety={c.safety_ratings}", flush=True)
            else:
                print(f"  DEBUG: no candidates", flush=True)
            return None
        content = r.text.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"  DEBUG: JSON parse failed: [{content[:300]}]", flush=True)
            return None
        return data
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:150]}", flush=True)
    return None


def decide_would_call(extraction: dict) -> tuple[bool, str | None, str]:
    """Deterministic decision based on extracted features."""
    if not extraction:
        return False, None, "extraction failed"

    names = extraction.get("names", [])
    plural = extraction.get("plural_pronouns", 0)
    singular = extraction.get("singular_pronouns", 0)
    owner_mentioned = extraction.get("owner_mentioned", False)
    gatekeeper = extraction.get("gatekeeper", False)
    total_reviews = extraction.get("total_reviews", 0)

    # Filter to names with at least some presence
    named = [n for n in names if n.get("mentions", 0) >= 1]
    unique_count = len(named)

    # Find the top name
    top = max(named, key=lambda x: x.get("mentions", 0)) if named else None
    top_name = top["name"] if top else None
    top_mentions = top.get("mentions", 0) if top else 0
    top_role = top.get("role", "unknown") if top else "unknown"

    # Count owners vs workers
    owner_names = [n for n in named if n.get("role") == "owner"]
    worker_names = [n for n in named if n.get("role") == "worker"]
    office_names = [n for n in named if n.get("role") == "office"]

    # Compute dominance: does the top name have way more mentions than others?
    second = sorted(named, key=lambda x: x.get("mentions", 0), reverse=True)[1] if len(named) >= 2 else None
    second_mentions = second.get("mentions", 0) if second else 0
    dominant = top_mentions >= 3 and top_mentions >= second_mentions * 2  # top has 2x+ the second

    # Rule 1: No names at all → NO
    if unique_count == 0:
        return False, None, "no names mentioned"

    # Rule 2: Very dominant name (5+ mentions, second ≤1) → YES
    very_dominant = top_mentions >= 5 and second_mentions <= 1
    if very_dominant:
        return True, top_name, f"very dominant name ({top_mentions} mentions)"

    # Rule 3: 4+ different names → staffed → NO
    if unique_count >= 4:
        # Exception: one name has 4+ mentions and next highest has 1 → still reachable
        if top_mentions >= 4 and second_mentions <= 1:
            return True, top_name, f"dominant name ({top_mentions}x) despite {unique_count} names"
        return False, top_name, f"{unique_count} different names = staffed operation"

    # Rule 4: Gatekeeper/office staff detected → NO
    # Unless dominant name (3+ mentions, 2x others) overrides
    if gatekeeper or len(office_names) >= 1:
        if dominant:
            return True, top_name, f"dominant name overrides gatekeeper ({top_mentions} mentions)"
        return False, top_name, "gatekeeper/office staff detected"

    # Rule 5: Explicit owner + small team → YES
    # But only if unique_count <= 3 (not a big staffed operation)
    if owner_mentioned and top_name and unique_count <= 3:
        # Check: is the "owner" actually the top name? Or is the owner unnamed?
        if top_role == "owner" or (owner_names and top_name == owner_names[0].get("name")):
            return True, top_name, "identified owner"
        # Owner mentioned but top name isn't the owner — could be unnamed owner + workers
        if top_mentions >= 2:
            return True, top_name, f"owner mentioned + {top_name} prominent ({top_mentions}x)"
    if owner_names and owner_names[0].get("mentions", 0) >= 2 and unique_count <= 3:
        return True, owner_names[0]["name"], "identified as owner"

    # Rule 6: Dominant name (3+ mentions, 2x others) → YES
    if dominant and unique_count <= 3:
        if top_role == "worker" and len(worker_names) >= 2:
            other_workers_prominent = any(w.get("mentions", 0) >= 2 for w in worker_names if w.get("name") != top_name)
            if other_workers_prominent:
                return False, top_name, "multiple prominent workers = staffed"
        if top_role == "worker" and not owner_mentioned and plural >= 3:
            return False, top_name, "worker with team language, no ownership signal"
        return True, top_name, f"dominant name ({top_mentions} mentions)"

    # Rule 7: 3 different names → likely staffed
    if unique_count == 3:
        if top_mentions >= 3:
            return True, top_name, f"dominant name despite 3 names ({top_mentions} mentions)"
        return False, top_name, "3 different names = likely staffed"

    # Rule 8: Name appears only once in many reviews → not confident
    if top_mentions <= 1 and total_reviews >= 6:
        return False, top_name, f"name appears only {top_mentions}x in {total_reviews} reviews"

    # Rule 9: Plural pronouns dominate with no clear owner
    if plural >= 3 and not owner_mentioned and top_mentions <= 2:
        return False, top_name, "plural pronouns dominate, no clear owner"

    # Rule 10: Two workers doing same job
    if unique_count == 2 and len(worker_names) == 2 and top_mentions < 3:
        return False, top_name, "two workers doing same job = staffed"

    # Rule 11: One name with 2+ mentions → YES
    if top_mentions >= 2:
        return True, top_name, f"name appears {top_mentions} times"

    # Rule 12: Micro business (few reviews) with a name → YES
    if total_reviews <= 5 and top_name:
        return True, top_name, "micro business with identifiable contact"

    # Rule 13: Single mention in many reviews → NO
    if top_mentions <= 1:
        return False, top_name, "name not prominent enough"

    return False, top_name, "insufficient signals"

def get_reviews(r, max_chars=5000):
    try:
        lst = json.loads(r.get("reviews_json") or "[]")
        if isinstance(lst, list):
            return "\n---\n".join(_clean_trainer_review_text(str(x)) for x in lst)[:max_chars]
    except Exception:
        pass
    return _clean_trainer_review_text(str(r.get("reviews_json") or ""))[:max_chars]

def process(args):
    idx, r = args
    biz = r.get("business_name","???")
    truth = r["would_call"].strip().lower()
    prompt = PROMPT_TEMPLATE.format(business_name=biz, reviews=get_reviews(r))
    extraction = call_gemini(prompt)
    would_call, person_name, reason = decide_would_call(extraction)
    result = {"would_call": would_call, "person_name": person_name, "reason": reason, "_extraction": extraction} if extraction else None
    return idx, biz, truth, result

start = time.time()
results = {}
with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
    futs = {ex.submit(process, (i, r)): i for i, r in enumerate(rows)}
    done = 0
    for f in as_completed(futs):
        idx, biz, truth, result = f.result()
        results[idx] = (biz, truth, result)
        done += 1
        print(f"  [{done}/{len(rows)}] {biz[:50]}", flush=True)

correct = total = 0
fn = []; fp = []
print("\n")
for idx in sorted(results):
    biz, truth, result = results[idx]
    if not result:
        print(f"  [ERROR] {biz}")
        continue
    pred = "yes" if result["would_call"] else "no"
    ok = pred == truth
    correct += ok; total += 1
    if not ok:
        (fn if truth=="yes" else fp).append((biz, result.get("person_name"), result.get("reason",""), result.get("_extraction")))
    status = "OK   " if ok else "WRONG"
    print(f"  [{status}] {biz[:45]:<45} truth={truth} pred={pred} name={result.get('person_name')}")

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%  ({len(fn)} FN, {len(fp)} FP)  [{time.time()-start:.0f}s]")
if fn:
    print(f"\nFALSE NEGATIVES:")
    for n,p,r,ext in fn:
        print(f"  {n[:50]} | {r[:70]}")
        if ext: print(f"    extraction: {json.dumps(ext, ensure_ascii=False)[:200]}")
if fp:
    print(f"\nFALSE POSITIVES:")
    for n,p,r,ext in fp:
        print(f"  {n[:50]} | {r[:70]}")
        if ext: print(f"    extraction: {json.dumps(ext, ensure_ascii=False)[:200]}")
