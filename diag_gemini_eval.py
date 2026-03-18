"""
Diagnostic: evaluate Gemini 2.5 Flash accuracy on labeled businesses.
Tests the "would I call?" prompt against all labels using parallel requests.
"""
import csv, json, os, re, sys, time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

from app import _clean_trainer_review_text

# ── Config ───────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: Set GOOGLE_API_KEY in .env")
    sys.exit(1)

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_FAST = "gemini-2.5-flash-lite"
MODEL_SMART = "gemini-2.5-pro"
PARALLEL = 15  # paid tier allows high concurrency

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

# ── Prompt ───────────────────────────────────────────────────────────

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


def call_gemini(prompt: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            full_prompt = SYSTEM_INSTRUCTION + "\n\n" + prompt
            response = client.models.generate_content(
                model=MODEL_FAST,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                ),
            )
            content = (response.text or "").strip()
            if not content:
                return None
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```\s*$", "", content)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"    JSON parse failed: [{content[:200]}]", flush=True)
                return None
        except Exception as e:
            if "503" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"    Gemini error: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return None


def decide_would_call(extraction: dict) -> tuple[bool, str | None, str, str]:
    """Deterministic decision based on extracted features.
    Returns (would_call, person_name, reason, confidence) where confidence is 'high' or 'low'."""
    if not extraction:
        return False, None, "extraction failed", "low"

    names = extraction.get("names", [])
    plural = extraction.get("plural_pronouns", 0)
    singular = extraction.get("singular_pronouns", 0)
    owner_mentioned = extraction.get("owner_mentioned", False)
    gatekeeper = extraction.get("gatekeeper", False)
    total_reviews = extraction.get("total_reviews", 0)

    named = [n for n in names if n.get("mentions", 0) >= 1]
    unique_count = len(named)

    top = max(named, key=lambda x: x.get("mentions", 0)) if named else None
    top_name = top["name"] if top else None
    top_mentions = top.get("mentions", 0) if top else 0
    top_role = top.get("role", "unknown") if top else "unknown"

    owner_names = [n for n in named if n.get("role") == "owner"]
    worker_names = [n for n in named if n.get("role") == "worker"]
    office_names = [n for n in named if n.get("role") == "office"]

    second = sorted(named, key=lambda x: x.get("mentions", 0), reverse=True)[1] if len(named) >= 2 else None
    second_mentions = second.get("mentions", 0) if second else 0
    dominant = top_mentions >= 3 and top_mentions >= second_mentions * 2

    # Rule 1: No names at all → NO (high confidence)
    if unique_count == 0:
        return False, None, "no names mentioned", "high"

    # Rule 2: Very dominant name (5+ mentions, 3x others) → YES
    very_dominant = top_mentions >= 5 and (second_mentions == 0 or top_mentions >= second_mentions * 3)
    if very_dominant:
        return True, top_name, f"very dominant name ({top_mentions} mentions)", "high"

    # Rule 3: 4+ different names → staffed → NO (unless one person is overwhelmingly dominant)
    if unique_count >= 4:
        if top_mentions >= 4 and top_mentions >= second_mentions * 2:
            return True, top_name, f"dominant name ({top_mentions}x) despite {unique_count} names", "high"
        return False, top_name, f"{unique_count} different names = staffed operation", "high"

    # Rule 4: Gatekeeper/office staff → NO (unless dominant owner/unknown overrides)
    # Micro businesses (<=3 reviews) with one named person likely have that person answering
    micro_solo = total_reviews <= 3 and unique_count == 1 and top_mentions >= 2
    if gatekeeper or len(office_names) >= 1:
        if dominant and (top_role in ("owner", "unknown") or owner_mentioned):
            return True, top_name, f"dominant name overrides gatekeeper ({top_mentions} mentions)", "low"
        if micro_solo:
            return True, top_name, f"micro business — {top_name} likely answers ({top_mentions}/{total_reviews})", "low"
        return False, top_name, "gatekeeper/office staff detected", "high"

    # Rule 5: Explicit owner + small team → YES
    # But if workers are as prominent as owner + team language → larger operation, skip
    max_worker_m = max((n.get("mentions", 0) for n in worker_names), default=0)
    big_operation = max_worker_m >= top_mentions and plural >= 3 and unique_count >= 3
    if owner_mentioned and top_name and unique_count <= 3 and not big_operation:
        if top_role == "owner" or (owner_names and top_name == owner_names[0].get("name")):
            return True, top_name, "identified owner", "low" if unique_count >= 3 else "high"
        if top_mentions >= 2:
            return True, top_name, f"owner mentioned + {top_name} prominent ({top_mentions}x)", "low"
    if owner_names and owner_names[0].get("mentions", 0) >= 2 and unique_count <= 3 and not big_operation:
        return True, owner_names[0]["name"], "identified as owner", "high"

    # Rule 6: Dominant name (3+ mentions, 2x others) → YES
    if dominant and unique_count <= 3:
        if top_role == "worker" and len(worker_names) >= 2:
            other_worker_max = max((w.get("mentions", 0) for w in worker_names if w.get("name") != top_name), default=0)
            if other_worker_max >= 2 and top_mentions < other_worker_max * 2:
                return False, top_name, "multiple prominent workers = staffed", "low"
        # Barely dominant worker with heavy team language (and low singular) → skeptical
        if top_mentions <= 3 and top_role == "worker" and plural >= 4 and singular <= plural // 2:
            return False, top_name, "worker with heavy team language", "low"
        # Multiple prominent names (even non-workers) = team
        if second_mentions >= 3 and unique_count >= 3:
            return False, top_name, "multiple prominent names = team operation", "low"
        return True, top_name, f"dominant name ({top_mentions} mentions)", "high"

    # Rule 7: 3 different names → likely staffed
    if unique_count == 3:
        if top_mentions >= 3:
            # All 3 names prominent = team, not one person
            third = sorted(named, key=lambda x: x.get("mentions", 0), reverse=True)[2]
            third_mentions = third.get("mentions", 0)
            if second_mentions >= 3 and third_mentions >= 2:
                return False, top_name, "3 prominent names = team operation", "low"
            return True, top_name, f"dominant name despite 3 names ({top_mentions} mentions)", "high"
        return False, top_name, "3 different names = likely staffed", "low"

    # Rule 8: Name appears only once in many reviews → not confident
    if top_mentions <= 1 and total_reviews >= 6:
        return False, top_name, f"name appears only {top_mentions}x in {total_reviews} reviews", "low"

    # Rule 9: Plural pronouns dominate with no clear owner
    if plural >= 3 and not owner_mentioned and top_mentions <= 2:
        return False, top_name, "plural pronouns dominate, no clear owner", "low"

    # Rule 10: Two workers doing same job
    if unique_count == 2 and len(worker_names) == 2 and top_mentions < 3:
        return False, top_name, "two workers doing same job = staffed", "low"

    # Rule 11: One name with 2+ mentions → YES
    if top_mentions >= 2:
        return True, top_name, f"name appears {top_mentions} times", "high"

    # Rule 12: Micro business (few reviews) with a name → YES
    if total_reviews <= 5 and top_name:
        return True, top_name, "micro business with identifiable contact", "high"

    # Rule 13: Single mention in many reviews → NO
    if top_mentions <= 1:
        return False, top_name, "name not prominent enough", "low"

    return False, top_name, "insufficient signals", "low"


PRO_PROMPT = """You are a cold-calling salesperson. You need to decide: should I call this business and ask for a specific person?

You MUST answer YES only if you are confident you can name a specific owner/decision-maker to ask for when you call. A name appearing once is NOT enough — you need repeated, prominent mentions.

Key signals for NO:
- Multiple different employee names doing the same work (staffed company)
- "They", "the team", "their crew" as dominant language (no clear owner)
- Office staff, receptionist, dispatcher mentioned (gatekeepers between you and decision-maker)
- Named person is called "technician", "installer", or similar employee title
- Name appears only 1 time across many reviews
- No name at all — just pronouns

Key signals for YES:
- One name appears repeatedly across reviews as THE person
- Ownership language: "the owner [Name]", "[Name] and his team" (possessive)
- Small/personal feel even with helpers

Business: "{business_name}"

REVIEWS:
{reviews}

Output ONLY JSON: {{"would_call": true or false, "person_name": "Name" or null, "reason": "one sentence"}}"""


def call_gemini_pro(biz_name: str, reviews: str) -> dict | None:
    """Second pass with Gemini 2.5 Pro for borderline cases."""
    try:
        prompt = PRO_PROMPT.format(business_name=biz_name, reviews=reviews)
        response = client.models.generate_content(
            model=MODEL_SMART,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
            ),
        )
        content = (response.text or "").strip()
        if not content:
            return None
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content)
        wc = re.search(r'"would_call"\s*:\s*(true|false)', content, re.IGNORECASE)
        pn = re.search(r'"person_name"\s*:\s*(?:"([^"]*?)"|null)', content, re.IGNORECASE)
        rs = re.search(r'"reason"\s*:\s*"([^"]*?)"', content, re.IGNORECASE)
        if wc:
            return {
                "would_call": wc.group(1).lower() == "true",
                "person_name": pn.group(1) if pn and pn.group(1) else None,
                "reason": rs.group(1) if rs else "",
            }
        return None
    except Exception as e:
        print(f"    Pro error: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None


def get_reviews_text(r, max_chars=6000) -> str:
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
    extraction = call_gemini(prompt)
    if extraction is None:
        return idx, biz_name, truth, None
    would_call, person_name, reason, confidence = decide_would_call(extraction)
    return idx, biz_name, truth, {
        "would_call": would_call,
        "person_name": person_name,
        "reason": reason,
        "confidence": confidence,
        "_reviews": reviews,
        "_extraction": extraction,
    }


# ── Pass 1: Flash-Lite extraction + rules ────────────────────────────

print(f"\n=== PASS 1: Flash-Lite extraction ({len(rows)} businesses, {PARALLEL} parallel) ===\n")

start = time.time()
results = {}

with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
    futures = {executor.submit(process_row, (i, r)): i for i, r in enumerate(rows)}
    done = 0
    for future in as_completed(futures):
        idx, biz_name, truth, result = future.result()
        results[idx] = (biz_name, truth, result)
        done += 1
        conf = result.get("confidence", "?") if result else "err"
        print(f"  [{done:>3}/{len(rows)}] {biz_name[:45]} [{conf}]", flush=True)

# Count borderline cases
borderline = {idx: (biz, truth, res) for idx, (biz, truth, res) in results.items()
              if res and res.get("confidence") == "low"}
high_conf = {idx: (biz, truth, res) for idx, (biz, truth, res) in results.items()
             if res and res.get("confidence") == "high"}

print(f"\nPass 1 done: {len(high_conf)} high-confidence, {len(borderline)} borderline")

SKIP_PRO = True  # Set to False when Pro model is available
# ── Pass 2: Gemini Pro for borderline cases ──────────────────────────

if borderline and not SKIP_PRO:
    print(f"\n=== PASS 2: Gemini 2.5 Pro on {len(borderline)} borderline cases ===\n")

    def process_borderline(args):
        idx, biz_name, reviews = args
        pro_result = call_gemini_pro(biz_name, reviews)
        return idx, pro_result

    bl_args = [(idx, biz, res.get("_reviews", ""))
               for idx, (biz, truth, res) in borderline.items()]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_borderline, a): a[0] for a in bl_args}
        done = 0
        for future in as_completed(futures):
            idx, pro_result = future.result()
            biz_name, truth, old_result = results[idx]
            done += 1

            if pro_result:
                old_pred = "yes" if old_result["would_call"] else "no"
                new_pred = "yes" if pro_result["would_call"] else "no"
                changed = " [CHANGED]" if old_pred != new_pred else ""
                print(f"  [{done}/{len(borderline)}] {biz_name[:45]} {old_pred}→{new_pred}{changed}", flush=True)
                # Override with Pro result
                results[idx] = (biz_name, truth, {
                    "would_call": pro_result["would_call"],
                    "person_name": pro_result.get("person_name"),
                    "reason": f"[Pro] {pro_result.get('reason', '')}",
                })
            else:
                print(f"  [{done}/{len(borderline)}] {biz_name[:45]} [Pro error, keeping Pass 1]", flush=True)

# ── Final Results ────────────────────────────────────────────────────

print("\n--- Final Results ---\n", flush=True)

correct = 0
total = 0
fn_cases = []
fp_cases = []
errors = 0

for idx in sorted(results):
    biz_name, truth, result = results[idx]
    if result is None:
        errors += 1
        print(f"  [{idx+1:>3}/{len(rows)}] [ERROR] {biz_name[:45]}", flush=True)
        continue

    would_call = bool(result.get("would_call", False))
    person = result.get("person_name")
    reason = str(result.get("reason", ""))[:80]
    pred = "yes" if would_call else "no"
    ok = pred == truth
    if ok:
        correct += 1
    total += 1

    if not ok:
        ext = result.get("_extraction", {})
        if truth == "yes":
            fn_cases.append((biz_name, person, reason, ext))
        else:
            fp_cases.append((biz_name, person, reason, ext))

    status = "[WRONG]" if not ok else "[  OK ]"
    print(f"  [{idx+1:>3}/{len(rows)}] {status} {biz_name[:40]:<40} truth={truth} pred={pred} name={person}", flush=True)

elapsed = time.time() - start

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"TWO-PASS EVALUATION RESULTS (Flash-Lite + Pro)")
print(f"{'='*70}")
acc = correct / total if total else 0
print(f"Accuracy: {correct}/{total} = {acc*100:.1f}%")
print(f"High-confidence (Pass 1): {len(high_conf)}")
print(f"Borderline → Pro (Pass 2): {len(borderline)}")
print(f"Time: {elapsed:.0f}s")
print(f"Errors: {errors}")
print(f"False Negatives (missed leads): {len(fn_cases)}")
print(f"False Positives (wasted calls): {len(fp_cases)}")

if fn_cases:
    print(f"\nFALSE NEGATIVES — Missed Leads (truth=YES, pred=NO):")
    for name, person, reason, ext in fn_cases:
        print(f"  {name[:50]:<50} name={person} reason={reason}")
        names_info = ext.get("names", [])
        print(f"    extraction: names={json.dumps(names_info)} plural={ext.get('plural_pronouns',0)} singular={ext.get('singular_pronouns',0)} owner={ext.get('owner_mentioned',False)} gatekeeper={ext.get('gatekeeper',False)} total_reviews={ext.get('total_reviews',0)}")

if fp_cases:
    print(f"\nFALSE POSITIVES — Wasted Calls (truth=NO, pred=YES):")
    for name, person, reason, ext in fp_cases:
        print(f"  {name[:50]:<50} name={person} reason={reason}")
        names_info = ext.get("names", [])
        print(f"    extraction: names={json.dumps(names_info)} plural={ext.get('plural_pronouns',0)} singular={ext.get('singular_pronouns',0)} owner={ext.get('owner_mentioned',False)} gatekeeper={ext.get('gatekeeper',False)} total_reviews={ext.get('total_reviews',0)}")

print(f"\nDone.")
