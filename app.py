"""
Google Maps Solo-Owner Leads Scraper
Uses Playwright for scraping and local Ollama (qwen2.5:7b) for owner detection.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import math
import os
import re
import sys
import time
import traceback
import hashlib
import html
import logging
from logging.handlers import RotatingFileHandler

# Windows: use ProactorEventLoop so Playwright can create browser subprocess.
# Streamlit (and other frameworks) may set a loop that doesn't support subprocess.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from dataforseo_client import DataForSeoError, ReviewRequest, fetch_reviews_batch, fetch_reviews_text, maps_search

# Eagerly import google-genai so threaded workers don't race on lazy imports.
# Use importlib to avoid namespace-package resolution issues with "from google import genai"
# which can fail under Streamlit's re-execution model.
try:
    import importlib as _importlib
    _genai_module = _importlib.import_module("google.genai")
    _genai_types = _importlib.import_module("google.genai.types")
    print("[STARTUP] google-genai loaded OK", flush=True)
except Exception as _genai_err:
    _genai_module = None  # type: ignore[assignment]
    _genai_types = None  # type: ignore[assignment]
    print(f"[STARTUP] google-genai import FAILED: {type(_genai_err).__name__}: {_genai_err}", flush=True)

try:
    from streamlit_extras.metric_cards import style_metric_cards
except Exception:
    def style_metric_cards():
        pass

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NICHES = [
    "plumber",
    "HVAC",
    "electrician",
    "locksmith",
    "garage door",
    "mobile detailer",
    "power washer",
    "floorer",
    "roofer",
    "dentist",
    "chiropractor",
]
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MIN_CONFIDENCE = 0.7
MAX_REVIEWS_TO_ANALYZE = 20
MIN_REVIEWS = 2
MAX_REVIEWS = 120
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
CACHE_SCHEMA_VERSION = 4
OWNER_WORKERS = 8
EARLY_STOP_REVIEW_SNIPPETS = 8
LISTING_WORKERS = 4
LISTING_TIMEOUT_SECONDS = 15
REVIEW_SCROLL_CYCLES = 6

# Prefer in-session scraping (more reliable + lower CPU) over spinning up parallel worker browsers.
USE_WORKER_BROWSERS = False
TRAINING_MODEL_FILE = "review_trainer_model.json"
TRAINER_REVIEW_SNIPPETS = 6
TRAINER_MIN_SUBSTANTIAL_REVIEW_CHARS = 45
TRAINER_MIN_SUBSTANTIAL_REVIEW_COUNT = 2
TRAINER_MIN_TOTAL_REVIEW_CHARS = 180
TRAINER_FEW_SHOT_EXAMPLES_PER_LABEL = 3
_FEW_SHOT_PROMPT_CACHE = {"key": None, "prompt": ""}

# ---------------------------------------------------------------------------
# Logging (debug-focused, file-based)
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception:
    # If we cannot create the directory, fall back to current working directory.
    LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def _init_root_logger() -> logging.Logger:
    logger = logging.getLogger("solo_app")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception:
        # If file handler fails (e.g. permission issues), at least keep console logging.
        pass

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    logger.addHandler(console)
    return logger


_ROOT_LOGGER = _init_root_logger()


def get_logger(component: str = "app") -> logging.Logger:
    """Return a component-specific logger (e.g. 'solo_app.scrape')."""
    return _ROOT_LOGGER.getChild(component)


def log_event(component: str, level: int, message: str, **fields) -> None:
    """Structured-ish logging helper for debugging.

    Example:
        log_event("scrape", logging.INFO, "scrape_start", city=city, niche=niche)
    """
    logger = get_logger(component)
    if fields:
        try:
            payload = json.dumps(fields, ensure_ascii=False, sort_keys=True)
            logger.log(level, "%s %s", message, payload)
        except Exception:
            logger.log(level, "%s %s", message, fields)
    else:
        logger.log(level, "%s", message)


def click_until_reviews_ready(click_action, ready_check, on_success=None) -> bool:
    try:
        click_action()
    except Exception:
        return False
    if not ready_check():
        return False
    if on_success:
        on_success()
    return True


def is_qualified_lead_row(row: dict) -> bool:
    owner = str(row.get("owner_name") or "").strip().lower()
    conf = float(row.get("confidence_score") or 0)
    return bool(row.get("solo")) and conf >= MIN_CONFIDENCE and owner not in ("", "unknown", "none", "null")


def append_review_labels(rows: list[dict], city: str, niche: str, labels_path: str | None = None) -> str:
    labels_path = labels_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_training_labels.csv")
    fieldnames = [
        "timestamp_utc",
        "city",
        "niche",
        "business_name",
        "address",
        "phone",
        "website",
        "rating",
        "num_reviews",
        "would_call",
        "reason",
        "evidence_quote",
        "highlighted_evidence_json",
        "owner_name_guess",
        "reviews_json",
    ]
    write_header = not os.path.exists(labels_path)
    try:
        with open(labels_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        log_event(
            "trainer",
            logging.DEBUG,
            "append_review_labels",
            path=labels_path,
            rows=len(rows),
            city=city,
            niche=niche,
        )
    except Exception as e:
        log_event(
            "trainer",
            logging.ERROR,
            "append_review_labels_failed",
            path=labels_path,
            rows=len(rows),
            error=str(e),
            trace=traceback.format_exc(limit=2),
        )
        raise
    return labels_path


_STOP_WORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "have", "were",
    "was", "are", "has", "had", "been", "but", "they", "them", "then",
    "than", "when", "what", "which", "who", "how", "can", "will", "just",
    "also", "very", "really", "about", "into", "our", "your", "you",
    "its", "there", "their", "she", "her", "his", "him", "too",
    # Domain/niche noise words that don't help detect solo ownership
    "plumbing", "plumber", "pipe", "drain", "water", "heater", "sewer",
    "cleaning", "clean", "cleaner", "washing", "wash", "pressure", "power",
    "detailing", "detail", "car", "auto", "mobile", "ceramic", "coating",
    "flooring", "floor", "carpet", "tile", "wood", "hardwood",
    "roofing", "roof", "roofer", "hvac", "ac", "heating", "cooling",
    "service", "services", "company", "business", "job", "work", "worker", "workers",
    "house", "home", "building", "property", "project", 
    "good", "great", "awesome", "excellent", "amazing", "best", "recommend", "highly",
    "time", "price", "cost", "quality", "responsive", "professional", "punctual", "value",
    "call", "called", "came", "fix", "fixed", "repair", "repaired", "install", "installed",
    "day", "days", "week", "weeks", "month", "months", "year", "years", "today", "tomorrow"
})
_NEGATION_CUES = frozenset({
    "not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "can't", "couldn't", "wouldn't",
    "shouldn't", "nor", "neither", "nobody", "nothing",
})


def _tokenize_for_training(text: str) -> list[str]:
    if not text:
        return []
    raw = re.findall(r"[a-z0-9']+", text.lower())
    tokens: list[str] = []
    negate_window = 0
    for w in raw:
        if w in _NEGATION_CUES:
            negate_window = 2
            continue
        if len(w) < 3 or w in _STOP_WORDS:
            continue
        if negate_window > 0:
            tokens.append(f"NOT_{w}")
            negate_window -= 1
        else:
            tokens.append(w)
    bigrams = [f"{tokens[j]}_{tokens[j+1]}" for j in range(len(tokens) - 1)]
    return tokens + bigrams


def _clean_trainer_review_text(text: str) -> str:
    """Safely remove Google Topics widget text and obvious owner replies from review text."""
    if not text:
        return ""
    # Remove owner reply blocks entirely if it starts with one
    if re.search(r"^(?:Dear|Hi|Hello)\s+[A-Za-z]+,\s*(?:Thank|We|I)", text[:60], re.IGNORECASE) or \
       re.search(r"^(?:Thank you|Thanks)(?:\s+so much)?\s+(?:for|for taking)", text[:60], re.IGNORECASE):
        return ""
    
    # Strip known widget texts from the end
    text = re.sub(r"\bresponse\s+from\s+the\s+owner\b[\s\S]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*(?:Price|Service|Quality)\s+assessment\b[\s\S]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bServices\s+(?:Power|Drain|Plumbing|Sewer|Auto|Car|Flooring|Service not listed|Water heater)\b[\s\S]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:Positive|Negative)\s+(?:Responsiveness|Punctuality|Quality|Professionalism|Value)\b[\s\S]*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_meta_features(text: str) -> list[str]:
    """Extract meta-features that predict: 'If I call this business, will I
    reach an owner / decision-maker?'  Not just 'is it solo-owned'."""
    feats: list[str] = []
    low = (text or "").lower()

    # ── Owner / personal signals ──────────────────────────────────────
    if re.search(r"\bowner\b", low):
        feats.append("_META_OWNER")
    if re.search(r"\b(himself|herself|personally|hands[- ]on|owner[- ]operated|owner[- ]run)\b", low):
        feats.append("_META_PERSONAL")
    if re.search(r"\b(family|husband|wife|son|daughter|brother|sister|mom|dad)\b", low):
        feats.append("_META_FAMILY")

    # ── Team language — split mild vs heavy ───────────────────────────
    team_hits = len(re.findall(r"\b(team|staff|crew|employees|workers)\b", low))
    if team_hits >= 3:
        feats.append("_META_TEAM_HEAVY")
    elif team_hits >= 1:
        feats.append("_META_TEAM_MILD")

    # ── Hard-NO signals (rotating staff, no single contact) ──────────
    if re.search(r"\b(different|rotating|various|multiple|another|several)\s+(technician|techs?|person|people|staff|crew|employees?|workers?|guys?|plumbers?|electricians?|cleaners?|painters?)\b", low):
        feats.append("_META_ROTATING")

    if re.search(r"\bsent\s+(a\s+)?(tech|technician|guy|someone)\b", low):
        feats.append("_META_DISPATCH_LANGUAGE")

    # ── Pronoun signals ────────────────────────────────────────────────
    he_count = len(re.findall(r"\b(he|his|him)\b", low))
    she_count = len(re.findall(r"\b(she|her|hers)\b", low))
    they_count = len(re.findall(r"\b(they|their|them)\b", low))
    if he_count > 0 and he_count > they_count * 2:
        feats.append("_META_HE_DOMINANT")
    if she_count > 0 and she_count > they_count * 2:
        feats.append("_META_SHE_DOMINANT")

    # ── Name-based features ──────────────────────────────────────────
    _NAME_STOP = frozenset({
        "The", "And", "This", "They", "He", "She", "It", "We", "I", "A",
        "In", "On", "At", "To", "For", "With", "Thank", "Thanks", "Great",
        "Good", "Very", "Awesome", "Excellent", "Highly", "Recommend", "Our",
        "My", "Your", "When", "If", "Just", "So", "Then", "As", "But",
        "Job", "Work", "Service", "Price", "Quality", "Professional",
        "Punctual", "Responsive", "Value", "Positive", "Negative", "Services",
        "Assessment", "Owner", "Team", "Staff", "Crew", "Guys", "Not", "No",
        "Yes", "Was", "Were", "Did", "Had", "Has", "Have", "Are", "Is",
        "Can", "Could", "Would", "Should", "Will", "Like", "Love", "Best",
        "Always", "Never", "Day", "Week", "Month", "Year", "Today",
        "Tomorrow", "After", "Before", "Called", "Came", "Made", "Got",
    })
    names_found = re.findall(r"\b[A-Z][a-z]{2,15}\b", text or "")
    valid_names = [n for n in names_found if n not in _NAME_STOP and n.lower() not in _STOP_WORDS]
    name_counts: dict[str, int] = {}
    for n in valid_names:
        name_counts[n] = name_counts.get(n, 0) + 1

    if name_counts:
        sorted_names = sorted(name_counts.items(), key=lambda kv: -kv[1])
        max_name = sorted_names[0][0]
        max_count = sorted_names[0][1]
        second_count = sorted_names[1][1] if len(sorted_names) >= 2 else 0

        # One name mentioned 3+ times
        if max_count >= 3:
            feats.append("_META_SINGLE_NAME_REPEATED")
        # One dominant name (≥2 mentions and 2× any other name)
        if max_count >= 2 and (second_count == 0 or max_count >= second_count * 2):
            feats.append("_META_DOMINANT_NAME")
        # ★ NEW: Very dominant name (5+ mentions, 3× any other)
        if max_count >= 5 and (second_count == 0 or max_count >= second_count * 3):
            feats.append("_META_NAME_VERY_DOMINANT")

    # ── Advanced structural features ─────────────────────────────────
    if re.search(r"\b(i|my|we)\b.{1,30}\b(owner|manager|boss)\b", low) or \
       re.search(r"\b(owner|manager|boss)\b.{1,30}\b(i|my|we)\b", low):
        feats.append("_META_FIRST_PERSON_QUOTE")

    if re.search(r"\bfamily\s+(owned|run|business)\b", low):
        feats.append("_META_FAMILY_BUSINESS_EXPLICIT")

    if re.search(r"\b(husband\s+and\s+wife|one\s+man\s+show|guy\s+and\s+his\s+(wife|son)|mom\s+and\s+pop)\b", low):
        feats.append("_META_HUSBAND_WIFE")

    if re.search(r"\b(owner\s+operated|did\s+the\s+work\s+himself|came\s+out\s+himself|runs\s+it\s+himself)\b", low):
        feats.append("_META_OWNER_OPERATED_EXPLICIT")

    # Owner-does-work: "owner" near an action verb
    if re.search(
        r"\bowner\b.{0,20}\b(came|fixed|installed|cleaned|did|does|handled|"
        r"arrived|showed|quoted|repaired|replaced|completed|answered|responded)\b",
        low,
    ) or re.search(
        r"\b(came|fixed|installed|cleaned|did|does|handled|arrived|showed|"
        r"quoted|repaired|replaced|completed|answered|responded)\b.{0,20}\bowner\b",
        low,
    ):
        feats.append("_META_OWNER_DOES_WORK")

    # "One man" / "one person" / "does everything himself"
    if re.search(r"\b(one\s+man|one\s+person|one\s+guy|does\s+everything|runs\s+it\s+(all\s+)?by\s+himself)\b", low):
        feats.append("_META_ONE_PERSON_EXPLICIT")

    # ── ★ NEW: Owner-reachability features ───────────────────────────

    # "Ask for NAME" / "request NAME" / "call NAME"
    _NON_NAME_CAPS = frozenset({
        "The", "They", "Their", "He", "She", "It", "We", "You", "Who",
        "What", "How", "My", "Our", "This", "That", "These", "Those",
        "There", "Then", "When", "Where", "And", "But", "For", "With",
    })
    if re.search(
        r"\b(?:ask\s+for|request|call|ask\s+about)\s+([A-Z][a-z]{2,23})\b",
        text or "",
    ):
        m = re.search(r"\b(?:ask\s+for|request|call|ask\s+about)\s+([A-Z][a-z]{2,23})\b", text or "")
        if m and m.group(1) not in _NON_NAME_CAPS:
            feats.append("_META_ASK_FOR_NAME")

    # "Owner/manager/president/founder NAME" or "NAME the owner/manager"
    if re.search(r"\b(?:owner|manager|president|founder)\s+[A-Z][a-z]{2,23}\b", text or "") or \
       re.search(r"\b[A-Z][a-z]{2,23}.{0,10}\b(?:the\s+owner|the\s+manager|the\s+founder)\b", text or ""):
        feats.append("_META_OWNER_TITLE")

    # NAME + action verb appearing in 3+ separate review-like contexts
    _ACTION_VERBS_RE = (
        r"(?:came\s+(?:out|to|and|over)|arrived|showed\s+up|fixed|installed|"
        r"quoted|repaired|cleaned|completed|handled|explained|helped|"
        r"replaced|inspected|responded|did|does|was\s+(?:great|awesome|amazing|fantastic|professional))"
    )
    name_action_hits: dict[str, int] = {}
    for m in re.finditer(
        rf"\b([A-Z][a-z]{{2,23}})\s+{_ACTION_VERBS_RE}\b", text or ""
    ):
        n = m.group(1)
        if n not in _NON_NAME_CAPS and n not in _NAME_STOP:
            name_action_hits[n] = name_action_hits.get(n, 0) + 1
    if any(c >= 3 for c in name_action_hits.values()):
        feats.append("_META_NAME_PLUS_ACTION")

    # General person-name-in-context signal (kept from original)
    def _has_person_name_signal(t: str) -> bool:
        if re.search(
            r"\b(?:ask(?:ed)?\s+for|worked\s+with|spoke\s+with|talked\s+to|"
            r"met\s+with|thanks\s+to)\s+[A-Z][a-z]{2,23}\b", t,
        ):
            return True
        for m in re.finditer(
            rf"\b([A-Z][a-z]{{2,23}})\s+{_ACTION_VERBS_RE}\b", t,
        ):
            if m.group(1) not in _NON_NAME_CAPS:
                return True
        return False
    if _has_person_name_signal(text or ""):
        feats.append("_META_PERSON_NAME")

    # ── ★ Interaction features (capture combinations) ────────────────
    has_name = "_META_SINGLE_NAME_REPEATED" in feats or "_META_DOMINANT_NAME" in feats
    has_team = "_META_TEAM_MILD" in feats or "_META_TEAM_HEAVY" in feats

    # Name + team = "has employees but identifiable owner" → still call
    if has_name and has_team:
        feats.append("_META_NAME_DESPITE_TEAM")

    # Strong name signal without any he/she pronoun dominance
    # (catches cases like "Cynthia" where reviewers use "they" generically)
    has_pronoun_signal = "_META_HE_DOMINANT" in feats or "_META_SHE_DOMINANT" in feats
    if has_name and not has_pronoun_signal:
        feats.append("_META_NAME_NO_PRONOUN")

    # "she/her" used significantly (any she_count > 2) = female owner signal
    if she_count >= 3:
        feats.append("_META_SHE_PRESENT")

    return feats


def _labels_file_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_training_labels.csv")


def _trainer_model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), TRAINING_MODEL_FILE)


def _training_biz_key(row: dict) -> str:
    """Stable identity key for a labeled business row (for deduplication)."""
    name = str(row.get("business_name") or "").strip().lower()
    addr = str(row.get("address") or "").strip().lower()
    phone = "".join(ch for ch in str(row.get("phone") or "") if ch.isdigit())
    return name + "|" + (addr or phone or "")


def _make_save_sig(row: dict) -> str:
    """Compute a deduplication signature for a labeled row.

    Uses only business identity + label so that updating highlights or notes on
    an already-saved entry still de-duplicates correctly within a session.
    Cross-session duplicates are prevented by pre-loading the CSV on session start.
    """
    return json.dumps(
        {k: str(row.get(k, "") or "") for k in sorted(["business_name", "address", "phone", "would_call"])},
        ensure_ascii=False,
        sort_keys=True,
    )


_TRAINER_FEATURE_NAMES: list[str] = [
    # Binary meta-features
    "_META_OWNER", "_META_PERSONAL", "_META_FAMILY",
    "_META_TEAM_MILD", "_META_TEAM_HEAVY",
    "_META_ROTATING", "_META_DISPATCH_LANGUAGE",
    "_META_HE_DOMINANT", "_META_SHE_DOMINANT",
    "_META_SINGLE_NAME_REPEATED", "_META_DOMINANT_NAME", "_META_NAME_VERY_DOMINANT",
    "_META_FIRST_PERSON_QUOTE",
    "_META_FAMILY_BUSINESS_EXPLICIT", "_META_HUSBAND_WIFE",
    "_META_OWNER_OPERATED_EXPLICIT", "_META_OWNER_DOES_WORK",
    "_META_ONE_PERSON_EXPLICIT",
    "_META_ASK_FOR_NAME", "_META_OWNER_TITLE",
    "_META_NAME_PLUS_ACTION", "_META_PERSON_NAME",
    # Interaction features
    "_META_NAME_DESPITE_TEAM", "_META_NAME_NO_PRONOUN", "_META_SHE_PRESENT",
    # Numeric features (continuous)
    "_NUM_TOP_NAME_COUNT",       # how many times the most-mentioned name appears
    "_NUM_NAME_DOMINANCE_RATIO", # top_name_count / second_name_count (0 if only 1 name)
    "_NUM_HE_COUNT",             # raw "he/his/him" count
    "_NUM_THEY_COUNT",           # raw "they/their/them" count
    "_NUM_HE_THEY_RATIO",        # he_count / (they_count + 1)
    "_NUM_TEAM_MENTIONS",        # count of team/staff/crew/employees/workers
    "_NUM_UNIQUE_NAMES",         # how many distinct person names found
    "_NUM_NAME_ACTION_HITS",     # how many NAME+action_verb matches
    "_NUM_OWNER_MENTIONS",       # how many times "owner" appears
    "_NUM_SHE_COUNT",            # raw "she/her/hers" count
    "_NUM_PERSONAL_PRONOUN_RATIO",  # (he + she) / (they + 1)
    "_NUM_TOP_NAME_RATIO",       # top_name_count / unique_names (how much does top name stand out?)
    "_NUM_ACTION_PER_NAME",      # action_hits / top_name_count (names in context vs just mentioned)
    "_NUM_REACHABILITY",         # composite reachability score
    # Per-review analysis features
    "_PRV_REVIEWS_WITH_NAME",    # how many reviews mention a person name
    "_PRV_REVIEWS_WITH_HE_SHE",  # how many reviews use he/she/her/him
    "_PRV_REVIEWS_WITH_THEY",    # how many reviews use they/their/them
    "_PRV_PCT_NAME",             # % of reviews with a person name
    "_PRV_PCT_HE_SHE",           # % of reviews with he/she
    "_PRV_PCT_THEY",             # % of reviews with they/their
    "_PRV_NAME_CONSISTENCY",     # max(reviews mentioning same name) / total reviews
    "_PRV_TOTAL_REVIEWS",        # total review count
]


def _extract_numeric_features(text: str) -> dict[str, float]:
    """Extract continuous numeric features from review text."""
    low = (text or "").lower()
    nums: dict[str, float] = {}

    # Name analysis
    _NAME_STOP = frozenset({
        "The", "And", "This", "They", "He", "She", "It", "We", "I", "A",
        "In", "On", "At", "To", "For", "With", "Thank", "Thanks", "Great",
        "Good", "Very", "Awesome", "Excellent", "Highly", "Recommend", "Our",
        "My", "Your", "When", "If", "Just", "So", "Then", "As", "But",
        "Job", "Work", "Service", "Price", "Quality", "Professional",
        "Punctual", "Responsive", "Value", "Positive", "Negative", "Services",
        "Assessment", "Owner", "Team", "Staff", "Crew", "Guys", "Not", "No",
        "Yes", "Was", "Were", "Did", "Had", "Has", "Have", "Are", "Is",
        "Can", "Could", "Would", "Should", "Will", "Like", "Love", "Best",
        "Always", "Never", "Day", "Week", "Month", "Year", "Today",
        "Tomorrow", "After", "Before", "Called", "Came", "Made", "Got",
    })
    names_found = re.findall(r"\b[A-Z][a-z]{2,15}\b", text or "")
    valid_names = [n for n in names_found if n not in _NAME_STOP and n.lower() not in _STOP_WORDS]
    name_counts: dict[str, int] = {}
    for n in valid_names:
        name_counts[n] = name_counts.get(n, 0) + 1

    if name_counts:
        sorted_counts = sorted(name_counts.values(), reverse=True)
        nums["_NUM_TOP_NAME_COUNT"] = float(sorted_counts[0])
        if len(sorted_counts) >= 2 and sorted_counts[1] > 0:
            nums["_NUM_NAME_DOMINANCE_RATIO"] = sorted_counts[0] / sorted_counts[1]
        else:
            nums["_NUM_NAME_DOMINANCE_RATIO"] = float(sorted_counts[0])  # infinite dominance
        nums["_NUM_UNIQUE_NAMES"] = float(len(name_counts))
    else:
        nums["_NUM_TOP_NAME_COUNT"] = 0.0
        nums["_NUM_NAME_DOMINANCE_RATIO"] = 0.0
        nums["_NUM_UNIQUE_NAMES"] = 0.0

    # Pronoun counts
    he_count = len(re.findall(r"\b(he|his|him)\b", low))
    she_count = len(re.findall(r"\b(she|her|hers)\b", low))
    they_count = len(re.findall(r"\b(they|their|them)\b", low))
    nums["_NUM_HE_COUNT"] = float(he_count)
    nums["_NUM_SHE_COUNT"] = float(she_count)
    nums["_NUM_THEY_COUNT"] = float(they_count)
    nums["_NUM_HE_THEY_RATIO"] = he_count / (they_count + 1.0)
    nums["_NUM_PERSONAL_PRONOUN_RATIO"] = (he_count + she_count) / (they_count + 1.0)

    # Team mentions
    nums["_NUM_TEAM_MENTIONS"] = float(len(re.findall(r"\b(team|staff|crew|employees|workers)\b", low)))

    # Name + action verb hits
    _ACTION_VERBS_RE = (
        r"(?:came\s+(?:out|to|and|over)|arrived|showed\s+up|fixed|installed|"
        r"quoted|repaired|cleaned|completed|handled|explained|helped|"
        r"replaced|inspected|responded|did|does|was\s+(?:great|awesome|amazing|fantastic|professional))"
    )
    _NON_NAME_CAPS = frozenset({
        "The", "They", "Their", "He", "She", "It", "We", "You", "Who",
        "What", "How", "My", "Our", "This", "That", "These", "Those",
        "There", "Then", "When", "Where", "And", "But", "For", "With",
    })
    action_hits = 0
    for m in re.finditer(rf"\b([A-Z][a-z]{{2,23}})\s+{_ACTION_VERBS_RE}\b", text or ""):
        if m.group(1) not in _NON_NAME_CAPS and m.group(1) not in _NAME_STOP:
            action_hits += 1
    nums["_NUM_NAME_ACTION_HITS"] = float(action_hits)

    # Owner mentions
    nums["_NUM_OWNER_MENTIONS"] = float(len(re.findall(r"\bowner\b", low)))

    # Name-to-unique-names ratio: how dominant is the top name relative to total names?
    # High ratio = one person stands out. Low ratio = many equal names (corporate).
    if nums["_NUM_UNIQUE_NAMES"] > 0:
        nums["_NUM_TOP_NAME_RATIO"] = nums["_NUM_TOP_NAME_COUNT"] / nums["_NUM_UNIQUE_NAMES"]
    else:
        nums["_NUM_TOP_NAME_RATIO"] = 0.0

    # Action-hits-to-name ratio: are names appearing in ACTION context or just mentioned?
    if nums["_NUM_TOP_NAME_COUNT"] > 0:
        nums["_NUM_ACTION_PER_NAME"] = nums["_NUM_NAME_ACTION_HITS"] / nums["_NUM_TOP_NAME_COUNT"]
    else:
        nums["_NUM_ACTION_PER_NAME"] = 0.0

    # Composite reachability signal: combines he-dominance with name presence
    # Higher = more likely to reach a specific person
    nums["_NUM_REACHABILITY"] = (
        nums["_NUM_PERSONAL_PRONOUN_RATIO"] * 2.0
        + nums["_NUM_TOP_NAME_RATIO"] * 3.0
        + nums["_NUM_NAME_ACTION_HITS"] * 0.5
        + nums["_NUM_OWNER_MENTIONS"] * 1.0
    )

    return nums


def _extract_per_review_features(reviews_json_str: str) -> dict[str, float]:
    """Analyze each review individually and aggregate — captures signals
    that get lost when all reviews are merged into one text blob."""
    nums: dict[str, float] = {}
    try:
        reviews = json.loads(reviews_json_str or "[]")
        if not isinstance(reviews, list):
            reviews = [str(reviews)]
    except (json.JSONDecodeError, TypeError):
        reviews = [str(reviews_json_str or "")]

    reviews = [_clean_trainer_review_text(str(r)) for r in reviews]
    reviews = [r for r in reviews if r and len(r) > 20]

    if not reviews:
        return {
            "_PRV_REVIEWS_WITH_NAME": 0.0, "_PRV_REVIEWS_WITH_HE_SHE": 0.0,
            "_PRV_REVIEWS_WITH_THEY": 0.0, "_PRV_PCT_NAME": 0.0,
            "_PRV_PCT_HE_SHE": 0.0, "_PRV_PCT_THEY": 0.0,
            "_PRV_NAME_CONSISTENCY": 0.0, "_PRV_TOTAL_REVIEWS": 0.0,
        }

    _NAME_STOP = frozenset({
        "The", "And", "This", "They", "He", "She", "It", "We", "I", "A",
        "In", "On", "At", "To", "For", "With", "Thank", "Thanks", "Great",
        "Good", "Very", "Awesome", "Excellent", "Highly", "Recommend", "Our",
        "My", "Your", "When", "If", "Just", "So", "Then", "As", "But",
        "Job", "Work", "Service", "Price", "Quality", "Professional",
        "Punctual", "Responsive", "Value", "Positive", "Negative", "Services",
        "Assessment", "Owner", "Team", "Staff", "Crew", "Guys", "Not", "No",
        "Yes", "Was", "Were", "Did", "Had", "Has", "Have", "Are", "Is",
        "Can", "Could", "Would", "Should", "Will", "Like", "Love", "Best",
        "Always", "Never", "Day", "Week", "Month", "Year", "Today",
        "Tomorrow", "After", "Before", "Called", "Came", "Made", "Got",
    })

    n_with_name = 0
    n_with_he_she = 0
    n_with_they = 0
    names_across_reviews: dict[str, int] = {}  # how many different reviews mention each name

    for rev in reviews:
        low = rev.lower()
        # Check for person name
        names_in_rev = set()
        for n in re.findall(r"\b[A-Z][a-z]{2,15}\b", rev):
            if n not in _NAME_STOP and n.lower() not in _STOP_WORDS:
                names_in_rev.add(n)
        if names_in_rev:
            n_with_name += 1
            for n in names_in_rev:
                names_across_reviews[n] = names_across_reviews.get(n, 0) + 1

        if re.search(r"\b(he|his|him|she|her|hers)\b", low):
            n_with_he_she += 1
        if re.search(r"\b(they|their|them)\b", low):
            n_with_they += 1

    n_reviews = len(reviews)
    nums["_PRV_REVIEWS_WITH_NAME"] = float(n_with_name)
    nums["_PRV_REVIEWS_WITH_HE_SHE"] = float(n_with_he_she)
    nums["_PRV_REVIEWS_WITH_THEY"] = float(n_with_they)
    nums["_PRV_PCT_NAME"] = n_with_name / n_reviews
    nums["_PRV_PCT_HE_SHE"] = n_with_he_she / n_reviews
    nums["_PRV_PCT_THEY"] = n_with_they / n_reviews
    nums["_PRV_TOTAL_REVIEWS"] = float(n_reviews)

    # Name consistency: how many reviews mention the SAME name?
    # High = one person keeps coming up. Low = different names in each review.
    if names_across_reviews:
        max_name_reviews = max(names_across_reviews.values())
        nums["_PRV_NAME_CONSISTENCY"] = max_name_reviews / n_reviews
    else:
        nums["_PRV_NAME_CONSISTENCY"] = 0.0

    return nums


def _meta_features_to_vector(feats: list[str], numeric_feats: dict[str, float] | None = None) -> list[float]:
    feat_set = set(feats)
    vec: list[float] = []
    for f in _TRAINER_FEATURE_NAMES:
        if f.startswith("_NUM_"):
            vec.append(float((numeric_feats or {}).get(f, 0.0)))
        else:
            vec.append(1.0 if f in feat_set else 0.0)
    return vec


def _load_labeled_rows(labels_path: str | None = None) -> list[dict]:
    labels_path = labels_path or _labels_file_path()
    if not os.path.exists(labels_path):
        raise RuntimeError("No labels file found yet.")
    all_labeled_rows: list[dict] = []
    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = str(row.get("would_call") or "").strip().lower()
            if label in ("yes", "no"):
                all_labeled_rows.append(row)
    seen_biz: dict[str, dict] = {}
    for row in all_labeled_rows:
        key = _training_biz_key(row)
        seen_biz[key if key.strip("|") else row.get("timestamp_utc", str(id(row)))] = row
    return list(seen_biz.values())


def _row_to_reviews_text(row: dict) -> str:
    try:
        review_list = json.loads(row.get("reviews_json") or "[]")
        if isinstance(review_list, list):
            clean_revs = [_clean_trainer_review_text(str(r)) for r in review_list]
            return " ".join(r for r in clean_revs if r)
    except (json.JSONDecodeError, TypeError):
        pass
    return _clean_trainer_review_text(str(row.get("reviews_json") or ""))


def train_review_preference_model(labels_path: str | None = None, model_path: str | None = None) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier
    import joblib

    labels_path = labels_path or _labels_file_path()
    model_path = model_path or _trainer_model_path()
    labeled_rows = _load_labeled_rows(labels_path)
    n_raw_labels = len(labeled_rows)

    X: list[list[float]] = []
    y: list[int] = []
    for row in labeled_rows:
        reviews_raw = _row_to_reviews_text(row)
        feats = _extract_meta_features(reviews_raw)
        nums = _extract_numeric_features(reviews_raw)
        prv = _extract_per_review_features(row.get("reviews_json") or "[]")
        nums.update(prv)
        vec = _meta_features_to_vector(feats, nums)
        label = 1 if str(row.get("would_call") or "").strip().lower() == "yes" else 0
        X.append(vec)
        y.append(label)

    if not X or sum(y) == 0 or sum(1 - v for v in y) == 0:
        raise RuntimeError("Need at least one yes and one no label to train.")

    n_yes_labels = sum(y)
    n_no_labels = len(y) - n_yes_labels

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X, y)

    # Save sklearn model with joblib
    sklearn_path = model_path.replace(".json", ".joblib")
    joblib.dump({"clf": clf, "feature_names": _TRAINER_FEATURE_NAMES}, sklearn_path)

    # Also save JSON metadata for compatibility
    model = {
        "version": 5,
        "model_type": "gradient_boosting",
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "labels_file": labels_path,
        "n_raw_labels": n_raw_labels,
        "n_yes_labels": n_yes_labels,
        "n_no_labels": n_no_labels,
        "vocab_size": len(_TRAINER_FEATURE_NAMES),
        "feature_names": _TRAINER_FEATURE_NAMES,
        "sklearn_model_path": sklearn_path,
        # Keep token_log_odds empty for backwards compat (old code checks for it)
        "token_log_odds": {},
        "prior_log_odds": 0.0,
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    log_event(
        "trainer",
        logging.INFO,
        "model_trained",
        n_raw_labels=n_raw_labels,
        n_yes=n_yes_labels,
        n_no=n_no_labels,
        vocab_size=len(_TRAINER_FEATURE_NAMES),
    )
    return model


def load_review_preference_model(model_path: str | None = None) -> dict | None:
    path = model_path or _trainer_model_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
        if not isinstance(model, dict) or not isinstance(model.get("token_log_odds"), dict):
            return None
        return model
    except Exception:
        return None


def score_would_call_probability(reviews_text: str, business_name: str = "", model_path: str | None = None) -> float | None:
    model = load_review_preference_model(model_path=model_path)
    if not model:
        return None

    # v5+: use sklearn model
    if model.get("model_type") == "gradient_boosting":
        import joblib
        sklearn_path = model.get("sklearn_model_path") or (model_path or _trainer_model_path()).replace(".json", ".joblib")
        if not os.path.exists(sklearn_path):
            return None
        data = joblib.load(sklearn_path)
        clf = data["clf"]
        feats = _extract_meta_features(reviews_text)
        nums = _extract_numeric_features(reviews_text)
        vec = _meta_features_to_vector(feats, nums)
        proba = clf.predict_proba([vec])[0][1]
        return _clamp_probability(float(proba))

    # v4 fallback: old Naive Bayes
    token_log_odds = model.get("token_log_odds") or {}
    logit = float(model.get("prior_log_odds") or 0.0)
    _n_labels = int(model.get("n_raw_labels", 0))
    if _n_labels >= 1000:
        toks = set(_tokenize_for_training(reviews_text))
        toks.update(_extract_meta_features(reviews_text))
    else:
        toks = set(_extract_meta_features(reviews_text))
    if not toks:
        return None
    for t in toks:
        if t in token_log_odds:
            logit += float(token_log_odds[t])
    logit = max(-20.0, min(20.0, logit))
    return _clamp_probability(1.0 / (1.0 + math.exp(-logit)))


def _clamp_probability(value: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, float(value)))


def _shorten_for_prompt(text: str, max_chars: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _load_user_preference_few_shot_prompt(
    labels_path: str | None = None,
    max_per_label: int = TRAINER_FEW_SHOT_EXAMPLES_PER_LABEL,
) -> str:
    global _FEW_SHOT_PROMPT_CACHE

    path = labels_path or _labels_file_path()
    if not os.path.exists(path):
        return ""

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return ""

    cache_key = (path, float(mtime), int(max_per_label))
    if _FEW_SHOT_PROMPT_CACHE.get("key") == cache_key:
        return str(_FEW_SHOT_PROMPT_CACHE.get("prompt") or "")

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return ""

    yes_examples: list[dict] = []
    no_examples: list[dict] = []
    for row in reversed(rows):
        label = str(row.get("would_call") or "").strip().lower()
        if label not in ("yes", "no"):
            continue
        bucket = yes_examples if label == "yes" else no_examples
        if len(bucket) >= max_per_label:
            continue

        try:
            review_list = json.loads(row.get("reviews_json") or "[]")
            if not isinstance(review_list, list):
                review_list = []
        except (json.JSONDecodeError, TypeError):
            review_list = []

        try:
            highlights = json.loads(row.get("highlighted_evidence_json") or "[]")
            if not isinstance(highlights, list):
                highlights = []
        except (json.JSONDecodeError, TypeError):
            highlights = []

        excerpt = " ".join(
            _shorten_for_prompt(_clean_trainer_review_text(str(s)), 160)
            for s in review_list[:2]
            if str(s or "").strip() and _clean_trainer_review_text(str(s))
        )
        highlight_text = " | ".join(
            _shorten_for_prompt(str(h.get("text") or ""), 120)
            for h in highlights[:2]
            if isinstance(h, dict) and str(h.get("text") or "").strip()
        )
        reason = _shorten_for_prompt(str(row.get("reason") or ""), 160)
        owner_guess = _shorten_for_prompt(str(row.get("owner_name_guess") or ""), 80)
        if not (excerpt or highlight_text or reason):
            continue

        bucket.append(
            {
                "label": label,
                "business_name": _shorten_for_prompt(str(row.get("business_name") or ""), 120),
                "owner_guess": owner_guess,
                "reason": reason,
                "highlight": highlight_text,
                "excerpt": excerpt,
            }
        )
        if len(yes_examples) >= max_per_label and len(no_examples) >= max_per_label:
            break

    lines: list[str] = []
    if yes_examples or no_examples:
        lines.extend(
            [
                "USER PREFERENCE EXAMPLES (soft guidance for borderline cases):",
                "These show which businesses the user WOULD or WOULD NOT call based on review patterns.",
                "Lean slightly toward recall: if reviews show a real owner name in owner-like context and no strong multi-staff evidence, prefer not to miss the lead.",
                "Still obey the strict owner-name rules and output only valid JSON.",
            ]
        )
        for group in (yes_examples, no_examples):
            for ex in group:
                lines.append(f"Example ({str(ex['label']).upper()})")
                if ex["business_name"]:
                    lines.append(f"- Business: {ex['business_name']}")
                if ex["owner_guess"]:
                    lines.append(f"- Owner guess: {ex['owner_guess']}")
                if ex["reason"]:
                    lines.append(f"- User note: {ex['reason']}")
                if ex["highlight"]:
                    lines.append(f"- Highlighted evidence: {ex['highlight']}")
                if ex["excerpt"]:
                    lines.append(f"- Review excerpt: {ex['excerpt']}")

    prompt = "\n".join(lines).strip()
    _FEW_SHOT_PROMPT_CACHE = {"key": cache_key, "prompt": prompt}
    return prompt


def _build_owner_detection_prompt(
    reviews_text: str,
    business_name: str = "",
    labels_path: str | None = None,
) -> str:
    prompt = """Analyze these reviews to determine if calling this business would likely connect a salesperson with the OWNER or a KEY DECISION-MAKER.

DECISION RULES:
- A specific person's name appearing repeatedly as the main contact/owner = YES, EVEN IF the business has employees, technicians, or a team. The key question is: "Can I ask for someone by name?"
- Reviews say "ask for [Name]", "[Name] is the owner", or one person clearly runs customer interactions = YES.
- He/him or she/her pronouns dominate, suggesting one person does the work = YES.
- Small husband/wife, family, or owner-operated business = YES.
- Reviews mention rotating/different technicians with no consistent contact = NO.
- No individual name appears — only "the company", "their team", "the office" = NO.
- Many different staff names appear equally (no clear main contact) = NO.
- IMPORTANT: Having employees does NOT automatically mean NO.

NAME RULES:
- owner_name must be a REAL PERSON NAME from reviews (e.g., Josh, Josh Smith, Maria).
- Never output pronouns or generic words as owner_name: he, she, they, owner, technician, plumber, team.
- If no clear real name appears, set owner_name to null and solo=false.

Output ONLY valid JSON: {\"owner_name\": \"Name\" or null, \"solo\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"brief explanation\"}
NOTE: "solo" here means "a specific reachable person exists", not "only one employee"."""
    if business_name:
        prompt += f"\n\nBUSINESS NAME:\n{_shorten_for_prompt(business_name, 180)}"
    few_shot = _load_user_preference_few_shot_prompt(labels_path=labels_path)
    if few_shot:
        prompt += f"\n\n{few_shot}"
    prompt += "\n\nREVIEWS:\n"
    prompt += reviews_text[:8000]
    return prompt


PERSON_NAME_BLOCKLIST = {
    "accept", "all", "am", "and", "book", "business", "call", "closed", "crew",
    "directions", "great", "help", "hours", "manager", "more", "my", "new",
    "not", "office", "open", "owner", "plumber", "professional", "rating", "rep",
    "repair", "results", "roof", "roofing", "sales", "schedule", "service",
    "staff", "suite", "team", "technician", "thank", "thanks", "website", "yes",
}
COMPANY_WORD_BLOCKLIST = {
    "and", "co", "company", "construction", "contracting", "contractor", "exteriors",
    "gutters", "home", "inc", "llc", "ltd", "pros", "repair", "replacement",
    "rescue", "restoration", "roof", "roofers", "roofing", "services", "solutions",
    "sons", "systems", "the", "works",
}
PERSON_CONTEXT_VERBS = (
    "answered", "arrived", "came", "cleaned", "communicated", "completed",
    "coordinated", "delivered", "did", "explained", "fixed", "handled", "helped",
    "installed", "inspected", "made", "quoted", "repaired", "replaced",
    "responded", "scheduled", "showed", "walked", "was", "were",
)

STEALTH_INIT_SCRIPT = r"""
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
"""

STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-dev-shm-usage",
]


def apply_stealth(context) -> None:
    try:
        context.add_init_script(STEALTH_INIT_SCRIPT)
    except Exception:
        pass


def normalize_person_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^A-Za-z' -]", " ", str(value)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return None
    parts = []
    for raw_part in cleaned.split():
        part = raw_part.strip("' -")
        if not part:
            continue
        if len(part) < 2 or len(part) > 24:
            return None
        lowered = part.lower()
        if lowered in PERSON_NAME_BLOCKLIST:
            return None
        parts.append(part.capitalize())
    if not parts:
        return None
    return " ".join(parts[:2])


def extract_contextual_person_names(reviews_text: str) -> dict[str, int]:
    if not reviews_text:
        return {}

    counts: dict[str, int] = {}
    patterns = [
        r"\b([A-Z][a-z]{1,23})\s+and\s+([A-Z][a-z]{1,23})\b",
        r"\b([A-Z][a-z]{1,23})\s+and\s+(?:his|her)\s+team\b",
        r"\b([A-Z][a-z]{1,23})\s+and\s+team\b",
        r"\b(?i:ask for|asked for|worked with|spoke with|talked to|met with|contacted|called|dealt with|thanks to|thank you)\s+([A-Z][a-z]{1,23})\b",
        r"\b(?i:owner|owners?|manager|project manager|sales rep|representative|technician)\s+(?i:named\s+)?([A-Z][a-z]{1,23})\b",
        rf"\b([A-Z][a-z]{{1,23}})\s+(?:{'|'.join(PERSON_CONTEXT_VERBS)})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, reviews_text):
            for group in match.groups():
                name = normalize_person_name(group)
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
    return counts


def extract_owner_name_from_replies(owner_answers: list[str]) -> str | None:
    """Extract owner's first name from reply signatures.

    Owner replies on Google Maps often end with a signature like:
    "Thank you! - Mike", "Best, Sarah", "Mike, Owner"
    or start with "Hi, this is Mike from..."
    """
    if not owner_answers:
        return None
    from collections import Counter
    name_counts: Counter = Counter()

    for reply in owner_answers:
        reply = reply.strip()
        if not reply:
            continue
        # Common sign-off patterns:
        # "Thank you! - Mike"  /  "Best regards, Sarah"  /  "- Mike"
        # "Cheers, Mike"  /  "Thanks, Mike Owner"
        sign_off = re.search(
            r"(?:^|\n)\s*[-–—]\s*([A-Z][a-z]{1,20})\s*[,.]?\s*(?:Owner|Manager|Founder|President|CEO|Operations)?\s*$",
            reply,
        )
        if sign_off:
            name = normalize_person_name(sign_off.group(1))
            if name:
                name_counts[name] += 1
                continue

        # "Best, Mike" / "Cheers, Mike" / "Thanks, Mike" / "Sincerely, Mike"
        sign_off2 = re.search(
            r"\b(?:best|cheers|thanks|sincerely|regards|warm regards|thank you|many thanks)[,!.]?\s+[-–—]?\s*([A-Z][a-z]{1,20})\s*[,.]?\s*(?:Owner|Manager|Founder)?\s*$",
            reply,
            re.IGNORECASE,
        )
        if sign_off2:
            name = normalize_person_name(sign_off2.group(1))
            if name:
                name_counts[name] += 1
                continue

        # "Hi, this is Mike from..." / "Hello, my name is Mike"
        intro = re.search(
            r"\b(?:this is|my name is|i'm|i am)\s+([A-Z][a-z]{1,20})\b",
            reply,
            re.IGNORECASE,
        )
        if intro:
            name = normalize_person_name(intro.group(1))
            if name:
                name_counts[name] += 1
                continue

        # "Mike here" at the start of a reply
        mike_here = re.match(r"^([A-Z][a-z]{1,20})\s+here\b", reply)
        if mike_here:
            name = normalize_person_name(mike_here.group(1))
            if name:
                name_counts[name] += 1

    if not name_counts:
        return None

    # Return the most frequently appearing name
    best_name, best_count = name_counts.most_common(1)[0]
    if best_count >= 1:
        return best_name
    return None


def has_owner_signals(reviews_text: str) -> bool:
    text = (reviews_text or "").strip()
    if len(text) < 40:
        return False
    # Fast deterministic signal: any contextual person names extracted?
    try:
        return len(extract_contextual_person_names(text)) > 0
    except Exception:
        return False


def closes_between_5_and_6_local(work_hours: dict | None) -> bool:
    """
    Returns True if any weekday (Mon–Fri) has a closing time around 5–6pm local.

    Uses DataForSEO `work_hours.timetable` structure:
      { "timetable": { "monday": [ { "open": {hour,minute}, "close": {hour,minute} }, ... ], ... } }
    """
    if not work_hours:
        return False
    timetable = work_hours.get("timetable") or {}
    weekdays = ("monday", "tuesday", "wednesday", "thursday", "friday")
    for day in weekdays:
        slots = timetable.get(day) or []
        for slot in slots:
            close = slot.get("close") or {}
            try:
                h = int(close.get("hour"))
            except Exception:
                continue
            # Treat 17:00–18:59 as acceptable “5–6pm” closing.
            if h in (17, 18):
                return True
    return False


def count_name_mentions(reviews_text: str, owner_name: str | None) -> int:
    if not reviews_text or not owner_name:
        return 0
    review_text_lower = reviews_text.lower()
    tokens = [t.lower() for t in owner_name.split() if t.strip()]
    candidates = set()
    if tokens:
        candidates.add(" ".join(tokens))
        candidates.add(tokens[0])
    mention_count = 0
    for cand in candidates:
        try:
            hits = len(re.findall(rf"\b{re.escape(cand)}\b", review_text_lower))
            mention_count = max(mention_count, hits)
        except Exception:
            continue
    return mention_count


def owner_has_person_context(reviews_text: str, owner_name: str | None) -> bool:
    normalized = normalize_person_name(owner_name)
    if not normalized:
        return False
    first = re.escape(normalized.split()[0])
    patterns = [
        rf"\b(?i:ask for|asked for|worked with|spoke with|talked to|met with|contacted|called|dealt with|thanks to|thank you)\s+{first}\b",
        rf"\b(?i:owner|owners?|manager|project manager|sales rep|representative|technician)\s+(?i:named\s+)?{first}\b",
        rf"\b{first}\b\s+(?:{'|'.join(PERSON_CONTEXT_VERBS)})\b",
        rf"\b{first}\b\s+and\s+[A-Z][a-z]{{1,23}}\b",
        rf"\b[A-Z][a-z]{{1,23}}\b\s+and\s+\b{first}\b",
        rf"\b{first}\b\s+and\s+(?:his|her)\s+team\b",
        rf"\b{first}\b\s+and\s+team\b",
    ]
    return any(re.search(pattern, reviews_text) for pattern in patterns)


def owner_is_paired_with_other_name(reviews_text: str, owner_name: str | None, other_names: list[str]) -> bool:
    normalized = normalize_person_name(owner_name)
    if not normalized or not other_names:
        return False
    first = re.escape(normalized.split()[0])
    for other_name in other_names:
        other_first = re.escape(other_name.split()[0])
        if re.search(rf"\b{first}\b\s+and\s+\b{other_first}\b", reviews_text):
            return True
        if re.search(rf"\b{other_first}\b\s+and\s+\b{first}\b", reviews_text):
            return True
    return False


def owner_name_matches_business_name(owner_name: str | None, business_name: str | None) -> bool:
    normalized_owner = normalize_person_name(owner_name)
    if not normalized_owner or not business_name:
        return False
    owner_tokens = {part.lower() for part in normalized_owner.split()}
    business_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z']+", business_name)
        if token and token.lower() not in COMPANY_WORD_BLOCKLIST
    }
    return bool(owner_tokens & business_tokens)


def validate_owner_detection(reviews_text: str, business_name: str, detected: dict) -> dict:
    owner_name = normalize_person_name(detected.get("owner_name"))
    solo = bool(detected.get("solo", False))
    confidence = float(detected.get("confidence", 0) or 0)
    reason = str(detected.get("reason", "") or "")[:200]
    review_lower = (reviews_text or "").lower()

    def extract_simple_first_names(text: str) -> set[str]:
        # Broad first-name-ish extraction for disqualifying obvious multi-staff cases.
        # Keep conservative: only used as a guardrail, not as a positive owner signal.
        if not text:
            return set()
        stop = {
            "i", "we", "our", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
            "dr", "mr", "mrs", "ms",
        }
        out: set[str] = set()
        for token in re.findall(r"\b[A-Z][a-z]{1,23}\b", text):
            name = normalize_person_name(token)
            if not name:
                continue
            first = name.split()[0].lower()
            if first in stop:
                continue
            out.add(name.split()[0])
        return out

    # Hard negative signal: the reviewer explicitly says it's different people/techs.
    rotation_phrases = (
        "different person",
        "different people",
        "different technician",
        "different tech",
        "almost every time",
        "every time we call",
        "various technicians",
        "multiple technicians",
    )
    if any(p in review_lower for p in rotation_phrases):
        names_seen = extract_simple_first_names(reviews_text)
        if len(names_seen) >= 2:
            solo = False
            confidence = min(confidence, 0.35)
            reason = (reason + " | Review implies multiple rotating staff; not an owner-run shop").strip(" |")

    if owner_name:
        mention_count = count_name_mentions(reviews_text, owner_name)
        contextual_names = extract_contextual_person_names(reviews_text)
        owner_context = contextual_names.get(owner_name, 0)
        if not owner_context and owner_has_person_context(reviews_text, owner_name):
            owner_context = 1

        other_names = sorted(
            name for name, count in contextual_names.items()
            if name != owner_name and count > 0
        )
        # Treat very small teams (1–2 consistently mentioned people) as effectively
        # “solo-owned” for calling purposes.
        small_team = len(contextual_names) <= 2

        if mention_count <= 1:
            confidence = min(confidence, 0.3)
            reason = (reason + " | Name appears only once in reviews").strip(" |")

        if mention_count == 0:
            owner_name = None
            solo = False
            confidence = min(confidence, 0.2)
            reason = (reason + " | Owner name is not supported by review text").strip(" |")
        elif owner_name_matches_business_name(owner_name, business_name) and owner_context == 0:
            owner_name = None
            solo = False
            confidence = min(confidence, 0.2)
            reason = (reason + " | Matched the company name, not a person mentioned in reviews").strip(" |")
        elif owner_is_paired_with_other_name(reviews_text, owner_name, other_names) and not small_team:
            # Even if paired, if the owner name dominates, still call
            if mention_count >= 4 and mention_count >= max((contextual_names.get(n, 0) for n in other_names), default=0) * 2:
                confidence = min(confidence, max(confidence, 0.7))
                reason = (reason + f" | Owner paired with staff but dominates mentions ({mention_count}x)").strip(" |")
            else:
                solo = False
                confidence = min(confidence, 0.35)
                reason = (reason + f" | Owner is repeatedly paired with another staff name: {', '.join(other_names[:2])}").strip(" |")
        elif len(other_names) >= 3 and not small_team:
            # Only force NO if 3+ other names AND owner doesn't clearly dominate
            other_max = max((contextual_names.get(n, 0) for n in other_names), default=0)
            if mention_count >= 4 and mention_count >= other_max * 2:
                # Owner dominates despite multiple staff — still a callable lead
                confidence = min(confidence, max(confidence, 0.7))
                reason = (reason + f" | Multiple staff but {owner_name} dominates ({mention_count} vs {other_max})").strip(" |")
            else:
                solo = False
                confidence = min(confidence, 0.35)
                reason = (reason + f" | Multiple service names found in reviews: {', '.join(other_names[:3])}").strip(" |")
        elif len(other_names) == 2 and not small_team:
            # Two other names: mild penalty, don't force NO
            confidence = min(confidence, 0.55)
            reason = (reason + f" | A couple of other staff names appear: {', '.join(other_names[:2])}").strip(" |")
        elif len(other_names) == 1:
            if small_team:
                # Example: “Jason and Teresa” – two-person shop; still call this.
                # Two-person shop (e.g. Randy and Mike, Josh and Danny) – still call and ask for one.
                confidence = min(confidence, max(confidence, 0.8))
                reason = (reason + f" | Two-person owner-style team detected: {owner_name} and {other_names[0]}").strip(" |")
                solo = True
            elif owner_context <= 1:
                confidence = min(confidence, 0.55)
                reason = (reason + f" | Another staff name also appears: {other_names[0]}").strip(" |")

    # Fallback: if the LLM did not return a usable owner but the reviews clearly
    # talk about one dominant person (e.g. “Jeff and his team”), infer that as
    # the owner so we do not miss good leads.
    #
    # IMPORTANT: keep this conservative. Never flip obvious multi-staff
    # businesses into "solo" just because one staff name is mentioned.
    if not owner_name:
        contextual_names = extract_contextual_person_names(reviews_text)
        if contextual_names:
            # Allow fallback when one name clearly dominates, even with multiple staff.
            if len(contextual_names) <= 4:
                # Block fallback for clearly multi-provider/professional settings.
                role_block = (" dr.", "dr ", "dentist", "hygienist", "technician", "trainee")
                if any(rb in review_lower for rb in role_block) and len(extract_simple_first_names(reviews_text)) >= 2:
                    # Keep as not-solo; do not infer an owner name from staff mentions.
                    pass
                else:
                    dominant_name, dominant_count = max(contextual_names.items(), key=lambda kv: kv[1])
                    total_mentions = sum(int(v) for v in contextual_names.values()) or 0
                    dominant_ratio = (dominant_count / total_mentions) if total_mentions else 0.0
                    # If it's a 2-person shop (e.g. spouses/partners), allow less dominance.
                    min_ratio = 0.55 if len(contextual_names) == 1 else 0.45
                    owner_hint = ("owner" in review_lower) or ("ask for" in review_lower)
                    if (
                        (dominant_count >= 2 or (len(contextual_names) == 2 and owner_hint and total_mentions >= 2))
                        and (dominant_ratio >= min_ratio or owner_hint)
                        and owner_has_person_context(reviews_text, dominant_name)
                    ):
                        owner_name = dominant_name
                        solo = True
                        confidence = max(confidence, 0.8)
                        reason = (
                            reason
                            + f" | Fallback owner inferred from reviews: {owner_name} dominates owner-like mentions"
                        ).strip(" |")
            # Partner-owners fallback: if the reviews explicitly say "owners" and only
            # two people are mentioned, treat as a call-worthy owner-run shop even
            # without a dominant single name.
            if not owner_name and "owner" in review_lower and len(contextual_names) == 2:
                names = sorted(contextual_names.items(), key=lambda kv: (-int(kv[1]), kv[0]))
                picked = names[0][0]
                if owner_has_person_context(reviews_text, picked):
                    owner_name = picked
                    solo = True
                    confidence = max(confidence, 0.8)
                    reason = (reason + f" | Two-owner shop inferred from reviews: {', '.join(n for n, _c in names)}").strip(" |")

            # Last resort for partner owners: if reviews contain an explicit "A and B"
            # name pair plus owner-style language, infer a call-worthy small shop.
            if not owner_name and ("owner" in review_lower or "ask for" in review_lower):
                pairs = re.findall(r"\b([A-Z][a-z]{1,23})\s+(?:and|&)\s+([A-Z][a-z]{1,23})\b", reviews_text)
                if pairs:
                    a, b = pairs[0]
                    a_n = normalize_person_name(a)
                    b_n = normalize_person_name(b)
                    if a_n and b_n and owner_has_person_context(reviews_text, a_n):
                        owner_name = a_n
                        solo = True
                        confidence = max(confidence, 0.8)
                        reason = (reason + f" | Two-owner name pair found in reviews: {a_n} and {b_n}").strip(" |")

    # Personalized trainer model overlay (learned from your yes/no labels).
    # Blends the trainer signal with the LLM detection instead of overriding.
    # The trainer weight scales with the number of labels — with < 100 labels,
    # the model is noisy and should only nudge the score, not override it.
    pref_prob = score_would_call_probability(reviews_text, business_name) if reviews_text else None
    if pref_prob is not None:
        # Weight: 0.3 at 50 labels, 0.5 at 100 labels, 0.7 at 200+ labels
        _model = load_review_preference_model()
        _n_labels = int((_model or {}).get("n_raw_labels", 50))
        _trainer_weight = min(0.7, max(0.2, _n_labels / 200.0))
        # Blend: weighted average of existing confidence and trainer probability
        blended = confidence * (1 - _trainer_weight) + pref_prob * _trainer_weight
        if pref_prob >= 0.60:
            confidence = max(confidence, min(0.95, blended))
            if owner_name and owner_has_person_context(reviews_text, owner_name):
                solo = True
            reason = (reason + f" | Trainer model suggests would-call ({pref_prob:.2f}, weight={_trainer_weight:.2f})").strip(" |")
        elif pref_prob <= 0.40:
            confidence = min(confidence, max(0.15, blended))
            if not owner_name:
                solo = False
            reason = (reason + f" | Trainer model suggests avoid-call ({pref_prob:.2f})").strip(" |")

    # STRICT NAME ENFORCEMENT
    # If we made it all the way down here and STILL don't have an actionable name,
    # completely override any statistical trainer models or fallbacks and push it to the NO pile.
    # The user's workflow demands a name to ask for in order to cold-call, so NO NAME = NO LEAD.
    #
    # EXCEPTION: If the trainer model is highly confident (≥0.80) that this is a
    # would-call business AND there is a dominant person name in the reviews,
    # rescue the lead by using that contextual name. This prevents the strict
    # enforcement from killing leads that the trainer correctly identified but
    # Ollama failed to parse a name for.
    _owner_check = str(owner_name or "").strip().lower()
    if not _owner_check or _owner_check in ("unknown", "none", "null"):
        rescued = False
        if pref_prob is not None and pref_prob >= 0.80:
            rescue_names = extract_contextual_person_names(reviews_text)
            if rescue_names:
                best_name, best_count = max(rescue_names.items(), key=lambda kv: kv[1])
                if best_count >= 2 and owner_has_person_context(reviews_text, best_name):
                    owner_name = best_name
                    solo = True
                    confidence = max(confidence, 0.75)
                    reason = (reason + f" | Trainer-rescued: high trainer prob ({pref_prob:.2f}) + dominant name '{best_name}' in reviews").strip(" |")
                    rescued = True
        if not rescued:
            owner_name = None
            solo = False
            confidence = min(confidence, 0.2)
            if not reason:
                reason = "No clear owner name found in reviews"

    return {
        "owner_name": owner_name,
        "solo": solo,
        "confidence": max(0.0, min(confidence, 1.0)),
        "reason": reason[:240],
    }


def normalize_place_href(href: str | None) -> str:
    cleaned = (href or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("/"):
        cleaned = "https://www.google.com" + cleaned
    if "/maps/place/" not in cleaned:
        return ""
    return cleaned


def normalize_name_for_match(value: str | None) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def names_roughly_match(left: str | None, right: str | None) -> bool:
    a = normalize_name_for_match(left)
    b = normalize_name_for_match(right)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def is_review_metadata_line(value: str | None) -> bool:
    cleaned = re.sub(r"[\ue000-\uf8ff]", " ", str(value or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|.:")
    if not cleaned:
        return True
    low = cleaned.lower()
    if re.fullmatch(r"(?:local guide\s*[·•]\s*)?\d+\s+reviews?(?:\s*[·•]\s*\d+\s+photos?)?(?:\s*[·•]\s*\d+\s+videos?)?", low):
        return True
    if re.fullmatch(r"\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago", low):
        return True
    if re.search(r"\bmentioned\s+in\s+\d+\s+reviews?\b", low):
        return True
    if re.fullmatch(r"(?:[a-z0-9][a-z0-9'’.&-]*\s+){1,5}\d+\s+reviews?(?:\s*[·•]\s*\d+\s+photos?)?(?:\s*[·•]\s*(?:local guide|\d+\s+photos?))?", low):
        return True
    if re.fullmatch(r"(?:[a-z0-9][a-z0-9'’.&-]*\s+){1,5}\d+\s+photos?", low):
        return True
    # Only mark as metadata if the line is PURELY a reviewer profile/count with no real content.
    # Do NOT drop short but real reviews like "Great work!" or "5 star service, highly recommend!"
    if (re.fullmatch(r"local guide\s*[^\w]*.*", low)
            and not re.search(r"[.!?]", cleaned)
            and len(cleaned) <= 120):
        return True
    # "Lia F. Local Guide · 420 reviews · 2,991 photos" — name + Local Guide badge + counts
    if (re.search(r"\blocal guide\b", low)
            and re.search(r"\b\d+\s+reviews?\b", low)
            and len(cleaned) <= 160):
        return True
    # Google Maps promotional / UI text
    if re.search(r"get the most out of google", low):
        return True
    return False


def is_business_card_snippet(value: str | None) -> bool:
    cleaned = re.sub(r"[\ue000-\uf8ff]", " ", str(value or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return True
    low = cleaned.lower()
    if re.search(r"\b[0-5]\.\d\s*\(\d{1,3}(?:,\d{3})*\)", cleaned):
        return True
    if re.search(r"\(\d{3}\)\s*\d{3}-\d{4}", cleaned):
        return True
    if re.search(r"\b(open|closed)\b.*\b\d{1,2}\s?(?:am|pm)\b", low):
        return True
    # Hours-of-operation block: "SundayClosed Monday9 AM..." or "Monday 9 AM – 5 PM"
    if re.search(r"\b(sunday|monday|tuesday|wednesday|thursday|friday|saturday)", low) and re.search(r"\d+\s*(?:am|pm)", low) and len(cleaned) <= 300:
        return True
    # Hours with "Open 24 hours" pattern: "SundayOpen 24 hours MondayOpen 24 hours..."
    if re.search(r"\b(sunday|monday|tuesday|wednesday|thursday|friday|saturday)", low) and re.search(r"open\s+24\s+hours", low) and len(cleaned) <= 300:
        return True
    # Google Plus Code: "WHGX+H2 El Paso, Texas"
    if re.match(r"^[a-z0-9]{4,8}\+[a-z0-9]{1,4}\s", low) and len(cleaned) <= 80:
        return True
    # Google boilerplate disclaimer
    if re.search(r"reviews\s+are\s+automatically\s+processed\s+to\s+detect", low):
        return True
    # Pure address: "1234 Street Name, City, ST 12345"
    if re.match(r"^\d+\s+[\w\s]+(?:ave|st|blvd|dr|rd|way|ct|ln|pl|pkwy|cir)\b", low) and re.search(r"\b[a-z]{2}\s+\d{5}\b", low) and len(cleaned) <= 200:
        return True
    # In-store shopping / pickup / delivery (Google business info)
    if re.search(r"\b(in-store shopping|in-store pickup|delivery)\b", low) and len(cleaned) <= 120:
        return True
    if re.search(r"\b(car detailing service|flooring store|contractor|plumber|electrician|hvac|roofing contractor|garage door supplier|pressure washing service)\b", low) and len(cleaned) <= 200:
        return True
    # Google Review Highlights / Topics widget: no sentence punctuation, contains
    # structured label pairs like "Price assessment Great price Services Power/pressure washing"
    if re.search(r"\b(price assessment|service assessment|quality assessment)\b", low):
        return True
    if re.search(r"\bservices\b.*\bwashing\b", low) and not re.search(r"[.!?]", cleaned) and len(cleaned) < 120:
        return True
    # Google Maps promotional / UI text
    if re.search(r"get the most out of google", low):
        return True
    if re.search(r"\b(sign in|create an account|download the app)\b", low) and len(cleaned) < 80:
        return True
    return False


def _is_review_highlight_fragment(value: str) -> bool:
    """Detect Google Review Highlights / Topics widget fragments.

    These are short, single-sentence quoted excerpts from actual reviews that
    Google displays as summary snippets.  They are NOT full reviews and must be
    dropped so they don't inflate the snippet count.

    Characteristics:
    - Typically one sentence (no paragraph structure).
    - Often ≤ 120 chars.
    - The raw text often arrives wrapped in literal double-quotes.
    - They contain no reviewer context (name, date, star rating).
    """
    cleaned = re.sub(r"\s+", " ", value).strip()
    # Strip surrounding quotes if present
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return False
    # A single short sentence (≤ 120 chars) with exactly one sentence-ending
    # punctuation mark or none at all is likely a highlight fragment.
    # Real reviews almost always exceed 120 chars or have multiple sentences.
    if len(cleaned) <= 120:
        sentence_count = len(re.findall(r"[.!?]+", cleaned))
        if sentence_count <= 1:
            return True
    return False


def clean_extracted_review_snippet(value: str | None) -> str:
    text = re.sub(r"[\ue000-\uf8ff]", " ", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    quoted_match = re.search(r'photo of reviewer who wrote\s+"([^\"]{20,400})"', text, flags=re.IGNORECASE)
    if quoted_match:
        text = quoted_match.group(1)
    # Detect if the entire raw text is a quoted fragment (Google Review Highlights)
    stripped = text.strip()
    _is_pure_quoted = (stripped.startswith('"') and stripped.endswith('"')
                       and stripped.count('"') == 2)
    lines = [re.sub(r"\s+", " ", line).strip(" -|") for line in text.split("\n")]
    lines = [line for line in lines if line]
    # Strip reviewer name + metadata prefix from each line BEFORE metadata-line detection
    # "Name Name Local Guide · N reviews · N photos N months ago ActualReview"
    _meta_prefix_re = re.compile(
        r"^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z'.]+){0,3}\s+)?(?:Local\s+Guide\s*[·•]\s*)?(?:\d+\s+reviews?\s*[·•]?\s*)?(?:\d+\s+photos?\s*[·•]?\s*)?(?:(?:\d+|a)\s+(?:day|week|month|year)s?\s+ago\s*)"
    )
    lines = [_meta_prefix_re.sub("", line).strip() or line for line in lines]
    while lines and is_review_metadata_line(lines[0]):
        lines.pop(0)
    while lines and is_review_metadata_line(lines[-1]):
        lines.pop()
    cleaned_lines = [line for line in lines if not is_review_metadata_line(line)]
    cleaned = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
    cleaned = re.sub(r"(?:\.\.\.\s*more|…\s*more|more)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"(?:\.\.\.|…)\s*$", "", cleaned).strip()
    # Strip owner reply text that may have been concatenated
    cleaned = re.sub(r"\bresponse\s+from\s+the\s+owner\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\(owner\)\s*.*$", "", cleaned, flags=re.IGNORECASE).strip()
    # Strip Google Review Topics/Highlights widget text appended to reviews
    cleaned = re.sub(r"\s*(?:Price|Service|Quality)\s+assessment\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    # Strip "Like Share" / "+N Like Share" action buttons from end
    cleaned = re.sub(r"\s*\+?\d*\s*Like\s+Share\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    # Strip "Hover to react" button text
    cleaned = re.sub(r"\s*Hover\s+to\s+react\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    # Final pass: strip reviewer metadata prefix from joined text
    cleaned = _meta_prefix_re.sub("", cleaned).strip()
    if not cleaned or is_review_metadata_line(cleaned) or is_business_card_snippet(cleaned):
        return ""
    # Drop Google Review Highlights widget fragments (short single-sentence quotes)
    if _is_pure_quoted and _is_review_highlight_fragment(cleaned):
        return ""
    # Also drop if the cleaned text itself is still wrapped in quotes (highlight fragment)
    if cleaned.startswith('"') and cleaned.endswith('"') and cleaned.count('"') == 2:
        inner = cleaned[1:-1].strip()
        if inner and _is_review_highlight_fragment(inner):
            return ""
    return cleaned


def _is_substring_of_existing(candidate: str, existing: list[str]) -> bool:
    """Return True if *candidate* is a substring of any item in *existing*,
    OR if any existing item is a substring of *candidate* (catches duplicates
    where one version has extra metadata prepended/appended)."""
    low = candidate.lower()
    for item in existing:
        item_low = item.lower()
        if low in item_low and len(low) < len(item_low):
            return True
        if item_low in low and len(item_low) < len(low):
            return True
    return False


def sanitize_review_snippets(snippets: list[str], max_items: int | None = None) -> list[str]:
    import logging as _logging
    _log = _logging.getLogger("solo_app.reviews")
    cleaned_items: list[str] = []
    seen = set()
    for idx, snippet in enumerate(snippets):
        cleaned = clean_extracted_review_snippet(snippet)
        if not cleaned:
            _log.info("sanitize_drop idx=%d reason=empty raw=%r", idx, (snippet or "")[:200])
            continue
        if re.search(r"\bresponse\s+from\s+the\s+owner\b", cleaned, flags=re.IGNORECASE):
            _log.info("sanitize_drop idx=%d reason=owner_reply cleaned=%r", idx, cleaned[:200])
            continue
        # Catch owner replies that don't use the "response from the owner" marker.
        # These typically start with grateful/thank-you language addressing a reviewer.
        if re.search(
            r"^(we(?:'re|'re)?\s+(?:incredibly\s+)?grateful\s+for\s+your|"
            r"thank\s+you\s+(?:so\s+much\s+)?for\s+(?:your\s+)?(?:kind\s+)?(?:review|words|feedback)|"
            r"thanks?\s+for\s+(?:the\s+)?(?:wonderful|great|amazing|kind)\s+(?:review|words|feedback)|"
            r"we\s+appreciate\s+(?:your|the)\s+(?:kind\s+)?(?:review|words|feedback))",
            cleaned, flags=re.IGNORECASE,
        ):
            _log.info("sanitize_drop idx=%d reason=owner_reply_pattern cleaned=%r", idx, cleaned[:200])
            continue
        key = cleaned.lower()
        if key in seen:
            _log.info("sanitize_drop idx=%d reason=duplicate cleaned=%r", idx, cleaned[:80])
            continue
        # Drop if this snippet is a substring of an already-accepted review
        if _is_substring_of_existing(cleaned, cleaned_items):
            _log.info("sanitize_drop idx=%d reason=substring_dup cleaned=%r", idx, cleaned[:80])
            continue
        seen.add(key)
        cleaned_items.append(cleaned)
        if max_items is not None and len(cleaned_items) >= max_items:
            break
    return cleaned_items


def detail_page_matches_candidate(
    detail_name: str | None,
    expected_name: str | None,
    place_url: str | None,
    expected_place_url: str | None,
) -> bool:
    if expected_name and not names_roughly_match(detail_name, expected_name):
        return False
    normalized_place_url = normalize_place_href(place_url)
    if expected_place_url and normalized_place_url and normalized_place_url != expected_place_url:
        return False
    return True

# ---------------------------------------------------------------------------
# Ollama owner detection
# ---------------------------------------------------------------------------
def _parse_ollama_json(content: str) -> dict | None:
    """Extract the first valid JSON object from Ollama response text."""
    start = content.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(content[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def detect_owner_with_ollama(reviews_text: str, business_name: str = "") -> dict:
    """Use local Ollama to detect solo owner from review text. Returns dict with owner_name, solo, confidence, reason."""
    if not reviews_text or len(reviews_text.strip()) < 50:
        return {"owner_name": None, "solo": False, "confidence": 0.0, "reason": "Insufficient review text"}

    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        prompt = _build_owner_detection_prompt(reviews_text, business_name=business_name)
        response = llm.invoke(prompt)
        content = (response.content if hasattr(response, "content") else str(response)).strip()

        data = _parse_ollama_json(content)

        # Retry with a simpler prompt if JSON parsing failed
        if data is None:
            retry_prompt = (
                f"Read these reviews and answer with ONLY a JSON object, nothing else.\n"
                f"Business: {_shorten_for_prompt(business_name, 100)}\n"
                f"Reviews: {reviews_text[:4000]}\n\n"
                f'Answer format: {{"owner_name": "FirstName" or null, "solo": true/false, "confidence": 0.0-1.0, "reason": "brief"}}'
            )
            retry_resp = llm.invoke(retry_prompt)
            retry_content = (retry_resp.content if hasattr(retry_resp, "content") else str(retry_resp)).strip()
            data = _parse_ollama_json(retry_content)

        if data is not None:
            return validate_owner_detection(
                reviews_text,
                business_name,
                {
                    "owner_name": data.get("owner_name"),
                    "solo": bool(data.get("solo", False)),
                    "confidence": float(data.get("confidence", 0)),
                    "reason": str(data.get("reason", ""))[:200],
                },
            )
    except Exception as e:
        return {"owner_name": None, "solo": False, "confidence": 0.0, "reason": f"Ollama error: {str(e)[:100]}"}

    return {"owner_name": None, "solo": False, "confidence": 0.0, "reason": "Could not parse model output"}


# ---------------------------------------------------------------------------
# Gemini extraction + rules owner detection (primary scorer)
# ---------------------------------------------------------------------------
_GEMINI_CLIENT = None
_GEMINI_MODEL = "gemini-2.5-flash-lite"

_GEMINI_EXTRACTION_PROMPT = """Extract information from business reviews. Do NOT decide yes/no — just extract facts.

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


def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or _genai_module is None:
            return None
        _GEMINI_CLIENT = _genai_module.Client(api_key=api_key)
    return _GEMINI_CLIENT


def _call_gemini_extraction(reviews_text: str, business_name: str, retries: int = 3) -> dict | None:
    """Call Gemini flash-lite to extract structured features from reviews."""
    _log = get_logger("gemini")
    client = _get_gemini_client()
    if not client:
        _log.error("Gemini client is None for '%s' (genai_module=%s, api_key_set=%s)",
                    business_name[:40], _genai_module is not None, bool(os.getenv("GOOGLE_API_KEY")))
        return None
    if _genai_types is None:
        _log.error("_genai_types is None — google-genai types import failed")
        return None
    types = _genai_types
    prompt = _GEMINI_EXTRACTION_PROMPT + f'\n\nBusiness: "{business_name}"\n\nREVIEWS:\n{reviews_text[:6000]}\n\nExtract facts as JSON. Do NOT decide would_call — just extract.'
    for attempt in range(retries):
        try:
            _log.info("Gemini API call attempt %d/%d for '%s' (prompt_len=%d)",
                       attempt + 1, retries, business_name[:40], len(prompt))
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                ),
            )
            raw_text = response.text
            _log.info("Gemini raw response for '%s': %s", business_name[:40],
                       (raw_text[:300] if raw_text else "<None>"))
            content = (raw_text or "").strip()
            if not content:
                _log.warning("Gemini returned EMPTY content for '%s'", business_name[:40])
                return None
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```\s*$", "", content)
            parsed = json.loads(content)
            _log.info("Gemini extraction OK for '%s': %s", business_name[:40], str(parsed)[:200])
            return parsed
        except json.JSONDecodeError as e:
            _log.error("JSON parse failed for '%s': %s — raw content: %s",
                        business_name[:40], e, content[:300])
            return None
        except Exception as e:
            _log.error("Gemini API exception for '%s' (attempt %d/%d): %s: %s",
                        business_name[:40], attempt + 1, retries, type(e).__name__, str(e)[:300])
            if "503" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def _decide_would_call(extraction: dict) -> tuple[bool, str | None, str, float]:
    """Deterministic decision based on extracted features.
    Returns (would_call, person_name, reason, confidence_score)."""
    if not extraction:
        return False, None, "extraction failed", 0.0

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

    # Rule 1: No names at all → NO
    if unique_count == 0:
        return False, None, "no names mentioned", 0.1

    # Rule 2: Very dominant name (5+ mentions, 3x others) → YES
    very_dominant = top_mentions >= 5 and (second_mentions == 0 or top_mentions >= second_mentions * 3)
    if very_dominant:
        return True, top_name, f"very dominant name ({top_mentions} mentions)", 0.95

    # Rule 3: 4+ different names → staffed → NO
    if unique_count >= 4:
        if top_mentions >= 4 and top_mentions >= second_mentions * 2:
            return True, top_name, f"dominant name ({top_mentions}x) despite {unique_count} names", 0.85
        return False, top_name, f"{unique_count} different names = staffed operation", 0.15

    # Rule 4: Gatekeeper/office staff → NO (unless dominant owner/unknown overrides)
    micro_solo = total_reviews <= 3 and unique_count == 1 and top_mentions >= 2
    if gatekeeper or len(office_names) >= 1:
        if dominant and (top_role in ("owner", "unknown") or owner_mentioned):
            return True, top_name, f"dominant name overrides gatekeeper ({top_mentions} mentions)", 0.75
        if micro_solo:
            return True, top_name, f"micro business — {top_name} likely answers ({top_mentions}/{total_reviews})", 0.75
        return False, top_name, "gatekeeper/office staff detected", 0.2

    # Rule 5: Explicit owner + small team → YES
    max_worker_m = max((n.get("mentions", 0) for n in worker_names), default=0)
    big_operation = max_worker_m >= top_mentions and plural >= 3 and unique_count >= 3
    if owner_mentioned and top_name and unique_count <= 3 and not big_operation:
        if top_role == "owner" or (owner_names and top_name == owner_names[0].get("name")):
            return True, top_name, "identified owner", 0.9 if unique_count < 3 else 0.75
        if top_mentions >= 2:
            return True, top_name, f"owner mentioned + {top_name} prominent ({top_mentions}x)", 0.75
    if owner_names and owner_names[0].get("mentions", 0) >= 2 and unique_count <= 3 and not big_operation:
        return True, owner_names[0]["name"], "identified as owner", 0.9

    # Rule 6: Dominant name (3+ mentions, 2x others) → YES
    if dominant and unique_count <= 3:
        if top_role == "worker" and len(worker_names) >= 2:
            other_worker_max = max((w.get("mentions", 0) for w in worker_names if w.get("name") != top_name), default=0)
            if other_worker_max >= 2 and top_mentions < other_worker_max * 2:
                return False, top_name, "multiple prominent workers = staffed", 0.3
        if top_mentions <= 3 and top_role == "worker" and plural >= 4 and singular <= plural // 2:
            return False, top_name, "worker with heavy team language", 0.3
        if second_mentions >= 3 and unique_count >= 3:
            return False, top_name, "multiple prominent names = team operation", 0.3
        return True, top_name, f"dominant name ({top_mentions} mentions)", 0.9

    # Rule 7: 3 different names → likely staffed
    if unique_count == 3:
        if top_mentions >= 3:
            third = sorted(named, key=lambda x: x.get("mentions", 0), reverse=True)[2]
            third_mentions = third.get("mentions", 0)
            if second_mentions >= 3 and third_mentions >= 2:
                return False, top_name, "3 prominent names = team operation", 0.25
            return True, top_name, f"dominant name despite 3 names ({top_mentions} mentions)", 0.85
        return False, top_name, "3 different names = likely staffed", 0.3

    # Rule 8: Name appears only once in many reviews → not confident
    if top_mentions <= 1 and total_reviews >= 6:
        return False, top_name, f"name appears only {top_mentions}x in {total_reviews} reviews", 0.3

    # Rule 9: Plural pronouns dominate with no clear owner
    if plural >= 3 and not owner_mentioned and top_mentions <= 2:
        return False, top_name, "plural pronouns dominate, no clear owner", 0.25

    # Rule 10: Two workers doing same job
    if unique_count == 2 and len(worker_names) == 2 and top_mentions < 3:
        return False, top_name, "two workers doing same job = staffed", 0.3

    # Rule 11: One name with 2+ mentions → YES
    if top_mentions >= 2:
        return True, top_name, f"name appears {top_mentions} times", 0.85

    # Rule 12: Micro business (few reviews) with a name → YES
    if total_reviews <= 5 and top_name:
        return True, top_name, "micro business with identifiable contact", 0.75

    # Rule 13: Single mention in many reviews → NO
    if top_mentions <= 1:
        return False, top_name, "name not prominent enough", 0.3

    return False, top_name, "insufficient signals", 0.3


def detect_owner_with_gemini(reviews_text: str, business_name: str = "") -> dict:
    """Use Gemini flash-lite extraction + deterministic rules to detect callable owner.
    Returns dict with owner_name, solo, confidence, reason — same shape as Ollama detector."""
    if not reviews_text or len(reviews_text.strip()) < 50:
        return {"owner_name": None, "solo": False, "confidence": 0.0, "reason": "Insufficient review text"}

    # Clean reviews the same way as training
    try:
        review_list = json.loads(reviews_text) if reviews_text.strip().startswith("[") else None
    except (json.JSONDecodeError, ValueError):
        review_list = None

    if review_list and isinstance(review_list, list):
        clean = [_clean_trainer_review_text(str(x)) for x in review_list]
        clean_text = "\n---\n".join(c for c in clean if c)
    else:
        clean_text = reviews_text

    extraction = _call_gemini_extraction(clean_text[:6000], business_name)
    if extraction is None:
        return {"owner_name": None, "solo": False, "confidence": 0.0, "reason": "Gemini extraction failed"}

    would_call, person_name, reason, confidence = _decide_would_call(extraction)

    return {
        "owner_name": person_name if would_call else (person_name or None),
        "solo": would_call,
        "confidence": confidence,
        "reason": f"[Gemini] {reason}",
    }


def detect_owner(reviews_text: str, business_name: str = "") -> dict:
    """Primary owner detection: Gemini if GOOGLE_API_KEY is set, else Ollama fallback."""
    _log = get_logger("owner_detection")
    if os.getenv("GOOGLE_API_KEY"):
        result = detect_owner_with_gemini(reviews_text, business_name)
        if result.get("confidence", 0) > 0 or result.get("solo"):
            _log.info("Gemini detection for '%s': solo=%s conf=%.2f reason=%s",
                       business_name[:40], result.get("solo"), result.get("confidence", 0), result.get("reason", "")[:80])
            return result
        _log.warning("Gemini returned zero confidence for '%s' (review_len=%d): %s — falling back to Ollama",
                      business_name[:40], len(reviews_text or ""), result.get("reason", ""))
    else:
        _log.info("No GOOGLE_API_KEY — using Ollama for '%s'", business_name[:40])
    return detect_owner_with_ollama(reviews_text, business_name)


# ---------------------------------------------------------------------------
# Google Maps scraping (Playwright)
# ---------------------------------------------------------------------------
def scrape_google_maps(
    city: str,
    niche: str,
    max_pages: int,
    progress_callback=None,
    status_callback=None,
    table_callback=None,
    headless: bool = True,
    max_businesses: int | None = None,
    target_leads: int | None = None,
    run_owner_detection: bool = True,
    review_snippets_target: int = EARLY_STOP_REVIEW_SNIPPETS,
):
    """
    Scrape Google Maps for businesses, then use Ollama to detect solo owners.
    Yields (row_dict, total_found) for live table updates.
    """
    from playwright.sync_api import sync_playwright

    run_id = f"{datetime.utcnow().isoformat()}Z"
    log_event(
        "scrape",
        logging.INFO,
        "scrape_google_maps_start",
        run_id=run_id,
        city=city,
        niche=niche,
        max_pages=max_pages,
        max_businesses=max_businesses,
        target_leads=target_leads,
        run_owner_detection=run_owner_detection,
        review_snippets_target=review_snippets_target,
        headless=headless,
    )

    # On Windows, ensure this thread uses an event loop that supports subprocess
    # (Streamlit may have set a SelectorEventLoop that raises NotImplementedError).
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

    search_query = f"{niche} in {city}"
    results = []
    seen_result_keys = set()
    qualified_found = 0

    def update_status(msg):
        if status_callback:
            status_callback(msg)
        log_event(
            "scrape",
            logging.DEBUG,
            "status",
            run_id=run_id,
            status=str(msg),
        )

    def update_progress(pct, msg=None):
        if progress_callback:
            progress_callback(pct, msg)
        log_event(
            "scrape",
            logging.DEBUG,
            "progress",
            run_id=run_id,
            pct=float(pct),
            progress_msg=str(msg or ""),
        )

    def configure_fast_context(ctx) -> None:
        """Speed-focused defaults: block heavy resources and reduce timeouts."""
        try:
            ctx.set_default_timeout(5000)
        except Exception:
            pass
        try:
            ctx.set_default_navigation_timeout(12000)
        except Exception:
            pass

        # Block the heaviest resource types. (Keep scripts + stylesheets.)
        try:
            def _route_handler(route):
                try:
                    req = route.request
                    if req.resource_type in ("image", "media", "font"):
                        return route.abort()
                    url = (req.url or "").lower()
                    if any(s in url for s in ("doubleclick.net", "googlesyndication", "adservice.google")):
                        return route.abort()
                except Exception:
                    pass
                return route.continue_()

            ctx.route("**/*", _route_handler)
        except Exception:
            pass

    def build_result_key(row: dict, fallback: str = "") -> str:
        name = (row.get("business_name") or "").strip().lower()
        address = (row.get("address") or "").strip().lower()
        phone = "".join(ch for ch in (row.get("phone") or "") if ch.isdigit())
        if name and address:
            return f"{name}|{address}"
        if name and phone:
            return f"{name}|{phone}"
        if name:
            return f"{name}|{fallback}"
        return ""

    def get_reject_reason(row: dict) -> str:
        owner = str(row.get("owner_name") or "").strip().lower()
        conf = float(row.get("confidence_score") or 0)
        if not bool(row.get("solo")):
            return "Not marked solo by owner detection"
        if owner in ("", "unknown", "none", "null"):
            return "No valid owner name found"
        if conf < MIN_CONFIDENCE:
            return f"Confidence below threshold ({conf:.2f} < {MIN_CONFIDENCE:.2f})"
        return "Qualified lead"

    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_places_cache.json")
    now_ts = int(time.time())
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                if isinstance(cache_data, dict):
                    cache = cache_data
                else:
                    cache = {}
        else:
            cache = {}
    except Exception as e:
        log_event(
            "scrape",
            logging.WARNING,
            "cache_load_failed",
            run_id=run_id,
            error=str(e),
            trace=traceback.format_exc(limit=2),
        )
        cache = {}

    def parse_review_count_from_text(text: str) -> int:
        if not text:
            return 0
        m = re.search(r"\((\d{1,3}(?:,\d{3})*)\)", text)
        if m:
            return int(m.group(1).replace(",", ""))
        m = re.search(r"(\d{1,3}(?:,\d{3})*)\s*reviews?", text, re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))
        return 0

    def parse_rating_from_text(text: str) -> float | None:
        if not text:
            return None
        # Prefer values directly tied to star glyph.
        m = re.search(r"([0-5](?:\.\d)?)\s*★", text)
        if not m:
            m = re.search(r"★\s*([0-5](?:\.\d)?)", text)
        if not m:
            # Fallback for "4.8 (123)" style card strings.
            m = re.search(r"\b([0-5](?:\.\d)?)\b\s*\(\d{1,3}(?:,\d{3})*\)", text)
        if not m:
            return None
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 5.0:
                return val
        except Exception:
            return None
        return None

    def wait_until(condition_fn, timeout_sec: float, poll_interval: float = 0.05) -> bool:
        """Return as soon as condition_fn() is True or timeout. Returns True if condition met."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                if condition_fn():
                    return True
            except Exception:
                pass
            time.sleep(poll_interval)
        return False

    def extract_name_guess(text: str) -> str:
        blocked = {
            "results",
            "sponsored",
            "share",
            "website",
            "directions",
            "collapse side panel",
            "rating",
            "hours",
            "all filters",
        }
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered in blocked:
                continue
            if "★" in line or re.search(r"\(\d", line):
                continue
            if any(ch.isalpha() for ch in line):
                return line
        return ""

    def normalize_field(value: str, missing: str = "Not listed") -> str:
        cleaned = (value or "").strip()
        return cleaned if cleaned else missing

    def extract_card_review_snippet(card_text: str) -> str:
        if not card_text:
            return ""
        quoted = re.findall(r"\"([^\"]{15,260})\"", card_text)
        if quoted:
            return quoted[-1].strip()
        lines = [ln.strip() for ln in card_text.splitlines() if ln.strip()]
        for ln in reversed(lines[-10:]):
            if len(ln) >= 20 and not re.search(r"\(\d{1,3}(?:,\d{3})*\)", ln):
                if any(ch.isalpha() for ch in ln):
                    return ln[:260].strip()
        return ""

    def get_detail_review_count(detail_page) -> int:
        try:
            return detail_page.evaluate(
                r"""
                () => {
                    const candidates = Array.from(document.querySelectorAll('button, [role="button"], span, div'));
                    for (const el of candidates) {
                        const txt = ((el.getAttribute && el.getAttribute('aria-label')) || el.textContent || '').trim();
                        if (!/reviews?/i.test(txt)) continue;
                        const m = txt.match(/(\d{1,3}(?:,\d{3})*)/);
                        if (m) return parseInt(m[1].replace(/,/g, ''), 10) || 0;
                    }
                    return 0;
                }
                """
            ) or 0
        except Exception:
            return 0

    def open_reviews_tab(detail_page) -> bool:
        # Mutable slots shared across closures.
        # _last_click_info[0]: set by whichever strategy fires, so we can log
        #   exactly what element was clicked (selector / tag / role / label).
        # _has_reviews_logged[0]: guards so has_reviews_loaded only emits one
        #   diagnostic log per open_reviews_tab() call (not once per poll).
        _last_click_info = [None]
        _has_reviews_logged = [False]

        def has_reviews_loaded() -> bool:
            try:
                # The FULL reviews panel has "Sort reviews" button and/or "Search reviews"
                # input.  These do NOT exist on the Overview tab (which only has "Write a
                # review" and 3 summary review cards).  Requiring these prevents false
                # positives from the Overview's pre-loaded review DOM.
                full_panel_controls = detail_page.query_selector_all(
                    'button[aria-label*="Sort reviews" i], input[aria-label*="Search reviews" i]'
                )
                n_controls = len(full_panel_controls)
                if not n_controls:
                    return False
                # Some Maps layouts render reviews outside a role="feed".
                els = detail_page.query_selector_all(
                    'div[data-review-id], div.jftiEf, div.GHT2ce, span.wiI7pd, span[data-expandable-section], .jftiEf .wiI7pd, .MyEned'
                )
                n_els = len(els)
                if n_els > 0:
                    if not _has_reviews_logged[0]:
                        _has_reviews_logged[0] = True
                        log_event("reviews", logging.WARNING, "has_reviews_loaded_ok",
                                  controls=n_controls, review_els=n_els, via="specific_selectors")
                    return True
                # Fallback layout: review text is present in generic body nodes.
                loose = detail_page.query_selector_all('.fontBodyMedium, span[dir="auto"], div[dir="auto"]')
                n_loose = len(loose)
                result = n_loose >= 6
                if result and not _has_reviews_logged[0]:
                    _has_reviews_logged[0] = True
                    log_event("reviews", logging.WARNING, "has_reviews_loaded_ok",
                              controls=n_controls, review_els=0, loose_els=n_loose, via="loose_fallback")
                return result
            except Exception:
                return False

        def review_tab_is_selected() -> bool:
            try:
                sel = detail_page.query_selector('button[role="tab"][aria-label*="Reviews" i][aria-selected="true"]')
                return sel is not None
            except Exception:
                return False

        def wait_for_reviews_ready(timeout_sec: float = 4.0) -> bool:
            deadline = time.monotonic() + max(0.2, timeout_sec)
            while time.monotonic() < deadline:
                if has_reviews_loaded():
                    return True
                time.sleep(0.05)
            return has_reviews_loaded()

        # Do NOT short-circuit on has_reviews_loaded() here: after clicking Back,
        # the previous listing's review DOM can still be present and would be wrong.
        # Helper JS to check if an element is a reviewer profile link (NOT the
        # Reviews tab).  Clicking these opens a new tab / navigates to the
        # contributor page and causes Maps to reset to a condensed 3-review view.
        _IS_REVIEWER_LINK_CHECK = r"""
                    // Skip <a> elements that link to reviewer profiles
                    const isReviewerLink = (el) => {
                        if (el.tagName === 'A') {
                            const href = (el.getAttribute('href') || '');
                            if (/\/maps\/contrib\//i.test(href)) return true;
                            // Reviewer links often contain "Local Guide" or "N reviews · M photos"
                            const text = (el.textContent || '').trim();
                            if (/local\s+guide/i.test(text) && /\d+\s+reviews?/i.test(text)) return true;
                        }
                        // Also skip if the element is INSIDE a reviewer profile link
                        let parent = el.parentElement;
                        while (parent) {
                            if (parent.tagName === 'A') {
                                const href = (parent.getAttribute('href') || '');
                                if (/\/maps\/contrib\//i.test(href)) return true;
                            }
                            parent = parent.parentElement;
                        }
                        return false;
                    };
        """

        # Helpers to wrap each strategy so _last_click_info is always populated.
        def _pw(selector, timeout):
            """Playwright click wrapper — records the matched selector."""
            detail_page.click(selector, timeout=timeout)
            _last_click_info[0] = {"method": "playwright", "selector": selector}

        def _pw_role(timeout):
            """Playwright get_by_role wrapper."""
            detail_page.get_by_role("tab", name=re.compile(r"Reviews", re.IGNORECASE)).first.click(timeout=timeout)
            _last_click_info[0] = {"method": "playwright_role", "selector": "tab[Reviews]"}

        def _js(js_code):
            """JS evaluate wrapper — JS must return null/false on failure or a
            dict {tag, role, label} on success. Result stored in _last_click_info."""
            result = detail_page.evaluate(js_code)
            if result:
                _last_click_info[0] = result  # dict from JS

        review_clickers = [
            # STRATEGY 0-4: Specific tab selectors first — these target the Reviews
            # tab button directly and will NOT match the Overview DOM's pre-loaded
            # 3-review cards.  Run these before any broad JS search.
            lambda: _pw('button[role="tab"][aria-label*="Reviews" i]', 2500),
            lambda: _pw_role(2500),
            lambda: _pw('button[role="tab"]:has-text("Reviews")', 2000),
            lambda: _pw('button[aria-label^="Reviews for" i]', 2000),
            lambda: _pw('button[aria-label*=" reviews" i]', 1500),
            # STRATEGY 5: Many listings have a dedicated "All reviews" button instead of a tab.
            lambda: _js(
                r"""
                () => {
                  """
                + _IS_REVIEWER_LINK_CHECK
                + r"""
                  const els = Array.from(document.querySelectorAll('button,[role="button"],a'));
                  for (const el of els) {
                    if (isReviewerLink(el)) continue;
                    const label = ((el.getAttribute && el.getAttribute('aria-label')) || el.textContent || '').trim();
                    if (!label) continue;
                    if (!/all\\s+reviews/i.test(label)) continue;
                    try { el.scrollIntoView({block:'center'}); } catch(e) {}
                    el.click();
                    return { tag: el.tagName, role: (el.getAttribute('role')||''), label: label.slice(0,120), method: 'all_reviews_js' };
                  }
                  return null;
                }
                """
            ),
            # STRATEGY 6: Broad JS — sorts all "reviews" candidates by count and
            # clicks the highest.  Only runs after all specific selectors have failed.
            # CRITICAL: exclude <a> links to reviewer profiles (/maps/contrib/).
            lambda: _js(
                r"""
                () => {
                  """
                + _IS_REVIEWER_LINK_CHECK
                + r"""
                  const candidates = Array.from(document.querySelectorAll('button,[role="button"],a,[role="tab"]'));
                  const scored = [];
                  for (const el of candidates) {
                    const label = ((el.getAttribute && el.getAttribute('aria-label')) || el.textContent || '').trim();
                    if (!label) continue;
                    if (!/reviews?/i.test(label)) continue;
                    if (isReviewerLink(el)) continue;
                    const m = label.match(/(\d{1,3}(?:,\d{3})*)/);
                    const count = m ? parseInt(m[1].replace(/,/g,''), 10) : 0;
                    scored.push({ el, count, label: label.slice(0, 120), tag: el.tagName, role: el.getAttribute('role')||'' });
                  }
                  scored.sort((a,b) => (b.count||0) - (a.count||0));
                  const pick = scored[0];
                  if (pick) {
                    try { pick.el.scrollIntoView({block:'center'}); } catch(e) {}
                    pick.el.click();
                    return { tag: pick.tag, role: pick.role, label: pick.label, count: pick.count, method: 'broad_js' };
                  }
                  return null;
                }
                """
            ),
            # STRATEGY 7 (Last resort): click any control with "review" + a number.
            # CRITICAL: exclude reviewer profile links.
            lambda: _js(
                r"""
                () => {
                  """
                + _IS_REVIEWER_LINK_CHECK
                + r"""
                  const els = Array.from(document.querySelectorAll('button,[role="button"],a'));
                  let best = null;
                  let bestCount = -1;
                  for (const el of els) {
                    if (isReviewerLink(el)) continue;
                    const label = ((el.getAttribute && el.getAttribute('aria-label')) || el.textContent || '').trim();
                    if (!label) continue;
                    if (!/review/i.test(label)) continue;
                    const m = label.match(/(\\d{1,3}(?:,\\d{3})*)/);
                    const count = m ? parseInt(m[1].replace(/,/g,''), 10) : -1;
                    if (count > bestCount) { bestCount = count; best = el; }
                  }
                  if (best) {
                    try { best.scrollIntoView({block:'center'}); } catch(e) {}
                    best.click();
                    return { tag: best.tagName, role: (best.getAttribute('role')||''), label: (best.getAttribute('aria-label')||best.textContent||'').slice(0,120), count: bestCount, method: 'last_resort_js' };
                  }
                  return null;
                }
                """
            ),
        ]

        # Before trying any strategy, log how many pages exist and the page URL
        try:
            _rtab_pre_pages = len(detail_page.context.pages)
            _rtab_pre_url = detail_page.url or ""
            log_event("reviews", logging.WARNING, "review_tab_start",
                      url=_rtab_pre_url[:80], pages=_rtab_pre_pages)
        except Exception:
            pass

        for idx, clicker in enumerate(review_clickers):
            # Snapshot page count before click
            try:
                _pages_before = len(detail_page.context.pages)
            except Exception:
                _pages_before = -1

            if click_until_reviews_ready(
                clicker,
                wait_for_reviews_ready,
            ):
                # Snapshot DOM state right at the moment reviews are declared ready.
                try:
                    _pages_after = len(detail_page.context.pages)
                    _post_url = detail_page.url or ""
                    _containers = detail_page.evaluate(
                        "() => document.querySelectorAll('div[data-review-id], div.jftiEf').length"
                    )
                    _tab_selected = detail_page.evaluate(
                        "() => !!document.querySelector('button[role=\"tab\"][aria-selected=\"true\"][aria-label]')"
                        " && (document.querySelector('button[role=\"tab\"][aria-selected=\"true\"]')"
                        "    ?.getAttribute('aria-label') || '')"
                    )
                    _sort_present = bool(detail_page.query_selector(
                        'button[aria-label*="Sort reviews" i], input[aria-label*="Search reviews" i]'
                    ))
                    log_event("reviews", logging.WARNING, "review_tab_strategy_ok",
                              strategy=idx,
                              clicked=_last_click_info[0],
                              pages_before=_pages_before,
                              pages_after=_pages_after,
                              containers=_containers,
                              tab_selected=_tab_selected,
                              sort_present=_sort_present,
                              url=_post_url[:80])
                    if _pages_after > _pages_before:
                        log_event("reviews", logging.WARNING, "review_tab_NEW_TAB_OPENED",
                                  strategy=idx, pages_before=_pages_before,
                                  pages_after=_pages_after)
                except Exception:
                    pass
                return True

            # Log failed strategy and check for tab leak
            try:
                _pages_after_fail = len(detail_page.context.pages)
                if _pages_after_fail > _pages_before:
                    log_event("reviews", logging.WARNING, "review_tab_strategy_fail_NEW_TAB",
                              strategy=idx, pages_before=_pages_before,
                              pages_after=_pages_after_fail)
            except Exception:
                pass

        # Avoid heavy/unstable evaluate fallbacks here. If strategies above fail,
        # return False quickly so the caller can fall back to card snippets.

        log_event("reviews", logging.WARNING, "open_reviews_tab_failed")
        return False

    def ensure_reviews_sorted_most_relevant(detail_page) -> bool:
        """Best-effort: keep review order aligned to Google's default 'Most relevant'.
        Uses TWO separate evaluate() calls with a Python-side wait between them so the
        sort dropdown has time to open before we click the menu item."""
        _CLICK_BY_TEXT_JS = r"""
            (regex_source) => {
                const re = new RegExp(regex_source, 'i');
                const nodes = Array.from(document.querySelectorAll(
                    'button,[role="button"],[role="menuitem"],[role="option"],[role="listitem"],div,span,a'
                ));
                for (const node of nodes) {
                    // Never click reviewer profile links
                    if (node.tagName === 'A' && /\/maps\/contrib\//i.test(node.getAttribute('href') || '')) continue;
                    let inContribLink = false;
                    let p = node.parentElement;
                    while (p) {
                        if (p.tagName === 'A' && /\/maps\/contrib\//i.test(p.getAttribute('href') || '')) { inContribLink = true; break; }
                        p = p.parentElement;
                    }
                    if (inContribLink) continue;
                    const label = (
                        (node.getAttribute && (node.getAttribute('aria-label') || node.getAttribute('data-value')))
                        || node.innerText
                        || node.textContent
                        || ''
                    ).replace(/\s+/g, ' ').trim();
                    if (!label) continue;
                    if (!re.test(label)) continue;
                    try { node.click(); return true; } catch (e) {}
                }
                return false;
            }
        """
        try:
            opened = detail_page.evaluate(_CLICK_BY_TEXT_JS, r"sort\s+reviews?")
            if not opened:
                return False
            # Wait for the sort dropdown to open before clicking the menu item.
            try:
                detail_page.wait_for_timeout(450)
            except Exception:
                pass
            picked = detail_page.evaluate(_CLICK_BY_TEXT_JS, r"^most\s+relevant$")
            if not picked:
                picked = detail_page.evaluate(_CLICK_BY_TEXT_JS, r"most\s+relevant")
            if picked:
                try:
                    detail_page.wait_for_timeout(300)
                except Exception:
                    pass
            return bool(picked)
        except Exception:
            return False

    def _pre_scroll_review_panel(page, steps: int = 5, wait_ms: int = 380) -> None:
        """Scroll the reviews feed from Python so the browser can lazy-load successive
        batches of review cards between evaluate() calls.  Each call to wait_for_timeout
        yields control back to the browser event loop, allowing IntersectionObserver
        callbacks and XHR responses to complete before the next scroll."""
        _SCROLL_JS = r"""
            () => {
                const reviewContainerSelector = 'div[data-review-id], div.jftiEf, div.GHT2ce, div[class*="jftiEf"], div[class*="GHT2ce"]';
                const sels = [
                    'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
                    'div[role="feed"]',
                    '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
                    '.WNBkOb',
                    '.XiKgde',
                    '.m6QErb[role="feed"]',
                    'div.section-layout.section-scrollbox',
                ];
                for (const sel of sels) {
                    const el = document.querySelector(sel);
                    if (el && el.scrollHeight > el.clientHeight) {
                        el.scrollTop += Math.max(400, (el.clientHeight || 700) * 0.85);
                        return true;
                    }
                }
                const reviews = Array.from(document.querySelectorAll(reviewContainerSelector));
                if (reviews.length > 0) {
                    let curr = reviews[0].parentElement;
                    while (curr && curr !== document.body) {
                        const style = window.getComputedStyle(curr);
                        if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                            curr.scrollTop += Math.max(400, (curr.clientHeight || 700) * 0.85);
                            return true;
                        }
                        curr = curr.parentElement;
                    }
                }
                return false;
            }
        """
        for _ in range(steps):
            try:
                page.evaluate(_SCROLL_JS)
                page.wait_for_timeout(wait_ms)
            except Exception:
                break
        # Scroll back to top so the JS extractor starts from the first reviews
        _SCROLL_TOP_JS = r"""
            () => {
                const sels = [
                    'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
                    'div[role="feed"]',
                    '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
                    '.WNBkOb',
                    '.XiKgde',
                    '.m6QErb[role="feed"]',
                    'div.section-layout.section-scrollbox',
                ];
                for (const sel of sels) {
                    const el = document.querySelector(sel);
                    if (el && el.scrollHeight > el.clientHeight) {
                        el.scrollTop = 0;
                        return true;
                    }
                }
                return false;
            }
        """
        try:
            page.evaluate(_SCROLL_TOP_JS)
            page.wait_for_timeout(300)
        except Exception:
            pass

    def collect_review_texts(
        detail_page,
        max_snippets: int = EARLY_STOP_REVIEW_SNIPPETS,
        max_scroll_cycles: int = REVIEW_SCROLL_CYCLES,
        listed_count: int | None = None,
    ) -> list[str]:
        # Fast scroll mode: single in-page loop that quickly scrolls, expands
        # "... More", and collects the first N customer reviews in relevance order.
        # Give enough time for lazy-rendered review cards to load.
        # Google Maps only renders .wiI7pd text when scrolled into view,
        # so we need generous timeouts per container.
        max_ms = 4000.0 if listed_count is None else float(min(6000, 2000 + min(listed_count, 80) * 40))
        if max_snippets >= TRAINER_REVIEW_SNIPPETS:
            max_ms = max(max_ms, 10000.0)
        try:
            review_texts = detail_page.evaluate(
                r"""
                async (params) => {
                    let { needed, maxMs, listedCount } = params;
                    // NOTE: We do NOT cap needed to listedCount here.
                    // The listed count from the card view is often wrong/stale.
                    // Always try to collect `needed` reviews — the Python-side
                    // sanitizer will handle any extras.

                    const start = performance.now();
                    const orderedKeys = [];
                    const textByKey = new Map();
                    const sleep = ms => new Promise(r => setTimeout(r, ms));

                    const shouldReplace = (oldText, newText) => {
                        const prev = String(oldText || '').trim();
                        const next = String(newText || '').trim();
                        if (!prev) return true;
                        if (!next) return false;
                        if (next.length >= prev.length + 12) return true;
                        if (/(?:\.\.\.|…)\s*$/.test(prev) && next.length > prev.length) return true;
                        return false;
                    };

                    const upsertText = (key, text) => {
                        const k = String(key || '').trim();
                        const t = String(text || '').trim();
                        if (!k || !t) return;
                        if (!textByKey.has(k)) {
                            orderedKeys.push(k);
                            textByKey.set(k, t);
                            return;
                        }
                        const prev = textByKey.get(k) || '';
                        if (shouldReplace(prev, t)) {
                            textByKey.set(k, t);
                        }
                    };

                    const snapshot = () => {
                        const out = [];
                        for (const k of orderedKeys) {
                            const t = (textByKey.get(k) || '').trim();
                            if (!t) continue;
                            out.push(t);
                            if (out.length >= needed) break;
                        }
                        return out;
                    };

                                        const reviewContainerSelector = 'div[data-review-id], div.jftiEf, div.GHT2ce, div[class*="jftiEf"], div[class*="GHT2ce"]';
                                        const primaryReviewBodySelectors = [
                                            '.wiI7pd',
                                            '[data-expandable-section]',
                                            '.MyEned'
                                        ];
                                        const secondaryReviewBodySelectors = [
                                            '.fontBodyMedium',
                                            'span[dir="auto"]',
                                            'div[dir="auto"]'
                                        ];

                                        const cleanReviewText = (value) => {
                                            let text = String(value || '').replace(/[\uE000-\uF8FF]/g, ' ').replace(/\s+/g, ' ').trim();
                                            if (!text) return '';
                                            text = text.replace(/(?:\.\.\.\s*more|…\s*more|more)\s*$/i, '').trim();
                                            // Strip owner reply text: "Response from the owner" and everything after
                                            text = text.replace(/\bresponse\s+from\s+the\s+owner\b[\s\S]*$/i, '').trim();
                                            // Also handle "(Owner)" prefix pattern sometimes seen
                                            text = text.replace(/\(owner\)\s*[\s\S]*$/i, '').trim();
                                            // Strip Google Review Topics/Highlights widget text appended to reviews
                                            text = text.replace(/\s*(?:Price|Service|Quality)\s+assessment\b[\s\S]*$/i, '').trim();
                                            // Strip "Like Share" / "+N Like Share" action buttons from end
                                            text = text.replace(/\s*\+?\d*\s*Like\s+Share\s*$/i, '').trim();
                                            // Strip "Hover to react" button text
                                            text = text.replace(/\s*Hover\s+to\s+react\s*$/i, '').trim();
                                            // Strip reviewer name + metadata prepended to review text
                                            // Pattern: "Name Name Local Guide · N reviews · N photos N months ago ActualReview"
                                            text = text.replace(/^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z'.]+){0,3}\s+)?(?:Local\s+Guide\s*[·•]\s*)?(?:\d+\s+reviews?\s*[·•]?\s*)?(?:\d+\s+photos?\s*[·•]?\s*)?(?:(?:\d+|a)\s+(?:day|week|month|year)s?\s+ago\s*)/i, '').trim();
                                            return text;
                                        };

                                        const isLikelyReviewerMeta = (value) => {
                                            const text = cleanReviewText(value);
                                            const low = text.toLowerCase();
                                            if (!text) return true;
                                            if (/(local guide|\b\d+\s+reviews?\b|\b\d+\s+photos?\b|\b\d+\s+videos?\b)/i.test(text)
                                                && !/[.!?]/.test(text)
                                                && text.length <= 120) {
                                                return true;
                                            }
                                            if (/^\d+\s+(day|days|week|weeks|month|months|year|years)\s+ago$/i.test(text)) {
                                                return true;
                                            }
                                            return false;
                                        };

                                        const isLikelyBusinessCard = (value) => {
                                            const text = cleanReviewText(value);
                                            const low = text.toLowerCase();
                                            if (!text) return true;
                                            if (/\b[0-5]\.\d\s*\(\d{1,3}(?:,\d{3})*\)/.test(text)) return true;
                                            if (/\(\d{3}\)\s*\d{3}-\d{4}/.test(text)) return true;
                                            if (/\b(open|closed)\b.*\b\d{1,2}\s?(am|pm)\b/.test(low)) return true;
                                            // Hours-of-operation block: "SundayClosed Monday9 AM..." or "Monday 9 AM – 5 PM"
                                            if (/\b(sunday|monday|tuesday|wednesday|thursday|friday|saturday)/i.test(text) && /\d+\s*(am|pm)/i.test(text) && text.length <= 300) return true;
                                            // Hours with "Open 24 hours" pattern: "SundayOpen 24 hours MondayOpen 24 hours..."
                                            if (/\b(sunday|monday|tuesday|wednesday|thursday|friday|saturday)/i.test(text) && /open\s+24\s+hours/i.test(text) && text.length <= 300) return true;
                                            // Google Plus Code: "WHGX+H2 El Paso, Texas"
                                            if (/^[A-Z0-9]{4,8}\+[A-Z0-9]{1,4}\s/i.test(text) && text.length <= 80) return true;
                                            // Google boilerplate disclaimer
                                            if (/reviews\s+are\s+automatically\s+processed\s+to\s+detect/i.test(text)) return true;
                                            // Pure address: "1234 Street Name, City, ST 12345"
                                            if (/^\d+\s+[\w\s]+(?:ave|st|blvd|dr|rd|way|ct|ln|pl|pkwy|cir)\b/i.test(text) && /\b[a-z]{2}\s+\d{5}\b/i.test(text) && text.length <= 200) return true;
                                            // In-store shopping / pickup / delivery (Google business info)
                                            if (/\b(in-store shopping|in-store pickup|delivery)\b/i.test(text) && text.length <= 120) return true;
                                            if (/\b(car detailing service|flooring store|contractor|plumber|electrician|hvac|roofing contractor|garage door supplier|pressure washing service)\b/.test(low) && text.length <= 200) {
                                                return true;
                                            }
                                            return false;
                                        };

                                        const isLikelyUiNoise = (value) => {
                                            const text = cleanReviewText(value).toLowerCase();
                                            if (!text) return true;
                                            if (text.length < 20) return true;
                                            if (/^photo of reviewer who wrote\b/.test(text)) return true;
                                            if (/^actions for .+ review$/.test(text)) return true;
                                            if (/^more information about the review summary$/.test(text)) return true;
                                            if (/^\d+\s+reviews?$/.test(text)) return true;
                                            if (/\bmentioned\s+in\s+\d+\s+reviews?\b/.test(text)) return true;
                                            if (/^(reviews?|all reviews|write a review|sort|newest|highest|lowest|most relevant|search reviews?|google maps|directions|website|call|share)$/.test(text)) {
                                                return true;
                                            }
                                            if (/^(open|closed)\b/.test(text) && text.length < 40) return true;
                                            if (isLikelyReviewerMeta(text) || isLikelyBusinessCard(text)) return true;
                                            // Quoted review highlight fragments: "Some short quote."
                                            if (/^"[^"]{5,120}"\.?$/.test(text.trim())) return true;
                                            // Google Maps promotional / UI text
                                            if (/get the most out of google/i.test(text)) return true;
                                            if (/sign in|create an account|download the app/i.test(text) && text.length < 80) return true;
                                            if (/\byour? (review|contribution|timeline)\b/i.test(text) && text.length < 80) return true;
                                            return false;
                                        };

                                        // Detect if a DOM node is inside the Review Highlights / Topics widget
                                        // (the section above the actual review list that shows quoted excerpts).
                                        const isInsideHighlightsWidget = (node) => {
                                            let cur = node;
                                            while (cur) {
                                                // Highlights widget containers are NOT inside review containers
                                                if (cur.hasAttribute && cur.hasAttribute('data-review-id')) return false;
                                                const cls = cur.className || '';
                                                // Known classes for the highlights / topics section
                                                if (/\bKNfEk\b|\bKHXTOb\b|\bm6QErb\b/.test(cls) && cur.querySelector && cur.querySelector('[data-review-id]') === null) {
                                                    // This container has no review elements inside — likely the highlights widget
                                                    const textContent = (cur.innerText || cur.textContent || '').trim();
                                                    if (/mentioned\s+in\s+\d+\s+reviews?/i.test(textContent)) return true;
                                                }
                                                cur = cur.parentElement;
                                            }
                                            return false;
                                        };

                                        const reviewContainers = () => {
                                            // Prefer data-review-id (unique per review, no nesting issues)
                                            let containers = Array.from(document.querySelectorAll('div[data-review-id]'));
                                            if (containers.length === 0) {
                                                // Fallback: class-based selectors, but filter out nested duplicates
                                                const raw = Array.from(document.querySelectorAll('div.jftiEf, div.GHT2ce, div[class*="jftiEf"], div[class*="GHT2ce"]'));
                                                // Remove any element that is a descendant of another element in the set
                                                const filtered = [];
                                                for (const el of raw) {
                                                    let isNested = false;
                                                    for (const other of raw) {
                                                        if (other !== el && other.contains(el)) {
                                                            isNested = true;
                                                            break;
                                                        }
                                                    }
                                                    if (!isNested) filtered.push(el);
                                                }
                                                containers = filtered;
                                            }
                                            return containers;
                                        };

                                        const hasReviewControls = () => {
                                            const controls = Array.from(document.querySelectorAll('button,[role="button"],a,input,[role="tab"]'));
                                            let matched = 0;
                                            for (const el of controls) {
                                                const label = ((el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('placeholder'))) || el.textContent || '').trim().toLowerCase();
                                                if (!label) continue;
                                                if (
                                                    label.includes('reviews for')
                                                    || label.includes('all reviews')
                                                    || label.includes('sort reviews')
                                                    || label.includes('search reviews')
                                                    || label.includes('write a review')
                                                ) {
                                                    matched += 1;
                                                    if (matched >= 1) return true;
                                                }
                                            }
                                            return false;
                                        };

                    const findScrollEl = () => {
                      const sels = [
                        'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
                        'div[role="feed"]',
                        '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
                        '.WNBkOb',
                        '.XiKgde',
                        '.m6QErb[role="feed"]',
                        'div.section-layout.section-scrollbox'
                      ];
                      for (const sel of sels) {
                        const el = document.querySelector(sel);
                        if (el && el.scrollHeight > el.clientHeight) return el;
                      }
                      const reviews = Array.from(document.querySelectorAll(reviewContainerSelector));
                      if (reviews.length > 0) {
                          let curr = reviews[0].parentElement;
                          while (curr && curr !== document.body) {
                              const style = window.getComputedStyle(curr);
                              if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                                  return curr;
                              }
                              curr = curr.parentElement;
                          }
                      }
                      return document.scrollingElement || document.documentElement || document.body;
                    };

                    const scrollEl = findScrollEl();
                    const scrollOnce = () => {
                      if (!scrollEl) return;
                      const maxScroll = (scrollEl.scrollHeight || 0) - (scrollEl.clientHeight || 0);
                      const step = Math.max(300, (scrollEl.clientHeight || 600) * 0.7);
                      const next = Math.min(maxScroll, (scrollEl.scrollTop || 0) + step);
                      scrollEl.scrollTop = next;
                    };

                    const expandMore = async () => {
                      const containers = reviewContainers();
                      for (const c of containers) {
                        // CSS approach: remove truncation constraints so the browser
                        // reveals any text already present in the DOM.
                        for (const el of c.querySelectorAll('.wiI7pd, [data-expandable-section], .MyEned')) {
                          el.style.maxHeight = 'none';
                          el.style.overflow = 'visible';
                          el.style.webkitLineClamp = 'unset';
                          el.style.display = '';
                          el.classList.remove('GHWMnc');
                        }
                        // Show hidden direct children inside text wrappers
                        for (const el of c.querySelectorAll('.wiI7pd > *, .MyEned > *')) {
                          if (el.tagName !== 'BUTTON' && el.tagName !== 'A') {
                            el.style.display = '';
                            el.style.visibility = '';
                          }
                        }
                        // aria-expanded approach
                        for (const el of c.querySelectorAll('[aria-expanded="false"]')) {
                          try { el.setAttribute('aria-expanded', 'true'); } catch(e) {}
                        }
                        for (const el of c.querySelectorAll('[data-expandable-section]')) {
                          el.style.display = '';
                          el.style.height = 'auto';
                          el.style.maxHeight = 'none';
                        }
                        // Click "More" / "See more" buttons inside this review card.
                        // These are safe <button> elements that only expand local review
                        // text — they are NOT reviewer profile links. We specifically
                        // exclude any <a> tag and any element whose href contains
                        // /maps/contrib/ to prevent SPA navigation side-effects.
                        for (const el of c.querySelectorAll('button, [role="button"]')) {
                          if (el.tagName === 'A') continue;
                          const href = el.getAttribute('href') || '';
                          if (/\/maps\/contrib\//i.test(href)) continue;
                          const label = (el.getAttribute('aria-label') || el.textContent || '').trim();
                          if (/^(more|see more|\.\.\.\s*more|…\s*more)$/i.test(label)) {
                            try { el.click(); } catch(e) {}
                          }
                        }
                      }
                    };

                                        // Check if a node is inside an owner-reply section
                                        const isInsideOwnerReply = (node) => {
                                            let cur = node;
                                            while (cur) {
                                                // Google Maps owner reply wrapper classes
                                                const cls = cur.className || '';
                                                if (/\bCDe7pd\b/.test(cls)) return true;
                                                // Check for "Response from the owner" label nearby
                                                const prev = cur.previousElementSibling;
                                                if (prev) {
                                                    const prevText = (prev.innerText || prev.textContent || '').trim().toLowerCase();
                                                    if (/response\s+from\s+the\s+owner/i.test(prevText)) return true;
                                                }
                                                // Check aria-label on parent divs
                                                if (cur.getAttribute && /owner/i.test(cur.getAttribute('aria-label') || '')) return true;
                                                cur = cur.parentElement;
                                                // Stop at the review container level
                                                if (cur && cur.hasAttribute && cur.hasAttribute('data-review-id')) break;
                                                if (cur && /\bjftiEf\b|\bGHT2ce\b/.test(cur.className || '')) break;
                                            }
                                            return false;
                                        };

                                        // Returns true if this element is a Google Maps review-attributes block
                                        // ("Positive Quality, Responsiveness Services Light fixture installation…").
                                        // These are NOT customer-written text — they are structured tag lists.
                                        // Heuristic: short, no sentence punctuation, starts with a sentiment word
                                        // or "Services", and contains at least one comma-separated noun phrase.
                                        const isReviewAttributesEl = (el) => {
                                            const text = (el.textContent || '').replace(/\s+/g, ' ').trim();
                                            if (!text || text.length > 400) return false;
                                            // Real review sentences always contain punctuation or verbs.
                                            if (/[.!?]/.test(text)) return false;
                                            if (/^(?:Positive|Negative|Critical)\s+\w/i.test(text)) return true;
                                            if (/^Services\s+\w/i.test(text) &&
                                                /\b(?:installation|repair|replacement|wiring|outlet|switch|fixture|lighting|electrical|plumbing|hvac|heating|cooling|construction|inspection|maintenance)\b/i.test(text)) return true;
                                            return false;
                                        };

                                        const extractBestText = (container) => {
                                            const collectFromSelectors = (selectors) => {
                                                const nodes = [];
                                                for (const sel of selectors) {
                                                    for (const node of container.querySelectorAll(sel)) {
                                                        if (isInsideOwnerReply(node)) continue;
                                                        if (isReviewAttributesEl(node)) continue;
                                                        nodes.push(node);
                                                    }
                                                }
                                                let best = '';
                                                for (const node of nodes) {
                                                    const text = cleanReviewText(node.innerText || node.textContent || '');
                                                    if (isLikelyUiNoise(text)) continue;
                                                    if (text.length > best.length) best = text;
                                                }
                                                return best;
                                            };

                                            let best = collectFromSelectors(primaryReviewBodySelectors);
                                            if (best) return best;
                                            best = collectFromSelectors(secondaryReviewBodySelectors);
                                            if (best) return best;

                                            const nodes = [];
                                            for (const sel of secondaryReviewBodySelectors) {
                                                for (const node of container.querySelectorAll(sel)) {
                                                    if (isInsideOwnerReply(node)) continue;
                                                    if (isReviewAttributesEl(node)) continue;
                                                    nodes.push(node);
                                                }
                                            }
                                            if (!nodes.length) {
                                                nodes.push(container);
                                            }
                                            best = '';
                                            for (const node of nodes) {
                                                if (isReviewAttributesEl(node)) continue;
                                                const text = cleanReviewText(node.innerText || node.textContent || '');
                                                if (isLikelyUiNoise(text)) continue;
                                                if (text.length > best.length) best = text;
                                            }
                                            return best;
                                        };

                    const collect = () => {
                                            const containers = reviewContainers();
                      let idx = 0;
                      for (const c of containers) {
                                                let skip = false;
                        const nameEl = c.querySelector('.d4r55, .kxklSb');
                        if (nameEl) {
                          const name = (nameEl.innerText || nameEl.textContent || '').toLowerCase();
                          if (name.includes('owner')) skip = true;
                        }
                        if (skip) continue;
                                                let text = extractBestText(c);
                        if (!text) continue;

                        // Primary key: stable review id if present; fallback: text prefix + index.
                        let key = c.getAttribute('data-review-id');
                        const prefix = text.substring(0, 160);
                        if (!key) {
                          key = `idx:${idx}:${prefix}`;
                        }
                        idx += 1;
                                                upsertText(key, text);
                                                if (orderedKeys.length >= needed) break;
                      }
                                            return snapshot();
                    };

                                        const collectLoose = () => {
                                            if (!hasReviewControls()) {
                                                return [];
                                            }
                                            const out = [];
                                            const scope = document.querySelector('div[role="feed"]') || findScrollEl() || document.body;
                                            const looseNodes = Array.from(
                                                scope.querySelectorAll('.wiI7pd, [data-expandable-section], .MyEned, .fontBodyMedium, span[dir="auto"], div[dir="auto"]')
                                            );
                                            let idx = 0;
                                            for (const node of looseNodes) {
                                                if (isInsideOwnerReply(node)) continue;
                                                if (isInsideHighlightsWidget(node)) continue;
                                                const text = cleanReviewText(node.innerText || node.textContent || '');
                                                if (isLikelyUiNoise(text)) continue;
                                                const prefix = text.substring(0, 160);
                                                const key = `loose:${prefix}`;
                                                idx += 1;
                                                upsertText(key, text);
                                                out.push(text);
                                                if (orderedKeys.length >= needed) break;
                                            }
                                            return snapshot();
                                        };

                    // Initial attempt without scrolling.
                    await expandMore();

                                        // Pre-scroll warmup: keep the first visible reviews in order and
                                        // give expansion controls a few fast passes before scrolling.
                                        for (let warm = 0; warm < 3 && snapshot().length < needed; warm++) {
                                            await expandMore();
                                            collect();
                                        }

                                                                                let texts = snapshot();
                                        if (texts.length >= needed) return texts.slice(0, needed);

                    // Scroll individual review containers into view to trigger lazy text rendering.
                    // Google Maps only renders .wiI7pd text when the review card is visible.
                    // Use generous sleep (350ms) so the browser has time to render the text element.
                    const containers = reviewContainers();
                    for (let ci = 0; ci < containers.length && snapshot().length < needed && performance.now() - start < maxMs; ci++) {
                        try {
                            containers[ci].scrollIntoView({ behavior: 'instant', block: 'center' });
                        } catch (e) {}
                        await sleep(350);
                        await expandMore();
                        collect();
                        // If text still not rendered, wait a bit more and re-collect
                        if (snapshot().length <= ci) {
                            await sleep(300);
                            collect();
                        }
                    }

                    texts = snapshot();
                    if (texts.length >= needed) return texts.slice(0, needed);

                    // Standard scroll loop for any remaining reviews that need loading
                    while (performance.now() - start < maxMs && texts.length < needed) {
                      scrollOnce();
                      await sleep(300);  // Give reviews time to render after scroll
                      await expandMore();
                      collect();
                                                                                        texts = snapshot();
                      if (texts.length >= needed) break;
                    }

                    // Second-chance: if still short, re-scroll each container with longer waits
                    if (snapshot().length < needed && performance.now() - start < maxMs) {
                        const containers2 = reviewContainers();
                        for (let ci = 0; ci < containers2.length && snapshot().length < needed && performance.now() - start < maxMs; ci++) {
                            try {
                                containers2[ci].scrollIntoView({ behavior: 'instant', block: 'center' });
                            } catch (e) {}
                            await sleep(500);
                            await expandMore();
                            collect();
                        }
                    }

                    // Final loose sweep ONLY if container-based extraction found NOTHING.
                    // collectLoose uses very broad selectors that pick up Google Maps UI text,
                    // so it should only be used as a last resort when no review containers exist.
                    if (snapshot().length === 0) {
                        collectLoose();
                    }
                                        return snapshot().slice(0, needed);
                }
                """,
                {"needed": int(max_snippets) + 4, "maxMs": float(max_ms), "listedCount": int(listed_count) if listed_count is not None else None},
            ) or []
        except Exception as e:
            log_event("reviews", logging.WARNING, "collect_review_texts_fallback_failed", error=str(e))
            review_texts = []
        raw_review_texts = list(review_texts or [])
        # Always allow up to max_snippets through sanitization.
        # Do NOT cap to listed_count — it's often wrong/stale from card view.
        review_texts = sanitize_review_snippets(review_texts, max_items=max_snippets)
        log_event(
            "reviews",
            logging.INFO,
            "collect_review_texts_done",
            max_scroll_cycles=max_scroll_cycles,
            max_snippets=max_snippets,
            snippets=len(review_texts),
            raw_snippets=len(raw_review_texts),
            sample_lengths=[len(s) for s in review_texts[:6]],
        )
        if raw_review_texts and not review_texts:
            log_event(
                "reviews",
                logging.INFO,
                "collect_review_texts_raw_rejected",
                raw_count=len(raw_review_texts),
                raw_samples=raw_review_texts[:6],
            )
        if not review_texts:
            try:
                zero_dom = detail_page.evaluate(
                    """
                    () => {
                        const counts = {};
                        const selectors = [
                            'div[data-review-id]',
                            'div.jftiEf',
                            'div.GHT2ce',
                            '.wiI7pd',
                            '[data-expandable-section]',
                            '.MyEned',
                            '.fontBodyMedium',
                            'div[role="feed"]'
                        ];
                        for (const sel of selectors) {
                            counts[sel] = document.querySelectorAll(sel).length;
                        }
                        const labels = [];
                        const els = Array.from(document.querySelectorAll('button,[role="button"],a,[role="tab"]'));
                        for (const el of els) {
                            const label = ((el.getAttribute && el.getAttribute('aria-label')) || el.textContent || '').trim();
                            if (!label) continue;
                            if (/review/i.test(label) && labels.length < 10) labels.push(label.slice(0, 120));
                        }
                        return {
                            counts,
                            review_controls: labels,
                            title: ((document.querySelector('h1') && document.querySelector('h1').textContent) || '').trim().slice(0, 120),
                        };
                    }
                    """
                ) or {}
                log_event("reviews", logging.INFO, "collect_review_texts_zero_dom", dom=zero_dom)
            except Exception as dom_error:
                log_event("reviews", logging.DEBUG, "collect_review_texts_zero_dom_failed", error=str(dom_error))
        log_event(
            "reviews",
            logging.INFO,
            "collect_review_texts_done",
            snippets=len(review_texts),
            max_snippets=max_snippets,
            max_scroll_cycles=max_scroll_cycles,
        )
        return review_texts[:MAX_REVIEWS_TO_ANALYZE]

    def collect_review_texts_precise(
        detail_page,
        max_snippets: int = TRAINER_REVIEW_SNIPPETS,
        listed_count: int | None = None,
    ) -> list[str]:
        """Slower but more deterministic review extraction used only as a trainer fallback."""
        max_ms = 8000.0 if listed_count is None else float(min(15000, 6000 + min(int(listed_count), 80) * 40))
        try:
            review_texts = detail_page.evaluate(
                r"""
                async (params) => {
                    let { needed, maxMs, listedCount } = params;
                    // Do NOT cap needed to listedCount — it's often wrong.

                    const start = performance.now();
                    const orderedKeys = [];
                    const textByKey = new Map();

                    const reviewContainerSelector = 'div[data-review-id], div.jftiEf, div.GHT2ce, div[class*="jftiEf"], div[class*="GHT2ce"]';
                    const textSelectors = [
                        '.wiI7pd',
                        '[data-expandable-section]',
                        '.MyEned',
                        '.fontBodyMedium',
                        'span[dir="auto"]',
                        'div[dir="auto"]',
                    ];

                    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

                    const shouldReplace = (oldText, newText) => {
                        const prev = String(oldText || '').trim();
                        const next = String(newText || '').trim();
                        if (!prev) return true;
                        if (!next) return false;
                        if (next.length >= prev.length + 12) return true;
                        if (/(?:\.\.\.|…)\s*$/.test(prev) && next.length > prev.length) return true;
                        return false;
                    };

                    const upsertText = (key, text) => {
                        const k = String(key || '').trim();
                        const t = String(text || '').trim();
                        if (!k || !t) return;
                        if (!textByKey.has(k)) {
                            orderedKeys.push(k);
                            textByKey.set(k, t);
                            return;
                        }
                        const prev = textByKey.get(k) || '';
                        if (shouldReplace(prev, t)) {
                            textByKey.set(k, t);
                        }
                    };

                    const snapshot = () => {
                        const out = [];
                        for (const key of orderedKeys) {
                            const text = (textByKey.get(key) || '').trim();
                            if (!text) continue;
                            out.push(text);
                            if (out.length >= needed) break;
                        }
                        return out;
                    };

                    const clean = (value) => {
                        let text = String(value || '').replace(/[\uE000-\uF8FF]/g, ' ');
                        text = text.replace(/\s+/g, ' ').trim();
                        text = text.replace(/(?:\.\.\.\s*more|…\s*more|more)\s*$/i, '').trim();
                        text = text.replace(/\bresponse\s+from\s+the\s+owner\b[\s\S]*$/i, '').trim();
                        text = text.replace(/\s*(?:Price|Service|Quality)\s+assessment\b[\s\S]*$/i, '').trim();
                        // Strip "Like Share" / "+N Like Share" action buttons
                        text = text.replace(/\s*\+?\d*\s*Like\s+Share\s*$/i, '').trim();
                        text = text.replace(/\s*Hover\s+to\s+react\s*$/i, '').trim();
                        // Strip reviewer name + metadata at start
                        text = text.replace(/^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z'.]+){0,3}\s+)?(?:Local\s+Guide\s*[·•]\s*)?(?:\d+\s+reviews?\s*[·•]?\s*)?(?:\d+\s+photos?\s*[·•]?\s*)?(?:(?:\d+|a)\s+(?:day|week|month|year)s?\s+ago\s*)/i, '').trim();
                        return text;
                    };

                    const getContainers = () => {
                        // Prefer data-review-id (unique per review, no nesting issues)
                        let containers = Array.from(document.querySelectorAll('div[data-review-id]'));
                        if (containers.length === 0) {
                            const raw = Array.from(document.querySelectorAll('div.jftiEf, div.GHT2ce, div[class*="jftiEf"], div[class*="GHT2ce"]'));
                            const filtered = [];
                            for (const el of raw) {
                                let isNested = false;
                                for (const other of raw) {
                                    if (other !== el && other.contains(el)) { isNested = true; break; }
                                }
                                if (!isNested) filtered.push(el);
                            }
                            containers = filtered;
                        }
                        return containers;
                    };

                    const findScrollEl = () => {
                        const sels = [
                            'div.m6QErb.DxyBCb.kA9KKe.dS8AEf.XiKgde',
                            'div[role="feed"]',
                            '.m6QErb.DxyBCb.kA9KKe.dS8AEf',
                            '.WNBkOb',
                            '.XiKgde',
                            '.m6QErb[role="feed"]',
                            'div.section-layout.section-scrollbox',
                        ];
                        for (const sel of sels) {
                            const el = document.querySelector(sel);
                            if (el && el.scrollHeight > el.clientHeight) return el;
                        }
                        const reviews = getContainers();
                        if (reviews.length > 0) {
                            let curr = reviews[0].parentElement;
                            while (curr && curr !== document.body) {
                                const style = window.getComputedStyle(curr);
                                if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                                    return curr;
                                }
                                curr = curr.parentElement;
                            }
                        }
                        return document.scrollingElement || document.documentElement || document.body;
                    };

                    const clickMoreInContainer = (container) => {
                        // ZERO-CLICK expansion: force-reveal full review text via CSS/DOM.
                        // No clicks at all — any click risks triggering Maps SPA navigation.
                        for (const el of container.querySelectorAll('.wiI7pd, [data-expandable-section], .MyEned')) {
                            el.style.maxHeight = 'none';
                            el.style.overflow = 'visible';
                            el.style.webkitLineClamp = 'unset';
                            el.style.display = '';
                            el.classList.remove('GHWMnc');
                        }
                        for (const el of container.querySelectorAll('[aria-expanded="false"]')) {
                            try { el.setAttribute('aria-expanded', 'true'); } catch(e) {}
                        }
                        for (const el of container.querySelectorAll('[data-expandable-section]')) {
                            el.style.display = '';
                            el.style.height = 'auto';
                            el.style.maxHeight = 'none';
                        }
                    };

                    // Check if a node is inside an owner-reply section (precise extractor)
                    const isOwnerReplyNode = (node, reviewContainer) => {
                        let cur = node;
                        while (cur && cur !== reviewContainer) {
                            const cls = cur.className || '';
                            if (/\bCDe7pd\b/.test(cls)) return true;
                            const prev = cur.previousElementSibling;
                            if (prev) {
                                const prevText = (prev.innerText || prev.textContent || '').trim();
                                if (/response\s+from\s+the\s+owner/i.test(prevText)) return true;
                            }
                            if (cur.getAttribute && /owner/i.test(cur.getAttribute('aria-label') || '')) return true;
                            cur = cur.parentElement;
                        }
                        return false;
                    };

                    const extractContainerText = (container) => {
                        // Skip containers where the reviewer IS the owner
                        const nameEl = container.querySelector('.d4r55, .kxklSb');
                        if (nameEl && /\bowner\b/i.test(nameEl.innerText || nameEl.textContent || '')) {
                            return '';
                        }
                        // Extract text from sub-elements, skipping owner reply sections
                        let best = '';
                        for (const sel of textSelectors) {
                            for (const node of container.querySelectorAll(sel)) {
                                if (isOwnerReplyNode(node, container)) continue;
                                let text = clean(node.innerText || node.textContent || '');
                                if (!text) continue;
                                // Strip any remaining owner reply text that got concatenated
                                text = text.replace(/\bresponse\s+from\s+the\s+owner\b[\s\S]*$/i, '').trim();
                                text = text.replace(/\(owner\)\s*[\s\S]*$/i, '').trim();
                                if (!text) continue;
                                if (text.length > best.length) best = text;
                            }
                        }
                        // Fallback: use full container text but strip owner reply portion
                        if (!best) {
                            let containerText = clean(container.innerText || container.textContent || '');
                            if (containerText) {
                                containerText = containerText.replace(/\bresponse\s+from\s+the\s+owner\b[\s\S]*$/i, '').trim();
                                containerText = containerText.replace(/\(owner\)\s*[\s\S]*$/i, '').trim();
                                // Strip "Like Share" / "+N Like Share" action buttons
                                containerText = containerText.replace(/\s*\+?\d*\s*Like\s+Share\s*$/i, '').trim();
                                containerText = containerText.replace(/\s*Hover\s+to\s+react\s*$/i, '').trim();
                                // Strip reviewer name + metadata at start
                                containerText = containerText.replace(/^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z'.]+){0,3}\s+)?(?:Local\s+Guide\s*[·•]\s*)?(?:\d+\s+reviews?\s*[·•]?\s*)?(?:\d+\s+photos?\s*[·•]?\s*)?(?:(?:\d+|a)\s+(?:day|week|month|year)s?\s+ago\s*)/i, '').trim();
                                best = containerText;
                            }
                        }
                        return clean(best);
                    };

                    const scrollEl = findScrollEl();
                    let rounds = 0;

                    // First pass: scroll each container into view to trigger lazy text rendering.
                    // Use generous 350ms sleep so Google Maps has time to render .wiI7pd text.
                    const initialContainers = getContainers();
                    for (let ci = 0; ci < initialContainers.length && snapshot().length < needed && performance.now() - start < maxMs; ci++) {
                        try {
                            initialContainers[ci].scrollIntoView({ behavior: 'instant', block: 'center' });
                        } catch (e) {}
                        await sleep(350);
                        clickMoreInContainer(initialContainers[ci]);
                        let text = extractContainerText(initialContainers[ci]);
                        // If text not yet rendered, wait a bit more
                        if (!text || text.length < 20) {
                            await sleep(300);
                            text = extractContainerText(initialContainers[ci]);
                        }
                        if (text && text.length >= 20) {
                            if (!/get the most out of google/i.test(text) && !/^"[^"]{5,120}"\.?$/.test(text.trim()) && !/reviews\s+are\s+automatically\s+processed/i.test(text)) {
                                const dataId = initialContainers[ci].getAttribute('data-review-id') || `precise:init:${ci}`;
                                upsertText(dataId, text);
                            }
                        }
                    }

                    // Second pass: scroll-based extraction for any remaining
                    while (performance.now() - start < maxMs && snapshot().length < needed) {
                        const containers = getContainers();
                        const active = containers.slice(0, Math.max(needed * 3, 18));

                        for (const c of active) {
                            clickMoreInContainer(c);
                        }

                        await sleep(150);

                        let idx = 0;
                        for (const c of active) {
                            const text = extractContainerText(c);
                            if (!text || text.length < 20) continue;
                            // Filter out Google UI noise that slipped through container extraction
                            if (/get the most out of google/i.test(text)) continue;
                            if (/^"[^"]{5,120}"\.?$/.test(text.trim())) continue;
                            if (/reviews\s+are\s+automatically\s+processed/i.test(text)) continue;
                            const dataId = c.getAttribute('data-review-id') || `precise:${rounds}:${idx}`;
                            idx += 1;
                            upsertText(dataId, text);
                            if (snapshot().length >= needed) break;
                        }

                        if (snapshot().length >= needed) break;

                        if (scrollEl) {
                            const maxScroll = Math.max(0, (scrollEl.scrollHeight || 0) - (scrollEl.clientHeight || 0));
                            const step = Math.max(320, (scrollEl.clientHeight || 600) * 0.7);
                            scrollEl.scrollTop = Math.min(maxScroll, (scrollEl.scrollTop || 0) + step);
                        }
                        rounds += 1;
                        await sleep(rounds < 3 ? 350 : 500);
                    }

                    return snapshot().slice(0, needed);
                }
                """,
                {"needed": int(max_snippets) + 4, "maxMs": float(max_ms), "listedCount": int(listed_count) if listed_count is not None else None},
            ) or []
        except Exception as e:
            log_event("reviews", logging.DEBUG, "collect_review_texts_precise_failed", error=str(e))
            review_texts = []

        # Always allow up to max_snippets — do NOT cap to listed_count.
        review_texts = sanitize_review_snippets(list(review_texts or []), max_items=max_snippets)
        log_event(
            "reviews",
            logging.INFO,
            "collect_review_texts_precise_done",
            snippets=len(review_texts),
            max_snippets=max_snippets,
        )
        return review_texts[:MAX_REVIEWS_TO_ANALYZE]

    def resolve_candidate_urls(main_page, selectors: list[str], candidates: list[dict]) -> list[dict]:
        resolved = []
        unresolved_count = 0
        for pos, candidate in enumerate(candidates):
            if candidate.get("href"):
                resolved.append(candidate)
                continue
            card_idx = int(candidate.get("cardIndex", pos))
            current_cards = []
            for sel in selectors:
                current_cards = main_page.query_selector_all(sel)
                if len(current_cards) > card_idx:
                    break
            if card_idx >= len(current_cards):
                unresolved_count += 1
                continue
            try:
                target = current_cards[card_idx]
                name_guess = candidate.get("name_guess") or ""
                if name_guess:
                    for maybe_card in current_cards:
                        try:
                            card_text = (maybe_card.inner_text() or "").strip()
                        except Exception:
                            continue
                        if names_roughly_match(extract_name_guess(card_text), name_guess):
                            target = maybe_card
                            break
                target.scroll_into_view_if_needed(timeout=2500)
                target.click(timeout=3500)
                detail_url = ""
                detail_name = ""
                for _ in range(6):
                    detail_url = main_page.url
                    try:
                        detail_name = (main_page.locator("h1").first.inner_text(timeout=600) or "").strip()
                    except Exception:
                        detail_name = ""
                    if "/maps/place/" in detail_url:
                        break
                    time.sleep(0.08)
                if "/maps/place/" in detail_url and (
                    not candidate.get("name_guess") or names_roughly_match(detail_name, candidate.get("name_guess"))
                ):
                    candidate["href"] = detail_url
                    resolved.append(candidate)
                else:
                    unresolved_count += 1
            except Exception:
                unresolved_count += 1
            finally:
                try:
                    back_btn = main_page.query_selector('button[aria-label^="Back"]')
                    if back_btn:
                        back_btn.click()
                        wait_until(lambda: main_page.query_selector('div[role="feed"]') is not None, timeout_sec=0.6)
                except Exception:
                    pass
        update_status(f"Resolved {sum(1 for c in resolved if c.get('href'))}/{len(candidates)} candidate links.")
        if unresolved_count:
            update_status(f"{unresolved_count} candidates could not be resolved to direct place links and may be skipped.")
        return resolved

    def scrape_place_url(candidate: dict, shared_context=None) -> dict | None:

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        place_url = candidate.get("href") or ""
        if not place_url:
            return None

        started = time.monotonic()
        row = {
            "business_name": candidate.get("name_guess") or "Unknown",
            "address": "Not listed",
            "phone": "Not listed",
            "website": "Not listed",
            "owner_name": "Unknown",
            "confidence_score": 0.0,
            "num_reviews": int(candidate.get("reviewCount") or 0),
            "solo": False,
            "_reviews_text": "",
            "_debug": {
                "place_url": place_url,
                "candidate_review_count": int(candidate.get("reviewCount") or 0),
                "card_review_snippet": (candidate.get("card_review_snippet") or "")[:260],
                "review_tab_opened": False,
                "review_snippet_count": 0,
                "sample_review_snippets": [],
                "limited_view": False,
                "used_card_snippet": False,
                "worker_timed_out": False,
                "detail_review_count": 0,
                "errors": [],
            },
        }

        # Preferred fast path: reuse the already-open browser context.
        if shared_context is not None:
            page_worker = shared_context.new_page()
            try:
                page_worker.goto(place_url, wait_until="domcontentloaded", timeout=12000)
                # Smart wait instead of fixed sleep.
                try:
                    page_worker.wait_for_selector("h1", timeout=2000)
                except Exception:
                    pass

                try:
                    body_text = page_worker.evaluate("() => (document.body && (document.body.innerText || '')) || ''") or ""
                    if "limited view of google maps" in str(body_text).lower():
                        row["_debug"]["limited_view"] = True
                except Exception:
                    pass

                name_el = page_worker.query_selector("h1")
                name_text = (name_el.inner_text() or "").strip() if name_el else ""
                if name_text and name_text.lower() != "results":
                    row["business_name"] = name_text

                addr_el = page_worker.query_selector('[data-item-id="address"]') or page_worker.query_selector('button[data-item-id="address"]')
                if addr_el:
                    row["address"] = normalize_field(addr_el.inner_text(), "Not listed")

                phone_el = page_worker.query_selector('a[href^="tel:"]')
                if phone_el:
                    row["phone"] = normalize_field(phone_el.inner_text().replace(" ", ""), "Not listed")

                website_el = (
                    page_worker.query_selector('a[data-item-id="authority"]')
                    or page_worker.query_selector('a[href^="http"][data-tooltip="Open website"]')
                    or page_worker.query_selector('a[aria-label*="Website"]')
                )
                if website_el:
                    row["website"] = normalize_field(website_el.get_attribute("href"), "Not listed")

                if row["num_reviews"] <= 0:
                    row["num_reviews"] = get_detail_review_count(page_worker)
                row["_debug"]["detail_review_count"] = row["num_reviews"]

                if row["num_reviews"] > 0 and not (MIN_REVIEWS <= row["num_reviews"] <= MAX_REVIEWS):
                    return None

                if time.monotonic() - started > LISTING_TIMEOUT_SECONDS:
                    row["_reviews_text"] = ""
                    row["_debug"]["worker_timed_out"] = True
                    return row

                if row["_debug"].get("limited_view"):
                    snippet = (candidate.get("card_review_snippet") or "").strip()
                    if snippet:
                        row["_reviews_text"] = snippet
                        row["_debug"]["used_card_snippet"] = True
                        row["_debug"]["review_snippet_count"] = 1
                        row["_debug"]["sample_review_snippets"] = [snippet[:260]]
                    return row

                # Block new-tab-opening clicks (reviewer profile links) on worker page
                try:
                    page_worker.evaluate(r"""
                        () => {
                            if (!window.__openBlocked) {
                                window.open = function() { return null; };
                                window.__openBlocked = true;
                            }
                            for (const a of document.querySelectorAll('a[target="_blank"]')) {
                                a.removeAttribute('target');
                            }
                            if (!window.__clickIntercepted) {
                                document.addEventListener('click', function(e) {
                                    const link = e.target.closest('a[target]');
                                    if (link && link.target && link.target !== '_self') {
                                        const href = link.getAttribute('href') || '';
                                        if (/\/maps\/contrib\//i.test(href)) {
                                            e.preventDefault();
                                            e.stopPropagation();
                                        }
                                    }
                                }, true);
                                window.__clickIntercepted = true;
                            }
                        }
                    """)
                except Exception:
                    pass

                row["_debug"]["review_tab_opened"] = open_reviews_tab(page_worker)

                # --- CONDENSED VIEW RECOVERY (worker path) ---
                if row["_debug"]["review_tab_opened"]:
                    try:
                        _cs = page_worker.evaluate(r"""
                            () => {
                                const containers = document.querySelectorAll('div[data-review-id]').length;
                                const hasSortBtn = !!document.querySelector('button[aria-label*="Sort reviews" i]');
                                let expandBtn = null;
                                let expandText = '';
                                for (const el of document.querySelectorAll('button, a, [role="button"], [role="link"]')) {
                                    const text = (el.textContent || el.getAttribute('aria-label') || '').trim();
                                    if (/more\s+reviews?\s*\(\d+\)/i.test(text) ||
                                        /see\s+all\s+\d+\s+reviews/i.test(text) ||
                                        /all\s+\d+\s+reviews/i.test(text) ||
                                        /show\s+\d+\s+more\s+reviews/i.test(text) ||
                                        /\d+\s+more\s+reviews/i.test(text)) {
                                        expandBtn = el;
                                        expandText = text.slice(0, 80);
                                        break;
                                    }
                                }
                                return {
                                    containers: containers,
                                    has_sort: hasSortBtn,
                                    has_expand_btn: !!expandBtn,
                                    expand_text: expandText,
                                    is_condensed: containers <= 4 && !!expandBtn && !hasSortBtn
                                };
                            }
                        """)
                        row["_debug"]["condensed_state"] = _cs
                        log_event("reviews", logging.WARNING, "review_diag_after_tab_worker",
                                  business=row.get("business_name", ""),
                                  condensed_info=_cs)
                        if _cs and _cs.get("is_condensed"):
                            log_event("reviews", logging.WARNING, "condensed_view_detected",
                                      business=row.get("business_name", ""),
                                      containers=_cs.get("containers"),
                                      expand_text=_cs.get("expand_text"))
                            _expanded = page_worker.evaluate(r"""
                                () => {
                                    for (const el of document.querySelectorAll('button, a, [role="button"], [role="link"]')) {
                                        const text = (el.textContent || el.getAttribute('aria-label') || '').trim();
                                        if (/more\s+reviews?\s*\(\d+\)/i.test(text) ||
                                            /see\s+all\s+\d+\s+reviews/i.test(text) ||
                                            /all\s+\d+\s+reviews/i.test(text) ||
                                            /show\s+\d+\s+more\s+reviews/i.test(text) ||
                                            /\d+\s+more\s+reviews/i.test(text)) {
                                            if (el.tagName === 'A' && /\/maps\/contrib\//i.test(el.getAttribute('href') || '')) continue;
                                            try { el.scrollIntoView({block:'center'}); } catch(e) {}
                                            el.click();
                                            return true;
                                        }
                                    }
                                    return false;
                                }
                            """)
                            if _expanded:
                                log_event("reviews", logging.WARNING, "condensed_view_expanded",
                                          business=row.get("business_name", ""))
                                time.sleep(1.5)
                                try:
                                    page_worker.wait_for_selector('div[data-review-id]', timeout=3000)
                                except Exception:
                                    pass
                                _new_c = page_worker.evaluate(
                                    "() => document.querySelectorAll('div[data-review-id]').length")
                                log_event("reviews", logging.WARNING, "condensed_view_after_expand",
                                          business=row.get("business_name", ""),
                                          containers=_new_c)
                                if _new_c <= 4:
                                    open_reviews_tab(page_worker)
                    except Exception as _ce:
                        log_event("reviews", logging.WARNING, "condensed_check_failed",
                                  error=str(_ce))

                if row["_debug"]["review_tab_opened"] and review_snippets_target >= TRAINER_REVIEW_SNIPPETS:
                    row["_debug"]["review_sort_enforced"] = ensure_reviews_sorted_most_relevant(page_worker)
                    # Pre-scroll so the browser lazy-loads review cards before sync extraction
                    _pre_scroll_review_panel(page_worker, steps=8, wait_ms=380)
                if not row["_debug"]["review_tab_opened"]:
                    try:
                        row["_debug"]["review_button_candidates"] = (
                            page_worker.evaluate(
                                """
                                () => {
                                  const out = [];
                                  const els = Array.from(document.querySelectorAll('button,[role="button"],a'));
                                  for (const el of els) {
                                    const label = ((el.getAttribute && el.getAttribute('aria-label')) || el.textContent || '').trim();
                                    if (!label) continue;
                                    if (/review/i.test(label) || /\\b\\d+[\\s,]*\\+?\\s*reviews?\\b/i.test(label)) {
                                      out.push(label.slice(0, 120));
                                    }
                                    if (out.length >= 25) break;
                                  }
                                  return out;
                                }
                                """
                            )
                            or []
                        )
                    except Exception:
                        row["_debug"]["review_button_candidates"] = []
                review_texts = collect_review_texts(page_worker, max_snippets=review_snippets_target, listed_count=row.get("num_reviews"))
                # Always retry with precise extractor if we got fewer reviews than target.
                # Don't gate on listed_count — it's often wrong.
                if len(review_texts) < review_snippets_target:
                    try:
                        row["_debug"]["review_retry_attempted"] = True
                        # Pre-scroll more to lazy-load additional review cards, then use the
                        # async precise extractor which yields between rounds.
                        _pre_scroll_review_panel(page_worker, steps=10, wait_ms=400)
                        precise_texts = collect_review_texts_precise(
                            page_worker,
                            max_snippets=review_snippets_target,
                            listed_count=None,  # Don't pass unreliable listed_count
                        )
                        if len(precise_texts) > len(review_texts):
                            review_texts = precise_texts
                            row["_debug"]["review_precise_retry_improved"] = True
                    except Exception:
                        pass
                if not review_texts:
                    snippet = (candidate.get("card_review_snippet") or "").strip()
                    if snippet:
                        review_texts = [snippet]
                        row["_debug"]["used_card_snippet"] = True
                row["num_reviews"] = max(row["num_reviews"], len(review_texts))
                row["_reviews_text"] = "\n\n".join(review_texts) if review_texts else ""
                row["_debug"]["review_snippet_count"] = len(review_texts)
                row["_debug"]["sample_review_snippets"] = review_texts[:10]
                return row
            except Exception as e:
                row["_debug"]["errors"].append(str(e)[:200])
                return row
            finally:
                try:
                    page_worker.close()
                except Exception:
                    pass

        # No shared context available: skip (this mode is disabled by default).
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=STEALTH_LAUNCH_ARGS)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
        )
        apply_stealth(context)
        configure_fast_context(context)
        page = context.new_page()
        # Close any NEW tabs opened after this point and log them for debugging.
        def _on_new_page(new_page):
            try:
                new_url = new_page.url or "(unknown)"
            except Exception:
                new_url = "(error reading url)"
            log_event("reviews", logging.WARNING, "unexpected_new_tab_opened",
                      new_tab_url=new_url)
            try:
                new_page.close()
            except Exception:
                pass
        context.on("page", _on_new_page)

        try:
            update_status("Opening Google Maps...")
            update_progress(0.05, "Loading Google Maps")
            page.goto("https://www.google.com/maps", wait_until="domcontentloaded", timeout=20000)
            # Smart wait: ensure the search box exists, no fixed sleep.
            try:
                page.wait_for_selector("#searchboxinput", timeout=4000)
            except Exception:
                pass

            # Dismiss cookie/consent if present so search and results are visible
            try:
                for btn_text in ["Accept all", "I agree", "Accept", "Agree"]:
                    btn = page.get_by_role("button", name=btn_text).first
                    try:
                        btn.click(timeout=2000)
                        try:
                            page.wait_for_timeout(250)
                        except Exception:
                            pass
                        break
                    except Exception:
                        continue
            except Exception:
                pass

            # Search using the search box (more reliable than direct URL)
            update_status("Searching for businesses...")
            update_progress(0.08, "Searching")
            search_input = page.query_selector("#searchboxinput")
            if search_input:
                search_input.fill(search_query)
                search_input.press("Enter")
            else:
                search_url = "https://www.google.com/maps/search/" + search_query.replace(" ", "+")
                page.goto(search_url, wait_until="domcontentloaded", timeout=30000)

            # Wait for results panel and let it load
            update_status("Waiting for results to load...")
            update_progress(0.1, "Waiting for results")
            try:
                page.wait_for_selector('div[role="feed"]', timeout=12000)
            except Exception:
                pass
            # Wait for feed to have cards (like a human waiting for results to appear)
            wait_until(lambda: len(page.query_selector_all('div[role="feed"] > div')) >= 3, timeout_sec=1.5)

            # FAST FLOW:
            # 1) Scroll list to load many cards
            # 2) Parse review count from each card text
            # 3) Keep only 2-120 review cards
            # 4) Resolve place URLs
            # 5) Scrape place URLs in parallel workers
            update_status("Loading businesses (scrolling to load more)...")
            scroll_rounds = max(12, max_pages * 3)
            if max_businesses is not None:
                scroll_rounds = min(scroll_rounds, max(6, max_businesses + 4))
            for i in range(scroll_rounds):
                scrollable = page.query_selector('div[role="feed"]')
                if not scrollable:
                    break
                card_count_before = len(page.query_selector_all('div[role="feed"] > div'))
                scrollable.evaluate("el => el.scrollTop = el.scrollHeight")
                # Proceed as soon as new cards appear or after short timeout (human-like)
                wait_until(lambda: len(page.query_selector_all('div[role="feed"] > div')) > card_count_before, timeout_sec=0.5)
                update_status(f"Loading businesses... scroll {i + 1}/{scroll_rounds}")

            card_selectors = ['div.Nv2PK', 'div[role="article"]', 'div[jsaction*="mouseover:pane"]', 'div[role="feed"] > div > div']
            selector_counts = {}
            cards = []
            for sel in card_selectors:
                current = page.query_selector_all(sel)
                selector_counts[sel] = len(current)
                if len(current) > len(cards):
                    cards = current

            update_status("Filtering cards by review count (2-120)...")
            candidate_items = []
            raw_cards = []
            for idx, card in enumerate(cards):
                try:
                    text = (card.inner_text() or "").strip()
                    name_guess = extract_name_guess(text)
                    rc = 0
                    for line in text.splitlines():
                        line = line.strip()
                        if "★" in line or "." in line:
                            rc = parse_review_count_from_text(line)
                            if rc > 0:
                                break
                    if rc <= 0:
                        rc = parse_review_count_from_text(text)

                    href = ""
                    link_el = card.query_selector('a[href*="/maps/place/"], a.hfpxzc, a[href*="maps/place"]')
                    if link_el:
                        href = normalize_place_href(link_el.get_attribute("href"))

                    raw_cards.append(
                        {
                            "cardIndex": idx,
                            "href": href,
                            "reviewCount": rc,
                            "rating": parse_rating_from_text(text),
                            "hasReviewCount": rc > 0,
                            "name_guess": name_guess,
                            "card_review_snippet": extract_card_review_snippet(text),
                            "sampleText": text[:220],
                        }
                    )
                    if MIN_REVIEWS <= rc <= MAX_REVIEWS:
                        candidate_items.append(
                            {
                                "cardIndex": idx,
                                "href": href,
                                "reviewCount": rc,
                                "rating": parse_rating_from_text(text),
                                "name_guess": name_guess,
                                "card_review_snippet": extract_card_review_snippet(text),
                            }
                        )
                except Exception:
                    continue

            # Fail-open if counts parsing found no in-range links
            if not candidate_items:
                # Fallback to first cards; we'll click by card index.
                fallback_n = min(len(cards), max(10, max_pages * 10))
                candidate_items = [{"cardIndex": i, "href": "", "reviewCount": 0, "rating": None} for i in range(fallback_n)]
                update_status(f"No in-range card counts found. Falling back to first {len(candidate_items)} cards.")
            else:
                with_href = sum(1 for c in candidate_items if c.get("href"))
                update_status(
                    f"Found {len(candidate_items)} candidates with {MIN_REVIEWS}-{MAX_REVIEWS} reviews ({with_href} with direct links)."
                )

            # Dedupe by href when available, otherwise by card index.
            deduped = []
            seen_keys = set()
            for c in candidate_items:
                key = c.get("href") if c.get("href") else f"idx:{c.get('cardIndex')}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped.append(c)
            candidate_items = deduped

            if max_businesses:
                candidate_items = candidate_items[:max_businesses]
                update_status(f"Debug mode: scraping only {len(candidate_items)} businesses.")

            # NOTE: We intentionally do NOT resolve/require direct place URLs here.
            # Clicking each card inside the same Maps session is significantly more reliable
            # (Google often serves a "limited view" when directly loading place URLs).

            # Save diagnostics
            try:
                debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_card_filter.json")
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "search_query": search_query,
                            "page_url": page.url,
                            "raw_cards_count": len(raw_cards),
                            "cards_with_counts": sum(1 for c in raw_cards if c.get("hasReviewCount")),
                            "in_range_count": len(candidate_items),
                            "min_reviews": MIN_REVIEWS,
                            "max_reviews": MAX_REVIEWS,
                            "sample_cards": raw_cards[:40],
                            "selector_counts": selector_counts,
                        },
                        f,
                        indent=2,
                    )
            except Exception:
                pass

            total = len(candidate_items)
            if total == 0:
                update_status("No businesses with 2-120 reviews found. Saving screenshot for debugging...")
                try:
                    screenshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_no_links.png")
                    page.screenshot(path=screenshot_path)
                    update_status("No links found. Screenshot saved as debug_no_links.png in the app folder.")
                except Exception:
                    update_status("No business links found. Try 'Show browser window' to see what Maps shows.")
                update_progress(1.0)
                browser.close()
                return []

            update_status(f"Preparing {total} candidate businesses...")
            rows_for_owner = []
            pending_candidates = []
            completed_candidates = 0
            for candidate in candidate_items:
                place_url = candidate.get("href") or ""
                cache_key = f"url:{place_url}|v:{CACHE_SCHEMA_VERSION}" if place_url else ""
                if cache_key and cache_key in cache:
                    cached = cache.get(cache_key, {})
                    cached_ts = int(cached.get("ts", 0))
                    if now_ts - cached_ts <= CACHE_MAX_AGE_SECONDS:
                        cached_row = cached.get("row", {})
                        if isinstance(cached_row, dict):
                            # For trainer pulls (review_snippets_target >= TRAINER_REVIEW_SNIPPETS),
                            # ignore cache entries that have many listed reviews but effectively no
                            # review text saved. This forces a re-scrape so the trainer is not
                            # stuck with 0 snippets for high-review businesses.
                            cached_text = str(cached_row.get("_reviews_text") or "").strip()
                            cached_listed = int(cached_row.get("num_reviews") or 0)
                            # Heuristic: approximate how many snippets we previously stored by
                            # splitting on double newlines. If a high-review business has fewer
                            # than TRAINER_REVIEW_SNIPPETS snippets cached, force a re-scrape
                            # so the trainer can pull a full set.
                            approx_snippets = 0
                            if cached_text:
                                approx_snippets = max(
                                    1,
                                    len([chunk for chunk in cached_text.split("\\n\\n") if chunk.strip()]),
                                )
                            weak_for_trainer = (
                                review_snippets_target >= TRAINER_REVIEW_SNIPPETS
                                and cached_listed >= TRAINER_REVIEW_SNIPPETS
                                and approx_snippets < TRAINER_REVIEW_SNIPPETS
                            )
                            if not weak_for_trainer:
                                cached_row["_debug"] = {
                                    "place_url": place_url,
                                    "source": "cache",
                                    "review_tab_opened": None,
                                    "review_snippet_count": None,
                                    "sample_review_snippets": [],
                                    "worker_timed_out": False,
                                    "detail_review_count": cached_row.get("num_reviews", 0),
                                    "errors": [],
                                }
                                result_key = build_result_key(cached_row, place_url)
                                if result_key and result_key not in seen_result_keys:
                                    seen_result_keys.add(result_key)
                                    results.append(dict(cached_row))
                                    if table_callback:
                                        table_callback(results)
                                completed_candidates += 1
                                update_progress(
                                    0.15 + (0.55 * (completed_candidates / total)),
                                    f"Prepared {completed_candidates}/{total}",
                                )
                                continue
                pending_candidates.append(candidate)

            def scrape_candidate_in_main(main_page, selectors: list[str], candidate: dict) -> dict | None:
                started_local = time.monotonic()
                card_idx = int(candidate.get("cardIndex", 0))
                name_guess = candidate.get("name_guess") or ""
                expected_place_url = normalize_place_href(candidate.get("href") or "")

                # Find a stable target card element in the current list.
                current_cards = []
                for sel in selectors:
                    current_cards = main_page.query_selector_all(sel)
                    if len(current_cards) > card_idx:
                        break
                if not current_cards:
                    return None

                target = current_cards[min(card_idx, len(current_cards) - 1)]
                if expected_place_url:
                    for maybe_card in current_cards:
                        try:
                            link_el = maybe_card.query_selector('a[href*="/maps/place/"], a.hfpxzc, a[href*="maps/place"]')
                            href = normalize_place_href(link_el.get_attribute("href")) if link_el else ""
                        except Exception:
                            href = ""
                        if href and href == expected_place_url:
                            target = maybe_card
                            break
                elif name_guess:
                    for maybe_card in current_cards:
                        try:
                            card_text = (maybe_card.inner_text() or "").strip()
                        except Exception:
                            continue
                        if names_roughly_match(extract_name_guess(card_text), name_guess):
                            target = maybe_card
                            break

                try:
                    target.scroll_into_view_if_needed(timeout=2500)
                    link = target.query_selector('a[href*="/maps/place/"], a.hfpxzc, a[href*="maps/place"]')
                    if link:
                        link.click(timeout=2200, force=True)
                    else:
                        target.click(timeout=2200, force=True)
                except Exception:
                    return None

                # Wait for the detail panel to open: URL change, Reviews tab, or a valid place name in an h1.
                detail_ready = False
                detail_name = ""
                for _ in range(18):
                    try:
                        url = main_page.url or ""
                        if "/maps/place/" in url:
                            detail_ready = True
                            break
                        if main_page.query_selector('button[role="tab"][aria-label*="Reviews" i]'):
                            detail_ready = True
                            break
                        # Panel may show place name before Reviews tab; accept any non-results h1.
                        for sel in ("h1.DUwDvf", "h1.fontHeadlineLarge", "h1"):
                            try:
                                el = main_page.locator("h1").nth(1) if sel == "h1" else main_page.locator(sel).first
                                detail_name = (el.inner_text(timeout=200) or "").strip()
                            except Exception:
                                continue
                            if detail_name and detail_name.lower() not in ("results", "google maps", ""):
                                detail_ready = True
                                break
                        if detail_ready:
                            break
                    except Exception:
                        pass
                    time.sleep(0.1)
                if not detail_ready:
                    return None

                # If we don't have the name yet, read it now (multiple selectors).
                if not detail_name:
                    try:
                        main_page.wait_for_selector("h1", timeout=2000)
                    except Exception:
                        pass
                    name_selectors = [
                        "h1.DUwDvf",
                        "h1.fontHeadlineLarge",
                        "[role='main'] h1",
                        "h1",
                    ]
                    for _ in range(10):
                        for sel in name_selectors:
                            try:
                                el = main_page.locator("h1").nth(1) if sel == "h1" else main_page.locator(sel).first
                                detail_name = (el.inner_text(timeout=300) or "").strip()
                            except Exception:
                                detail_name = ""
                            if detail_name and detail_name.lower() not in ("results", "google maps", ""):
                                break
                        if detail_name:
                            break
                        time.sleep(0.05)

                if not detail_name or detail_name.lower() in ("results", "google maps"):
                    # Did not successfully open details; skip this candidate.
                    return None

                expected_name = (name_guess or "").strip()

                # Maps may not always update the address bar to /maps/place/ immediately.
                # Try to extract a canonical place URL from DOM links as a stable cache key.
                place_url = ""
                try:
                    place_url = normalize_place_href(main_page.evaluate(
                        """
                        () => {
                          const a = document.querySelector('a[href*="/maps/place/"], a.hfpxzc[href*="maps/place"]');
                          return a ? (a.getAttribute('href') || '') : '';
                        }
                        """
                    ))
                except Exception:
                    place_url = ""
                if not place_url:
                    place_url = main_page.url

                if not detail_page_matches_candidate(detail_name, expected_name, place_url, expected_place_url):
                    try:
                        back_btn = main_page.query_selector('button[aria-label^="Back"]')
                        if back_btn:
                            back_btn.click()
                            time.sleep(0.4)
                    except Exception:
                        pass
                    return None

                row = {
                    "business_name": detail_name or name_guess or "Unknown",
                    "address": "Not listed",
                    "phone": "Not listed",
                    "website": "Not listed",
                    "rating": float(candidate.get("rating")) if candidate.get("rating") is not None else None,
                    "owner_name": "Unknown",
                    "confidence_score": 0.0,
                    "num_reviews": int(candidate.get("reviewCount") or 0),
                    "solo": False,
                    "_reviews_text": "",
                    "_debug": {
                        "place_url": place_url,
                        "candidate_review_count": int(candidate.get("reviewCount") or 0),
                        "card_review_snippet": (candidate.get("card_review_snippet") or "")[:260],
                        "review_tab_opened": False,
                        "review_snippet_count": 0,
                        "sample_review_snippets": [],
                        "limited_view": False,
                        "used_card_snippet": False,
                        "worker_timed_out": False,
                        "detail_review_count": 0,
                        "errors": [],
                        "source": "in_session",
                    },
                }

                try:
                    body_text = main_page.evaluate("() => (document.body && (document.body.innerText || '')) || ''") or ""
                    if "limited view of google maps" in str(body_text).lower():
                        row["_debug"]["limited_view"] = True
                except Exception:
                    pass

                try:
                    addr_el = main_page.query_selector('[data-item-id="address"]') or main_page.query_selector('button[data-item-id="address"]')
                    if addr_el:
                        row["address"] = normalize_field(addr_el.inner_text(), "Not listed")
                except Exception:
                    pass

                try:
                    phone_el = main_page.query_selector('a[href^="tel:"]')
                    if phone_el:
                        row["phone"] = normalize_field(phone_el.inner_text().replace(" ", ""), "Not listed")
                except Exception:
                    pass

                try:
                    website_el = (
                        main_page.query_selector('a[data-item-id="authority"]')
                        or main_page.query_selector('a[href^="http"][data-tooltip="Open website"]')
                        or main_page.query_selector('a[aria-label*="Website"]')
                    )
                    if website_el:
                        row["website"] = normalize_field(website_el.get_attribute("href"), "Not listed")
                except Exception:
                    pass

                if row["num_reviews"] <= 0:
                    row["num_reviews"] = get_detail_review_count(main_page)
                row["_debug"]["detail_review_count"] = row["num_reviews"]

                if row["num_reviews"] > 0 and not (MIN_REVIEWS <= row["num_reviews"] <= MAX_REVIEWS):
                    # Navigate back before skipping.
                    try:
                        back_btn = main_page.query_selector('button[aria-label^="Back"]')
                        if back_btn:
                            back_btn.click()
                            wait_until(lambda: main_page.query_selector('div[role="feed"]') is not None, timeout_sec=0.6)
                    except Exception:
                        pass
                    return None

                if time.monotonic() - started_local > LISTING_TIMEOUT_SECONDS:
                    row["_debug"]["worker_timed_out"] = True
                    row["_debug"]["review_pull_gap"] = {
                        "listed": int(row.get("num_reviews") or 0),
                        "pulled": 0,
                        "target": review_snippets_target,
                        "note": "Google Maps page load timed out before reviews could be extracted.",
                    }
                    return row

                if row["_debug"].get("limited_view"):
                    snippet = (candidate.get("card_review_snippet") or "").strip()
                    if snippet:
                        row["_reviews_text"] = snippet
                        row["_debug"]["used_card_snippet"] = True
                        row["_debug"]["review_snippet_count"] = 1
                        row["_debug"]["sample_review_snippets"] = [snippet[:260]]
                    row["_debug"]["review_pull_gap"] = {
                        "listed": int(row.get("num_reviews") or 0),
                        "pulled": 1 if snippet else 0,
                        "target": review_snippets_target,
                        "note": "Google is showing a limited view of Maps (bot detection). Only a card snippet was available.",
                    }
                    return row

                # --- SAFETY: Block any new tabs from opening during review extraction ---
                # This prevents reviewer profile links from causing Maps to reset to
                # condensed 3-review view, even if a click accidentally targets one.
                try:
                    main_page.evaluate(r"""
                        () => {
                            // Block window.open (reviewer profile links)
                            if (!window.__openBlocked) {
                                window.__origOpen = window.open;
                                window.open = function() {
                                    console.warn('[TAB-BLOCK] Blocked window.open call');
                                    return null;
                                };
                                window.__openBlocked = true;
                            }
                            // Neutralize target="_blank" on all links
                            for (const a of document.querySelectorAll('a[target="_blank"]')) {
                                a.removeAttribute('target');
                            }
                            // Intercept future target="_blank" clicks
                            if (!window.__clickIntercepted) {
                                document.addEventListener('click', function(e) {
                                    const link = e.target.closest('a[target]');
                                    if (link && link.target && link.target !== '_self') {
                                        const href = link.getAttribute('href') || '';
                                        if (/\/maps\/contrib\//i.test(href)) {
                                            console.warn('[TAB-BLOCK] Blocked click on contributor link: ' + href);
                                            e.preventDefault();
                                            e.stopPropagation();
                                        }
                                    }
                                }, true);
                                window.__clickIntercepted = true;
                            }
                        }
                    """)
                except Exception:
                    pass

                # --- TAB-LEAK DIAGNOSTIC: snapshot page state before reviews ---
                try:
                    _pre_url = main_page.url or ""
                    _pre_containers = main_page.evaluate("() => document.querySelectorAll('div[data-review-id]').length")
                    _pre_pages = len(context.pages)
                    # Check for condensed "More reviews" / "Show X more" button
                    _condensed_info = main_page.evaluate(r"""
                        () => {
                            const btns = [];
                            for (const el of document.querySelectorAll('button, a, [role="button"]')) {
                                const text = (el.textContent || el.getAttribute('aria-label') || '').trim();
                                if (/more\s+reviews|show\s+\d+\s+more|see\s+all\s+\d+\s+reviews/i.test(text)) {
                                    btns.push({tag: el.tagName, text: text.slice(0,80), href: (el.getAttribute('href')||'').slice(0,80)});
                                }
                            }
                            return {
                                review_containers: document.querySelectorAll('div[data-review-id]').length,
                                has_condensed_button: btns.length > 0,
                                condensed_buttons: btns.slice(0, 3),
                                has_sort_reviews: !!document.querySelector('button[aria-label*="Sort reviews" i]'),
                                has_search_reviews: !!document.querySelector('input[aria-label*="Search reviews" i]'),
                                url: window.location.href.slice(0, 100)
                            };
                        }
                    """)
                    log_event("reviews", logging.WARNING, "review_diag_before_tab",
                              business=detail_name, containers=_pre_containers,
                              pages=_pre_pages, condensed_info=_condensed_info)
                except Exception as _de:
                    log_event("reviews", logging.WARNING, "review_diag_before_tab_failed", error=str(_de))

                row["_debug"]["review_tab_opened"] = open_reviews_tab(main_page)

                # --- TAB-LEAK DIAGNOSTIC: snapshot after opening reviews tab ---
                try:
                    _post_tab_url = main_page.url or ""
                    _post_tab_containers = main_page.evaluate("() => document.querySelectorAll('div[data-review-id]').length")
                    _post_tab_pages = len(context.pages)
                    _post_condensed = main_page.evaluate(r"""
                        () => {
                            const btns = [];
                            for (const el of document.querySelectorAll('button, a, [role="button"]')) {
                                const text = (el.textContent || el.getAttribute('aria-label') || '').trim();
                                if (/more\s+reviews|show\s+\d+\s+more|see\s+all\s+\d+\s+reviews/i.test(text)) {
                                    btns.push({tag: el.tagName, text: text.slice(0,80), href: (el.getAttribute('href')||'').slice(0,80)});
                                }
                            }
                            return {
                                review_containers: document.querySelectorAll('div[data-review-id]').length,
                                has_condensed_button: btns.length > 0,
                                condensed_buttons: btns.slice(0, 3),
                                has_sort_reviews: !!document.querySelector('button[aria-label*="Sort reviews" i]'),
                                has_search_reviews: !!document.querySelector('input[aria-label*="Search reviews" i]'),
                            };
                        }
                    """)
                    log_event("reviews", logging.WARNING, "review_diag_after_tab",
                              business=detail_name,
                              tab_opened=row["_debug"]["review_tab_opened"],
                              containers=_post_tab_containers, pages=_post_tab_pages,
                              condensed_info=_post_condensed)
                except Exception as _de:
                    log_event("reviews", logging.WARNING, "review_diag_after_tab_failed", error=str(_de))

                # --- CONDENSED VIEW RECOVERY ---
                # Google Maps sometimes shows a "condensed" view with only ~3 reviews
                # plus a "More reviews (N)" or "See all reviews" button.  This happens
                # when a reviewer profile link was accidentally clicked (opening a new
                # tab), or when Maps just decides to show the summary.  Detect and fix
                # by clicking the expansion button.
                if row["_debug"]["review_tab_opened"]:
                    try:
                        _condensed_state = main_page.evaluate(r"""
                            () => {
                                const containers = document.querySelectorAll('div[data-review-id]').length;
                                const hasSortBtn = !!document.querySelector('button[aria-label*="Sort reviews" i]');
                                // Look for "More reviews (N)" or "See all N reviews" buttons
                                let expandBtn = null;
                                let expandText = '';
                                for (const el of document.querySelectorAll('button, a, [role="button"], [role="link"]')) {
                                    const text = (el.textContent || el.getAttribute('aria-label') || '').trim();
                                    if (/more\s+reviews?\s*\(\d+\)/i.test(text) ||
                                        /see\s+all\s+\d+\s+reviews/i.test(text) ||
                                        /all\s+\d+\s+reviews/i.test(text) ||
                                        /show\s+\d+\s+more\s+reviews/i.test(text) ||
                                        /\d+\s+more\s+reviews/i.test(text)) {
                                        expandBtn = el;
                                        expandText = text.slice(0, 80);
                                        break;
                                    }
                                }
                                return {
                                    containers: containers,
                                    has_sort: hasSortBtn,
                                    has_expand_btn: !!expandBtn,
                                    expand_text: expandText,
                                    is_condensed: containers <= 4 && !!expandBtn && !hasSortBtn
                                };
                            }
                        """)
                        row["_debug"]["condensed_state"] = _condensed_state
                        if _condensed_state and _condensed_state.get("is_condensed"):
                            log_event("reviews", logging.WARNING, "condensed_view_detected",
                                      business=detail_name,
                                      containers=_condensed_state.get("containers"),
                                      expand_text=_condensed_state.get("expand_text"))
                            # Click the "More reviews" button to expand
                            _expanded = main_page.evaluate(r"""
                                () => {
                                    for (const el of document.querySelectorAll('button, a, [role="button"], [role="link"]')) {
                                        const text = (el.textContent || el.getAttribute('aria-label') || '').trim();
                                        if (/more\s+reviews?\s*\(\d+\)/i.test(text) ||
                                            /see\s+all\s+\d+\s+reviews/i.test(text) ||
                                            /all\s+\d+\s+reviews/i.test(text) ||
                                            /show\s+\d+\s+more\s+reviews/i.test(text) ||
                                            /\d+\s+more\s+reviews/i.test(text)) {
                                            // Skip if it's a reviewer profile link
                                            if (el.tagName === 'A' && /\/maps\/contrib\//i.test(el.getAttribute('href') || '')) continue;
                                            try { el.scrollIntoView({block:'center'}); } catch(e) {}
                                            el.click();
                                            return true;
                                        }
                                    }
                                    return false;
                                }
                            """)
                            if _expanded:
                                log_event("reviews", logging.WARNING, "condensed_view_expanded",
                                          business=detail_name)
                                # Wait for the full reviews to load
                                time.sleep(1.5)
                                # Re-open reviews tab if needed
                                try:
                                    main_page.wait_for_selector('div[data-review-id]', timeout=3000)
                                except Exception:
                                    pass
                                # Check if we now have more containers
                                try:
                                    _new_containers = main_page.evaluate(
                                        "() => document.querySelectorAll('div[data-review-id]').length")
                                    log_event("reviews", logging.WARNING, "condensed_view_after_expand",
                                              business=detail_name, containers=_new_containers)
                                    # If still condensed, try open_reviews_tab again
                                    if _new_containers <= 4:
                                        open_reviews_tab(main_page)
                                except Exception:
                                    pass
                    except Exception as _ce:
                        log_event("reviews", logging.WARNING, "condensed_view_check_failed",
                                  error=str(_ce))

                if row["_debug"]["review_tab_opened"] and review_snippets_target >= TRAINER_REVIEW_SNIPPETS:
                    row["_debug"]["review_sort_enforced"] = ensure_reviews_sorted_most_relevant(main_page)
                    # Pre-scroll so the browser lazy-loads review cards before sync extraction
                    _pre_scroll_review_panel(main_page, steps=8, wait_ms=380)

                    # --- TAB-LEAK DIAGNOSTIC: snapshot after sort + pre-scroll ---
                    try:
                        _post_scroll_containers = main_page.evaluate("() => document.querySelectorAll('div[data-review-id]').length")
                        _post_scroll_pages = len(context.pages)
                        log_event("reviews", logging.WARNING, "review_diag_after_scroll",
                                  business=detail_name, containers=_post_scroll_containers,
                                  pages=_post_scroll_pages)
                    except Exception as _de:
                        log_event("reviews", logging.WARNING, "review_diag_after_scroll_failed", error=str(_de))

                if not row["_debug"]["review_tab_opened"]:
                    # Don't waste time trying to scroll/extract if we couldn't open reviews.
                    snippet = (candidate.get("card_review_snippet") or "").strip()
                    if snippet:
                        row["_reviews_text"] = snippet
                        row["_debug"]["used_card_snippet"] = True
                        row["_debug"]["review_snippet_count"] = 1
                        row["_debug"]["sample_review_snippets"] = [snippet[:260]]
                    else:
                        row["_reviews_text"] = ""
                        row["_debug"]["review_snippet_count"] = 0
                        row["_debug"]["errors"].append("Reviews tab/button could not be opened")
                    row["_debug"]["review_pull_gap"] = {
                        "listed": int(row.get("num_reviews") or 0),
                        "pulled": 1 if snippet else 0,
                        "target": review_snippets_target,
                        "note": "Google Maps reviews tab could not be opened. Only a card snippet was available." if snippet
                               else "Google Maps reviews tab could not be opened and no card snippet was available.",
                    }
                    # Return to list panel for next candidate.
                    try:
                        back_btn = main_page.query_selector('button[aria-label^="Back"]')
                        if back_btn:
                            back_btn.click()
                            wait_until(lambda: main_page.query_selector('div[role="feed"]') is not None, timeout_sec=0.6)
                    except Exception:
                        pass
                    return row

                # DOM is pre-scrolled; collect_review_texts will find cards already loaded
                scroll_cycles = REVIEW_SCROLL_CYCLES
                if review_snippets_target >= TRAINER_REVIEW_SNIPPETS:
                    scroll_cycles = max(REVIEW_SCROLL_CYCLES, 7)
                review_texts = collect_review_texts(
                    main_page,
                    max_snippets=review_snippets_target,
                    max_scroll_cycles=scroll_cycles,
                    listed_count=int(row.get("num_reviews") or 0),
                )
                # Always retry with precise extractor if under target
                if len(review_texts) < review_snippets_target:
                    try:
                        row["_debug"]["review_retry_attempted"] = True
                        _pre_scroll_review_panel(main_page, steps=10, wait_ms=400)
                        precise_texts = collect_review_texts_precise(
                            main_page,
                            max_snippets=review_snippets_target,
                            listed_count=None,
                        )
                        if len(precise_texts) > len(review_texts):
                            review_texts = precise_texts
                            row["_debug"]["review_precise_retry_improved"] = True
                    except Exception:
                        pass

                # Verify page didn't drift to a different business during review collection.
                post_collect_name = ""
                try:
                    post_collect_name = (main_page.locator("h1.DUwDvf").first.inner_text(timeout=800) or "").strip()
                except Exception:
                    pass
                if post_collect_name and detail_name and not names_roughly_match(post_collect_name, detail_name):
                    row["_debug"]["errors"].append(f"Page drifted: expected '{detail_name}', got '{post_collect_name}'")
                    review_texts = []
                if not review_texts:
                    snippet = (candidate.get("card_review_snippet") or "").strip()
                    if snippet:
                        review_texts = [snippet]
                        row["_debug"]["used_card_snippet"] = True
                # Extract owner reply texts for name extraction
                try:
                    owner_replies = detail_page.evaluate(r"""
                        () => {
                            const replies = [];
                            // Look for owner reply sections
                            for (const container of document.querySelectorAll('div[data-review-id]')) {
                                // Find elements with CDe7pd class (owner reply wrapper)
                                for (const el of container.querySelectorAll('.CDe7pd')) {
                                    const text = (el.innerText || el.textContent || '').trim();
                                    if (text.length >= 10) replies.push(text);
                                }
                                // Also check for "Response from the owner" siblings
                                for (const el of container.querySelectorAll('span, div')) {
                                    const t = (el.innerText || '').trim();
                                    if (/response\s+from\s+the\s+owner/i.test(t)) {
                                        const next = el.nextElementSibling;
                                        if (next) {
                                            const replyText = (next.innerText || next.textContent || '').trim();
                                            if (replyText.length >= 10) replies.push(replyText);
                                        }
                                    }
                                }
                            }
                            return replies.slice(0, 10);
                        }
                    """) or []
                except Exception:
                    owner_replies = []
                reply_owner_name = extract_owner_name_from_replies(owner_replies) if owner_replies else None
                row["_reply_owner_name"] = reply_owner_name
                row["_debug"]["reply_owner_name"] = reply_owner_name

                listed_count = int(row.get("num_reviews") or 0)
                pulled_count = len(review_texts)
                row["num_reviews"] = max(listed_count, pulled_count)
                row["_reviews_text"] = "\n\n".join(review_texts) if review_texts else ""
                row["_debug"]["review_snippet_count"] = pulled_count
                row["_debug"]["sample_review_snippets"] = review_texts[:review_snippets_target]
                expected = min(review_snippets_target, listed_count) if listed_count > 0 else review_snippets_target
                if pulled_count < expected and listed_count > pulled_count:
                    if pulled_count <= 3 and listed_count >= 10:
                        reason = "Most reviews are star-only ratings without text."
                    else:
                        reason = "Some reviews may be star-only or too short to extract."
                    row["_debug"]["review_pull_gap"] = {
                        "listed": listed_count,
                        "pulled": pulled_count,
                        "target": review_snippets_target,
                        "note": f"Pulled {pulled_count}/{listed_count} reviews (target {review_snippets_target}). {reason}",
                    }
                elif pulled_count < expected and listed_count <= pulled_count:
                    row["_debug"]["review_pull_gap"] = {
                        "listed": listed_count,
                        "pulled": pulled_count,
                        "target": review_snippets_target,
                        "note": f"Business only has {listed_count} reviews; pulled all available.",
                    }

                # Return to list panel for next candidate.
                try:
                    back_btn = main_page.query_selector('button[aria-label^="Back"]')
                    if back_btn:
                        back_btn.click()
                        wait_until(lambda: main_page.query_selector('div[role="feed"]') is not None, timeout_sec=0.6)
                except Exception:
                    pass

                return row

            def scrape_candidate_via_direct_url(main_page, candidate: dict, review_snippets_target: int) -> dict | None:
                """When in-session click fails, open place URL directly and collect reviews (trainer fallback)."""
                href = (candidate.get("href") or "").strip()
                if not href or "/maps/place/" not in href:
                    return None
                try:
                    main_page.goto(href, wait_until="domcontentloaded", timeout=12000)
                except Exception:
                    return None
                # Proceed when place panel is ready (h1 visible), not after a fixed delay
                try:
                    main_page.wait_for_selector("h1", timeout=3000)
                except Exception:
                    pass
                place_url = normalize_place_href(href)
                row = {
                    "business_name": candidate.get("name_guess") or "Unknown",
                    "address": "Not listed",
                    "phone": "Not listed",
                    "website": "Not listed",
                    "rating": float(candidate.get("rating")) if candidate.get("rating") is not None else None,
                    "owner_name": "Unknown",
                    "confidence_score": 0.0,
                    "num_reviews": int(candidate.get("reviewCount") or 0),
                    "solo": False,
                    "_reviews_text": "",
                    "_debug": {
                        "place_url": place_url,
                        "candidate_review_count": int(candidate.get("reviewCount") or 0),
                        "card_review_snippet": (candidate.get("card_review_snippet") or "")[:260],
                        "review_tab_opened": False,
                        "review_snippet_count": 0,
                        "sample_review_snippets": [],
                        "limited_view": False,
                        "used_card_snippet": False,
                        "worker_timed_out": False,
                        "detail_review_count": 0,
                        "errors": [],
                        "source": "direct_url_fallback",
                    },
                }
                try:
                    body_text = main_page.evaluate("() => (document.body && (document.body.innerText || '')) || ''") or ""
                    if "limited view of google maps" in str(body_text).lower():
                        row["_debug"]["limited_view"] = True
                except Exception:
                    pass
                try:
                    name_el = main_page.query_selector("h1")
                    if name_el:
                        name_text = (name_el.inner_text() or "").strip()
                        if name_text and name_text.lower() not in ("results", "google maps"):
                            row["business_name"] = name_text
                except Exception:
                    pass
                try:
                    addr_el = main_page.query_selector('[data-item-id="address"]') or main_page.query_selector('button[data-item-id="address"]')
                    if addr_el:
                        row["address"] = normalize_field(addr_el.inner_text(), "Not listed")
                except Exception:
                    pass
                try:
                    phone_el = main_page.query_selector('a[href^="tel:"]')
                    if phone_el:
                        row["phone"] = normalize_field(phone_el.inner_text().replace(" ", ""), "Not listed")
                except Exception:
                    pass
                try:
                    website_el = (
                        main_page.query_selector('a[data-item-id="authority"]')
                        or main_page.query_selector('a[href^="http"][data-tooltip="Open website"]')
                        or main_page.query_selector('a[aria-label*="Website"]')
                    )
                    if website_el:
                        row["website"] = normalize_field(website_el.get_attribute("href"), "Not listed")
                except Exception:
                    pass
                if row["num_reviews"] <= 0:
                    row["num_reviews"] = get_detail_review_count(main_page)
                row["_debug"]["detail_review_count"] = row["num_reviews"]
                if row["num_reviews"] > 0 and not (MIN_REVIEWS <= row["num_reviews"] <= MAX_REVIEWS):
                    try:
                        back_btn = main_page.query_selector('button[aria-label^="Back"]')
                        if back_btn:
                            back_btn.click()
                            wait_until(lambda: main_page.query_selector('div[role="feed"]') is not None, timeout_sec=0.6)
                    except Exception:
                        pass
                    return None
                if row["_debug"].get("limited_view"):
                    snippet = (candidate.get("card_review_snippet") or "").strip()
                    if snippet:
                        row["_reviews_text"] = snippet
                        row["_debug"]["used_card_snippet"] = True
                        row["_debug"]["review_snippet_count"] = 1
                        row["_debug"]["sample_review_snippets"] = [snippet[:260]]
                    row["_debug"]["review_pull_gap"] = {
                        "listed": int(row.get("num_reviews") or 0),
                        "pulled": 1 if snippet else 0,
                        "target": review_snippets_target,
                        "note": "Google is showing a limited view of Maps (bot detection). Only a card snippet was available.",
                    }
                    try:
                        back_btn = main_page.query_selector('button[aria-label^="Back"]')
                        if back_btn:
                            back_btn.click()
                            wait_until(lambda: main_page.query_selector('div[role="feed"]') is not None, timeout_sec=0.6)
                    except Exception:
                        pass
                    return row
                # Block new-tab-opening clicks (reviewer profile links)
                try:
                    main_page.evaluate(r"""
                        () => {
                            if (!window.__openBlocked) {
                                window.open = function() { return null; };
                                window.__openBlocked = true;
                            }
                            for (const a of document.querySelectorAll('a[target="_blank"]')) {
                                a.removeAttribute('target');
                            }
                            if (!window.__clickIntercepted) {
                                document.addEventListener('click', function(e) {
                                    const link = e.target.closest('a[target]');
                                    if (link && link.target && link.target !== '_self') {
                                        const href = link.getAttribute('href') || '';
                                        if (/\/maps\/contrib\//i.test(href)) {
                                            e.preventDefault();
                                            e.stopPropagation();
                                        }
                                    }
                                }, true);
                                window.__clickIntercepted = true;
                            }
                        }
                    """)
                except Exception:
                    pass
                row["_debug"]["review_tab_opened"] = open_reviews_tab(main_page)
                if row["_debug"]["review_tab_opened"] and review_snippets_target >= TRAINER_REVIEW_SNIPPETS:
                    row["_debug"]["review_sort_enforced"] = ensure_reviews_sorted_most_relevant(main_page)
                    # Pre-scroll so the browser lazy-loads review cards before sync extraction
                    _pre_scroll_review_panel(main_page, steps=8, wait_ms=380)
                scroll_cycles = max(REVIEW_SCROLL_CYCLES, 7) if review_snippets_target >= TRAINER_REVIEW_SNIPPETS else REVIEW_SCROLL_CYCLES
                review_texts = collect_review_texts(
                    main_page,
                    max_snippets=review_snippets_target,
                    max_scroll_cycles=scroll_cycles,
                    listed_count=int(row.get("num_reviews") or 0),
                )
                # Always retry with precise extractor if under target
                if len(review_texts) < review_snippets_target:
                    try:
                        row["_debug"]["review_retry_attempted"] = True
                        _pre_scroll_review_panel(main_page, steps=10, wait_ms=400)
                        precise_texts = collect_review_texts_precise(
                            main_page,
                            max_snippets=review_snippets_target,
                            listed_count=None,
                        )
                        if len(precise_texts) > len(review_texts):
                            review_texts = precise_texts
                            row["_debug"]["review_precise_retry_improved"] = True
                    except Exception:
                        pass
                if not review_texts:
                    snippet = (candidate.get("card_review_snippet") or "").strip()
                    if snippet:
                        review_texts = [snippet]
                        row["_debug"]["used_card_snippet"] = True
                pulled_count = len(review_texts)
                row["_debug"]["review_tab_opened"] = pulled_count > 0  # We opened tab and collected
                row["_reviews_text"] = "\n\n".join(review_texts) if review_texts else ""
                row["_debug"]["review_snippet_count"] = pulled_count
                row["_debug"]["sample_review_snippets"] = review_texts[:review_snippets_target]
                listed_count = int(row.get("num_reviews") or 0)
                expected = min(review_snippets_target, listed_count) if listed_count > 0 else review_snippets_target
                if pulled_count < expected and listed_count > pulled_count:
                    if pulled_count <= 3 and listed_count >= 10:
                        reason = "Most reviews are star-only ratings without text."
                    else:
                        reason = "Some reviews may be star-only or too short to extract."
                    row["_debug"]["review_pull_gap"] = {
                        "listed": listed_count,
                        "pulled": pulled_count,
                        "target": review_snippets_target,
                        "note": f"Pulled {pulled_count}/{listed_count} reviews (target {review_snippets_target}). {reason}",
                    }
                elif pulled_count < expected and listed_count <= pulled_count:
                    row["_debug"]["review_pull_gap"] = {
                        "listed": listed_count,
                        "pulled": pulled_count,
                        "target": review_snippets_target,
                        "note": f"Business only has {listed_count} reviews; pulled all available.",
                    }
                try:
                    back_btn = main_page.query_selector('button[aria-label^="Back"]')
                    if back_btn:
                        back_btn.click()
                        # Proceed when list is back, not after fixed delay
                        wait_until(lambda: main_page.query_selector('div[role="feed"]') is not None, timeout_sec=0.6)
                except Exception:
                    pass
                return row

            if pending_candidates:
                if USE_WORKER_BROWSERS:
                    worker_count = min(LISTING_WORKERS, len(pending_candidates))
                    update_status(f"Scraping {len(pending_candidates)} listings with {worker_count} workers...")
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        future_map = {executor.submit(scrape_place_url, candidate): candidate for candidate in pending_candidates}
                        for future in as_completed(future_map):
                            candidate = future_map[future]
                            completed_candidates += 1
                            try:
                                row = future.result()
                            except Exception:
                                row = None

                            if row:
                                place_url = row.get("_debug", {}).get("place_url") or candidate.get("href") or ""
                                result_key = build_result_key(row, place_url or f"idx:{candidate.get('cardIndex')}")
                                if result_key and result_key not in seen_result_keys:
                                    seen_result_keys.add(result_key)
                                    results.append(row)
                                    rows_for_owner.append((len(results) - 1, place_url))
                                    if table_callback:
                                        table_callback(results)

                            update_progress(
                                0.15 + (0.55 * (completed_candidates / total)),
                                f"Scraped {completed_candidates}/{total}",
                            )
                else:
                    update_status(f"Scraping {len(pending_candidates)} listings in-session (reliable mode)...")
                    for candidate in pending_candidates:
                        completed_candidates += 1
                        row = None
                        try:
                            row = scrape_candidate_in_main(page, card_selectors, candidate)
                        except Exception:
                            row = None
                        if row and review_snippets_target >= TRAINER_REVIEW_SNIPPETS:
                            debug_info = row.get("_debug", {}) or {}
                            pulled_count = int(debug_info.get("review_snippet_count") or 0)
                            listed_count = int(row.get("num_reviews") or 0)
                            thin_in_session = (
                                listed_count >= TRAINER_REVIEW_SNIPPETS
                                and pulled_count < min(review_snippets_target, listed_count)
                            )
                            if thin_in_session and (candidate.get("href") or "").strip():
                                try:
                                    fallback_row = scrape_candidate_via_direct_url(page, candidate, review_snippets_target)
                                except Exception:
                                    fallback_row = None
                                if fallback_row:
                                    fallback_pulled = int((fallback_row.get("_debug", {}) or {}).get("review_snippet_count") or 0)
                                    if fallback_pulled > pulled_count:
                                        row = fallback_row
                        if not row and (candidate.get("href") or "").strip() and review_snippets_target >= TRAINER_REVIEW_SNIPPETS:
                            try:
                                row = scrape_candidate_via_direct_url(page, candidate, review_snippets_target)
                            except Exception:
                                pass
                        if not row:
                            # Fallback: keep the candidate as a "card-only" row so we don't drop it entirely.
                            snippet = (candidate.get("card_review_snippet") or "").strip()
                            if snippet or (candidate.get("name_guess") or "").strip():
                                row = {
                                    "business_name": candidate.get("name_guess") or "Unknown",
                                    "address": "Not listed",
                                    "phone": "Not listed",
                                    "website": "Not listed",
                                    "rating": float(candidate.get("rating")) if candidate.get("rating") is not None else None,
                                    "owner_name": "Unknown",
                                    "confidence_score": 0.0,
                                    "num_reviews": int(candidate.get("reviewCount") or 0),
                                    "solo": False,
                                    "_reviews_text": snippet,
                                    "_debug": {
                                        "place_url": candidate.get("href") or "",
                                        "candidate_review_count": int(candidate.get("reviewCount") or 0),
                                        "card_review_snippet": snippet[:260],
                                        "review_tab_opened": False,
                                        "review_snippet_count": 1 if snippet else 0,
                                        "sample_review_snippets": [snippet[:260]] if snippet else [],
                                        "limited_view": False,
                                        "used_card_snippet": bool(snippet),
                                        "worker_timed_out": False,
                                        "detail_review_count": int(candidate.get("reviewCount") or 0),
                                        "errors": ["In-session click failed; using card snippet fallback"],
                                        "source": "card_only",
                                    },
                                }

                        if row:
                            place_url = row.get("_debug", {}).get("place_url") or candidate.get("href") or ""
                            result_key = build_result_key(row, place_url or f"idx:{candidate.get('cardIndex')}")
                            if result_key and result_key not in seen_result_keys:
                                seen_result_keys.add(result_key)
                                results.append(row)
                                rows_for_owner.append((len(results) - 1, place_url))
                                if table_callback:
                                    table_callback(results)

                                # Inline owner detection + early stop when we hit the target.
                                if run_owner_detection and target_leads is not None and (row.get("_reviews_text") or "").strip():
                                    try:
                                        det = detect_owner(
                                            row["_reviews_text"],
                                            row.get("business_name", ""),
                                        )
                                    except Exception:
                                        det = {"owner_name": None, "solo": False, "confidence": 0.0, "reason": "Owner detection failed"}
                                    row["owner_name"] = det.get("owner_name") or "Unknown"
                                    row["confidence_score"] = det.get("confidence", 0)
                                    row["solo"] = det.get("solo", False)
                                    # Fallback: if Ollama didn't find a name, try reply signature name
                                    reply_name = row.get("_reply_owner_name")
                                    _oc = str(row["owner_name"]).strip().lower()
                                    if reply_name and _oc in ("", "unknown", "none", "null"):
                                        reviews_text_inline = (row.get("_reviews_text") or "")
                                        if owner_has_person_context(reviews_text_inline, reply_name) or count_name_mentions(reviews_text_inline, reply_name) >= 1:
                                            row["owner_name"] = reply_name
                                            row["solo"] = True
                                            row["confidence_score"] = max(row["confidence_score"], 0.80)
                                        elif reply_name:
                                            row["owner_name"] = reply_name
                                            row["solo"] = True
                                            row["confidence_score"] = max(row["confidence_score"], 0.70)
                                    row.setdefault("_debug", {})["owner_detection"] = {
                                        "owner_name": row["owner_name"],
                                        "confidence_score": row["confidence_score"],
                                        "solo": row["solo"],
                                        "reason": det.get("reason", ""),
                                    }
                                    if is_qualified_lead_row(row):
                                        qualified_found += 1
                                    if target_leads is not None and qualified_found >= target_leads:
                                        update_status(f"Target of {target_leads} qualified leads reached; stopping early.")
                                        # We have enough leads; return immediately instead of scraping more.
                                        return results

                            # Cache the scraped row by URL when we have one.
                            if place_url:
                                cache[f"url:{place_url}|v:{CACHE_SCHEMA_VERSION}"] = {
                                    "ts": now_ts,
                                    "row": {k: v for k, v in row.items() if not str(k).startswith("_")},
                                }

                        update_progress(
                            0.15 + (0.55 * (completed_candidates / total)),
                            f"Scraped {completed_candidates}/{total}",
                        )

                    # If we already hit the target within this batch, stop processing more candidates.
                    # (Handled by early return above.)

            # Parallelize owner detection for any remaining rows that still need it.
            owner_candidates = []
            if run_owner_detection:
                owner_candidates = [
                    (i, url)
                    for i, url in rows_for_owner
                    if (results[i].get("_reviews_text") or "").strip()
                    and not results[i].get("_debug", {}).get("owner_detection")
                ]
            if owner_candidates and not (target_leads is not None and qualified_found >= target_leads):
                update_status(f"Running owner detection in parallel ({OWNER_WORKERS} workers)...")
                with ThreadPoolExecutor(max_workers=OWNER_WORKERS) as executor:
                    future_map = {
                        executor.submit(
                            detect_owner,
                            results[i]["_reviews_text"],
                            results[i].get("business_name", ""),
                        ): (i, url)
                        for i, url in owner_candidates
                    }
                    done_count = 0
                    for fut in as_completed(future_map):
                        i, url = future_map[fut]
                        done_count += 1
                        try:
                            det = fut.result()
                        except Exception:
                            det = {"owner_name": None, "solo": False, "confidence": 0.0}
                        results[i]["owner_name"] = det.get("owner_name") or "Unknown"
                        results[i]["confidence_score"] = det.get("confidence", 0)
                        results[i]["solo"] = det.get("solo", False)
                        if "_debug" not in results[i]:
                            results[i]["_debug"] = {}
                        results[i]["_debug"]["owner_detection"] = {
                            "owner_name": results[i]["owner_name"],
                            "confidence_score": results[i]["confidence_score"],
                            "solo": results[i]["solo"],
                            "reason": det.get("reason", ""),
                        }
                        if is_qualified_lead_row(results[i]):
                            qualified_found += 1
                        if table_callback:
                            table_callback(results)
                        update_status(f"Owner detection {done_count}/{len(owner_candidates)} complete")

                        # Cache final row by URL
                        if url:
                            cache[f"url:{url}|v:{CACHE_SCHEMA_VERSION}"] = {
                                "ts": now_ts,
                                "row": {
                                    k: v for k, v in results[i].items() if not str(k).startswith("_")
                                },
                            }
                        if target_leads is not None and qualified_found >= target_leads:
                            update_status(
                                f"Target of {target_leads} qualified leads reached during owner detection; stopping early."
                            )
                            # Early return: we've already got enough qualified leads.
                            return results

            # Save per-run debug report with reject reasons.
            try:
                debug_report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_lead_report.json")
                debug_rows = []
                for r in results:
                    debug_info = dict(r.get("_debug", {}))
                    debug_info["qualified_lead"] = (
                        bool(r.get("solo"))
                        and float(r.get("confidence_score") or 0) >= MIN_CONFIDENCE
                        and str(r.get("owner_name") or "").strip().lower() not in ("", "unknown", "none", "null")
                    )
                    debug_info["reject_reason"] = get_reject_reason(r)
                    debug_rows.append(
                        {
                            "business_name": r.get("business_name", ""),
                            "address": r.get("address", ""),
                            "phone": r.get("phone", ""),
                            "website": r.get("website", ""),
                            "owner_name": r.get("owner_name", ""),
                            "confidence_score": r.get("confidence_score", 0),
                            "solo": r.get("solo", False),
                            "num_reviews": r.get("num_reviews", 0),
                            "debug": debug_info,
                        }
                    )
                with open(debug_report_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "search_query": search_query,
                            "qualified_count": sum(1 for r in debug_rows if r["debug"].get("qualified_lead")),
                            "processed_count": len(debug_rows),
                            "rows": debug_rows,
                        },
                        f,
                        indent=2,
                    )
            except Exception:
                pass

            # Cleanup temp fields and persist cache
            for r in results:
                if "_reviews_text" in r:
                    r.pop("_reviews_text", None)
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            except Exception:
                pass

            update_progress(1.0, "Complete")
            update_status("Scraping finished.")

        except Exception as e:
            update_status(f"Error: {str(e)}")
            raise
        finally:
            browser.close()

    return results


def scrape_dataforseo(
    *,
    city: str,
    niche: str,
    max_businesses: int | None,
    progress_callback=None,
    status_callback=None,
    dataforseo_location_name: str,
    reviews_depth: int,
    reviews_priority: int,
) -> list[dict]:
    _log = get_logger("dataforseo")
    login = (os.getenv("DATAFORSEO_LOGIN") or "").strip()
    password = (os.getenv("DATAFORSEO_PASSWORD") or "").strip()
    if not login or not password:
        raise RuntimeError("Missing DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD in environment (.env)")

    def update_status(msg):
        if status_callback:
            status_callback(msg)

    def update_progress(pct, msg=None):
        if progress_callback:
            progress_callback(pct, msg)

    keyword = f"{niche} in {city}"
    update_status("DataForSEO: fetching Google Maps results...")
    update_progress(0.1, "DataForSEO: maps search")
    places = maps_search(
        login=login,
        password=password,
        keyword=keyword,
        location_name=dataforseo_location_name,
        language_code="en",
        depth=100,
    )
    _log.info("maps_search returned %d places for '%s'", len(places), keyword)

    # Filter obvious non-targets before paying for reviews.
    filtered_places = []
    skipped_no_reviews = 0
    skipped_review_range = 0
    skipped_hours = 0
    for p in places:
        rc = int(p.reviews_count or 0)
        if rc <= 0:
            skipped_no_reviews += 1
            continue
        # Reuse the same 2–120 window you had in the Playwright path.
        if rc < MIN_REVIEWS or rc > MAX_REVIEWS:
            skipped_review_range += 1
            continue
        # NEW: only consider businesses whose local work hours show closing around 5–6pm.
        if not closes_between_5_and_6_local(getattr(p, "work_hours", None)):
            skipped_hours += 1
            continue
        filtered_places.append(p)
    _log.info("Filtering: %d passed, %d no reviews, %d outside %d-%d range, %d failed hours filter",
              len(filtered_places), skipped_no_reviews, skipped_review_range, MIN_REVIEWS, MAX_REVIEWS, skipped_hours)

    # Cap how many businesses we ever send to Reviews API to keep credit usage predictable.
    effective_max = max_businesses if max_businesses is not None else 50
    places = filtered_places[:effective_max]
    _log.info("Processing %d businesses (cap=%s)", len(places), effective_max)

    results: list[dict] = []
    rows_for_owner: list[tuple[int, str]] = []
    reviews_fetched_count = 0
    reviews_empty_count = 0
    reviews_error_count = 0

    # --- Batch-post all review tasks at once, then poll in parallel ---
    update_status(f"DataForSEO: posting {len(places)} review tasks...")
    update_progress(0.10, f"Posting {len(places)} review tasks")

    review_reqs = [
        ReviewRequest(index=idx, place_id=p.place_id, cid=p.cid, keyword_fallback=p.title)
        for idx, p in enumerate(places)
    ]
    try:
        batch_results = fetch_reviews_batch(
            login=login,
            password=password,
            requests_list=review_reqs,
            location_name=dataforseo_location_name,
            language_code="en",
            depth=reviews_depth,
            priority=reviews_priority,
        )
    except DataForSeoError as e:
        _log.error("Batch review fetch failed entirely: %s", str(e)[:300])
        batch_results = {}

    update_status(f"DataForSEO: processing {len(places)} results...")
    update_progress(0.70, "Building rows")

    for idx, p in enumerate(places):
        reviews_texts, debug_reviews_payload = batch_results.get(idx, ([], {"error": "not in batch"}))

        if isinstance(debug_reviews_payload, dict) and "error" in debug_reviews_payload:
            reviews_error_count += 1
            _log.error("fetch_reviews FAILED for '%s': %s", p.title[:40], debug_reviews_payload["error"])
            reviews: list[str] = []
        elif reviews_texts:
            reviews = reviews_texts
            reviews_fetched_count += 1
        else:
            reviews = []
            reviews_empty_count += 1
            _log.warning("fetch_reviews returned 0 reviews for '%s' (place_id=%s, cid=%s)",
                         p.title[:40], p.place_id, p.cid)

        reviews_text = "\n\n".join(reviews) if reviews else ""
        _log.info("[%d/%d] '%s': %d reviews fetched, %d chars text",
                  idx + 1, len(places), p.title[:35], len(reviews), len(reviews_text))

        # Extract owner name from reply signatures (free signal, no Ollama needed)
        owner_answers = (debug_reviews_payload or {}).get("_owner_answers", [])
        reply_owner_name = extract_owner_name_from_replies(owner_answers) if owner_answers else None

        row = {
            "business_name": p.title,
            "address": p.address or "Not listed",
            "phone": p.phone or "Not listed",
            "website": p.website or "Not listed",
            "rating": float(p.rating_value) if p.rating_value is not None else None,
            "owner_name": "Unknown",
            "confidence_score": 0.0,
            "num_reviews": int(p.reviews_count or 0),
            "solo": False,
            "_reviews_text": reviews_text,
            "_reply_owner_name": reply_owner_name,
            "_debug": {
                "source": "dataforseo",
                "place_id": p.place_id,
                "cid": p.cid,
                "dataforseo_location_name": dataforseo_location_name,
                "reviews_depth": reviews_depth,
                "reviews_priority": reviews_priority,
                "review_snippet_count": len(reviews),
                "sample_review_snippets": reviews[:10],
                "dataforseo_reviews_payload": debug_reviews_payload,
                "reply_owner_name": reply_owner_name,
            },
        }
        results.append(row)
        rows_for_owner.append((len(results) - 1, p.place_id or p.cid or p.title))

    _log.info("Reviews summary: %d fetched OK, %d empty, %d errors out of %d businesses",
              reviews_fetched_count, reviews_empty_count, reviews_error_count, len(places))

    # Owner detection — run on every business that has review text.
    owner_candidates = [
        (i, key)
        for i, key in rows_for_owner
        if (results[i].get("_reviews_text") or "").strip()
    ]
    _log.info("Owner detection: %d/%d businesses have review text → running detection",
              len(owner_candidates), len(results))
    if len(owner_candidates) == 0 and len(results) > 0:
        _log.warning("NO businesses have review text! Owner detection will be SKIPPED for all %d businesses. "
                      "This is likely why all confidence_scores are 0.", len(results))

    if owner_candidates:
        update_status(f"Running owner detection in parallel ({OWNER_WORKERS} workers)...")
        with ThreadPoolExecutor(max_workers=OWNER_WORKERS) as executor:
            future_map = {
                executor.submit(detect_owner, results[i]["_reviews_text"], results[i].get("business_name", "")): i
                for i, _key in owner_candidates
            }
            done = 0
            for fut in as_completed(future_map):
                i = future_map[fut]
                done += 1
                try:
                    det = fut.result()
                except Exception as exc:
                    det = {"owner_name": None, "solo": False, "confidence": 0.0, "reason": "Owner detection failed"}
                    _log.error("detect_owner EXCEPTION for '%s': %s", results[i].get("business_name", "?")[:40], str(exc)[:200])
                results[i]["owner_name"] = det.get("owner_name") or "Unknown"
                results[i]["confidence_score"] = det.get("confidence", 0)
                results[i]["solo"] = det.get("solo", False)
                _log.info("detect_owner '%s': owner=%s conf=%.2f solo=%s reason=%s",
                          results[i].get("business_name", "?")[:35],
                          results[i]["owner_name"], results[i]["confidence_score"],
                          results[i]["solo"], det.get("reason", "")[:80])

                # Fallback: if detection didn't find a name, try the reply signature name
                reply_name = results[i].get("_reply_owner_name")
                owner_check = str(results[i]["owner_name"]).strip().lower()
                if reply_name and owner_check in ("", "unknown", "none", "null"):
                    reviews_text = results[i].get("_reviews_text", "")
                    if owner_has_person_context(reviews_text, reply_name) or count_name_mentions(reviews_text, reply_name) >= 1:
                        results[i]["owner_name"] = reply_name
                        results[i]["solo"] = True
                        results[i]["confidence_score"] = max(results[i]["confidence_score"], 0.80)
                        det["reason"] = (det.get("reason", "") + f" | Owner name from reply signature: {reply_name}").strip(" |")
                        _log.info("Reply signature fallback for '%s': using '%s' (in reviews)", results[i].get("business_name", "?")[:35], reply_name)
                    elif reply_name:
                        # Name found in replies but not in reviews — still use it but lower confidence
                        results[i]["owner_name"] = reply_name
                        results[i]["solo"] = True
                        results[i]["confidence_score"] = max(results[i]["confidence_score"], 0.70)
                        det["reason"] = (det.get("reason", "") + f" | Owner name from reply signature only: {reply_name}").strip(" |")
                        _log.info("Reply signature fallback for '%s': using '%s' (NOT in reviews, conf=0.70)", results[i].get("business_name", "?")[:35], reply_name)

                results[i].setdefault("_debug", {})["owner_detection"] = {
                    "owner_name": results[i]["owner_name"],
                    "confidence_score": results[i]["confidence_score"],
                    "solo": results[i]["solo"],
                    "reason": det.get("reason", ""),
                }
                update_status(f"Owner detection {done}/{len(owner_candidates)} complete")

    # Final summary
    qualified_count = sum(1 for r in results if is_qualified_lead_row(r))
    _log.info("PIPELINE COMPLETE: %d businesses → %d qualified leads (%.0f%%)",
              len(results), qualified_count, 100 * qualified_count / len(results) if results else 0)

    update_progress(1.0, "Complete")
    update_status("Scraping finished.")

    # cleanup
    for r in results:
        r.pop("_reviews_text", None)
        r.pop("_reply_owner_name", None)

    return results


def _review_snippets_from_row(row: dict, max_items: int = 10) -> list[str]:
    snippets = list((row.get("_debug", {}) or {}).get("sample_review_snippets") or [])
    return sanitize_review_snippets(snippets, max_items=max_items)


def _review_text_from_row(row: dict, max_items: int = 10) -> str:
    return "\n\n".join(_review_snippets_from_row(row, max_items=max_items))


def assess_trainer_listing_quality(row: dict) -> dict:
    snippets = _review_snippets_from_row(row)
    lengths = [len(s) for s in snippets]
    substantial_count = sum(1 for n in lengths if n >= TRAINER_MIN_SUBSTANTIAL_REVIEW_CHARS)
    total_chars = sum(lengths)
    reviews_text = "\n".join(snippets)
    owner_signals = has_owner_signals(reviews_text)
    # Quality score: text volume (50%) + substantial reviews (30%) + owner signals (20%).
    # Owner signals matter because reviews without person names or owner language
    # teach the model nothing about solo-owner patterns.
    score = (
        min(1.0, substantial_count / max(1, TRAINER_MIN_SUBSTANTIAL_REVIEW_COUNT)) * 0.50
        + min(1.0, total_chars / max(1, TRAINER_MIN_TOTAL_REVIEW_CHARS)) * 0.30
        + (0.20 if owner_signals else 0.0)
    )
    text_ok = (
        (substantial_count >= TRAINER_MIN_SUBSTANTIAL_REVIEW_COUNT and total_chars >= TRAINER_MIN_TOTAL_REVIEW_CHARS)
        or total_chars >= (TRAINER_MIN_TOTAL_REVIEW_CHARS + 120)
        or (substantial_count >= 1 and total_chars >= 200)
    )
    # A listing is strong for training only when it has enough text AND
    # contains owner-identifying signals (person names, "owner", "ask for", etc.).
    # Reviews that just say "great service" waste labeling effort.
    is_strong = text_ok and owner_signals
    if not snippets:
        weak_reason = "No text review snippets available"
    elif is_strong:
        weak_reason = ""
    elif text_ok and not owner_signals:
        weak_reason = "Reviews lack owner/person signals (no names or owner language found)"
    elif substantial_count < TRAINER_MIN_SUBSTANTIAL_REVIEW_COUNT and total_chars < TRAINER_MIN_TOTAL_REVIEW_CHARS:
        weak_reason = "Too little review text for a strong training example"
    elif substantial_count < TRAINER_MIN_SUBSTANTIAL_REVIEW_COUNT:
        weak_reason = "Not enough substantial text reviews"
    else:
        weak_reason = "Review text is still thin"
    debug = row.get("_debug") or {}
    pull_gap = debug.get("review_pull_gap") or {}
    listed = int(pull_gap.get("listed") or debug.get("detail_review_count") or row.get("num_reviews") or 0)
    pulled = int(pull_gap.get("pulled") or debug.get("review_snippet_count") or len(snippets))
    return {
        "snippet_count": len(snippets),
        "substantial_reviews": substantial_count,
        "total_chars": total_chars,
        "has_owner_signals": owner_signals,
        "score": round(score, 3),
        "is_strong": is_strong,
        "weak_reason": weak_reason,
        "listed_reviews": listed,
        "pulled_reviews": pulled,
        "pull_gap_note": str(pull_gap.get("note") or ""),
    }


def estimate_rule_would_call_probability(reviews_text: str, business_name: str = "") -> float:
    low = (reviews_text or "").lower()
    
    # HARD OVERRIDES - Definitive 95% Secrets
    feats = set(_extract_meta_features(reviews_text))
    # Definitive NO: 3+ distinct names doing the work
    if "_META_MULTI_NAMES" in feats or "_META_DISPATCH_LANGUAGE" in feats:
        return 0.05
    # Definitive YES: "husband and wife team", "owner operated"
    if "_META_HUSBAND_WIFE" in feats or "_META_OWNER_OPERATED_EXPLICIT" in feats:
        return 0.95
    
    score = 0.50
    if "_META_OWNER" in feats:
        score += 0.14
    if "_META_FAMILY" in feats:
        score += 0.10
    if "_META_PERSONAL" in feats:
        score += 0.12
    if re.search(r"\bask for [a-z][a-z]+\b", low):
        score += 0.10
    if re.search(r"\b(owner|owners?)\b", low) and re.search(r"\b(partner|wife|husband|family)\b", low):
        score += 0.08
    if "_META_TEAM" in feats and "_META_OWNER" not in feats:
        score -= 0.08
    if "_META_ROLE" in feats:
        score -= 0.08
    if "_META_ROTATING" in feats:
        score -= 0.35
    if re.search(r"\b(different|multiple|various|several)\b", low):
        score -= 0.10
    return _clamp_probability(score)



def _trainer_priority_sort_key(row: dict) -> tuple[float, float, float, int]:
    trainer = row.get("_trainer") or {}
    quality = trainer.get("quality") or {}
    priority = trainer.get("priority") or {}
    return (
        1.0 if quality.get("is_strong") else 0.0,
        float(priority.get("priority_score", 0.0)),
        float(quality.get("score", 0.0)),
        int(row.get("num_reviews") or 0),
    )


def prepare_trainer_rows_for_labeling(
    rows: list[dict],
    status_callback=None,
    run_llm: bool = True,
    sort_rows: bool = True,
    filter_weak: bool = True,
) -> tuple[list[dict], dict]:
    working = list(rows or [])
    if not working:
        return [], {
            "input_count": 0,
            "kept_count": 0,
            "strong_count": 0,
            "filtered_weak_count": 0,
            "fallback_used": False,
        }

    strong_candidates: list[dict] = []
    for row in working:
        trainer = row.setdefault("_trainer", {})
        trainer["reviews_text"] = _review_text_from_row(row)
        trainer["quality"] = assess_trainer_listing_quality(row)
        if trainer["quality"].get("is_strong") and trainer["reviews_text"]:
            strong_candidates.append(row)

    if run_llm and strong_candidates:
        llm_candidates = [r for r in strong_candidates if not (r.get("_trainer", {}) or {}).get("llm_detection")]
        if llm_candidates:
            if status_callback:
                status_callback(f"Local AI triage for {len(llm_candidates)} trainer examples...")
            with ThreadPoolExecutor(max_workers=max(1, min(OWNER_WORKERS, len(llm_candidates)))) as executor:
                future_map = {
                    executor.submit(detect_owner, r.get("_trainer", {}).get("reviews_text", ""), r.get("business_name", "")): r
                    for r in llm_candidates
                }
                done = 0
                for fut in as_completed(future_map):
                    row = future_map[fut]
                    done += 1
                    try:
                        det = fut.result()
                    except Exception:
                        det = {
                            "owner_name": None,
                            "solo": False,
                            "confidence": 0.0,
                            "reason": "Trainer triage failed",
                        }
                    row.setdefault("_trainer", {})["llm_detection"] = det
                    if status_callback and (done == len(llm_candidates) or done % 2 == 0):
                        status_callback(f"Local AI triage {done}/{len(llm_candidates)} complete...")

    for row in working:
        trainer = row.setdefault("_trainer", {})
        quality = trainer.get("quality") or assess_trainer_listing_quality(row)
        reviews_text = str(trainer.get("reviews_text") or "")
        trainer_prob = score_would_call_probability(reviews_text, row.get("business_name", "")) if reviews_text else None
        rule_prob = estimate_rule_would_call_probability(reviews_text, row.get("business_name", "")) if reviews_text else None
        det = trainer.get("llm_detection") if isinstance(trainer.get("llm_detection"), dict) else None
        llm_prob = None
        if det is not None:
            conf = float(det.get("confidence", 0.0) or 0.0)
            llm_prob = _clamp_probability(0.5 + (conf / 2.0 if bool(det.get("solo")) else -conf / 2.0))
        probs = [p for p in (trainer_prob, rule_prob, llm_prob) if p is not None]
        avg_prob = (sum(probs) / len(probs)) if probs else 0.5
        disagreement = (max(probs) - min(probs)) if len(probs) >= 2 else 0.0
        uncertainty = max(0.0, 1.0 - abs(avg_prob - 0.5) * 2.0)
        if not quality.get("is_strong"):
            queue_label = "Weak evidence"
        elif disagreement >= 0.25:
            queue_label = "High-value disagreement"
        elif uncertainty >= 0.55:
            queue_label = "Borderline / uncertain"
        else:
            queue_label = "Cleaner consensus"
        trainer["priority"] = {
            "trainer_prob": None if trainer_prob is None else round(float(trainer_prob), 3),
            "rule_prob": None if rule_prob is None else round(float(rule_prob), 3),
            "llm_prob": None if llm_prob is None else round(float(llm_prob), 3),
            "avg_prob": round(float(avg_prob), 3),
            "disagreement": round(float(disagreement), 3),
            "uncertainty": round(float(uncertainty), 3),
            "priority_score": round(disagreement * 1.4 + uncertainty * 0.8 + float(quality.get("score", 0.0)) * 0.25, 4),
            "queue_label": queue_label,
        }

    strong_rows = [r for r in working if ((r.get("_trainer") or {}).get("quality") or {}).get("is_strong")]
    fallback_used = False
    filtered_weak_count = 0
    output_rows = working
    if filter_weak:
        if strong_rows:
            output_rows = strong_rows
            filtered_weak_count = len(working) - len(strong_rows)
        else:
            fallback_used = True
            output_rows = working
    if sort_rows:
        output_rows = sorted(output_rows, key=_trainer_priority_sort_key, reverse=True)
    return output_rows, {
        "input_count": len(working),
        "kept_count": len(output_rows),
        "strong_count": len(strong_rows),
        "filtered_weak_count": filtered_weak_count,
        "fallback_used": fallback_used,
    }


# ---------------------------------------------------------------------------
# Review training page
# ---------------------------------------------------------------------------
def _compute_training_readiness() -> dict:
    """Compute how 'trained' the model is on a 0-100 scale with breakdown."""
    labels_path = _labels_file_path()
    n_yes = 0
    n_no = 0
    n_with_highlights = 0
    n_with_notes = 0
    total_csv_rows = 0
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    label = str(row.get("would_call") or "").strip().lower()
                    if label == "yes":
                        n_yes += 1
                        total_csv_rows += 1
                    elif label == "no":
                        n_no += 1
                        total_csv_rows += 1
                    hl = str(row.get("highlighted_evidence_json") or "")
                    if hl and hl not in ("[]", ""):
                        n_with_highlights += 1
                    if str(row.get("reason") or "").strip():
                        n_with_notes += 1
        except Exception:
            pass
    total_labels = n_yes + n_no
    model = load_review_preference_model()
    has_model = model is not None
    vocab_size = int(model.get("vocab_size", 0)) if model else 0

    # Label volume: 50 points, requires 50 total labels for max.
    label_score = min(1.0, total_labels / 50) * 50
    # Class balance: 20 points. Requires both classes AND reasonable ratio.
    # A model with all-YES or all-NO labels is useless regardless of count.
    balance_score = 0.0
    if n_yes >= 3 and n_no >= 3:
        ratio = min(n_yes, n_no) / max(n_yes, n_no)
        balance_score = ratio * 20
    elif n_yes > 0 and n_no > 0:
        balance_score = 5.0
    # Highlight quality: 10 points, maxes when 70%+ of labels have highlights.
    highlight_rate = n_with_highlights / max(1, total_labels)
    highlight_score = min(1.0, highlight_rate / 0.7) * 10
    # Model existence: 5 points.
    model_score = 5 if has_model else 0
    # Per-class depth: 15 points. Requires at least 15 in the minority class.
    # This penalises imbalanced datasets where one class dominates.
    minority = min(n_yes, n_no)
    depth_score = min(1.0, minority / 15) * 15

    readiness = int(min(100, label_score + balance_score + highlight_score + model_score + depth_score))

    if readiness < 15:
        tier = "Untrained"
        color = "#6e7681"
    elif readiness < 35:
        tier = "Getting started"
        color = "#58a6ff"
    elif readiness < 55:
        tier = "Learning"
        color = "#3b82f6"
    elif readiness < 75:
        tier = "Good"
        color = "#0078d4"
    elif readiness < 90:
        tier = "Strong"
        color = "#0098ff"
    else:
        tier = "Expert"
        color = "#79c0ff"

    # Count labels added since the model was last trained.
    model_trained_at = str(model.get("trained_at_utc", "")) if model else ""
    labels_since_train = 0
    if model_trained_at and os.path.exists(labels_path):
        try:
            model_ts = datetime.fromisoformat(model_trained_at.rstrip("Z"))
            with open(labels_path, "r", encoding="utf-8", newline="") as _f:
                for _row in csv.DictReader(_f):
                    _ts_str = str(_row.get("timestamp_utc") or "").rstrip("Z")
                    if not _ts_str:
                        continue
                    try:
                        _ts = datetime.fromisoformat(_ts_str)
                        if _ts > model_ts:
                            labels_since_train += 1
                    except Exception:
                        pass
        except Exception:
            pass

    return {
        "readiness": readiness,
        "tier": tier,
        "color": color,
        "total_labels": total_labels,
        "n_yes": n_yes,
        "n_no": n_no,
        "n_with_highlights": n_with_highlights,
        "n_with_notes": n_with_notes,
        "has_model": has_model,
        "vocab_size": vocab_size,
        "labels_since_train": labels_since_train,
    }


def _render_circular_progress(readiness: int, color: str, tier: str, size: int = 150) -> str:
    """Return an HTML/SVG circular progress bar."""
    radius = 54
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - readiness / 100)
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:8px 0;">
      <svg width="{size}" height="{size}" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="{radius}" fill="none" stroke="#242424" stroke-width="8"/>
        <circle cx="60" cy="60" r="{radius}" fill="none" stroke="{color}" stroke-width="8"
                stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                stroke-linecap="round" transform="rotate(-90 60 60)"
                style="transition: stroke-dashoffset 0.6s ease;"/>
        <text x="60" y="55" text-anchor="middle" fill="{color}" font-size="26" font-weight="600" font-family="Inter, system-ui, sans-serif">{readiness}%</text>
        <text x="60" y="72" text-anchor="middle" fill="#6B6865" font-size="10" font-family="Inter, system-ui, sans-serif">{tier}</text>
      </svg>
    </div>"""


def render_review_training_page() -> None:
    st.title("Review Trainer")

    with st.sidebar:
        st.header("Review Trainer Settings")

        tr = _compute_training_readiness()
        st.markdown(_render_circular_progress(tr["readiness"], tr["color"], tr["tier"]), unsafe_allow_html=True)
        st.caption(
            f'{tr["total_labels"]} labels ({tr["n_yes"]} yes / {tr["n_no"]} no) '
            f'| {tr["n_with_highlights"]} with highlights'
        )
        if tr["has_model"]:
            _since = int(tr.get("labels_since_train", 0))
            if _since > 0:
                st.caption(f'Model trained (vocab: {tr["vocab_size"]}) — ⚠ {_since} new label(s) since last train')
            else:
                st.caption(f'Model trained (vocab: {tr["vocab_size"]}) — up to date')
        else:
            if tr["total_labels"] >= 2:
                st.caption("Model not trained yet -- click Train below")
            else:
                st.caption("Model not trained yet -- label some listings first")

        st.markdown("---")
        city = st.text_input("City (trainer)", value="Seattle", key="trainer_city")
        niche = st.selectbox("Niche (trainer)", options=NICHES, key="trainer_niche")
        max_pages = st.slider("Max pages (trainer)", 1, 10, 2, key="trainer_max_pages")
        trainer_chunk_size = st.slider("Listings per chunk", 1, 20, 5, key="trainer_chunk_size")
        show_browser = st.checkbox("Show browser window (trainer)", value=False, key="trainer_show_browser")
        run_trainer_triage = st.checkbox(
            "Local AI triage (slower, optional)",
            value=False,
            key="trainer_run_triage",
            help="Runs local Ollama analysis on each pulled listing to prioritize 'Next Best Example'. Turning this off makes pulling chunks much faster.",
        )
        pull_initial_clicked = st.button("Pull first chunk", type="primary", use_container_width=True)
        pull_more_clicked = st.button("Load next chunk", use_container_width=True)
        train_clicked = st.button("Train preference model from labels", use_container_width=True)

        st.markdown("---")
        reset_session_clicked = st.button("Reset current session", use_container_width=True,
                                          help="Clears pulled listings from this session. Saved labels in the CSV are kept.")
        if "confirm_reset_all" not in st.session_state:
            st.session_state.confirm_reset_all = False
        reset_all_clicked = st.button("Reset ALL training data", use_container_width=True,
                                      help="Deletes the labels CSV and trained model. Cannot be undone.")
        if reset_all_clicked:
            st.session_state.confirm_reset_all = True
        if st.session_state.get("confirm_reset_all"):
            st.warning("This will delete all saved labels and the trained model.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete everything", use_container_width=True):
                    for fp in [_labels_file_path(), _trainer_model_path()]:
                        if os.path.exists(fp):
                            os.remove(fp)
                    st.session_state.trainer_rows = []
                    st.session_state.trainer_current_idx = 0
                    st.session_state.trainer_saved_signatures = []
                    st.session_state.trainer_label_cache = {}
                    st.session_state.confirm_reset_all = False
                    st.success("All training data deleted.")
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_reset_all = False
                    st.rerun()

    if "trainer_rows" not in st.session_state:
        st.session_state.trainer_rows = []
    if "trainer_current_idx" not in st.session_state:
        st.session_state.trainer_current_idx = 0
    if "trainer_saved_signatures" not in st.session_state:
        # Pre-populate from the existing labels CSV so we don't re-save rows
        # that were saved in a previous session.
        _pre_sigs: list[str] = []
        if os.path.exists(_labels_file_path()):
            try:
                with open(_labels_file_path(), newline="", encoding="utf-8") as _pf:
                    for _pr in csv.DictReader(_pf):
                        _pre_sigs.append(_make_save_sig(_pr))
            except Exception:
                pass
        st.session_state.trainer_saved_signatures = list(set(_pre_sigs))
    if "trainer_label_cache" not in st.session_state:
        st.session_state.trainer_label_cache = {}

    if reset_session_clicked:
        st.session_state.trainer_rows = []
        st.session_state.trainer_current_idx = 0
        st.session_state.trainer_saved_signatures = []
        st.session_state.trainer_label_cache = {}
        st.info("Session cleared. Pull a new chunk to start fresh.")
        st.rerun()

    if train_clicked:
        try:
            model = train_review_preference_model()
            existing_rows = list(st.session_state.get("trainer_rows") or [])
            if existing_rows:
                refreshed_rows, _ = prepare_trainer_rows_for_labeling(
                    existing_rows,
                    status_callback=None,
                    run_llm=False,
                    sort_rows=False,
                    filter_weak=False,
                )
                st.session_state.trainer_rows = refreshed_rows
            _n_raw = int(model.get("n_raw_labels", 0))
            _n_yes_labels = int(model.get("n_yes_labels", 0))
            _n_no_labels = int(model.get("n_no_labels", 0))
            st.success(
                f"Trained model from {_n_raw} labels: {_n_yes_labels} yes / {_n_no_labels} no."
            )
        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    def trainer_row_key(row: dict) -> str:
        name = str(row.get("business_name") or "").strip().lower()
        address = str(row.get("address") or "").strip().lower()
        phone = "".join(ch for ch in str(row.get("phone") or "") if ch.isdigit())
        if name and address:
            return f"{name}|{address}"
        if name and phone:
            return f"{name}|{phone}"
        return name

    if pull_initial_clicked or pull_more_clicked:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        existing_rows = list(st.session_state.get("trainer_rows") or [])
        if pull_initial_clicked:
            desired_total = int(trainer_chunk_size)
            existing_rows = []
            st.session_state.trainer_current_idx = 0
            st.session_state.trainer_label_cache = {}
            progress_placeholder.progress(0, text="Starting trainer pull...")
            status_placeholder.info("Pulling first chunk of businesses and reviews...")
        else:
            desired_total = len(existing_rows) + int(trainer_chunk_size)
            progress_placeholder.progress(0, text="Loading more...")
            status_placeholder.info("Loading next chunk (free Google Maps mode)...")

        def set_progress(pct, msg=None):
            progress_placeholder.progress(min(1.0, max(0.0, pct)), text=msg)

        def set_status(msg):
            status_placeholder.info(msg)

        rows = scrape_google_maps(
            city=city,
            niche=niche,
            max_pages=max_pages,
            progress_callback=set_progress,
            status_callback=set_status,
            table_callback=None,
            headless=not show_browser,
            max_businesses=desired_total,
            target_leads=None,
            run_owner_detection=False,
            review_snippets_target=TRAINER_REVIEW_SNIPPETS,
        )

        merged: list[dict] = list(existing_rows)
        seen_keys = {trainer_row_key(r) for r in merged if trainer_row_key(r)}
        seen_reviews = set()
        for r in merged:
            snips = _review_snippets_from_row(r)
            if snips:
                seen_reviews.add(snips[0][:80])
        fresh_rows: list[dict] = []
        for row in rows:
            k = trainer_row_key(row)
            if k and k in seen_keys:
                continue
            snips = _review_snippets_from_row(row)
            review_fp = snips[0][:80] if snips else ""
            if review_fp and review_fp in seen_reviews:
                continue
            if k:
                seen_keys.add(k)
            if review_fp:
                seen_reviews.add(review_fp)
            fresh_rows.append(row)

        prepared_rows, prep_stats = prepare_trainer_rows_for_labeling(
            fresh_rows,
            status_callback=set_status,
            run_llm=bool(run_trainer_triage),
        )
        first_new_idx = len(merged)
        merged.extend(prepared_rows)

        st.session_state.trainer_rows = merged
        if pull_more_clicked and prepared_rows:
            st.session_state.trainer_current_idx = first_new_idx
        progress_placeholder.empty()
        if pull_initial_clicked:
            if prep_stats.get("fallback_used"):
                status_placeholder.warning(
                    f"Pulled {len(merged)} listings. Google only exposed thin review text this round, so the trainer kept the best available examples."
                )
            else:
                skipped = int(prep_stats.get("filtered_weak_count", 0))
                extra = f" Skipped {skipped} weak listings." if skipped else ""
                status_placeholder.success(f"Pulled {len(merged)} high-value listings.{extra}")
        else:
            skipped = int(prep_stats.get("filtered_weak_count", 0))
            if prep_stats.get("fallback_used"):
                status_placeholder.warning(
                    f"Added {len(prepared_rows)} listings (total: {len(merged)}). Google only exposed weak review text in this batch."
                )
            else:
                extra = f" Skipped {skipped} weak listings." if skipped else ""
                status_placeholder.success(f"Added {len(prepared_rows)} new high-value listings (total: {len(merged)}).{extra}")

    with st.expander("How to use the Review Trainer", expanded="trainer_rows" not in st.session_state or not st.session_state.get("trainer_rows")):
        st.markdown("""
**Goal:** Teach the AI to recognize which businesses are worth cold-calling by showing it your decision-making process.

**Step 1 -- Pull listings**
- In the sidebar, pick a **city** and **niche** you'd normally prospect in.
- Click **Pull first chunk**. The scraper will grab businesses and up to 6 of their Google reviews each (free, no API credits).
- Need more? Click **Load next chunk** to add another batch.

**Step 2 -- Label each listing**
- Read through the reviews like you would on Google Maps.
- Ask yourself: *"Would I pick up the phone and call this business?"*
- Select **yes** or **no**.
- **Highlight key phrases** -- in each review, paste the exact words that influenced you (e.g. *"Jeff the owner"*, *"family-run shop"*). This is the most valuable signal for the AI.
- Optionally add a **note** explaining your reasoning and guess the **owner's name** if you spotted one.
- Click **Save & Next >** to save your label and jump to the next highest-value example.

**Step 3 -- Train the model**
- After labeling **at least 10-15 listings** (mix of yes and no), click **Train preference model** in the sidebar.
- The circular progress bar shows your overall training readiness. Aim for **60%+** before relying on the model.
- You can keep labeling and retraining -- each round makes it smarter.

**Tips for best results:**
- Label a **balanced mix** of yes and no (don't only label easy ones).
- Highlights matter more than notes -- the AI weights them 3x.
- Use **Next Best Example** to label the listings where the AI is most uncertain (active learning).
- After 30-40 labels with good highlights, the model becomes quite reliable.
- The progress bar tracks labels, balance, highlights, and vocabulary -- all factors that improve accuracy.
""")

    rows = st.session_state.get("trainer_rows", [])
    if not rows:
        st.info("Click **Pull first chunk** in the sidebar to get started.")
        return

    st.subheader("Label Listings")

    def trainer_uid(row: dict, idx: int) -> str:
        debug = row.get("_debug") or {}
        place_url = str(debug.get("place_url") or "").strip()
        raw = place_url or trainer_row_key(row) or ""
        # IMPORTANT: uid must be stable across reruns and list re-ordering.
        # Using `idx` in the hash makes the uid unstable when rows change order,
        # which can cause the labeled % meter to decrease unexpectedly.
        if not raw:
            # Fall back to a deterministic identity derived from stable fields,
            # NOT the row index.
            name = str(row.get("business_name") or "").strip().lower()
            addr = str(row.get("address") or "").strip().lower()
            phone = "".join(ch for ch in str(row.get("phone") or "") if ch.isdigit())
            website = str(row.get("website") or "").strip().lower()
            raw = "|".join(p for p in (name, addr, phone, website) if p)
        if not raw:
            # Extremely rare: keep stable within this chunk by hashing the full row.
            # (Still better than idx-based.)
            try:
                raw = json.dumps({k: row.get(k) for k in sorted(row.keys()) if not str(k).startswith("_")}, ensure_ascii=False)
            except Exception:
                raw = "unknown"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _cache_listing_state(j: int) -> None:
        """Persist current widget values for listing j into the label cache."""
        r = rows[j]
        r_uid = trainer_uid(r, j)
        dec = str(st.session_state.get(f"trainer_decision_{r_uid}", "unlabeled")).lower().strip()
        reason = str(st.session_state.get(f"trainer_reason_{r_uid}", "")).strip()
        owner_guess = str(st.session_state.get(f"trainer_owner_guess_{r_uid}", "")).strip()
        highlights: dict[str, str] = {}
        revs = _review_snippets_from_row(r)
        for ridx in range(1, len(revs) + 1):
            h = str(st.session_state.get(f"trainer_highlight_{r_uid}_{ridx}", "")).strip()
            if h:
                highlights[str(ridx)] = h
        cache = st.session_state.setdefault("trainer_label_cache", {})
        existing = cache.get(r_uid, {}) if isinstance(cache, dict) else {}
        # Safety: don't let a transient rerun overwrite a real label with an empty unlabeled state.
        # This prevents the progress meter from dropping to 0/20.
        if (
            str(existing.get("decision", "")).lower() in ("yes", "no")
            and dec == "unlabeled"
            and not reason
            and not owner_guess
            and not highlights
        ):
            return
        cache[r_uid] = {
            "decision": dec,
            "reason": reason,
            "owner_guess": owner_guess,
            "highlights": highlights,
        }

    label_cache: dict = st.session_state.get("trainer_label_cache", {})
    # One-time migration: older builds used sha1(f"{raw}:{idx}") as uid.
    # If the user already labeled rows, keep those labels after switching uid scheme.
    try:
        migrated = 0
        for _j in range(len(rows)):
            _debug = rows[_j].get("_debug") or {}
            _place_url = str(_debug.get("place_url") or "").strip()
            _raw = _place_url or trainer_row_key(rows[_j]) or ""
            if not _raw:
                continue
            _old_uid = hashlib.sha1(f"{_raw}:{_j}".encode("utf-8")).hexdigest()[:12]
            _new_uid = hashlib.sha1(_raw.encode("utf-8")).hexdigest()[:12]
            if _old_uid in label_cache and _new_uid not in label_cache:
                label_cache[_new_uid] = label_cache.get(_old_uid, {})
                migrated += 1
        if migrated:
            st.session_state.trainer_label_cache = label_cache
    except Exception:
        pass
    labeled_count = 0
    unlabeled_indices: list[int] = []
    for idx in range(len(rows)):
        uid = trainer_uid(rows[idx], idx)
        decision = str(label_cache.get(uid, {}).get("decision", "unlabeled")).lower()
        if decision in ("yes", "no"):
            labeled_count += 1
        else:
            unlabeled_indices.append(idx)

    def trainer_priority_score(idx: int) -> float:
        trainer = rows[idx].get("_trainer") or {}
        quality = trainer.get("quality") or {}
        priority = trainer.get("priority") or {}
        return float(priority.get("priority_score", 0.0)) + (0.15 if quality.get("is_strong") else 0.0)

    def next_best_unlabeled_idx(current_idx: int | None = None) -> int | None:
        ordered = sorted(unlabeled_indices, key=trainer_priority_score, reverse=True)
        for idx in ordered:
            if current_idx is None or idx != current_idx:
                return idx
        return None

    progress_pct = labeled_count / max(1, len(rows))
    st.progress(progress_pct, text=f"Labeled {labeled_count}/{len(rows)}")

    cur = int(st.session_state.get("trainer_current_idx", 0))
    cur = max(0, min(cur, len(rows) - 1))
    st.session_state.trainer_current_idx = cur

    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        if st.button("Previous", use_container_width=True, disabled=(cur <= 0)):
            _cache_listing_state(cur)
            st.session_state.trainer_current_idx = max(0, cur - 1)
            st.rerun()
    with nav2:
        if st.button("Next", use_container_width=True, disabled=(cur >= len(rows) - 1)):
            _cache_listing_state(cur)
            st.session_state.trainer_current_idx = min(len(rows) - 1, cur + 1)
            st.rerun()
    with nav3:
        next_best = next_best_unlabeled_idx(cur)
        if st.button("Next Best Example", use_container_width=True, disabled=(next_best is None)):
            _cache_listing_state(cur)
            st.session_state.trainer_current_idx = int(next_best)
            st.rerun()
    # Keep the Jump-to widget synchronized with the current index.
    # Without this, typing into any input triggers a rerun, and Streamlit may
    # reuse the old selectbox value and jump you back to a previous listing.
    st.session_state["trainer_jump_to"] = int(cur) + 1
    with nav4:
        chosen = st.selectbox(
            "Jump to",
            options=list(range(1, len(rows) + 1)),
            index=cur,
            label_visibility="collapsed",
            key="trainer_jump_to",
        )
        chosen_idx = int(chosen) - 1
        if chosen_idx != cur:
            _cache_listing_state(cur)
            st.session_state.trainer_current_idx = chosen_idx
            st.rerun()

    i = int(st.session_state.trainer_current_idx)
    row = rows[i]
    uid = trainer_uid(row, i)
    biz = row.get("business_name", "Unknown")
    addr = row.get("address", "Not listed")
    rating_val = row.get("rating")
    reviews = _review_snippets_from_row(row, TRAINER_REVIEW_SNIPPETS)
    trainer_info = row.get("_trainer") or {}
    quality_info = trainer_info.get("quality") or assess_trainer_listing_quality(row)
    priority_info = trainer_info.get("priority") or {}

    # Restore cached values into widget keys so previously labeled listings
    # display their saved answers even after Streamlit garbage-collects the
    # widget keys (which happens whenever a widget is not rendered).
    cached_label = label_cache.get(uid, {})
    if cached_label:
        _wk_decision = f"trainer_decision_{uid}"
        if _wk_decision not in st.session_state:
            st.session_state[_wk_decision] = cached_label.get("decision", "unlabeled")
        _wk_reason = f"trainer_reason_{uid}"
        if _wk_reason not in st.session_state:
            st.session_state[_wk_reason] = cached_label.get("reason", "")
        _wk_owner = f"trainer_owner_guess_{uid}"
        if _wk_owner not in st.session_state:
            st.session_state[_wk_owner] = cached_label.get("owner_guess", "")
        for _hridx_str, _htxt in (cached_label.get("highlights") or {}).items():
            _wk_hl = f"trainer_highlight_{uid}_{_hridx_str}"
            if _wk_hl not in st.session_state:
                st.session_state[_wk_hl] = _htxt

    st.markdown(f"### {i + 1}. {biz}")
    meta1, meta2 = st.columns(2)
    with meta1:
        if rating_val is not None:
            st.write(f"**Star rating:** {float(rating_val):.1f}/5")
        else:
            st.write("**Star rating:** Unknown")
    with meta2:
        st.write(f"**Review count:** {int(row.get('num_reviews') or 0)}")
    st.write(f"**Address:** {addr}")
    st.write(f"**Phone:** {row.get('phone', 'Not listed')}")
    st.write(f"**Website:** {row.get('website', 'Not listed')}")
    summary_bits = []
    listed_n = int(quality_info.get("listed_reviews") or row.get("num_reviews") or 0)
    pulled_n = int(quality_info.get("pulled_reviews") or len(reviews))
    if listed_n > 0 and pulled_n < min(TRAINER_REVIEW_SNIPPETS, listed_n):
        summary_bits.append(f"Reviews pulled: {pulled_n}/{listed_n} (Google limited view)")
    elif listed_n > 0:
        summary_bits.append(f"Reviews pulled: {pulled_n}/{listed_n}")
    if quality_info.get("is_strong"):
        summary_bits.append(f"Training quality: good ({int(quality_info.get('substantial_reviews', 0))} substantial)")
    else:
        summary_bits.append(f"Training quality: weak ({quality_info.get('weak_reason') or 'limited review text'})")
    if priority_info.get("queue_label"):
        summary_bits.append(f"Queue: {priority_info.get('queue_label')}")
    if summary_bits:
        st.caption(" | ".join(summary_bits))

    # Show a clear indicator when fewer than 6 reviews were pulled, explaining
    # the specific Google-side reason so the user knows it's not an app bug.
    debug_info = row.get("_debug") or {}
    gap_info = debug_info.get("review_pull_gap") or {}
    gap_note_text = str(gap_info.get("note") or quality_info.get("pull_gap_note") or "")
    if pulled_n < TRAINER_REVIEW_SNIPPETS:
        if gap_note_text:
            st.warning(f"Only {pulled_n}/{TRAINER_REVIEW_SNIPPETS} reviews pulled: {gap_note_text}")
        elif debug_info.get("limited_view"):
            st.warning(f"Only {pulled_n}/{TRAINER_REVIEW_SNIPPETS} reviews pulled: Google showed a limited/bot-detected view of this page.")
        elif debug_info.get("used_card_snippet"):
            st.warning(f"Only {pulled_n}/{TRAINER_REVIEW_SNIPPETS} reviews pulled: Google did not expose the full reviews panel. Only the card preview snippet was available.")
        elif debug_info.get("worker_timed_out"):
            st.warning(f"Only {pulled_n}/{TRAINER_REVIEW_SNIPPETS} reviews pulled: Page load timed out before reviews could be extracted.")
        elif listed_n > 0 and listed_n < TRAINER_REVIEW_SNIPPETS:
            st.info(f"Only {pulled_n}/{TRAINER_REVIEW_SNIPPETS} reviews available: this business only has {listed_n} reviews on Google.")
        elif listed_n > pulled_n:
            st.warning(f"Only {pulled_n}/{TRAINER_REVIEW_SNIPPETS} reviews pulled: most of the {listed_n} listed reviews are star-only ratings without text.")
        else:
            st.warning(f"Only {pulled_n}/{TRAINER_REVIEW_SNIPPETS} reviews pulled: Google did not expose enough review text for this listing.")
    elif gap_note_text:
        st.caption(gap_note_text)

    if reviews:
        st.write("**Review snippets (up to 6):**")
        # Wrap reviews to a narrower column to make skimming easier.
        for ridx, rv in enumerate(reviews, start=1):
            safe_text = html.escape(str(rv or ""))
            st.markdown(
                f"<div style='max-width:65ch; white-space:normal; line-height:1.4; margin-bottom:0.35rem;'><strong>{ridx}.</strong> {safe_text}</div>",
                unsafe_allow_html=True,
            )
            st.text_input(
                f"Highlight from review {ridx} (optional)",
                key=f"trainer_highlight_{uid}_{ridx}",
                placeholder="Paste the exact words/phrase you used for your decision from this review.",
            )
    else:
        st.info("No review snippets available for this listing.")

    st.markdown("---")
    dec_col, owner_col = st.columns([1, 1])
    with dec_col:
        st.radio(
            "Would you call this lead?",
            options=["unlabeled", "yes", "no"],
            horizontal=True,
            key=f"trainer_decision_{uid}",
        )
    with owner_col:
        st.text_input(
            "Owner name guess (optional)",
            key=f"trainer_owner_guess_{uid}",
            placeholder="e.g. Jeff, Chris Hays",
        )
    st.text_area(
        "Notes (optional)",
        key=f"trainer_reason_{uid}",
        placeholder="Why yes/no? Any observations.",
        height=68,
    )

    def _build_label_payload(j: int) -> dict | None:
        """Build a save payload for a single listing by index. Returns None if unlabeled."""
        r = rows[j]
        r_uid = trainer_uid(r, j)
        dec = str(st.session_state.get(f"trainer_decision_{r_uid}", "unlabeled")).lower().strip()
        if dec not in ("yes", "no"):
            return None
        revs = _review_snippets_from_row(r)
        
        # Clean extracted reviews before saving
        clean_revs = [clean_extracted_review_snippet(rv) for rv in revs]
        # Keep only the non-empty ones and rebuild list
        clean_revs = [cr for cr in clean_revs if cr]
        
        hl = []
        for ridx, _rv in enumerate(clean_revs, start=1):
            h = str(st.session_state.get(f"trainer_highlight_{r_uid}_{ridx}", "")).strip()
            if h:
                hl.append({"review_index": ridx, "text": h})
        return {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "city": city,
            "niche": niche,
            "business_name": r.get("business_name", ""),
            "address": r.get("address", ""),
            "phone": r.get("phone", ""),
            "website": r.get("website", ""),
            "rating": r.get("rating", None),
            "num_reviews": int(r.get("num_reviews") or 0),
            "would_call": dec,
            "reason": str(st.session_state.get(f"trainer_reason_{r_uid}", "")).strip(),
            "evidence_quote": "",
            "highlighted_evidence_json": json.dumps(hl, ensure_ascii=False),
            "owner_name_guess": str(st.session_state.get(f"trainer_owner_guess_{r_uid}", "")).strip(),
            "reviews_json": json.dumps(clean_revs, ensure_ascii=False),
        }

    def _save_payloads(payloads: list[dict]) -> int:
        """Deduplicate and append payloads to the labels CSV. Returns count saved.

        Deduplication uses (business_name, address, phone, would_call) so that:
        - The same label saved twice (same decision) is skipped.
        - A relabeled business (yes→no or no→yes) IS saved; training deduplication
          in train_review_preference_model() keeps only the most recent label.
        - Cross-session duplicates are prevented because trainer_saved_signatures
          is pre-populated from the CSV at session start.
        """
        saved_sigs = set(st.session_state.get("trainer_saved_signatures") or [])
        new_rows = []
        for p in payloads:
            sig = _make_save_sig(p)
            if sig not in saved_sigs:
                saved_sigs.add(sig)
                new_rows.append(p)
        if new_rows:
            append_review_labels(new_rows, city=city, niche=niche)
            st.session_state.trainer_saved_signatures = list(saved_sigs)
            log_event(
                "trainer",
                logging.INFO,
                "trainer_labels_saved",
                count=len(new_rows),
                city=city,
                niche=niche,
            )
        else:
            log_event(
                "trainer",
                logging.DEBUG,
                "trainer_labels_no_new_rows",
                incoming=len(payloads),
            )
        return len(new_rows)

    save_col, next_col = st.columns(2)
    with save_col:
        save_all = st.button("Save all labels", use_container_width=True)
    with next_col:
        save_next = st.button("Save & Next >", type="primary", use_container_width=True)

    if save_next:
        _cache_listing_state(i)
        payload = _build_label_payload(i)
        if payload is None:
            st.warning("Set a decision (yes/no) before saving.")
        else:
            _save_payloads([payload])
            # Recompute unlabeled indices from the just-updated cache.
            _fresh_cache = st.session_state.get("trainer_label_cache", {})
            _fresh_unlabeled = []
            for _j in range(len(rows)):
                _j_uid = trainer_uid(rows[_j], _j)
                _j_dec = str(_fresh_cache.get(_j_uid, {}).get("decision", "unlabeled")).lower()
                if _j_dec not in ("yes", "no"):
                    _fresh_unlabeled.append(_j)
            nxt = None
            for _j in _fresh_unlabeled:
                if _j > i:
                    nxt = _j
                    break
            if nxt is None and _fresh_unlabeled:
                nxt = _fresh_unlabeled[0]
            if nxt is not None:
                st.session_state.trainer_current_idx = nxt
                st.rerun()
            else:
                st.info("All listings labeled! Click 'Train preference model' in the sidebar.")
    elif save_all:
        for j in range(len(rows)):
            _cache_listing_state(j)
        payloads = [p for p in (_build_label_payload(j) for j in range(len(rows))) if p]
        if not payloads:
            st.warning("No labeled rows to save yet. Set at least one decision to yes/no.")
        else:
            n = _save_payloads(payloads)
            if n > 0:
                st.success(f"Saved {n} labeled rows.")
            else:
                st.info("No new or changed labels to save.")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
from pathlib import Path


def _inject_tel_patch(tel_to_website: dict[str, str]) -> None:
    """Inject JS that patches window.open for tel: links + maps phone→website."""
    import json
    import streamlit.components.v1 as components
    mapping_json = json.dumps(tel_to_website)
    components.html(
        f"""<script>
        var w = window.parent;
        // Always update the mapping (data changes between reruns)
        w._telToWebsite = {mapping_json};
        console.log('[TEL PATCH] mapping updated:', Object.keys(w._telToWebsite).length, 'entries', w._telToWebsite);
        // Re-patch every time (Streamlit reruns may reset window.open)
        if(!w._origWindowOpen) w._origWindowOpen = w.open.bind(w);
        w.open = function(url){{
            console.log('[TEL PATCH] window.open called with:', url);
            if(url && typeof url === 'string' && url.startsWith('tel:')){{
                console.log('[TEL PATCH] intercepted tel:', url);
                var f = w.document.createElement('iframe');
                f.style.display = 'none';
                f.src = url;
                w.document.body.appendChild(f);
                setTimeout(function(){{ w.document.body.removeChild(f); }}, 2000);
                var site = w._telToWebsite[url];
                console.log('[TEL PATCH] website lookup:', site);
                if(site){{
                    var bg = w._origWindowOpen(site, '_blank');
                    try {{ bg.blur(); }} catch(e){{}}
                    w.focus();
                    setTimeout(function(){{ w.focus(); }}, 50);
                    setTimeout(function(){{ w.focus(); }}, 150);
                }}
                return null;
            }}
            return w._origWindowOpen.apply(w, arguments);
        }};
        </script>""",
        height=0,
    )


def _inject_saas_theme() -> None:
    """Inject local CSS to give the app a SaaS-style shell."""
    try:
        css_path = Path(__file__).with_name("saas_theme.css")
        css = css_path.read_text(encoding="utf-8")
    except Exception:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    # tel: patch is injected per-table (needs phone→website mapping from data)


def main():
    st.set_page_config(page_title="Solo Owner Leads", page_icon="📞", layout="wide")
    style_metric_cards()
    _inject_saas_theme()

    # --- Lead database ---
    from leads_db import init_db, upsert_leads, update_lead_tracking, get_all_leads, get_lead_stats
    if "db_conn" not in st.session_state:
        _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leads.db")
        st.session_state.db_conn = init_db(_db_path)

    page = st.sidebar.radio(
        "Page",
        options=["Lead Finder", "Review Trainer"],
        key="app_page",
        label_visibility="collapsed",
    )
    if page == "Review Trainer":
        render_review_training_page()
        return

    # --- Hero header ---
    st.markdown(
        "<div class='solo-hero solo-fade-in'>"
        "<div class='solo-hero-inner'>"
        "<div class='solo-hero-badge'>LEAD FINDER</div>"
        "<h1 class='solo-hero-title'>Solo Owner Leads</h1>"
        "<p class='solo-hero-subtitle'>Find cold-callable solo-owned local businesses from Google Maps reviews.</p>"
        "<div class='solo-hero-chips'>"
        "<span class='solo-chip'>Google Maps</span>"
        "<span class='solo-chip'>AI Classification</span>"
        "<span class='solo-chip'>DataForSEO</span>"
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        # --- Branding ---
        st.markdown(
            "<div class='solo-sidebar-brand'>"
            "<span class='solo-sidebar-logo'>S</span>"
            "<span class='solo-sidebar-name'>Solo Leads</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # --- Search section ---
        st.markdown("<div class='solo-sidebar-section-label'>SEARCH</div>", unsafe_allow_html=True)
        city = st.text_input("City", value="Dallas", key="city")
        niche = st.selectbox("Business type / Niche", options=NICHES, key="niche")
        target_leads = st.slider(
            "Target qualified leads",
            min_value=1,
            max_value=50,
            value=5,
            help="Stop early once this many call-ready leads are found.",
        )
        max_pages = st.slider("Max result pages", min_value=1, max_value=10, value=3, key="max_pages")

        # --- Data source section ---
        st.markdown("<div class='solo-sidebar-section-label'>DATA SOURCE</div>", unsafe_allow_html=True)
        use_dataforseo = st.checkbox(
            "Use DataForSEO API",
            value=False,
            help="Fetch Google Maps results + reviews via API. Requires credentials in .env.",
        )
        if use_dataforseo:
            dataforseo_location_name = st.text_input(
                "Location",
                value=f"{city},United States",
                help='"City,State,United States" for best accuracy.',
                disabled=not use_dataforseo,
            )
            reviews_depth = st.slider(
                "Reviews per business",
                min_value=10,
                max_value=200,
                value=40,
                step=10,
                disabled=not use_dataforseo,
            )
            reviews_priority = st.selectbox(
                "Queue priority",
                options=[1, 2],
                index=1,
                help="1 = fast (≤1 min, 2x cost). 2 = cheap (up to 45 min, half price).",
                disabled=not use_dataforseo,
            )
        else:
            dataforseo_location_name = f"{city},United States"
            reviews_depth = 40
            reviews_priority = 2

        # --- Advanced section ---
        with st.expander("Advanced", expanded=False):
            show_browser = st.checkbox("Show browser window", value=False, help="Opens a visible Chromium window for debugging.")
            debug_mode = st.checkbox("Debug mode", value=False, help="Limit businesses processed for testing.")
            debug_max_businesses = st.slider(
                "Max businesses (debug)",
                min_value=1,
                max_value=50,
                value=3,
                disabled=not debug_mode,
            )

        # --- Action ---
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        start_clicked = st.button("Start Scraping", type="primary", use_container_width=True)

        # --- Database stats ---
        st.markdown("<div class='solo-sidebar-section-label'>DATABASE</div>", unsafe_allow_html=True)
        _stats = get_lead_stats(st.session_state.db_conn)
        st.markdown(
            f"<div style='font-size:0.8rem; color:#706C68; line-height:1.8;'>"
            f"<span style='color:#A8A4A0;'>{_stats['total']}</span> total leads<br>"
            f"<span style='color:#A8A4A0;'>{_stats['qualified']}</span> qualified<br>"
            f"<span style='color:#A8A4A0;'>{_stats['called']}</span> contacted<br>"
            f"<span style='color:#D4A574;'>{_stats['interested']}</span> interested"
            f"</div>",
            unsafe_allow_html=True,
        )
        show_history = st.button("View All Saved Leads", use_container_width=True)

    # Session state for results
    if "leads_df" not in st.session_state:
        st.session_state.leads_df = None
    if "scraping_done" not in st.session_state:
        st.session_state.scraping_done = False
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

    if show_history:
        st.session_state.show_history = True
    if st.session_state.show_history and not start_clicked:
        _all_leads = get_all_leads(st.session_state.db_conn)
        if not _all_leads:
            st.markdown(
                "<div class='solo-empty-state'>"
                "<div class='solo-empty-icon'>&#x1f4c2;</div>"
                "<div class='solo-empty-title'>No saved leads yet</div>"
                "<div class='solo-empty-desc'>Run a search to start building your lead database.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='solo-section-header'>"
                "<span class='solo-section-title'>All Saved Leads</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            _STATUS_OPTIONS_HIST = ["New", "Called", "No Answer", "Answered", "Interested", "Not Interested"]
            _hist_df = pd.DataFrame(_all_leads)
            _hist_display_cols = ["business_name", "phone", "website", "owner_name",
                                  "confidence_score", "search_city", "search_niche", "status", "notes"]
            for c in _hist_display_cols:
                if c not in _hist_df.columns:
                    _hist_df[c] = ""
            _hist_ids = _hist_df["id"].tolist()
            _hist_old_status = _hist_df["status"].tolist()
            _hist_old_notes = _hist_df["notes"].tolist()
            _hist_show = _hist_df[_hist_display_cols].copy()

            # Normalize phone to tel: URLs and website to http URLs
            if "phone" in _hist_show.columns:
                _hp = []
                for val in _hist_show["phone"].astype(str).tolist():
                    v = val.strip()
                    if not v or v.lower() in ("not listed", "none", "null", ""):
                        _hp.append("")
                        continue
                    digits = "".join(ch for ch in v if ch.isdigit())
                    if not digits:
                        _hp.append("")
                        continue
                    if len(digits) == 10:
                        digits = "1" + digits
                    _hp.append(f"tel:+{digits}")
                _hist_show["phone"] = _hp
            if "website" in _hist_show.columns:
                _hw = []
                for val in _hist_show["website"].astype(str).tolist():
                    v = val.strip()
                    if not v or v.lower() in ("not listed", "none", "null", ""):
                        _hw.append("")
                        continue
                    if not v.startswith(("http://", "https://")):
                        v = "https://" + v.lstrip("/")
                    _hw.append(v)
                _hist_show["website"] = _hw

            # KPI row
            _hcol1, _hcol2, _hcol3, _hcol4 = st.columns(4)
            with _hcol1:
                st.metric("Total Leads", len(_all_leads))
            with _hcol2:
                st.metric("Qualified", sum(1 for r in _all_leads if r.get("solo") and float(r.get("confidence_score", 0)) > 0.7))
            with _hcol3:
                st.metric("Contacted", sum(1 for r in _all_leads if r.get("status", "New") != "New"))
            with _hcol4:
                st.metric("Interested", sum(1 for r in _all_leads if r.get("status") == "Interested"))

            # Inject tel: patch with phone→website mapping
            _hist_tel_map = {}
            if "phone" in _hist_show.columns and "website" in _hist_show.columns:
                for _, _hr in _hist_show.iterrows():
                    _hp = str(_hr.get("phone", "")).strip()
                    _hw = str(_hr.get("website", "")).strip()
                    if _hp and _hw and _hp.startswith("tel:"):
                        _hist_tel_map[_hp] = _hw
            _inject_tel_patch(_hist_tel_map)

            _hist_edited = st.data_editor(
                _hist_show,
                use_container_width=True,
                height=500,
                num_rows="fixed",
                disabled=["business_name", "phone", "website", "owner_name",
                          "confidence_score", "search_city", "search_niche"],
                column_config={
                    "phone": st.column_config.LinkColumn("Phone", display_text="Call"),
                    "website": st.column_config.LinkColumn("Website", display_text="Open"),
                    "status": st.column_config.SelectboxColumn("Status", options=_STATUS_OPTIONS_HIST, default="New", width="small"),
                    "notes": st.column_config.TextColumn("Notes", width="medium"),
                    "business_name": st.column_config.TextColumn("Business", width="medium"),
                    "confidence_score": st.column_config.NumberColumn("Confidence", format="%.2f"),
                    "search_city": st.column_config.TextColumn("City", width="small"),
                    "search_niche": st.column_config.TextColumn("Niche", width="small"),
                },
                key="history_editor",
            )
            # Save edits
            if _hist_edited is not None:
                for i in range(len(_hist_edited)):
                    db_id = _hist_ids[i] if i < len(_hist_ids) else 0
                    if not db_id:
                        continue
                    ns = str(_hist_edited.iloc[i].get("status", "New"))
                    nn = str(_hist_edited.iloc[i].get("notes", ""))
                    if ns != (_hist_old_status[i] or "New") or nn != (_hist_old_notes[i] or ""):
                        update_lead_tracking(st.session_state.db_conn, db_id, ns, nn)

            # Download all leads
            _dl_df = _hist_df[["business_name", "address", "phone", "website", "owner_name",
                               "confidence_score", "num_reviews", "search_city", "search_niche",
                               "status", "notes", "last_called"]].copy()
            st.download_button(
                label=f"Download All Leads ({len(_dl_df)})",
                data=_dl_df.to_csv(index=False).encode("utf-8"),
                file_name="all_saved_leads.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if st.button("Back to Lead Finder"):
                st.session_state.show_history = False
                st.rerun()
        return

    def get_reject_reason_ui(row: dict) -> str:
        owner = str(row.get("owner_name") or "").strip().lower()
        conf = float(row.get("confidence_score") or 0)
        if not bool(row.get("solo")):
            return "Not marked solo by owner detection"
        if owner in ("", "unknown", "none", "null"):
            return "No valid owner name found"
        if conf < MIN_CONFIDENCE:
            return f"Confidence below threshold ({conf:.2f} < {MIN_CONFIDENCE:.2f})"
        return "Qualified lead"

    if start_clicked:
        st.session_state.last_error = None
        st.session_state.show_history = False
        # Run scraper on MAIN THREAD (required: Playwright/asyncio subprocess fails in a thread on Windows)
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        table_placeholder = st.empty()

        progress_placeholder.progress(0, text="Starting...")
        status_placeholder.info("Launching browser (a few seconds is normal). Then scraping — keep this tab open for 2–10 min.")

        def set_progress(pct, msg=None):
            progress_placeholder.progress(min(1.0, max(0.0, pct)), text=msg)

        def set_status(msg):
            status_placeholder.info(msg)

        def update_table(rows):
            # Intentionally disabled during scraping to reduce frontend CPU usage.
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            status_placeholder.error(
                "Playwright is not installed. Run: **pip install playwright** then **playwright install chromium**"
            )
            st.session_state.leads_df = None
            st.session_state.scraping_done = True
            st.stop()

        try:
            # Decide how many businesses to process.
            # - In debug mode we honor the explicit cap.
            # - Otherwise we scale with the target number of qualified leads so we
            #   don't over-scrape thousands of listings when you only need a few.
            if debug_mode:
                effective_max = debug_max_businesses
            else:
                # Heuristic: process up to ~6x the desired leads, with sane bounds.
                multiplier = 6
                upper_cap = 60
                effective_max = max(target_leads * multiplier, target_leads + 2)
                effective_max = min(effective_max, upper_cap)
            if use_dataforseo:
                rows = scrape_dataforseo(
                    city=city,
                    niche=niche,
                    max_businesses=effective_max,
                    progress_callback=set_progress,
                    status_callback=set_status,
                    dataforseo_location_name=dataforseo_location_name,
                    reviews_depth=int(reviews_depth),
                    reviews_priority=int(reviews_priority),
                )
            else:
                rows = scrape_google_maps(
                    city=city,
                    niche=niche,
                    max_pages=max_pages,
                    progress_callback=set_progress,
                    status_callback=set_status,
                    table_callback=None,
                    headless=not show_browser,
                    max_businesses=effective_max,
                    target_leads=int(target_leads),
                )

            if not rows:
                status_placeholder.warning("No businesses were scraped. Try a different city/niche or check your connection.")
                st.session_state.leads_df = None
                st.session_state.raw_rows = []
                st.session_state.scraping_done = True
            else:
                qualified_rows = [r for r in rows if is_qualified_lead_row(r)]
                display_cols = ["business_name", "phone", "website", "owner_name", "confidence_score", "num_reviews"]
                st.session_state.raw_rows = list(rows)
                st.session_state.scraping_done = True
                # Save to DB (adds _db_id, _db_status, etc. to each row dict)
                new_count, updated_count = upsert_leads(st.session_state.db_conn, rows, city, niche)
                if not qualified_rows:
                    progress_placeholder.empty()
                    limited_count = sum(1 for r in rows if r.get("_debug", {}).get("limited_view"))
                    df_all = pd.DataFrame(rows)
                    for c in display_cols:
                        if c not in df_all.columns:
                            df_all[c] = ""
                    df_all["confidence_score"] = pd.to_numeric(df_all.get("confidence_score"), errors="coerce").fillna(0).round(2)
                    df_all["reject_reason"] = [get_reject_reason_ui(r) for r in rows]
                    df_all["source"] = [str((r.get("_debug", {}) or {}).get("source") or "") for r in rows]
                    df_all = df_all[display_cols + ["reject_reason", "source"]]
                    st.session_state.leads_df = df_all.copy()
                    st.session_state.showing_all_results = True
                    if limited_count:
                        status_placeholder.warning(
                            f"Scraping complete, but no leads met confidence criteria. "
                            f"Note: Google served a limited Maps view for {limited_count}/{len(rows)} listings, "
                            f"which prevents full review extraction (owners may show as Unknown). "
                            f"Try enabling 'Show browser window (debug)' and rerun."
                        )
                    else:
                        status_placeholder.warning(
                            "Scraping complete, but no leads met confidence criteria. "
                            "Showing all results below. Tip: try increasing 'Reviews per business' to 120+ "
                            "or lowering the confidence threshold."
                        )
                    table_placeholder.empty()
                    # Fall through and render results section below
                else:
                    df = pd.DataFrame(qualified_rows)
                    for c in display_cols:
                        if c not in df.columns:
                            df[c] = ""
                    df = df[display_cols]
                    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0).round(2)
                    st.session_state.leads_df = df.copy()
                    st.session_state.showing_all_results = False
                    # Clear placeholders so only the results section below is shown (avoids duplicate table)
                    progress_placeholder.empty()
                    _db_msg = f" ({new_count} new, {updated_count} updated)" if updated_count else ""
                    status_placeholder.success(
                        f"Done! Found {len(df)} qualified leads (target was {target_leads}).{_db_msg} Saved to database."
                    )
                    table_placeholder.empty()
                # Do NOT rerun: fall through and render results section below

        except Exception as e:
            status_placeholder.error(f"Something went wrong: {str(e)}")
            st.session_state.last_error = traceback.format_exc()
            with st.expander("Error details (click to expand)", expanded=True):
                st.code(st.session_state.last_error)
            st.session_state.leads_df = None
            st.session_state.scraping_done = False

    if st.session_state.last_error:
        st.error("Last run failed. See details below.")
        with st.expander("Last error details", expanded=False):
            st.code(st.session_state.last_error)

    # After run: show table and download
    if st.session_state.leads_df is not None:
        df = st.session_state.leads_df
        raw_rows = st.session_state.get("raw_rows", [])
        showing_all = bool(st.session_state.get("showing_all_results", False))

        total_biz = len(raw_rows) if raw_rows else len(df)
        solo_count = sum(1 for r in raw_rows if is_qualified_lead_row(r))
        hit_rate = f"{(solo_count / total_biz * 100):.0f}%" if total_biz else "0%"

        # --- KPI strip ---
        st.markdown("<div class='solo-kpi-strip'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Businesses Scanned", total_biz)
        with col2:
            st.metric("Qualified Leads", solo_count)
        with col3:
            st.metric("Hit Rate", hit_rate)
        with col4:
            st.metric("Confidence Threshold", f"{MIN_CONFIDENCE*100:.0f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Results table ---
        st.markdown(
            "<div class='solo-section-header'>"
            "<span class='solo-section-title'>Results</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Build display DataFrame with call tracking columns
        df_display = df.copy()

        # Add call tracking columns from DB data stored on raw_rows
        _status_list = []
        _notes_list = []
        _db_ids = []
        for i in range(len(df_display)):
            # Match by business_name to find the raw_row
            bname = str(df_display.iloc[i].get("business_name", ""))
            matched_row = next((r for r in raw_rows if r.get("business_name") == bname), None)
            if matched_row:
                _status_list.append(matched_row.get("_db_status", "New"))
                _notes_list.append(matched_row.get("_db_notes", ""))
                _db_ids.append(matched_row.get("_db_id", 0))
            else:
                _status_list.append("New")
                _notes_list.append("")
                _db_ids.append(0)
        df_display["Status"] = _status_list
        df_display["Notes"] = _notes_list
        df_display["_db_id"] = _db_ids

        # Normalize phone into tel: URLs and website into http URLs
        try:
            if "phone" in df_display.columns:
                norm_phones: list[str] = []
                for val in df_display["phone"].astype(str).tolist():
                    v = val.strip()
                    if not v or v.lower() in ("not listed", "none", "null"):
                        norm_phones.append("")
                        continue
                    digits = "".join(ch for ch in v if ch.isdigit())
                    if not digits:
                        norm_phones.append("")
                        continue
                    if len(digits) == 10:
                        digits = "1" + digits
                    norm_phones.append(f"tel:+{digits}")
                df_display["phone"] = norm_phones
            if "website" in df_display.columns:
                norm_sites: list[str] = []
                for val in df_display["website"].astype(str).tolist():
                    v = val.strip()
                    if not v or v.lower() in ("not listed", "none", "null"):
                        norm_sites.append("")
                        continue
                    if not v.startswith(("http://", "https://")):
                        v = "https://" + v.lstrip("/")
                    norm_sites.append(v)
                df_display["website"] = norm_sites
        except Exception:
            pass

        _STATUS_OPTIONS = ["New", "Called", "No Answer", "Answered", "Interested", "Not Interested"]

        # Inject tel: patch with phone→website mapping
        _tel_web_map = {}
        if "phone" in df_display.columns and "website" in df_display.columns:
            for _, row in df_display.iterrows():
                p = str(row.get("phone", "")).strip()
                w = str(row.get("website", "")).strip()
                if p and w and p.startswith("tel:"):
                    _tel_web_map[p] = w
        _inject_tel_patch(_tel_web_map)

        edited_df = st.data_editor(
            df_display.drop(columns=["_db_id"], errors="ignore"),
            use_container_width=True,
            height=450,
            num_rows="fixed",
            disabled=["business_name", "phone", "website", "owner_name", "confidence_score", "num_reviews", "reject_reason", "source"],
            column_config={
                "phone": st.column_config.LinkColumn("Phone", display_text="Call"),
                "website": st.column_config.LinkColumn("Website", display_text="Open"),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=_STATUS_OPTIONS,
                    default="New",
                    width="small",
                ),
                "Notes": st.column_config.TextColumn(
                    "Notes",
                    width="medium",
                ),
                "confidence_score": st.column_config.NumberColumn("Confidence", format="%.2f"),
                "business_name": st.column_config.TextColumn("Business", width="medium"),
            },
            key="leads_editor",
        )

        # Persist edits back to DB
        if edited_df is not None:
            for i in range(len(edited_df)):
                db_id = _db_ids[i] if i < len(_db_ids) else 0
                if not db_id:
                    continue
                new_status = str(edited_df.iloc[i].get("Status", "New"))
                new_notes = str(edited_df.iloc[i].get("Notes", ""))
                old_status = _status_list[i] if i < len(_status_list) else "New"
                old_notes = _notes_list[i] if i < len(_notes_list) else ""
                if new_status != old_status or new_notes != old_notes:
                    update_lead_tracking(st.session_state.db_conn, db_id, new_status, new_notes)
                    # Update raw_rows so next rerun reflects changes
                    matched = next((r for r in raw_rows if r.get("_db_id") == db_id), None)
                    if matched:
                        matched["_db_status"] = new_status
                        matched["_db_notes"] = new_notes

        # --- Downloads ---
        solo_rows = [r for r in raw_rows if is_qualified_lead_row(r)]
        solo_df = pd.DataFrame(solo_rows) if solo_rows else pd.DataFrame()

        dl_col1, dl_col2 = st.columns(2)
        if showing_all:
            all_rows_df = pd.DataFrame(raw_rows) if raw_rows else df.copy()
            with dl_col1:
                st.download_button(
                    label=f"Download All Results ({len(all_rows_df)})",
                    data=all_rows_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"all_results_{city}_{niche}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        if not solo_df.empty:
            cols = ["business_name", "address", "phone", "website", "owner_name", "confidence_score", "num_reviews"]
            for c in cols:
                if c not in solo_df.columns:
                    solo_df[c] = ""
            solo_df = solo_df[cols]
            solo_count = len(solo_df)
            with dl_col2:
                st.download_button(
                    label=f"Download Solo Owners ({solo_count} leads)",
                    data=solo_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"solo_owners_{city}_{niche}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )
        else:
            st.info("No businesses met the solo-owner filter. Adjust filters or try more results.")

    elif st.session_state.scraping_done and st.session_state.leads_df is None:
        st.markdown(
            "<div class='solo-empty-state'>"
            "<div class='solo-empty-icon'>0</div>"
            "<div class='solo-empty-title'>No results found</div>"
            "<div class='solo-empty-desc'>Try a different city or niche, or adjust your filters in the sidebar.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            "<div class='solo-empty-state'>"
            "<div class='solo-empty-icon'>&#x2192;</div>"
            "<div class='solo-empty-title'>Ready to find leads</div>"
            "<div class='solo-empty-desc'>Choose a city and niche in the sidebar, then click <strong>Start Scraping</strong> to begin.</div>"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
