from __future__ import annotations

import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests

_log = logging.getLogger("solo_app.dataforseo")


API_BASE = "https://api.dataforseo.com"


@dataclass(frozen=True)
class MapsPlace:
    title: str
    address: str | None
    phone: str | None
    website: str | None
    place_id: str | None
    cid: str | None
    url: str | None
    reviews_count: int | None
    rating_value: float | None
    work_hours: dict | None


class DataForSeoError(RuntimeError):
    pass


def _post_json(login: str, password: str, path: str, payload: list[dict[str, Any]]) -> dict[str, Any]:
    r = requests.post(
        f"{API_BASE}{path}",
        auth=(login, password),
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if r.status_code >= 400:
        raise DataForSeoError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def _get_json(login: str, password: str, path: str) -> dict[str, Any]:
    r = requests.get(
        f"{API_BASE}{path}",
        auth=(login, password),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    if r.status_code >= 400:
        raise DataForSeoError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


_LOCATION_CODE_CACHE: dict[str, int] = {}


def resolve_location_code(*, login: str, password: str, location_name: str) -> int:
    """
    Resolve a human-readable location name to DataForSEO `location_code`.

    Uses the free endpoint:
      GET /v3/serp/google/locations/$country
    Docs:
      https://docs.dataforseo.com/v3/serp/google/locations
    """
    key = (location_name or "").strip()
    if not key:
        raise DataForSeoError("Empty DataForSEO location_name")
    if key in _LOCATION_CODE_CACHE:
        return _LOCATION_CODE_CACHE[key]

    # Common fast-path.
    if key.lower() in {"united states", "us", "usa"}:
        _LOCATION_CODE_CACHE[key] = 2840
        return 2840

    # Try to fetch US locations list (smaller than full global list).
    data = _get_json(login, password, "/v3/serp/google/locations/us")
    tasks = data.get("tasks") or []
    if not tasks or tasks[0].get("status_code") != 20000:
        msg = (tasks[0].get("status_message") if tasks else data.get("status_message")) or "Unknown error"
        raise DataForSeoError(f"Failed to load locations list: {msg}")

    results = tasks[0].get("result") or []
    # Exact match first.
    for loc in results:
        if str(loc.get("location_name") or "").strip().lower() == key.lower():
            code = int(loc.get("location_code"))
            _LOCATION_CODE_CACHE[key] = code
            return code

    # Fuzzy contains match (e.g. user typed "Phoenix, AZ" or partial).
    key_tokens = [t.strip().lower() for t in re.split(r"[,\s]+", key) if t.strip()]
    best_code = None
    best_score = -1
    for loc in results:
        name = str(loc.get("location_name") or "").strip().lower()
        if not name:
            continue
        score = sum(1 for tok in key_tokens if tok in name)
        if score > best_score:
            best_score = score
            best_code = loc.get("location_code")
    if best_code is not None and best_score >= 1:
        code = int(best_code)
        _LOCATION_CODE_CACHE[key] = code
        return code

    raise DataForSeoError(
        f"Could not resolve DataForSEO location_code for location_name='{location_name}'. "
        f"Try full format like 'Phoenix,Arizona,United States'."
    )


def maps_search(
    *,
    login: str,
    password: str,
    keyword: str,
    location_name: str,
    language_code: str = "en",
    depth: int = 100,
    device: str = "desktop",
    os_name: str = "windows",
) -> list[MapsPlace]:
    """
    Calls:
      POST /v3/serp/google/maps/live/advanced
    Docs:
      https://docs.dataforseo.com/v3/serp/google/maps/live/advanced
    """
    def do_request(task: dict[str, Any]) -> dict[str, Any]:
        return _post_json(login, password, "/v3/serp/google/maps/live/advanced", [task])

    location_code = resolve_location_code(login=login, password=password, location_name=location_name)

    base_task: dict[str, Any] = {
        "keyword": keyword,
        "location_code": int(location_code),
        "language_code": language_code,
        "depth": depth,
    }
    full_task: dict[str, Any] = {
        **base_task,
        "device": device,
        "os": os_name,
        "search_places": False,
        "search_this_area": True,
    }

    data = do_request(full_task)
    tasks = data.get("tasks") or []
    if not tasks or tasks[0].get("status_code") not in (20000, 20100):
        msg = (tasks[0].get("status_message") if tasks else data.get("status_message")) or "Unknown error"
        # Some accounts/plans or API variations reject optional fields with "Invalid Field".
        # Retry with required fields only.
        if "invalid field" in str(msg).lower():
            data = do_request(base_task)
            tasks = data.get("tasks") or []
            if not tasks or tasks[0].get("status_code") not in (20000, 20100):
                msg2 = (tasks[0].get("status_message") if tasks else data.get("status_message")) or "Unknown error"
                raise DataForSeoError(f"Maps search failed (minimal retry): {msg2}")
        else:
            raise DataForSeoError(f"Maps search failed: {msg}")

    results = tasks[0].get("result") or []
    if not results:
        return []
    items = results[0].get("items") or []

    out: list[MapsPlace] = []
    for it in items:
        if (it.get("type") or "") not in ("maps_search", "maps_paid_item"):
            continue
        rating = it.get("rating") or {}
        rating_value = rating.get("value")
        try:
            rating_value = float(rating_value) if rating_value is not None else None
        except Exception:
            rating_value = None

        votes_count = None
        try:
            votes_count = int(rating.get("votes_count")) if rating.get("votes_count") is not None else None
        except Exception:
            votes_count = None

        out.append(
            MapsPlace(
                title=str(it.get("title") or "").strip(),
                address=(it.get("address") or it.get("snippet") or None),
                phone=(it.get("phone") or None),
                website=(it.get("url") or None),
                place_id=(it.get("place_id") or None),
                cid=(it.get("cid") or None),
                url=(it.get("url") or None),
                reviews_count=votes_count,
                rating_value=rating_value,
                work_hours=(it.get("work_hours") or None),
            )
        )
    return [p for p in out if p.title]


def fetch_reviews_text(
    *,
    login: str,
    password: str,
    place_id: str | None,
    cid: str | None,
    keyword_fallback: str,
    location_name: str,
    language_code: str = "en",
    depth: int = 60,
    priority: int = 2,
    poll_timeout_s: int = 90,
) -> tuple[list[str], dict[str, Any]]:
    """
    Calls:
      POST /v3/business_data/google/reviews/task_post
      GET  /v3/business_data/google/reviews/task_get/$id
    Docs:
      https://docs.dataforseo.com/v3/business_data-google-reviews-task_post/
      https://docs.dataforseo.com/v3/business_data/google/reviews/task_get/
    """
    # The Reviews API requires location_code (integer), NOT location_name (string).
    # Resolve the human-readable name to a numeric code, same as maps_search does.
    location_code = resolve_location_code(login=login, password=password, location_name=location_name)

    task: dict[str, Any] = {
        "location_code": int(location_code),
        "language_code": language_code,
        "depth": int(depth),
        "priority": int(priority),
        "sort_by": "relevant",
    }
    if place_id:
        task["place_id"] = place_id
    elif cid:
        task["cid"] = cid
    else:
        task["keyword"] = keyword_fallback

    post = _post_json(login, password, "/v3/business_data/google/reviews/task_post", [task])
    tasks = post.get("tasks") or []
    if not tasks or tasks[0].get("status_code") not in (20000, 20100):
        msg = (tasks[0].get("status_message") if tasks else post.get("status_message")) or "Unknown error"
        raise DataForSeoError(f"Reviews task_post failed: {msg}")

    task_id = tasks[0].get("id")
    if not task_id:
        raise DataForSeoError("Reviews task_post returned no task id")

    started = time.monotonic()
    last_payload: dict[str, Any] = {}
    while time.monotonic() - started < poll_timeout_s:
        got = _get_json(login, password, f"/v3/business_data/google/reviews/task_get/{task_id}")
        last_payload = got
        t = (got.get("tasks") or [{}])[0]
        # 20000 indicates success; 40602 "Task in progress" can appear transiently.
        if t.get("status_code") == 20000:
            result = (t.get("result") or [{}])[0]
            items = result.get("items") or []
            texts: list[str] = []
            owner_answers: list[str] = []
            for item in items:
                txt = (item.get("review_text") or item.get("original_review_text") or "").strip()
                if len(txt) >= 20:
                    texts.append(txt)
                # Collect owner replies — these often contain the owner's name
                owner_ans = (item.get("owner_answer") or item.get("original_owner_answer") or "").strip()
                if owner_ans:
                    owner_answers.append(owner_ans)
            # Attach owner answers to the payload for downstream extraction
            got["_owner_answers"] = owner_answers
            return texts, got
        time.sleep(2)

    raise DataForSeoError("Reviews task_get timed out waiting for results")


# ---------------------------------------------------------------------------
# Batch reviews — post all tasks at once, poll in parallel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewRequest:
    """One review-fetch request for batch posting."""
    index: int  # caller's index so results can be matched back
    place_id: str | None
    cid: str | None
    keyword_fallback: str


def fetch_reviews_batch(
    *,
    login: str,
    password: str,
    requests_list: list[ReviewRequest],
    location_name: str,
    language_code: str = "en",
    depth: int = 60,
    priority: int = 2,
    poll_timeout_s: int = 90,
) -> dict[int, tuple[list[str], dict[str, Any]]]:
    """
    Post ALL review tasks in a single API call, then poll results in parallel.

    Returns a dict mapping each ReviewRequest.index -> (texts, raw_payload).
    Entries that failed are mapped to ([], {"error": "..."}).
    """
    if not requests_list:
        return {}

    location_code = resolve_location_code(login=login, password=password, location_name=location_name)

    # Build the batch payload — one task dict per business
    task_payloads: list[dict[str, Any]] = []
    for req in requests_list:
        task: dict[str, Any] = {
            "location_code": int(location_code),
            "language_code": language_code,
            "depth": int(depth),
            "priority": int(priority),
            "sort_by": "relevant",
        }
        if req.place_id:
            task["place_id"] = req.place_id
        elif req.cid:
            task["cid"] = req.cid
        else:
            task["keyword"] = req.keyword_fallback
        task_payloads.append(task)

    _log.info("Posting %d review tasks in one batch call", len(task_payloads))

    # Single POST with all tasks
    post = _post_json(login, password, "/v3/business_data/google/reviews/task_post", task_payloads)
    tasks_resp = post.get("tasks") or []

    # Map each posted task to its task_id (or error)
    task_ids: dict[int, str] = {}  # req.index -> task_id
    out: dict[int, tuple[list[str], dict[str, Any]]] = {}

    for i, req in enumerate(requests_list):
        if i < len(tasks_resp):
            t = tasks_resp[i]
            if t.get("status_code") in (20000, 20100) and t.get("id"):
                task_ids[req.index] = t["id"]
            else:
                msg = t.get("status_message") or "Unknown error"
                _log.warning("Batch task_post failed for index %d: %s", req.index, msg)
                out[req.index] = ([], {"error": f"task_post: {msg}"})
        else:
            out[req.index] = ([], {"error": "No task response returned"})

    _log.info("Batch post: %d task IDs received, %d failed", len(task_ids), len(out))

    if not task_ids:
        return out

    # Poll all task IDs in parallel threads
    def _poll_one(idx: int, tid: str) -> tuple[int, list[str], dict[str, Any]]:
        started = time.monotonic()
        while time.monotonic() - started < poll_timeout_s:
            got = _get_json(login, password, f"/v3/business_data/google/reviews/task_get/{tid}")
            t = (got.get("tasks") or [{}])[0]
            if t.get("status_code") == 20000:
                result = (t.get("result") or [{}])[0]
                items = result.get("items") or []
                texts: list[str] = []
                owner_answers: list[str] = []
                for item in items:
                    txt = (item.get("review_text") or item.get("original_review_text") or "").strip()
                    if len(txt) >= 20:
                        texts.append(txt)
                    owner_ans = (item.get("owner_answer") or item.get("original_owner_answer") or "").strip()
                    if owner_ans:
                        owner_answers.append(owner_ans)
                got["_owner_answers"] = owner_answers
                return idx, texts, got
            time.sleep(2)
        return idx, [], {"error": "poll timed out"}

    with ThreadPoolExecutor(max_workers=min(len(task_ids), 8)) as pool:
        futures = {pool.submit(_poll_one, idx, tid): idx for idx, tid in task_ids.items()}
        for future in as_completed(futures):
            idx, texts, payload = future.result()
            out[idx] = (texts, payload)
            _log.info("Poll complete for index %d: %d reviews", idx, len(texts))

    return out

