import json
import os
import shutil

from app import (
    _build_owner_detection_prompt,
    _extract_meta_features,
    _tokenize_for_training,
    append_review_labels,
    detect_owner_with_ollama,
    prepare_trainer_rows_for_labeling,
    score_would_call_probability,
    scrape_google_maps,
    train_review_preference_model,
    validate_owner_detection,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_LABELS = os.path.join(BASE_DIR, "trainer_test_labels.csv")
TEST_MODEL = os.path.join(BASE_DIR, "trainer_test_model.json")
PROD_MODEL = os.path.join(BASE_DIR, "review_trainer_model.json")
PROD_LABELS = os.path.join(BASE_DIR, "review_training_labels.csv")
PROD_MODEL_BAK = os.path.join(BASE_DIR, "review_trainer_model.backup_for_test.json")
PROD_LABELS_BAK = os.path.join(BASE_DIR, "review_training_labels.backup_for_test.csv")


def cleanup_test_artifacts() -> None:
    for p in (TEST_LABELS, TEST_MODEL):
        if os.path.exists(p):
            os.remove(p)


def stash_prod_files() -> None:
    if os.path.exists(PROD_MODEL_BAK):
        os.remove(PROD_MODEL_BAK)
    if os.path.exists(PROD_LABELS_BAK):
        os.remove(PROD_LABELS_BAK)
    if os.path.exists(PROD_MODEL):
        shutil.copy2(PROD_MODEL, PROD_MODEL_BAK)
    if os.path.exists(PROD_LABELS):
        shutil.copy2(PROD_LABELS, PROD_LABELS_BAK)


def restore_prod_files() -> None:
    if os.path.exists(PROD_MODEL_BAK):
        shutil.copy2(PROD_MODEL_BAK, PROD_MODEL)
        os.remove(PROD_MODEL_BAK)
    elif os.path.exists(PROD_MODEL):
        os.remove(PROD_MODEL)

    if os.path.exists(PROD_LABELS_BAK):
        shutil.copy2(PROD_LABELS_BAK, PROD_LABELS)
        os.remove(PROD_LABELS_BAK)
    elif os.path.exists(PROD_LABELS):
        os.remove(PROD_LABELS)


def install_test_training_context() -> None:
    shutil.copy2(TEST_MODEL, PROD_MODEL)
    shutil.copy2(TEST_LABELS, PROD_LABELS)


def run_scrape_test(city: str, niche: str, n: int) -> list[dict]:
    print(f"\n--- Scraping {n} businesses: {niche} in {city} ---")
    rows = scrape_google_maps(
        city=city,
        niche=niche,
        max_pages=1,
        headless=True,
        max_businesses=n,
        run_owner_detection=False,
        review_snippets_target=6,
    )
    print(f"  scraped {len(rows)} rows")
    for idx, r in enumerate(rows):
        debug = r.get("_debug") or {}
        snips = list(debug.get("sample_review_snippets") or [])
        listed = int(r.get("num_reviews") or 0)
        pulled = int(debug.get("review_snippet_count") or len(snips))
        gap = debug.get("review_pull_gap")
        gap_str = ""
        if gap:
            gap_str = f"  gap_note={gap.get('note', '')[:60]}"
        print(
            f"  [{idx}] {r.get('business_name', '?')[:42]}  "
            f"rating={r.get('rating')}  listed={listed}  pulled={pulled}  "
            f"snippets={len(snips)}{gap_str}"
        )
        if listed > 0 and listed < 10:
            assert pulled <= listed, (
                f"Pulled {pulled} but only {listed} listed for {r.get('business_name')}"
            )
        if pulled > 0:
            assert len(snips) > 0, f"snippet list empty despite pulled={pulled}"
    return rows


def run_tokenizer_test() -> None:
    print("\n--- Tokenizer unit tests ---")
    t1 = _tokenize_for_training("Jeff is the owner and his team did great work")
    assert "jeff" in t1, f"missing jeff in {t1}"
    assert any("jeff" in b and "owner" in b for b in t1), f"missing jeff_owner bigram in {t1}"
    print(f"  basic tokens OK: {t1[:8]}...")

    t2 = _tokenize_for_training("Not the owner, never seen Jeff personally")
    assert "NOT_owner" in t2, f"missing NOT_owner in {t2}"
    assert "NOT_seen" in t2 or "NOT_jeff" in t2, f"missing negated token in {t2}"
    print(f"  negation tokens OK: {t2[:8]}...")

    t3 = _tokenize_for_training("Family run, owner operated small shop")
    assert any("family" in tok for tok in t3), f"missing family in {t3}"
    print(f"  phrase tokens OK: {t3[:8]}...")
    print("  PASS")


def run_meta_features_test() -> None:
    print("\n--- Meta-features unit tests ---")
    m1 = _extract_meta_features("The owner Jeff runs this himself with his team")
    assert "_META_OWNER" in m1
    assert "_META_TEAM" in m1
    assert "_META_PERSONAL" in m1
    print(f"  positive features: {m1}")

    m2 = _extract_meta_features("Different technician each visit, multiple crew members")
    assert "_META_ROTATING" in m2
    assert "_META_ROLE" in m2
    print(f"  negative features: {m2}")

    m3 = _extract_meta_features("Great family business, husband and wife team")
    assert "_META_FAMILY" in m3
    print(f"  family features: {m3}")
    print("  PASS")


def build_synthetic_labels(rows: list[dict]) -> list[dict]:
    label_rows = []
    for idx, r in enumerate(rows):
        snips = list((r.get("_debug", {}) or {}).get("sample_review_snippets") or [])[:6]
        is_yes = idx % 2 == 0
        if is_yes:
            reason = "Owner-run shop, ask for owner directly, family-run, hands-on personal service"
            evidence_hl = [{"review_index": 1, "text": "owner was on site"}] if snips else []
        else:
            reason = "Multiple different technicians, rotating crew, large team operation"
            evidence_hl = [{"review_index": 1, "text": "different technician each time"}] if snips else []
        label_rows.append(
            {
                "timestamp_utc": "test",
                "city": "Test",
                "niche": "test",
                "business_name": r.get("business_name", ""),
                "address": r.get("address", ""),
                "phone": r.get("phone", ""),
                "website": r.get("website", ""),
                "rating": r.get("rating", None),
                "num_reviews": int(r.get("num_reviews") or 0),
                "would_call": "yes" if is_yes else "no",
                "reason": reason,
                "evidence_quote": "",
                "highlighted_evidence_json": json.dumps(evidence_hl, ensure_ascii=False),
                "owner_name_guess": "Jeff" if is_yes else "",
                "reviews_json": json.dumps(snips, ensure_ascii=False),
            }
        )
    if label_rows and not any(r["would_call"] == "yes" for r in label_rows):
        extra = dict(label_rows[0])
        extra["would_call"] = "yes"
        extra["reason"] = "Owner-run family shop"
        label_rows.append(extra)
    if label_rows and not any(r["would_call"] == "no" for r in label_rows):
        extra = dict(label_rows[0])
        extra["would_call"] = "no"
        extra["reason"] = "Large rotating crew, multiple technicians"
        label_rows.append(extra)
    return label_rows


def run_prompt_builder_test() -> None:
    print("\n--- Few-shot prompt builder test ---")
    prompt = _build_owner_detection_prompt(
        "Jeff is mentioned as the owner in multiple reviews.",
        "Test Detail Shop",
        labels_path=TEST_LABELS,
    )
    assert "USER PREFERENCE EXAMPLES" in prompt, "Prompt missing few-shot section"
    assert "Example (YES)" in prompt, "Prompt missing positive example"
    assert "Example (NO)" in prompt, "Prompt missing negative example"
    assert "owner was on site" in prompt or "different technician each time" in prompt
    print("  prompt few-shot examples embedded")
    print("  PASS")


def run_training_test(label_rows: list[dict]) -> None:
    print("\n--- Training + scoring test ---")
    append_review_labels(label_rows, city="Test", niche="test", labels_path=TEST_LABELS)
    n_yes = sum(1 for r in label_rows if r["would_call"] == "yes")
    n_no = sum(1 for r in label_rows if r["would_call"] == "no")
    print(f"  wrote {len(label_rows)} labels ({n_yes} yes, {n_no} no)")

    model = train_review_preference_model(labels_path=TEST_LABELS, model_path=TEST_MODEL)
    print(
        f"  model: version={model.get('version')} yes={model.get('n_yes')} "
        f"no={model.get('n_no')} vocab={model.get('vocab_size')}"
    )
    assert model.get("version") == 2, f"Expected model version 2, got {model.get('version')}"

    pos_text = "Ask for Jeff, owner was on site and Jeff and his team did excellent work."
    neg_text = "Different technician each visit and multiple crew members handled the job."
    negated_text = "Jeff was not the owner and different technicians came each time."
    p_yes = score_would_call_probability(pos_text, "Jeff's Detail Shop", model_path=TEST_MODEL)
    p_no = score_would_call_probability(neg_text, "City HVAC Services", model_path=TEST_MODEL)
    p_negated = score_would_call_probability(negated_text, "City HVAC Services", model_path=TEST_MODEL)
    print(f"  score_yes={p_yes:.4f}  score_no={p_no:.4f}  score_negated={p_negated:.4f}")
    assert p_yes > p_no, f"Expected yes score > no score but {p_yes:.4f} <= {p_no:.4f}"
    assert p_yes > 0.60, f"Yes score too low: {p_yes:.4f}"
    assert p_no < 0.40, f"No score too high: {p_no:.4f}"
    assert p_negated < p_yes, f"Negated owner sentence scored too high: {p_negated:.4f}"
    print("  PASS")


def run_trainer_queue_test() -> None:
    print("\n--- Trainer filtering + priority test ---")
    rows = [
        {
            "business_name": "Jeff's Detail Shop",
            "address": "123 Main",
            "phone": "111",
            "website": "",
            "rating": 4.9,
            "num_reviews": 31,
            "_debug": {
                "sample_review_snippets": [
                    "Ask for Jeff, the owner. Jeff personally handled everything from the quote to the final delivery, and his team was excellent throughout the whole job.",
                    "Owner was on site, answered every question himself, and this family-run shop did a great job from start to finish with clear communication.",
                ]
            },
        },
        {
            "business_name": "Borderline Auto Care",
            "address": "456 Pine",
            "phone": "222",
            "website": "",
            "rating": 4.8,
            "num_reviews": 44,
            "_debug": {
                "sample_review_snippets": [
                    "Jeff helped me and the owner was friendly, but I had a different technician on the second visit and another crew member handled pickup the first time.",
                    "Good work overall, though multiple crew members were involved and I dealt with several different people during scheduling and delivery.",
                ]
            },
        },
        {
            "business_name": "Weak Listing",
            "address": "789 Oak",
            "phone": "333",
            "website": "",
            "rating": 5.0,
            "num_reviews": 12,
            "_debug": {"sample_review_snippets": ["Great service"]},
        },
    ]
    prepared, stats = prepare_trainer_rows_for_labeling(rows, run_llm=False)
    print(f"  stats={stats}")
    assert len(prepared) == 2, f"Expected weak row filtered out, got {len(prepared)} rows"
    assert stats.get("filtered_weak_count") == 1, f"Expected 1 weak row filtered, got {stats}"
    assert all((r.get("_trainer") or {}).get("priority") for r in prepared), "Missing priority metadata"
    print(
        f"  top queue labels={[((r.get('_trainer') or {}).get('priority') or {}).get('queue_label') for r in prepared]}"
    )
    print("  PASS")


def run_ollama_prompt_smoke_test() -> None:
    print("\n--- Ollama few-shot smoke test ---")
    sample_reviews = (
        "Jeff is the owner and personally handled my car. Ask for Jeff if you call. "
        "Jeff and his team did great work, but Jeff was clearly the point person."
    )
    det = detect_owner_with_ollama(sample_reviews, "Jeff's Auto Spa")
    print(
        f"  det owner={det.get('owner_name')} solo={det.get('solo')} "
        f"conf={det.get('confidence')} reason={det.get('reason')}"
    )
    assert "Ollama error" not in str(det.get("reason") or ""), det
    assert det.get("reason"), "Expected a non-empty Ollama reason"
    print("  PASS")


def run_live_trainer_pipeline_test(rows: list[dict]) -> None:
    print("\n--- Live trainer preparation test ---")
    prepared, stats = prepare_trainer_rows_for_labeling(rows, run_llm=True)
    print(f"  live stats={stats}")
    assert prepared, "Trainer preparation returned no rows"
    top = prepared[0]
    top_trainer = top.get("_trainer") or {}
    quality = top_trainer.get("quality") or {}
    priority = top_trainer.get("priority") or {}
    print(
        f"  top={top.get('business_name')} quality={quality} priority={priority} "
        f"llm={top_trainer.get('llm_detection')}"
    )
    assert priority, "Missing trainer priority on live prepared row"
    print("  PASS")


def run_overlay_test() -> None:
    print("\n--- Overlay integration test ---")
    pos_text = "Jeff is the owner. Jeff personally handled my car. Jeff and his team are great."
    det_pos = validate_owner_detection(
        pos_text,
        "Jeff's Auto Spa",
        {"owner_name": "Jeff", "solo": True, "confidence": 0.50, "reason": "base"},
    )
    print(
        f"  positive: owner={det_pos['owner_name']} solo={det_pos['solo']} "
        f"conf={det_pos['confidence']:.2f} reason={det_pos['reason'][:100]}"
    )
    assert det_pos["solo"] is True
    assert det_pos["confidence"] > 0.50, f"Expected confidence boost, got {det_pos['confidence']}"
    assert "Trainer model" in det_pos["reason"]

    neg_text = "Different technician each visit. Multiple crew members. Various workers handled it."
    det_neg = validate_owner_detection(
        neg_text,
        "Big City HVAC",
        {"owner_name": None, "solo": False, "confidence": 0.30, "reason": "no owner"},
    )
    print(
        f"  negative: owner={det_neg['owner_name']} solo={det_neg['solo']} "
        f"conf={det_neg['confidence']:.2f} reason={det_neg['reason'][:100]}"
    )
    assert det_neg["solo"] is False
    print("  PASS")


def main() -> None:
    cleanup_test_artifacts()
    stash_prod_files()
    try:
        run_tokenizer_test()
        run_meta_features_test()

        print("\n======= LIVE SCRAPE A: mobile detailer in Seattle (3 businesses) =======")
        rows1 = run_scrape_test("Seattle", "mobile detailer", 3)
        assert len(rows1) >= 1, "Scrape A returned 0 rows"

        print("\n======= LIVE SCRAPE B: auto detailer in Portland (3 businesses) =======")
        rows2 = run_scrape_test("Portland", "auto detailer", 3)
        assert len(rows2) >= 1, "Scrape B returned 0 rows"

        all_rows = rows1 + rows2
        labels = build_synthetic_labels(all_rows)
        run_training_test(labels)
        run_prompt_builder_test()

        install_test_training_context()
        run_trainer_queue_test()
        run_ollama_prompt_smoke_test()
        run_live_trainer_pipeline_test(all_rows)
        run_overlay_test()
    finally:
        restore_prod_files()
        cleanup_test_artifacts()

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
