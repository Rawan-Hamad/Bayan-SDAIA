"""Runner contracts use controlled predictors, not model-quality measurements."""
from bayan.evaluation.behavioural import build_cases, run_behavioural_suite


def test_missing_models_are_not_passes():
    report = run_behavioural_suite()
    assert [report["summary"][k]["total"] for k in ("invariance", "directional", "mft")] == [200, 200, 16]
    for group in report["summary"].values():
        assert group["evaluated"] == 0
        assert group["pass_rate"] is None
        assert group["target_met"] is None


def test_negation_pairs_are_distinct_and_not_double_negated():
    for case in build_cases():
        if case["test_type"] == "directional":
            token = "is not working" if case["lang"] == "en" else "\u0644\u0627 \u062a\u0639\u0645\u0644"
            assert token not in case["baseline"]
            assert case["text"].count(token) == 1


def test_successful_predictors_and_deduplicated_coverage():
    expected = {c["text"]: c["expected"] for c in build_cases() if c["test_type"] == "mft"}
    def sentiment(text):
        return -1.0 if "not working" in text or "\u0644\u0627 \u062a\u0639\u0645\u0644" in text else 1.0
    report = run_behavioural_suite(lambda text: expected.get(text, "roads"), sentiment)
    for group in report["summary"].values():
        assert group["pass_rate"] == 1.0
    assert report["summary"]["invariance"]["target_met"] is True
    assert report["summary"]["mft"]["target_met"] is True
    assert report["summary"]["invariance"]["unique_cases"] < 200


def test_regressions_fail_and_are_recorded():
    def topic(text):
        return "water" if text.startswith("  ") else "roads"
    def sentiment(text):
        return 1.0 if "not working" in text or "\u0644\u0627 \u062a\u0639\u0645\u0644" in text else -1.0
    report = run_behavioural_suite(topic, sentiment)
    assert report["summary"]["invariance"]["failure_rate"] == 1.0
    assert report["summary"]["directional"]["failure_rate"] == 1.0
    assert report["summary"]["mft"]["passed"] == 2
    assert report["summary"]["mft"]["target_met"] is False


def test_invalid_predictions_and_errors_count_as_failures():
    def broken(text):
        raise RuntimeError("model failure")
    report = run_behavioural_suite(broken, lambda text: float("nan"))
    assert all(row["status"] == "failed" for row in report["results"])
    assert all(group["pass_rate"] == 0.0 for group in report["summary"].values())
