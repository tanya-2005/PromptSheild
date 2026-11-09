import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from promptshield.confidence_calibration import (
    BASELINE_POLICY_ID,
    brier_score,
    build_report,
    evaluate_abstention,
    evaluate_policy_grid,
    expected_calibration_error,
    pareto_frontier,
    render_markdown,
    reliability_bins,
    run_study,
    smallest_abstention_for_error_reduction,
    validate_scores,
    write_reports,
)


class ScoreValidationTests(unittest.TestCase):
    def test_rejects_empty_mismatched_and_out_of_range_inputs(self):
        with self.assertRaisesRegex(ValueError, "no rows"):
            validate_scores([], [])
        with self.assertRaisesRegex(ValueError, "expected 2"):
            validate_scores([0, 1], [0.1, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            validate_scores([0], [1.5])
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_scores([0], [float("nan")])
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_scores([2], [0.5])


class ReliabilityTests(unittest.TestCase):
    def test_perfectly_calibrated_scores_give_zero_ece_and_brier(self):
        labels = [0, 0, 1, 1]
        scores = [0.0, 0.0, 1.0, 1.0]
        bins = reliability_bins(labels, scores, bin_count=10)
        self.assertEqual(expected_calibration_error(bins, len(labels)), 0.0)
        self.assertEqual(brier_score(labels, scores), 0.0)

    def test_single_bin_hand_computed_ece_and_brier(self):
        # One bin [0.5, 0.6): three examples, mean score 0.55, positive
        # rate 1/3, gap 0.55 - 1/3; ECE = 1 * |gap|.
        labels = [1, 0, 0]
        scores = [0.55, 0.55, 0.55]
        bins = reliability_bins(labels, scores, bin_count=10)
        non_empty = [entry for entry in bins if entry["count"]]
        self.assertEqual(len(non_empty), 1)
        entry = non_empty[0]
        self.assertEqual(entry["interval"], "[0.50, 0.60)")
        self.assertAlmostEqual(entry["mean_score"], 0.55)
        self.assertAlmostEqual(entry["empirical_positive_rate"], 1 / 3)
        expected_gap = abs(0.55 - 1 / 3)
        self.assertAlmostEqual(entry["absolute_gap"], expected_gap)
        self.assertAlmostEqual(
            expected_calibration_error(bins, 3), expected_gap
        )
        expected_brier = (
            (0.55 - 1) ** 2 + (0.55 - 0) ** 2 + (0.55 - 0) ** 2
        ) / 3
        self.assertAlmostEqual(brier_score(labels, scores), expected_brier)

    def test_boundary_scores_go_to_upper_bin_and_one_to_final_bin(self):
        labels = [0, 1, 0]
        scores = [0.3, 0.6, 1.0]
        bins = reliability_bins(labels, scores, bin_count=10)
        occupied = {entry["index"] for entry in bins if entry["count"]}
        self.assertEqual(occupied, {3, 6, 9})
        final = bins[-1]
        self.assertEqual(final["interval"], "[0.90, 1.00]")
        self.assertEqual(final["count"], 1)

    def test_empty_bins_are_reported_and_bin_count_is_validated(self):
        bins = reliability_bins([0], [0.95], bin_count=4)
        self.assertEqual(len(bins), 4)
        self.assertEqual(sum(entry["count"] for entry in bins), 1)
        self.assertEqual(
            [entry["count"] for entry in bins], [0, 0, 0, 1]
        )
        self.assertIsNone(bins[0]["mean_score"])
        with self.assertRaisesRegex(ValueError, "bin_count"):
            reliability_bins([0], [0.5], bin_count=1)


class SelectivePredictionTests(unittest.TestCase):
    def test_baseline_with_no_abstention_equals_deployed_rule(self):
        labels = [0, 0, 1, 1]
        scores = [0.05, 0.95, 0.10, 0.99]
        result = evaluate_abstention(labels, scores)
        self.assertEqual(result["policy_id"], BASELINE_POLICY_ID)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["abstention_rate"], 0.0)
        self.assertEqual(result["accepted_true_negative"], 1)
        self.assertEqual(result["accepted_false_positive"], 1)
        self.assertEqual(result["accepted_false_negative"], 1)
        self.assertEqual(result["accepted_true_positive"], 1)
        self.assertEqual(result["accepted_case_error_rate"], 0.5)
        self.assertEqual(result["abstained_count"], 0)
        self.assertEqual(result["fraction_of_all_errors_caught"], 0.0)

    def test_accounting_identities_hold_for_band_policy(self):
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores = [0.10, 0.55, 0.85, 0.95, 0.20, 0.60, 0.92, 0.99]
        result = evaluate_abstention(
            labels, scores, benign_accept_max=0.50, malicious_accept_min=0.95
        )
        total = len(labels)
        self.assertEqual(result["accepted_count"] + result["abstained_count"], total)
        self.assertEqual(result["coverage"] + result["abstention_rate"], 1.0)
        self.assertEqual(result["abstained_count"], 4)
        self.assertEqual(result["abstained_benign"], 2)
        self.assertEqual(result["abstained_malicious"], 2)
        # Accepted benign: 0.10 (TN), 0.20 (FN). Accepted malicious:
        # 0.95 (FP), 0.99 (TP). Errors among accepted: FP+FN = 2 of 4.
        self.assertEqual(result["accepted_false_positive"], 1)
        self.assertEqual(result["accepted_false_negative"], 1)
        self.assertEqual(result["accepted_true_negative"], 1)
        self.assertEqual(result["accepted_true_positive"], 1)
        self.assertEqual(result["accepted_case_error_rate"], 0.5)

    def test_errors_caught_counts_deployed_rule_errors_in_band(self):
        # Band straddles the 0.90 threshold: abstained cases on the benign
        # side that are malicious and on the malicious side that are benign
        # are the errors caught.
        labels = [0, 1, 0, 1]
        scores = [0.85, 0.85, 0.95, 0.95]
        result = evaluate_abstention(
            labels, scores, benign_accept_max=0.50, malicious_accept_min=0.99
        )
        self.assertEqual(result["abstained_count"], 4)
        # Deployed rule (>=0.90 -> malicious): 0.85/label1 is an error caught;
        # 0.85/label0 correct; 0.95/label0 error caught; 0.95/label1 correct.
        self.assertEqual(result["abstained_deployed_error_rate"], 0.5)
        self.assertEqual(result["fraction_of_all_errors_caught"], 1.0)
        # Nothing is accepted, so the accepted-case error rate is undefined.
        self.assertIsNone(result["accepted_case_error_rate"])

    def test_all_abstention_gives_null_accepted_error_rate(self):
        result = evaluate_abstention(
            [0, 1], [0.5, 0.6],
            benign_accept_max=0.0, malicious_accept_min=1.0,
        )
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["accepted_case_error_rate"])
        self.assertEqual(result["abstained_count"], 2)

    def test_invalid_band_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "benign_accept_max"):
            evaluate_abstention([0], [0.5], benign_accept_max=0.95)
        with self.assertRaisesRegex(ValueError, "malicious_accept_min"):
            evaluate_abstention([0], [0.5], malicious_accept_min=0.85)


class CandidateSelectionTests(unittest.TestCase):
    def test_grid_contains_baseline_first_and_no_threshold_change(self):
        policies = evaluate_policy_grid([0, 1], [0.1, 0.9])
        self.assertEqual(policies[0]["policy_id"], BASELINE_POLICY_ID)
        self.assertEqual(policies[0]["coverage"], 1.0)
        # 4 benign candidates x 3 malicious candidates.
        self.assertEqual(len(policies), 12)
        for policy in policies:
            self.assertLessEqual(policy["benign_accept_max"], 0.90)
            self.assertGreaterEqual(policy["malicious_accept_min"], 0.90)

    def test_pareto_frontier_excludes_dominated_policies(self):
        policies = [
            {"policy_id": "a", "coverage": 1.0, "accepted_case_error_rate": 0.20},
            {"policy_id": "b", "coverage": 0.8, "accepted_case_error_rate": 0.20},
            {"policy_id": "c", "coverage": 0.9, "accepted_case_error_rate": 0.10},
            {"policy_id": "d", "coverage": 0.5, "accepted_case_error_rate": 0.30},
        ]
        frontier_ids = [
            policy["policy_id"] for policy in pareto_frontier(policies)
        ]
        self.assertEqual(frontier_ids, ["a", "c"])

    def test_reference_rule_picks_smallest_abstention_meeting_target(self):
        baseline = {
            "policy_id": "no_abstention",
            "accepted_case_error_rate": 0.10,
        }
        policies = [
            baseline,
            {"policy_id": "x", "accepted_case_error_rate": 0.04, "abstention_rate": 0.25, "coverage": 0.75},
            {"policy_id": "y", "accepted_case_error_rate": 0.05, "abstention_rate": 0.10, "coverage": 0.90},
            {"policy_id": "z", "accepted_case_error_rate": 0.09, "abstention_rate": 0.01, "coverage": 0.99},
        ]
        # Target is 0.05; y meets it with the smallest abstention rate.
        chosen = smallest_abstention_for_error_reduction(policies, baseline)
        self.assertEqual(chosen["policy_id"], "y")

    def test_reference_rule_returns_none_when_target_unmet_or_zero(self):
        baseline = {
            "policy_id": "no_abstention",
            "accepted_case_error_rate": 0.10,
        }
        unmet = [
            baseline,
            {"policy_id": "x", "accepted_case_error_rate": 0.08, "abstention_rate": 0.1},
        ]
        self.assertIsNone(
            smallest_abstention_for_error_reduction(unmet, baseline)
        )
        perfect = {
            "policy_id": "no_abstention",
            "accepted_case_error_rate": 0.0,
        }
        self.assertIsNone(
            smallest_abstention_for_error_reduction([perfect], perfect)
        )


class ReportPrivacyTests(unittest.TestCase):
    SECRET_PROMPT = "ZZUNIQUESECRETSTEGLPROMPTZZ"

    def _synthetic_report(self):
        labels = [0, 1, 0, 1, 1, 0]
        scores = [0.05, 0.95, 0.92, 0.40, 0.99, 0.02]
        predictions = [1 if score >= 0.90 else 0 for score in scores]
        return build_report(labels, scores, predictions)

    def test_report_contains_no_prompt_text_and_privacy_flags(self):
        report = self._synthetic_report()
        serialized = json.dumps(report)
        markdown = render_markdown(report)
        self.assertNotIn(self.SECRET_PROMPT, serialized)
        self.assertNotIn(self.SECRET_PROMPT, markdown)
        # No per-example payloads anywhere: reports must be aggregate-only.
        forbidden_keys = {
            "text", "texts", "prompt", "prompts", "row_scores", "scores",
            "examples", "rows_data", "items",
        }

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertNotIn(key, forbidden_keys)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(report)
        self.assertFalse(report["test_split_loaded"])
        self.assertFalse(report["prompt_text_included"])
        self.assertTrue(report["validation_only"])

    def test_report_declares_no_optimality_and_threshold_unchanged(self):
        report = self._synthetic_report()
        selective = report["selective_prediction"]
        self.assertFalse(selective["optimality_claimed"])
        self.assertFalse(selective["decision_threshold_modified"])
        self.assertEqual(selective["decision_threshold"], 0.90)
        self.assertEqual(selective["baseline_policy_id"], BASELINE_POLICY_ID)
        self.assertIn(
            "optimistically biased", " ".join(selective["selection_notes"])
        )

    def test_written_reports_are_timestamped_and_clean(self):
        report = self._synthetic_report()
        with tempfile.TemporaryDirectory() as directory_name:
            json_path, markdown_path = write_reports(
                report, Path(directory_name)
            )
            self.assertTrue(json_path.name.startswith(
                "deberta-confidence-calibration-"
            ))
            self.assertTrue(markdown_path.name.startswith(
                "deberta-confidence-calibration-"
            ))
            json_text = json_path.read_text(encoding="utf-8")
            markdown_text = markdown_path.read_text(encoding="utf-8")
            self.assertNotIn(self.SECRET_PROMPT, json_text)
            self.assertNotIn(self.SECRET_PROMPT, markdown_text)

    def test_rendered_markdown_includes_core_sections(self):
        markdown = render_markdown(self._synthetic_report())
        for section in (
            "Reliability bins",
            "Score distributions",
            "Selective prediction",
            "Limitations",
            "No candidate is declared optimal",
        ):
            self.assertIn(section, markdown)


class RunStudyTests(unittest.TestCase):
    def test_only_approved_validation_path_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "approved"):
            run_study(
                Path("data/raw/test.parquet"),
                Path("models/deberta-v3-small"),
            )

    def test_mocked_study_produces_aggregate_report(self):
        fake_service = SimpleNamespace(
            threshold=0.90,
            device="cpu",
            _torch=SimpleNamespace(__version__="synthetic"),
            _tokenizer=SimpleNamespace(model_max_length=128),
            _model=SimpleNamespace(config=SimpleNamespace(
                architectures=["DebertaV2ForSequenceClassification"],
                model_type="deberta-v2",
                num_labels=2,
                id2label={"0": "LABEL_0", "1": "LABEL_1"},
                label2id={"LABEL_0": 0, "LABEL_1": 1},
            )),
        )
        fake_service.load = lambda: None
        fake_service.predict_logits_batch = lambda texts, **kwargs: [
            [2.0, 0.0],
            [0.0, 3.0],
            [0.2, 0.1],
        ]
        from promptshield.deberta_decision_audit import APPROVED_VALIDATION_PATH
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint = Path(directory_name)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "promptshield.confidence_calibration.load_labeled_parquet",
                return_value=(["a", "b", "c"], [0, 1, 1]),
            ), patch(
                "promptshield.confidence_calibration.DebertaInference",
                return_value=fake_service,
            ):
                report = run_study(APPROVED_VALIDATION_PATH, checkpoint)
        self.assertEqual(report["validation_rows"], 3)
        self.assertFalse(report["test_split_loaded"])
        self.assertFalse(report["prompt_text_included"])
        # Score for row 3 is sigmoid(0.1 - 0.2) ~= 0.475 -> predicted benign.
        self.assertEqual(
            report["reliability"]["bin_count"], 10
        )
        baseline = report["selective_prediction"]["policies"][0]
        self.assertEqual(baseline["policy_id"], BASELINE_POLICY_ID)
        self.assertEqual(baseline["coverage"], 1.0)

    def test_mismatched_logit_rows_fail_the_study(self):
        fake_service = SimpleNamespace(threshold=0.90)
        fake_service.load = lambda: None
        fake_service.predict_logits_batch = lambda texts, **kwargs: [[1.0, 0.0]]
        from promptshield.deberta_decision_audit import APPROVED_VALIDATION_PATH
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint = Path(directory_name)
            with patch(
                "promptshield.confidence_calibration.load_labeled_parquet",
                return_value=(["a", "b"], [0, 1]),
            ), patch(
                "promptshield.confidence_calibration.DebertaInference",
                return_value=fake_service,
            ):
                with self.assertRaisesRegex(ValueError, "logit rows"):
                    run_study(APPROVED_VALIDATION_PATH, checkpoint)


if __name__ == "__main__":
    unittest.main()
