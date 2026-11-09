"""Tests for the validation-only source reliability and gateway policy audit."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from promptshield.source_reliability import (
    ILLUSTRATIVE_MIN_COVERAGE,
    MAX_STABLE_INTERVAL_WIDTH,
    MIN_STABLE_DENOMINATOR,
    Z_95,
    build_report,
    gateway_policy_note,
    instability_findings,
    metadata_label_consistency,
    rate_summary,
    render_markdown,
    signal_availability_audit,
    stratified_rates,
    wilson_interval,
    write_reports,
)


def _error_analysis_report() -> dict:
    return {
        "validation_rows": 60,
        "label_distribution": {"0": 50, "1": 10},
        "model_configuration": {"threshold": 0.9},
        "counts": {
            "true_negative": 41,
            "false_positive": 9,
            "false_negative": 2,
            "true_positive": 8,
        },
        "metadata_breakdown": {
            "source": {
                "wildguard_judgecomp": {
                    "sample_count": 30,
                    "actual_label_counts": {"0": 29, "1": 1},
                    "counts": {
                        "true_negative": 20,
                        "false_positive": 9,
                        "false_negative": 0,
                        "true_positive": 1,
                    },
                },
                "neuralchemy_v1": {
                    "sample_count": 30,
                    "actual_label_counts": {"0": 21, "1": 9},
                    "counts": {
                        "true_negative": 21,
                        "false_positive": 0,
                        "false_negative": 2,
                        "true_positive": 7,
                    },
                },
            },
            "category": {},
        },
    }


def _investigation_report() -> dict:
    return {
        "metadata_label_consistency": {
            "source": {
                "potential_mixed_label_values": ["wildguard_judgecomp"],
                "label_counts_by_value": {
                    "wildguard_judgecomp": {"0": 29, "1": 1},
                    "neuralchemy_v1": {"0": 21, "1": 9},
                },
            }
        }
    }


def _calibration_report() -> dict:
    return {
        "selective_prediction": {
            "policies": [
                {
                    "policy_id": "no_abstention",
                    "coverage": 1.0,
                    "abstention_rate": 0.0,
                    "accepted_false_positive": 10,
                    "accepted_false_negative": 2,
                    "abstained_benign": 0,
                    "abstained_malicious": 0,
                    "abstained_count": 0,
                    "fraction_of_all_errors_caught": 0.0,
                },
                {
                    "policy_id": "band_wide",
                    "coverage": 0.9,
                    "abstention_rate": 0.1,
                    "accepted_false_positive": 4,
                    "accepted_false_negative": 1,
                    "abstained_benign": 6,
                    "abstained_malicious": 1,
                    "abstained_count": 7,
                    "fraction_of_all_errors_caught": 0.5,
                },
                {
                    "policy_id": "band_narrow",
                    "coverage": 0.97,
                    "abstention_rate": 0.03,
                    "accepted_false_positive": 7,
                    "accepted_false_negative": 2,
                    "abstained_benign": 3,
                    "abstained_malicious": 0,
                    "abstained_count": 3,
                    "fraction_of_all_errors_caught": 0.2,
                },
                {
                    "policy_id": "band_low_coverage",
                    "coverage": 0.8,
                    "abstention_rate": 0.2,
                    "accepted_false_positive": 1,
                    "accepted_false_negative": 0,
                    "abstained_benign": 9,
                    "abstained_malicious": 2,
                    "abstained_count": 11,
                    "fraction_of_all_errors_caught": 0.9,
                },
            ]
        }
    }


def _inputs() -> dict:
    return {
        "error_analysis": "reports/a.json",
        "investigation": "reports/b.json",
        "calibration": "reports/c.json",
    }


class WilsonIntervalTests(unittest.TestCase):
    def test_zero_denominator_returns_none(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))

    def test_hand_computed_value(self):
        lower, upper = wilson_interval(27, 33)
        self.assertAlmostEqual(lower, 0.656117, places=5)
        self.assertAlmostEqual(upper, 0.913890, places=5)

    def test_interval_contains_point_estimate(self):
        lower, upper = wilson_interval(7, 18)
        self.assertLess(lower, 7 / 18)
        self.assertGreater(upper, 7 / 18)

    def test_clamped_to_unit_interval_at_extremes(self):
        lower, upper = wilson_interval(0, 14)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLess(upper, 1.0)
        lower, upper = wilson_interval(14, 14)
        self.assertGreater(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_narrows_as_denominator_grows(self):
        wide = wilson_interval(5, 20)
        narrow = wilson_interval(500, 2000)
        self.assertGreater(
            wide[1] - wide[0], narrow[1] - narrow[0]
        )

    def test_rejects_inconsistent_counts(self):
        with self.assertRaises(ValueError):
            wilson_interval(-1, 10)
        with self.assertRaises(ValueError):
            wilson_interval(5, -1)
        with self.assertRaises(ValueError):
            wilson_interval(11, 10)

    def test_matches_closed_form_at_zero_and_full(self):
        lower, upper = wilson_interval(0, 10, z=Z_95)
        z2 = Z_95 * Z_95
        expected_upper = (z2 / 10) / (1 + z2 / 10)
        self.assertAlmostEqual(upper, expected_upper, places=12)
        self.assertAlmostEqual(lower, 0.0, places=12)


class RateSummaryTests(unittest.TestCase):
    def test_stable_when_large_denominator_and_narrow(self):
        summary = rate_summary(20, 400, "source:big")
        self.assertTrue(summary["stable"])
        self.assertEqual(summary["unstable_reasons"], [])

    def test_unstable_when_small_denominator(self):
        summary = rate_summary(7, 18, "source:small")
        self.assertFalse(summary["stable"])
        self.assertIn(
            f"denominator 18 < {MIN_STABLE_DENOMINATOR}",
            summary["unstable_reasons"],
        )

    def test_unstable_when_interval_wide_despite_size(self):
        # n=30 meets the denominator floor but a 50% rate still yields a
        # Wilson width above the threshold, so it must still be flagged.
        summary = rate_summary(15, 30, "source:mixed")
        self.assertFalse(summary["stable"])
        self.assertTrue(
            any(
                "interval width" in reason
                for reason in summary["unstable_reasons"]
            )
        )
        self.assertGreater(
            summary["wilson_95_width"], MAX_STABLE_INTERVAL_WIDTH
        )

    def test_zero_denominator_is_unstable_and_valueless(self):
        summary = rate_summary(0, 0, "source:absent")
        self.assertIsNone(summary["value"])
        self.assertFalse(summary["stable"])


class StratificationTests(unittest.TestCase):
    def test_stratifies_by_source_with_correct_denominators(self):
        report = build_report(
            _error_analysis_report(),
            _investigation_report(),
            _calibration_report(),
            _inputs(),
        )
        source = report["stratified_rates"]["source"]["wildguard_judgecomp"]
        fpr = source["false_positive_rate"]
        self.assertEqual(fpr["numerator"], 9)
        self.assertEqual(fpr["denominator"], 29)
        fnr = source["false_negative_rate"]
        self.assertEqual(fnr["numerator"], 0)
        self.assertEqual(fnr["denominator"], 1)

    def test_instability_findings_lists_small_groups(self):
        stratified = stratified_rates(_error_analysis_report())
        findings = instability_findings(stratified)
        unstable_groups = [
            item["group"] for item in findings["unstable"]
        ]
        self.assertIn(
            "source:wildguard_judgecomp:false_positive_rate", unstable_groups
        )
        self.assertIn(
            "source:wildguard_judgecomp:false_negative_rate", unstable_groups
        )
        self.assertTrue(findings["rule"].startswith("flagged unstable"))

    def test_instability_findings_skips_empty_denominators(self):
        stratified = stratified_rates(_error_analysis_report())
        findings = instability_findings(stratified)
        for item in findings["unstable"]:
            self.assertGreater(item["denominator"], 0)

    def test_missing_category_breakdown_yields_empty(self):
        report = build_report(
            _error_analysis_report(),
            _investigation_report(),
            _calibration_report(),
            _inputs(),
        )
        self.assertEqual(report["stratified_rates"]["category"], {})


class MetadataConsistencyTests(unittest.TestCase):
    def test_surfaces_mixed_labels_without_relabeling(self):
        consistency = metadata_label_consistency(_investigation_report())
        self.assertEqual(
            consistency["mixed_label_source_values"], ["wildguard_judgecomp"]
        )
        self.assertIn("NOT treated as ground truth", consistency["interpretation"])

    def test_hypotheses_remain_unresolved(self):
        consistency = metadata_label_consistency(_investigation_report())
        joined = " ".join(consistency["hypotheses_not_resolved"])
        self.assertIn("label noise", joined)
        self.assertIn("model bias", joined)
        self.assertIn("not distinguishable", joined)

    def test_missing_consistency_block_is_empty(self):
        consistency = metadata_label_consistency({})
        self.assertEqual(consistency["mixed_label_source_values"], [])


class SignalAvailabilityTests(unittest.TestCase):
    def test_metadata_is_offline_only(self):
        audit = signal_availability_audit()
        for field in ("source", "category", "severity", "augmented"):
            self.assertFalse(
                audit["offline_diagnostic_fields"][field][
                    "available_at_inference"
                ]
            )
            self.assertEqual(
                audit["offline_diagnostic_fields"][field]["classification"],
                "offline diagnostic only",
            )

    def test_prompt_is_available(self):
        audit = signal_availability_audit()
        self.assertTrue(audit["text_available_at_inference"])
        self.assertIn("prompt", audit["inference_available_signals"])

    def test_conclusion_blocks_allowlist_implementation(self):
        audit = signal_availability_audit()
        self.assertIn("no source-based", audit["conclusion"])
        self.assertIn("without changing the request contract", audit["conclusion"])


class GatewayPolicyTests(unittest.TestCase):
    def setUp(self):
        self.note = gateway_policy_note(
            _calibration_report(), _error_analysis_report()
        )

    def test_baseline_reports_deployed_counts(self):
        baseline = self.note["baseline_no_abstention"]
        self.assertEqual(baseline["fail_open"]["false_positives"], 10)
        self.assertEqual(baseline["fail_open"]["false_negatives"], 2)

    def test_fail_open_moves_abstained_malicious_to_false_negatives(self):
        behaviour = next(
            item
            for item in self.note["behaviours"]
            if item["policy_id"] == "band_wide"
        )
        self.assertEqual(behaviour["fail_open"]["false_positives"], 4)
        self.assertEqual(behaviour["fail_open"]["false_negatives"], 2)

    def test_fail_closed_moves_abstained_benign_to_false_positives(self):
        behaviour = next(
            item
            for item in self.note["behaviours"]
            if item["policy_id"] == "band_wide"
        )
        self.assertEqual(behaviour["fail_closed"]["false_positives"], 10)
        self.assertEqual(behaviour["fail_closed"]["false_negatives"], 1)

    def test_perfect_review_costs_capacity_only(self):
        behaviour = next(
            item
            for item in self.note["behaviours"]
            if item["policy_id"] == "band_wide"
        )
        review = behaviour["perfect_human_review"]
        self.assertEqual(review["false_positives"], 4)
        self.assertEqual(review["false_negatives"], 1)
        self.assertEqual(review["rows_sent_to_review"], 7)
        self.assertIn("Optimistic upper bound", review["note"])

    def test_fail_open_and_fail_closed_sum_to_accepted_plus_abstained(self):
        for behaviour in self.note["behaviours"]:
            fp_open = behaviour["fail_open"]["false_positives"]
            fp_closed = behaviour["fail_closed"]["false_positives"]
            fn_open = behaviour["fail_open"]["false_negatives"]
            fn_closed = behaviour["fail_closed"]["false_negatives"]
            self.assertEqual(
                fp_closed - fp_open, behaviour["abstained_benign"]
            )
            self.assertEqual(
                fn_open - fn_closed, behaviour["abstained_malicious"]
            )

    def test_illustrative_band_respects_coverage_floor(self):
        illustrative = self.note["illustrative_band"]
        self.assertFalse(illustrative["optimality_claimed"])
        self.assertEqual(illustrative["policy_id"], "band_narrow")
        policy = next(
            item
            for item in self.note["behaviours"]
            if item["policy_id"] == "band_narrow"
        )
        self.assertGreaterEqual(
            policy["coverage"], ILLUSTRATIVE_MIN_COVERAGE
        )

    def test_illustrative_band_skips_low_coverage_even_if_better(self):
        self.assertNotEqual(
            self.note["illustrative_band"]["policy_id"],
            "band_low_coverage",
        )

    def test_no_eligible_band_returns_none(self):
        calibration = {
            "selective_prediction": {
                "policies": [
                    {
                        "policy_id": "no_abstention",
                        "coverage": 1.0,
                        "abstention_rate": 0.0,
                        "fraction_of_all_errors_caught": 0.0,
                    }
                ]
            }
        }
        note = gateway_policy_note(
            calibration, _error_analysis_report()
        )
        self.assertIsNone(note["illustrative_band"]["policy_id"])
        self.assertIsNone(note["illustrative_band"]["behaviour"])


class ReportAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.report = build_report(
            _error_analysis_report(),
            _investigation_report(),
            _calibration_report(),
            _inputs(),
        )

    def test_reuses_predictions_and_runs_no_inference(self):
        self.assertTrue(self.report["reuses_existing_predictions"])
        self.assertFalse(self.report["new_inference_run"])
        self.assertTrue(self.report["validation_only"])

    def test_test_split_never_loaded(self):
        self.assertFalse(self.report["test_split_loaded"])
        self.assertFalse(self.report["prompt_text_included"])

    def test_threshold_recorded_and_not_modified(self):
        self.assertEqual(self.report["threshold"], 0.9)
        self.assertFalse(
            self.report["gateway_policy_design"]["deployed_decision_rule"][
                "modified"
            ]
        )

    def test_confidence_interval_method_documented(self):
        method = self.report["confidence_interval_method"]
        self.assertEqual(method["interval"], "Wilson score, 95%")
        self.assertEqual(method["z"], Z_95)

    def test_review_ceiling_is_offline_only(self):
        ceiling = self.report["review_escalation_ceiling"]
        self.assertIn("NOT implementable", ceiling["status"])
        for scenario in ceiling["scenarios"]:
            self.assertFalse(scenario["implementable_at_inference"])

    def test_review_ceiling_reports_largest_fp_source_first(self):
        ceiling = self.report["review_escalation_ceiling"]
        sources = [item["source"] for item in ceiling["scenarios"]]
        if len(sources) > 1:
            fp_counts = [
                item["false_positives_moved_to_review"]
                for item in ceiling["scenarios"]
            ]
            self.assertEqual(fp_counts, sorted(fp_counts, reverse=True))

    def test_limitations_present(self):
        self.assertTrue(
            any("validation-only" in item for item in self.report["limitations"])
        )
        self.assertTrue(
            any("ground truth" in item for item in self.report["limitations"])
        )


class PrivacyTests(unittest.TestCase):
    # Keys that would indicate raw prompt text was persisted. The report's own
    # boolean audit flags (prompt_text_included, text_available_at_inference)
    # are metadata about privacy and are intentionally allowed.
    FORBIDDEN_KEYS = {"prompt_text", "raw_prompt", "user_prompt", "prompts"}

    def setUp(self):
        self.report = build_report(
            _error_analysis_report(),
            _investigation_report(),
            _calibration_report(),
            _inputs(),
        )

    def _walk_keys(self, node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from self._walk_keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from self._walk_keys(item)

    def _walk_string_values(self, node):
        if isinstance(node, dict):
            for value in node.values():
                yield from self._walk_string_values(value)
        elif isinstance(node, list):
            for item in node:
                yield from self._walk_string_values(item)
        elif isinstance(node, str):
            yield node

    def test_report_excludes_raw_prompt_keys(self):
        keys = set(self._walk_keys(self.report))
        self.assertFalse(keys & self.FORBIDDEN_KEYS)

    def test_report_boolean_flags_declare_no_prompt_text(self):
        self.assertFalse(self.report["prompt_text_included"])
        self.assertFalse(self.report["test_split_loaded"])

    def test_no_long_prompt_like_string_values(self):
        # No persisted string should approach prompt length; every value here
        # is a short aggregate label or note.
        for value in self._walk_string_values(self.report):
            self.assertLess(len(value), 400)

    def test_rendered_markdown_excludes_raw_prompt_keys(self):
        rendered = render_markdown(self.report).lower()
        for forbidden in ("raw_prompt", "user_prompt"):
            self.assertNotIn(forbidden, rendered)


class RenderAndWriteTests(unittest.TestCase):
    def setUp(self):
        self.report = build_report(
            _error_analysis_report(),
            _investigation_report(),
            _calibration_report(),
            _inputs(),
        )

    def test_markdown_mentions_offline_metadata_conclusion(self):
        rendered = render_markdown(self.report)
        self.assertIn("offline diagnostic only", rendered)
        self.assertIn("no source-based", rendered)

    def test_markdown_flags_unstable_rates(self):
        rendered = render_markdown(self.report)
        self.assertIn("!unstable", rendered)

    def test_markdown_lists_mixed_label_sources(self):
        rendered = render_markdown(self.report)
        self.assertIn("wildguard_judgecomp", rendered)

    def test_write_reports_creates_unique_files(self):
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory)
            json_path, markdown_path = write_reports(
                self.report, reports_dir
            )
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertIn("source-reliability-policy-", json_path.name)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["experiment"],
                "deberta_source_reliability_and_gateway_policy",
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("gateway policy note", markdown)


if __name__ == "__main__":
    unittest.main()
