import unittest

from promptshield.diagnostic_validation import validate_records


def record(record_id="d-1", label=0, status="approved"):
    return {
        "id": record_id,
        "text": "A diagnostic example.",
        "label": label,
        "category": "ordinary_benign",
        "rationale": "The text requests a normal task.",
        "difficulty": "easy",
        "authoring_provenance": "PromptShield-authored",
        "version": "v1",
        "review_status": status,
    }


class DiagnosticValidationTests(unittest.TestCase):
    def test_counts_approved_labels_and_excludes_review_records(self):
        report = validate_records([
            record("d-1", 0),
            record("d-2", 1),
            record("d-3", None, "review"),
        ])

        self.assertTrue(report.valid)
        self.assertEqual(report.valid_rows, 2)
        self.assertEqual(report.excluded_for_review, 1)
        self.assertEqual(report.label_counts, {0: 1, 1: 1})

    def test_rejects_duplicate_ids_and_invalid_approved_labels(self):
        report = validate_records([
            record("d-1", 0),
            record("d-1", 2),
        ])

        self.assertFalse(report.valid)
        self.assertEqual(report.duplicate_ids, 1)
        self.assertTrue(any("approved records must have label 0 or 1" in error
                            for error in report.invalid_rows))

    def test_requires_nonempty_text_and_null_label_for_review(self):
        invalid = record("d-1", 0, "review")
        invalid["text"] = ""
        report = validate_records([invalid])

        self.assertFalse(report.valid)
        self.assertEqual(report.excluded_for_review, 0)
        self.assertIn("text must be a non-empty string", report.invalid_rows[0])
        self.assertIn("records under review must have a null label",
                      report.invalid_rows[0])


if __name__ == "__main__":
    unittest.main()
