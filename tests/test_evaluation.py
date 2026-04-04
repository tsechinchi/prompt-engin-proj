from __future__ import annotations

import unittest

from src.evaluation import score_answer, track_usage


class EvaluationTests(unittest.TestCase):
    def test_score_answer_returns_expected_metrics(self) -> None:
        metrics = score_answer(
            reference="The add/drop deadline is September 10.",
            prediction="The add drop deadline is September 10.",
        )

        for key in [
            "bleu",
            "rouge1_f",
            "rougeL_f",
            "token_precision",
            "token_recall",
            "token_f1",
            "exact_match",
        ]:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_score_answer_includes_optional_llm_judge(self) -> None:
        metrics = score_answer(
            reference="Reference text",
            prediction="Prediction text",
            judge_fn=lambda _ref, _pred: 1.4,
        )
        self.assertEqual(metrics["llm_judge"], 1.0)

    def test_track_usage_reports_total(self) -> None:
        usage = track_usage(prompt_tokens=12, completion_tokens=7)
        self.assertEqual(usage["total_tokens"], 19)


if __name__ == "__main__":
    unittest.main()