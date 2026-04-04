"""Evaluation helpers for answer quality."""

from __future__ import annotations

from collections.abc import Callable
import re

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


JudgeFn = Callable[[str, str], float]


def score_answer(
    reference: str,
    prediction: str,
    *,
    judge_fn: JudgeFn | None = None,
) -> dict[str, float]:
    """Compute lexical metrics and optional LLM-judge score.

    Returns BLEU, ROUGE-1/L, and token overlap metrics in ``[0, 1]``.
    If ``judge_fn`` is provided, its score is added as ``llm_judge``.
    """

    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)

    bleu = _bleu_score(ref_tokens, pred_tokens)
    rouge1_f, rougel_f = _rouge_scores(reference, prediction)
    precision, recall, f1 = _token_overlap_scores(ref_tokens, pred_tokens)

    metrics: dict[str, float] = {
        "bleu": bleu,
        "rouge1_f": rouge1_f,
        "rougeL_f": rougel_f,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "exact_match": float(reference.strip() == prediction.strip()),
    }

    if judge_fn is not None:
        raw = judge_fn(reference, prediction)
        metrics["llm_judge"] = _clamp01(raw)

    return metrics


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _bleu_score(reference_tokens: list[str], prediction_tokens: list[str]) -> float:
    if not reference_tokens or not prediction_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothing))


def _rouge_scores(reference: str, prediction: str) -> tuple[float, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return float(scores["rouge1"].fmeasure), float(scores["rougeL"].fmeasure)


def _token_overlap_scores(reference_tokens: list[str], prediction_tokens: list[str]) -> tuple[float, float, float]:
    if not reference_tokens or not prediction_tokens:
        return 0.0, 0.0, 0.0

    ref_set = set(reference_tokens)
    pred_set = set(prediction_tokens)
    overlap = len(ref_set.intersection(pred_set))

    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))

