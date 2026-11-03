"""Task-specific metrics for LongBench-Pro, routed by secondary task.

The benchmark's answers are ordered lists and its prompts require the final
answer to follow an ``[Answer]`` / ``[答案]`` marker.  The normalization here
therefore deliberately matches :mod:`evaluation.scoring`: case-fold, trim, collapse
internal whitespace, and score only the final answer area.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .scoring import (
    F1_Score,
    normalize_answers,
    normalize_prediction,
    strip_thinking,
)


def _lines(values: Sequence[str] | str) -> list[str]:
    if isinstance(values, str):
        return normalize_prediction(values)
    return normalize_answers(list(values))


def substring_exact_match(answers: Sequence[str], prediction: str) -> float:
    """LongBench-Pro SubEM: fraction of gold lines exactly predicted."""
    gold = _lines(answers)
    pred = _lines(prediction)
    if not gold or not pred:
        return 0.0
    return sum(float(answer in pred) for answer in gold) / len(gold)


def accuracy(answers: Sequence[str], prediction: str) -> float:
    """First-line exact accuracy for T3/T11 multiple-choice questions."""
    gold = _lines(answers)
    pred = _lines(prediction)
    return float(bool(gold) and bool(pred) and gold[0] == pred[0])


def pairwise_accuracy(answers: Sequence[str], prediction: str) -> float:
    """Gold pairs in correct order, divided by all predicted-item pairs."""
    gold = _lines(answers)
    pred = _lines(prediction)
    if len(gold) < 2 or len(pred) < 2:
        return 0.0
    ranks = {value: index for index, value in enumerate(pred)}
    correct = 0
    for left_index, left in enumerate(gold):
        for right in gold[left_index + 1 :]:
            if left in ranks and right in ranks and ranks[left] < ranks[right]:
                correct += 1
    total = len(pred) * (len(pred) - 1) // 2
    return correct / total


def ndcg_at_k(answers: Sequence[str], prediction: str) -> float:
    """TREC NDCG@k with gold rank as linear graded relevance."""
    gold = _lines(answers)
    pred = _lines(prediction)
    if not gold or not pred:
        return 0.0
    k = len(gold)
    relevance = {value: k - index for index, value in enumerate(gold)}
    run_scores = {value: len(pred) - index for index, value in enumerate(pred)}
    ranked = sorted(run_scores, key=run_scores.get, reverse=True)[:k]
    dcg = sum(
        relevance.get(value, 0) / math.log2(rank + 2.0)
        for rank, value in enumerate(ranked)
    )
    ideal = sum(
        value / math.log2(rank + 2.0)
        for rank, value in enumerate(sorted(relevance.values(), reverse=True))
    )
    return dcg / ideal if ideal else 0.0


_F1_PREFIXES = ("T5.", "T6.2", "T7.", "T9.")
_SUBEM_PREFIXES = ("T6.1", "T8.", "T10.")


def metric_family(secondary_task: str) -> str:
    task = str(secondary_task).strip()
    if task.startswith("T1."):
        return "ndcg"
    if task.startswith("T2.") or task.startswith("T6.3"):
        return "pairwise_accuracy"
    if task.startswith(("T3.", "T11.")):
        return "accuracy"
    if task.startswith(_F1_PREFIXES):
        return "f1"
    if task.startswith(_SUBEM_PREFIXES):
        return "subem"
    if task.startswith("T4."):
        return "summary"
    raise ValueError(f"no LongBench-Pro metric registered for {secondary_task!r}")


_METRICS: dict[str, Callable[[Sequence[str], str], float]] = {
    "ndcg": ndcg_at_k,
    "pairwise_accuracy": pairwise_accuracy,
    "accuracy": accuracy,
    "f1": F1_Score,
    "subem": substring_exact_match,
}


def score(secondary_task: str, answers: Sequence[str], prediction: str) -> float:
    family = metric_family(secondary_task)
    if family == "summary":
        raise ValueError("T4 summary rows must be scored after generation")
    return float(_METRICS[family](answers, prediction))


def normalize_summary_prediction(prediction: str) -> str:
    """Official T4 normalization applied to the final answer only."""
    return " ".join(strip_thinking(str(prediction or "")).lower().strip().split())


def summary_components(
    embedding_model,
    answers: Sequence[str],
    prediction: str,
    *,
    is_zh: bool,
    jieba_module=None,
    rouge_class=None,
) -> dict[str, float]:
    """Return the two official T4 components and their 50/50 score."""
    normalized_answers = normalize_answers(list(answers))
    normalized_prediction = normalize_summary_prediction(prediction)
    if not normalized_answers or not normalized_prediction:
        return {"semantic_similarity": 0.0, "rouge_l": 0.0, "score": 0.0}

    answer_embeddings = embedding_model.encode(normalized_answers)
    prediction_embeddings = embedding_model.encode([normalized_prediction])
    similarities = embedding_model.similarity(
        answer_embeddings, prediction_embeddings
    )
    semantic_similarity = float(similarities.max().cpu().numpy())

    rouge_answers = normalized_answers
    rouge_prediction = normalized_prediction
    if is_zh:
        if jieba_module is None:
            import jieba as jieba_module

        rouge_answers = [
            " ".join(list(jieba_module.cut(answer, cut_all=False)))
            for answer in rouge_answers
        ]
        rouge_prediction = " ".join(
            list(jieba_module.cut(rouge_prediction, cut_all=False))
        )
    if rouge_class is None:
        from rouge import Rouge as rouge_class

    scores = rouge_class().get_scores(
        [rouge_prediction] * len(rouge_answers), rouge_answers, avg=False
    )
    rouge_l = max(float(value["rouge-l"]["f"]) for value in scores)
    combined = 0.5 * semantic_similarity + 0.5 * rouge_l
    return {
        "semantic_similarity": semantic_similarity,
        "rouge_l": rouge_l,
        "score": combined,
    }
