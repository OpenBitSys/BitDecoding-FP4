"""LongBench-Pro benchmark with its task-specific metrics.

Loads all secondary tasks, sends each prompt as a single-user-message chat,
and routes scoring on ``secondary_task``. T4 generations are deliberately
left unscored until the generation engines have exited.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Callable, List

from . import config
from .longbench_pro_metrics import metric_family, score, summary_components
from .scoring import strip_thinking

_MAX_TRUNCATION_PASSES = 16


def load_examples(data_dir: str = None) -> List[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(data_dir or config.LONGBENCH_PRO_DIR)
    return [dict(ex) for ex in ds]


def build_prompt(example: dict, *, thinking: bool = False) -> str:
    """The context and the question, in the phrasing the mode calls for.

    The dataset ships the question twice. ``question_nonthinking`` asks for the
    answer directly; ``question_thinking`` is the one written for a model that
    reasons first. Reports 1 and 4 used the former because they ran the model
    non-thinking, so the field is chosen rather than fixed.
    """
    key = "question_thinking" if thinking else "question_nonthinking"
    return f"{example['context']}\n\n\n\n{example[key]}"


def _middle_truncate(tokens: List[int], budget: int) -> List[int]:
    if len(tokens) <= budget:
        return tokens
    head = (budget + 1) // 2
    tail = budget - head
    return tokens[:head] if tail == 0 else tokens[:head] + tokens[-tail:]


def _fit_budget(prompt, *, tokenize, detokenize, budget):
    tokens = tokenize(prompt)
    if len(tokens) <= budget:
        return prompt, tokens, False
    target = budget
    for _ in range(_MAX_TRUNCATION_PASSES):
        prompt = detokenize(_middle_truncate(tokens, target))
        tokens = tokenize(prompt)
        if len(tokens) <= budget:
            return prompt, tokens, True
        target = max(1, target - (len(tokens) - budget))
    raise RuntimeError(
        f"prompt still exceeds {budget} tokens after {_MAX_TRUNCATION_PASSES} passes"
    )


def build(examples, *, tokenize: Callable, detokenize: Callable,
          prompt_budget=None, thinking: bool = False,
          reject_truncation: bool = False):
    """Return (conversations, prepared) for the vLLM chat runner."""
    budget = prompt_budget or config.LONGBENCH_PROMPT_BUDGET
    conversations, prepared = [], []
    for ex in examples:
        p, toks, was_truncated = _fit_budget(
            build_prompt(ex, thinking=thinking),
            tokenize=tokenize, detokenize=detokenize, budget=budget,
        )
        if reject_truncation and was_truncated:
            raise ValueError(
                f"LongBench-Pro row {ex.get('id')!r} exceeds the "
                f"{budget}-token prompt budget; truncation is forbidden"
            )
        conversations.append([{"role": "user", "content": p}])
        prepared.append({"example": ex, "prompt_tokens": len(toks),
                         "prompt_was_truncated": was_truncated})
    return conversations, prepared


def grade(prepared, completions) -> List[dict]:
    records = []
    for item, comp in zip(prepared, completions, strict=True):
        ex = item["example"]
        raw_completion = str(comp.get("text") or "")
        pred = strip_thinking(raw_completion)
        secondary = str(ex.get("secondary_task") or "")
        family = metric_family(secondary)
        value = None if family == "summary" else score(
            secondary, ex["answer"], pred
        )
        records.append({
            "id": ex.get("id"),
            "primary_task": ex.get("primary_task"),
            "secondary_task": secondary,
            "metric_family": family,
            "language": ex.get("language"),
            "token_length": ex.get("token_length"),
            "contextual_requirement": ex.get("contextual_requirement"),
            "difficulty": ex.get("difficulty"),
            "answers": ex.get("answer"),
            "prediction": pred,
            "raw_completion": raw_completion,
            "score": value,
            "f1": value if family == "f1" else None,
            "prompt_tokens": comp.get("prompt_tokens", item["prompt_tokens"]),
            "prompt_was_truncated": item["prompt_was_truncated"],
            "gen_tokens": comp.get("gen_tokens"),
            "finish_reason": comp.get("finish_reason"),
        })
    return records


METRIC_KEY = "score"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_hashes(path: str) -> dict[str, str]:
    root = Path(path)
    if not root.is_dir():
        return {}
    return {
        str(candidate.relative_to(root)): _sha256(str(candidate))
        for candidate in sorted(root.rglob("*"))
        if candidate.is_file()
    }


def _package_versions(names) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


class T4SummaryScorer:
    """Lazy official-compatible LongBench-Pro T4 scorer."""

    def __init__(self, embedding_model_path: str):
        from sentence_transformers import SentenceTransformer

        self.embedding_model_path = embedding_model_path
        self.embedding_model = SentenceTransformer(
            embedding_model_path,
            tokenizer_kwargs={"padding_side": "left"},
        )

    def components(self, answers, prediction, *, is_zh: bool):
        return summary_components(
            self.embedding_model, answers, prediction, is_zh=is_zh
        )


def _load_official_utils(path: str):
    spec = importlib.util.spec_from_file_location(
        "longbench_pro_official_utils", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official evaluator utilities at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_official_summary_parity(
    scorer: T4SummaryScorer, official_utils_path: str
) -> dict:
    """Compare local and pinned official components on bilingual fixtures."""
    official = _load_official_utils(official_utils_path)
    fixtures = {
        "english": {
            "answers": [
                "A red fox crosses the river at dawn.",
                "At dawn, a red fox crosses the river.",
                "The river is crossed by a red fox at dawn.",
            ],
            "prediction": "A red fox crosses the river at dawn.",
            "is_zh": False,
        },
        "chinese": {
            "answers": [
                "清晨，一只红狐穿过河流。",
                "一只红色狐狸在清晨过河。",
                "红狐于清晨穿过了河流。",
            ],
            "prediction": "清晨，一只红狐穿过河流。",
            "is_zh": True,
        },
    }
    results = {}
    for name, fixture in fixtures.items():
        answers = official.normalize_answers(fixture["answers"])
        prediction = official.normalize_prediction_abstract(
            fixture["prediction"]
        )
        official_semantic = float(
            official.Summary_Max_Semantic_Similarity(
                scorer.embedding_model, answers, prediction
            )
        )
        official_rouge = float(
            official.Summary_Max_Rouge_L(
                answers, prediction, fixture["is_zh"]
            )
        )
        official_score = float(
            official.Summary(
                scorer.embedding_model,
                fixture["answers"],
                fixture["prediction"],
                fixture["is_zh"],
            )
        )
        local = scorer.components(
            fixture["answers"],
            fixture["prediction"],
            is_zh=fixture["is_zh"],
        )
        official_values = {
            "semantic_similarity": official_semantic,
            "rouge_l": official_rouge,
            "score": official_score,
        }
        differences = {
            key: abs(local[key] - official_values[key]) for key in local
        }
        if any(value > 1e-6 for value in differences.values()):
            raise RuntimeError(
                f"official T4 parity failed for {name}: {differences}"
            )
        results[name] = {
            "local": local,
            "official": official_values,
            "absolute_difference": differences,
        }
    return results


def score_t4_run(
    run_dir: str,
    *,
    embedding_model_path: str,
    official_utils_path: str,
    evaluator_commit: str,
    embedding_revision: str,
) -> dict:
    """Post-score immutable sharded generations into sibling scored files."""
    source_paths = sorted(
        path
        for path in glob.glob(
            os.path.join(run_dir, "shard_*", "longbench_pro__*.jsonl")
        )
        if not path.endswith(".scored.jsonl")
    )
    if not source_paths:
        raise ValueError(f"no LongBench-Pro generation JSONL files under {run_dir}")

    source_hashes = {path: _sha256(path) for path in source_paths}
    scorer = T4SummaryScorer(embedding_model_path)
    fixture_results = verify_official_summary_parity(
        scorer, official_utils_path
    )
    total_rows = 0
    t4_rows = 0
    output_paths = []
    for source_path in source_paths:
        output_path = source_path.removesuffix(".jsonl") + ".scored.jsonl"
        temporary = f"{output_path}.tmp.{os.getpid()}"
        with open(source_path, encoding="utf-8") as source, open(
            temporary, "w", encoding="utf-8"
        ) as destination:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_rows += 1
                if str(row.get("secondary_task", "")).startswith("T4."):
                    answers = row.get("answers")
                    if not isinstance(answers, list):
                        raise ValueError(
                            f"T4 row {row.get('id')!r} has no saved references"
                        )
                    values = scorer.components(
                        answers,
                        row.get("prediction", ""),
                        is_zh=row.get("language") == "Chinese",
                    )
                    row["semantic_similarity"] = values["semantic_similarity"]
                    row["rouge_l"] = values["rouge_l"]
                    row["score"] = values["score"]
                    t4_rows += 1
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output_path)
        output_paths.append(output_path)

    unchanged = {path: _sha256(path) for path in source_paths}
    if unchanged != source_hashes:
        raise RuntimeError("a generation JSONL changed during T4 scoring")

    parameter = next(scorer.embedding_model.parameters(), None)
    manifest = {
        "schema": 1,
        "run_dir": os.path.realpath(run_dir),
        "source_files": source_hashes,
        "scored_files": {path: _sha256(path) for path in output_paths},
        "rows": total_rows,
        "t4_rows": t4_rows,
        "official_evaluator": {
            "commit": evaluator_commit,
            "utils_path": os.path.realpath(official_utils_path),
            "utils_sha256": _sha256(official_utils_path),
        },
        "embedding_model": {
            "path": os.path.realpath(embedding_model_path),
            "revision": embedding_revision,
            "files": _model_hashes(embedding_model_path),
            "device": str(getattr(scorer.embedding_model, "device", "unknown")),
            "dtype": str(getattr(parameter, "dtype", "unknown")),
        },
        "packages": _package_versions(
            (
                "torch",
                "transformers",
                "sentence-transformers",
                "torchao",
                "jieba",
                "rouge",
            )
        ),
        "fixture_parity": fixture_results,
    }
    manifest_path = os.path.join(run_dir, "t4_scoring_manifest.json")
    temporary = f"{manifest_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, manifest_path)
    return manifest
