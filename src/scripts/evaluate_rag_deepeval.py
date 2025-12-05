"""
Evaluate the RAG system with DeepEval metrics using the full agent.

Adds:
  - llm_output (agent answer)
  - relevant_snippets (text snippets from retrieved contexts)

Metrics (DeepEval):
  - ContextualRecallMetric
  - ContextualRelevancyMetric
  - AnswerRelevancyMetric
  - FaithfulnessMetric

Flags mirror evaluate_rag.py:
  --dataset (required)
  --mode [text|multimodal] (multimodal nudges image search)
  --k
  --inplace (overwrite dataset)
  --output PATH (optional, ignored if --inplace)
  --eval-model (LLM name for DeepEval metrics (e.g., gpt-4o-mini). If omitted, metric defaults are used (gpt-4.1).)
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from src.agent.agent import build_the_batch_agent


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def extract_articles_from_intermediate_steps(intermediate_steps) -> List[Dict[str, Any]]:
    """
    Pull article results from the agent's tool calls (same logic as Streamlit app).
    """
    collected: List[Dict[str, Any]] = []

    for step in intermediate_steps:
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            continue
        action, result = step

        tool_name = getattr(action, "tool", None)
        if tool_name not in {"the_batch_multimodal_search", "image_search"}:
            continue

        payload: Optional[Any] = result
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None

        if isinstance(payload, dict):
            article_list = payload.get("results") or []
        elif isinstance(payload, list):
            article_list = payload
        else:
            article_list = []

        collected.extend(article_list)

    return collected


def run_agent_query(agent, query: str, use_image_prompt: bool, k: int) -> Tuple[str, List[str]]:
    """
    Execute the agent for a query and return (llm_output, text_snippets).
    """
    session_id = f"eval:{uuid.uuid4()}"

    input_text = (
        f"{query}\n[Evaluator note: include image search if helpful.]"
        if use_image_prompt
        else query
    )

    res = agent.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}},
    )
    intermediate_steps = res.get("intermediate_steps", [])
    articles = extract_articles_from_intermediate_steps(intermediate_steps)

    snippets: List[str] = []
    for art in articles:
        for snip in art.get("text_snippets") or []:
            if snip not in snippets:
                snippets.append(snip)
            if len(snippets) >= k:
                break
        if len(snippets) >= k:
            break

    llm_output = res.get("output", "") or ""
    return llm_output, snippets


def format_row(values: Tuple[Any, ...], widths: Tuple[int, ...]) -> str:
    return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))


def summarize(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_dataset(
    dataset_path: Path,
    mode: str,
    k: int,
    inplace: bool,
    output: Path | None,
    eval_model: Optional[str],
) -> None:
    data = load_dataset(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(data)}")

    use_image_prompt = mode == "multimodal"
    out_path = dataset_path if inplace else (output or dataset_path.with_name(f"{dataset_path.stem}_with_results.json"))

    print(f"[INFO] Building agent (mode={mode}, k={k}, use_image_prompt={use_image_prompt})")
    agent = build_the_batch_agent()

    metrics = [
        ContextualRecallMetric(model=eval_model) if eval_model else ContextualRecallMetric(),
        ContextualRelevancyMetric(model=eval_model) if eval_model else ContextualRelevancyMetric(),
        AnswerRelevancyMetric(model=eval_model) if eval_model else AnswerRelevancyMetric(),
        FaithfulnessMetric(model=eval_model) if eval_model else FaithfulnessMetric(),
    ]
    metric_names = [getattr(m, "name", getattr(m, "metric_name", m.__class__.__name__)) for m in metrics]
    metric_scores: Dict[str, List[float]] = {name: [] for name in metric_names}

    for entry in data:
        query = entry.get("query", "").strip()
        expected_answer = entry.get("expected_answer", "").strip()
        if not query:
            raise ValueError("Each dataset entry must have a 'query' string.")
        if not expected_answer:
            raise ValueError("Each dataset entry must have an 'expected_answer' string.")

        llm_output, snippets = run_agent_query(agent, query=query, use_image_prompt=use_image_prompt, k=k)
        entry["llm_output"] = llm_output
        entry["relevant_snippets"] = snippets

        test_case = LLMTestCase(
            input=query,
            actual_output=llm_output,
            expected_output=expected_answer,
            retrieval_context=snippets,
        )

        for m, name in zip(metrics, metric_names):
            m.measure(test_case)
            metric_scores[name].append(m.score)

    save_dataset(data, out_path)
    print(f"[INFO] Wrote annotated dataset with agent outputs to: {out_path}")

    widths = (22, 8)
    header = ("Metric", "Avg")
    divider = "---".join("-" * w for w in widths)
    print()
    print(format_row(header, widths))
    print(divider)
    for name in metric_names:
        avg = summarize(metric_scores[name])
        print(format_row((name, f"{avg:.3f}"), widths))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + generation with DeepEval metrics.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to JSON dataset with 'query' and 'expected_answer' fields.",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "multimodal"],
        default="text",
        help="Retrieval mode. 'multimodal' nudges the agent to include image search.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Max snippets to keep per query (also used as cutoff for snippet collection).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="If set, overwrite the input dataset; otherwise write to *_with_results.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output path. Ignored if --inplace is provided.",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        help="LLM name for DeepEval metrics (e.g., gpt-4o-mini). If omitted, metric defaults are used. (gpt-4.1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_dataset(
        dataset_path=args.dataset,
        mode=args.mode,
        k=args.k,
        inplace=args.inplace,
        output=args.output,
        eval_model=args.eval_model,
    )


if __name__ == "__main__":
    main()
