"""
Evaluate retrieval quality for the Multimodal RAG system using the agent end-to-end.

Features:
  - Loads a dataset of queries with expected article_ids.
  - Runs the LangChain agent to retrieve articles.
  - Appends 'retrieved_relevant_article_ids' to each record based on tool outputs.
  - Computes Recall@K, Precision@K, MRR, and HitRate@K and prints a summary.

Usage examples:
  # Text-only dataset, save annotated copy next to the input file
  python -m src.scripts.evaluate_rag --dataset data_test/metrics/text_only.json --mode text --k 5

  # Multimodal dataset (nudges agent to use image search), write in place
  python -m src.scripts.evaluate_rag --dataset data_test/metrics/multimodal.json --mode multimodal --k 5 --inplace
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agent.agent import build_the_batch_agent


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def compute_metrics(expected: Sequence[str], retrieved: Sequence[str], k: int) -> Dict[str, float]:
    """
    Compute basic retrieval metrics at cutoff k.
    - Recall@K: fraction of expected ids found in top-k.
    - Precision@K: fraction of retrieved top-k that are relevant.
    - MRR: reciprocal rank of first relevant in top-k (0 if none).
    - HitRate@K: 1 if any relevant in top-k else 0.
    """
    expected_set = set(expected)
    topk = list(retrieved[:k])

    hits = [aid for aid in topk if aid in expected_set]

    recall = (len(set(hits)) / len(expected_set)) if expected_set else 0.0
    precision = (len(hits) / len(topk)) if topk else 0.0

    mrr = 0.0
    for idx, aid in enumerate(topk, start=1):
        if aid in expected_set:
            mrr = 1.0 / idx
            break

    hit_rate = 1.0 if hits else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "hit_rate": hit_rate,
    }


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


def run_agent_query(agent, query: str, use_image_prompt: bool) -> List[str]:
    """
    Execute the agent for a query and return ordered article_ids from tool outputs.
    """
    session_id = f"eval:{uuid.uuid4()}"

    # For multimodal mode, I lightly nudge the agent to consider image search.
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
    return [a.get("article_id") for a in articles if a.get("article_id")]


def summarize_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "hit_rate": 0.0}

    def avg(key: str) -> float:
        return sum(m[key] for m in metrics) / len(metrics)

    return {
        "recall": avg("recall"),
        "precision": avg("precision"),
        "mrr": avg("mrr"),
        "hit_rate": avg("hit_rate"),
    }


def format_row(values: Tuple[Any, ...], widths: Tuple[int, ...]) -> str:
    return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))


def print_report(
    per_query_metrics: List[Dict[str, float]],
    data: List[Dict[str, Any]],
    k: int,
) -> None:
    widths = (4, 10, 12, 8, 10, 40)
    header = ("#", f"Recall@{k}", f"Precision@{k}", "MRR", f"Hit@{k}", "Query (truncated)")
    divider = "-+-".join("-" * w for w in widths)

    print()
    print(format_row(header, widths))
    print(divider)

    for idx, (entry, metrics) in enumerate(zip(data, per_query_metrics), start=1):
        query_preview = entry["query"][:37] + "..." if len(entry["query"]) > 40 else entry["query"]
        row = (
            idx,
            f"{metrics['recall']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['mrr']:.3f}",
            f"{metrics['hit_rate']:.3f}",
            query_preview,
        )
        print(format_row(row, widths))

    agg = summarize_metrics(per_query_metrics)
    print(divider)
    print(
        format_row(
            (
                "AVG",
                f"{agg['recall']:.3f}",
                f"{agg['precision']:.3f}",
                f"{agg['mrr']:.3f}",
                f"{agg['hit_rate']:.3f}",
                "",
            ),
            widths,
        )
    )
    print()


def evaluate_dataset(dataset_path: Path, mode: str, k: int, inplace: bool, output: Path | None) -> None:
    data = load_dataset(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(data)}")

    use_image_prompt = mode == "multimodal"
    out_path = dataset_path if inplace else (output or dataset_path.with_name(f"{dataset_path.stem}_with_results.json"))

    print(f"[INFO] Building agent (mode={mode}, k={k}, use_image_prompt={use_image_prompt})")
    agent = build_the_batch_agent()

    per_query_metrics: List[Dict[str, float]] = []

    for entry in data:
        query = entry.get("query", "").strip()
        expected = entry.get("expected_relevant_article_ids") or []
        if not query:
            raise ValueError("Each dataset entry must have a 'query' string.")

        retrieved = run_agent_query(agent, query=query, use_image_prompt=use_image_prompt)[:k]
        entry["retrieved_relevant_article_ids"] = retrieved

        per_query_metrics.append(compute_metrics(expected, retrieved, k))

    save_dataset(data, out_path)
    print(f"[INFO] Wrote annotated dataset with retrieval results to: {out_path}")

    print_report(per_query_metrics, data, k=k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and annotate datasets.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to JSON dataset with 'query' and 'expected_relevant_article_ids' fields.",
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
        help="Cutoff for Recall@K, Precision@K, HitRate@K, and ranking depth for MRR.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_dataset(
        dataset_path=args.dataset,
        mode=args.mode,
        k=args.k,
        inplace=args.inplace,
        output=args.output,
    )


if __name__ == "__main__":
    main()
