from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CapabilityCounts:
    preserved: int
    expansion: int
    shrinkage: int
    out_of_support: int
    esr: float | str


@dataclass(frozen=True)
class MetricSummary:
    base_pass_at_k: float
    rl_pass_at_k: float
    counts: CapabilityCounts


@dataclass(frozen=True)
class DualRewardSummary:
    train: MetricSummary
    independent: MetricSummary
    delta_esr: float | str


def _normalise_direction(direction: str) -> str:
    direction = direction.lower()
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    return direction


def _is_success(scores: pd.Series, threshold: float, direction: str) -> pd.Series:
    direction = _normalise_direction(direction)
    if direction == "higher":
        return scores >= threshold
    return scores <= threshold


def _model_successes(
    frame: pd.DataFrame,
    *,
    context_col: str,
    model_col: str,
    score_col: str,
    model_name: str,
    k: int,
    threshold: float,
    direction: str,
    rank_col: str | None,
) -> set[str]:
    subset = frame.loc[frame[model_col] == model_name].copy()
    if subset.empty:
        return set()

    if rank_col:
        subset = subset.sort_values([context_col, rank_col], kind="mergesort")
    else:
        subset = subset.sort_index(kind="mergesort")

    top_k = subset.groupby(context_col, sort=False).head(k)
    success_mask = _is_success(top_k[score_col], threshold, direction)
    return set(top_k.loc[success_mask, context_col].astype(str))


def _contexts(frame: pd.DataFrame, context_col: str) -> set[str]:
    return set(frame[context_col].dropna().astype(str))


def _esr(expansion: int, shrinkage: int) -> float | str:
    if shrinkage == 0:
        if expansion == 0:
            return 0.0
        return "inf"
    return expansion / shrinkage


def compute_metrics(
    frame: pd.DataFrame,
    *,
    context_col: str,
    model_col: str,
    score_col: str,
    base_model: str,
    rl_model: str,
    k: int,
    threshold: float,
    direction: str = "higher",
    rank_col: str | None = None,
) -> MetricSummary:
    if k <= 0:
        raise ValueError("k must be positive")
    required = {context_col, model_col, score_col}
    if rank_col:
        required.add(rank_col)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    all_contexts = _contexts(frame, context_col)
    if not all_contexts:
        raise ValueError("No contexts found")

    base_solved = _model_successes(
        frame,
        context_col=context_col,
        model_col=model_col,
        score_col=score_col,
        model_name=base_model,
        k=k,
        threshold=threshold,
        direction=direction,
        rank_col=rank_col,
    )
    rl_solved = _model_successes(
        frame,
        context_col=context_col,
        model_col=model_col,
        score_col=score_col,
        model_name=rl_model,
        k=k,
        threshold=threshold,
        direction=direction,
        rank_col=rank_col,
    )

    preserved = len(base_solved & rl_solved)
    expansion = len(rl_solved - base_solved)
    shrinkage = len(base_solved - rl_solved)
    out_of_support = len(all_contexts - (base_solved | rl_solved))
    counts = CapabilityCounts(
        preserved=preserved,
        expansion=expansion,
        shrinkage=shrinkage,
        out_of_support=out_of_support,
        esr=_esr(expansion, shrinkage),
    )
    denom = len(all_contexts)
    return MetricSummary(
        base_pass_at_k=len(base_solved) / denom,
        rl_pass_at_k=len(rl_solved) / denom,
        counts=counts,
    )


def _numeric_esr(value: float | str) -> float:
    if value == "inf":
        return math.inf
    return float(value)


def _format_delta(value: float) -> float | str:
    if math.isinf(value):
        return "inf"
    return value


def compute_dual_reward_metrics(
    frame: pd.DataFrame,
    *,
    context_col: str,
    model_col: str,
    train_score_col: str,
    independent_score_col: str,
    base_model: str,
    rl_model: str,
    k: int,
    train_threshold: float,
    independent_threshold: float,
    train_direction: str = "higher",
    independent_direction: str = "higher",
    rank_col: str | None = None,
) -> DualRewardSummary:
    train = compute_metrics(
        frame,
        context_col=context_col,
        model_col=model_col,
        score_col=train_score_col,
        base_model=base_model,
        rl_model=rl_model,
        k=k,
        threshold=train_threshold,
        direction=train_direction,
        rank_col=rank_col,
    )
    independent = compute_metrics(
        frame,
        context_col=context_col,
        model_col=model_col,
        score_col=independent_score_col,
        base_model=base_model,
        rl_model=rl_model,
        k=k,
        threshold=independent_threshold,
        direction=independent_direction,
        rank_col=rank_col,
    )
    delta = _numeric_esr(train.counts.esr) - _numeric_esr(independent.counts.esr)
    return DualRewardSummary(
        train=train,
        independent=independent,
        delta_esr=_format_delta(delta),
    )


def _jsonable(obj):
    if isinstance(obj, (MetricSummary, CapabilityCounts, DualRewardSummary)):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_or_print(summary, output: Path | None) -> None:
    text = json.dumps(summary, default=_jsonable, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute pass@k, ESR, Dual-Reward ESR, and delta-ESR from scored samples."
    )
    parser.add_argument("--samples", type=Path, required=True, help="CSV containing generated samples and scores.")
    parser.add_argument("--context-col", default="context_id", help="Column identifying each design context/problem.")
    parser.add_argument("--model-col", default="model", help="Column identifying base vs RL samples.")
    parser.add_argument("--base-model", default="base", help="Value in --model-col for the base PLM.")
    parser.add_argument("--rl-model", default="rl", help="Value in --model-col for the RL-tuned PLM.")
    parser.add_argument("--rank-col", default=None, help="Optional within-context sample rank column.")
    parser.add_argument("--k", type=int, required=True, help="Number of samples per context for pass@k/ESR.")
    parser.add_argument("--score-col", help="Single reward/evaluator score column.")
    parser.add_argument("--threshold", type=float, help="Success threshold for --score-col.")
    parser.add_argument("--direction", default="higher", choices=["higher", "lower"], help="Whether larger scores are better.")
    parser.add_argument("--train-score-col", help="Training-reward score column for Dual-Reward ESR.")
    parser.add_argument("--independent-score-col", help="Independent-evaluator score column for Dual-Reward ESR.")
    parser.add_argument("--train-threshold", type=float, help="Success threshold for --train-score-col.")
    parser.add_argument("--independent-threshold", type=float, help="Success threshold for --independent-score-col.")
    parser.add_argument("--train-direction", default="higher", choices=["higher", "lower"])
    parser.add_argument("--independent-direction", default="higher", choices=["higher", "lower"])
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.samples)

    dual_args = [
        args.train_score_col,
        args.independent_score_col,
        args.train_threshold,
        args.independent_threshold,
    ]
    if any(value is not None for value in dual_args):
        if not all(value is not None for value in dual_args):
            raise SystemExit(
                "Dual-Reward ESR requires --train-score-col, --independent-score-col, "
                "--train-threshold, and --independent-threshold."
            )
        summary = compute_dual_reward_metrics(
            frame,
            context_col=args.context_col,
            model_col=args.model_col,
            train_score_col=args.train_score_col,
            independent_score_col=args.independent_score_col,
            base_model=args.base_model,
            rl_model=args.rl_model,
            k=args.k,
            train_threshold=args.train_threshold,
            independent_threshold=args.independent_threshold,
            train_direction=args.train_direction,
            independent_direction=args.independent_direction,
            rank_col=args.rank_col,
        )
    else:
        if args.score_col is None or args.threshold is None:
            raise SystemExit("Single-reward ESR requires --score-col and --threshold.")
        summary = compute_metrics(
            frame,
            context_col=args.context_col,
            model_col=args.model_col,
            score_col=args.score_col,
            base_model=args.base_model,
            rl_model=args.rl_model,
            k=args.k,
            threshold=args.threshold,
            direction=args.direction,
            rank_col=args.rank_col,
        )
    _write_or_print(summary, args.output)


if __name__ == "__main__":
    main()
