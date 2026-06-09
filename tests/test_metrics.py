from __future__ import annotations

import math

import pandas as pd

from analysis.metrics import compute_dual_reward_metrics, compute_metrics


def test_compute_metrics_partitions_contexts() -> None:
    frame = pd.DataFrame(
        [
            {"context_id": "a", "model": "base", "rank": 1, "score": 1.0},
            {"context_id": "a", "model": "rl", "rank": 1, "score": 1.0},
            {"context_id": "b", "model": "base", "rank": 1, "score": 1.0},
            {"context_id": "b", "model": "rl", "rank": 1, "score": 0.0},
            {"context_id": "c", "model": "base", "rank": 1, "score": 0.0},
            {"context_id": "c", "model": "rl", "rank": 1, "score": 1.0},
            {"context_id": "d", "model": "base", "rank": 1, "score": 0.0},
            {"context_id": "d", "model": "rl", "rank": 1, "score": 0.0},
        ]
    )
    summary = compute_metrics(
        frame,
        context_col="context_id",
        model_col="model",
        score_col="score",
        base_model="base",
        rl_model="rl",
        k=1,
        threshold=0.5,
        rank_col="rank",
    )

    assert summary.base_pass_at_k == 0.5
    assert summary.rl_pass_at_k == 0.5
    assert summary.counts.preserved == 1
    assert summary.counts.expansion == 1
    assert summary.counts.shrinkage == 1
    assert summary.counts.out_of_support == 1
    assert summary.counts.esr == 1.0


def test_compute_dual_reward_metrics_delta() -> None:
    frame = pd.DataFrame(
        [
            {"context_id": "a", "model": "base", "rank": 1, "train": 1.0, "indep": 1.0},
            {"context_id": "a", "model": "rl", "rank": 1, "train": 1.0, "indep": 1.0},
            {"context_id": "b", "model": "base", "rank": 1, "train": 1.0, "indep": 1.0},
            {"context_id": "b", "model": "rl", "rank": 1, "train": 1.0, "indep": 0.0},
            {"context_id": "c", "model": "base", "rank": 1, "train": 0.0, "indep": 0.0},
            {"context_id": "c", "model": "rl", "rank": 1, "train": 1.0, "indep": 0.0},
        ]
    )
    summary = compute_dual_reward_metrics(
        frame,
        context_col="context_id",
        model_col="model",
        train_score_col="train",
        independent_score_col="indep",
        base_model="base",
        rl_model="rl",
        k=1,
        train_threshold=0.5,
        independent_threshold=0.5,
        rank_col="rank",
    )

    assert summary.train.counts.esr == "inf"
    assert summary.independent.counts.esr == 0.0
    assert summary.delta_esr == "inf"
