"""Shared same-team competition-aware selection helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_competition_penalty_matrix(
    probabilities: np.ndarray,
    sigma: float,
    coefficient: float,
    power: float,
    mean_center: float,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("competition sigma must be positive.")
    if coefficient < 0:
        raise ValueError("competition coefficient must be non-negative.")
    if power <= 0:
        raise ValueError("competition power must be positive.")

    probs = probabilities.astype(np.float64, copy=False)
    valid = (probs > 0.0) & (probs < 1.0)
    shifted_mean = ((probs[:, None] - mean_center) + (probs[None, :] - mean_center)) / 2.0
    distance = probs[:, None] - probs[None, :]
    matrix = coefficient * np.power(shifted_mean, power) * np.exp(
        -(distance * distance) / (2.0 * sigma * sigma)
    )
    matrix *= valid[:, None] & valid[None, :]
    np.fill_diagonal(matrix, 0.0)
    return matrix


def greedy_team_selection(
    team_rows: pd.DataFrame,
    *,
    over_prob_col: str,
    bet_side_col: str,
    bet_ev_col: str,
    lambda_: float,
    sigma: float,
    coefficient: float,
    power: float,
    mean_center: float,
) -> list[dict]:
    selected = []
    if team_rows.empty:
        return selected

    records = team_rows.to_dict("records")
    if lambda_ <= 0.0:
        for record in records:
            base_ev = safe_float(record.get(bet_ev_col))
            if base_ev is None or base_ev <= 0.0:
                continue
            chosen = dict(record)
            chosen["competition_adjusted_ev"] = float(base_ev)
            chosen["competition_penalty"] = 0.0
            selected.append(chosen)
        return selected

    over_records = []
    over_base_evs = []
    over_probs = []
    for record in records:
        base_ev = safe_float(record.get(bet_ev_col))
        if base_ev is None or base_ev <= 0.0:
            continue
        if record.get(bet_side_col, "") != "over":
            chosen = dict(record)
            chosen["competition_adjusted_ev"] = float(base_ev)
            chosen["competition_penalty"] = 0.0
            selected.append(chosen)
            continue
        over_records.append(record)
        over_base_evs.append(float(base_ev))
        over_probs.append(float(safe_float(record.get(over_prob_col)) or 0.0))

    if not over_records:
        return selected

    over_base_evs = np.asarray(over_base_evs, dtype=np.float64)
    over_probs = np.asarray(over_probs, dtype=np.float64)
    penalty_matrix = build_competition_penalty_matrix(
        over_probs,
        sigma=sigma,
        coefficient=coefficient,
        power=power,
        mean_center=mean_center,
    )
    cumulative_penalties = np.zeros(len(over_records), dtype=np.float64)
    remaining = np.ones(len(over_records), dtype=bool)

    while remaining.any():
        adjusted = over_base_evs - lambda_ * cumulative_penalties
        adjusted[~remaining] = -np.inf
        best_index = int(np.argmax(adjusted))
        best_adjusted_ev = float(adjusted[best_index])
        if not math.isfinite(best_adjusted_ev) or best_adjusted_ev <= 0.0:
            break

        chosen = dict(over_records[best_index])
        chosen["competition_adjusted_ev"] = best_adjusted_ev
        chosen["competition_penalty"] = float(cumulative_penalties[best_index])
        selected.append(chosen)

        remaining[best_index] = False
        cumulative_penalties[remaining] += penalty_matrix[best_index, remaining]

    return selected


def greedy_same_team_selection(
    bet_rows: pd.DataFrame,
    *,
    over_prob_col: str,
    bet_side_col: str,
    bet_ev_col: str,
    lambda_: float,
    sigma: float,
    coefficient: float,
    power: float,
    mean_center: float,
    max_bets_per_game: int,
) -> pd.DataFrame:
    if bet_rows.empty:
        empty = bet_rows.head(0).copy()
        empty["competition_adjusted_ev"] = pd.Series(dtype=float)
        empty["competition_penalty"] = pd.Series(dtype=float)
        return empty

    selected_records = []
    for _, game_rows in bet_rows.groupby(["game_id"], sort=False):
        game_selected = []
        team_group_col = (
            game_rows["team_id"].fillna("__missing__").astype(str)
            if "team_id" in game_rows.columns
            else pd.Series(["__missing__"] * len(game_rows), index=game_rows.index)
        )
        for _, team_rows in game_rows.groupby(team_group_col, sort=False):
            game_selected.extend(
                greedy_team_selection(
                    team_rows=team_rows,
                    over_prob_col=over_prob_col,
                    bet_side_col=bet_side_col,
                    bet_ev_col=bet_ev_col,
                    lambda_=lambda_,
                    sigma=sigma,
                    coefficient=coefficient,
                    power=power,
                    mean_center=mean_center,
                )
            )

        if not game_selected:
            continue

        if max_bets_per_game > 0:
            game_selected = sorted(
                game_selected,
                key=lambda row: (
                    safe_float(row.get("competition_adjusted_ev")) or float("-inf"),
                    safe_float(row.get(bet_ev_col)) or float("-inf"),
                ),
                reverse=True,
            )[:max_bets_per_game]

        selected_records.extend(game_selected)

    if selected_records:
        return pd.DataFrame(selected_records)

    empty = bet_rows.head(0).copy()
    empty["competition_adjusted_ev"] = pd.Series(dtype=float)
    empty["competition_penalty"] = pd.Series(dtype=float)
    return empty
