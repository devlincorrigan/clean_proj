#!/usr/bin/env python3
"""Run a monthly walk-forward backtest for the stacked Stage 2 model.

This script keeps Stage 1 fixed and retrains Stage 2 month by month using
only sportsbook rows strictly earlier than the evaluation month.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from multi_output_threshold_mlp.selection import greedy_same_team_selection
except ImportError:
    from selection import greedy_same_team_selection


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "multi_output_threshold_mlp_predictions.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "walk_forward_stacked_selected_bets.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "walk_forward_stacked_period_summary.csv"
DEFAULT_PAIR_SUMMARY = SCRIPT_DIR / "walk_forward_stacked_pair_type_summary.csv"
PRIMARY_MIN_BET_PRICE = 2.0
PRIMARY_MAX_BET_PRICE = 4.99
PRIMARY_GLOBAL_MIN_SCORE = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monthly walk-forward backtest for the Stage 2 stacked bet-quality "
            "model using only prior rows for each evaluation month."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--pair-summary-output", default=str(DEFAULT_PAIR_SUMMARY))
    parser.add_argument(
        "--bet-source",
        choices=["stacked", "raw"],
        default="stacked",
        help="Use the stacked Stage 2 score or the raw Stage 1 betting signal.",
    )
    parser.add_argument("--players-csv", default="")
    parser.add_argument("--teams-csv", default="")
    parser.add_argument("--player-window", type=int, default=60)
    parser.add_argument("--recent-window", type=int, default=5)
    parser.add_argument("--team-window", type=int, default=40)
    parser.add_argument(
        "--source-splits",
        default="val,test",
        help="Comma-separated splits allowed as historical Stage 2 fit rows.",
    )
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--eval-window",
        choices=["week", "month", "quarter"],
        default="month",
        help="Walk-forward evaluation period size.",
    )
    parser.add_argument(
        "--fit-history-days",
        type=int,
        default=0,
        help="Optional trailing fit-history window in days. 0 uses all earlier rows.",
    )
    parser.add_argument(
        "--fit-min-bet-price",
        type=float,
        default=2.0,
        help="Minimum decimal price to include in Stage 2 fit rows.",
    )
    parser.add_argument(
        "--fit-max-bet-price",
        type=float,
        default=0.0,
        help="Maximum decimal price to include in Stage 2 fit rows. 0 disables.",
    )
    parser.add_argument(
        "--recency-weighting",
        choices=["none", "exponential"],
        default="none",
    )
    parser.add_argument("--recency-half-life-days", type=float, default=30.0)
    parser.add_argument(
        "--regime",
        choices=["global", "odds_band", "longshot"],
        default="global",
    )
    parser.add_argument("--min-regime-rows", type=int, default=5000)
    parser.add_argument("--min-fit-rows", type=int, default=10000)
    parser.add_argument(
        "--model-type",
        choices=["logistic", "hist_gbdt"],
        default="logistic",
    )
    parser.add_argument("--c", type=float, default=0.03)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--ridge-alpha", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument(
        "--stage2-score-col",
        choices=[
            "trust_bet_ev",
            "trust_weighted_raw_ev",
            "trust_score",
            "trust_return_pred",
            "trust_blend_score",
        ],
        default="trust_blend_score",
    )
    parser.add_argument(
        "--global-min-score",
        type=float,
        default=PRIMARY_GLOBAL_MIN_SCORE,
        help="Single score floor applied across all odds bands.",
    )
    parser.add_argument("--min-bet-price", type=float, default=PRIMARY_MIN_BET_PRICE)
    parser.add_argument("--max-bet-price", type=float, default=PRIMARY_MAX_BET_PRICE)
    parser.add_argument("--min-score-band-0-2", type=float, default=0.0)
    parser.add_argument("--min-score-band-2-3", type=float, default=0.20)
    parser.add_argument("--min-score-band-3-5", type=float, default=0.20)
    parser.add_argument("--min-score-band-5-inf", type=float, default=0.0)
    parser.add_argument("--competition-lambda", type=float, default=0.10)
    parser.add_argument("--competition-sigma", type=float, default=0.15)
    parser.add_argument("--competition-coefficient", type=float, default=12.0)
    parser.add_argument("--competition-power", type=float, default=2.0)
    parser.add_argument("--competition-mean-center", type=float, default=0.65)
    parser.add_argument("--competition-max-bets-per-game", type=int, default=0)
    parser.add_argument("--pair-low-cutoff", type=float, default=0.20)
    parser.add_argument("--pair-mid-cutoff", type=float, default=0.60)
    return parser.parse_args()


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def odds_band(price: float) -> str:
    if not np.isfinite(price):
        return "missing"
    if price < 2.0:
        return "[0, 2)"
    if price < 3.0:
        return "[2, 3)"
    if price < 5.0:
        return "[3, 5)"
    return "[5, inf)"


def threshold_for_band(band: str, args: argparse.Namespace) -> float:
    if args.global_min_score is not None:
        return float(args.global_min_score)
    mapping = {
        "[0, 2)": args.min_score_band_0_2,
        "[2, 3)": args.min_score_band_2_3,
        "[3, 5)": args.min_score_band_3_5,
        "[5, inf)": args.min_score_band_5_inf,
    }
    return float(mapping.get(band, 0.0))


def profit_from_rows(df: pd.DataFrame, price_col: str) -> np.ndarray:
    prices = pd.to_numeric(df[price_col], errors="coerce")
    return np.where(
        df["bet_result"] == "win",
        prices - 1.0,
        np.where(df["bet_result"] == "loss", -1.0, np.nan),
    )


def probability_bucket(prob: float, low_cutoff: float, mid_cutoff: float) -> str:
    if prob < low_cutoff:
        return "L"
    if prob < mid_cutoff:
        return "M"
    return "H"


def assign_eval_periods(game_dates: pd.Series, eval_window: str) -> pd.Series:
    normalized = pd.to_datetime(game_dates, errors="coerce")
    if eval_window == "week":
        return normalized.dt.to_period("W").astype(str)
    if eval_window == "month":
        return normalized.dt.to_period("M").astype(str)
    if eval_window == "quarter":
        return normalized.dt.to_period("Q").astype(str)
    raise ValueError(f"Unsupported eval window: {eval_window}")


def build_same_team_pairs(
    df: pd.DataFrame,
    *,
    stage_label: str,
    low_cutoff: float,
    mid_cutoff: float,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    required = [
        "game_id",
        "team_id",
        "eval_period",
        "model_over_prob",
        "analysis_profit",
        "row_uid",
    ]
    working = df.copy()
    for column in required:
        if column not in working.columns:
            working[column] = np.nan
    working = working[
        working["game_id"].notna()
        & working["team_id"].notna()
        & working["model_over_prob"].notna()
        & working["analysis_profit"].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame()

    pair_rows: list[dict] = []
    for (eval_period, game_id, team_id), group in working.groupby(
        ["eval_period", "game_id", "team_id"],
        sort=False,
    ):
        if len(group) < 2:
            continue
        probs = group["model_over_prob"].to_numpy(dtype=np.float64)
        profits = group["analysis_profit"].to_numpy(dtype=np.float64)
        row_uids = group["row_uid"].astype(str).to_numpy()
        for i in range(len(group) - 1):
            for j in range(i + 1, len(group)):
                prob_low = float(min(probs[i], probs[j]))
                prob_high = float(max(probs[i], probs[j]))
                row_id_low, row_id_high = sorted([row_uids[i], row_uids[j]])
                pair_rows.append(
                    {
                        "eval_period": eval_period,
                        "game_id": game_id,
                        "team_id": team_id,
                        "pair_key": f"{eval_period}|{game_id}|{team_id}|{row_id_low}|{row_id_high}",
                        "stage": stage_label,
                        "pair_type": (
                            probability_bucket(prob_low, low_cutoff, mid_cutoff)
                            + probability_bucket(prob_high, low_cutoff, mid_cutoff)
                        ),
                        "min_model_over_prob": prob_low,
                        "max_model_over_prob": prob_high,
                        "pair_profit": float(profits[i] + profits[j]),
                        "pair_roi": float((profits[i] + profits[j]) / 2.0),
                    }
                )
    return pd.DataFrame(pair_rows)


def summarize_pair_types(pair_rows: pd.DataFrame) -> pd.DataFrame:
    pair_types = ["LL", "LM", "LH", "MM", "MH", "HH"]
    if pair_rows.empty:
        return pd.DataFrame(
            {
                "pair_type": pair_types,
                "before_pair_count": 0,
                "after_pair_count": 0,
                "removed_pair_count": 0,
                "removed_share": np.nan,
                "after_pair_roi": np.nan,
                "after_pair_profit": np.nan,
                "removed_pair_roi": np.nan,
                "removed_pair_profit": np.nan,
            }
        )

    before_rows = pair_rows[pair_rows["stage"] == "before"].copy()
    after_rows = pair_rows[pair_rows["stage"] == "after"].copy()
    kept_keys = set(after_rows["pair_key"].astype(str))
    removed_rows = before_rows[~before_rows["pair_key"].astype(str).isin(kept_keys)].copy()

    before = (
        before_rows.groupby("pair_type")
        .agg(before_pair_count=("pair_type", "size"))
    )
    after = (
        after_rows.groupby("pair_type")
        .agg(
            after_pair_count=("pair_type", "size"),
            after_pair_roi=("pair_roi", "mean"),
            after_pair_profit=("pair_profit", "mean"),
        )
    )
    removed = (
        removed_rows.groupby("pair_type")
        .agg(
            removed_pair_count=("pair_type", "size"),
            removed_pair_roi=("pair_roi", "mean"),
            removed_pair_profit=("pair_profit", "mean"),
        )
    )
    summary = pd.DataFrame({"pair_type": pair_types}).set_index("pair_type")
    summary = summary.join(before, how="left").join(after, how="left").join(removed, how="left").fillna(
        {
            "before_pair_count": 0,
            "after_pair_count": 0,
            "removed_pair_count": 0,
        }
    )
    summary["before_pair_count"] = summary["before_pair_count"].astype(int)
    summary["after_pair_count"] = summary["after_pair_count"].astype(int)
    summary["removed_pair_count"] = summary["removed_pair_count"].astype(int)
    summary["removed_share"] = np.where(
        summary["before_pair_count"] > 0,
        summary["removed_pair_count"] / summary["before_pair_count"],
        np.nan,
    )
    return summary.reset_index()


def select_rows(
    rows: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = rows.copy()
    if args.bet_source == "stacked":
        score_col = args.stage2_score_col
        side_col = "trust_bet_side"
        price_col = "trust_bet_price"
        prob_col = "trust_win_prob"
    else:
        score_col = "bet_ev"
        side_col = "bet_side"
        price_col = "bet_price"
        prob_col = "raw_bet_win_prob"
    candidate_mask = (
        working[side_col].fillna("").ne("")
        & working[price_col].notna()
        & working[score_col].notna()
        & working[prob_col].notna()
        & working["bet_resolved"]
    )
    if args.min_bet_price > 0.0:
        candidate_mask &= pd.to_numeric(working[price_col], errors="coerce") >= args.min_bet_price
    if args.max_bet_price > 0.0:
        candidate_mask &= pd.to_numeric(working[price_col], errors="coerce") <= args.max_bet_price
    candidate_prices = pd.to_numeric(working[price_col], errors="coerce")
    candidate_bands = candidate_prices.map(
        lambda value: odds_band(float(value)) if pd.notna(value) else "missing"
    )
    candidate_thresholds = candidate_bands.map(lambda band: threshold_for_band(band, args))
    candidate_mask &= pd.to_numeric(working[score_col], errors="coerce") > candidate_thresholds
    candidates = working.loc[candidate_mask].copy()
    if candidates.empty:
        return candidates, candidates
    candidates["row_uid"] = candidates.index.astype(str)
    candidates["analysis_price"] = pd.to_numeric(candidates[price_col], errors="coerce")
    candidates["analysis_profit"] = profit_from_rows(candidates, price_col)
    candidates["selected_source"] = args.bet_source

    selected = greedy_same_team_selection(
        bet_rows=candidates,
        over_prob_col="model_over_prob",
        bet_side_col=side_col,
        bet_ev_col=score_col,
        lambda_=args.competition_lambda,
        sigma=args.competition_sigma,
        coefficient=args.competition_coefficient,
        power=args.competition_power,
        mean_center=args.competition_mean_center,
        max_bets_per_game=args.competition_max_bets_per_game,
    )
    if not selected.empty:
        if "row_uid" not in selected.columns:
            selected["row_uid"] = selected.index.astype(str)
        selected["analysis_price"] = pd.to_numeric(selected[price_col], errors="coerce")
        selected["analysis_profit"] = profit_from_rows(selected, price_col)
        selected["profit"] = profit_from_rows(selected, price_col)
        selected["odds_band"] = pd.to_numeric(selected[price_col], errors="coerce").map(
            lambda value: odds_band(float(value)) if pd.notna(value) else "missing"
        )
        selected["selected_source"] = args.bet_source
    return candidates, selected


def main() -> None:
    args = parse_args()
    stage2_module = load_module(
        SCRIPT_DIR / "stacked_bet_quality_model.py",
        "stacked_stage2_module",
    )

    players_csv = args.players_csv or str(stage2_module.DEFAULT_PLAYERS)
    teams_csv = args.teams_csv or str(stage2_module.DEFAULT_TEAMS)

    predictions = stage2_module.load_predictions(args.input)
    base = stage2_module.build_stage2_base_frame(
        predictions=predictions,
        players_csv=players_csv,
        teams_csv=teams_csv,
        player_window=args.player_window,
        recent_window=args.recent_window,
        team_window=args.team_window,
    )
    stage2 = stage2_module.add_stage2_features(base, longshot_cutoff=stage2_module.DEFAULT_LONGSHOT_CUTOFF)
    numeric_features, categorical_features = stage2_module.stage2_feature_lists(stage2)

    source_splits = {split.strip() for split in args.source_splits.split(",") if split.strip()}
    eval_rows = stage2[
        (stage2["dataset_split"] == args.eval_split)
        & stage2["candidate_row"]
        & stage2["bet_resolved"]
        & stage2["game_date"].notna()
    ].copy()
    if eval_rows.empty:
        raise ValueError(f"No resolved candidate rows found for eval_split={args.eval_split}")

    eval_rows["eval_period"] = assign_eval_periods(eval_rows["game_date"], args.eval_window)
    periods = sorted(eval_rows["eval_period"].dropna().unique())

    selected_frames = []
    period_summaries = []
    pair_frames = []

    for period in periods:
        period_rows = eval_rows[eval_rows["eval_period"] == period].copy()
        period_start = period_rows["game_date"].min().normalize()
        fit_mask = (
            stage2["dataset_split"].isin(source_splits)
            & stage2["candidate_row"]
            & stage2["bet_resolved"]
            & stage2["game_date"].lt(period_start)
        )
        if args.fit_history_days > 0:
            fit_mask &= stage2["game_date"].ge(
                period_start - pd.Timedelta(days=args.fit_history_days)
            )
        if args.fit_min_bet_price > 0.0:
            fit_mask &= stage2["bet_price"].ge(args.fit_min_bet_price)
        if args.fit_max_bet_price > 0.0:
            fit_mask &= stage2["bet_price"].le(args.fit_max_bet_price)
        fit_df = stage2.loc[fit_mask].copy()
        if args.bet_source == "stacked" and len(fit_df) < args.min_fit_rows:
            period_summaries.append(
                {
                    "eval_period": period,
                    "status": "skipped",
                    "fit_rows": int(len(fit_df)),
                    "candidate_rows": int(len(period_rows)),
                    "selected_bets": 0,
                    "roi": np.nan,
                    "win_rate": np.nan,
                }
            )
            continue

        if args.bet_source == "stacked":
            fitted = stage2_module.fit_stage2_models(
                fit_df,
                args=args,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            scored = stage2_module.score_candidate_rows(
                period_rows,
                regime=args.regime,
                fitted=fitted,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            fit_weight_min = fitted["fit_weight_summary"]["min"]
            fit_weight_mean = fitted["fit_weight_summary"]["mean"]
            fit_weight_max = fitted["fit_weight_summary"]["max"]
        else:
            scored = period_rows.copy()
            fit_weight_min = np.nan
            fit_weight_mean = np.nan
            fit_weight_max = np.nan
        candidates, selected = select_rows(scored, args=args)
        if not candidates.empty:
            candidates["eval_period"] = period
            pair_frames.append(
                build_same_team_pairs(
                    candidates,
                    stage_label="before",
                    low_cutoff=args.pair_low_cutoff,
                    mid_cutoff=args.pair_mid_cutoff,
                )
            )
        if not selected.empty:
            selected["eval_period"] = period
            selected["fit_rows"] = int(len(fit_df))
            selected_frames.append(selected)
            pair_frames.append(
                build_same_team_pairs(
                    selected,
                    stage_label="after",
                    low_cutoff=args.pair_low_cutoff,
                    mid_cutoff=args.pair_mid_cutoff,
                )
            )

        roi = float(np.nanmean(selected["profit"])) if not selected.empty else np.nan
        win_rate = float(selected["bet_win_target"].mean()) if not selected.empty else np.nan
        period_summaries.append(
            {
                "eval_period": period,
                "status": "ok",
                "fit_rows": int(len(fit_df)),
                "candidate_rows": int(len(candidates)),
                "selected_bets": int(len(selected)),
                "roi": roi,
                "win_rate": win_rate,
                "fit_weight_min": fit_weight_min,
                "fit_weight_mean": fit_weight_mean,
                "fit_weight_max": fit_weight_max,
                "bet_source": args.bet_source,
            }
        )
        print(
            f"{period}: fit_rows={len(fit_df)}, candidate_rows={len(candidates)}, "
            f"selected_bets={len(selected)}, roi={roi:.4%}" if selected is not None else ""
        )

    selected_all = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else pd.DataFrame()
    )
    summary_df = pd.DataFrame(period_summaries)
    pair_rows = (
        pd.concat([frame for frame in pair_frames if not frame.empty], ignore_index=True)
        if pair_frames
        else pd.DataFrame()
    )
    pair_summary = summarize_pair_types(pair_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_all.to_csv(output_path, index=False)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    pair_summary_path = Path(args.pair_summary_output)
    pair_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pair_summary.to_csv(pair_summary_path, index=False)

    print(f"Wrote selected walk-forward bets to {output_path}")
    print(f"Wrote period summary to {summary_path}")
    print(f"Wrote pair-type summary to {pair_summary_path}")
    print(
        f"configuration: bet_source={args.bet_source}, "
        f"eval_window={args.eval_window}, "
        f"fit_history_days={args.fit_history_days if args.fit_history_days > 0 else 'all'}, "
        f"score_col={args.stage2_score_col if args.bet_source == 'stacked' else 'bet_ev'}, "
        f"price_range=[{args.min_bet_price}, "
        f"{'inf' if args.max_bet_price <= 0 else args.max_bet_price}], "
        f"global_min_score={args.global_min_score}"
    )
    if not selected_all.empty:
        overall_roi = float(np.nanmean(selected_all["profit"]))
        overall_win_rate = float(selected_all["bet_win_target"].mean())
        print(
            f"overall: bets={len(selected_all)}, win_rate={overall_win_rate:.4%}, roi={overall_roi:.4%}"
        )
        print("overall ROI by odds band:")
        for band in ["[0, 2)", "[2, 3)", "[3, 5)", "[5, inf)", "missing"]:
            band_rows = selected_all[selected_all["odds_band"] == band]
            if band_rows.empty:
                continue
            print(
                f"  {band}: bets={len(band_rows)}, "
                f"win_rate={band_rows['bet_win_target'].mean():.4%}, "
                f"roi={np.nanmean(band_rows['profit']):.4%}"
            )
    if not pair_summary.empty:
        print("same-team pair types (before vs after competition filter):")
        for _, row in pair_summary.iterrows():
            print(
                f"  {row['pair_type']}: "
                f"before={int(row['before_pair_count'])}, "
                f"after={int(row['after_pair_count'])}, "
                f"removed={int(row['removed_pair_count'])}, "
                f"removed_share={row['removed_share']:.4%}" if pd.notna(row["removed_share"]) else
                f"  {row['pair_type']}: before={int(row['before_pair_count'])}, after={int(row['after_pair_count'])}, removed={int(row['removed_pair_count'])}, removed_share=nan"
            )
        print("kept same-team pair ROI by type:")
        for _, row in pair_summary.iterrows():
            if int(row["after_pair_count"]) <= 0:
                continue
            print(
                f"  {row['pair_type']}: pairs={int(row['after_pair_count'])}, "
                f"pair_roi={row['after_pair_roi']:.4%}, "
                f"pair_profit={row['after_pair_profit']:.4f}"
            )
        print("removed same-team pair ROI by type:")
        for _, row in pair_summary.iterrows():
            if int(row["removed_pair_count"]) <= 0:
                continue
            print(
                f"  {row['pair_type']}: pairs={int(row['removed_pair_count'])}, "
                f"pair_roi={row['removed_pair_roi']:.4%}, "
                f"pair_profit={row['removed_pair_profit']:.4f}"
            )


if __name__ == "__main__":
    main()
