"""Artifact training and inference for the local quantile model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import RAW_PLAYER_SEQUENCE_COLS, TEAM_INFERENCE_COLS
from .model import DEFAULT_QUANTILES, PinballLoss, PlayerPropTransformer

TARGET_COLUMN = "PTS"
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 40
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EARLY_STOPPING_PATIENCE = 6
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_SEQUENCE_LENGTH = 20

ID_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "GAME_DATE",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
)

CONTEXT_COLUMNS: tuple[str, ...] = ("is_playoff", "home", "days_of_rest")


@dataclass
class QuantileModelArtifacts:
    """Saved artifact bundle for local quantile-model inference."""

    model: torch.nn.Module
    feature_columns: list[str]
    quantiles: tuple[float, ...]
    train_end_date: pd.Timestamp
    fit_rows: int
    validation_rows: int
    feature_mean: pd.Series
    feature_std: pd.Series
    epochs_trained: int
    train_loss: float
    val_loss: float
    model_type: str = "transformer"
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH


@dataclass
class QuantileTestSetEvaluation:
    """Held-out evaluation bundle for a split-based quantile artifact."""

    summary: dict[str, float]
    quantile_metrics: pd.DataFrame
    predictions: pd.DataFrame


def feature_columns_from_frame(df: pd.DataFrame) -> list[str]:
    preferred = [*CONTEXT_COLUMNS, *RAW_PLAYER_SEQUENCE_COLS, *TEAM_INFERENCE_COLS]
    return [column for column in preferred if column in df.columns]


def _to_timestamp(split_date: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.to_datetime(split_date)
    if pd.isna(parsed):
        raise ValueError("Invalid split date.")
    return pd.Timestamp(parsed)


def _to_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(values.astype(np.float32), dtype=torch.float32)


def _normalize_feature_frame(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_mean: pd.Series,
    feature_std: pd.Series,
) -> pd.DataFrame:
    normalized = df.copy()
    normalized[feature_cols] = normalized[feature_cols].apply(pd.to_numeric, errors="coerce")
    normalized[feature_cols] = (normalized[feature_cols] - feature_mean) / feature_std
    numeric = normalized[feature_cols].astype(float)
    numeric = numeric.mask(~np.isfinite(numeric), np.nan).fillna(0.0)
    normalized[feature_cols] = numeric
    return normalized


def _build_training_tensors(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    sequence_length: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    ordered = df.copy()
    ordered["GAME_DATE"] = pd.to_datetime(ordered["GAME_DATE"])
    ordered = ordered.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    features = np.nan_to_num(
        ordered[feature_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    targets = np.nan_to_num(
        ordered[[TARGET_COLUMN]].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    player_ids = ordered["PLAYER_ID"].to_numpy()

    n_rows = len(ordered)
    feat_dim = len(feature_cols)
    sequence = np.zeros((n_rows, sequence_length, feat_dim), dtype=np.float32)
    padding_mask = np.ones((n_rows, sequence_length), dtype=bool)

    start_idx = 0
    while start_idx < n_rows:
        end_idx = start_idx
        current_player = player_ids[start_idx]
        while end_idx < n_rows and player_ids[end_idx] == current_player:
            end_idx += 1

        for local_idx in range(end_idx - start_idx):
            global_idx = start_idx + local_idx
            history_start = max(0, local_idx - sequence_length)
            history = features[start_idx + history_start : start_idx + local_idx]
            if len(history) == 0:
                # Match the original implementation: keep one unmasked zero token
                # so attention never sees an all-masked sequence.
                sequence[global_idx, -1, :] = 0.0
                padding_mask[global_idx, -1] = False
                continue

            history_len = history.shape[0]
            sequence[global_idx, sequence_length - history_len :, :] = history
            padding_mask[global_idx, sequence_length - history_len :] = False

        start_idx = end_idx

    return ordered, sequence, padding_mask, targets


def _fit_artifact_from_indices(
    ordered_df: pd.DataFrame,
    sequence: np.ndarray,
    padding_mask: np.ndarray,
    targets: np.ndarray,
    *,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
    quantiles: Sequence[float],
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    sequence_length: int,
    feature_mean: pd.Series,
    feature_std: pd.Series,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> QuantileModelArtifacts:
    train_ds = TensorDataset(
        _to_tensor(sequence[fit_idx]),
        torch.tensor(padding_mask[fit_idx], dtype=torch.bool),
        _to_tensor(targets[fit_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_seq_tensor = _to_tensor(sequence[val_idx])
    val_mask_tensor = torch.tensor(padding_mask[val_idx], dtype=torch.bool)
    val_y_tensor = _to_tensor(targets[val_idx])

    model = PlayerPropTransformer(
        input_size=len(feature_cols),
        quantiles=tuple(float(q) for q in quantiles),
        max_len=max(sequence_length, 32),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = PinballLoss(quantiles=quantiles)

    best_state_dict = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss_total = 0.0
        epoch_batch_count = 0
        for batch_seq, batch_mask, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_seq, padding_mask=batch_mask), batch_y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss encountered.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss_total += float(loss.item())
            epoch_batch_count += 1

        model.eval()
        with torch.no_grad():
            current_val_loss = float(
                criterion(model(val_seq_tensor, padding_mask=val_mask_tensor), val_y_tensor).item()
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "train_loss": (
                        epoch_loss_total / epoch_batch_count if epoch_batch_count > 0 else float("nan")
                    ),
                    "val_loss": current_val_loss,
                    "best_val_loss": best_val_loss,
                }
            )
        if current_val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = current_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    with torch.no_grad():
        fit_seq_tensor = _to_tensor(sequence[fit_idx])
        fit_mask_tensor = torch.tensor(padding_mask[fit_idx], dtype=torch.bool)
        fit_y_tensor = _to_tensor(targets[fit_idx])
        train_loss = float(
            criterion(model(fit_seq_tensor, padding_mask=fit_mask_tensor), fit_y_tensor).item()
        )
        val_loss = float(
            criterion(model(val_seq_tensor, padding_mask=val_mask_tensor), val_y_tensor).item()
        )

    return QuantileModelArtifacts(
        model=model,
        feature_columns=feature_cols,
        quantiles=tuple(float(q) for q in quantiles),
        train_end_date=pd.to_datetime(ordered_df.loc[fit_idx, "GAME_DATE"].max()),
        fit_rows=int(len(fit_idx)),
        validation_rows=int(len(val_idx)),
        feature_mean=feature_mean,
        feature_std=feature_std,
        epochs_trained=best_epoch if best_epoch > 0 else max_epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        model_type="transformer",
        sequence_length=sequence_length,
    )


def train_split_artifact(
    df: pd.DataFrame,
    *,
    split_date: str | date | datetime | pd.Timestamp,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    val_fraction: float = DEFAULT_VALIDATION_FRACTION,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    random_seed: int = 42,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> QuantileModelArtifacts:
    """Train with a holdout split date for evaluation-oriented artifacts."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in training data.")
    split_ts = _to_timestamp(split_date)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    feature_cols = feature_columns_from_frame(df)
    working = df.copy()
    working["GAME_DATE"] = pd.to_datetime(working["GAME_DATE"])
    train_mask = working["GAME_DATE"] <= split_ts
    if not train_mask.any():
        raise ValueError("No training rows on or before split_date.")
    train_features = working.loc[train_mask, feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_mean = train_features.mean(axis=0).fillna(0.0)
    feature_std = (
        train_features.std(axis=0).replace(0.0, 1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    )

    normalized = _normalize_feature_frame(
        working,
        feature_cols=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    ordered_df, sequence, padding_mask, targets = _build_training_tensors(
        normalized,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
    )
    train_idx = ordered_df[ordered_df["GAME_DATE"] <= split_ts].index.to_numpy()
    if len(train_idx) < 2:
        raise ValueError("Not enough training rows for train/validation split.")
    val_size = max(1, int(len(train_idx) * val_fraction))
    fit_idx = train_idx[:-val_size]
    val_idx = train_idx[-val_size:]
    if len(fit_idx) < 1:
        raise ValueError("Not enough fitting rows after validation split.")

    return _fit_artifact_from_indices(
        ordered_df,
        sequence,
        padding_mask,
        targets,
        fit_idx=fit_idx,
        val_idx=val_idx,
        feature_cols=feature_cols,
        quantiles=quantiles,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        sequence_length=sequence_length,
        feature_mean=feature_mean,
        feature_std=feature_std,
        progress_callback=progress_callback,
    )


def train_production_artifact(
    df: pd.DataFrame,
    *,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    val_fraction: float = DEFAULT_VALIDATION_FRACTION,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    random_seed: int = 42,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> QuantileModelArtifacts:
    """Train a streamlit-ready artifact on all local historical data."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in training data.")
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5.")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    feature_cols = feature_columns_from_frame(df)
    working = df.copy()
    working["GAME_DATE"] = pd.to_datetime(working["GAME_DATE"])
    feature_mean = working[feature_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0).fillna(0.0)
    feature_std = (
        working[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .std(axis=0)
        .replace(0.0, 1.0)
        .replace([np.inf, -np.inf], 1.0)
        .fillna(1.0)
    )

    normalized = _normalize_feature_frame(
        working,
        feature_cols=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    ordered_df, sequence, padding_mask, targets = _build_training_tensors(
        normalized,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
    )
    ordered_idx = ordered_df.sort_values("GAME_DATE").index.to_numpy()
    if len(ordered_idx) < 2:
        raise ValueError("Not enough rows to train a production artifact.")

    val_size = max(1, int(len(ordered_idx) * val_fraction))
    fit_idx = ordered_idx[:-val_size]
    val_idx = ordered_idx[-val_size:]
    if len(fit_idx) < 1:
        raise ValueError("Not enough fitting rows after validation split.")

    artifact = _fit_artifact_from_indices(
        ordered_df,
        sequence,
        padding_mask,
        targets,
        fit_idx=fit_idx,
        val_idx=val_idx,
        feature_cols=feature_cols,
        quantiles=quantiles,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        sequence_length=sequence_length,
        feature_mean=feature_mean,
        feature_std=feature_std,
        progress_callback=progress_callback,
    )
    artifact.train_end_date = pd.to_datetime(ordered_df["GAME_DATE"].max())
    return artifact


def save_artifact(artifact: QuantileModelArtifacts, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_artifact(artifact_path: str | Path) -> QuantileModelArtifacts:
    artifact = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
    if not isinstance(artifact, QuantileModelArtifacts):
        raise TypeError(f"Unexpected artifact type: {type(artifact)!r}")
    artifact.model.eval()
    return artifact


def _normalized_features(features: pd.DataFrame, artifacts: QuantileModelArtifacts) -> pd.DataFrame:
    aligned = features.copy()
    for column in artifacts.feature_columns:
        if column not in aligned.columns:
            aligned[column] = float(artifacts.feature_mean.get(column, 0.0))
    aligned = aligned[artifacts.feature_columns].copy()
    normalized = (aligned - artifacts.feature_mean) / artifacts.feature_std
    normalized = normalized.astype(float)
    return normalized.mask(~np.isfinite(normalized), np.nan).fillna(0.0)


def _pinball_loss_numpy(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def evaluate_test_set(
    df: pd.DataFrame,
    artifacts: QuantileModelArtifacts,
) -> QuantileTestSetEvaluation:
    """Evaluate a split-based artifact on rows strictly after train_end_date."""
    feature_cols = artifacts.feature_columns
    sequence_length = int(getattr(artifacts, "sequence_length", DEFAULT_SEQUENCE_LENGTH))

    working = df.copy()
    working["GAME_DATE"] = pd.to_datetime(working["GAME_DATE"], errors="coerce")
    normalized = _normalize_feature_frame(
        working,
        feature_cols=feature_cols,
        feature_mean=artifacts.feature_mean,
        feature_std=artifacts.feature_std,
    )
    ordered_df, sequence, padding_mask, targets = _build_training_tensors(
        normalized,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
    )
    test_idx = ordered_df[ordered_df["GAME_DATE"] > pd.Timestamp(artifacts.train_end_date)].index.to_numpy()
    if len(test_idx) == 0:
        raise ValueError("No held-out test rows found after artifact train_end_date.")

    seq_tensor = _to_tensor(sequence[test_idx])
    mask_tensor = torch.tensor(padding_mask[test_idx], dtype=torch.bool)
    with torch.no_grad():
        preds = artifacts.model(seq_tensor, padding_mask=mask_tensor).numpy()

    test_df = ordered_df.loc[test_idx].reset_index(drop=True)
    y_true = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_var = float(np.var(y_true))

    quantile_rows: list[dict[str, float]] = []
    for idx, q in enumerate(artifacts.quantiles):
        y_q = preds[:, idx]
        quantile_rows.append(
            {
                "quantile": float(q),
                "pinball_loss": _pinball_loss_numpy(y_true, y_q, q),
                "empirical_coverage": float(np.mean(y_true <= y_q)),
            }
        )
    quantile_metrics = pd.DataFrame(quantile_rows)
    quantile_metrics["nominal_quantile"] = quantile_metrics["quantile"]
    quantile_metrics["calibration_gap"] = (
        quantile_metrics["empirical_coverage"] - quantile_metrics["nominal_quantile"]
    )

    median_idx = int(np.argmin(np.abs(np.array(artifacts.quantiles) - 0.5)))
    y_median = preds[:, median_idx]
    mae = float(np.mean(np.abs(y_true - y_median)))
    rmse = float(np.sqrt(np.mean((y_true - y_median) ** 2)))
    r2 = float(1.0 - (np.mean((y_true - y_median) ** 2) / y_var)) if y_var > 0 else float("nan")

    q10 = np.full_like(y_true, np.nan, dtype=float)
    q90 = np.full_like(y_true, np.nan, dtype=float)
    interval_width = float("nan")
    interval_coverage = float("nan")
    if 0.10 in artifacts.quantiles and 0.90 in artifacts.quantiles:
        q10_idx = artifacts.quantiles.index(0.10)
        q90_idx = artifacts.quantiles.index(0.90)
        q10 = preds[:, q10_idx]
        q90 = preds[:, q90_idx]
        interval_width = float(np.mean(q90 - q10))
        interval_coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))

    summary: dict[str, float] = {
        "test_rows": float(len(test_df)),
        "mae_q50": mae,
        "rmse_q50": rmse,
        "r2_q50": r2,
        "interval_width_q10_q90": interval_width,
        "interval_coverage_q10_q90": interval_coverage,
    }
    for idx, q in enumerate(artifacts.quantiles):
        q_label = f"q{int(q * 100)}"
        y_q = preds[:, idx]
        summary[f"mae_{q_label}"] = float(np.mean(np.abs(y_true - y_q)))
        summary[f"rmse_{q_label}"] = float(np.sqrt(np.mean((y_true - y_q) ** 2)))
        summary[f"r2_{q_label}"] = (
            float(1.0 - (np.mean((y_true - y_q) ** 2) / y_var)) if y_var > 0 else float("nan")
        )

    pred_cols = [f"q{int(q * 100)}" for q in artifacts.quantiles]
    predictions = test_df[list(ID_COLUMNS) + [TARGET_COLUMN]].copy()
    predictions = predictions.rename(columns={TARGET_COLUMN: "actual"})
    for idx, column in enumerate(pred_cols):
        predictions[column] = preds[:, idx]
    predictions["residual_q50"] = predictions["actual"] - y_median
    predictions["abs_error_q50"] = predictions["residual_q50"].abs()
    predictions["interval_width_q10_q90"] = q90 - q10
    predictions["within_interval_q10_q90"] = (
        (predictions["actual"] >= q10) & (predictions["actual"] <= q90)
    )

    return QuantileTestSetEvaluation(
        summary=summary,
        quantile_metrics=quantile_metrics,
        predictions=predictions,
    )


def _team_context_row(current_teams: pd.DataFrame, team_id: str) -> pd.Series:
    rows = current_teams[current_teams["TEAM_ID"].astype(str) == str(team_id)]
    if rows.empty:
        raise ValueError(f"TEAM_ID {team_id} not found in current_teams.")
    return rows.iloc[0]


def _players_for_team(current_players: pd.DataFrame, team_id: str) -> pd.DataFrame:
    rows = current_players[current_players["TEAM_ID"].astype(str) == str(team_id)].copy()
    if rows.empty:
        raise ValueError(f"No players found for TEAM_ID {team_id}.")
    return rows.drop_duplicates(subset=["PLAYER_ID"], keep="first").reset_index(drop=True)


def _filter_to_roster(
    players: pd.DataFrame,
    *,
    roster_df: pd.DataFrame | None,
    team_id: str,
) -> pd.DataFrame:
    if roster_df is None or roster_df.empty:
        return players.reset_index(drop=True)
    valid_ids = set(
        roster_df.loc[
            roster_df["TEAM_ID"].astype(str) == str(team_id),
            "PLAYER_ID",
        ].astype(str)
    )
    if not valid_ids:
        return players.reset_index(drop=True)
    filtered = players[players["PLAYER_ID"].astype(str).isin(valid_ids)].copy()
    if filtered.empty:
        return players.reset_index(drop=True)
    return filtered.reset_index(drop=True)


def _build_matchup_features(
    team_players: pd.DataFrame,
    *,
    own_team: pd.Series,
    opp_team: pd.Series,
    event_date: pd.Timestamp,
    is_playoff: bool,
    home_flag: int,
) -> pd.DataFrame:
    feature_df = team_players.copy()
    last_game_dates = pd.to_datetime(feature_df["GAME_DATE"], errors="coerce")
    inferred_rest = (pd.Timestamp(event_date) - last_game_dates).dt.days
    feature_df["days_of_rest"] = inferred_rest.fillna(10).clip(lower=0, upper=10).astype(float)
    feature_df["GAME_DATE"] = pd.Timestamp(event_date)
    feature_df["is_playoff"] = int(is_playoff)
    feature_df["home"] = int(home_flag)

    for column in TEAM_INFERENCE_COLS:
        mirrored_team_col = column.replace("Opp_", "Team_", 1)
        feature_df[column] = float(opp_team.get(mirrored_team_col, 0.0))

    return feature_df


def _build_inference_sequence(
    history_df: pd.DataFrame,
    current_features: pd.DataFrame,
    artifacts: QuantileModelArtifacts,
) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = artifacts.feature_columns
    sequence_length = int(getattr(artifacts, "sequence_length", DEFAULT_SEQUENCE_LENGTH))
    history_matrix = np.nan_to_num(
        history_df[feature_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    current_matrix = np.nan_to_num(
        current_features[feature_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sequence = np.zeros((sequence_length, len(feature_cols)), dtype=np.float32)
    padding_mask = np.ones((sequence_length,), dtype=bool)
    history_slots = max(sequence_length - 1, 0)
    history_len = min(history_slots, history_matrix.shape[0])
    if history_len > 0:
        sequence[history_slots - history_len : history_slots, :] = history_matrix[-history_len:]
        padding_mask[history_slots - history_len : history_slots] = False
    sequence[-1, :] = current_matrix[0]
    padding_mask[-1] = False
    return sequence, padding_mask


def _predict_from_features(
    feature_df: pd.DataFrame,
    *,
    history_df: pd.DataFrame,
    artifacts: QuantileModelArtifacts,
) -> np.ndarray:
    sequence_batch: list[np.ndarray] = []
    mask_batch: list[np.ndarray] = []
    history_df = history_df.copy()
    history_df["GAME_DATE"] = pd.to_datetime(history_df["GAME_DATE"], errors="coerce")

    for _, current_row in feature_df.iterrows():
        player_id = str(current_row["PLAYER_ID"])
        prior_rows = history_df[
            (history_df["PLAYER_ID"].astype(str) == player_id)
            & (history_df["GAME_DATE"] < current_row["GAME_DATE"])
        ].copy()
        prior_rows = prior_rows.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        prior_rows_norm = _normalized_features(prior_rows, artifacts)
        current_row_norm = _normalized_features(current_row.to_frame().T, artifacts)
        sequence, padding_mask = _build_inference_sequence(
            prior_rows_norm,
            current_row_norm,
            artifacts,
        )
        sequence_batch.append(sequence)
        mask_batch.append(padding_mask)

    if not sequence_batch:
        return np.empty((0, len(artifacts.quantiles)), dtype=np.float32)

    seq_tensor = _to_tensor(np.stack(sequence_batch, axis=0))
    mask_tensor = torch.tensor(np.stack(mask_batch, axis=0), dtype=torch.bool)
    with torch.no_grad():
        return artifacts.model(seq_tensor, padding_mask=mask_tensor).numpy()


def predict_matchup(
    *,
    artifacts: QuantileModelArtifacts,
    current_players: pd.DataFrame,
    current_teams: pd.DataFrame,
    history_df: pd.DataFrame,
    home_team_id: str | int,
    away_team_id: str | int,
    event_date: pd.Timestamp,
    is_playoff: bool,
    roster_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict q10/q50/q90 for all players in a matchup."""
    home_team_id = str(home_team_id)
    away_team_id = str(away_team_id)
    event_date = pd.Timestamp(event_date)

    home_team = _team_context_row(current_teams, home_team_id)
    away_team = _team_context_row(current_teams, away_team_id)

    home_players = _filter_to_roster(
        _players_for_team(current_players, home_team_id),
        roster_df=roster_df,
        team_id=home_team_id,
    )
    away_players = _filter_to_roster(
        _players_for_team(current_players, away_team_id),
        roster_df=roster_df,
        team_id=away_team_id,
    )

    home_features = _build_matchup_features(
        home_players,
        own_team=home_team,
        opp_team=away_team,
        event_date=event_date,
        is_playoff=is_playoff,
        home_flag=1,
    )
    away_features = _build_matchup_features(
        away_players,
        own_team=away_team,
        opp_team=home_team,
        event_date=event_date,
        is_playoff=is_playoff,
        home_flag=0,
    )

    home_preds = _predict_from_features(
        home_features,
        history_df=history_df,
        artifacts=artifacts,
    )
    away_preds = _predict_from_features(
        away_features,
        history_df=history_df,
        artifacts=artifacts,
    )

    pred_cols = [f"q{int(q * 100)}" for q in artifacts.quantiles]
    home_out = home_features[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "home"]].copy()
    away_out = away_features[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "home"]].copy()
    for index, column in enumerate(pred_cols):
        home_out[column] = home_preds[:, index]
        away_out[column] = away_preds[:, index]

    output = pd.concat([home_out, away_out], ignore_index=True)
    output["is_playoff"] = int(is_playoff)
    return output.sort_values(["home", "q50"], ascending=[False, False]).reset_index(drop=True)


def artifact_summary(artifact: QuantileModelArtifacts) -> dict[str, Any]:
    return {
        "train_end_date": str(pd.Timestamp(artifact.train_end_date).date()),
        "fit_rows": int(artifact.fit_rows),
        "validation_rows": int(artifact.validation_rows),
        "epochs_trained": int(artifact.epochs_trained),
        "train_loss": float(artifact.train_loss),
        "val_loss": float(artifact.val_loss),
        "feature_count": len(artifact.feature_columns),
        "quantiles": list(artifact.quantiles),
        "sequence_length": int(artifact.sequence_length),
    }


__all__ = [
    "DEFAULT_SEQUENCE_LENGTH",
    "QuantileModelArtifacts",
    "QuantileTestSetEvaluation",
    "artifact_summary",
    "evaluate_test_set",
    "feature_columns_from_frame",
    "load_artifact",
    "predict_matchup",
    "save_artifact",
    "train_production_artifact",
    "train_split_artifact",
]
