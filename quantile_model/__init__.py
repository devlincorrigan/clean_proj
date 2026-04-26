"""Local quantile regression package for NBA player points O/U modeling."""

from .data import (
    OPP_LAST_GAME_COLS,
    RAW_PLAYER_SEQUENCE_COLS,
    TEAM_INFERENCE_COLS,
    TEAM_LAST_GAME_COLS,
    build_local_datasets,
)
from .model import DEFAULT_QUANTILES, PinballLoss, PlayerPropTransformer
from .service import (
    QuantileModelArtifacts,
    load_artifact,
    predict_matchup,
    save_artifact,
    train_production_artifact,
    train_split_artifact,
)

__all__ = [
    "DEFAULT_QUANTILES",
    "OPP_LAST_GAME_COLS",
    "PinballLoss",
    "PlayerPropTransformer",
    "QuantileModelArtifacts",
    "RAW_PLAYER_SEQUENCE_COLS",
    "TEAM_INFERENCE_COLS",
    "TEAM_LAST_GAME_COLS",
    "build_local_datasets",
    "load_artifact",
    "predict_matchup",
    "save_artifact",
    "train_production_artifact",
    "train_split_artifact",
]
