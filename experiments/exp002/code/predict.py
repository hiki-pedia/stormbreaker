"""Generate exp001 predictions and submission file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from metrics import GROUP_CAPACITY_KWH, clip_predictions


ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = list(GROUP_CAPACITY_KWH)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fill_missing_predictions(preds: pd.DataFrame, fallback: dict) -> pd.DataFrame:
    filled = preds.copy()
    dt = pd.to_datetime(filled["kst_dtm"])
    keys = dt.dt.strftime("%m-%H")
    for target in TARGET_COLUMNS:
        month_hour = fallback[target]["month_hour_median"]
        global_median = fallback[target]["global_median"]
        missing = filled[target].isna()
        if missing.any():
            replacements = keys.map(month_hour).astype(float)
            filled.loc[missing, target] = replacements[missing]
        filled[target] = filled[target].fillna(global_median)
    return filled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp001_baseline.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["outputs"]["dir"]
    model_dir = output_dir / "models"

    test = pd.read_parquet(output_dir / "test_features.parquet")
    test["kst_dtm"] = pd.to_datetime(test["kst_dtm"])

    with (output_dir / "feature_columns.json").open("r", encoding="utf-8") as f:
        cols = json.load(f)
    with (output_dir / "fallback_values.json").open("r", encoding="utf-8") as f:
        fallback = json.load(f)

    preds = test[["forecast_id", "kst_dtm"]].copy()
    for target in TARGET_COLUMNS:
        model = joblib.load(model_dir / f"{target}.joblib")
        preds[target] = model.predict(test[cols])

    preds = fill_missing_predictions(preds, fallback)
    preds[TARGET_COLUMNS] = clip_predictions(preds[TARGET_COLUMNS])

    sample = pd.read_csv(ROOT / config["data"]["sample_submission"], encoding="utf-8-sig")
    submission = sample[["forecast_id", "forecast_kst_dtm"]].merge(
        preds[["forecast_id", *TARGET_COLUMNS]], on="forecast_id", how="left"
    )
    for target in TARGET_COLUMNS:
        if submission[target].isna().any():
            submission[target] = submission[target].fillna(float(np.nanmedian(preds[target])))

    submission_path = ROOT / config["outputs"]["submission"]
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")
    submission.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(submission)),
        "submission": str(submission_path.relative_to(ROOT)),
        "missing_cells": int(submission[TARGET_COLUMNS].isna().sum().sum()),
        "min": {target: float(submission[target].min()) for target in TARGET_COLUMNS},
        "max": {target: float(submission[target].max()) for target in TARGET_COLUMNS},
    }
    with (output_dir / "prediction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
