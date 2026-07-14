"""Feature generation for exp001.

Original data files are read only. Generated features are written under
`outputs/<experiment_id>/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

WIND_PAIRS = {
    "ldaps": [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "h10"),
        ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "h50_max"),
        ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "h50_min"),
        ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "h5_blws"),
    ],
    "gfs": [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "h10"),
        ("heightAboveGround_80_u", "heightAboveGround_80_v", "h80"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "h100"),
        ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "pbl"),
        ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "iso850"),
        ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "iso700"),
        ("isobaricInhPa_500_u", "isobaricInhPa_500_v", "iso500"),
    ],
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_wind_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    for u_col, v_col, name in WIND_PAIRS[source]:
        if u_col not in df.columns or v_col not in df.columns:
            continue
        speed = np.sqrt(np.square(df[u_col]) + np.square(df[v_col]))
        df[f"{name}_wind_speed"] = speed
        safe_speed = speed.replace(0, np.nan)
        df[f"{name}_wind_dir_cos"] = df[u_col] / safe_speed
        df[f"{name}_wind_dir_sin"] = df[v_col] / safe_speed
        df[f"{name}_wind_speed_sq"] = np.square(speed)
        df[f"{name}_wind_speed_cube"] = np.power(speed, 3)
    return df


def aggregate_weather(path: Path, source: str, aggregations: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["data_available_kst_dtm"] = pd.to_datetime(df["data_available_kst_dtm"])
    df = add_wind_features(df, source)

    exclude = {"grid_id", "latitude", "longitude"}
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude
    ]

    grouped = df.groupby("forecast_kst_dtm", sort=True)
    features = grouped[numeric_cols].agg(aggregations)
    features.columns = [f"{source}_{col}_{agg}" for col, agg in features.columns]
    features = features.reset_index().rename(columns={"forecast_kst_dtm": "kst_dtm"})

    available = grouped["data_available_kst_dtm"].first().reset_index().rename(
        columns={
            "forecast_kst_dtm": "kst_dtm",
            "data_available_kst_dtm": f"{source}_data_available_kst_dtm",
        }
    )
    features = features.merge(available, on="kst_dtm", how="left")
    lead = features["kst_dtm"] - features[f"{source}_data_available_kst_dtm"]
    features[f"{source}_lead_hours"] = lead.dt.total_seconds() / 3600.0
    features = features.drop(columns=[f"{source}_data_available_kst_dtm"])
    return features


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["kst_dtm"])
    hour = dt.dt.hour
    month = dt.dt.month
    dayofyear = dt.dt.dayofyear

    df["hour"] = hour
    df["month"] = month
    df["dayofyear"] = dayofyear
    df["dayofweek"] = dt.dt.dayofweek
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 366.0)
    df["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 366.0)
    return df


def build_features(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = config["data"]
    aggregations = config["features"]["grid_aggregations"]

    print("LDAPS train 집계 중")
    train_ldaps = aggregate_weather(ROOT / data["train_ldaps"], "ldaps", aggregations)
    print("GFS train 집계 중")
    train_gfs = aggregate_weather(ROOT / data["train_gfs"], "gfs", aggregations)
    print("LDAPS test 집계 중")
    test_ldaps = aggregate_weather(ROOT / data["test_ldaps"], "ldaps", aggregations)
    print("GFS test 집계 중")
    test_gfs = aggregate_weather(ROOT / data["test_gfs"], "gfs", aggregations)

    labels = pd.read_csv(ROOT / data["train_labels"], encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])

    sample = pd.read_csv(ROOT / data["sample_submission"], encoding="utf-8-sig")
    sample["kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])

    train = labels.merge(train_ldaps, on="kst_dtm", how="left").merge(train_gfs, on="kst_dtm", how="left")
    test = sample[["forecast_id", "kst_dtm"]].merge(test_ldaps, on="kst_dtm", how="left").merge(test_gfs, on="kst_dtm", how="left")

    train = add_calendar_features(train)
    test = add_calendar_features(test)

    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp001_baseline.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["outputs"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test = build_features(config)
    train_path = output_dir / "train_features.parquet"
    test_path = output_dir / "test_features.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train),
                "columns": train.shape[1],
                "missing_cells": int(train.isna().sum().sum()),
            },
            {
                "split": "test",
                "rows": len(test),
                "columns": test.shape[1],
                "missing_cells": int(test.isna().sum().sum()),
            },
        ]
    )
    summary.to_csv(output_dir / "feature_summary.csv", index=False, encoding="utf-8-sig")

    print(f"저장 완료: {train_path.relative_to(ROOT)}")
    print(f"저장 완료: {test_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
