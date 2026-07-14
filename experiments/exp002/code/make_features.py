"""Feature generation for exp001.

Original data files are read only. Generated features are written under
`outputs/<experiment_id>/`.
"""

from __future__ import annotations

import argparse
import re
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


def dms_to_decimal(text: object) -> tuple[float, float] | tuple[None, None]:
    if not isinstance(text, str):
        return None, None
    pattern = r"(\d+)°(\d+)'([\d.]+)\"([NSEW])"
    parts = re.findall(pattern, text)
    if len(parts) < 2:
        return None, None

    values: list[float] = []
    for deg, minutes, seconds, direction in parts[:2]:
        value = float(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0
        if direction in {"S", "W"}:
            value *= -1
        values.append(value)
    return values[0], values[1]


def read_turbine_info(path: Path) -> pd.DataFrame:
    info = pd.read_excel(path, sheet_name="info", header=3)
    info = info.dropna(how="all").copy()
    info = info[info["호기"].notna()].copy()
    info["KPX그룹"] = info["KPX그룹"].ffill().astype(int)
    info[["lat", "lon"]] = info["좌표(Google)"].apply(lambda x: pd.Series(dms_to_decimal(x)))
    return info


def haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius_km = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


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


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"grid_id", "latitude", "longitude"}
    return [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude
    ]


def add_group_spatial_features(
    df: pd.DataFrame,
    source: str,
    numeric_cols: list[str],
    turbine_info: pd.DataFrame | None,
    methods: list[str],
) -> pd.DataFrame | None:
    if turbine_info is None or not methods:
        return None

    grid_points = df[["grid_id", "latitude", "longitude"]].drop_duplicates("grid_id").copy()
    frames = []

    for group_id, turbines in turbine_info.groupby("KPX그룹"):
        center_lat = float(turbines["lat"].mean())
        center_lon = float(turbines["lon"].mean())
        distances = haversine_km(center_lat, center_lon, grid_points["latitude"], grid_points["longitude"])
        grid_weights = grid_points[["grid_id"]].copy()
        grid_weights["distance_km"] = distances
        grid_weights["weight"] = 1.0 / np.square(grid_weights["distance_km"] + 0.1)
        grid_weights["weight"] = grid_weights["weight"] / grid_weights["weight"].sum()

        if "weighted_mean" in methods:
            weighted = df[["forecast_kst_dtm", "grid_id", *numeric_cols]].merge(
                grid_weights[["grid_id", "weight"]], on="grid_id", how="left"
            )
            values = weighted[numeric_cols].multiply(weighted["weight"], axis=0)
            values["forecast_kst_dtm"] = weighted["forecast_kst_dtm"]
            wmean = values.groupby("forecast_kst_dtm", sort=True)[numeric_cols].sum(min_count=1).reset_index()
            wmean.columns = ["kst_dtm", *[f"{source}_g{group_id}_{col}_wmean" for col in numeric_cols]]
            frames.append(wmean)

        if "nearest_grid" in methods:
            nearest_grid_id = int(grid_weights.sort_values("distance_km").iloc[0]["grid_id"])
            nearest = df.loc[df["grid_id"] == nearest_grid_id, ["forecast_kst_dtm", *numeric_cols]].copy()
            nearest = nearest.rename(columns={"forecast_kst_dtm": "kst_dtm"})
            nearest.columns = ["kst_dtm", *[f"{source}_g{group_id}_{col}_nearest" for col in numeric_cols]]
            frames.append(nearest)

    if not frames:
        return None

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="kst_dtm", how="left")
    return out


def aggregate_weather(
    path: Path,
    source: str,
    aggregations: list[str],
    turbine_info: pd.DataFrame | None = None,
    spatial_methods: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["data_available_kst_dtm"] = pd.to_datetime(df["data_available_kst_dtm"])
    df = add_wind_features(df, source)

    numeric_cols = numeric_feature_columns(df)

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

    spatial = add_group_spatial_features(
        df,
        source,
        numeric_cols,
        turbine_info,
        spatial_methods or [],
    )
    if spatial is not None:
        features = features.merge(spatial, on="kst_dtm", how="left")
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
    use_spatial = bool(config["features"].get("use_spatial_features", False))
    spatial_methods = config["features"].get("spatial_methods", []) if use_spatial else []
    turbine_info = read_turbine_info(ROOT / data.get("info", "info.xlsx")) if use_spatial else None

    print("LDAPS train 집계 중")
    train_ldaps = aggregate_weather(ROOT / data["train_ldaps"], "ldaps", aggregations, turbine_info, spatial_methods)
    print("GFS train 집계 중")
    train_gfs = aggregate_weather(ROOT / data["train_gfs"], "gfs", aggregations, turbine_info, spatial_methods)
    print("LDAPS test 집계 중")
    test_ldaps = aggregate_weather(ROOT / data["test_ldaps"], "ldaps", aggregations, turbine_info, spatial_methods)
    print("GFS test 집계 중")
    test_gfs = aggregate_weather(ROOT / data["test_gfs"], "gfs", aggregations, turbine_info, spatial_methods)

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
