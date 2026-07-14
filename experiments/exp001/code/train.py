"""Train exp001 LightGBM baseline models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import yaml

from metrics import GROUP_CAPACITY_KWH, clip_predictions, group_nmae, mean_nmae, one_minus_nmae


ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = list(GROUP_CAPACITY_KWH)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"kst_dtm", "forecast_id", *TARGET_COLUMNS}
    return [col for col in df.columns if col not in excluded]


def make_model(random_state: int = 42, n_estimators: int = 4000) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def split_masks(df: pd.DataFrame, config: dict, target: str) -> tuple[pd.Series, pd.Series]:
    val = config["validation"]
    dt = pd.to_datetime(df["kst_dtm"])

    train_start = pd.Timestamp(val["train_start"])
    if target == "kpx_group_3":
        train_start = pd.Timestamp(val["group_3_train_start"])

    train_mask = (
        (dt >= train_start)
        & (dt <= pd.Timestamp(val["train_end"]))
        & df[target].notna()
    )
    valid_mask = (
        (dt >= pd.Timestamp(val["valid_start"]))
        & (dt <= pd.Timestamp(val["valid_end"]))
        & df[target].notna()
    )
    return train_mask, valid_mask


def all_available_mask(df: pd.DataFrame, target: str) -> pd.Series:
    return df[target].notna()


def build_fallback_values(train: pd.DataFrame) -> dict:
    dt = pd.to_datetime(train["kst_dtm"])
    temp = train.copy()
    temp["_month"] = dt.dt.month
    temp["_hour"] = dt.dt.hour

    fallback: dict[str, dict] = {}
    for target in TARGET_COLUMNS:
        month_hour = (
            temp.dropna(subset=[target])
            .groupby(["_month", "_hour"])[target]
            .median()
        )
        fallback[target] = {
            "global_median": float(temp[target].median()),
            "month_hour_median": {
                f"{int(month):02d}-{int(hour):02d}": float(value)
                for (month, hour), value in month_hour.items()
            },
        }
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp001_baseline.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["outputs"]["dir"]
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(output_dir / "train_features.parquet")
    train["kst_dtm"] = pd.to_datetime(train["kst_dtm"])
    cols = feature_columns(train)

    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(cols, f, ensure_ascii=False, indent=2)

    fallback = build_fallback_values(train)
    with (output_dir / "fallback_values.json").open("w", encoding="utf-8") as f:
        json.dump(fallback, f, ensure_ascii=False, indent=2)

    valid_pred = train[["kst_dtm", *TARGET_COLUMNS]].copy()
    importance_frames = []
    group_metrics = {}

    for i, target in enumerate(TARGET_COLUMNS):
        print(f"{target} 검증 모델 학습 중")
        train_mask, valid_mask = split_masks(train, config, target)
        model = make_model(random_state=42 + i)
        model.fit(
            train.loc[train_mask, cols],
            train.loc[train_mask, target],
            eval_set=[(train.loc[valid_mask, cols], train.loc[valid_mask, target])],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )

        raw_pred = model.predict(train.loc[valid_mask, cols], num_iteration=model.best_iteration_)
        pred_frame = pd.DataFrame({target: raw_pred}, index=train.index[valid_mask])
        pred_frame = clip_predictions(pred_frame)
        valid_pred.loc[valid_mask, f"pred_{target}"] = pred_frame[target]

        cap = GROUP_CAPACITY_KWH[target]
        group_metrics[target] = {
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "valid_nmae_all": group_nmae(train.loc[valid_mask, target], valid_pred.loc[valid_mask, f"pred_{target}"], cap),
            "valid_nmae_actual_ge_10pct": group_nmae(
                train.loc[valid_mask, target],
                valid_pred.loc[valid_mask, f"pred_{target}"],
                cap,
                min_actual_ratio=0.1,
            ),
        }

        importance = pd.DataFrame(
            {
                "target": target,
                "feature": cols,
                "importance": model.feature_importances_,
            }
        )
        importance_frames.append(importance)

        print(f"{target} 최종 모델 학습 중")
        final_mask = all_available_mask(train, target)
        final_model = make_model(random_state=42 + i, n_estimators=group_metrics[target]["best_iteration"])
        final_model.fit(train.loc[final_mask, cols], train.loc[final_mask, target])
        joblib.dump(final_model, model_dir / f"{target}.joblib")

    pred_cols = [f"pred_{target}" for target in TARGET_COLUMNS]
    rename = {f"pred_{target}": target for target in TARGET_COLUMNS}
    valid_only = valid_pred.dropna(subset=pred_cols, how="all").copy()
    y_true = valid_only[TARGET_COLUMNS]
    y_pred = valid_only[pred_cols].rename(columns=rename)

    metrics = {
        "experiment_id": config["experiment_id"],
        "model": config["model"]["type"],
        "feature_count": len(cols),
        "groups": group_metrics,
        "valid_mean_nmae_all": mean_nmae(y_true, y_pred),
        "valid_1_minus_nmae_all": one_minus_nmae(y_true, y_pred),
        "valid_mean_nmae_actual_ge_10pct": mean_nmae(y_true, y_pred, min_actual_ratio=0.1),
        "valid_1_minus_nmae_actual_ge_10pct": one_minus_nmae(y_true, y_pred, min_actual_ratio=0.1),
    }

    valid_only.to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(importance_frames, ignore_index=True).sort_values(
        ["target", "importance"], ascending=[True, False]
    ).to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
