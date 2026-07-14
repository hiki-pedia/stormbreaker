"""풍력 발전량 예측 대회용 1차 EDA 시각화를 생성합니다."""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "eda" / "figures"

CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}

GROUP_LABELS = {
    "kpx_group_1": "그룹 1",
    "kpx_group_2": "그룹 2",
    "kpx_group_3": "그룹 3",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ["AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic", "Arial Unicode MS"]:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(path.relative_to(ROOT))


def read_labels() -> pd.DataFrame:
    labels = pd.read_csv(ROOT / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    return labels.set_index("kst_dtm").sort_index()


def add_capacity_factor(labels: pd.DataFrame) -> pd.DataFrame:
    cf = pd.DataFrame(index=labels.index)
    for col, cap in CAPACITY_KWH.items():
        cf[col] = labels[col] / cap
    return cf


def plot_target_daily_capacity_factor(labels: pd.DataFrame) -> None:
    cf = add_capacity_factor(labels)
    daily = cf.resample("D").mean()
    rolling = daily.rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(15, 6))
    for col in CAPACITY_KWH:
        ax.plot(daily.index, daily[col], alpha=0.2, linewidth=0.8, label=f"{GROUP_LABELS[col]} 일별")
        ax.plot(rolling.index, rolling[col], linewidth=2.0, label=f"{GROUP_LABELS[col]} 7일 평균")
    ax.set_title("KPX 그룹별 일별 이용률")
    ax.set_xlabel("날짜")
    ax.set_ylabel("이용률")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(ncol=3, loc="upper right")
    savefig("01_target_daily_capacity_factor.png")


def plot_target_monthly_distribution(labels: pd.DataFrame) -> None:
    cf = add_capacity_factor(labels).reset_index()
    cf["month"] = cf["kst_dtm"].dt.month
    long = cf.melt(id_vars=["kst_dtm", "month"], value_vars=list(CAPACITY_KWH), var_name="group", value_name="capacity_factor")
    long["group"] = long["group"].map(GROUP_LABELS)

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(data=long, x="month", y="capacity_factor", hue="group", showfliers=False, ax=ax)
    ax.set_title("월별 시간 단위 이용률 분포")
    ax.set_xlabel("월")
    ax.set_ylabel("이용률")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="", ncol=3, loc="upper right")
    savefig("02_target_monthly_distribution.png")


def month_index() -> pd.PeriodIndex:
    return pd.period_range("2022-01", "2025-12", freq="M")


def monthly_hour_coverage(path: Path, date_col: str, freq: str) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=[date_col])
    times = pd.to_datetime(df[date_col]).drop_duplicates()
    months = month_index()
    out = pd.Series(0.0, index=months)
    for month, count in times.groupby(times.dt.to_period("M")).size().items():
        if month not in out.index:
            continue
        start = month.to_timestamp()
        end = (month + 1).to_timestamp() - pd.Timedelta(minutes=1)
        expected = len(pd.date_range(start, end, freq=freq))
        out.loc[month] = min(float(count) / float(expected), 1.0)
    return out


def plot_data_coverage(labels: pd.DataFrame) -> None:
    months = month_index()
    rows: dict[str, pd.Series] = {}

    for col in CAPACITY_KWH:
        monthly = labels[col].notna().groupby(labels.index.to_period("M")).mean()
        rows[f"정답 {GROUP_LABELS[col]}"] = monthly.reindex(months, fill_value=0.0)

    rows["LDAPS 학습 예보"] = monthly_hour_coverage(ROOT / "train" / "ldaps_train.csv", "forecast_kst_dtm", "h")
    rows["GFS 학습 예보"] = monthly_hour_coverage(ROOT / "train" / "gfs_train.csv", "forecast_kst_dtm", "h")
    rows["VESTAS SCADA"] = monthly_hour_coverage(ROOT / "train" / "scada_vestas_train.csv", "kst_dtm", "10min")
    rows["UNISON SCADA"] = monthly_hour_coverage(ROOT / "train" / "scada_unison_train.csv", "kst_dtm", "10min")
    rows["LDAPS 평가 예보"] = monthly_hour_coverage(ROOT / "test" / "ldaps_test.csv", "forecast_kst_dtm", "h")
    rows["GFS 평가 예보"] = monthly_hour_coverage(ROOT / "test" / "gfs_test.csv", "forecast_kst_dtm", "h")

    coverage = pd.DataFrame(rows).T
    coverage.columns = [str(c) for c in coverage.columns]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(coverage, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "데이터 존재 비율"}, ax=ax)
    ax.set_title("월별 데이터 커버리지")
    ax.set_xlabel("월")
    ax.set_ylabel("")
    tick_positions = np.arange(0.5, len(coverage.columns), 3)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([coverage.columns[int(i - 0.5)] for i in tick_positions], rotation=45, ha="right")
    savefig("03_data_coverage_heatmap.png")


def weather_wind_frame(path: Path, source: str, split: str, u_col: str, v_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["forecast_kst_dtm", "grid_id", u_col, v_col])
    ws = np.sqrt(np.square(df[u_col]) + np.square(df[v_col]))
    return pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(df["forecast_kst_dtm"]),
            "grid_id": df["grid_id"],
            "wind_speed": ws,
            "source": source,
            "split": split,
        }
    )


def plot_weather_wind_distribution() -> None:
    frames = [
        weather_wind_frame(ROOT / "train" / "ldaps_train.csv", "LDAPS", "학습", "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        weather_wind_frame(ROOT / "test" / "ldaps_test.csv", "LDAPS", "평가", "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        weather_wind_frame(ROOT / "train" / "gfs_train.csv", "GFS", "학습", "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        weather_wind_frame(ROOT / "test" / "gfs_test.csv", "GFS", "평가", "heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    ]
    wind = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, source in zip(axes, ["LDAPS", "GFS"], strict=True):
        sns.histplot(
            data=wind[wind["source"] == source],
            x="wind_speed",
            hue="split",
            bins=60,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_title(f"{source} 10m 풍속 분포")
        ax.set_xlabel("풍속")
        ax.set_ylabel("밀도")
        if ax.get_legend() is not None:
            ax.get_legend().set_title("구분")
    savefig("04_weather_wind_distribution.png")


def aggregate_weather_wind(path: Path, prefix: str, u_col: str, v_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["forecast_kst_dtm", "grid_id", u_col, v_col])
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["wind_speed"] = np.sqrt(np.square(df[u_col]) + np.square(df[v_col]))
    agg = df.groupby("forecast_kst_dtm")["wind_speed"].agg(["mean", "max", "min", "std"]).reset_index()
    agg.columns = ["kst_dtm", f"{prefix}_ws_mean", f"{prefix}_ws_max", f"{prefix}_ws_min", f"{prefix}_ws_std"]
    return agg


def binned_curve(data: pd.DataFrame, x_col: str, y_col: str, bins: np.ndarray) -> pd.DataFrame:
    temp = data[[x_col, y_col]].dropna().copy()
    temp["bin"] = pd.cut(temp[x_col], bins=bins, include_lowest=True)
    curve = temp.groupby("bin", observed=True).agg(x=(x_col, "mean"), y=(y_col, "mean"), n=(y_col, "size")).reset_index(drop=True)
    return curve[curve["n"] >= 20]


def plot_wind_power_curve(labels: pd.DataFrame) -> None:
    ldaps = aggregate_weather_wind(ROOT / "train" / "ldaps_train.csv", "ldaps", "heightAboveGround_10_10u", "heightAboveGround_10_10v")
    gfs = aggregate_weather_wind(ROOT / "train" / "gfs_train.csv", "gfs", "heightAboveGround_10_10u", "heightAboveGround_10_10v")

    data = labels.reset_index().merge(ldaps, on="kst_dtm", how="left").merge(gfs, on="kst_dtm", how="left")
    for col, cap in CAPACITY_KWH.items():
        data[f"{col}_cf"] = data[col] / cap

    bins = np.linspace(0, max(data["ldaps_ws_mean"].max(), data["gfs_ws_mean"].max()) + 0.5, 40)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    rng = np.random.default_rng(42)
    for ax, col in zip(axes, CAPACITY_KWH, strict=True):
        y_col = f"{col}_cf"
        valid = data[["ldaps_ws_mean", "gfs_ws_mean", y_col]].dropna()
        if len(valid) > 5000:
            valid = valid.iloc[rng.choice(len(valid), size=5000, replace=False)]
        ax.scatter(valid["ldaps_ws_mean"], valid[y_col], s=5, alpha=0.08, color="#3568a8", label="LDAPS 시간별 관측점")

        ldaps_curve = binned_curve(data, "ldaps_ws_mean", y_col, bins)
        gfs_curve = binned_curve(data, "gfs_ws_mean", y_col, bins)
        ax.plot(ldaps_curve["x"], ldaps_curve["y"], color="#145da0", linewidth=2.5, label="LDAPS 구간 평균")
        ax.plot(gfs_curve["x"], gfs_curve["y"], color="#c23b22", linewidth=2.5, label="GFS 구간 평균")
        ax.set_title(GROUP_LABELS[col])
        ax.set_xlabel("격자 평균 10m 풍속")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("이용률")
    axes[-1].legend(loc="lower right")
    savefig("05_wind_power_curve.png")


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


def read_turbine_info() -> pd.DataFrame:
    info = pd.read_excel(ROOT / "info.xlsx", sheet_name="info", header=3)
    info = info.dropna(how="all").copy()
    info = info[info["호기"].notna()].copy()
    info["KPX그룹"] = info["KPX그룹"].ffill()
    info[["lat", "lon"]] = info["좌표(Google)"].apply(lambda x: pd.Series(dms_to_decimal(x)))
    info["group"] = info["KPX그룹"].astype(int)
    info["label"] = info["제작사"].astype(str) + " " + info["호기"].astype(int).astype(str)
    return info


def read_grid_points(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["grid_id", "latitude", "longitude"])
    points = df.drop_duplicates("grid_id").copy()
    points["source"] = source
    return points


def plot_spatial_layout() -> None:
    turbines = read_turbine_info()
    ldaps = read_grid_points(ROOT / "train" / "ldaps_train.csv", "LDAPS")
    gfs = read_grid_points(ROOT / "train" / "gfs_train.csv", "GFS")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette = {1: "#4daf4a", 2: "#984ea3", 3: "#e41a1c"}

    def base_plot(ax: plt.Axes, label_points: bool) -> None:
        ax.scatter(ldaps["longitude"], ldaps["latitude"], marker="s", s=80, color="#377eb8", alpha=0.75, label="LDAPS 격자")
        ax.scatter(gfs["longitude"], gfs["latitude"], marker="^", s=100, color="#ff7f00", alpha=0.75, label="GFS 격자")
        for group, temp in turbines.groupby("group"):
            ax.scatter(
                temp["lon"],
                temp["lat"],
                s=70,
                color=palette.get(group, "black"),
                edgecolor="white",
                linewidth=0.8,
                label=f"그룹 {group} 터빈",
            )
            if label_points:
                for _, row in temp.iterrows():
                    ax.text(row["lon"] + 0.001, row["lat"] + 0.001, row["label"], fontsize=6)

        if label_points:
            for _, row in ldaps.iterrows():
                ax.text(row["longitude"] + 0.001, row["latitude"] + 0.001, f"L{int(row['grid_id'])}", fontsize=7, color="#24547a")
        else:
            for _, row in gfs.iterrows():
                ax.text(row["longitude"] + 0.006, row["latitude"] + 0.006, f"G{int(row['grid_id'])}", fontsize=7, color="#a44b00")

        ax.set_xlabel("경도")
        ax.set_ylabel("위도")
        ax.set_aspect("equal", adjustable="box")

    base_plot(axes[0], label_points=False)
    axes[0].set_title("전체 격자 범위")

    base_plot(axes[1], label_points=True)
    pad = 0.025
    xmin = min(turbines["lon"].min(), ldaps["longitude"].min()) - pad
    xmax = max(turbines["lon"].max(), ldaps["longitude"].max()) + pad
    ymin = min(turbines["lat"].min(), ldaps["latitude"].min()) - pad
    ymax = max(turbines["lat"].max(), ldaps["latitude"].max()) + pad
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)
    axes[1].set_title("확대: 터빈과 LDAPS 격자")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle("터빈 위치와 기상 격자 배치", y=1.08, fontsize=18, fontweight="bold")
    savefig("06_spatial_grid_turbine_map.png")


def scada_power_curve(path: Path, prefix: str, turbine_count: int, capacity_kw: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    curves = []
    bins = np.linspace(0, 30, 61)
    for i in range(1, turbine_count + 1):
        turbine = f"{prefix}_wtg{i:02d}"
        ws_col = f"{turbine}_ws"
        power_col = f"{turbine}_power_kw10m"
        temp = df[[ws_col, power_col]].rename(columns={ws_col: "wind_speed", power_col: "power"}).dropna()
        temp = temp[(temp["wind_speed"] >= 0) & (temp["power"] >= 0)].copy()
        temp["capacity_factor"] = temp["power"] / capacity_kw
        temp["bin"] = pd.cut(temp["wind_speed"], bins=bins, include_lowest=True)
        curve = temp.groupby("bin", observed=True).agg(
            wind_speed=("wind_speed", "mean"),
            capacity_factor=("capacity_factor", "mean"),
            n=("capacity_factor", "size"),
        )
        curve = curve[curve["n"] >= 20].reset_index(drop=True)
        curve["turbine"] = turbine
        curves.append(curve)
    return pd.concat(curves, ignore_index=True)


def plot_scada_power_curve() -> None:
    vestas = scada_power_curve(ROOT / "train" / "scada_vestas_train.csv", "vestas", 12, 3600.0)
    unison = scada_power_curve(ROOT / "train" / "scada_unison_train.csv", "unison", 5, 4200.0)
    vestas["manufacturer"] = "VESTAS"
    unison["manufacturer"] = "UNISON"
    data = pd.concat([vestas, unison], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="wind_speed", y="capacity_factor", hue="manufacturer", errorbar=("pi", 80), linewidth=2.5, ax=ax)
    ax.set_title("SCADA 기준 풍속-터빈 출력 곡선")
    ax.set_xlabel("터빈 실측 풍속")
    ax.set_ylabel("터빈 이용률")
    ax.set_ylim(-0.05, 1.2)
    savefig("07_scada_power_curve.png")


def write_index() -> None:
    lines = [
        "# EDA 시각화",
        "",
        "- `01_target_daily_capacity_factor.png`: 일별 및 7일 평균 KPX 그룹별 이용률.",
        "- `02_target_monthly_distribution.png`: 월별 시간 단위 이용률 분포.",
        "- `03_data_coverage_heatmap.png`: 데이터셋별 월별 커버리지.",
        "- `04_weather_wind_distribution.png`: 학습/평가 기상예보 풍속 분포.",
        "- `05_wind_power_curve.png`: 예보 풍속과 실제 발전량의 관계.",
        "- `06_spatial_grid_turbine_map.png`: 터빈 위치와 기상 격자 배치.",
        "- `07_scada_power_curve.png`: 과거 SCADA 기준 풍속-출력 관계.",
        "",
    ]
    path = FIG_DIR.parent / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path.relative_to(ROOT))


def main() -> None:
    setup_style()
    labels = read_labels()

    plot_target_daily_capacity_factor(labels)
    plot_target_monthly_distribution(labels)
    plot_data_coverage(labels)
    plot_weather_wind_distribution()
    plot_wind_power_curve(labels)
    plot_spatial_layout()
    plot_scada_power_curve()
    write_index()


if __name__ == "__main__":
    main()
