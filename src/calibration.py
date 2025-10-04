"""Calibration utilities for ShadowTrader market simulator.

This script loads 1-minute OHLCV data stored under ``assets/data/`` and
computes log return statistics grouped by time of day.  The aggregated
statistics are written to ``assets/data/calibration_stats.csv`` so that the
simulator can be tuned to realistic intraday dynamics.

Usage
-----
python -m src.calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"
OUTPUT_FILE = DATA_DIR / "calibration_stats.csv"


def _load_frames(files: Iterable[Path]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if df.empty or "close" not in df.columns:
            continue
        frame = df.copy()
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        else:
            raise ValueError(f"{path} is missing a 'timestamp' column")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame["log_close"] = np.log(frame["close"].astype(float))
        frame["log_return"] = frame["log_close"].diff()
        frame = frame.dropna(subset=["log_return"])
        if frame.empty:
            continue
        frame["time_label"] = frame["timestamp"].dt.strftime("%H:%M")
        frames.append(frame)
    return frames


def compute_statistics() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. Please add 1-minute OHLCV data."
        )
    frames = _load_frames(files)
    if not frames:
        raise RuntimeError("Loaded data is empty after preprocessing")
    data = pd.concat(frames, ignore_index=True)
    grouped = data.groupby("time_label")["log_return"]
    stats = grouped.agg(["mean", "std", "count"])
    stats.rename(
        columns={"mean": "log_return_mean", "std": "log_return_std"},
        inplace=True,
    )
    autocorr = grouped.apply(lambda x: x.autocorr(lag=1))
    stats["log_return_autocorr_lag1"] = autocorr
    stats = stats[stats["count"] > 0].drop(columns=["count"])
    stats = stats.sort_index()
    return stats


def save_statistics(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate intraday statistics")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Path to write the aggregated statistics CSV",
    )
    args = parser.parse_args(argv)
    stats = compute_statistics()
    save_statistics(stats, args.output)
    print(f"Saved calibration statistics to {args.output}")


if __name__ == "__main__":
    main()
