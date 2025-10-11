from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .market import Bar


def sma(series: List[float], n: int = 20) -> List[float]:
    s = pd.Series(series, dtype=float).rolling(n).mean()
    return s.tolist()


def vwap_from_bars(bars: List[Bar]) -> List[float]:
    res = []
    cum_pv = 0.0
    cum_v = 0.0
    for b in bars:
        price = (b.h + b.l + b.c) / 3.0
        cum_pv += price * b.v
        cum_v += b.v
        res.append(cum_pv / max(cum_v, 1e-9))
    return res


def bbands(series: List[float], n: int = 20, k: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    s = pd.Series(series, dtype=float)
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma.tolist(), upper.tolist(), lower.tolist()


def rsi(series: List[float], n: int = 14) -> List[float]:
    s = pd.Series(series, dtype=float)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.tolist()


__all__ = ["sma", "vwap_from_bars", "bbands", "rsi"]
