from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Bar:
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class Tick:
    price: float
    volume: float
    side: str
    minute_bar: Optional[Bar] = None


@dataclass
class OrderBookSnapshot:
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    market_buy_qty: float
    market_sell_qty: float
    indicative_buy: float
    indicative_sell: float


class MarketSim:
    MODES = {
        "preopen": {
            "bias_pct": -0.155,
            "vol_pct": 1.27,
            "autocorr": -0.014,
            "mean_revert": 0.45,
            "event_rate": 0.12,
        },
        "morning": {
            "bias_pct": 0.005,
            "vol_pct": 0.17,
            "autocorr": -0.087,
            "mean_revert": 0.32,
            "event_rate": 0.09,
        },
        "afternoon": {
            "bias_pct": 0.004,
            "vol_pct": 0.16,
            "autocorr": 0.054,
            "mean_revert": 0.38,
            "event_rate": 0.07,
        },
        "close": {
            "bias_pct": 0.037,
            "vol_pct": 0.16,
            "autocorr": 0.099,
            "mean_revert": 0.28,
            "event_rate": 0.11,
        },
    }

    EVENTS = [
        "大型投資家の成行買い",
        "主要指数先物が急落",
        "業績上方修正の噂",
        "為替の急変",
        "要人発言で市場揺れる",
        "AI銘柄に資金流入",
        "規制関連のネガティブヘッドライン",
    ]

    def __init__(
        self,
        start: float = 9200.0,
        seed: int = 77,
        *,
        ticks_per_minute: int = 60,
        board_levels: int = 5,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.base_start = float(start)
        self.mode = "preopen"
        self.event_bias_pct = 0.0
        self.event_decay = 0
        self.bias_pct = 0.0
        self.vol_pct = 0.2
        self.autocorr = 0.0
        self.last_minute_return_pct: Optional[float] = None
        self.ticks_per_minute = max(1, int(ticks_per_minute))
        self.board_levels = max(1, int(board_levels))
        self.set_mode(self.mode)

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            mode = "preopen"
        self.mode = mode
        params = self.MODES[mode]
        self.bias_pct = params["bias_pct"]
        self.vol_pct = params["vol_pct"]
        self.autocorr = params.get("autocorr", 0.0)
        self.mean_revert = params["mean_revert"]
        self.event_rate = params["event_rate"]
        self.reset_state()

    def reset_state(self) -> None:
        self.last = float(self.base_start)
        self.anchor = float(self.last)
        self.regime = "normal"
        self.regime_len = 0
        self.tick = 0
        self.event_bias_pct = 0.0
        self.event_decay = 0
        self.last_minute_return_pct = None
        self._minute_price_path: List[float] = []
        self._minute_bar_pending: Optional[Bar] = None
        self._minute_volume_total = 0.0
        self._minute_high_index: Optional[int] = None
        self._minute_low_index: Optional[int] = None
        self._minute_observed_high: Optional[float] = None
        self._minute_observed_low: Optional[float] = None
        self._minute_volatility_span = 0.0

    def _step_params(self) -> Tuple[float, bool]:
        self.regime_len += 1
        if self.regime_len > self.rng.integers(120, 280):
            self.regime = "highvol" if self.regime == "normal" else "normal"
            self.regime_len = 0
        sigma_factor = 1.6 if self.regime == "highvol" else 1.0
        shock = self.rng.random() < (0.08 if self.regime == "highvol" else 0.035)
        return sigma_factor, shock

    def _apply_event_decay(self) -> None:
        if self.event_decay > 0:
            self.event_decay -= 1
            self.event_bias_pct *= 0.9
        else:
            self.event_bias_pct = 0.0

    def maybe_event(self) -> Optional[str]:
        if self.rng.random() < self.event_rate:
            headline = self.rng.choice(self.EVENTS)
            vol_scale = max(0.3, self.vol_pct * 1.8)
            bias_pct = float(self.rng.normal(0.0, vol_scale))
            self.event_bias_pct = bias_pct
            self.event_decay = self.rng.integers(40, 80)
            return headline
        return None

    def order_book(self) -> OrderBookSnapshot:
        spread = max(self.last * 0.0004, self.last * (self.vol_pct / 100.0) * 0.6)
        bids: List[Tuple[float, float]] = []
        asks: List[Tuple[float, float]] = []
        for i in range(1, self.board_levels + 1):
            level_spread = spread * i
            bids.append((max(0.1, self.last - level_spread), float(self.rng.integers(80, 180))))
            asks.append((self.last + level_spread, float(self.rng.integers(80, 180))))
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        market_buy_qty = float(self.rng.integers(40, 180))
        market_sell_qty = float(self.rng.integers(40, 180))
        best_bid = bids[0][0] if bids else max(0.1, self.last - spread)
        best_ask = asks[0][0] if asks else self.last + spread

        return OrderBookSnapshot(
            bids=bids,
            asks=asks,
            market_buy_qty=market_buy_qty,
            market_sell_qty=market_sell_qty,
            indicative_buy=best_ask,
            indicative_sell=best_bid,
        )

    def _prepare_minute_path(self, sigma_factor: float, shock: bool) -> None:
        open_price = float(self.last)
        open_log = math.log(max(open_price, 0.1))
        baseline_pct = (self.bias_pct + self.event_bias_pct) / 100.0
        if self.last_minute_return_pct is not None:
            baseline_pct += (self.autocorr * (self.last_minute_return_pct / 100.0))
        anchor_term = 0.0
        if self.anchor > 0:
            anchor_term = math.log(max(self.anchor, 0.1) / max(open_price, 0.1)) * self.mean_revert
        combined_bias = baseline_pct + anchor_term

        base_vol_pct = max(self.vol_pct / 100.0, 0.0004)
        volatility_span = base_vol_pct * sigma_factor * (1.6 if shock else 1.0)
        close_move = float(self.rng.normal(combined_bias, volatility_span))
        close_log = open_log + close_move

        wick_span = volatility_span * (1.25 + self.rng.random() * 0.9)
        if shock:
            wick_span *= 1.4
        high_log = max(open_log, close_log) + abs(self.rng.normal(wick_span * 0.6, wick_span * 0.35))
        low_log = min(open_log, close_log) - abs(self.rng.normal(wick_span * 0.6, wick_span * 0.35))

        if high_log - low_log < wick_span * 0.6:
            pad = wick_span * 0.4
            high_log += pad
            low_log -= pad

        high_log = max(high_log, open_log, close_log) + 1e-6
        low_log = min(low_log, open_log, close_log)
        if high_log - low_log < 1e-4:
            bump = max(wick_span * 0.5, 0.0002)
            high_log += bump
            low_log -= bump

        first_pos = float(self.rng.uniform(0.18, 0.55))
        second_pos = float(self.rng.uniform(first_pos + 0.15, 0.92))
        if self.rng.random() < 0.5:
            positions = [0.0, first_pos, second_pos, 1.0]
            values = [open_log, high_log, low_log, close_log]
            high_pos = first_pos
            low_pos = second_pos
        else:
            positions = [0.0, first_pos, second_pos, 1.0]
            values = [open_log, low_log, high_log, close_log]
            low_pos = first_pos
            high_pos = second_pos

        ticks = self.ticks_per_minute
        tick_positions = np.linspace(0.0, 1.0, ticks)
        log_prices = np.interp(tick_positions, positions, values)
        noise = self.rng.normal(0.0, volatility_span * 0.18, size=ticks)
        kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=float)
        noise = np.convolve(noise, kernel, mode="same")
        log_prices = log_prices + noise

        high_idx = int(np.clip(round(high_pos * (ticks - 1)), 0, ticks - 1))
        low_idx = int(np.clip(round(low_pos * (ticks - 1)), 0, ticks - 1))
        log_prices = np.clip(log_prices, low_log, high_log)
        log_prices[0] = open_log
        log_prices[-1] = close_log
        log_prices[high_idx] = max(log_prices[high_idx], high_log)
        log_prices[low_idx] = min(log_prices[low_idx], low_log)

        observed_high_log = float(np.max(log_prices))
        observed_low_log = float(np.min(log_prices))
        if observed_high_log <= observed_low_log:
            observed_high_log = observed_low_log + max(volatility_span * 0.4, 0.0002)
            idx = max(high_idx, low_idx)
            log_prices[idx] = observed_high_log

        self._minute_price_path = [float(p) for p in log_prices]
        self._minute_bar_pending = Bar(
            float(math.exp(open_log)),
            float(math.exp(observed_high_log)),
            float(math.exp(observed_low_log)),
            float(math.exp(close_log)),
            0.0,
        )
        self._minute_volume_total = 0.0
        self._minute_high_index = int(high_idx)
        self._minute_low_index = int(low_idx)
        self._minute_observed_high = float(math.exp(observed_high_log))
        self._minute_observed_low = float(math.exp(observed_low_log))
        self._minute_volatility_span = float(volatility_span)

    def step_ticks(self, count: int) -> Tuple[List[Tick], Optional[str]]:
        ticks: List[Tick] = []
        headline: Optional[str] = None
        for _ in range(count):
            sigma_factor, shock = self._step_params()
            if not self._minute_price_path:
                self._prepare_minute_path(sigma_factor, shock)
            minute_step = self.tick % self.ticks_per_minute
            target_log_price = float(self._minute_price_path.pop(0))
            volatility_span = max(self._minute_volatility_span, 1e-6)
            jitter_scale = volatility_span * (0.25 if not shock else 0.4)
            jitter = float(self.rng.normal(0.0, jitter_scale))
            log_price = target_log_price + jitter
            if self._minute_bar_pending is not None:
                lower_bound = math.log(max(self._minute_bar_pending.l, 0.1))
                upper_bound = math.log(max(self._minute_bar_pending.h, 0.1))
            else:
                lower_bound = target_log_price - volatility_span * 2.0
                upper_bound = target_log_price + volatility_span * 2.0
            log_price = min(max(log_price, lower_bound), upper_bound)
            if (
                self._minute_high_index is not None
                and minute_step == self._minute_high_index
                and self._minute_bar_pending is not None
            ):
                log_price = math.log(max(self._minute_bar_pending.h, 0.1))
            if (
                self._minute_low_index is not None
                and minute_step == self._minute_low_index
                and self._minute_bar_pending is not None
            ):
                log_price = math.log(max(self._minute_bar_pending.l, 0.1))
            if not self._minute_price_path and self._minute_bar_pending is not None:
                log_price = math.log(max(self._minute_bar_pending.c, 0.1))
            price = max(0.1, math.exp(log_price))
            side = "BUY" if price >= self.last else "SELL"
            base_vol = float(self.rng.integers(10, 40))
            if self._minute_high_index is not None and minute_step == self._minute_high_index:
                base_vol *= 1.4
            if self._minute_low_index is not None and minute_step == self._minute_low_index:
                base_vol *= 1.3
            vol = float(base_vol)
            self.last = price
            self.tick += 1
            self._minute_volume_total += vol
            if self._minute_observed_high is not None:
                self._minute_observed_high = max(self._minute_observed_high, price)
            if self._minute_observed_low is not None:
                self._minute_observed_low = min(self._minute_observed_low, price)

            minute_bar = None
            if not self._minute_price_path and self._minute_bar_pending is not None:
                observed_high = (
                    self._minute_observed_high if self._minute_observed_high is not None else price
                )
                observed_low = (
                    self._minute_observed_low if self._minute_observed_low is not None else price
                )
                minute_bar = Bar(
                    float(self._minute_bar_pending.o),
                    float(observed_high),
                    float(observed_low),
                    float(price),
                    float(self._minute_volume_total),
                )
                self._minute_bar_pending = None
                self._minute_high_index = None
                self._minute_low_index = None
                self._minute_volume_total = 0.0
                self._minute_observed_high = None
                self._minute_observed_low = None
                if minute_bar.o > 0:
                    self.last_minute_return_pct = (
                        math.log(max(minute_bar.c, 0.1) / max(minute_bar.o, 0.1)) * 100.0
                    )
                else:
                    self.last_minute_return_pct = 0.0

            ticks.append(Tick(price, vol, side, minute_bar))
            if headline is None and self.rng.random() < 1.0 / max(90, 240 - self.event_decay * 2):
                headline = self.maybe_event()
            self._apply_event_decay()
        return ticks, headline


__all__ = ["Bar", "Tick", "OrderBookSnapshot", "MarketSim"]
