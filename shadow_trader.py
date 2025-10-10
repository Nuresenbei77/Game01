# shadow_trader.py
# ShadowTrader v0.3 ー テクニカル観察＆質問フェーズを備えた予測ゲーム
# 操作概要:
#  - 観察フェーズ: ←/→ 確信度変更, M モード切替, 1-4 インジケータ切替
#  - 質問フェーズ: ↑ 上昇, ↓ 下落, Space もみ合い, Enter 回答確定
#  - 共通: H ヘルプ表示, P 一時停止, R リセット
#
# スコア: 選択に確信度を割り当てる対数スコア方式（連勝ボーナス付き）
# データ: 歩み値ベースで1分足・5分足を生成、板情報・イベントを擬似表示

import sys
import os
import math
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque

import pygame
import pygame.freetype
import numpy as np
import pandas as pd

# -------------------------
# 画面・ゲーム設定
# -------------------------
WIDTH, HEIGHT = 1100, 760
FPS = 30
HISTORY = 90              # 表示するバー本数
SIM_SECONDS_PER_MINUTE = 30  # 疑似1分足が経過する現実秒
TICKS_PER_MINUTE = 60        # 疑似1分を構成する歩み値数
PREDICTION_MINUTES = 5       # 予測の評価に用いる未来の分数
TICKS_PER_SECOND = TICKS_PER_MINUTE / SIM_SECONDS_PER_MINUTE
TICKS_PER_FRAME_TARGET = TICKS_PER_SECOND / FPS
STREAM_SECONDS = 20          # 観察フェーズ（実時間秒）
POST_DISPLAY_SECONDS = 6     # 結果表示のための短い観察時間
STREAM_FRAMES = int(STREAM_SECONDS * FPS)
POST_DISPLAY_FRAMES = int(POST_DISPLAY_SECONDS * FPS)

# フォント候補
FONT_BUNDLED = os.path.join(os.path.dirname(__file__), "assets", "fonts", "NotoSansJP-Regular.otf")
FONT_CANDIDATES = ["Meiryo", "Yu Gothic", "MS Gothic"]

# 色
COL_BG = (12, 14, 22)
COL_GRID = (38, 44, 64)
COL_WHITE = (240, 240, 240)
COL_DIM = (160, 160, 180)
COL_GREEN = (60, 205, 100)
COL_RED = (235, 70, 80)
COL_YELLOW = (240, 210, 90)
COL_BLUE = (90, 170, 250)
COL_PURPLE = (160, 120, 255)

# -------------------------
# データ構造
# -------------------------
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
    side: str  # "BUY" or "SELL"
    minute_bar: Optional[Bar] = None


@dataclass
class OrderBookSnapshot:
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    market_buy_qty: float
    market_sell_qty: float
    indicative_buy: float
    indicative_sell: float

# -------------------------
# 擬似マーケット（OHLCV生成）
# -------------------------
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

    def __init__(self, start: float = 9200.0, seed: int = 77):
        self.rng = np.random.default_rng(seed)
        self.base_start = float(start)
        self.mode = "preopen"
        self.event_bias_pct = 0.0
        self.event_decay = 0
        self.bias_pct = 0.0
        self.vol_pct = 0.2
        self.autocorr = 0.0
        self.last_minute_return_pct: Optional[float] = None
        self.set_mode(self.mode)

    def set_mode(self, mode: str):
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

    def reset_state(self):
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

    def _step_params(self):
        self.regime_len += 1
        if self.regime_len > self.rng.integers(120, 280):
            self.regime = "highvol" if self.regime == "normal" else "normal"
            self.regime_len = 0
        sigma_factor = 1.6 if self.regime == "highvol" else 1.0
        shock = self.rng.random() < (0.08 if self.regime == "highvol" else 0.035)
        return sigma_factor, shock

    def _apply_event_decay(self):
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
        for i in range(1, 6):
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

    def _prepare_minute_path(self, sigma_factor: float, shock: bool):
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

        ticks = TICKS_PER_MINUTE
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
            minute_step = self.tick % TICKS_PER_MINUTE
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
                observed_high = self._minute_observed_high if self._minute_observed_high is not None else price
                observed_low = self._minute_observed_low if self._minute_observed_low is not None else price
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
                    self.last_minute_return_pct = math.log(max(minute_bar.c, 0.1) / max(minute_bar.o, 0.1)) * 100.0
                else:
                    self.last_minute_return_pct = 0.0

            ticks.append(Tick(price, vol, side, minute_bar))
            if headline is None and self.rng.random() < 1.0 / max(90, 240 - self.event_decay * 2):
                headline = self.maybe_event()
            self._apply_event_decay()
        return ticks, headline

# -------------------------
# 指標計算
# -------------------------
def sma(series: List[float], n=20) -> List[float]:
    s = pd.Series(series, dtype=float).rolling(n).mean()
    return s.tolist()

def vwap_from_bars(bars: List[Bar]) -> List[float]:
    res = []
    cum_pv = 0.0
    cum_v = 0.0
    for b in bars:
        price = (b.h + b.l + b.c)/3.0
        cum_pv += price * b.v
        cum_v += b.v
        res.append(cum_pv/max(cum_v, 1e-9))
    return res

def bbands(series: List[float], n=20, k=2.0) -> Tuple[List[float], List[float], List[float]]:
    s = pd.Series(series, dtype=float)
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    upper = ma + k*sd
    lower = ma - k*sd
    return ma.tolist(), upper.tolist(), lower.tolist()

def rsi(series: List[float], n=14) -> List[float]:
    s = pd.Series(series, dtype=float)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.tolist()

# -------------------------
# ゲーム本体
# -------------------------
class ShadowTrader:
    def __init__(self):
        pygame.init()
        pygame.freetype.init()
        pygame.display.set_caption("ShadowTrader - Candle & Tech MVP")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.load_fonts()

        # チャート領域
        self.chart_rect = pygame.Rect(50, 50, 760, 420)
        self.chart_5m_rect = pygame.Rect(50, 480, 760, 110)
        self.tape_rect = pygame.Rect(50, 600, 760, 80)
        self.side_rect = pygame.Rect(840, 50, 220, 520)

        # 状態
        self.mode_cycle = ["preopen", "morning", "afternoon", "close"]
        self.mode_labels = {
            "preopen": "寄り付き前: 気配とギャップを意識",
            "morning": "前場: トレンドが走りやすい",
            "afternoon": "後場: もみ合い・反転注視",
            "close": "引け直前: 需給と引け成行",
        }
        self.mode = "preopen"
        self.sim = MarketSim()
        self.sim.set_mode(self.mode)
        self.event_log: Deque[Tuple[str, int]] = deque(maxlen=6)
        self.tick_tape: Deque[Tick] = deque(maxlen=40)
        self.tick_history: Deque[Tick] = deque(maxlen=600)
        self.order_book: OrderBookSnapshot = OrderBookSnapshot([], [], 0.0, 0.0, 0.0, 0.0)
        self.last_result: Optional[str] = None
        self.pred_choice: Optional[str] = None
        self.question_reason: str = ""
        self.phase: str = "stream"
        self.phase_timer: int = STREAM_FRAMES
        self.anchor_price: Optional[float] = None
        self.range_threshold = 0.0015
        self.recent_event: Optional[str] = None
        self.result_flash_timer = 0
        self.tick_accumulator = 0.0
        self.post_round_summary: str = ""
        self.post_summary_ready: bool = False
        self.preopen_minute_bars: List[Bar] = []
        self.preopen_five_minute_bars: List[Bar] = []
        self.preopen_tick_digest: List[Tick] = []
        self.preopen_order_book: OrderBookSnapshot = OrderBookSnapshot([], [], 0.0, 0.0, 0.0, 0.0)
        self.cached_live_minute_bars: List[Bar] = []
        self.cached_live_five_min_bars: List[Bar] = []
        self.cached_live_tick_history: List[Tick] = []
        self.cached_live_tick_tape: List[Tick] = []
        self.cached_live_order_book: OrderBookSnapshot = OrderBookSnapshot([], [], 0.0, 0.0, 0.0, 0.0)
        self.previous_mode: Optional[str] = None
        self.reset_game()

        # 表示切替
        self.show_sma = True
        self.show_bbands = True
        self.show_vwap = True
        self.show_rsi = True
        self.show_help = True
        self.paused = False

    def reset_game(self):
        previous_mode = self.previous_mode if self.previous_mode is not None else self.mode
        preopen_mode = self.mode == "preopen"
        entering_preopen = preopen_mode and previous_mode != "preopen"
        exiting_preopen = previous_mode == "preopen" and not preopen_mode

        if entering_preopen:
            self.cached_live_minute_bars = list(getattr(self, "minute_bars", []))
            self.cached_live_five_min_bars = list(getattr(self, "five_min_bars", []))
            existing_history: Deque[Tick] = getattr(
                self, "tick_history", deque(maxlen=600)
            )
            existing_tape: Deque[Tick] = getattr(
                self, "tick_tape", deque(maxlen=40)
            )
            self.cached_live_tick_history = list(existing_history)
            self.cached_live_tick_tape = list(existing_tape)
            current_book = getattr(
                self,
                "order_book",
                OrderBookSnapshot([], [], 0.0, 0.0, 0.0, 0.0),
            )
            self.cached_live_order_book = self._clone_order_book(current_book)

        if preopen_mode:
            snapshot_start = (
                self.cached_live_minute_bars[-1].c
                if self.cached_live_minute_bars
                else self.sim.last
            )
            (
                snapshot_minutes,
                snapshot_fives,
                snapshot_ticks,
                snapshot_book,
            ) = self._generate_preopen_snapshot(snapshot_start)
            self.preopen_minute_bars = snapshot_minutes
            self.preopen_five_minute_bars = snapshot_fives
            self.preopen_tick_digest = snapshot_ticks
            self.preopen_order_book = self._clone_order_book(snapshot_book)
            if self.preopen_minute_bars:
                self.sim.base_start = self.preopen_minute_bars[-1].c
        elif exiting_preopen and self.cached_live_minute_bars:
            self.sim.base_start = self.cached_live_minute_bars[-1].c

        if preopen_mode:
            minute_bars = list(self.preopen_minute_bars)
            five_minute_bars = list(self.preopen_five_minute_bars)
            tick_history_seed = list(self.preopen_tick_digest)
            tick_tape_seed = list(self.preopen_tick_digest)
            order_book_seed: Optional[OrderBookSnapshot] = self.preopen_order_book
        elif exiting_preopen and self.cached_live_minute_bars:
            minute_bars = list(self.cached_live_minute_bars)
            five_minute_bars = list(self.cached_live_five_min_bars)
            tick_history_seed = list(self.cached_live_tick_history)
            tick_tape_seed = list(self.cached_live_tick_tape)
            order_book_seed = self.cached_live_order_book
        else:
            minute_bars = []
            five_minute_bars = []
            tick_history_seed = []
            tick_tape_seed = []
            order_book_seed = None

        self.tick_history = deque(tick_history_seed, maxlen=600)
        self.tick_tape = deque(tick_tape_seed, maxlen=40)
        self.minute_bars = minute_bars
        self.five_min_bars = five_minute_bars
        self.current_minute_bar: Optional[List[float]] = None
        self.current_minute_ticks = 0
        self.current_five_bucket: List[Bar] = []

        if not preopen_mode and self.minute_bars:
            self.sim.base_start = self.minute_bars[-1].c

        self.sim.set_mode(self.mode)

        if order_book_seed is not None:
            self.order_book = self._clone_order_book(order_book_seed)
        else:
            self.order_book = self.sim.order_book()

        self.round = 1
        self.score = 0.0
        self.streak = 0
        self.conf = 0.6
        self.phase = "stream"
        self.phase_timer = STREAM_FRAMES
        self.pred_choice = None
        self.question_reason = ""
        self.last_result = None
        self.result_flash_timer = 0
        self.anchor_price = None
        self.recent_event = None
        self.tick_accumulator = 0.0
        self.post_round_summary = ""
        self.post_summary_ready = False

        if preopen_mode:
            pass
        else:
            # 初期化のため疑似的に履歴を生成
            if not self.minute_bars:
                bootstrap_ticks = (HISTORY + 10) * TICKS_PER_MINUTE
                remaining = max(bootstrap_ticks, 1)
                while remaining > 0:
                    step = min(remaining, TICKS_PER_MINUTE)
                    ticks, _ = self.sim.step_ticks(step)
                    self._ingest_market_updates(ticks, headline=None)
                    remaining -= step
            self.cached_live_minute_bars = list(self.minute_bars)
            self.cached_live_five_min_bars = list(self.five_min_bars)
            self.cached_live_tick_history = list(self.tick_history)
            self.cached_live_tick_tape = list(self.tick_tape)
            self.cached_live_order_book = self._clone_order_book(self.order_book)
        # ブートストラップ時はイベントをクリアして新鮮な状態にする
        self.event_log.clear()
        self.recent_event = None
        self.previous_mode = self.mode

    def load_fonts(self):
        """日本語フォントを優先して読み込む。"""
        font_path = None
        fallback_name = None
        bundled_error = None

        if os.path.exists(FONT_BUNDLED):
            try:
                pygame.freetype.Font(FONT_BUNDLED, 18)
                font_path = FONT_BUNDLED
            except (OSError, FileNotFoundError, pygame.error) as exc:
                bundled_error = exc
        else:
            bundled_error = FileNotFoundError(FONT_BUNDLED)

        if font_path is None:
            for name in FONT_CANDIDATES:
                matched = pygame.font.match_font(name)
                if not matched:
                    continue
                try:
                    pygame.freetype.Font(matched, 18)
                except (OSError, FileNotFoundError, pygame.error):
                    continue
                font_path = matched
                fallback_name = name
                break

        if bundled_error is not None:
            if font_path is not None:
                chosen = fallback_name if fallback_name else os.path.basename(font_path)
                print(
                    f"[ShadowTrader] Failed to load bundled font '{FONT_BUNDLED}': {bundled_error}. Using fallback font '{chosen}'.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[ShadowTrader] Failed to load bundled font '{FONT_BUNDLED}': {bundled_error}. Using generic system font.",
                    file=sys.stderr,
                )

        def make(size, bold=False):
            nonlocal font_path
            if font_path:
                try:
                    font_obj = pygame.freetype.Font(font_path, size)
                except (OSError, FileNotFoundError, pygame.error) as exc:
                    print(
                        f"[ShadowTrader] Failed to instantiate font '{font_path}' (size {size}): {exc}. Using generic system font.",
                        file=sys.stderr,
                    )
                    font_obj = pygame.freetype.SysFont(None, size)
            else:
                font_obj = pygame.freetype.SysFont(None, size)
            if bold:
                font_obj.style |= pygame.freetype.STYLE_STRONG
            return font_obj
        self.font = make(18)
        self.font_small = make(14)
        self.font_footer = make(16)
        self.font_big = make(24)
        self.font_big_bold = make(24, bold=True)

    # -------- 進行 --------
    def _clone_order_book(self, book: OrderBookSnapshot) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            bids=[(float(price), float(vol)) for price, vol in book.bids],
            asks=[(float(price), float(vol)) for price, vol in book.asks],
            market_buy_qty=float(book.market_buy_qty),
            market_sell_qty=float(book.market_sell_qty),
            indicative_buy=float(book.indicative_buy),
            indicative_sell=float(book.indicative_sell),
        )

    def _ingest_market_updates(self, ticks: List[Tick], headline: Optional[str]):
        if ticks:
            self.order_book = self.sim.order_book()
        for tick in ticks:
            self.tick_history.append(tick)
            self.tick_tape.appendleft(tick)
            if len(self.tick_tape) > self.tick_tape.maxlen:
                self.tick_tape.pop()
            self._update_minute_bar(tick)
        if headline:
            self.recent_event = headline
            self.event_log.appendleft((headline, int(FPS * 8)))

    def _generate_preopen_snapshot(
        self, start_price: float
    ) -> Tuple[List[Bar], List[Bar], List[Tick], OrderBookSnapshot]:
        seed = int(np.random.default_rng().integers(0, 1_000_000_000))
        snapshot_sim = MarketSim(start=start_price, seed=seed)
        snapshot_sim.set_mode("close")

        minute_bars: List[Bar] = []
        five_minute_bars: List[Bar] = []
        tick_digest: Deque[Tick] = deque(maxlen=40)
        current_bar: Optional[List[float]] = None
        ticks_in_minute = 0
        five_bucket: List[Bar] = []
        target_minutes = max(HISTORY + 30, 120)

        while len(minute_bars) < target_minutes:
            ticks, _ = snapshot_sim.step_ticks(TICKS_PER_MINUTE)
            for tick in ticks:
                tick_digest.appendleft(tick)
                if current_bar is None:
                    current_bar = [tick.price, tick.price, tick.price, tick.price, 0.0]
                current_bar[1] = max(current_bar[1], tick.price)
                current_bar[2] = min(current_bar[2], tick.price)
                current_bar[3] = tick.price
                current_bar[4] += tick.volume
                ticks_in_minute += 1
                finalized_bar: Optional[Bar] = None
                if tick.minute_bar is not None:
                    summary = tick.minute_bar
                    finalized_bar = Bar(
                        float(summary.o),
                        float(summary.h),
                        float(summary.l),
                        float(summary.c),
                        float(summary.v),
                    )
                elif ticks_in_minute >= TICKS_PER_MINUTE and current_bar is not None:
                    finalized_bar = Bar(
                        float(current_bar[0]),
                        float(current_bar[1]),
                        float(current_bar[2]),
                        float(current_bar[3]),
                        float(current_bar[4]),
                    )

                if finalized_bar is not None:
                    minute_bars.append(finalized_bar)
                    snapshot_sim.anchor = finalized_bar.c
                    five_bucket.append(finalized_bar)
                    if len(five_bucket) == 5:
                        highs = [b.h for b in five_bucket]
                        lows = [b.l for b in five_bucket]
                        aggregated = Bar(
                            five_bucket[0].o,
                            max(highs),
                            min(lows),
                            five_bucket[-1].c,
                            sum(b.v for b in five_bucket),
                        )
                        five_minute_bars.append(aggregated)
                        five_bucket = []
                    current_bar = None
                    ticks_in_minute = 0

        order_book = snapshot_sim.order_book()

        return minute_bars, five_minute_bars, list(tick_digest), order_book

    def advance_market(self, tick_budget: Optional[int] = None):
        if self.mode == "preopen":
            if self.preopen_order_book.bids or self.preopen_order_book.asks:
                self.order_book = self._clone_order_book(self.preopen_order_book)
            self._decay_events()
            return
        if tick_budget is None:
            self.tick_accumulator += TICKS_PER_FRAME_TARGET
            tick_budget = int(self.tick_accumulator)
            self.tick_accumulator -= tick_budget
        ticks: List[Tick] = []
        headline: Optional[str] = None
        if tick_budget > 0:
            ticks, headline = self.sim.step_ticks(tick_budget)
            self._ingest_market_updates(ticks, headline)
        self._decay_events()

    def _decay_events(self):
        updated: Deque[Tuple[str, int]] = deque(maxlen=self.event_log.maxlen)
        for msg, timer in list(self.event_log):
            nt = timer - 1
            if nt > 0:
                updated.append((msg, nt))
        self.event_log = updated
        if not self.event_log:
            self.recent_event = None

    def _update_minute_bar(self, tick: Tick):
        price = tick.price
        volume = tick.volume
        if self.current_minute_bar is None:
            self.current_minute_bar = [price, price, price, price, 0.0]
            self.current_minute_ticks = 0
        bar = self.current_minute_bar
        bar[1] = max(bar[1], price)
        bar[2] = min(bar[2], price)
        bar[3] = price
        bar[4] += volume
        self.current_minute_ticks += 1
        finalized: Optional[Bar] = None
        if tick.minute_bar is not None:
            summary = tick.minute_bar
            finalized = Bar(
                float(summary.o),
                float(summary.h),
                float(summary.l),
                float(summary.c),
                float(summary.v),
            )
        elif self.current_minute_ticks >= TICKS_PER_MINUTE:
            finalized = Bar(float(bar[0]), float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]))
        if finalized is not None:
            self.minute_bars.append(finalized)
            self.sim.anchor = finalized.c
            self.current_minute_bar = None
            self.current_minute_ticks = 0
            self._update_five_minute_bar(finalized)

    def _update_five_minute_bar(self, bar: Bar):
        self.current_five_bucket.append(bar)
        if len(self.current_five_bucket) == 5:
            highs = [b.h for b in self.current_five_bucket]
            lows = [b.l for b in self.current_five_bucket]
            aggregated = Bar(
                self.current_five_bucket[0].o,
                max(highs),
                min(lows),
                self.current_five_bucket[-1].c,
                sum(b.v for b in self.current_five_bucket),
            )
            self.five_min_bars.append(aggregated)
            self.current_five_bucket = []

    def update(self):
        if self.paused:
            return
        if self.mode == "preopen":
            if self.preopen_order_book.bids or self.preopen_order_book.asks:
                self.order_book = self._clone_order_book(self.preopen_order_book)
            self._decay_events()
            if self.result_flash_timer > 0:
                self.result_flash_timer -= 1
            return
        if self.phase == "stream":
            self.advance_market()
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                if len(self.minute_bars) < 5:
                    self.phase_timer = STREAM_FRAMES // 2
                else:
                    self.prepare_question()
        elif self.phase == "question":
            # 質問中は描画のみ。イベント寿命だけ進める
            self._decay_events()
        elif self.phase == "post":
            if not self.post_summary_ready:
                self.resolve_round()
                self.post_summary_ready = True
                self.phase_timer = POST_DISPLAY_FRAMES
            else:
                self.phase_timer -= 1
                if self.phase_timer <= 0:
                    self.start_stream_phase()
        if self.result_flash_timer > 0:
            self.result_flash_timer -= 1

    def start_stream_phase(self):
        self.phase = "stream"
        self.phase_timer = STREAM_FRAMES
        self.pred_choice = None
        self.question_reason = ""
        self.anchor_price = None
        self.post_round_summary = ""
        self.post_summary_ready = False
        self.round += 1

    def cycle_mode(self):
        if self.mode not in self.mode_cycle:
            next_mode = self.mode_cycle[0]
        else:
            idx = self.mode_cycle.index(self.mode)
            next_mode = self.mode_cycle[(idx + 1) % len(self.mode_cycle)]
        self.previous_mode = self.mode
        self.mode = next_mode
        self.reset_game()

    def prepare_question(self):
        self.phase = "question"
        self.pred_choice = None
        self.anchor_price = self.sim.last
        self.question_reason = self.detect_setup_reason()
        self.result_flash_timer = FPS * 2

    def start_post_phase(self):
        self.phase = "post"
        self.post_round_summary = ""
        self.post_summary_ready = False
        if self.anchor_price is None:
            self.anchor_price = self.sim.last
        future_ticks = int(PREDICTION_MINUTES * TICKS_PER_MINUTE)
        self.advance_market(tick_budget=future_ticks)
        self.resolve_round()
        self.post_summary_ready = True
        self.phase_timer = POST_DISPLAY_FRAMES

    def resolve_round(self):
        if self.anchor_price is None:
            self.post_round_summary = "観測データ不足のため評価できませんでした"
            self.last_result = "結果: 評価対象データ不足"
            self.result_flash_timer = FPS * 2
            return
        current_price = self.sim.last
        delta = (current_price - self.anchor_price) / max(self.anchor_price, 1e-6)
        if abs(delta) < self.range_threshold:
            actual = "RANGE"
        elif delta > 0:
            actual = "UP"
        else:
            actual = "DOWN"

        self.post_round_summary = self.build_post_round_summary(actual, delta, self.anchor_price, current_price)
        self.last_result = f"結果(5分先): {actual} (Δ {delta*100:+.2f}%)"
        self.result_flash_timer = FPS * 3
        self.evaluate_score(actual)

    def detect_setup_reason(self) -> str:
        if not self.minute_bars:
            return "初期化中"
        closes = [b.c for b in self.minute_bars[-HISTORY:]]
        volumes = [b.v for b in self.minute_bars[-HISTORY:]]
        reasons: List[str] = []
        latest_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else latest_close
        if len(closes) >= 20:
            ma20 = sma(closes, 20)
            if ma20 and not math.isnan(ma20[-1]):
                ma_now = ma20[-1]
                ma_prev = ma20[-2] if len(ma20) > 1 and not math.isnan(ma20[-2]) else ma_now
                if prev_close < ma_prev <= latest_close:
                    reasons.append("SMA20上抜け")
                elif prev_close > ma_prev >= latest_close:
                    reasons.append("SMA20下抜け")
                gap = (latest_close / ma_now - 1.0)
                if abs(gap) > 0.01:
                    reasons.append(f"価格がSMA20から{gap*100:+.1f}%乖離")
        if len(closes) >= 22:
            ma, upper, lower = bbands(closes, 20, 2.0)
            if upper and not math.isnan(upper[-1]) and latest_close >= upper[-1]:
                reasons.append("ボリンジャー上限タッチ")
            if lower and not math.isnan(lower[-1]) and latest_close <= lower[-1]:
                reasons.append("ボリンジャー下限タッチ")
        if len(closes) >= 15:
            rsi_vals = rsi(closes, 14)
            if rsi_vals and not math.isnan(rsi_vals[-1]):
                r = rsi_vals[-1]
                if r >= 70:
                    reasons.append(f"RSI {r:.0f} → 買われ過ぎ")
                elif r <= 30:
                    reasons.append(f"RSI {r:.0f} → 売られ過ぎ")
        if len(volumes) >= 6:
            avg_vol = sum(volumes[-6:-1]) / max(len(volumes[-6:-1]), 1)
            if avg_vol and volumes[-1] > avg_vol * 1.6:
                reasons.append("出来高急増")
        move = (latest_close - prev_close) / max(prev_close, 1e-6)
        if abs(move) > 0.015:
            reasons.append(f"直近1本で{move*100:+.1f}%変動")
        if not reasons:
            reasons.append("テクニカル中立、次の展開を予測")
        if self.recent_event:
            reasons.append(f"イベント: {self.recent_event}")
        return " / ".join(reasons[:3])

    def build_post_round_summary(
        self, actual: str, delta: float, anchor: float, current_price: float
    ) -> str:
        direction_map = {"UP": "上昇", "DOWN": "下落", "RANGE": "もみ合い"}
        actual_label = direction_map.get(actual, actual)
        lines: List[str] = []
        lines.append("予測から5分先の高速シミュレーション結果です。")
        lines.append(
            f"実現値動き: {anchor:,.2f}→{current_price:,.2f} ({delta*100:+.2f}%) → {actual_label}"
        )

        closes = [b.c for b in self.minute_bars[-HISTORY:]]
        volumes = [b.v for b in self.minute_bars[-HISTORY:]]
        indicator_parts: List[str] = []

        if len(closes) >= 20:
            ma20 = sma(closes, 20)
            if ma20 and not math.isnan(ma20[-1]):
                ma_now = ma20[-1]
                ma_prev = ma20[-2] if len(ma20) > 1 and not math.isnan(ma20[-2]) else ma_now
                slope = ma_now - ma_prev
                slope_txt = "上向き" if slope > 0 else ("下向き" if slope < 0 else "横ばい")
                gap = (current_price / ma_now - 1.0) if ma_now else 0.0
                indicator_parts.append(f"SMA20 {slope_txt} 乖離{gap*100:+.2f}%")

        if len(closes) >= 22:
            ma_arr, upper, lower = bbands(closes, 20, 2.0)
            if (
                upper
                and lower
                and not math.isnan(upper[-1])
                and not math.isnan(lower[-1])
            ):
                upper_now = upper[-1]
                lower_now = lower[-1]
                if current_price >= upper_now:
                    indicator_parts.append("ボリンジャー+2σ上抜け")
                elif current_price <= lower_now:
                    indicator_parts.append("ボリンジャー-2σ下抜け")
                elif ma_arr and not math.isnan(ma_arr[-1]):
                    center = ma_arr[-1]
                    if current_price >= center:
                        indicator_parts.append("ボリンジャー上半分")
                    else:
                        indicator_parts.append("ボリンジャー下半分")

        if len(closes) >= 15:
            rsi_vals = rsi(closes, 14)
            if rsi_vals and not math.isnan(rsi_vals[-1]):
                r = rsi_vals[-1]
                if r >= 70:
                    indicator_parts.append(f"RSI14 {r:.0f} (買われ過ぎ)")
                elif r <= 30:
                    indicator_parts.append(f"RSI14 {r:.0f} (売られ過ぎ)")
                else:
                    indicator_parts.append(f"RSI14 {r:.0f}")

        vwap_list = vwap_from_bars(self.minute_bars[-HISTORY:])
        if vwap_list:
            vwap_val = vwap_list[-1]
            if vwap_val:
                vwap_gap = (current_price / vwap_val - 1.0)
                indicator_parts.append(f"VWAP乖離 {vwap_gap*100:+.2f}%")

        if len(volumes) >= 2:
            reference = volumes[-6:-1]
            if reference:
                avg_vol = sum(reference) / len(reference)
                if avg_vol > 0:
                    ratio = volumes[-1] / avg_vol
                    if ratio >= 1.6:
                        indicator_parts.append(f"出来高 {ratio:.1f}倍 (急増)")
                    else:
                        indicator_parts.append(f"出来高 {ratio:.1f}倍")

        if indicator_parts:
            lines.append("主要指標: " + " / ".join(indicator_parts[:4]))

        reason = self.detect_setup_reason()
        if reason:
            lines.append(f"検出シグナル: {reason}")

        return "\n".join(lines)

    def prediction_probs(self, choice: str) -> dict:
        rest = max(0.0, 1.0 - self.conf)
        if choice == "UP":
            return {"UP": self.conf, "DOWN": rest/2, "RANGE": rest/2}
        if choice == "DOWN":
            return {"UP": rest/2, "DOWN": self.conf, "RANGE": rest/2}
        # RANGE
        return {"UP": rest/2, "DOWN": rest/2, "RANGE": self.conf}

    def evaluate_score(self, actual: str):
        if self.pred_choice is None:
            self.last_result += " / 回答なし"
            self.streak = 0
            return
        probs = self.prediction_probs(self.pred_choice)
        p_actual = probs.get(actual, max(1.0/3, 1e-6))
        gain = math.log(max(p_actual, 1e-6))
        if actual == self.pred_choice:
            self.streak += 1
        else:
            self.streak = 0
        multiplier = 1 + 0.1 * self.streak
        self.score += gain * multiplier
        outcome = "◎" if actual == self.pred_choice else "×"
        self.last_result += f" / 予想: {self.pred_choice} ({outcome})"
        self.last_result += f" / スコアΔ {gain*multiplier:+.2f}"

    # -------- 描画 --------
    def draw(self):
        self.screen.fill(COL_BG)
        self.draw_chart()
        self.draw_5m_chart()
        self.draw_tick_tape()
        self.draw_sidebar()
        self.draw_footer()
        if self.result_flash_timer > 0 and self.last_result:
            self.draw_result_banner()
        if self.phase == "question":
            self.draw_question_overlay()
        elif self.phase == "post":
            self.draw_post_overlay()

        if self.show_help:
            self.draw_help()

        pygame.display.flip()

    def draw_chart(self):
        pygame.draw.rect(self.screen, COL_GRID, self.chart_rect, 1)

        bars = list(self.minute_bars[-HISTORY:])
        if self.current_minute_bar is not None:
            partial = Bar(
                self.current_minute_bar[0],
                self.current_minute_bar[1],
                self.current_minute_bar[2],
                self.current_minute_bar[3],
                self.current_minute_bar[4],
            )
            bars.append(partial)
        if not bars:
            return

        # スケール
        highs = [b.h for b in bars]
        lows = [b.l for b in bars]
        min_p = min(lows)
        max_p = max(highs)
        rng = max(max_p - min_p, 1e-6)
        px_w = self.chart_rect.width / max(len(bars), 1)

        # グリッド横線
        for y_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = self.chart_rect.top + self.chart_rect.height*(1-y_frac)
            pygame.draw.line(self.screen, (28,32,50), (self.chart_rect.left, y), (self.chart_rect.right, y), 1)
            price = min_p + rng*y_frac
            lab, _ = self.font_small.render(f"{price:,.2f}", COL_DIM)
            self.screen.blit(lab, (self.chart_rect.right+8, y-8))

        # 指標描画のためクローズ配列も用意
        closes = [b.c for b in bars]

        # SMA
        if self.show_sma:
            sma20 = sma(closes, 20)
            self._draw_line_indicator(sma20, min_p, rng, COL_YELLOW, px_w)

        # VWAP
        if self.show_vwap:
            vwap_list = vwap_from_bars(bars)
            self._draw_line_indicator(vwap_list, min_p, rng, COL_PURPLE, px_w, width=2)

        # ボリンジャー
        if self.show_bbands:
            ma, upper, lower = bbands(closes, 20, 2.0)
            self._draw_line_indicator(upper, min_p, rng, COL_BLUE, px_w)
            self._draw_line_indicator(lower, min_p, rng, COL_BLUE, px_w)

        # ローソク
        for i, b in enumerate(bars):
            x = self.chart_rect.left + i*px_w
            cx = x + px_w*0.5
            # スケール変換
            def Y(p): return self.chart_rect.bottom - (p - min_p)/rng * self.chart_rect.height
            y_o, y_h, y_l, y_c = map(Y, (b.o, b.h, b.l, b.c))

            col = COL_RED if b.c >= b.o else COL_GREEN
            # ヒゲ
            pygame.draw.line(self.screen, col, (cx, y_h), (cx, y_l), 1)
            # 実体
            body_top = min(y_o, y_c)
            body_h = max(2, abs(y_c - y_o))
            pygame.draw.rect(self.screen, col, pygame.Rect(x+px_w*0.15, body_top, px_w*0.7, body_h))

    def draw_5m_chart(self):
        pygame.draw.rect(self.screen, COL_GRID, self.chart_5m_rect, 1)
        bars = self.five_min_bars[-60:]
        if not bars:
            return
        highs = [b.h for b in bars]
        lows = [b.l for b in bars]
        min_p = min(lows)
        max_p = max(highs)
        rng = max(max_p - min_p, 1e-6)
        px_w = self.chart_5m_rect.width / max(len(bars), 1)
        for i, b in enumerate(bars):
            x = self.chart_5m_rect.left + i*px_w
            cx = x + px_w*0.5

            def Y(p):
                return self.chart_5m_rect.bottom - (p - min_p)/rng * self.chart_5m_rect.height

            y_h = Y(b.h)
            y_l = Y(b.l)
            y_o = Y(b.o)
            y_c = Y(b.c)
            col = COL_RED if b.c >= b.o else COL_GREEN
            pygame.draw.line(self.screen, col, (cx, y_h), (cx, y_l), 1)
            body_top = min(y_o, y_c)
            body_h = max(2, abs(y_c - y_o))
            pygame.draw.rect(self.screen, col, pygame.Rect(x+px_w*0.3, body_top, px_w*0.4, body_h))

        label, _ = self.font_small.render("5分足", COL_DIM)
        self.screen.blit(label, (self.chart_5m_rect.left + 6, self.chart_5m_rect.top + 4))

    def draw_tick_tape(self):
        pygame.draw.rect(self.screen, COL_GRID, self.tape_rect, 1)
        if not self.tick_tape:
            return
        x = self.tape_rect.left + 10
        y = self.tape_rect.top + 10
        step = 18
        max_entries = 24
        for i, tick in enumerate(list(self.tick_tape)[:max_entries]):
            col = COL_RED if tick.side == "BUY" else COL_GREEN
            text = f"{tick.price:,.2f} ({int(tick.volume)})"
            surf, _ = self.font_small.render(text, col)
            self.screen.blit(surf, (x, y))
            y += step
            if y > self.tape_rect.bottom - 20:
                x += 150
                y = self.tape_rect.top + 10
                if x > self.tape_rect.right - 140:
                    break
        label, _ = self.font_small.render("歩み値 (最新→上)", COL_DIM)
        self.screen.blit(label, (self.tape_rect.left + 6, self.tape_rect.top + 4))

    def draw_result_banner(self):
        if self.result_flash_timer > 0:
            alpha = max(120, min(220, int(255 * (self.result_flash_timer / (FPS * 3 + 1)))))
        else:
            alpha = 150
        surf = pygame.Surface((WIDTH-100, 36), pygame.SRCALPHA)
        surf.fill((0, 0, 0, alpha))
        rect = surf.get_rect()
        rect.center = (WIDTH//2, 28)
        self.screen.blit(surf, rect)
        if self.last_result:
            label, _ = self.font_big.render(self.last_result, COL_YELLOW)
            self.screen.blit(label, (rect.left + 20, rect.top + 6))

    def _wrap_lines(
        self, text: str, font_obj: pygame.freetype.Font, max_width: int
    ) -> List[str]:
        lines: List[str] = []
        for raw_line in text.replace(" / ", "\n").split("\n"):
            if raw_line == "":
                lines.append("")
                continue
            current = ""
            for ch in raw_line:
                if ch == "\r":
                    continue
                candidate = current + ch
                surf, _ = font_obj.render(candidate, COL_WHITE)
                if surf.get_width() <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = ch
            if current:
                lines.append(current)
        return lines

    def draw_question_overlay(self):
        panel = pygame.Surface(self.side_rect.size, pygame.SRCALPHA)
        panel.fill((8, 12, 24, 225))
        pygame.draw.rect(panel, (120, 140, 200, 140), panel.get_rect(), 1, border_radius=10)

        margin = 16
        inner_width = panel.get_width() - margin * 2
        y = margin

        title, _ = self.font_big_bold.render("テクニカル注目ポイント", COL_YELLOW)
        panel.blit(title, (margin, y))
        y += title.get_height() + 10

        bullet_color = COL_WHITE
        for reason in self.question_reason.split(" / "):
            bullet = "・" if reason else ""
            wrapped = (
                self._wrap_lines(reason, self.font_small, inner_width - 18)
                if reason
                else [""]
            )
            for i, line in enumerate(wrapped):
                prefix = bullet if i == 0 else "  "
                text_line = f"{prefix}{line}" if line else prefix
                surf, _ = self.font_small.render(text_line, bullet_color)
                panel.blit(surf, (margin, y))
                y += surf.get_height() + 4
            y += 2

        y += 8
        prompt_lines = self._wrap_lines("次の展開は？ (Enterで決定)", self.font, inner_width)
        for line in prompt_lines:
            surf, _ = self.font.render(line, COL_WHITE)
            panel.blit(surf, (margin, y))
            y += surf.get_height() + 6

        conf_text = f"現在の確信度: {self.conf:.1f}"
        conf_surf, conf_rect = self.font_small.render(conf_text, COL_BLUE)
        panel.blit(conf_surf, (margin, y))
        y += conf_rect.height + 10

        choices = [
            ("UP", "上昇する", pygame.K_UP),
            ("DOWN", "下落する", pygame.K_DOWN),
            ("RANGE", "もみ合う", pygame.K_SPACE),
        ]
        choice_box_w = inner_width
        choice_colors = {
            "UP": COL_RED,
            "DOWN": COL_GREEN,
            "RANGE": COL_BLUE,
        }
        for key, text, _ in choices:
            selected = key == self.pred_choice
            marker = "▶" if selected else "  "
            label, label_rect = self.font_big.render(f"{marker} {text}", COL_WHITE)
            line_height = label_rect.height
            if selected:
                highlight = pygame.Surface((choice_box_w, line_height + 8), pygame.SRCALPHA)
                sel_col = choice_colors.get(key, COL_YELLOW)
                highlight.fill((*sel_col, 70))
                panel.blit(highlight, (margin - 4, y - 4))
                label, _ = self.font_big.render(f"{marker} {text}", choice_colors.get(key, COL_YELLOW))
            panel.blit(label, (margin, y))
            y += line_height + 12

        hint = "↑/↓/Spaceで選択  ←/→で確信度  Enterで回答"
        hint_surf, hint_rect = self.font_small.render(hint, COL_YELLOW)
        hint_bg = pygame.Surface((inner_width, hint_rect.height + 10), pygame.SRCALPHA)
        hint_bg.fill((255, 255, 255, 30))
        hint_y = panel.get_height() - margin - hint_rect.height - 6
        panel.blit(hint_bg, (margin, hint_y - 4))
        panel.blit(hint_surf, (margin + 4, hint_y))

        self.screen.blit(panel, self.side_rect)

    def draw_post_overlay(self):
        panel = pygame.Surface(self.side_rect.size, pygame.SRCALPHA)
        panel.fill((10, 16, 30, 235))
        pygame.draw.rect(panel, (200, 180, 80, 140), panel.get_rect(), 1, border_radius=10)

        margin = 16
        inner_width = panel.get_width() - margin * 2
        y = margin

        if self.post_summary_ready and self.post_round_summary:
            title_text = "ラウンド結果サマリー (5分先)"
            sections = [line.strip() for line in self.post_round_summary.split("\n")]
        else:
            title_text = "結果観察フェーズ (5分先)"
            remaining = max(0, math.ceil(self.phase_timer / FPS))
            sections = [
                "5分先の値動きを高速で確定しています。",
                f"残り {remaining}s",
            ]

        title_surf, _ = self.font_big_bold.render(title_text, COL_YELLOW)
        panel.blit(title_surf, (margin, y))
        y += title_surf.get_height() + 12

        info_color = COL_WHITE
        for raw_section in sections:
            if raw_section == "":
                y += 6
                continue
            wrapped_lines = self._wrap_lines(raw_section, self.font_small, inner_width - 18)
            for i, line in enumerate(wrapped_lines):
                prefix = "・" if i == 0 else "  "
                text_line = f"{prefix}{line}" if line else prefix
                surf, _ = self.font_small.render(text_line, info_color)
                panel.blit(surf, (margin, y))
                y += surf.get_height() + 4
            y += 4

        y += 6
        if self.post_summary_ready:
            last_result_text = self.last_result or "結果集計中"
        else:
            last_result_text = "結果集計中"
        result_lines = self._wrap_lines(last_result_text, self.font_small, inner_width)
        for line in result_lines:
            surf, _ = self.font_small.render(line, COL_BLUE)
            panel.blit(surf, (margin, y))
            y += surf.get_height() + 4

        choice_display = self.pred_choice or "-"
        choice_text = f"回答: {choice_display} / 確信度 {self.conf:.1f}"
        choice_surf, choice_rect = self.font_small.render(choice_text, COL_DIM)
        panel.blit(choice_surf, (margin, panel.get_height() - margin - choice_rect.height))

        self.screen.blit(panel, self.side_rect)

    def _draw_line_indicator(self, arr, min_p, rng, color, px_w, width=1):
        pts = []
        for i, v in enumerate(arr[-HISTORY:]):
            if v is None or math.isnan(v):
                continue
            x = self.chart_rect.left + i*px_w
            y = self.chart_rect.bottom - (v - min_p)/rng * self.chart_rect.height
            pts.append((x, y))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, width)

    def draw_sidebar(self):
        # サイド情報（ヒント類）
        pygame.draw.rect(self.screen, (20, 24, 36), self.side_rect)
        pygame.draw.rect(self.screen, COL_GRID, self.side_rect, 1)

        closes = [b.c for b in self.minute_bars[-HISTORY:]]
        txt_y = self.side_rect.top + 10

        def put(text: str, col=COL_WHITE):
            nonlocal txt_y
            if text is None:
                txt_y += 24
                return
            surf, _ = self.font.render(text, col)
            self.screen.blit(surf, (self.side_rect.left + 10, txt_y))
            txt_y += 24

        put("テクニカル概要", COL_YELLOW)
        if closes:
            mom_k = (
                pd.Series(closes).pct_change().rolling(5).mean().iloc[-1]
                if len(closes) >= 6
                else 0.0
            )
            mom_k = 0.0 if pd.isna(mom_k) else float(mom_k)
            arrow = "▲" if mom_k > 0 else ("▼" if mom_k < 0 else "→")
            put(f"{arrow} Momentum(5): {mom_k * 100:+.2f}%")
            vwap_list = vwap_from_bars(self.minute_bars[-HISTORY:])
            if vwap_list:
                vwap_val = vwap_list[-1]
                gap = (closes[-1] / vwap_val - 1.0) if vwap_val else 0.0
                put(f"± VWAP 乖離: {gap * 100:+.2f}%")
            if self.show_rsi:
                rsi_arr = rsi(closes, 14)
                r = rsi_arr[-1] if len(rsi_arr) else float("nan")
                if not math.isnan(r):
                    bars_txt = "▁▂▃▄▅▆▇█"
                    level = int(max(0, min(7, r / 100 * 7)))
                    put(f"{bars_txt[level]} RSI(14): {r:5.1f}")
        else:
            put("データ準備中", COL_DIM)

        put("")
        put("板情報", COL_YELLOW)

        book = self.order_book
        show_market_rows = self.mode in ("preopen", "close")
        bids = list(book.bids)
        asks = list(book.asks)
        ask_levels = list(reversed(asks[:5]))
        bid_levels = bids[:5]
        while len(ask_levels) < 5:
            ask_levels.append((None, None))
        while len(bid_levels) < 5:
            bid_levels.append((None, None))

        ask_total = int(sum(vol for _, vol in asks[:5])) if asks else 0
        bid_total = int(sum(vol for _, vol in bids[:5])) if bids else 0
        indicative_mid = (
            (book.indicative_buy + book.indicative_sell) / 2.0
            if book.indicative_buy and book.indicative_sell
            else (book.indicative_buy or book.indicative_sell or 0.0)
        )

        rows: List[Tuple[str, str, str]] = []
        if show_market_rows:
            rows.append(("market_sell", "成行売り", f"{int(book.market_sell_qty):,}"))
        rows.append(("indicative_sell", "売気配", f"{book.indicative_sell:,.2f}"))
        rows.append(("header", "価格", "数量"))
        for price, vol in ask_levels:
            price_text = "-" if price is None else f"{price:,.2f}"
            qty_text = "-" if (vol is None or vol <= 0) else f"{int(vol):,}"
            rows.append(("ask", price_text, qty_text))
        rows.append(("ask_total", "売合計", "-" if ask_total == 0 else f"{ask_total:,}"))
        rows.append(("indicative_mid", "気配値", f"{indicative_mid:,.2f}"))
        for price, vol in bid_levels:
            price_text = "-" if price is None else f"{price:,.2f}"
            qty_text = "-" if (vol is None or vol <= 0) else f"{int(vol):,}"
            rows.append(("bid", price_text, qty_text))
        rows.append(("bid_total", "買合計", "-" if bid_total == 0 else f"{bid_total:,}"))
        rows.append(("indicative_buy", "買気配", f"{book.indicative_buy:,.2f}"))
        if show_market_rows:
            rows.append(("market_buy", "成行買い", f"{int(book.market_buy_qty):,}"))

        table_margin = 10
        table_x = self.side_rect.left + table_margin
        table_width = self.side_rect.width - table_margin * 2
        row_height = self.font_small.get_sized_height() + 6
        table_height = max(row_height * len(rows), row_height)
        table_y = txt_y
        table_rect = pygame.Rect(table_x, table_y, table_width, table_height)
        pygame.draw.rect(self.screen, (18, 22, 34), table_rect)
        pygame.draw.rect(self.screen, COL_GRID, table_rect, 1)

        price_col_width = int(table_width * 0.55)
        col_split = table_rect.left + price_col_width

        color_map = {
            "header": COL_DIM,
            "ask": COL_GREEN,
            "ask_total": COL_GREEN,
            "market_sell": COL_GREEN,
            "indicative_sell": COL_GREEN,
            "bid": COL_RED,
            "bid_total": COL_RED,
            "market_buy": COL_RED,
            "indicative_buy": COL_RED,
            "indicative_mid": COL_YELLOW,
        }

        bg_map = {
            "header": (26, 30, 46),
            "ask": (32, 24, 28),
            "ask_total": (40, 30, 42),
            "indicative_sell": (30, 32, 42),
            "indicative_mid": (34, 34, 46),
            "bid": (24, 32, 28),
            "bid_total": (40, 30, 42),
            "indicative_buy": (30, 32, 42),
            "market_sell": (34, 28, 36),
            "market_buy": (34, 28, 36),
        }

        def blit_cell(text: str, rect: pygame.Rect, align: str, color):
            if not text:
                return
            color_to_use = COL_DIM if text == "-" else color
            surf, surf_rect = self.font_small.render(text, color_to_use)
            if align == "right":
                surf_rect.topright = (
                    rect.right - 6,
                    rect.top + (row_height - surf_rect.height) // 2,
                )
            elif align == "center":
                surf_rect.center = (
                    rect.centerx,
                    rect.top + row_height // 2,
                )
            else:
                surf_rect.topleft = (
                    rect.left + 6,
                    rect.top + (row_height - surf_rect.height) // 2,
                )
            self.screen.blit(surf, surf_rect)

        prev_type: Optional[str] = None
        for i, (row_type, price_text, qty_text) in enumerate(rows):
            row_rect = pygame.Rect(
                table_rect.left,
                table_rect.top + i * row_height,
                table_width,
                row_height,
            )
            bg = bg_map.get(row_type, (22, 26, 38))
            pygame.draw.rect(self.screen, bg, row_rect)

            if prev_type not in {None, row_type} and row_type in {"bid", "market_buy"}:
                pygame.draw.line(
                    self.screen,
                    COL_GRID,
                    (row_rect.left, row_rect.top),
                    (row_rect.right, row_rect.top),
                    2,
                )
            if row_type in {"indicative_mid", "indicative_buy"}:
                pygame.draw.line(
                    self.screen,
                    COL_GRID,
                    (row_rect.left, row_rect.top),
                    (row_rect.right, row_rect.top),
                    2,
                )

            price_rect = pygame.Rect(row_rect.left, row_rect.top, price_col_width, row_height)
            qty_rect = pygame.Rect(col_split, row_rect.top, table_width - price_col_width, row_height)

            pygame.draw.line(
                self.screen,
                COL_GRID,
                (col_split, row_rect.top),
                (col_split, row_rect.bottom),
            )
            pygame.draw.line(
                self.screen,
                COL_GRID,
                (row_rect.left, row_rect.bottom),
                (row_rect.right, row_rect.bottom),
            )

            color = color_map.get(row_type, COL_WHITE)
            price_align = "right" if row_type in {"ask", "bid"} else "left"
            qty_align = "right"
            blit_cell(price_text, price_rect, price_align, color)
            if qty_text:
                blit_cell(qty_text, qty_rect, qty_align, color)

            prev_type = row_type

        pygame.draw.rect(self.screen, COL_GRID, table_rect, 1)

        txt_y = table_rect.bottom + 16

        put("イベント", COL_YELLOW)
        if self.event_log:
            for msg, _ in list(self.event_log):
                put(f"・{msg}", COL_DIM)
        else:
            put("静穏", COL_DIM)

        put("")
        put("インジケータ表示", COL_YELLOW)
        put("[1] SMA20  " + ("ON" if self.show_sma else "OFF"), COL_DIM)
        put("[2] BBand  " + ("ON" if self.show_bbands else "OFF"), COL_DIM)
        put("[3] VWAP   " + ("ON" if self.show_vwap else "OFF"), COL_DIM)
        put("[4] RSI    " + ("ON" if self.show_rsi else "OFF"), COL_DIM)

    def draw_footer(self):
        # 画面最下部の帯にフッター情報を表示
        footer_rect = pygame.Rect(0, HEIGHT - 60, WIDTH, 60)
        pygame.draw.rect(self.screen, (20, 24, 36), footer_rect)
        pygame.draw.rect(self.screen, COL_GRID, footer_rect, 1)

        margin_x = 20
        available_width = footer_rect.width - margin_x * 2
        phase_label = {
            "stream": "観察",
            "question": "予測入力",
            "post": "結果待ち",
        }.get(self.phase, self.phase)
        timer_text = f"{self.phase_timer//FPS}s" if self.phase != "question" else "---"
        mode_label = self.mode_labels.get(self.mode, self.mode)
        infos = [
            f"Round {self.round}",
            f"Mode: {mode_label}",
            f"Phase: {phase_label}",
            f"Timer {timer_text}",
            f"Score {self.score:.2f}",
            f"Streak {self.streak}",
            f"確信度 {self.conf:.1f}",
            f"選択 {self.pred_choice or '-'}",
        ]
        info_spacing = 18
        rows: List[List[Tuple[pygame.Surface, pygame.Rect]]] = []
        current_row: List[Tuple[pygame.Surface, pygame.Rect]] = []
        current_width = 0

        for text in infos:
            label, label_rect = self.font_footer.render(text, COL_WHITE)
            required_width = label_rect.width if not current_row else current_width + info_spacing + label_rect.width
            if current_row and required_width > available_width:
                rows.append(current_row)
                current_row = []
                current_width = 0
            if current_row:
                current_width += info_spacing + label_rect.width
            else:
                current_width = label_rect.width
            current_row.append((label, label_rect))
        if current_row:
            rows.append(current_row)

        row_heights: List[int] = [max(label_rect.height for _, label_rect in row) for row in rows]

        controls_text = "↑=上昇 ↓=下落 Space=もみ合い ←/→=確信度 Enter=決定 M=モード 1-4=指標 H=ヘルプ P=一時停止 R=リセット"

        def wrap_controls(text: str) -> List[str]:
            wrapped: List[str] = []
            current = ""
            for word in text.split():
                candidate = word if not current else f"{current} {word}"
                surf, _ = self.font_small.render(candidate, COL_DIM)
                if surf.get_width() <= available_width:
                    current = candidate
                else:
                    if current:
                        wrapped.append(current)
                    current = word
            if current:
                wrapped.append(current)
            return wrapped or [text]

        control_lines = wrap_controls(controls_text)
        control_surfs = [self.font_small.render(line, COL_DIM)[0] for line in control_lines]

        row_spacing = 4 if len(rows) > 1 else 0
        controls_spacing = 6 if rows else 0
        control_line_spacing = 2 if len(control_surfs) > 1 else 0
        top_margin = 6
        bottom_margin = 6

        def total_height() -> int:
            metrics_height = sum(row_heights)
            if len(row_heights) > 1:
                metrics_height += row_spacing * (len(row_heights) - 1)
            controls_height = sum(s.get_height() for s in control_surfs)
            if len(control_surfs) > 1:
                controls_height += control_line_spacing * (len(control_surfs) - 1)
            spacing = controls_spacing if control_surfs else 0
            return top_margin + metrics_height + spacing + controls_height + bottom_margin

        min_top = 2
        min_bottom = 2
        min_row_spacing = 2 if len(rows) > 1 else 0
        min_controls_spacing = 2 if control_surfs else 0
        min_control_line_spacing = 1 if len(control_surfs) > 1 else 0

        while total_height() > footer_rect.height:
            adjusted = False
            if top_margin > min_top:
                top_margin -= 1
                adjusted = True
            elif bottom_margin > min_bottom:
                bottom_margin -= 1
                adjusted = True
            elif row_spacing > min_row_spacing:
                row_spacing -= 1
                adjusted = True
            elif controls_spacing > min_controls_spacing:
                controls_spacing -= 1
                adjusted = True
            elif control_line_spacing > min_control_line_spacing:
                control_line_spacing -= 1
                adjusted = True
            else:
                break
        y = footer_rect.top + top_margin
        for idx, row in enumerate(rows):
            row_height = row_heights[idx]
            x = footer_rect.left + margin_x
            for label, label_rect in row:
                self.screen.blit(label, (x, y))
                x += label_rect.width + info_spacing
            if idx < len(rows) - 1:
                y += row_height + row_spacing
            else:
                y += row_height

        if control_surfs:
            y += controls_spacing
            x = footer_rect.left + margin_x
            for idx, surf in enumerate(control_surfs):
                self.screen.blit(surf, (x, y))
                if idx < len(control_surfs) - 1:
                    y += surf.get_height() + control_line_spacing

    def draw_help(self):
        # 半透明パネル
        surf = pygame.Surface((WIDTH-120, HEIGHT-160), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 240))  # 背景を濃くして視認性向上
        rect = surf.get_rect()
        rect.topleft = (60, 80)
        self.screen.blit(surf, rect)
        lines = [
            "ShadowTrader 強化版チュートリアル",
            "",
            "目的: テクニカルの節目で立ち止まり『上昇 / 下落 / もみ合い』を判断する。",
            "観察フェーズで1分足・5分足・歩み値・板情報を確認し、質問フェーズで回答する。",
            "",
            "インジケータ:",
            "  - SMA20 / ボリンジャー / VWAP / RSI を切替表示 (1〜4)。",
            "  - 5分足ミニチャート、歩み値テープ、イベントログを追加。",
            "",
            "操作:",
            "  - 観察: ←/→で確信度 0.1〜0.9 調整、Mでモード切替 (寄り前/前場/後場/引け)。",
            "  - 質問中: ↑=上昇、↓=下落、Space=もみ合い を選択し Enter で回答。",
            "  - P=ポーズ, R=リセット, H=このヘルプ。",
            "",
            "スコア: 選んだ方向に確信度を割り当てる対数スコア方式。連勝すると倍率ボーナス。",
            "大きく外すと減点が増えるので迷ったら確信度を下げる。",
            "",
            "モード:",
            "  - 寄り付き前: ギャップと成行気配が激しい。",
            "  - 前場: トレンド追随が効きやすい。",
            "  - 後場: もみ合い・戻りを意識。",
            "  - 引け: 成行注文とイベントで乱高下。",
            "",
            "Hで閉じる。",
        ]
        y = 100
        for i, ln in enumerate(lines):
            font = self.font_big_bold if i == 0 else self.font
            label, _ = font.render(ln, COL_WHITE)
            self.screen.blit(label, (80, y))
            y += 26

        pygame.draw.rect(self.screen, COL_WHITE, rect, 2)

    # -------- 入力 --------
    def handle_event(self, e: pygame.event.Event):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_UP:
                if self.phase == "question":
                    self.pred_choice = "UP"
            elif e.key == pygame.K_DOWN:
                if self.phase == "question":
                    self.pred_choice = "DOWN"
            elif e.key == pygame.K_SPACE:
                if self.phase == "question":
                    self.pred_choice = "RANGE"
            elif e.key == pygame.K_RETURN:
                if self.phase == "question" and self.pred_choice is not None:
                    self.start_post_phase()
            elif e.key == pygame.K_LEFT:
                self.conf = max(0.1, self.conf - 0.1)
            elif e.key == pygame.K_RIGHT:
                self.conf = min(0.9, self.conf + 0.1)
            elif e.key == pygame.K_1:
                self.show_sma = not self.show_sma
            elif e.key == pygame.K_2:
                self.show_bbands = not self.show_bbands
            elif e.key == pygame.K_3:
                self.show_vwap = not self.show_vwap
            elif e.key == pygame.K_4:
                self.show_rsi = not self.show_rsi
            elif e.key == pygame.K_m:
                self.cycle_mode()
            elif e.key == pygame.K_h:
                self.show_help = not self.show_help
            elif e.key == pygame.K_r:
                self.reset_game()
            elif e.key == pygame.K_p:
                self.paused = not self.paused

# -------------------------
# メインループ
# -------------------------
def main():
    game = ShadowTrader()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            else:
                game.handle_event(e)
        game.update()
        game.draw()
        game.clock.tick(FPS)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
