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
TICKS_PER_SECOND = TICKS_PER_MINUTE / SIM_SECONDS_PER_MINUTE
TICKS_PER_FRAME_TARGET = TICKS_PER_SECOND / FPS
STREAM_SECONDS = 20          # 観察フェーズ（実時間秒）
POST_SECONDS = 10            # 回答後の結果観察（実時間秒）
STREAM_FRAMES = int(STREAM_SECONDS * FPS)
POST_FRAMES = int(POST_SECONDS * FPS)

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

# -------------------------
# 擬似マーケット（OHLCV生成）
# -------------------------
class MarketSim:
    MODES = {
        "preopen": {
            "bias": 0.02,
            "vol": 0.35,
            "mean_revert": 0.15,
            "event_rate": 0.12,
        },
        "morning": {
            "bias": 0.01,
            "vol": 0.28,
            "mean_revert": 0.1,
            "event_rate": 0.09,
        },
        "afternoon": {
            "bias": 0.0,
            "vol": 0.22,
            "mean_revert": 0.2,
            "event_rate": 0.07,
        },
        "close": {
            "bias": -0.005,
            "vol": 0.32,
            "mean_revert": 0.05,
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

    def __init__(self, start=100.0, seed=77):
        self.rng = np.random.default_rng(seed)
        self.base_start = start
        self.mode = "preopen"
        self.event_bias = 0.0
        self.event_decay = 0
        self.set_mode(self.mode)

    def set_mode(self, mode: str):
        if mode not in self.MODES:
            mode = "preopen"
        self.mode = mode
        params = self.MODES[mode]
        self.bias = params["bias"]
        self.vol = params["vol"]
        self.mean_revert = params["mean_revert"]
        self.event_rate = params["event_rate"]
        self.reset_state()

    def reset_state(self):
        self.last = self.base_start
        self.anchor = self.last
        self.regime = "normal"
        self.regime_len = 0
        self.tick = 0
        self.event_bias = 0.0
        self.event_decay = 0

    def _step_params(self):
        self.regime_len += 1
        if self.regime_len > self.rng.integers(120, 280):
            self.regime = "highvol" if self.regime == "normal" else "normal"
            self.regime_len = 0
        sigma = self.vol * (1.6 if self.regime == "highvol" else 1.0)
        shock = self.rng.random() < (0.08 if self.regime == "highvol" else 0.035)
        return sigma, shock

    def _apply_event_decay(self):
        if self.event_decay > 0:
            self.event_decay -= 1
            self.event_bias *= 0.9
        else:
            self.event_bias = 0.0

    def maybe_event(self) -> Optional[str]:
        if self.rng.random() < self.event_rate:
            headline = self.rng.choice(self.EVENTS)
            bias = self.rng.normal(0.0, self.vol * 2.5)
            self.event_bias = bias
            self.event_decay = self.rng.integers(40, 80)
            return headline
        return None

    def order_book(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        spread = max(0.02, self.vol * 0.4)
        bids = []
        asks = []
        for i in range(1, 6):
            level_spread = spread * i
            bids.append((max(0.1, self.last - level_spread), float(self.rng.integers(80, 180))))
            asks.append((self.last + level_spread, float(self.rng.integers(80, 180))))
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return bids, asks

    def step_ticks(self, count: int) -> Tuple[List[Tick], Optional[str]]:
        ticks: List[Tick] = []
        headline: Optional[str] = None
        for _ in range(count):
            sigma, shock = self._step_params()
            drift = self.bias + self.event_bias
            mean_revert = (self.anchor - self.last) * self.mean_revert
            move = self.rng.normal(drift + mean_revert, sigma * (2.0 if shock else 1.0))
            new_price = max(0.1, self.last + move)
            side = "BUY" if new_price >= self.last else "SELL"
            vol = float(self.rng.integers(10, 40))
            self.last = new_price
            self.tick += 1
            ticks.append(Tick(new_price, vol, side))
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
        self.order_book: Tuple[List[Tuple[float, float]], List[Tuple[float, float]]] = ([], [])
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
        self.reset_game()

        # 表示切替
        self.show_sma = True
        self.show_bbands = True
        self.show_vwap = True
        self.show_rsi = True
        self.show_help = True
        self.paused = False

    def reset_game(self):
        self.tick_history.clear()
        self.tick_tape.clear()
        self.minute_bars: List[Bar] = []
        self.five_min_bars: List[Bar] = []
        self.current_minute_bar: Optional[List[float]] = None
        self.current_minute_ticks = 0
        self.current_five_bucket: List[Bar] = []
        self.sim.set_mode(self.mode)
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
        self.order_book = self.sim.order_book()
        self.tick_accumulator = 0.0

        # 初期化のため疑似的に履歴を生成
        bootstrap_ticks = (HISTORY + 10) * TICKS_PER_MINUTE
        remaining = max(bootstrap_ticks, 1)
        while remaining > 0:
            step = min(remaining, TICKS_PER_MINUTE)
            ticks, _ = self.sim.step_ticks(step)
            self._ingest_market_updates(ticks, headline=None)
            remaining -= step
        # ブートストラップ時はイベントをクリアして新鮮な状態にする
        self.event_log.clear()
        self.recent_event = None

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
        self.font_big = make(24)
        self.font_big_bold = make(24, bold=True)

    # -------- 進行 --------
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

    def advance_market(self, tick_budget: Optional[int] = None):
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
        if self.current_minute_ticks >= TICKS_PER_MINUTE:
            finalized = Bar(bar[0], bar[1], bar[2], bar[3], bar[4])
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
            self.advance_market()
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.resolve_round()
                self.start_stream_phase()
        if self.result_flash_timer > 0:
            self.result_flash_timer -= 1

    def start_stream_phase(self):
        self.phase = "stream"
        self.phase_timer = STREAM_FRAMES
        self.pred_choice = None
        self.question_reason = ""
        self.anchor_price = None
        self.round += 1

    def cycle_mode(self):
        if self.mode not in self.mode_cycle:
            self.mode = self.mode_cycle[0]
        else:
            idx = self.mode_cycle.index(self.mode)
            self.mode = self.mode_cycle[(idx + 1) % len(self.mode_cycle)]
        self.reset_game()

    def prepare_question(self):
        self.phase = "question"
        self.pred_choice = None
        self.anchor_price = self.sim.last
        self.question_reason = self.detect_setup_reason()
        self.result_flash_timer = FPS * 2

    def start_post_phase(self):
        self.phase = "post"
        self.phase_timer = POST_FRAMES

    def resolve_round(self):
        if self.anchor_price is None:
            return
        current_price = self.sim.last
        delta = (current_price - self.anchor_price) / max(self.anchor_price, 1e-6)
        if abs(delta) < self.range_threshold:
            actual = "RANGE"
        elif delta > 0:
            actual = "UP"
        else:
            actual = "DOWN"

        self.last_result = f"結果: {actual} (Δ {delta*100:+.2f}%)"
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

            col = COL_GREEN if b.c >= b.o else COL_RED
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
            col = COL_GREEN if b.c >= b.o else COL_RED
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
            col = COL_GREEN if tick.side == "BUY" else COL_RED
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

    def draw_question_overlay(self):
        overlay = pygame.Surface((WIDTH-140, HEIGHT-200), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 220))
        rect = overlay.get_rect()
        rect.topleft = (70, 90)
        self.screen.blit(overlay, rect)

        lines = [
            "テクニカル注目ポイント", 
            self.question_reason,
            "",
            "次の展開は？ (Enterで決定)",
            f"現在の確信度: {self.conf:.1f}",
        ]
        y = rect.top + 30
        for i, ln in enumerate(lines):
            font = self.font_big_bold if i == 0 else self.font
            label, _ = font.render(ln, COL_WHITE)
            self.screen.blit(label, (rect.left + 40, y))
            y += 32

        choices = [
            ("UP", "上昇する", pygame.K_UP),
            ("DOWN", "下落する", pygame.K_DOWN),
            ("RANGE", "もみ合う", pygame.K_SPACE),
        ]
        y += 10
        for key, text, _ in choices:
            selected = key == self.pred_choice
            col = COL_GREEN if selected else COL_WHITE
            marker = "▶" if selected else "  "
            label, _ = self.font_big.render(f"{marker} {text}", col)
            self.screen.blit(label, (rect.left + 80, y))
            y += 40

        hint = "↑/↓/Spaceで選択  ←/→で確信度  Enterで回答"
        surf, _ = self.font_small.render(hint, COL_DIM)
        self.screen.blit(surf, (rect.left + 40, rect.bottom - 60))

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
        pygame.draw.rect(self.screen, (20,24,36), self.side_rect)
        pygame.draw.rect(self.screen, COL_GRID, self.side_rect, 1)

        closes = [b.c for b in self.minute_bars[-HISTORY:]]
        txt_y = self.side_rect.top + 10

        def put(text, col=COL_WHITE):
            nonlocal txt_y
            surf, _ = self.font.render(text, col)
            self.screen.blit(surf, (self.side_rect.left + 10, txt_y))
            txt_y += 24

        put("テクニカル概要", COL_YELLOW)
        if closes:
            mom_k = pd.Series(closes).pct_change().rolling(5).mean().iloc[-1] if len(closes) >= 6 else 0.0
            mom_k = 0.0 if pd.isna(mom_k) else float(mom_k)
            arrow = "▲" if mom_k > 0 else ("▼" if mom_k < 0 else "→")
            put(f"{arrow} Momentum(5): {mom_k*100:+.2f}%")
            vwap_list = vwap_from_bars(self.minute_bars[-HISTORY:])
            if vwap_list:
                vwap_val = vwap_list[-1]
                gap = (closes[-1] / vwap_val - 1.0) if vwap_val else 0.0
                put(f"± VWAP 乖離: {gap*100:+.2f}%")
            if self.show_rsi:
                rsi_arr = rsi(closes, 14)
                r = rsi_arr[-1] if len(rsi_arr) else float("nan")
                if not math.isnan(r):
                    bars_txt = "▁▂▃▄▅▆▇█"
                    level = int(max(0, min(7, r/100*7)))
                    put(f"{bars_txt[level]} RSI(14): {r:5.1f}")
        else:
            put("データ準備中", COL_DIM)

        put("")
        put("板情報", COL_YELLOW)
        bids, asks = self.order_book
        put("Ask", COL_RED)
        for price, vol in asks[:5]:
            put(f" {price:,.2f} x {int(vol)}", COL_RED)
        put("Bid", COL_GREEN)
        for price, vol in bids[:5]:
            put(f" {price:,.2f} x {int(vol)}", COL_GREEN)

        put("")
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
        footer_rect = pygame.Rect(0, HEIGHT-60, WIDTH, 60)
        pygame.draw.rect(self.screen, (20, 24, 36), footer_rect)
        pygame.draw.rect(self.screen, COL_GRID, footer_rect, 1)
        base_y = footer_rect.top + 10
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
        x = 20
        for t in infos:
            label, _ = self.font.render(t, COL_WHITE)
            self.screen.blit(label, (x, base_y))
            x += 150

        controls = "↑=上昇 ↓=下落 Space=もみ合い ←/→=確信度 Enter=決定 M=モード 1-4=指標 H=ヘルプ P=一時停止 R=リセット"
        surf, _ = self.font_small.render(controls, COL_DIM)
        self.screen.blit(surf, (20, base_y+26))

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
