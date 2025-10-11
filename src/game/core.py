from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from ..engine.indicators import bbands, rsi, sma, vwap_from_bars
from ..engine.market import Bar, MarketSim, OrderBookSnapshot, Tick
from ..engine.scenario import ScenarioGenerator
from .config import GameConfig
from .modes import DEFAULT_MODE, MODE_CYCLE, MODE_LABELS


@dataclass
class RoundLogRecord:
    round_index: int
    mode: str
    phase: str
    prediction: Optional[str]
    confidence: float
    actual: Optional[str]
    delta: Optional[float]
    anchor_price: Optional[float]
    settle_price: Optional[float]
    score: float
    streak: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class GameCore:
    EVENT_LOG_LENGTH = 6
    TICK_TAPE_LENGTH = 40
    TICK_HISTORY_LENGTH = 600

    def __init__(
        self,
        config: GameConfig,
        *,
        market: Optional[MarketSim] = None,
        scenario: Optional[ScenarioGenerator] = None,
    ) -> None:
        self.config = config
        self.sim = market or MarketSim(
            ticks_per_minute=config.ticks_per_minute,
            board_levels=config.board_levels,
        )
        self.scenario = scenario or ScenarioGenerator(config)
        self.mode_cycle = MODE_CYCLE
        self.mode_labels = MODE_LABELS
        self.mode = DEFAULT_MODE
        self.event_log: Deque[Tuple[str, int]] = deque(maxlen=self.EVENT_LOG_LENGTH)
        self.tick_tape: Deque[Tick] = deque(maxlen=self.TICK_TAPE_LENGTH)
        self.tick_history: Deque[Tick] = deque(maxlen=self.TICK_HISTORY_LENGTH)
        self.order_book: OrderBookSnapshot = OrderBookSnapshot([], [], 0.0, 0.0, 0.0, 0.0)
        self.last_result: Optional[str] = None
        self.pred_choice: Optional[str] = None
        self.question_reason: str = ""
        self.phase: str = "stream"
        self.phase_timer: int = self.config.stream_frames
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
        self.round = 1
        self.score = 0.0
        self.streak = 0
        self.conf = 0.6
        self.minute_bars: List[Bar] = []
        self.five_min_bars: List[Bar] = []
        self.current_minute_bar: Optional[List[float]] = None
        self.current_minute_ticks = 0
        self.current_five_bucket: List[Bar] = []
        self.paused = False
        self.log_path = self._initialise_log_path()
        self.reset_game()

    def _initialise_log_path(self) -> Path:
        root = Path(self.config.log_root)
        date_dir = root / datetime.now().strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / "round_log.jsonl"

    def log_round(
        self,
        *,
        actual: Optional[str],
        delta: Optional[float],
        anchor_price: Optional[float],
        settle_price: Optional[float],
    ) -> None:
        record = RoundLogRecord(
            round_index=self.round,
            mode=self.mode,
            phase=self.phase,
            prediction=self.pred_choice,
            confidence=self.conf,
            actual=actual,
            delta=delta,
            anchor_price=anchor_price,
            settle_price=settle_price,
            score=self.score,
            streak=self.streak,
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def _clone_order_book(self, book: OrderBookSnapshot) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            bids=[(float(p), float(q)) for p, q in book.bids],
            asks=[(float(p), float(q)) for p, q in book.asks],
            market_buy_qty=float(book.market_buy_qty),
            market_sell_qty=float(book.market_sell_qty),
            indicative_buy=float(book.indicative_buy),
            indicative_sell=float(book.indicative_sell),
        )

    def reset_game(self) -> None:
        previous_mode = self.previous_mode if self.previous_mode is not None else self.mode
        preopen_mode = self.mode == "preopen"
        entering_preopen = preopen_mode and previous_mode != "preopen"
        exiting_preopen = previous_mode == "preopen" and not preopen_mode

        if entering_preopen:
            self.cached_live_minute_bars = list(getattr(self, "minute_bars", []))
            self.cached_live_five_min_bars = list(getattr(self, "five_min_bars", []))
            existing_history: Deque[Tick] = getattr(self, "tick_history", deque(maxlen=self.TICK_HISTORY_LENGTH))
            existing_tape: Deque[Tick] = getattr(self, "tick_tape", deque(maxlen=self.TICK_TAPE_LENGTH))
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
            snapshot = self.scenario.generate_preopen_snapshot(snapshot_start)
            self.preopen_minute_bars = snapshot.minute_bars
            self.preopen_five_minute_bars = snapshot.five_minute_bars
            self.preopen_tick_digest = snapshot.tick_digest
            self.preopen_order_book = self._clone_order_book(snapshot.order_book)
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

        self.tick_history = deque(tick_history_seed, maxlen=self.TICK_HISTORY_LENGTH)
        self.tick_tape = deque(tick_tape_seed, maxlen=self.TICK_TAPE_LENGTH)
        self.minute_bars = minute_bars
        self.five_min_bars = five_minute_bars
        self.current_minute_bar = None
        self.current_minute_ticks = 0
        self.current_five_bucket = []

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
        self.phase_timer = self.config.stream_frames
        self.pred_choice = None
        self.question_reason = ""
        self.last_result = None
        self.result_flash_timer = 0
        self.anchor_price = None
        self.recent_event = None
        self.tick_accumulator = 0.0
        self.post_round_summary = ""
        self.post_summary_ready = False

        if not preopen_mode:
            if not self.minute_bars:
                bootstrap_ticks = (self.config.history_length + 10) * self.config.ticks_per_minute
                remaining = max(bootstrap_ticks, 1)
                while remaining > 0:
                    step = min(remaining, self.config.ticks_per_minute)
                    ticks, _ = self.sim.step_ticks(step)
                    self._ingest_market_updates(ticks, headline=None)
                    remaining -= step
            self.cached_live_minute_bars = list(self.minute_bars)
            self.cached_live_five_min_bars = list(self.five_min_bars)
            self.cached_live_tick_history = list(self.tick_history)
            self.cached_live_tick_tape = list(self.tick_tape)
            self.cached_live_order_book = self._clone_order_book(self.order_book)
        self.event_log.clear()
        self.recent_event = None
        self.previous_mode = self.mode

    def _ingest_market_updates(self, ticks: List[Tick], headline: Optional[str]) -> None:
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
            self.event_log.appendleft((headline, int(self.config.fps * 8)))

    def advance_market(self, tick_budget: Optional[int] = None) -> None:
        if self.mode == "preopen":
            if self.preopen_order_book.bids or self.preopen_order_book.asks:
                self.order_book = self._clone_order_book(self.preopen_order_book)
            self._decay_events()
            return
        if tick_budget is None:
            self.tick_accumulator += self.config.ticks_per_frame_target
            tick_budget = int(self.tick_accumulator)
            self.tick_accumulator -= tick_budget
        ticks: List[Tick] = []
        headline: Optional[str] = None
        if tick_budget > 0:
            ticks, headline = self.sim.step_ticks(tick_budget)
            self._ingest_market_updates(ticks, headline)
        self._decay_events()

    def _decay_events(self) -> None:
        updated: Deque[Tuple[str, int]] = deque(maxlen=self.event_log.maxlen)
        for msg, timer in list(self.event_log):
            nt = timer - 1
            if nt > 0:
                updated.append((msg, nt))
        self.event_log = updated
        if not self.event_log:
            self.recent_event = None

    def _update_minute_bar(self, tick: Tick) -> None:
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
        elif self.current_minute_ticks >= self.config.ticks_per_minute:
            finalized = Bar(float(bar[0]), float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]))
        if finalized is not None:
            self.minute_bars.append(finalized)
            self.sim.anchor = finalized.c
            self.current_minute_bar = None
            self.current_minute_ticks = 0
            self._update_five_minute_bar(finalized)

    def _update_five_minute_bar(self, bar: Bar) -> None:
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

    def update(self) -> None:
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
                    self.phase_timer = self.config.stream_frames // 2
                else:
                    self.prepare_question()
        elif self.phase == "question":
            self._decay_events()
        elif self.phase == "post":
            if not self.post_summary_ready:
                self.resolve_round()
                self.post_summary_ready = True
                self.phase_timer = self.config.post_display_frames
            else:
                self.phase_timer -= 1
                if self.phase_timer <= 0:
                    self.start_stream_phase()
        if self.result_flash_timer > 0:
            self.result_flash_timer -= 1

    def start_stream_phase(self) -> None:
        self.phase = "stream"
        self.phase_timer = self.config.stream_frames
        self.pred_choice = None
        self.question_reason = ""
        self.anchor_price = None
        self.post_round_summary = ""
        self.post_summary_ready = False
        self.round += 1

    def cycle_mode(self) -> None:
        if self.mode not in self.mode_cycle:
            next_mode = self.mode_cycle[0]
        else:
            idx = self.mode_cycle.index(self.mode)
            next_mode = self.mode_cycle[(idx + 1) % len(self.mode_cycle)]
        self.previous_mode = self.mode
        self.mode = next_mode
        self.reset_game()

    def prepare_question(self) -> None:
        self.phase = "question"
        self.pred_choice = None
        self.anchor_price = self.sim.last
        self.question_reason = self.detect_setup_reason()
        self.result_flash_timer = int(self.config.fps * 2)

    def start_post_phase(self) -> None:
        self.phase = "post"
        self.post_round_summary = ""
        self.post_summary_ready = False
        if self.anchor_price is None:
            self.anchor_price = self.sim.last
        future_ticks = int(self.config.prediction_minutes * self.config.ticks_per_minute)
        self.advance_market(tick_budget=future_ticks)
        self.resolve_round()
        self.post_summary_ready = True
        self.phase_timer = self.config.post_display_frames

    def resolve_round(self) -> None:
        if self.anchor_price is None:
            self.post_round_summary = "観測データ不足のため評価できませんでした"
            self.last_result = "結果: 評価対象データ不足"
            self.result_flash_timer = int(self.config.fps * 2)
            self.log_round(actual=None, delta=None, anchor_price=None, settle_price=None)
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
        self.result_flash_timer = int(self.config.fps * 3)
        self.evaluate_score(actual)
        self.log_round(actual=actual, delta=delta, anchor_price=self.anchor_price, settle_price=current_price)

    def detect_setup_reason(self) -> str:
        if not self.minute_bars:
            return "初期化中"
        closes = [b.c for b in self.minute_bars[-self.config.history_length :]]
        volumes = [b.v for b in self.minute_bars[-self.config.history_length :]]
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

        closes = [b.c for b in self.minute_bars[-self.config.history_length :]]
        volumes = [b.v for b in self.minute_bars[-self.config.history_length :]]
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

        vwap_list = vwap_from_bars(self.minute_bars[-self.config.history_length :])
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
            return {"UP": self.conf, "DOWN": rest / 2, "RANGE": rest / 2}
        if choice == "DOWN":
            return {"UP": rest / 2, "DOWN": self.conf, "RANGE": rest / 2}
        return {"UP": rest / 2, "DOWN": rest / 2, "RANGE": self.conf}

    def evaluate_score(self, actual: str) -> None:
        if self.pred_choice is None:
            if self.last_result is not None:
                self.last_result += " / 回答なし"
            self.streak = 0
            return
        probs = self.prediction_probs(self.pred_choice)
        p_actual = probs.get(actual, max(1.0 / 3, 1e-6))
        gain = math.log(max(p_actual, 1e-6))
        if actual == self.pred_choice:
            self.streak += 1
        else:
            self.streak = 0
        multiplier = 1 + 0.1 * self.streak
        self.score += gain * multiplier
        outcome = "◎" if actual == self.pred_choice else "×"
        if self.last_result is not None:
            self.last_result += f" / 予想: {self.pred_choice} ({outcome})"
            self.last_result += f" / スコアΔ {gain*multiplier:+.2f}"

    def adjust_confidence(self, delta: float) -> None:
        self.conf = min(0.9, max(0.1, self.conf + delta))

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def get_chart_bars(self) -> List[Bar]:
        bars = list(self.minute_bars[-self.config.history_length :])
        if self.current_minute_bar is not None:
            bars.append(
                Bar(
                    float(self.current_minute_bar[0]),
                    float(self.current_minute_bar[1]),
                    float(self.current_minute_bar[2]),
                    float(self.current_minute_bar[3]),
                    float(self.current_minute_bar[4]),
                )
            )
        return bars

    def get_sma(self, period: int = 20) -> List[float]:
        closes = [b.c for b in self.minute_bars[-self.config.history_length :]]
        if len(closes) < period:
            return []
        return sma(closes, period)

    def get_vwap(self) -> List[float]:
        return vwap_from_bars(self.minute_bars[-self.config.history_length :])

    def get_bbands(self, period: int = 20, k: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        closes = [b.c for b in self.minute_bars[-self.config.history_length :]]
        if len(closes) < period + 2:
            return [], [], []
        return bbands(closes, period, k)

    def get_rsi(self, period: int = 14) -> List[float]:
        closes = [b.c for b in self.minute_bars[-self.config.history_length :]]
        if len(closes) < period + 1:
            return []
        return rsi(closes, period)


__all__ = ["GameCore", "RoundLogRecord"]
