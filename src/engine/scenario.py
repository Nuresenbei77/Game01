from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from .market import Bar, MarketSim, OrderBookSnapshot, Tick
from ..game.config import GameConfig


@dataclass
class PreopenSnapshot:
    minute_bars: List[Bar]
    five_minute_bars: List[Bar]
    tick_digest: List[Tick]
    order_book: OrderBookSnapshot


class ScenarioGenerator:
    def __init__(self, config: GameConfig) -> None:
        self.config = config

    def generate_preopen_snapshot(self, start_price: float) -> PreopenSnapshot:
        seed = int(np.random.default_rng().integers(0, 1_000_000_000))
        snapshot_sim = MarketSim(
            start=start_price,
            seed=seed,
            ticks_per_minute=self.config.ticks_per_minute,
            board_levels=self.config.board_levels,
        )
        snapshot_sim.set_mode("close")

        minute_bars: List[Bar] = []
        five_minute_bars: List[Bar] = []
        tick_digest: Deque[Tick] = deque(maxlen=40)
        current_bar: Optional[List[float]] = None
        ticks_in_minute = 0
        five_bucket: List[Bar] = []
        target_minutes = max(self.config.history_length + 30, 120)

        while len(minute_bars) < target_minutes:
            ticks, _ = snapshot_sim.step_ticks(self.config.ticks_per_minute)
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
                elif ticks_in_minute >= self.config.ticks_per_minute and current_bar is not None:
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

        return PreopenSnapshot(
            minute_bars=minute_bars,
            five_minute_bars=five_minute_bars,
            tick_digest=list(tick_digest),
            order_book=order_book,
        )


__all__ = ["PreopenSnapshot", "ScenarioGenerator"]
