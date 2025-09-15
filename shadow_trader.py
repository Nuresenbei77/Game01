# shadow_trader.py
# ShadowTrader v0.2 ー ローソク足＋テクニカル選択＆ヒント付き予測ゲーム（MVP）
# 操作:
#  - ↑: 上予想  ↓: 下予想  Space: PASS
#  - ←/→: 確信度 0.1〜0.9
#  - 1: SMA20 切替, 2: Bollinger(20,2) 切替, 3: VWAP 切替, 4: RSIパネル 切替
#  - H: ヘルプ/チュートリアル表示,  R: リセット
#  - P: 一時停止/再開
#
# スコア: 対数スコア（正しく当てるほど加点。確信過剰で外すと減点が大きい）
# データ: 擬似1分足風のOHLCVを内部生成（ボラレジーム＋ランダムショック）

import sys
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import pygame
import numpy as np
import pandas as pd

# -------------------------
# 画面・ゲーム設定
# -------------------------
WIDTH, HEIGHT = 950, 700
FPS = 30
HISTORY = 80          # 表示するバー本数
ROUND_FRAMES = FPS*3  # 1ラウンド（次の1本確定まで）の秒数

# フォント（英語/日本語切替用）
FONT_NAME_EN = "DejaVu Sans"
FONT_NAME_JP = "Noto Sans CJK JP"  # 日本語表示用（例: Meiryo, Noto Sans 等）

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

# -------------------------
# 擬似マーケット（OHLCV生成）
# -------------------------
class MarketSim:
    def __init__(self, start=100.0, seed=77):
        self.rng = np.random.default_rng(seed)
        self.last = start
        self._cum_pv = 0.0
        self._cum_v = 0.0
        self.tick = 0
        self.regime = "normal"  # normal / highvol
        self.regime_len = 0

    def _step_params(self):
        # ボラレジームを時々切替
        self.regime_len += 1
        if self.regime_len > self.rng.integers(40, 180):
            self.regime = "highvol" if self.regime == "normal" else "normal"
            self.regime_len = 0
        sigma = 0.4 if self.regime == "highvol" else 0.15  # 歩幅係数
        shock = self.rng.random() < (0.07 if self.regime == "highvol" else 0.03)
        return sigma, shock

    def next_bar(self, vwap_hint: float, mom_hint: float) -> Bar:
        # 前終値を中心に擬似的な1本を形成
        sigma, shock = self._step_params()
        step_dir = 1 if self.rng.random() < 0.5 + 0.12*np.sign(mom_hint) else -1
        # VWAPからの回帰を弱く
        step_dir += (-1 if self.last > vwap_hint else 1) * 0.2
        step_dir = 1 if step_dir >= 0 else -1

        base_step = self.rng.normal(0.0, sigma)
        if shock:
            base_step += self.rng.normal(0.0, sigma*2.0)

        # 1本の中での上下動を生成
        rng1 = abs(base_step) + self.rng.random()*sigma
        open_ = self.last
        close = max(0.1, open_ + step_dir * rng1)
        high = max(open_, close) + self.rng.random()*sigma*0.8
        low  = min(open_, close) - self.rng.random()*sigma*0.8
        vol = float(self.rng.integers(80, 200))
        self.last = close
        return Bar(open_, high, low, close, vol)

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
        pygame.display.set_caption("ShadowTrader - Candle & Tech MVP")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.use_jp_font = False  # フォント切替フラグ
        self.load_fonts()

        # チャート領域
        self.chart_rect = pygame.Rect(50, 50, 680, 460)
        self.side_rect = pygame.Rect(750, 50, 150, 460)

        # 状態
        self.reset_game()

        # 表示切替
        self.show_sma = True
        self.show_bbands = True
        self.show_vwap = True
        self.show_rsi = True
        self.show_help = True
        self.paused = False

    def reset_game(self):
        self.bars: List[Bar] = []
        self.sim = MarketSim()
        # 初期バーを数本作る
        for _ in range(20):
            vw = self.sim.last
            mom = 0.0
            self.bars.append(self.sim.next_bar(vw, mom))
        self.round_timer = ROUND_FRAMES
        self.round = 1
        self.score = 0.0
        self.streak = 0
        self.pred = None         # "UP" / "DOWN" / "PASS" / None
        self.conf = 0.6

    def load_fonts(self):
        """現在の設定に応じたフォントを読み込む。"""
        name = FONT_NAME_JP if self.use_jp_font else FONT_NAME_EN
        self.font = pygame.font.SysFont(name, 18)
        self.font_small = pygame.font.SysFont(name, 14)
        self.font_big = pygame.font.SysFont(name, 24)
        self.font_big_bold = pygame.font.SysFont(name, 24, bold=True)

    # -------- 進行 --------
    def update(self):
        if self.paused:
            return
        self.round_timer -= 1
        if self.round_timer <= 0:
            self.resolve_round()
            self.round += 1
            self.round_timer = ROUND_FRAMES
            self.pred = None

    def resolve_round(self):
        closes = [b.c for b in self.bars]
        mom_k = pd.Series(closes).pct_change().rolling(5).mean().iloc[-1]
        mom_k = 0.0 if pd.isna(mom_k) else float(mom_k)
        vwap_val = vwap_from_bars(self.bars)[-1]

        new_bar = self.sim.next_bar(vwap_val, mom_k)
        prev_c = self.bars[-1].c
        self.bars.append(new_bar)

        actual = "UP" if new_bar.c > prev_c else "DOWN"
        if self.pred in ("UP", "DOWN"):
            p = self.conf if self.pred == "UP" else 1 - self.conf
            # 対数スコア
            gain = math.log(max(p, 1e-6)) if actual == self.pred else math.log(max(1-p, 1e-6))
            if actual == self.pred:
                self.streak += 1
            else:
                self.streak = 0
            self.score += gain * (1 + 0.1*self.streak)

    # -------- 描画 --------
    def draw(self):
        self.screen.fill(COL_BG)
        self.draw_chart()
        self.draw_sidebar()
        self.draw_footer()

        if self.show_help:
            self.draw_help()

        pygame.display.flip()

    def draw_chart(self):
        pygame.draw.rect(self.screen, COL_GRID, self.chart_rect, 1)

        bars = self.bars[-HISTORY:]
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
            lab = self.font_small.render(f"{price:,.2f}", True, COL_DIM)
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

        closes = [b.c for b in self.bars[-HISTORY:]]
        txt_y = self.side_rect.top + 10

        def put(text, col=COL_WHITE):
            nonlocal txt_y
            self.screen.blit(self.font.render(text, True, col), (self.side_rect.left + 10, txt_y))
            txt_y += 24

        put("Hints", COL_YELLOW)
        # Momentum
        mom_k = pd.Series(closes).pct_change().rolling(5).mean().iloc[-1] if len(closes) >= 6 else 0.0
        mom_k = 0.0 if pd.isna(mom_k) else float(mom_k)
        arrow = "▲" if mom_k > 0 else ("▼" if mom_k < 0 else "→")
        put(f"{arrow} Momentum(5): {mom_k*100:+.2f}%")

        # VWAP 乖離
        vwap_list = vwap_from_bars(self.bars[-HISTORY:])
        if vwap_list:
            vwap_val = vwap_list[-1]
            gap = (closes[-1] / vwap_val - 1.0) if vwap_val else 0.0
            put(f"± VWAP dev: {gap*100:+.2f}%")

        # RSI
        if self.show_rsi:
            rsi_arr = rsi(closes, 14)
            r = rsi_arr[-1] if len(rsi_arr) else float("nan")
            if not math.isnan(r):
                bars = "▁▂▃▄▅▆▇█"
                level = int(max(0, min(7, r/100*7)))
                gauge = bars[level]
                put(f"{gauge} RSI(14): {r:5.1f}")
                # 簡易ヒント
                tip = ""
                if r > 70: tip = "Overbought? ↓"
                elif r < 30: tip = "Oversold? ↑"
                elif abs(mom_k) > 0.005:
                    tip = "Momentum→"
                else:
                    tip = "Neutral"
                put(f"Tip: {tip}", COL_DIM)
            else:
                put("RSI: n/a", COL_DIM)

        put("")
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
        infos = [
            f"Round {self.round}",
            f"Timer {self.round_timer//FPS}s",
            f"Score {self.score:.2f}",
            f"Streak {self.streak}",
            f"Confidence {self.conf:.1f}",
            f"Pred {self.pred}",
        ]
        x = 20
        for t in infos:
            label = self.font.render(t, True, COL_WHITE)
            self.screen.blit(label, (x, base_y))
            x += 150

        controls = "↑UP  ↓DOWN  Space=PASS  ←/→=Confidence  1/2/3/4=Indicators  F=Font  H=Help  P=Pause  R=Reset"
        self.screen.blit(self.font_small.render(controls, True, COL_DIM), (20, base_y+26))

    def draw_help(self):
        # 半透明パネル
        surf = pygame.Surface((WIDTH-120, HEIGHT-160), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 240))  # 背景を濃くして視認性向上
        rect = surf.get_rect()
        rect.topleft = (60, 80)
        self.screen.blit(surf, rect)
        lines = [
            "ShadowTrader チュートリアル（超入門）",
            "",
            "目的: 次の1本で価格が上がるか下がるかを予測してスコアを稼ぐ。",
            "",
            "ローソク足: 緑=上昇, 赤=下落。上ヒゲ/下ヒゲで高値・安値。",
            "テクニカル:",
            "  - SMA20: 20本の平均線。価格がSMAを上抜け/下抜け後はトレンド継続しやすいことがある。",
            "  - Bollinger(20,2): ±2σのバンド。上限付近は反落、下限付近は反発が起こりやすいことがある。",
            "  - VWAP: その日の出来高加重平均。価格が大きく乖離すると戻る“磁力”が働く場面がある。",
            "  - RSI(14): 70超=買われ過ぎ、30未満=売られ過ぎの目安。",
            "",
            "操作: ↑=上, ↓=下, Space=見送り, ←/→=確信度, 1/2/3/4=指標表示切替, F=フォント切替, P=一時停止, R=リセット, H=この画面",
            "スコア: 対数スコア。確信を高く宣言して外すと減点が大きい（現実のリスク管理に近い）。",
            "",
            "戦略ヒント:",
            "  - トレンド強いときはモメンタム寄り、バンド端では逆張りが有利なことがある。",
            "  - VWAP乖離が大きいときは“戻り”を意識。",
            "  - わからないときは PASS で損失回避も有効（ただし得点機会は減る）。",
            "",
            "Hで閉じる。",
        ]
        y = 100
        for i, ln in enumerate(lines):
            font = self.font_big_bold if i == 0 else self.font
            label = font.render(ln, True, COL_WHITE)
            self.screen.blit(label, (80, y))
            y += 26

        pygame.draw.rect(self.screen, COL_WHITE, rect, 2)

    # -------- 入力 --------
    def handle_event(self, e: pygame.event.Event):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_UP:
                self.pred = "UP"
            elif e.key == pygame.K_DOWN:
                self.pred = "DOWN"
            elif e.key == pygame.K_SPACE:
                self.pred = "PASS"
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
            elif e.key == pygame.K_f:
                self.use_jp_font = not self.use_jp_font
                self.load_fonts()
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
