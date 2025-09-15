# shadow_trader.py
# MVP版 ShadowTrader（描画のIndexError修正版）
# プレイヤーは次のローソク足の方向を3秒以内に予測する

import pygame
import sys
import random
import numpy as np
import pandas as pd
from math import log

# -------------------------
# パラメータ
# -------------------------
WIDTH, HEIGHT = 800, 600
FPS = 30
HISTORY_LEN = 60
ROUND_TIME = 90  # フレーム数で約3秒
FONT_NAME = "Arial"

# -------------------------
# 疑似データ生成
# -------------------------
def generate_next_price(prev_price, vwap, mom, shock=False):
    """市場原理を簡略化して方向確率を決定"""
    p_up = 0.5
    # モメンタム効果
    p_up += 0.2 * np.sign(mom)
    # VWAP回帰
    if prev_price > vwap:
        p_up -= 0.1
    else:
        p_up += 0.1
    # ニュースショック
    if shock:
        p_up += random.choice([-0.3, 0.3])
    p_up = np.clip(p_up, 0.05, 0.95)

    step = np.random.choice([1, -1], p=[p_up, 1 - p_up])
    new_price = prev_price + step * random.uniform(0.5, 2.0)
    return new_price, p_up

# -------------------------
# ゲーム本体
# -------------------------
class ShadowTraderGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("ShadowTrader MVP")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT_NAME, 20)

        # チャートデータ
        self.prices = [100.0]
        self.vwap = [100.0]
        self.volumes = [100]

        self.round_timer = ROUND_TIME
        self.round = 1
        self.score = 0.0
        self.streak = 0

        self.prediction = None
        self.confidence = 0.5

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.prediction = "UP"
                    elif event.key == pygame.K_DOWN:
                        self.prediction = "DOWN"
                    elif event.key == pygame.K_SPACE:
                        self.prediction = "PASS"
                    elif event.key == pygame.K_LEFT:
                        self.confidence = max(0.1, self.confidence - 0.1)
                    elif event.key == pygame.K_RIGHT:
                        self.confidence = min(0.9, self.confidence + 0.1)

            # ラウンド進行
            self.round_timer -= 1
            if self.round_timer <= 0:
                self.resolve_round()
                self.round += 1
                self.round_timer = ROUND_TIME
                self.prediction = None

            self.draw()

        pygame.quit()
        sys.exit()

    def resolve_round(self):
        prev_price = self.prices[-1]
        vwap = np.mean(self.prices)
        mom = np.mean(np.diff(self.prices[-5:])) if len(self.prices) > 5 else 0.0
        shock = random.random() < 0.1

        new_price, p_up = generate_next_price(prev_price, vwap, mom, shock)
        self.prices.append(new_price)
        self.vwap.append(np.mean(self.prices))
        self.volumes.append(random.randint(80, 120))

        actual = "UP" if new_price > prev_price else "DOWN"

        if self.prediction in ["UP", "DOWN"]:
            p_declared = self.confidence if self.prediction == "UP" else 1 - self.confidence
            # 対数スコア
            if actual == self.prediction:
                delta = log(max(p_declared, 1e-6))
                self.streak += 1
            else:
                delta = log(max(1 - p_declared, 1e-6))
                self.streak = 0
            self.score += delta * (1 + 0.1 * self.streak)

    def draw(self):
        self.screen.fill((10, 10, 30))

        # チャート枠
        chart_area = pygame.Rect(50, 50, 700, 400)
        pygame.draw.rect(self.screen, (40, 40, 60), chart_area, 1)

        # 可視セグメントをローカル配列として扱う（序盤の短履歴でも安全）
        segment = self.prices[-HISTORY_LEN:]
        if len(segment) >= 2:
            min_val = min(segment)
            max_val = max(segment)
            rng = max(max_val - min_val, 1e-6)  # 値幅0対策
            scale_y = chart_area.height / rng
            base_y = chart_area.bottom
            step_x = chart_area.width / max(len(segment) - 1, 1)

            # 折れ線で価格推移
            for i in range(1, len(segment)):
                x1 = chart_area.left + (i - 1) * step_x
                y1 = base_y - (segment[i - 1] - min_val) * scale_y
                x2 = chart_area.left + i * step_x
                y2 = base_y - (segment[i] - min_val) * scale_y
                pygame.draw.line(self.screen, (0, 200, 0), (x1, y1), (x2, y2), 2)

            # VWAP（移動平均の簡易版）も描く
            vwap_seg = pd.Series(segment).expanding().mean().tolist()
            for i in range(1, len(vwap_seg)):
                x1 = chart_area.left + (i - 1) * step_x
                y1 = base_y - (vwap_seg[i - 1] - min_val) * scale_y
                x2 = chart_area.left + i * step_x
                y2 = base_y - (vwap_seg[i] - min_val) * scale_y
                pygame.draw.line(self.screen, (200, 200, 50), (x1, y1), (x2, y2), 1)

        # テキスト表示
        txts = [
            f"Round: {self.round}",
            f"Timer: {self.round_timer//FPS}",
            f"Score: {self.score:.2f}",
            f"Streak: {self.streak}",
            f"Confidence: {self.confidence:.1f}",
            f"Prediction: {self.prediction}",
            "Keys: Up=UP / Down=DOWN / Space=PASS / Left-Right=Confidence",
        ]
        for i, t in enumerate(txts):
            self.screen.blit(self.font.render(t, True, (255, 255, 255)), (50, 480 + i * 20))

        pygame.display.flip()

# -------------------------
# メイン
# -------------------------
if __name__ == "__main__":
    game = ShadowTraderGame()
    game.run()
