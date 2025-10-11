from __future__ import annotations

import math
from typing import List, Tuple

import pygame
import pygame.freetype

from ..game.core import GameCore
from . import theme


class ShadowTraderUI:
    def __init__(self, core: GameCore) -> None:
        self.core = core
        pygame.init()
        pygame.freetype.init()
        pygame.display.set_caption("ShadowTrader - Candle & Tech MVP")
        self.screen = pygame.display.set_mode((core.config.width, core.config.height))
        self.clock = pygame.time.Clock()
        self.load_fonts()

        self.chart_rect = pygame.Rect(50, 50, 760, 420)
        self.chart_5m_rect = pygame.Rect(50, 480, 760, 110)
        self.tape_rect = pygame.Rect(50, 600, 760, 80)
        self.side_rect = pygame.Rect(840, 50, 220, 520)

        self.show_sma = True
        self.show_bbands = True
        self.show_vwap = True
        self.show_rsi = True
        self.show_help = True

    def load_fonts(self) -> None:
        font_path = None
        fallback_name = None
        bundled_error = None

        if theme.FONT_BUNDLED.exists():
            try:
                pygame.freetype.Font(str(theme.FONT_BUNDLED), 18)
                font_path = str(theme.FONT_BUNDLED)
            except (OSError, FileNotFoundError, pygame.error) as exc:
                bundled_error = exc
        else:
            bundled_error = FileNotFoundError(str(theme.FONT_BUNDLED))

        if font_path is None:
            for name in theme.FONT_CANDIDATES:
                try:
                    pygame.freetype.Font(name, 18)
                    fallback_name = name
                    break
                except (OSError, FileNotFoundError, pygame.error):
                    continue
        if font_path is None and fallback_name is None:
            raise RuntimeError("Unable to load any Japanese-capable font") from bundled_error

        def make(size: int, bold: bool = False) -> pygame.freetype.Font:
            if font_path:
                font = pygame.freetype.Font(font_path, size)
            else:
                font = pygame.freetype.SysFont(fallback_name, size)
            font.pad = True
            font.strong = bold
            return font

        self.font = make(18)
        self.font_small = make(15)
        self.font_tiny = make(13)
        self.font_big = make(26)
        self.font_big_bold = make(28, bold=True)
        self.font_footer = make(16)

    def draw(self) -> None:
        self.screen.fill(theme.COL_BG)
        self.draw_chart()
        self.draw_5m_chart()
        self.draw_tick_tape()
        self.draw_sidebar()
        self.draw_footer()
        if self.core.result_flash_timer > 0 and self.core.last_result:
            self.draw_result_banner()
        if self.core.phase == "question":
            self.draw_question_overlay()
        elif self.core.phase == "post":
            self.draw_post_overlay()
        if self.show_help:
            self.draw_help()
        pygame.display.flip()

    def draw_chart(self) -> None:
        pygame.draw.rect(self.screen, theme.COL_GRID, self.chart_rect, 1)
        bars = self.core.get_chart_bars()
        if not bars:
            return

        highs = [b.h for b in bars]
        lows = [b.l for b in bars]
        min_p = min(lows)
        max_p = max(highs)
        rng = max(max_p - min_p, 1e-6)
        px_w = self.chart_rect.width / max(len(bars), 1)

        for y_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = self.chart_rect.top + self.chart_rect.height * (1 - y_frac)
            pygame.draw.line(
                self.screen,
                (28, 32, 50),
                (self.chart_rect.left, y),
                (self.chart_rect.right, y),
                1,
            )
            price = min_p + rng * y_frac
            lab, _ = self.font_small.render(f"{price:,.2f}", theme.COL_DIM)
            self.screen.blit(lab, (self.chart_rect.right + 8, y - 8))

        if self.show_sma:
            sma20 = self.core.get_sma(20)
            self._draw_line_indicator(sma20, min_p, rng, theme.COL_YELLOW, px_w, len(bars))

        if self.show_vwap:
            vwap_list = self.core.get_vwap()
            self._draw_line_indicator(vwap_list, min_p, rng, theme.COL_PURPLE, px_w, len(bars), width=2)

        if self.show_bbands:
            _, upper, lower = self.core.get_bbands(20, 2.0)
            self._draw_line_indicator(upper, min_p, rng, theme.COL_BLUE, px_w, len(bars))
            self._draw_line_indicator(lower, min_p, rng, theme.COL_BLUE, px_w, len(bars))

        for i, bar in enumerate(bars):
            x = self.chart_rect.left + i * px_w
            cx = x + px_w * 0.5

            def Y(p: float) -> float:
                return self.chart_rect.bottom - (p - min_p) / rng * self.chart_rect.height

            y_o, y_h, y_l, y_c = map(Y, (bar.o, bar.h, bar.l, bar.c))
            col = theme.COL_RED if bar.c >= bar.o else theme.COL_GREEN
            pygame.draw.line(self.screen, col, (cx, y_h), (cx, y_l), 1)
            body_top = min(y_o, y_c)
            body_h = max(2, abs(y_c - y_o))
            pygame.draw.rect(self.screen, col, pygame.Rect(x + px_w * 0.15, body_top, px_w * 0.7, body_h))

    def draw_5m_chart(self) -> None:
        pygame.draw.rect(self.screen, theme.COL_GRID, self.chart_5m_rect, 1)
        bars = self.core.five_min_bars[-60:]
        if not bars:
            return
        highs = [b.h for b in bars]
        lows = [b.l for b in bars]
        min_p = min(lows)
        max_p = max(highs)
        rng = max(max_p - min_p, 1e-6)
        px_w = self.chart_5m_rect.width / max(len(bars), 1)
        for i, bar in enumerate(bars):
            x = self.chart_5m_rect.left + i * px_w
            cx = x + px_w * 0.5

            def Y(p: float) -> float:
                return self.chart_5m_rect.bottom - (p - min_p) / rng * self.chart_5m_rect.height

            y_h = Y(bar.h)
            y_l = Y(bar.l)
            y_o = Y(bar.o)
            y_c = Y(bar.c)
            col = theme.COL_RED if bar.c >= bar.o else theme.COL_GREEN
            pygame.draw.line(self.screen, col, (cx, y_h), (cx, y_l), 1)
            body_top = min(y_o, y_c)
            body_h = max(2, abs(y_c - y_o))
            pygame.draw.rect(self.screen, col, pygame.Rect(x + px_w * 0.3, body_top, px_w * 0.4, body_h))
        label, _ = self.font_small.render("5分足", theme.COL_DIM)
        self.screen.blit(label, (self.chart_5m_rect.left + 6, self.chart_5m_rect.top + 4))

    def draw_tick_tape(self) -> None:
        pygame.draw.rect(self.screen, theme.COL_GRID, self.tape_rect, 1)
        if not self.core.tick_tape:
            return
        x = self.tape_rect.left + 10
        y = self.tape_rect.top + 10
        step = 18
        max_entries = 24
        for tick in list(self.core.tick_tape)[:max_entries]:
            col = theme.COL_RED if tick.side == "BUY" else theme.COL_GREEN
            text = f"{tick.price:,.2f} ({int(tick.volume)})"
            surf, _ = self.font_small.render(text, col)
            self.screen.blit(surf, (x, y))
            y += step
            if y > self.tape_rect.bottom - 20:
                x += 150
                y = self.tape_rect.top + 10
                if x > self.tape_rect.right - 140:
                    break
        label, _ = self.font_small.render("歩み値 (最新→上)", theme.COL_DIM)
        self.screen.blit(label, (self.tape_rect.left + 6, self.tape_rect.top + 4))

    def draw_result_banner(self) -> None:
        if self.core.result_flash_timer > 0:
            alpha = max(120, min(220, int(255 * (self.core.result_flash_timer / (self.core.config.fps * 3 + 1)))))
        else:
            alpha = 150
        surf = pygame.Surface((self.core.config.width - 100, 36), pygame.SRCALPHA)
        surf.fill((0, 0, 0, alpha))
        rect = surf.get_rect()
        rect.center = (self.core.config.width // 2, 28)
        self.screen.blit(surf, rect)
        if self.core.last_result:
            label, _ = self.font_big.render(self.core.last_result, theme.COL_YELLOW)
            self.screen.blit(label, (rect.left + 20, rect.top + 6))

    def _wrap_lines(self, text: str, font_obj: pygame.freetype.Font, max_width: int) -> List[str]:
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
                surf, _ = font_obj.render(candidate, theme.COL_WHITE)
                if surf.get_width() <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = ch
            if current:
                lines.append(current)
        return lines

    def draw_question_overlay(self) -> None:
        panel = pygame.Surface(self.side_rect.size, pygame.SRCALPHA)
        panel.fill((8, 12, 24, 225))
        pygame.draw.rect(panel, (120, 140, 200, 140), panel.get_rect(), 1, border_radius=10)

        margin = 16
        inner_width = panel.get_width() - margin * 2
        y = margin

        title, _ = self.font_big_bold.render("テクニカル注目ポイント", theme.COL_YELLOW)
        panel.blit(title, (margin, y))
        y += title.get_height() + 10

        bullet_color = theme.COL_WHITE
        for reason in self.core.question_reason.split(" / "):
            bullet = "・" if reason else ""
            wrapped = self._wrap_lines(reason, self.font_small, inner_width - 18) if reason else [""]
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
            surf, _ = self.font.render(line, theme.COL_WHITE)
            panel.blit(surf, (margin, y))
            y += surf.get_height() + 6

        conf_text = f"現在の確信度: {self.core.conf:.1f}"
        conf_surf, conf_rect = self.font_small.render(conf_text, theme.COL_BLUE)
        panel.blit(conf_surf, (margin, y))
        y += conf_rect.height + 10

        choices = [
            ("UP", "上昇する", pygame.K_UP),
            ("DOWN", "下落する", pygame.K_DOWN),
            ("RANGE", "もみ合う", pygame.K_SPACE),
        ]
        choice_box_w = inner_width
        choice_colors = {
            "UP": theme.COL_RED,
            "DOWN": theme.COL_GREEN,
            "RANGE": theme.COL_BLUE,
        }
        for key, text, _ in choices:
            selected = key == self.core.pred_choice
            marker = "▶" if selected else "  "
            label, label_rect = self.font_big.render(f"{marker} {text}", theme.COL_WHITE)
            line_height = label_rect.height
            if selected:
                highlight = pygame.Surface((choice_box_w, line_height + 8), pygame.SRCALPHA)
                sel_col = choice_colors.get(key, theme.COL_YELLOW)
                highlight.fill((*sel_col, 70))
                panel.blit(highlight, (margin - 4, y - 4))
                label, _ = self.font_big.render(f"{marker} {text}", sel_col)
            panel.blit(label, (margin, y))
            y += line_height + 12

        hint = "↑/↓/Spaceで選択  ←/→で確信度  Enterで回答"
        hint_surf, hint_rect = self.font_small.render(hint, theme.COL_YELLOW)
        hint_bg = pygame.Surface((inner_width, hint_rect.height + 10), pygame.SRCALPHA)
        hint_bg.fill((255, 255, 255, 30))
        hint_y = panel.get_height() - margin - hint_rect.height - 6
        panel.blit(hint_bg, (margin, hint_y - 4))
        panel.blit(hint_surf, (margin + 4, hint_y))

        self.screen.blit(panel, self.side_rect)

    def draw_post_overlay(self) -> None:
        panel = pygame.Surface(self.side_rect.size, pygame.SRCALPHA)
        panel.fill((10, 16, 30, 235))
        pygame.draw.rect(panel, (200, 180, 80, 140), panel.get_rect(), 1, border_radius=10)

        margin = 16
        inner_width = panel.get_width() - margin * 2
        y = margin

        if self.core.post_summary_ready and self.core.post_round_summary:
            title_text = "ラウンド結果サマリー (5分先)"
            sections = [line.strip() for line in self.core.post_round_summary.split("\n")]
        else:
            title_text = "結果観察フェーズ (5分先)"
            remaining = max(0, math.ceil(self.core.phase_timer / self.core.config.fps))
            sections = [
                "5分先の値動きを高速で確定しています。",
                f"残り {remaining}s",
            ]

        title_surf, _ = self.font_big_bold.render(title_text, theme.COL_YELLOW)
        panel.blit(title_surf, (margin, y))
        y += title_surf.get_height() + 12

        info_color = theme.COL_WHITE
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
        last_result_text = self.core.last_result or "結果集計中"
        result_lines = self._wrap_lines(last_result_text, self.font_small, inner_width)
        for line in result_lines:
            surf, _ = self.font_small.render(line, theme.COL_BLUE)
            panel.blit(surf, (margin, y))
            y += surf.get_height() + 4

        choice_display = self.core.pred_choice or "-"
        choice_text = f"回答: {choice_display} / 確信度 {self.core.conf:.1f}"
        choice_surf, choice_rect = self.font_small.render(choice_text, theme.COL_DIM)
        panel.blit(choice_surf, (margin, panel.get_height() - margin - choice_rect.height))

        self.screen.blit(panel, self.side_rect)

    def _draw_line_indicator(
        self, arr: List[float], min_p: float, rng: float, color, px_w: float, count: int, width: int = 1
    ) -> None:
        pts = []
        for i, v in enumerate(arr[-count:]):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            x = self.chart_rect.left + i * px_w
            y = self.chart_rect.bottom - (v - min_p) / rng * self.chart_rect.height
            pts.append((x, y))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, width)

    def draw_sidebar(self) -> None:
        pygame.draw.rect(self.screen, (20, 24, 36), self.side_rect)
        pygame.draw.rect(self.screen, theme.COL_GRID, self.side_rect, 1)

        closes = [b.c for b in self.core.minute_bars[-self.core.config.history_length :]]
        txt_y = self.side_rect.top + 10

        def put(text: str, col=theme.COL_WHITE) -> None:
            nonlocal txt_y
            if text is None:
                txt_y += 24
                return
            surf, _ = self.font.render(text, col)
            self.screen.blit(surf, (self.side_rect.left + 10, txt_y))
            txt_y += 24

        put("テクニカル概要", theme.COL_YELLOW)
        if closes:
            if len(closes) >= 6:
                recent = closes[-6:]
                base = recent[0]
                mom_k = (recent[-1] / base - 1.0) if base else 0.0
            else:
                mom_k = 0.0
            arrow = "▲" if mom_k > 0 else ("▼" if mom_k < 0 else "→")
            put(f"{arrow} Momentum(5): {mom_k * 100:+.2f}%")
            vwap_list = self.core.get_vwap()
            if vwap_list:
                vwap_val = vwap_list[-1]
                gap = (closes[-1] / vwap_val - 1.0) if vwap_val else 0.0
                put(f"± VWAP 乖離: {gap * 100:+.2f}%")
            if self.show_rsi:
                rsi_arr = self.core.get_rsi(14)
                r = rsi_arr[-1] if rsi_arr else float("nan")
                if not math.isnan(r):
                    bars_txt = "▁▂▃▄▅▆▇█"
                    level = int(max(0, min(7, r / 100 * 7)))
                    put(f"{bars_txt[level]} RSI(14): {r:5.1f}")
        else:
            put("データ準備中", theme.COL_DIM)

        put("")
        put("板情報", theme.COL_YELLOW)

        book = self.core.order_book
        show_market_rows = self.core.mode in ("preopen", "close")
        bids = list(book.bids)
        asks = list(book.asks)
        ask_levels = list(reversed(asks[: self.core.config.board_levels]))
        bid_levels = bids[: self.core.config.board_levels]
        while len(ask_levels) < self.core.config.board_levels:
            ask_levels.append((None, None))
        while len(bid_levels) < self.core.config.board_levels:
            bid_levels.append((None, None))

        ask_total = int(sum(vol for _, vol in asks[: self.core.config.board_levels])) if asks else 0
        bid_total = int(sum(vol for _, vol in bids[: self.core.config.board_levels])) if bids else 0
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
        pygame.draw.rect(self.screen, theme.COL_GRID, table_rect, 1)

        price_col_width = int(table_width * 0.55)
        col_split = table_rect.left + price_col_width

        color_map = {
            "header": theme.COL_DIM,
            "ask": theme.COL_GREEN,
            "ask_total": theme.COL_GREEN,
            "market_sell": theme.COL_GREEN,
            "indicative_sell": theme.COL_GREEN,
            "bid": theme.COL_RED,
            "bid_total": theme.COL_RED,
            "market_buy": theme.COL_RED,
            "indicative_buy": theme.COL_RED,
            "indicative_mid": theme.COL_YELLOW,
        }

        bg_map = {
            "header": (26, 30, 46),
            "ask": (32, 24, 28),
            "ask_total": (40, 30, 42),
            "indicative_sell": (30, 32, 42),
            "indicative_mid": (34, 34, 46),
            "bid": (24, 32, 28),
            "bid_total": (40, 30, 42),
            "market_sell": (30, 28, 40),
            "market_buy": (28, 40, 32),
        }

        for idx, (row_type, price_text, qty_text) in enumerate(rows):
            cell_rect = pygame.Rect(table_rect.left, table_rect.top + idx * row_height, table_width, row_height)
            bg = bg_map.get(row_type)
            if bg:
                pygame.draw.rect(self.screen, bg, cell_rect)
            pygame.draw.line(self.screen, theme.COL_GRID, (cell_rect.left, cell_rect.bottom), (cell_rect.right, cell_rect.bottom), 1)
            color = color_map.get(row_type, theme.COL_WHITE)
            label_price, _ = self.font_small.render(price_text, color)
            label_qty, _ = self.font_small.render(qty_text, color)
            self.screen.blit(label_price, (cell_rect.left + 6, cell_rect.top + 4))
            self.screen.blit(label_qty, (col_split + 6, cell_rect.top + 4))

        txt_y = table_rect.bottom + 16
        put("イベントログ", theme.COL_YELLOW)
        for msg, timer in list(self.core.event_log):
            label, _ = self.font_small.render(f"{msg} ({timer//self.core.config.fps}s)", theme.COL_DIM)
            self.screen.blit(label, (self.side_rect.left + 12, txt_y))
            txt_y += 18

    def draw_footer(self) -> None:
        footer_rect = pygame.Rect(50, self.core.config.height - 60, self.core.config.width - 100, 44)
        pygame.draw.rect(self.screen, (18, 22, 34), footer_rect)
        pygame.draw.rect(self.screen, theme.COL_GRID, footer_rect, 1)

        margin_x = 12
        available_width = footer_rect.width - margin_x * 2
        mode_label = self.core.mode_labels.get(self.core.mode, self.core.mode)
        phase_label = self.core.phase
        timer_text = f"残 {self.core.phase_timer / self.core.config.fps:.1f}s" if self.core.phase == "stream" else ""
        infos = [
            f"Round {self.core.round}",
            f"Mode: {mode_label}",
            f"Phase: {phase_label}",
            f"Timer {timer_text}",
            f"Score {self.core.score:.2f}",
            f"Streak {self.core.streak}",
            f"確信度 {self.core.conf:.1f}",
            f"選択 {self.core.pred_choice or '-'}",
        ]
        info_spacing = 18
        rows: List[List[Tuple[pygame.Surface, pygame.Rect]]] = []
        current_row: List[Tuple[pygame.Surface, pygame.Rect]] = []
        current_width = 0

        for text in infos:
            label, label_rect = self.font_footer.render(text, theme.COL_WHITE)
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
                surf, _ = self.font_small.render(candidate, theme.COL_DIM)
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
        control_surfs = [self.font_small.render(line, theme.COL_DIM)[0] for line in control_lines]

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

    def draw_help(self) -> None:
        surf = pygame.Surface((self.core.config.width - 120, self.core.config.height - 160), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 240))
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
            label, _ = font.render(ln, theme.COL_WHITE)
            self.screen.blit(label, (80, y))
            y += 26
        pygame.draw.rect(self.screen, theme.COL_WHITE, rect, 2)

    def handle_event(self, e: pygame.event.Event) -> None:
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_UP:
                if self.core.phase == "question":
                    self.core.pred_choice = "UP"
            elif e.key == pygame.K_DOWN:
                if self.core.phase == "question":
                    self.core.pred_choice = "DOWN"
            elif e.key == pygame.K_SPACE:
                if self.core.phase == "question":
                    self.core.pred_choice = "RANGE"
            elif e.key == pygame.K_RETURN:
                if self.core.phase == "question" and self.core.pred_choice is not None:
                    self.core.start_post_phase()
                elif self.core.phase == "post":
                    self.core.start_stream_phase()
            elif e.key == pygame.K_LEFT:
                self.core.adjust_confidence(-0.1)
            elif e.key == pygame.K_RIGHT:
                self.core.adjust_confidence(0.1)
            elif e.key == pygame.K_1:
                self.show_sma = not self.show_sma
            elif e.key == pygame.K_2:
                self.show_bbands = not self.show_bbands
            elif e.key == pygame.K_3:
                self.show_vwap = not self.show_vwap
            elif e.key == pygame.K_4:
                self.show_rsi = not self.show_rsi
            elif e.key == pygame.K_m:
                self.core.cycle_mode()
            elif e.key == pygame.K_h:
                self.show_help = not self.show_help
            elif e.key == pygame.K_r:
                self.core.reset_game()
            elif e.key == pygame.K_p:
                self.core.toggle_pause()

    def run(self) -> None:
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                else:
                    self.handle_event(e)
            self.core.update()
            self.draw()
            self.clock.tick(self.core.config.fps)
        pygame.quit()


__all__ = ["ShadowTraderUI"]
