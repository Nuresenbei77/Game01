from __future__ import annotations

MODE_CYCLE = ["preopen", "morning", "afternoon", "close"]

MODE_LABELS = {
    "preopen": "寄り付き前: 気配とギャップを意識",
    "morning": "前場: トレンドが走りやすい",
    "afternoon": "後場: もみ合い・反転注視",
    "close": "引け直前: 需給と引け成行",
}

DEFAULT_MODE = MODE_CYCLE[0]

__all__ = ["MODE_CYCLE", "MODE_LABELS", "DEFAULT_MODE"]
