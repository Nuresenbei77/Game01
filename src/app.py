from __future__ import annotations

from .game.config import ConfigLoader
from .game.core import GameCore
from .ui.screens import ShadowTraderUI


def main() -> None:
    config = ConfigLoader().load()
    core = GameCore(config)
    ui = ShadowTraderUI(core)
    ui.run()


if __name__ == "__main__":
    main()
