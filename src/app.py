from __future__ import annotations

import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from ui_main import MainWindow


def setup_logging() -> None:
    os.makedirs("outputs", exist_ok=True)
    logging.basicConfig(
        filename=os.path.join("outputs", "app.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    setup_logging()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(980, 860)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
