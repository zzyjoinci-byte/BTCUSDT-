from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import logging

from PySide6.QtCore import QObject, Signal


@dataclass
class ProgressDetail:
    phase: str
    percent: int
    message: str = ""
    counts: Dict[str, Any] | None = None


class AppState(QObject):
    log_signal = Signal(str)
    status_signal = Signal(str)
    phase_signal = Signal(str)
    progress_signal = Signal(int)
    detail_signal = Signal(dict)
    result_signal = Signal(dict)
    error_signal = Signal(str)

    def log(self, text: str) -> None:
        logging.getLogger("app").info(text)
        self.log_signal.emit(text)

    def set_status(self, text: str) -> None:
        logging.getLogger("app").info(text)
        self.status_signal.emit(text)

    def set_phase(self, text: str) -> None:
        self.phase_signal.emit(text)

    def set_progress(self, percent: int) -> None:
        self.progress_signal.emit(int(percent))

    def update_detail(self, detail: Dict[str, Any]) -> None:
        self.detail_signal.emit(detail)

    def push_result(self, payload: Dict[str, Any]) -> None:
        self.result_signal.emit(payload)

    def error(self, text: str) -> None:
        logging.getLogger("app").error(text)
        self.error_signal.emit(text)
