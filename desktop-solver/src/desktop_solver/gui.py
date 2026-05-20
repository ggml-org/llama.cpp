"""PyQt6 GUI for the desktop GCSE solver.

The window splits into:

  +---------------------------------------------------------------+
  |  Problem (plain text or LaTeX in $...$)                       |
  |  [ ............................................. ]  [ Solve ]|
  +---------------------------------------------------------------+
  |  Answer (rendered LaTeX)                                      |
  |                                                               |
  +---------------------------------------------------------------+
  |  Working (steps)            |  SymPy code & verification       |
  |                             |                                  |
  +-----------------------------+----------------------------------+
  |  Log                                                          |
  +---------------------------------------------------------------+

Solving runs on a worker thread so the GUI stays responsive while llama.cpp
chews on the prompt.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .latex_io import render_latex_png
from .llm_engine import EngineConfig, LLMEngine
from .solver import SolveResult, solve

log = logging.getLogger(__name__)


# --- worker -----------------------------------------------------------------

@dataclass
class _SolveJob:
    problem: str


class _SolveWorker(QObject):
    """Owns the LLM engine and processes one job at a time on its thread."""

    finished = pyqtSignal(object)     # SolveResult
    failed   = pyqtSignal(str)        # error message
    log_line = pyqtSignal(str)

    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self._config = config
        self._engine: LLMEngine | None = None

    def load(self) -> None:
        try:
            self.log_line.emit(f"loading model: {self._config.model_path}")
            self._engine = LLMEngine(self._config)
            self.log_line.emit("model ready")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"failed to load model: {e}")

    def solve(self, job: _SolveJob) -> None:
        if self._engine is None:
            self.failed.emit("engine not loaded")
            return
        try:
            self.log_line.emit(f"solving: {job.problem[:80]}")
            result = solve(self._engine, job.problem)
            self.finished.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"solve failed: {type(e).__name__}: {e}")


# --- main window ------------------------------------------------------------

class _LatexLabel(QLabel):
    """QLabel that holds a rendered LaTeX image and re-renders on demand."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(80)
        self.setStyleSheet("background-color: #fafafa; border: 1px solid #ddd;")
        self.setText("(no answer yet)")

    def show_latex(self, latex: str) -> None:
        if not latex.strip():
            self.setPixmap(QPixmap())
            self.setText("(no LaTeX)")
            return
        try:
            png = render_latex_png(latex, font_size=28)
            pm = QPixmap()
            pm.loadFromData(png)
            self.setPixmap(pm)
            self.setText("")
        except Exception as e:  # noqa: BLE001
            self.setText(f"LaTeX render error: {e}\n\n{latex}")


class MainWindow(QMainWindow):
    request_load = pyqtSignal()
    request_solve = pyqtSignal(object)  # _SolveJob

    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self.setWindowTitle("GCSE Math Solver — Desktop")
        self.resize(QSize(1100, 800))
        self._config = config
        self._build_ui()
        self._start_worker()

    # -- UI assembly ---------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Problem row
        prob_row = QHBoxLayout()
        self.problem_edit = QPlainTextEdit()
        self.problem_edit.setPlaceholderText(
            "Enter a GCSE math problem. LaTeX in $...$ is supported.\n"
            "Examples:\n"
            "  Solve x^2 + 5x + 6 = 0\n"
            "  $\\int_0^1 (2x + 3)\\,dx$\n"
            "  The probability of rain is 0.3. Find P(no rain in 5 days)."
        )
        self.problem_edit.setMaximumHeight(110)
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.setFixedWidth(120)
        self.solve_btn.setEnabled(False)
        self.solve_btn.clicked.connect(self._on_solve_clicked)
        prob_row.addWidget(self.problem_edit, 1)
        prob_row.addWidget(self.solve_btn, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(prob_row)

        # Answer (rendered LaTeX)
        self.answer_label = _LatexLabel()
        answer_scroll = QScrollArea()
        answer_scroll.setWidgetResizable(True)
        answer_scroll.setWidget(self.answer_label)
        answer_scroll.setMinimumHeight(140)
        root.addWidget(self._titled("Answer", answer_scroll))

        # Working + code split
        split = QSplitter(Qt.Orientation.Horizontal)
        self.steps_view = QTextEdit()
        self.steps_view.setReadOnly(True)
        self.steps_view.setFont(QFont("Sans", 11))
        self.code_view = QTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setFont(QFont("Monospace", 10))
        split.addWidget(self._titled("Working", self.steps_view))
        split.addWidget(self._titled("SymPy code + verification", self.code_view))
        split.setSizes([500, 500])
        root.addWidget(split, 1)

        # Log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        self.log_view.setFont(QFont("Monospace", 9))
        root.addWidget(self._titled("Log", self.log_view))

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Loading model — please wait")

    def _titled(self, title: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(label)
        layout.addWidget(widget)
        return container

    # -- worker thread management -------------------------------------------

    def _start_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = _SolveWorker(self._config)
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._on_solve_finished)
        self._worker.failed.connect(self._on_solve_failed)
        self._worker.log_line.connect(self._append_log)
        self.request_load.connect(self._worker.load)
        self.request_solve.connect(self._worker.solve)
        self._thread.start()
        # Defer model load so the window paints first.
        self.request_load.emit()
        # Re-enable solve button as soon as the load completes.
        self._worker.log_line.connect(self._maybe_enable_solve)

    def _maybe_enable_solve(self, line: str) -> None:
        if line == "model ready":
            self.solve_btn.setEnabled(True)
            self.status.showMessage("Ready")

    # -- handlers ------------------------------------------------------------

    def _on_solve_clicked(self) -> None:
        problem = self.problem_edit.toPlainText().strip()
        if not problem:
            QMessageBox.information(self, "No input", "Please enter a problem.")
            return
        self.solve_btn.setEnabled(False)
        self.status.showMessage("Solving…")
        self.answer_label.show_latex("")
        self.steps_view.clear()
        self.code_view.clear()
        self.request_solve.emit(_SolveJob(problem=problem))

    def _on_solve_finished(self, result: SolveResult) -> None:
        self.solve_btn.setEnabled(True)
        if not result.success:
            self.status.showMessage(f"Error: {result.error}")
            self.answer_label.show_latex(r"\text{error}")
            self.code_view.setPlainText(result.code or "(no code)")
            self.steps_view.setPlainText(result.error)
            return

        self.answer_label.show_latex(result.latex or result.answer_repr)
        verified_tag = "VERIFIED" if result.verified else f"UNVERIFIED ({result.verify_reason})"
        self.status.showMessage(f"{verified_tag} — {result.answer_repr}")
        self.steps_view.setPlainText(result.steps or "(no steps provided)")
        code_text = (
            f"# answer: {result.answer_repr}\n"
            f"# {verified_tag}\n\n"
            f"{result.code}\n"
        )
        self.code_view.setPlainText(code_text)
        self._append_log(f"done: {result.answer_repr}  [{verified_tag}]")

    def _on_solve_failed(self, msg: str) -> None:
        self.solve_btn.setEnabled(True)
        self.status.showMessage(f"Error: {msg}")
        self._append_log(f"ERROR: {msg}")

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    # -- lifecycle -----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)


# --- entry ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="GCSE math solver — desktop GUI")
    parser.add_argument("--model", required=False, help="Path to GGUF model file")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window")
    parser.add_argument("--n-threads", type=int, default=None, help="Threads for inference")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="Layers to offload to GPU")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv if argv is None else argv)

    model_path = args.model
    if not model_path:
        picked, _ = QFileDialog.getOpenFileName(
            None, "Choose GGUF model", str(Path.home()), "GGUF (*.gguf)"
        )
        if not picked:
            QMessageBox.critical(None, "No model", "A GGUF model is required.")
            return 1
        model_path = picked

    config = EngineConfig(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
