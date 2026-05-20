"""Chat-style PyQt6 GUI for the desktop GCSE solver.

Layout::

  +-----------------+----------------------------------------+
  | Chats           | Chat title                             |
  |                 |                                        |
  | [+ New chat]    |  user: Solve x^2 + 5x + 6 = 0          |
  |                 |                                        |
  | • Solve x^2 ... |  assistant:                            |
  |   Solve sin(...)|    [rendered LaTeX answer]             |
  |                 |    verified                            |
  |                 |    ▸ Working / SymPy code              |
  |                 +----------------------------------------+
  | [Delete]        | [Input area...]              [Send]    |
  +-----------------+----------------------------------------+

Model selection is automatic: the filesystem is scanned on startup, the
highest-scoring local GGUF is picked, and the engine loads in the
background. If no model is found, a manual file picker opens.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import scanner
from .chat_store import Chat, ChatStore, Turn
from .latex_io import render_latex_png
from .llm_engine import EngineConfig, LLMEngine
from .solver import SolveResult, solve

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class ScanWorker(QObject):
    """Filesystem walk on a background thread."""
    found = pyqtSignal(object)              # FoundModel
    progress = pyqtSignal(str)              # current directory
    done = pyqtSignal(list, object)         # list[FoundModel], best pick

    def run(self) -> None:
        try:
            models = scanner.scan(
                on_dir=lambda p: self.progress.emit(str(p)),
                on_found=lambda m: self.found.emit(m),
            )
            best = scanner.auto_pick(models)
            self.done.emit(models, best)
        except Exception as e:  # noqa: BLE001
            log.exception("scan crashed")
            self.progress.emit(f"scan error: {e}")
            self.done.emit([], None)


@dataclass
class SolveJob:
    problem: str
    chat_id: str


class SolveWorker(QObject):
    """Owns the LLM engine. Loads on demand; processes solve jobs serially."""
    loaded = pyqtSignal(str)                # human-readable status
    load_failed = pyqtSignal(str)
    solved = pyqtSignal(str, object)        # chat_id, SolveResult
    failed = pyqtSignal(str, str)           # chat_id, error
    log_line = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._engine: Optional[LLMEngine] = None

    def load(self, model_path: str) -> None:
        try:
            self.log_line.emit(f"loading {Path(model_path).name}")
            self._engine = LLMEngine(EngineConfig(model_path=model_path))
            self.loaded.emit(f"ready — {Path(model_path).name}")
        except Exception as e:  # noqa: BLE001
            log.exception("load failed")
            self.load_failed.emit(str(e))

    def solve(self, job: SolveJob) -> None:
        if self._engine is None:
            self.failed.emit(job.chat_id, "engine not loaded")
            return
        try:
            self.log_line.emit(f"solving: {job.problem[:80]}")
            result = solve(self._engine, job.problem)
            self.solved.emit(job.chat_id, result)
        except Exception as e:  # noqa: BLE001
            log.exception("solve failed")
            self.failed.emit(job.chat_id, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Message widgets
# ---------------------------------------------------------------------------

_BUBBLE_USER = "background:#dceaf7;border-radius:10px;padding:8px;"
_BUBBLE_BOT  = "background:#f4f4f4;border-radius:10px;padding:8px;"
_BUBBLE_ERR  = "background:#fde2e2;border-radius:10px;padding:8px;"


class _LatexWidget(QLabel):
    """Renders a LaTeX string as a PNG. Falls back to plain text on error."""

    def __init__(self, latex: str, font_size: int = 24, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if not latex.strip():
            self.setText("(no answer)")
            return
        try:
            png = render_latex_png(latex, font_size=font_size)
            pm = QPixmap()
            pm.loadFromData(png)
            self.setPixmap(pm)
        except Exception as e:  # noqa: BLE001
            log.warning("LaTeX render failed: %s", e)
            self.setText(latex)


class _MessageBubble(QFrame):
    """A single chat bubble. Renders either a user or an assistant turn."""

    def __init__(self, turn: Turn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)

        if turn.role == "user":
            outer.addStretch(1)
            outer.addWidget(self._build_user(turn), 0)
        else:
            outer.addWidget(self._build_assistant(turn), 0)
            outer.addStretch(1)

    def _build_user(self, turn: Turn) -> QWidget:
        box = QFrame()
        box.setStyleSheet(_BUBBLE_USER)
        box.setMaximumWidth(720)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(2)
        v.addWidget(_label("You", bold=True, color="#345"))
        text = QLabel(turn.problem)
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(text)
        return box

    def _build_assistant(self, turn: Turn) -> QWidget:
        is_error = bool(turn.error and not turn.answer)
        box = QFrame()
        box.setStyleSheet(_BUBBLE_ERR if is_error else _BUBBLE_BOT)
        box.setMaximumWidth(820)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        header_row.addWidget(_label("Solver", bold=True, color="#345"))
        header_row.addStretch(1)
        if is_error:
            header_row.addWidget(_pill("ERROR", "#a33"))
        elif turn.verified:
            header_row.addWidget(_pill("verified", "#274"))
        elif turn.answer:
            reason = turn.verify_reason or "no check"
            header_row.addWidget(_pill(f"unverified — {reason}", "#a64"))
        v.addLayout(header_row)

        if is_error:
            err = QLabel(turn.error)
            err.setWordWrap(True)
            v.addWidget(err)
            return box

        latex_to_show = turn.latex or turn.answer
        v.addWidget(_LatexWidget(latex_to_show, font_size=26))

        if turn.answer:
            ans = QLabel(f"answer: {turn.answer}")
            ans.setStyleSheet("color:#456; font-family:monospace;")
            ans.setWordWrap(True)
            ans.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v.addWidget(ans)

        if turn.steps:
            v.addWidget(_collapsible("Working", turn.steps))
        if turn.code:
            v.addWidget(_collapsible("SymPy code", turn.code, mono=True))

        return box


def _label(text: str, *, bold: bool = False, color: Optional[str] = None) -> QLabel:
    lab = QLabel(text)
    style = []
    if bold:
        style.append("font-weight:bold;")
    if color:
        style.append(f"color:{color};")
    if style:
        lab.setStyleSheet("".join(style))
    return lab


def _pill(text: str, color: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(
        f"color:white; background:{color}; padding:2px 8px;"
        f"border-radius:8px; font-size:11px;"
    )
    return lab


def _collapsible(title: str, body: str, *, mono: bool = False) -> QWidget:
    box = QFrame()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    btn = QToolButton()
    btn.setText(f"▸ {title}")
    btn.setStyleSheet("border:none; color:#345; font-weight:bold; text-align:left;")
    body_label = QLabel(body)
    body_label.setWordWrap(True)
    body_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    body_label.setVisible(False)
    if mono:
        body_label.setFont(QFont("Monospace", 10))
        body_label.setStyleSheet(
            "background:#fff; padding:6px; border:1px solid #ddd; font-family:monospace;"
        )
    layout.addWidget(btn)
    layout.addWidget(body_label)

    def _toggle() -> None:
        visible = not body_label.isVisible()
        body_label.setVisible(visible)
        btn.setText(f"{'▾' if visible else '▸'} {title}")

    btn.clicked.connect(_toggle)
    return box


# ---------------------------------------------------------------------------
# Chat view (scrollable list of bubbles)
# ---------------------------------------------------------------------------

class ChatView(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet("background:white;")
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)
        self.setWidget(self._container)
        self._pending_label: Optional[QLabel] = None

    def clear(self) -> None:
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._pending_label = None

    def add_turn(self, turn: Turn) -> None:
        self._dismiss_pending()
        self._layout.insertWidget(self._layout.count() - 1, _MessageBubble(turn))
        self._scroll_to_bottom()

    def add_pending(self, text: str = "thinking…") -> None:
        self._dismiss_pending()
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#888; font-style:italic; padding:8px;")
        self._pending_label = lbl
        self._layout.insertWidget(self._layout.count() - 1, lbl)
        self._scroll_to_bottom()

    def _dismiss_pending(self) -> None:
        if self._pending_label is not None:
            self._pending_label.deleteLater()
            self._pending_label = None

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    request_scan       = pyqtSignal()
    request_load_model = pyqtSignal(str)
    request_solve      = pyqtSignal(object)  # SolveJob

    def __init__(self, *, model_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("GCSE Math Solver — Chat")
        self.resize(QSize(1200, 820))

        self._store = ChatStore()
        self._chat: Chat = self._store.new_chat()
        self._engine_ready = False

        self._build_ui()
        self._start_workers()
        self._refresh_chat_list()
        self._render_chat()

        if model_path:
            self.status.showMessage(f"loading {Path(model_path).name}…")
            self.request_load_model.emit(model_path)
        else:
            self.status.showMessage("scanning filesystem for GGUF models…")
            self.request_scan.emit()

    def _build_ui(self) -> None:
        split = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(split)

        # Sidebar
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(6)
        self.new_btn = QPushButton("+ New chat")
        self.new_btn.clicked.connect(self._on_new_chat)
        side_layout.addWidget(self.new_btn)

        self.chat_list = QListWidget()
        self.chat_list.itemClicked.connect(self._on_chat_picked)
        side_layout.addWidget(self.chat_list, 1)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete_chat)
        side_layout.addWidget(self.delete_btn)

        split.addWidget(side)

        # Right pane
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        self.title_label = QLabel("New chat")
        self.title_label.setStyleSheet("font-size:16px; font-weight:bold; padding:4px;")
        right_layout.addWidget(self.title_label)

        self.chat_view = ChatView()
        right_layout.addWidget(self.chat_view, 1)

        input_row = QHBoxLayout()
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText(
            "Type a GCSE math problem and press Ctrl+Enter (or click Send). "
            "LaTeX in $...$ is supported."
        )
        self.input_edit.setFixedHeight(90)
        self.input_edit.installEventFilter(self)
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedWidth(110)
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.send_btn, 0, Qt.AlignmentFlag.AlignBottom)
        right_layout.addLayout(input_row)

        split.addWidget(right)
        split.setSizes([260, 940])

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.input_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) \
                    and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    def _start_workers(self) -> None:
        self._scan_thread = QThread(self)
        self._scan_worker = ScanWorker()
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.done.connect(self._on_scan_done)
        self.request_scan.connect(self._scan_worker.run)
        self._scan_thread.start()

        self._solve_thread = QThread(self)
        self._solve_worker = SolveWorker()
        self._solve_worker.moveToThread(self._solve_thread)
        self._solve_worker.loaded.connect(self._on_loaded)
        self._solve_worker.load_failed.connect(self._on_load_failed)
        self._solve_worker.solved.connect(self._on_solved)
        self._solve_worker.failed.connect(self._on_solve_failed)
        self._solve_worker.log_line.connect(lambda s: log.info("worker: %s", s))
        self.request_load_model.connect(self._solve_worker.load)
        self.request_solve.connect(self._solve_worker.solve)
        self._solve_thread.start()

    def closeEvent(self, event) -> None:  # noqa: N802
        for thread in (self._scan_thread, self._solve_thread):
            thread.quit()
            thread.wait(2000)
        super().closeEvent(event)

    # -- scan handlers ---------------------------------------------------

    def _on_scan_progress(self, where: str) -> None:
        shown = where if len(where) <= 80 else "…" + where[-79:]
        self.status.showMessage(f"scanning {shown}")

    def _on_scan_done(self, models: list, best) -> None:
        if not models:
            self.status.showMessage("no GGUF models found — please pick one manually")
            picked, _ = QFileDialog.getOpenFileName(
                self, "Choose GGUF model", str(Path.home()), "GGUF (*.gguf)"
            )
            if not picked:
                QMessageBox.critical(self, "No model",
                                     "A GGUF model is required to use the solver.")
                return
            self.request_load_model.emit(picked)
            return

        for m in models:
            log.info("found: %s (%.2f GB)", m.path, m.size_gb)
        if best is None:
            best = models[0]
        self.status.showMessage(
            f"found {len(models)} model(s); loading {best.path.name} ({best.size_gb:.1f} GB)"
        )
        self.request_load_model.emit(str(best.path))

    # -- engine + solve handlers ----------------------------------------

    def _on_loaded(self, status_msg: str) -> None:
        self._engine_ready = True
        self.send_btn.setEnabled(True)
        self.status.showMessage(status_msg)

    def _on_load_failed(self, msg: str) -> None:
        self._engine_ready = False
        self.send_btn.setEnabled(False)
        self.status.showMessage(f"model load failed: {msg}")
        QMessageBox.critical(self, "Load failed", msg)

    def _on_solved(self, chat_id: str, result: SolveResult) -> None:
        # The user might have switched chats mid-solve; only accept results
        # that belong to the chat currently displayed.
        if chat_id != self._chat.id:
            return
        self._chat.append_assistant(result)
        self._store.save(self._chat)
        self._refresh_chat_list()
        self.chat_view.add_turn(self._chat.turns[-1])
        self.send_btn.setEnabled(self._engine_ready)
        if result.error and not result.answer_repr:
            self.status.showMessage(f"error: {result.error}")
        elif result.verified:
            self.status.showMessage(f"verified — {result.answer_repr}")
        else:
            self.status.showMessage(f"unverified — {result.answer_repr}")

    def _on_solve_failed(self, chat_id: str, msg: str) -> None:
        if chat_id != self._chat.id:
            return
        synthetic = SolveResult(problem="", error=msg)
        self._chat.append_assistant(synthetic)
        self._store.save(self._chat)
        self.chat_view.add_turn(self._chat.turns[-1])
        self.send_btn.setEnabled(self._engine_ready)
        self.status.showMessage(f"error: {msg}")

    # -- chat lifecycle --------------------------------------------------

    def _refresh_chat_list(self) -> None:
        self.chat_list.clear()
        chats = self._store.list_chats()
        ids_in_store = {c.id for c in chats}
        ordered: list[Chat] = []
        if self._chat.id not in ids_in_store:
            ordered.append(self._chat)
        ordered.extend(chats)
        for chat in ordered:
            item = QListWidgetItem(chat.title)
            item.setData(Qt.ItemDataRole.UserRole, chat.id)
            if chat.id == self._chat.id:
                item.setSelected(True)
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.chat_list.addItem(item)

    def _render_chat(self) -> None:
        self.title_label.setText(self._chat.title)
        self.chat_view.clear()
        for turn in self._chat.turns:
            self.chat_view.add_turn(turn)

    def _on_new_chat(self) -> None:
        self._chat = self._store.new_chat()
        self._refresh_chat_list()
        self._render_chat()
        self.input_edit.setFocus()

    def _on_chat_picked(self, item: QListWidgetItem) -> None:
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        if chat_id == self._chat.id:
            return
        loaded = self._store.load(chat_id)
        if loaded is None:
            return
        self._chat = loaded
        self._render_chat()
        self._refresh_chat_list()

    def _on_delete_chat(self) -> None:
        if not self._chat.turns:
            return
        confirm = QMessageBox.question(
            self, "Delete chat",
            f"Delete '{self._chat.title}'? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._store.delete(self._chat.id)
        self._chat = self._store.new_chat()
        self._refresh_chat_list()
        self._render_chat()

    # -- input -----------------------------------------------------------

    def _on_send(self) -> None:
        if not self._engine_ready:
            self.status.showMessage("model not ready yet")
            return
        problem = self.input_edit.toPlainText().strip()
        if not problem:
            return
        self.input_edit.clear()
        self._chat.append_user(problem)
        self._store.save(self._chat)
        self._refresh_chat_list()
        self.title_label.setText(self._chat.title)
        self.chat_view.add_turn(self._chat.turns[-1])
        self.chat_view.add_pending("thinking…")
        self.send_btn.setEnabled(False)
        self.request_solve.emit(SolveJob(problem=problem, chat_id=self._chat.id))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="GCSE math solver — chat GUI")
    parser.add_argument("--model", help="Skip the filesystem scan; load this GGUF directly")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv if argv is None else argv)
    window = MainWindow(model_path=args.model)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
