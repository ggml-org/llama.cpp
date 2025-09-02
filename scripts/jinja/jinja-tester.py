import sys
import json
import traceback
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPlainTextEdit, QTextEdit, QPushButton
)
from PySide6.QtGui import QColor, QTextCursor, QTextFormat
from PySide6.QtCore import Qt, QRect, QSize
from jinja2 import Environment, TemplateSyntaxError, TemplateError


# ------------------------
# Line Number Widget
# ------------------------
class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.code_editor = editor

    def sizeHint(self):
        return QSize(self.code_editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.code_editor.line_number_area_paint_event(event)


class CodeEditor(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.line_number_area = LineNumberArea(self)

        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)

        self.update_line_number_area_width(0)
        self.highlight_current_line()

    def line_number_area_width(self):
        digits = len(str(self.blockCount()))
        space = 3 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())

        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))

    def line_number_area_paint_event(self, event):
        from PySide6.QtGui import QPainter
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), Qt.lightGray)

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(Qt.black)
                painter.drawText(0, top, self.line_number_area.width() - 2,
                                 self.fontMetrics().height(),
                                 Qt.AlignRight, number)
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def highlight_current_line(self):
        extra_selections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            line_color = QColor(Qt.yellow).lighter(160)
            selection.format.setBackground(line_color)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)
        self.setExtraSelections(extra_selections)

    def highlight_position(self, lineno: int, col: int, color: QColor):
        block = self.document().findBlockByLineNumber(lineno - 1)
        if block.isValid():
            cursor = QTextCursor(block)
            text = block.text()
            start = block.position() + max(0, col - 1)
            cursor.setPosition(start)
            if col <= len(text):
                cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)

            extra = QTextEdit.ExtraSelection()
            extra.format.setBackground(color.lighter(160))
            extra.cursor = cursor

            self.setExtraSelections(self.extraSelections() + [extra])

    def highlight_line(self, lineno: int, color: QColor):
        block = self.document().findBlockByLineNumber(lineno - 1)
        if block.isValid():
            cursor = QTextCursor(block)
            cursor.select(QTextCursor.LineUnderCursor)

            extra = QTextEdit.ExtraSelection()
            extra.format.setBackground(color.lighter(160))
            extra.cursor = cursor

            self.setExtraSelections(self.extraSelections() + [extra])

    def clear_highlighting(self):
        self.highlight_current_line()


# ------------------------
# Main App
# ------------------------
class JinjaTester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jinja Template Tester")
        self.resize(1200, 800)

        central = QWidget()
        main_layout = QVBoxLayout(central)

        # -------- Top input area --------
        input_layout = QHBoxLayout()

        # Template editor with label
        template_layout = QVBoxLayout()
        template_label = QLabel("Jinja2 Template")
        template_layout.addWidget(template_label)
        self.template_edit = CodeEditor()
        template_layout.addWidget(self.template_edit)
        input_layout.addLayout(template_layout)

        # JSON editor with label
        json_layout = QVBoxLayout()
        json_label = QLabel("Context (JSON)")
        json_layout.addWidget(json_label)
        self.json_edit = CodeEditor()
        json_layout.addWidget(self.json_edit)
        input_layout.addLayout(json_layout)

        main_layout.addLayout(input_layout)

        # -------- Rendered output area --------
        output_label = QLabel("Rendered Output")
        main_layout.addWidget(output_label)
        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        main_layout.addWidget(self.output_edit)

        # -------- Render button and status --------
        btn_layout = QHBoxLayout()
        self.render_btn = QPushButton("Render")
        self.render_btn.clicked.connect(self.render_template)
        btn_layout.addWidget(self.render_btn)
        self.status_label = QLabel("Ready")
        btn_layout.addWidget(self.status_label)
        main_layout.addLayout(btn_layout)

        self.setCentralWidget(central)

    def render_template(self):
        self.template_edit.clear_highlighting()
        self.output_edit.clear()

        template_str = self.template_edit.toPlainText()
        json_str = self.json_edit.toPlainText()

        # Parse JSON context
        try:
            context = json.loads(json_str) if json_str.strip() else {}
        except Exception as e:
            self.status_label.setText(f"❌ JSON Error: {e}")
            return

        env = Environment()
        try:
            template = env.from_string(template_str)
            output = template.render(context)
            self.output_edit.setPlainText(output)
            self.status_label.setText("✅ Render successful")
        except TemplateSyntaxError as e:
            self.status_label.setText(f"❌ Syntax Error (line {e.lineno}): {e.message}")
            if e.lineno:
                self.template_edit.highlight_line(e.lineno, QColor("red"))
        except Exception as e:
            # Catch all runtime errors
            # Try to extract template line number
            lineno = None
            tb = e.__traceback__
            while tb:
                frame = tb.tb_frame
                if frame.f_code.co_filename == "<template>":
                    lineno = tb.tb_lineno
                    break
                tb = tb.tb_next

            error_msg = f"Runtime Error: {type(e).__name__}: {e}"
            if lineno:
                error_msg = f"Runtime Error at line {lineno} in template: {type(e).__name__}: {e}"
                self.template_edit.highlight_line(lineno, QColor("orange"))

            self.output_edit.setPlainText(error_msg)
            self.status_label.setText(f"❌ {error_msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JinjaTester()
    window.show()
    sys.exit(app.exec())
