#!/usr/bin/env python3
import sys
import json
import argparse
import jinja2.ext as jinja2_ext
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QTextEdit,
    QPushButton,
    QFileDialog,
)
from PySide6.QtGui import QColor, QColorConstants, QTextCursor, QTextFormat
from PySide6.QtCore import Qt, QRect, QSize
from jinja2 import TemplateSyntaxError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from datetime import datetime


def format_template_content(template_content):
    """Format the Jinja template content using Jinja2's lexer."""
    if not template_content.strip():
        return template_content

    env = ImmutableSandboxedEnvironment()
    tc_rstrip = template_content.rstrip()
    tokens = list(env.lex(tc_rstrip))
    result = ""
    indent_level = 0
    i = 0

    while i < len(tokens):
        token = tokens[i]
        _, token_type, token_value = token

        if token_type == "block_begin":
            block_start = i
            # Collect all tokens for this block construct
            construct_content = token_value
            end_token_type = token_type.replace("_begin", "_end")
            j = i + 1
            while j < len(tokens) and tokens[j][1] != end_token_type:
                construct_content += tokens[j][2]
                j += 1

            if j < len(tokens):  # Found the end token
                construct_content += tokens[j][2]
                i = j  # Skip to the end token

                # Check for control structure keywords for indentation
                stripped_content = construct_content.strip()
                instr = block_start + 1
                while tokens[instr][1] == "whitespace":
                    instr = instr + 1

                instruction_token = tokens[instr][2]
                start_control_tokens = ["if", "for", "macro", "call", "block"]
                end_control_tokens = ["end" + t for t in start_control_tokens]
                is_control_start = any(
                    instruction_token.startswith(kw) for kw in start_control_tokens
                )
                is_control_end = any(
                    instruction_token.startswith(kw) for kw in end_control_tokens
                )

                # Adjust indentation for control structures
                # For control end blocks, decrease indent BEFORE adding the content
                if is_control_end:
                    indent_level = max(0, indent_level - 1)

                # Remove all previous whitespace before this block
                result = result.rstrip()

                # Add proper indent, but only if this is not the first token
                added_newline = False
                if result:  # Only add newline and indent if there's already content
                    result += (
                        "\n" + "  " * indent_level
                    )  # Use 2 spaces per indent level
                    added_newline = True
                else:  # For the first token, don't add any indent
                    result += ""

                # Add the block content
                result += stripped_content

                # Add '-' after '%' if it wasn't there and we added a newline or indent
                if (
                    added_newline
                    and stripped_content.startswith("{%")
                    and not stripped_content.startswith("{%-")
                ):
                    # Add '-' at the beginning
                    result = (
                        result[: result.rfind("{%")]
                        + "{%-"
                        + result[result.rfind("{%") + 2 :]
                    )
                if stripped_content.endswith("%}") and not stripped_content.endswith(
                    "-%}"
                ):
                    # Only add '-' if this is not the last token or if there's content after
                    if i + 1 < len(tokens) and tokens[i + 1][1] != "eof":
                        result = result[:-2] + "-%}"

                # For control start blocks, increase indent AFTER adding the content
                if is_control_start:
                    indent_level += 1
            else:
                # Malformed template, just add the token
                result += token_value
        elif token_type == "variable_begin":
            # Collect all tokens for this variable construct
            construct_content = token_value
            end_token_type = token_type.replace("_begin", "_end")
            j = i + 1
            while j < len(tokens) and tokens[j][1] != end_token_type:
                construct_content += tokens[j][2]
                j += 1

            if j < len(tokens):  # Found the end token
                construct_content += tokens[j][2]
                i = j  # Skip to the end token

                # For variable constructs, leave them alone
                # Do not add indent or whitespace before or after them
                result += construct_content
            else:
                # Malformed template, just add the token
                result += token_value
        elif token_type == "data":
            # Handle data (text between Jinja constructs)
            # For data content, preserve it as is
            result += token_value
        else:
            # Handle any other tokens
            result += token_value

        i += 1

    # Clean up trailing newlines and spaces
    result = result.rstrip()

    # Copy the newline / space count from the original
    if (trailing_length := len(template_content) - len(tc_rstrip)):
        result += template_content[-trailing_length:]

    return result


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
            self.line_number_area.update(
                0, rect.y(), self.line_number_area.width(), rect.height()
            )

        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event):
        from PySide6.QtGui import QPainter

        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColorConstants.LightGray)

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(
            self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        )
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColorConstants.Black)
                painter.drawText(
                    0,
                    top,
                    self.line_number_area.width() - 2,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    number,
                )
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def highlight_current_line(self):
        extra_selections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            line_color = QColorConstants.Yellow.lighter(160)
            selection.format.setBackground(line_color)  # pyright: ignore[reportAttributeAccessIssue]
            selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)  # pyright: ignore[reportAttributeAccessIssue]
            selection.cursor = self.textCursor()  # pyright: ignore[reportAttributeAccessIssue]
            selection.cursor.clearSelection()  # pyright: ignore[reportAttributeAccessIssue]
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
                cursor.movePosition(
                    QTextCursor.MoveOperation.NextCharacter,
                    QTextCursor.MoveMode.KeepAnchor,
                )

            extra = QTextEdit.ExtraSelection()
            extra.format.setBackground(color.lighter(160))  # pyright: ignore[reportAttributeAccessIssue]
            extra.cursor = cursor  # pyright: ignore[reportAttributeAccessIssue]

            self.setExtraSelections(self.extraSelections() + [extra])

    def highlight_line(self, lineno: int, color: QColor):
        block = self.document().findBlockByLineNumber(lineno - 1)
        if block.isValid():
            cursor = QTextCursor(block)
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)

            extra = QTextEdit.ExtraSelection()
            extra.format.setBackground(color.lighter(160))  # pyright: ignore[reportAttributeAccessIssue]
            extra.cursor = cursor  # pyright: ignore[reportAttributeAccessIssue]

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
        self.json_edit.setPlainText("""
{
    "add_generation_prompt": true,
    "bos_token": "",
    "eos_token": "",
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of Poland?"
        }
    ]
}
        """.strip())
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

        # Load template button
        self.load_btn = QPushButton("Load Template")
        self.load_btn.clicked.connect(self.load_template)
        btn_layout.addWidget(self.load_btn)

        # Format template button
        self.format_btn = QPushButton("Format")
        self.format_btn.clicked.connect(self.format_template)
        btn_layout.addWidget(self.format_btn)

        self.render_btn = QPushButton("Render")
        self.render_btn.clicked.connect(self.render_template)
        btn_layout.addWidget(self.render_btn)
        main_layout.addLayout(btn_layout)

        # Status label below buttons
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

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

        def raise_exception(text: str) -> str:
            raise RuntimeError(text)

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2_ext.loopcontrols],
        )
        env.filters["tojson"] = (
            lambda x,
            indent=None,
            separators=None,
            sort_keys=False,
            ensure_ascii=False: json.dumps(
                x,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
                ensure_ascii=ensure_ascii,
            )
        )
        env.globals["strftime_now"] = lambda format: datetime.now().strftime(format)
        env.globals["raise_exception"] = raise_exception
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

    def load_template(self):
        """Load a Jinja template from a file using a file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Jinja Template",
            "",
            "Template Files (*.jinja *.j2 *.html *.txt);;All Files (*)",
        )

        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    self.template_edit.setPlainText(content)
                    self.status_label.setText(f"✅ Loaded template from {file_path}")
            except Exception as e:
                self.status_label.setText(f"❌ Error loading file: {str(e)}")

    def format_template(self):
        """Format the Jinja template using Jinja2's lexer for proper parsing."""
        try:
            template_content = self.template_edit.toPlainText()
            if not template_content.strip():
                self.status_label.setText("⚠️ Template is empty")
                return

            formatted_content = format_template_content(template_content)
            self.template_edit.setPlainText(formatted_content)
            self.status_label.setText("✅ Template formatted")
        except Exception as e:
            self.status_label.setText(f"❌ Error formatting template: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode
        parser = argparse.ArgumentParser(description="Jinja Template Tester")
        parser.add_argument(
            "--template", required=True, help="Path to Jinja template file"
        )
        parser.add_argument("--context", required=True, help="JSON string for context")
        parser.add_argument(
            "--action",
            choices=["format", "render"],
            default="render",
            help="Action to perform",
        )
        args = parser.parse_args()

        # Load template
        with open(args.template, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Load JSON
        context = json.loads(args.context)
        # Add missing variables
        context.setdefault("bos_token", "")
        context.setdefault("eos_token", "")
        context.setdefault("add_generation_prompt", False)

        env = ImmutableSandboxedEnvironment()

        if args.action == "format":
            formatted = format_template_content(template_content)
            print(formatted) # noqa: NP100
        elif args.action == "render":
            template = env.from_string(template_content)
            output = template.render(context)
            print(output) # noqa: NP100

    else:
        # GUI mode
        app = QApplication(sys.argv)
        window = JinjaTester()
        window.show()
        sys.exit(app.exec())
