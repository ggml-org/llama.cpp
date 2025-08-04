#!/usr/bin/env python3
from __future__ import annotations

import logging
import argparse
import os
import sys
import numpy
import enum
from pathlib import Path
from typing import Any, Optional, Tuple, Type
import warnings

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QComboBox, QMessageBox, QTabWidget,
    QTextEdit, QFormLayout,
    QHeaderView, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType, ReaderField
from gguf.constants import TokenType, RopeScalingType, PoolingType, GGMLQuantizationType

logger = logging.getLogger("gguf-editor-gui")

# Map of key names to enum types for automatic enum interpretation
KEY_TO_ENUM_TYPE = {
    gguf.Keys.Tokenizer.TOKEN_TYPE: TokenType,
    gguf.Keys.Rope.SCALING_TYPE: RopeScalingType,
    gguf.Keys.LLM.POOLING_TYPE: PoolingType,
    gguf.Keys.General.FILE_TYPE: GGMLQuantizationType,
}

# Define the tokenizer keys that should be edited together
TOKENIZER_LINKED_KEYS = [
    gguf.Keys.Tokenizer.LIST,
    gguf.Keys.Tokenizer.TOKEN_TYPE,
    gguf.Keys.Tokenizer.SCORES
]


class TokenizerEditorDialog(QDialog):
    def __init__(self, tokens, token_types, scores, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Tokenizer Data")
        self.resize(900, 600)

        self.tokens = tokens.copy() if tokens else []
        self.token_types = token_types.copy() if token_types else []
        self.scores = scores.copy() if scores else []

        # Ensure all arrays have the same length
        max_len = max(len(self.tokens), len(self.token_types), len(self.scores))
        if len(self.tokens) < max_len:
            self.tokens.extend([""] * (max_len - len(self.tokens)))
        if len(self.token_types) < max_len:
            self.token_types.extend([0] * (max_len - len(self.token_types)))
        if len(self.scores) < max_len:
            self.scores.extend([0.0] * (max_len - len(self.scores)))

        layout = QVBoxLayout(self)

        # Add filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Type to filter tokens...")
        self.filter_edit.textChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_edit)

        # Add page controls
        self.page_size = 100  # Show 100 items per page
        self.current_page = 0
        self.total_pages = max(1, (len(self.tokens) + self.page_size - 1) // self.page_size)

        self.page_label = QLabel(f"Page 1 of {self.total_pages}")
        filter_layout.addWidget(self.page_label)

        prev_page = QPushButton("Previous")
        prev_page.clicked.connect(self.previous_page)
        filter_layout.addWidget(prev_page)

        next_page = QPushButton("Next")
        next_page.clicked.connect(self.next_page)
        filter_layout.addWidget(next_page)

        layout.addLayout(filter_layout)

        # Tokenizer data table
        self.tokens_table = QTableWidget()
        self.tokens_table.setColumnCount(4)
        self.tokens_table.setHorizontalHeaderLabels(["Index", "Token", "Type", "Score"])
        self.tokens_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tokens_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tokens_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tokens_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.tokens_table)

        # Controls
        controls_layout = QHBoxLayout()

        add_button = QPushButton("Add Token")
        add_button.clicked.connect(self.add_token)
        controls_layout.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected)
        controls_layout.addWidget(remove_button)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initialize the filtered values
        self.filtered_indices = list(range(len(self.tokens)))

        # Load data for the first page
        self.load_page()

    def apply_filter(self):
        """Filter the tokens based on the search text."""
        filter_text = self.filter_edit.text().lower()

        if not filter_text:
            # No filter, show all values
            self.filtered_indices = list(range(len(self.tokens)))
        else:
            # Apply filter
            self.filtered_indices = []
            for i, token in enumerate(self.tokens):
                if filter_text in str(token).lower():
                    self.filtered_indices.append(i)

        # Reset to first page and reload
        self.total_pages = max(1, (len(self.filtered_indices) + self.page_size - 1) // self.page_size)
        self.current_page = 0
        self.page_label.setText(f"Page 1 of {self.total_pages}")
        self.load_page()

    def previous_page(self):
        """Go to the previous page of results."""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_label.setText(f"Page {self.current_page + 1} of {self.total_pages}")
            self.load_page()

    def next_page(self):
        """Go to the next page of results."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.page_label.setText(f"Page {self.current_page + 1} of {self.total_pages}")
            self.load_page()

    def load_page(self):
        """Load the current page of tokenizer data."""
        self.tokens_table.setRowCount(0)  # Clear the table

        # Calculate start and end indices for the current page
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.filtered_indices))

        # Pre-allocate rows for better performance
        self.tokens_table.setRowCount(end_idx - start_idx)

        for row, i in enumerate(range(start_idx, end_idx)):
            orig_idx = self.filtered_indices[i]

            # Index
            index_item = QTableWidgetItem(str(orig_idx))
            index_item.setData(Qt.ItemDataRole.UserRole, orig_idx)  # Store original index
            index_item.setFlags(index_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tokens_table.setItem(row, 0, index_item)

            # Token
            token_item = QTableWidgetItem(str(self.tokens[orig_idx]))
            self.tokens_table.setItem(row, 1, token_item)

            # Token Type
            token_type = self.token_types[orig_idx] if orig_idx < len(self.token_types) else 0
            try:
                enum_val = TokenType(token_type)
                display_text = f"{enum_val.name} ({token_type})"
            except (ValueError, KeyError):
                display_text = f"Unknown ({token_type})"

            type_item = QTableWidgetItem(display_text)
            type_item.setData(Qt.ItemDataRole.UserRole, token_type)

            # Make type cell editable with a double-click handler
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tokens_table.setItem(row, 2, type_item)

            # Score
            score = self.scores[orig_idx] if orig_idx < len(self.scores) else 0.0
            score_item = QTableWidgetItem(str(score))
            self.tokens_table.setItem(row, 3, score_item)

        # Connect double-click handler for token type cells
        self.tokens_table.cellDoubleClicked.connect(self.handle_cell_double_click)

    def handle_cell_double_click(self, row, column):
        """Handle double-click on a cell, specifically for token type editing."""
        if column == 2:  # Token Type column
            orig_item = self.tokens_table.item(row, 0)
            if orig_item:
                orig_idx = orig_item.data(Qt.ItemDataRole.UserRole)
                self.edit_token_type(row, orig_idx)

    def edit_token_type(self, row, orig_idx):
        """Edit a token type using a dialog with a dropdown of all enum options."""
        current_value = self.token_types[orig_idx] if orig_idx < len(self.token_types) else 0

        # Create a dialog with enum options
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Token Type")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        for enum_val in TokenType:
            combo.addItem(f"{enum_val.name} ({enum_val.value})", enum_val.value)

        # Set current value
        try:
            if isinstance(current_value, int):
                enum_val = TokenType(current_value)
                combo.setCurrentText(f"{enum_val.name} ({current_value})")
        except (ValueError, KeyError):
            pass

        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get the selected value
            new_value = combo.currentData()
            enum_val = TokenType(new_value)
            display_text = f"{enum_val.name} ({new_value})"

            # Update the display
            type_item = self.tokens_table.item(row, 2)
            if type_item:
                type_item.setText(display_text)
                type_item.setData(Qt.ItemDataRole.UserRole, new_value)

            # Update the actual value
            self.token_types[orig_idx] = new_value

    def add_token(self):
        """Add a new token to the end of the list."""
        # Add to the end of the arrays
        self.tokens.append("")
        self.token_types.append(0)  # Default to normal token
        self.scores.append(0.0)

        orig_idx = len(self.tokens) - 1

        # Add to filtered indices if it matches the current filter
        filter_text = self.filter_edit.text().lower()
        if not filter_text or filter_text in "":
            self.filtered_indices.append(orig_idx)

        # Update pagination
        self.total_pages = max(1, (len(self.filtered_indices) + self.page_size - 1) // self.page_size)

        # Go to the last page to show the new item
        self.current_page = self.total_pages - 1
        self.page_label.setText(f"Page {self.current_page + 1} of {self.total_pages}")

        # Reload the page
        self.load_page()

    def remove_selected(self):
        """Remove selected tokens from all arrays."""
        selected_rows = []
        for item in self.tokens_table.selectedItems():
            row = item.row()
            if row not in selected_rows:
                selected_rows.append(row)

        if not selected_rows:
            return

        # Get original indices in descending order to avoid index shifting
        orig_indices = []
        for row in selected_rows:
            orig_item = self.tokens_table.item(row, 0)
            if orig_item:
                orig_indices.append(orig_item.data(Qt.ItemDataRole.UserRole))
        orig_indices.sort(reverse=True)

        # Remove from all arrays
        for idx in orig_indices:
            if idx < len(self.tokens):
                del self.tokens[idx]
            if idx < len(self.token_types):
                del self.token_types[idx]
            if idx < len(self.scores):
                del self.scores[idx]

        # Rebuild filtered_indices
        self.filtered_indices = []
        filter_text = self.filter_edit.text().lower()

        for i, token in enumerate(self.tokens):
            if not filter_text or filter_text in str(token).lower():
                self.filtered_indices.append(i)

        # Update pagination
        self.total_pages = max(1, (len(self.filtered_indices) + self.page_size - 1) // self.page_size)
        self.current_page = min(self.current_page, self.total_pages - 1)
        self.page_label.setText(f"Page {self.current_page + 1} of {self.total_pages}")

        # Reload the page
        self.load_page()

    def get_data(self):
        """Return the edited tokenizer data."""
        return self.tokens, self.token_types, self.scores


class ArrayEditorDialog(QDialog):
    def __init__(self, array_values, element_type, key=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Array Values")
        self.resize(700, 500)

        self.array_values = array_values
        self.element_type = element_type
        self.key = key

        # Get enum type for this array if applicable
        self.enum_type = None
        if key in KEY_TO_ENUM_TYPE and element_type == GGUFValueType.INT32:
            self.enum_type = KEY_TO_ENUM_TYPE[key]

        layout = QVBoxLayout(self)

        # Add enum type information if applicable
        if self.enum_type is not None:
            enum_info_layout = QHBoxLayout()
            enum_label = QLabel(f"Editing {self.enum_type.__name__} values:")
            enum_info_layout.addWidget(enum_label)

            # Add a legend for the enum values
            enum_values = ", ".join([f"{e.name}={e.value}" for e in self.enum_type])
            enum_values_label = QLabel(f"Available values: {enum_values}")
            enum_values_label.setWordWrap(True)
            enum_info_layout.addWidget(enum_values_label, 1)

            layout.addLayout(enum_info_layout)

        # Add search/filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Type to filter values...")
        self.filter_edit.textChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_edit)

        # Add page controls for large arrays
        self.page_size = 100  # Show 100 items per page
        self.current_page = 0
        self.total_pages = max(1, (len(array_values) + self.page_size - 1) // self.page_size)

        self.page_label = QLabel(f"Page 1 of {self.total_pages}")
        filter_layout.addWidget(self.page_label)

        prev_page = QPushButton("Previous")
        prev_page.clicked.connect(self.previous_page)
        filter_layout.addWidget(prev_page)

        next_page = QPushButton("Next")
        next_page.clicked.connect(self.next_page)
        filter_layout.addWidget(next_page)

        layout.addLayout(filter_layout)

        # Array items table
        self.items_table = QTableWidget()

        # Set up columns based on whether we have an enum type
        if self.enum_type is not None:
            self.items_table.setColumnCount(3)
            self.items_table.setHorizontalHeaderLabels(["Index", "Value", "Actions"])
            self.items_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.items_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self.items_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        else:
            self.items_table.setColumnCount(2)
            self.items_table.setHorizontalHeaderLabels(["Index", "Value"])
            self.items_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.items_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.items_table)

        # Controls
        controls_layout = QHBoxLayout()

        add_button = QPushButton("Add Item")
        add_button.clicked.connect(self.add_item)
        controls_layout.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected)
        controls_layout.addWidget(remove_button)

        # Add bulk edit button for enum arrays
        if self.enum_type is not None:
            bulk_edit_button = QPushButton("Bulk Edit Selected")
            bulk_edit_button.clicked.connect(self.bulk_edit_selected)
            controls_layout.addWidget(bulk_edit_button)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initialize the filtered values
        self.filtered_indices = list(range(len(self.array_values)))

        # Load array values for the first page
        self.load_page()

    def apply_filter(self):
        """Filter the array values based on the search text."""
        filter_text = self.filter_edit.text().lower()

        if not filter_text:
            # No filter, show all values
            self.filtered_indices = list(range(len(self.array_values)))
        else:
            # Apply filter
            self.filtered_indices = []
            for i, value in enumerate(self.array_values):
                # For enum values, search in both name and value
                if self.enum_type is not None and isinstance(value, int):
                    try:
                        enum_val = self.enum_type(value)
                        display_text = f"{enum_val.name} ({value})".lower()
                        if filter_text in display_text:
                            self.filtered_indices.append(i)
                    except (ValueError, KeyError):
                        # If not a valid enum value, just check the raw value
                        if filter_text in str(value).lower():
                            self.filtered_indices.append(i)
                else:
                    # For non-enum values, just check the string representation
                    if filter_text in str(value).lower():
                        self.filtered_indices.append(i)

        # Reset to first page and reload
        self.total_pages = max(1, (len(self.filtered_indices) + self.page_size - 1) // self.page_size)
        self.current_page = 0
        self.page_label.setText(f"Page 1 of {self.total_pages}")
        self.load_page()

    def previous_page(self):
        """Go to the previous page of results."""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_label.setText(f"Page {self.current_page + 1} of {self.total_pages}")
            self.load_page()

    def next_page(self):
        """Go to the next page of results."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.page_label.setText(f"Page {self.current_page + 1} of {self.total_pages}")
            self.load_page()

    def load_page(self):
        """Load the current page of array values."""
        self.items_table.setRowCount(0)  # Clear the table

        # Calculate start and end indices for the current page
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.filtered_indices))

        # Pre-allocate rows for better performance
        self.items_table.setRowCount(end_idx - start_idx)

        for row, i in enumerate(range(start_idx, end_idx)):
            orig_idx = self.filtered_indices[i]
            value = self.array_values[orig_idx]

            # Index
            index_item = QTableWidgetItem(str(orig_idx))
            index_item.setData(Qt.ItemDataRole.UserRole, orig_idx)  # Store original index
            index_item.setFlags(index_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.items_table.setItem(row, 0, index_item)

            # Value
            if self.enum_type is not None:
                # Display enum value and name
                try:
                    if isinstance(value, (int, numpy.signedinteger)):
                        enum_val = self.enum_type(value)
                        display_text = f"{enum_val.name} ({value})"
                    else:
                        display_text = str(value)
                except (ValueError, KeyError):
                    display_text = f"Unknown ({value})"

                # Store the enum value in the item
                value_item = QTableWidgetItem(display_text)
                value_item.setData(Qt.ItemDataRole.UserRole, value)
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.items_table.setItem(row, 1, value_item)

                # Add an edit button in a separate column
                edit_button = QPushButton("Edit")
                edit_button.setProperty("row", row)
                edit_button.clicked.connect(self.edit_array_enum_value)

                # Create a widget to hold the button
                button_widget = QWidget()
                button_layout = QHBoxLayout(button_widget)
                button_layout.setContentsMargins(2, 2, 2, 2)
                button_layout.addWidget(edit_button)
                button_layout.addStretch()

                self.items_table.setCellWidget(row, 2, button_widget)
            else:
                value_item = QTableWidgetItem(str(value))
                self.items_table.setItem(row, 1, value_item)

    def edit_array_enum_value(self):
        """Handle editing an enum value in the array editor."""
        button = self.sender()
        row = button.property("row")

        # Get the original index from the table item
        orig_item = self.items_table.item(row, 0)
        new_item = self.items_table.item(row, 1)
        if orig_item and new_item and self.enum_type and self.edit_enum_value(row, self.enum_type):
            orig_idx = orig_item.data(Qt.ItemDataRole.UserRole)
            new_value = new_item.data(Qt.ItemDataRole.UserRole)
            # Update the stored value in the array
            if isinstance(new_value, (int, float, str, bool)):
                self.array_values[orig_idx] = new_value

    def bulk_edit_selected(self):
        """Edit multiple enum values at once."""
        if not self.enum_type:
            return

        selected_rows = set()
        for item in self.items_table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select at least one row to edit.")
            return

        # Create a dialog with enum options
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Bulk Edit {self.enum_type.__name__} Values")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel(f"Set {len(selected_rows)} selected items to:"))

        combo = QComboBox()
        for enum_val in self.enum_type:
            combo.addItem(f"{enum_val.name} ({enum_val.value})", enum_val.value)

        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get the selected value
            new_value = combo.currentData()
            enum_val = self.enum_type(new_value)
            display_text = f"{enum_val.name} ({new_value})"

            # Update all selected rows
            for row in selected_rows:
                orig_item = self.items_table.item(row, 0)
                new_item = self.items_table.item(row, 1)
                if orig_item and new_item:
                    orig_idx = orig_item.data(Qt.ItemDataRole.UserRole)
                    self.array_values[orig_idx] = new_value

                    # Update the display
                    new_item.setText(display_text)
                    new_item.setData(Qt.ItemDataRole.UserRole, new_value)

    def add_item(self):
        # Add to the end of the array
        orig_idx = len(self.array_values)

        # Add default value based on type
        if self.enum_type is not None:
            # Default to first enum value
            default_value = list(self.enum_type)[0].value
            self.array_values.append(default_value)
        else:
            if self.element_type == GGUFValueType.STRING:
                self.array_values.append("")
            else:
                self.array_values.append(0)

        # Add to filtered indices if it matches the current filter
        self.filtered_indices.append(orig_idx)

        # Update pagination
        self.total_pages = max(1, (len(self.filtered_indices) + self.page_size - 1) // self.page_size)

        # Go to the last page to show the new item
        self.current_page = self.total_pages - 1
        self.page_label.setText(f"Page {self.current_page + 1} of {self.total_pages}")

        # Reload the page
        self.load_page()

    def remove_selected(self):
        selected_rows = []
        for item in self.items_table.selectedItems():
            row = item.row()
            if row not in selected_rows:
                selected_rows.append(row)

        if not selected_rows:
            return

        # Get original indices in descending order to avoid index shifting
        orig_indices = list()
        for row in selected_rows:
            orig_item = self.items_table.item(row, 0)
            if orig_item:
                orig_indices.append(orig_item.data(Qt.ItemDataRole.UserRole))
        orig_indices.sort(reverse=True)

        # Remove from array_values
        for idx in orig_indices:
            del self.array_values[idx]

        # Rebuild filtered_indices
        self.filtered_indices = []
        filter_text = self.filter_edit.text().lower()

        for i, value in enumerate(self.array_values):
            if not filter_text:
                self.filtered_indices.append(i)
            else:
                # Apply filter
                if self.enum_type is not None and isinstance(value, int):
                    try:
                        enum_val = self.enum_type(value)
                        display_text = f"{enum_val.name} ({value})".lower()
                        if filter_text in display_text:
                            self.filtered_indices.append(i)
                    except (ValueError, KeyError):
                        if filter_text in str(value).lower():
                            self.filtered_indices.append(i)
                else:
                    if filter_text in str(value).lower():
                        self.filtered_indices.append(i)

        # Update pagination
        self.total_pages = max(1, (len(self.filtered_indices) + self.page_size - 1) // self.page_size)
        self.current_page = min(self.current_page, self.total_pages - 1)
        self.page_label.setText(f"Page {self.current_page + 1} of {self.total_pages}")

        # Reload the page
        self.load_page()

    def edit_enum_value(self, row: int, enum_type: Type[enum.Enum]):
        """Edit an enum value using a dialog with a dropdown of all enum options."""
        # Get the original index from the table item
        orig_item = self.items_table.item(row, 0)
        if orig_item:
            orig_idx = orig_item.data(Qt.ItemDataRole.UserRole)
        else:
            return
        current_value = self.array_values[orig_idx]

        # Create a dialog with enum options
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select {enum_type.__name__} Value")
        layout = QVBoxLayout(dialog)

        # Add description
        description = QLabel(f"Select a {enum_type.__name__} value:")
        layout.addWidget(description)

        # Use a combo box for quick selection
        combo = QComboBox()
        for enum_val in enum_type:
            combo.addItem(f"{enum_val.name} ({enum_val.value})", enum_val.value)

        # Set current value
        try:
            if isinstance(current_value, int):
                enum_val = enum_type(current_value)
                combo.setCurrentText(f"{enum_val.name} ({current_value})")
        except (ValueError, KeyError):
            pass

        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Update the value display and stored data
            new_value = combo.currentData()
            enum_val = enum_type(new_value)
            display_text = f"{enum_val.name} ({new_value})"

            new_item = self.items_table.item(row, 1)
            if new_item:
                new_item.setText(display_text)
                new_item.setData(Qt.ItemDataRole.UserRole, new_value)

            # Update the actual array value
            self.array_values[orig_idx] = new_value
            return True
        return False

    def get_array_values(self):
        # The array_values list is kept up-to-date as edits are made
        return self.array_values


class AddMetadataDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Metadata")
        self.resize(400, 200)

        layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        self.key_edit = QLineEdit()
        form_layout.addRow("Key:", self.key_edit)

        self.type_combo = QComboBox()
        for value_type in GGUFValueType:
            if value_type != GGUFValueType.ARRAY:  # Skip array type for simplicity
                self.type_combo.addItem(value_type.name, value_type)
        form_layout.addRow("Type:", self.type_combo)

        self.value_edit = QTextEdit()
        form_layout.addRow("Value:", self.value_edit)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> Tuple[str, GGUFValueType, Any]:
        key = self.key_edit.text()
        value_type = self.type_combo.currentData()
        value_text = self.value_edit.toPlainText()

        # Convert value based on type
        if value_type == GGUFValueType.UINT8:
            value = np.uint8(int(value_text))
        elif value_type == GGUFValueType.INT8:
            value = np.int8(int(value_text))
        elif value_type == GGUFValueType.UINT16:
            value = np.uint16(int(value_text))
        elif value_type == GGUFValueType.INT16:
            value = np.int16(int(value_text))
        elif value_type == GGUFValueType.UINT32:
            value = np.uint32(int(value_text))
        elif value_type == GGUFValueType.INT32:
            value = np.int32(int(value_text))
        elif value_type == GGUFValueType.FLOAT32:
            value = np.float32(float(value_text))
        elif value_type == GGUFValueType.BOOL:
            value = value_text.lower() in ('true', 'yes', '1')
        elif value_type == GGUFValueType.STRING:
            value = value_text
        else:
            value = value_text

        return key, value_type, value


class GGUFEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GGUF Editor")
        self.resize(1000, 800)

        self.current_file = None
        self.reader = None
        self.modified = False
        self.metadata_changes = {}  # Store changes to apply when saving
        self.metadata_to_remove = set()  # Store keys to remove when saving
        self.on_metadata_changed_is_connected = False

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # File controls
        file_layout = QHBoxLayout()

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        file_layout.addWidget(self.file_path_edit)

        open_button = QPushButton("Open GGUF")
        open_button.clicked.connect(self.open_file)
        file_layout.addWidget(open_button)

        save_button = QPushButton("Save As...")
        save_button.clicked.connect(self.save_file)
        file_layout.addWidget(save_button)

        main_layout.addLayout(file_layout)

        # Tabs for different views
        self.tabs = QTabWidget()

        # Metadata tab
        self.metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(self.metadata_tab)

        # Metadata table
        self.metadata_table = QTableWidget()
        self.metadata_table.setColumnCount(4)
        self.metadata_table.setHorizontalHeaderLabels(["Key", "Type", "Value", "Actions"])
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.metadata_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.metadata_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.metadata_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        metadata_layout.addWidget(self.metadata_table)

        # Metadata controls
        metadata_controls = QHBoxLayout()

        add_metadata_button = QPushButton("Add Metadata")
        add_metadata_button.clicked.connect(self.add_metadata)
        metadata_controls.addWidget(add_metadata_button)

        metadata_controls.addStretch()

        metadata_layout.addLayout(metadata_controls)

        # Tensors tab
        self.tensors_tab = QWidget()
        tensors_layout = QVBoxLayout(self.tensors_tab)

        self.tensors_table = QTableWidget()
        self.tensors_table.setColumnCount(5)
        self.tensors_table.setHorizontalHeaderLabels(["Name", "Type", "Shape", "Elements", "Size (bytes)"])
        self.tensors_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tensors_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tensors_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tensors_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.tensors_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        tensors_layout.addWidget(self.tensors_table)

        # Add tabs to tab widget
        self.tabs.addTab(self.metadata_tab, "Metadata")
        self.tabs.addTab(self.tensors_tab, "Tensors")

        main_layout.addWidget(self.tabs)

        # Status bar
        self.statusBar().showMessage("Ready")

    def load_file(self, file_path):
        """Load a GGUF file by path"""
        try:
            self.statusBar().showMessage(f"Loading {file_path}...")
            QApplication.processEvents()

            self.reader = GGUFReader(file_path, 'r')
            self.current_file = file_path
            self.file_path_edit.setText(file_path)

            self.load_metadata()
            self.load_tensors()

            self.metadata_changes = {}
            self.metadata_to_remove = set()
            self.modified = False

            self.statusBar().showMessage(f"Loaded {file_path}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")
            self.statusBar().showMessage("Error loading file")
            return False

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open GGUF File", "", "GGUF Files (*.gguf);;All Files (*)"
        )

        if not file_path:
            return

        self.load_file(file_path)

    def load_metadata(self):
        self.metadata_table.setRowCount(0)

        if not self.reader:
            return

        # Disconnect to prevent triggering during loading
        if self.on_metadata_changed_is_connected:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.metadata_table.itemChanged.disconnect(self.on_metadata_changed)
            self.on_metadata_changed_is_connected = False

        for i, (key, field) in enumerate(self.reader.fields.items()):
            self.metadata_table.insertRow(i)

            # Key
            key_item = QTableWidgetItem(key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.metadata_table.setItem(i, 0, key_item)

            # Type
            if not field.types:
                type_str = "N/A"
            elif field.types[0] == GGUFValueType.ARRAY:
                nest_count = len(field.types) - 1
                element_type = field.types[-1].name
                # Check if this is an enum array
                enum_type = self.get_enum_for_key(key)
                if enum_type is not None and field.types[-1] == GGUFValueType.INT32:
                    element_type = enum_type.__name__
                type_str = '[' * nest_count + element_type + ']' * nest_count
            else:
                type_str = str(field.types[0].name)
                # Check if this is an enum field
                enum_type = self.get_enum_for_key(key)
                if enum_type is not None and field.types[0] == GGUFValueType.INT32:
                    type_str = enum_type.__name__

            type_item = QTableWidgetItem(type_str)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.metadata_table.setItem(i, 1, type_item)

            # Value
            value_str = self.format_field_value(field)
            value_item = QTableWidgetItem(value_str)

            # Make only simple values editable
            if len(field.types) == 1 and field.types[0] != GGUFValueType.ARRAY:
                value_item.setFlags(value_item.flags() | Qt.ItemFlag.ItemIsEditable)
            else:
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.metadata_table.setItem(i, 2, value_item)

            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)

            # Add Edit button for arrays and enum fields
            if field.types and field.types[0] == GGUFValueType.ARRAY:
                edit_button = QPushButton("Edit")
                edit_button.setProperty("row", i)
                edit_button.setProperty("key", key)
                edit_button.clicked.connect(self.edit_array_metadata)
                actions_layout.addWidget(edit_button)

                # Add special label for tokenizer linked fields
                if key in TOKENIZER_LINKED_KEYS:
                    edit_button.setText("Edit Tokenizer")
                    edit_button.setToolTip("Edit all tokenizer data together")
            elif len(field.types) == 1 and self.get_enum_for_key(key) is not None:
                edit_button = QPushButton("Edit")
                edit_button.setProperty("row", i)
                edit_button.setProperty("key", key)
                edit_button.clicked.connect(self.edit_metadata_enum)
                actions_layout.addWidget(edit_button)

            remove_button = QPushButton("Remove")
            remove_button.setProperty("row", i)
            remove_button.setProperty("key", key)
            remove_button.clicked.connect(self.remove_metadata)
            actions_layout.addWidget(remove_button)

            self.metadata_table.setCellWidget(i, 3, actions_widget)

        # Reconnect after loading
        self.metadata_table.itemChanged.connect(self.on_metadata_changed)
        self.on_metadata_changed_is_connected = True

    def extract_array_values(self, field: ReaderField) -> list:
        """Extract all values from an array field."""
        if not field.types or field.types[0] != GGUFValueType.ARRAY:
            return []

        curr_type = field.types[1]
        array_values = []
        total_elements = len(field.data)

        if curr_type == GGUFValueType.STRING:
            for element_pos in range(total_elements):
                value_string = str(bytes(field.parts[-1 - (total_elements - element_pos - 1) * 2]), encoding='utf-8')
                array_values.append(value_string)
        elif self.reader and curr_type in self.reader.gguf_scalar_to_np:
            for element_pos in range(total_elements):
                array_values.append(field.parts[-1 - (total_elements - element_pos - 1)][0])

        return array_values

    def get_enum_for_key(self, key: str) -> Optional[Type[enum.Enum]]:
        """Get the enum type for a given key if it exists."""
        return KEY_TO_ENUM_TYPE.get(key)

    def format_enum_value(self, value: Any, enum_type: Type[enum.Enum]) -> str:
        """Format a value as an enum if possible."""
        try:
            if isinstance(value, (int, str)):
                enum_value = enum_type(value)
                return f"{enum_value.name} ({value})"
        except (ValueError, KeyError):
            pass
        return str(value)

    def format_field_value(self, field: ReaderField) -> str:
        if not field.types:
            return "N/A"

        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                return str(bytes(field.parts[-1]), encoding='utf-8')
            elif self.reader and curr_type in self.reader.gguf_scalar_to_np:
                value = field.parts[-1][0]
                # Check if this field has an enum type
                enum_type = self.get_enum_for_key(field.name)
                if enum_type is not None:
                    return self.format_enum_value(value, enum_type)
                return str(value)

        if field.types[0] == GGUFValueType.ARRAY:
            array_values = self.extract_array_values(field)
            render_element = min(5, len(array_values))

            # Get enum type for this array if applicable
            enum_type = self.get_enum_for_key(field.name)

            if enum_type is not None:
                array_elements = []
                for i in range(render_element):
                    array_elements.append(self.format_enum_value(array_values[i], enum_type))
            else:
                array_elements = [str(array_values[i]) for i in range(render_element)]

            return f"[ {', '.join(array_elements).strip()}{', ...' if len(array_values) > len(array_elements) else ''} ]"

        return "Complex value"

    def load_tensors(self):
        self.tensors_table.setRowCount(0)

        if not self.reader:
            return

        for i, tensor in enumerate(self.reader.tensors):
            self.tensors_table.insertRow(i)

            # Name
            name_item = QTableWidgetItem(tensor.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tensors_table.setItem(i, 0, name_item)

            # Type
            type_item = QTableWidgetItem(tensor.tensor_type.name)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tensors_table.setItem(i, 1, type_item)

            # Shape
            shape_str = " × ".join(str(d) for d in tensor.shape)
            shape_item = QTableWidgetItem(shape_str)
            shape_item.setFlags(shape_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tensors_table.setItem(i, 2, shape_item)

            # Elements
            elements_item = QTableWidgetItem(str(tensor.n_elements))
            elements_item.setFlags(elements_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tensors_table.setItem(i, 3, elements_item)

            # Size
            size_item = QTableWidgetItem(f"{tensor.n_bytes:,}")
            size_item.setFlags(size_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tensors_table.setItem(i, 4, size_item)

    def on_metadata_changed(self, item):
        if item.column() != 2:  # Only handle value column changes
            return

        row = item.row()
        orig_item = self.metadata_table.item(row, 0)
        key = None
        if orig_item:
            key = orig_item.text()
        new_value = item.text()

        field = None
        if self.reader and key:
            field = self.reader.get_field(key)
        if not field or not field.types or not key:
            return

        value_type = field.types[0]

        # Check if this is an enum field
        enum_type = self.get_enum_for_key(key)
        if enum_type is not None and value_type == GGUFValueType.INT32:
            # Try to parse the enum value from the text
            try:
                # Check if it's a name
                try:
                    enum_val = enum_type[new_value]
                    converted_value = enum_val.value
                except (KeyError, AttributeError):
                    # Check if it's a number or "NAME (value)" format
                    if '(' in new_value and ')' in new_value:
                        # Extract the value from "NAME (value)" format
                        value_part = new_value.split('(')[1].split(')')[0].strip()
                        converted_value = int(value_part)
                    else:
                        # Try to convert directly to int
                        converted_value = int(new_value)

                # Validate that it's a valid enum value
                enum_type(converted_value)

                # Store the change
                self.metadata_changes[key] = (value_type, converted_value)
                self.modified = True

                # Update display with formatted enum value
                formatted_value = self.format_enum_value(converted_value, enum_type)
                item.setText(formatted_value)

                self.statusBar().showMessage(f"Changed {key} to {formatted_value}")
                return
            except (ValueError, KeyError) as e:
                QMessageBox.warning(
                    self,
                    f"Invalid Enum Value ({e})",
                    f"'{new_value}' is not a valid {enum_type.__name__} value.\n"
                    f"Valid values are: {', '.join(v.name for v in enum_type)}")

                # Revert to original value
                original_value = self.format_field_value(field)
                item.setText(original_value)
                return

        try:
            # Convert the string value to the appropriate type
            if value_type == GGUFValueType.UINT8:
                converted_value = np.uint8(int(new_value))
            elif value_type == GGUFValueType.INT8:
                converted_value = np.int8(int(new_value))
            elif value_type == GGUFValueType.UINT16:
                converted_value = np.uint16(int(new_value))
            elif value_type == GGUFValueType.INT16:
                converted_value = np.int16(int(new_value))
            elif value_type == GGUFValueType.UINT32:
                converted_value = np.uint32(int(new_value))
            elif value_type == GGUFValueType.INT32:
                converted_value = np.int32(int(new_value))
            elif value_type == GGUFValueType.FLOAT32:
                converted_value = np.float32(float(new_value))
            elif value_type == GGUFValueType.BOOL:
                converted_value = new_value.lower() in ('true', 'yes', '1')
            elif value_type == GGUFValueType.STRING:
                converted_value = new_value
            else:
                # Unsupported type for editing
                return

            # Store the change
            self.metadata_changes[key] = (value_type, converted_value)
            self.modified = True

            self.statusBar().showMessage(f"Changed {key} to {new_value}")
        except ValueError:
            QMessageBox.warning(self, "Invalid Value", f"The value '{new_value}' is not valid for type {value_type.name}")

            # Revert to original value
            original_value = self.format_field_value(field)
            item.setText(original_value)

    def remove_metadata(self):
        button = self.sender()
        key = button.property("key")
        row = button.property("row")

        reply = QMessageBox.question(
            self, "Confirm Removal",
            f"Are you sure you want to remove the metadata key '{key}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.metadata_table.removeRow(row)
            self.metadata_to_remove.add(key)

            # If we previously had changes for this key, remove them
            if key in self.metadata_changes:
                del self.metadata_changes[key]

            self.modified = True
            self.statusBar().showMessage(f"Marked {key} for removal")

    def edit_metadata_enum(self):
        """Edit an enum metadata field."""
        button = self.sender()
        key = button.property("key")
        row = button.property("row")

        field = None
        if self.reader:
            field = self.reader.get_field(key)
        if not field or not field.types:
            return

        enum_type = self.get_enum_for_key(key)
        if enum_type is None:
            return

        # Get current value
        current_value = field.contents()

        # Create a dialog with enum options
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select {enum_type.__name__} Value")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        for enum_val in enum_type:
            combo.addItem(f"{enum_val.name} ({enum_val.value})", enum_val.value)

        # Set current value
        try:
            if isinstance(current_value, (int, str)):
                enum_val = enum_type(current_value)
                combo.setCurrentText(f"{enum_val.name} ({current_value})")
        except (ValueError, KeyError):
            pass

        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get the selected value
            new_value = combo.currentData()
            enum_val = enum_type(new_value)

            # Store the change
            self.metadata_changes[key] = (field.types[0], new_value)
            self.modified = True

            # Update display
            display_text = f"{enum_val.name} ({new_value})"
            target_item = self.metadata_table.item(row, 2)
            if target_item:
                target_item.setText(display_text)

            self.statusBar().showMessage(f"Changed {key} to {display_text}")

    def edit_array_metadata(self):
        button = self.sender()
        key = button.property("key")
        row = button.property("row")

        # Check if this is one of the linked tokenizer keys
        if key in TOKENIZER_LINKED_KEYS:
            self.edit_tokenizer_metadata(key)
            return

        field = None
        if self.reader:
            field = self.reader.get_field(key)
        if not field or not field.types or field.types[0] != GGUFValueType.ARRAY:
            return

        # Get array element type
        element_type = field.types[1]

        # Extract array values
        array_values = self.extract_array_values(field)

        # Open array editor dialog
        dialog = ArrayEditorDialog(array_values, element_type, key, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_values = dialog.get_array_values()

            # Store the change
            self.metadata_changes[key] = (GGUFValueType.ARRAY, (element_type, new_values))
            self.modified = True

            # Update display
            enum_type = self.get_enum_for_key(key)
            if enum_type is not None and element_type == GGUFValueType.INT32:
                value_str = f"[ {', '.join(self.format_enum_value(v, enum_type) for v in new_values[:5])}{', ...' if len(new_values) > 5 else ''} ]"
            else:
                value_str = f"[ {', '.join(str(v) for v in new_values[:5])}{', ...' if len(new_values) > 5 else ''} ]"
            target_item = self.metadata_table.item(row, 2)
            if target_item:
                target_item.setText(value_str)

            self.statusBar().showMessage(f"Updated array values for {key}")

    def edit_tokenizer_metadata(self, trigger_key):
        """Edit the linked tokenizer metadata arrays together."""
        if not self.reader:
            return

        # Get all three fields
        tokens_field = self.reader.get_field(gguf.Keys.Tokenizer.LIST)
        token_types_field = self.reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE)
        scores_field = self.reader.get_field(gguf.Keys.Tokenizer.SCORES)

        # Extract values from each field
        tokens = self.extract_array_values(tokens_field) if tokens_field else []
        token_types = self.extract_array_values(token_types_field) if token_types_field else []
        scores = self.extract_array_values(scores_field) if scores_field else []

        # Apply any pending changes
        if gguf.Keys.Tokenizer.LIST in self.metadata_changes:
            _, (_, tokens) = self.metadata_changes[gguf.Keys.Tokenizer.LIST]
        if gguf.Keys.Tokenizer.TOKEN_TYPE in self.metadata_changes:
            _, (_, token_types) = self.metadata_changes[gguf.Keys.Tokenizer.TOKEN_TYPE]
        if gguf.Keys.Tokenizer.SCORES in self.metadata_changes:
            _, (_, scores) = self.metadata_changes[gguf.Keys.Tokenizer.SCORES]

        # Open the tokenizer editor dialog
        dialog = TokenizerEditorDialog(tokens, token_types, scores, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_tokens, new_token_types, new_scores = dialog.get_data()

            # Store changes for all three arrays
            if tokens_field:
                self.metadata_changes[gguf.Keys.Tokenizer.LIST] = (
                    GGUFValueType.ARRAY,
                    (tokens_field.types[1], new_tokens)
                )

            if token_types_field:
                self.metadata_changes[gguf.Keys.Tokenizer.TOKEN_TYPE] = (
                    GGUFValueType.ARRAY,
                    (token_types_field.types[1], new_token_types)
                )

            if scores_field:
                self.metadata_changes[gguf.Keys.Tokenizer.SCORES] = (
                    GGUFValueType.ARRAY,
                    (scores_field.types[1], new_scores)
                )

            self.modified = True

            # Update display for all three fields
            self.update_tokenizer_display(gguf.Keys.Tokenizer.LIST, new_tokens)
            self.update_tokenizer_display(gguf.Keys.Tokenizer.TOKEN_TYPE, new_token_types)
            self.update_tokenizer_display(gguf.Keys.Tokenizer.SCORES, new_scores)

            self.statusBar().showMessage("Updated tokenizer data")

    def update_tokenizer_display(self, key, values):
        """Update the display of a tokenizer field in the metadata table."""
        for row in range(self.metadata_table.rowCount()):
            key_item = self.metadata_table.item(row, 0)
            if key_item and key_item.text() == key:
                value_str = f"[ {', '.join(str(v) for v in values[:5])}{', ...' if len(values) > 5 else ''} ]"
                value_item = self.metadata_table.item(row, 2)
                if value_item:
                    value_item.setText(value_str)
                break

    def add_metadata(self):
        dialog = AddMetadataDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            key, value_type, value = dialog.get_data()

            if not key:
                QMessageBox.warning(self, "Invalid Key", "Key cannot be empty")
                return

            # Check if key already exists
            for row in range(self.metadata_table.rowCount()):
                orig_item = self.metadata_table.item(row, 0)
                if orig_item and orig_item.text() == key:
                    QMessageBox.warning(self, "Duplicate Key", f"Key '{key}' already exists")
                    return

            # Add to table
            row = self.metadata_table.rowCount()
            self.metadata_table.insertRow(row)

            # Key
            key_item = QTableWidgetItem(key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.metadata_table.setItem(row, 0, key_item)

            # Type
            type_item = QTableWidgetItem(value_type.name)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.metadata_table.setItem(row, 1, type_item)

            # Value
            value_item = QTableWidgetItem(str(value))
            value_item.setFlags(value_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.metadata_table.setItem(row, 2, value_item)

            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)

            remove_button = QPushButton("Remove")
            remove_button.setProperty("row", row)
            remove_button.setProperty("key", key)
            remove_button.clicked.connect(self.remove_metadata)
            actions_layout.addWidget(remove_button)

            self.metadata_table.setCellWidget(row, 3, actions_widget)

            # Store the change
            self.metadata_changes[key] = (value_type, value)
            self.modified = True

            self.statusBar().showMessage(f"Added new metadata key {key}")

    def save_file(self):
        if not self.reader:
            QMessageBox.warning(self, "No File Open", "Please open a GGUF file first")
            return

        if not self.modified and not self.metadata_changes and not self.metadata_to_remove:
            QMessageBox.information(self, "No Changes", "No changes to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save GGUF File As", "", "GGUF Files (*.gguf);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self.statusBar().showMessage(f"Saving to {file_path}...")
            QApplication.processEvents()

            # Get architecture and endianness from the original file
            arch = 'unknown'
            field = self.reader.get_field(gguf.Keys.General.ARCHITECTURE)
            if field:
                arch = field.contents()

            # Create writer
            writer = GGUFWriter(file_path, arch=arch, endianess=self.reader.endianess)

            # Get alignment if present
            alignment = None
            field = self.reader.get_field(gguf.Keys.General.ALIGNMENT)
            if field:
                alignment = field.contents()
                if alignment is not None:
                    writer.data_alignment = alignment

            # Copy metadata with changes
            for field in self.reader.fields.values():
                # Skip virtual fields and fields written by GGUFWriter
                if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
                    continue

                # Skip fields marked for removal
                if field.name in self.metadata_to_remove:
                    continue

                # Apply changes if any
                sub_type = None
                if field.name in self.metadata_changes:
                    value_type, value = self.metadata_changes[field.name]
                    if value_type == GGUFValueType.ARRAY:
                        # Handle array values
                        sub_type, value = value
                else:
                    # Copy original value
                    value = field.contents()
                    value_type = field.types[0]
                    if value_type == GGUFValueType.ARRAY:
                        sub_type = field.types[-1]

                if value is not None:
                    writer.add_key_value(field.name, value, value_type, sub_type=sub_type)

            # Add new metadata
            for key, (value_type, value) in self.metadata_changes.items():
                # Skip if the key already existed (we handled it above)
                if self.reader.get_field(key) is not None:
                    continue

                sub_type = None
                if value_type == GGUFValueType.ARRAY:
                    # Handle array values
                    sub_type, value = value

                writer.add_key_value(key, value, value_type, sub_type=sub_type)

            # Add tensors (including data)
            for tensor in self.reader.tensors:
                writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.data.shape, raw_dtype=tensor.tensor_type)

            # Write header and metadata
            writer.open_output_file(Path(file_path))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()

            # Write tensor data using the optimized method
            writer.write_tensors_to_file(progress=False)

            writer.close()

            self.statusBar().showMessage(f"Saved to {file_path}")

            # Ask if user wants to open the new file
            reply = QMessageBox.question(
                self, "Open Saved File",
                "Would you like to open the newly saved file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.reader = GGUFReader(file_path, 'r')
                self.current_file = file_path
                self.file_path_edit.setText(file_path)

                self.load_metadata()
                self.load_tensors()

                self.metadata_changes = {}
                self.metadata_to_remove = set()
                self.modified = False

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
            self.statusBar().showMessage("Error saving file")


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI GGUF Editor")
    parser.add_argument("model_path", nargs="?", help="path to GGUF model file to load at startup")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    app = QApplication(sys.argv)
    window = GGUFEditorWindow()
    window.show()

    # Load model if specified
    if args.model_path:
        if os.path.isfile(args.model_path) and args.model_path.endswith('.gguf'):
            window.load_file(args.model_path)
        else:
            logger.error(f"Invalid model path: {args.model_path}")
            QMessageBox.warning(
                window,
                "Invalid Model Path",
                f"The specified file does not exist or is not a GGUF file: {args.model_path}")

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
