# Copyright (c) 2025 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Dialog for mapping CSV columns used by `VideoPlayer`.

`ColumnMappingDialog` is a simple Tkinter dialog that lets the user pick
the CSV columns for X/Y gaze coordinates, timestamps (or durations),
session IDs, and any extra filters.  After pressing **OK** the dialog
stores the mapping in `self.result`; if the user cancels, `self.result`
is `None`.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk


class ColumnMappingDialog(simpledialog.Dialog):
    """Dialog for defining how CSV columns map to `VideoPlayer` fields.

    The user is asked to enter

    * X- and Y-coordinate column names (required)
    * either a timestamp column **or** a duration column (at least one
      required)
    * the column that identifies a recording session (required)
    * optional extra filters in the form
      ``col=value1|value2, other_col=foo``

    After the user clicks **OK** the dialog stores a dictionary in
    ``self.result``::


        {
            "pixel_x":                <str>,
            "pixel_y":                <str>,
            "image_column":           <str>,
            "grouping_parameters":    <str> | None>,
            "filter_columns":         {<str>: list[str], ...|None}
        }

    If the user cancels or validation fails, ``self.result`` is ``None``.

    Attributes
    ----------
    result : dict | None
        The mapping returned by the dialog, or None if the user cancelled.

    Parameters
    ----------
    parent : tk.Misc | None
        The parent window (can be withdrawn).
    title : str | None, optional
        Window title; if None, the default dialog title is used.

    Notes
    -----
    `simpledialog.Dialog` shows the window immediately during construction;
    When done, read ``ColumnMappingDialog(...).result`` to get the mapping.
    """

    result: dict | None

    def __init__(self, parent: tk.Misc | None, title: str | None = None):
        super().__init__(parent, title)
        self.pixel_x_entry = None
        self.pixel_y_entry = None
        self.image_column_entry = None
        self.grouping_entry = None
        self.filters_entry = None

    def body(self, master):
        """Build and lay out the dialog widgets; return the widget to focus."""
        self.title('Configure Column Mapping')

        (
            ttk.Label(master, text='X-coordinate column; e.g. CURRENT_FIX_X:')
            .grid(row=0, column=0, sticky='w', pady=2)
        )
        self.pixel_x_entry = ttk.Entry(master, width=30)
        self.pixel_x_entry.grid(row=0, column=1, pady=2)

        (
            ttk.Label(master, text='Y-coordinate column; e.g. CURRENT_FIX_Y:')
            .grid(row=1, column=0, sticky='w', pady=2)
        )
        self.pixel_y_entry = ttk.Entry(master, width=30)
        self.pixel_y_entry.grid(row=1, column=1, pady=2)

        (
            ttk.Label(
                master,
                text='Image name column; e.g. page_name:',
            )
            .grid(row=2, column=0, sticky='w', pady=2)
        )
        self.image_column_entry = ttk.Entry(master, width=30)
        self.image_column_entry.grid(row=2, column=1, pady=2)

        (
            ttk.Label(
                master, text=(
                    'Grouping parameters (optional) (Comma-separated); e.g. RECORDING_SESSION_LABEL, trial_number:'
                ),
            ).grid(row=4, column=0, sticky='w', pady=2)
        )
        self.grouping_entry = ttk.Entry(master, width=30)
        self.grouping_entry.grid(row=4, column=1, pady=2)

        (
            ttk.Label(
                master,
                text=(
                    'Filter columns '
                    "(comma-separated, use '=' for column "
                    "and '|' for alternatives; "
                    'e.g.  RECORDING_SESSION_LABEL=msd002|msd003, '
                    'page_name=recording-dickens-1):'
                ),
            ).grid(row=6, column=0, columnspan=2, sticky='w', pady=2)
        )
        self.filters_entry = ttk.Entry(master, width=50)
        self.filters_entry.grid(row=7, column=0, columnspan=2, pady=2)

        return self.pixel_x_entry

    def validate(self) -> bool:
        """Validate inputs before closing the dialog."""
        pixel_x = self.pixel_x_entry.get().strip()
        pixel_y = self.pixel_y_entry.get().strip()
        image = self.image_column_entry.get().strip()
        raw_grouping = self.grouping_entry.get().strip()
        raw_filters = self.filters_entry.get().strip()

        if not (pixel_x and pixel_y):
            messagebox.showerror(
                'Error',
                'X and Y coordinate column names are required.',
            )
            return False

        if not image:
            messagebox.showerror(
                'Error',
                'Image column name is required.',
            )
            return False

        # Check grouping format
        if raw_grouping:
            try:
                [v.strip() for v in raw_grouping.split('|')]
            except Exception as err:
                messagebox.showerror('Grouping Format Error', str(err))
                return False

        # Check filter format
        if raw_filters:
            try:
                for pair in (p.strip() for p in raw_filters.split(',') if p.strip()):
                    if '=' not in pair:
                        raise ValueError(
                            f"Missing '=' in filter pair: '{pair}'")
                    col, val = pair.split('=', 1)
                    values = [v.strip() for v in val.split('|') if v.strip()]
                    if not values:
                        raise ValueError(
                            f"No value specified for column '{col.strip()}'")
            except ValueError as err:
                messagebox.showerror('Filter Format Error', str(err))
                return False

        return True

    def apply(self) -> None:
        """Save the mapping to self.result (assumes validation already passed)."""
        pixel_x = self.pixel_x_entry.get().strip()
        pixel_y = self.pixel_y_entry.get().strip()
        image = self.image_column_entry.get().strip()
        raw_grouping = self.grouping_entry.get().strip()
        raw_filters = self.filters_entry.get().strip()

        grouping = [v.strip()
                    for v in raw_grouping.split('|')] if raw_grouping else []
        grouping.append(image)

        filters: dict[str, list[str]] = {}
        if raw_filters:
            for pair in (p.strip() for p in raw_filters.split(',') if p.strip()):
                col, val = pair.split('=', 1)
                values = [v.strip() for v in val.split('|') if v.strip()]
                filters[col.strip()] = values

        self.result = {
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'image_column': image,
            'grouping': grouping,
            'filter_columns': filters,
        }
