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
    """
    Dialog for defining how CSV columns map to the fields `VideoPlayer`
    needs.

    The user is asked to enter

    * X- and Y-coordinate column names (required)
    * either a timestamp column **or** a duration column (at least one
      required)
    * the column that identifies a recording session (required)
    * optional extra filters in the form
      ``col=value1|value2, other_col=foo``

    After the user clicks **OK** the dialog stores a dictionary in
    ``self.result``:

    ```
    {
        "pixel_x":                <str>,
        "pixel_y":                <str>,
        "time":                   <str | None>,
        "duration":               <str | None>,
        "recording_session_col":  <str>,
        "filter_columns":         {<str>: list[str], ...}
    }
    ```

    If the user cancels or validation fails, ``self.result`` is ``None``.

    Parameters
    ----------
    parent : tk.Misc | None
        The parent window (can be withdrawn).
    title : str | None, default "Configure Column Mapping"
        Window title.

    Notes
    -----
    `simpledialog.Dialog` shows the window immediately during
    construction; when it closes, just read
    ``ColumnMappingDialog(...).result`` to obtain the mapping.
    """
    result: dict | None

    def __init__(self, parent: tk.Misc | None, title: str | None = ...):
        super().__init__(parent, title)
        self.pixel_x_entry = None
        self.pixel_y_entry = None
        self.session_entry = None
        self.page_name_entry = None
        self.time_entry = None
        self.duration_entry = None
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
                master, text='Recording session column; e.g. RECORDING_SESSION_LABEL:',
            )
            .grid(row=2, column=0, sticky='w', pady=2)
        )
        self.session_entry = ttk.Entry(master, width=30)
        self.session_entry.grid(row=2, column=1, pady=2)

        (
            ttk.Label(master, text='Page name column; e.g. page_name:')
            .grid(row=3, column=0, sticky='w', pady=2)
        )
        self.page_name_entry = ttk.Entry(master, width=30)
        self.page_name_entry.grid(row=3, column=1, pady=2)

        (
            ttk.Label(master, text='Timestamp column (optional):')
            .grid(row=4, column=0, sticky='w', pady=2)
        )
        self.time_entry = ttk.Entry(master, width=30)
        self.time_entry.grid(row=4, column=1, pady=2)

        (
            ttk.Label(
                master, text='Duration column (optional); e.g. CURRENT_FIX_DURATION:',
            )
            .grid(row=5, column=0, sticky='w', pady=2)
        )
        self.duration_entry = ttk.Entry(master, width=30)
        self.duration_entry.grid(row=5, column=1, pady=2)

        (
            ttk.Label(
                master,
                text=(
                    'Other filters '
                    "(comma-separated, use '=' for column and '|' for alternatives; "
                    'e.g.  trial_date=1998-06-02, '
                    'trial_number=1|2):'
                ),
            ).grid(row=6, column=0, columnspan=2, sticky='w', pady=2)
        )
        self.filters_entry = ttk.Entry(master, width=50)
        self.filters_entry.grid(row=7, column=0, columnspan=2, pady=2)

        return self.pixel_x_entry

    def apply(self):
        """Validate entries, build the mapping dict, and store it in self.result."""
        pixel_x = self.pixel_x_entry.get().strip()
        pixel_y = self.pixel_y_entry.get().strip()
        session = self.session_entry.get().strip()
        page_name = self.page_name_entry.get().strip()
        time = self.time_entry.get().strip()
        duration = self.duration_entry.get().strip()
        raw_filters = self.filters_entry.get().strip()

        if not (pixel_x and pixel_y):
            messagebox.showerror(
                'Error',
                'X and Y coordinate column names are required.',
            )
            return None

        if not page_name:
            messagebox.showerror(
                'Error',
                'Page name column name is required.',
            )
            return None

        if not session:
            messagebox.showerror(
                'Error',
                'Recording session column name is required.',
            )
            return None

        if not (time or duration):
            messagebox.showerror(
                'Error',
                'You must provide at least a timestamp or a duration column name.',
            )
            return None

        filters: dict[str, list[str]] = {}
        if raw_filters:
            try:
                for pair in (p.strip() for p in raw_filters.split(',') if p.strip()):
                    if '=' not in pair:
                        raise ValueError(
                            f"Missing '=' in filter pair: '{pair}'",
                        )
                    col, val = pair.split('=', 1)
                    values = [v.strip() for v in val.split('|') if v.strip()]

                    if not values:
                        raise ValueError(
                            f"No value specified for column '{col.strip()}'",
                        )
                    filters[col.strip()] = values

            except ValueError as err:
                messagebox.showerror('Filter Format Error', str(err))
                return None

        self.result = {
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'recording_session': session,
            'page_name': page_name,
            'time': time or None,
            'duration': duration or None,
            'filter_columns': filters,
        }
