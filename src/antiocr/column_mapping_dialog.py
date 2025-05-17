"""Dialog for mapping CSV columns used by `AntiOCR`.

`ColumnMappingDialog` is a small Tkinter dialog that lets the user pick
the CSV columns for X/Y gaze coordinates, an interest-area label, the page
name, the recording-session ID, and any extra filters.  After pressing
**OK** the dialog stores the mapping in `self.result`; if the user cancels,
`self.result` is `None`.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk


class ColumnMappingDialog(simpledialog.Dialog):
    """
    Dialog for defining how CSV columns map to the fields `AntiOCR` needs.

    The user is asked to enter

    * X- and Y-coordinate column names (required)
    * interest-area label column name (required)
    * the column that identifies a recording session (required)
    * the column that holds the page / stimulus name (required)
    * optional extra filters in the form
      ``col=value1|value2, other_col=foo``

    After the user clicks **OK** the dialog stores a dictionary in
    ``self.result``::

        {
            "pixel_x":           <str>,
            "pixel_y":           <str>,
            "interest_area_label": <str>,
            "recording_session": <str>,
            "page_name":         <str>,
            "filter_columns":    {<str>: list[str], ...}
        }

    If the user cancels or validation fails, ``self.result`` is *None*.

    Parameters
    ----------
    parent : tk.Misc | None
        The parent (possibly withdrawn) window.
    title : str | None, default "Configure Column Mapping"
        Dialog title.

    Notes
    -----
    `simpledialog.Dialog` shows the window immediately during construction;
    once it closes you can read
    ``ColumnMappingDialog(parent, title).result`` to obtain the mapping.
    """
    result: dict | None

    def __init__(self, parent: tk.Misc | None, title: str | None = ...):
        super().__init__(parent, title)
        self.pixel_x_entry = None
        self.pixel_y_entry = None
        self.interest_area_label_entry = None
        self.session_entry = None
        self.page_name_entry = None
        self.filters_entry = None

    def body(self, master):
        """Build and lay out the dialog widgets; return the widget to focus."""
        self.title('Configure Column Mapping')

        (
            ttk.Label(
                master, text='X-coordinate column; e.g. CURRENT_FIX_X_INTEREST_AREA:',
            )
            .grid(row=0, column=0, sticky='w', pady=2)
        )
        self.pixel_x_entry = ttk.Entry(master, width=30)
        self.pixel_x_entry.grid(row=0, column=1, pady=2)

        (
            ttk.Label(
                master, text='Y-coordinate column; e.g. CURRENT_FIX_Y_INTEREST_AREA:',
            )
            .grid(row=1, column=0, sticky='w', pady=2)
        )
        self.pixel_y_entry = ttk.Entry(master, width=30)
        self.pixel_y_entry.grid(row=1, column=1, pady=2)

        (
            ttk.Label(
                master, text='Interest area label column; e.g. CURRENT_FIX_INTEREST_AREA_LABEL:',
            )
            .grid(row=2, column=0, sticky='w', pady=2)
        )
        self.interest_area_label_entry = ttk.Entry(master, width=30)
        self.interest_area_label_entry.grid(row=2, column=1, pady=2)

        (
            ttk.Label(
                master, text='Recording session column; e.g. RECORDING_SESSION_LABEL:',
            )
            .grid(row=3, column=0, sticky='w', pady=2)
        )
        self.session_entry = ttk.Entry(master, width=30)
        self.session_entry.grid(row=3, column=1, pady=2)

        (
            ttk.Label(master, text='Page name column; e.g. page_name:')
            .grid(row=4, column=0, sticky='w', pady=2)
        )
        self.page_name_entry = ttk.Entry(master, width=30)
        self.page_name_entry.grid(row=4, column=1, pady=2)

        (
            ttk.Label(
                master,
                text=(
                    'Other filters '
                    "(comma-separated, use '=' for column and '|' for alternatives; "
                    'e.g.  trial_date=1998-06-02, '
                    'trial_number=1|2):'
                ),
            ).grid(row=5, column=0, columnspan=2, sticky='w', pady=2)
        )
        self.filters_entry = ttk.Entry(master, width=50)
        self.filters_entry.grid(row=6, column=0, columnspan=2, pady=2)

        return self.pixel_x_entry

    def apply(self):
        """Validate entries, build the mapping dict, and store it in self.result."""
        pixel_x = self.pixel_x_entry.get().strip()
        pixel_y = self.pixel_y_entry.get().strip()
        interest_area_label = self.interest_area_label_entry.get().strip()
        session = self.session_entry.get().strip()
        page_name = self.page_name_entry.get().strip()
        raw_filters = self.filters_entry.get().strip()

        if not (pixel_x and pixel_y):
            messagebox.showerror(
                'Error', 'X and Y coordinate column names are required.',
            )
            return None

        if not interest_area_label:
            messagebox.showerror(
                'Error', 'Interest area label column name is required.',
            )
            return None

        if not page_name:
            messagebox.showerror('Error', 'Page name column name is required.')
            return None

        if not session:
            messagebox.showerror(
                'Error', 'Recording session column name is required.',
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
            'interest_area_label': interest_area_label,
            'recording_session': session,
            'page_name': page_name,
            'filter_columns': filters,
        }
