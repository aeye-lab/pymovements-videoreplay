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
"""Dialog for choosing a recording session used by `VideoPlayer`.

`SessionSelectDialog` is a small Tkinter dialog that displays a list of
available recording sessions.  The user selects a session and presses
**OK**.  The chosen session name is stored in `self.result`; if the user
cancels, `self.result` is `None`.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from tkinter import ttk


class SessionSelectDialog(simpledialog.Dialog):
    """Dialog for letting the user choose one of the loaded recording sessions.

    The dialog shows a list box populated with the available session names.
    The user selects a session and presses **OK** (or double-clicks).
    The chosen name is stored in ``self.result``; if the user cancels,
    ``self.result`` is ``None``.

    Parameters
    ----------
    parent : tk.Misc
        The (possibly withdrawn) parent window.
    sessions : list[str]
        Names of the recording sessions to display.
    title : str, optional
        Window title. Defaults to "Select session".

    Notes
    -----
    As with other `simpledialog.Dialog` classes, the window appears
    immediately during construction; once it closes you can read
    ``SessionSelectDialog(parent, sessions).result`` to obtain the choice.
    """

    def __init__(
            self,
            parent: tk.Misc,
            sessions: list[str],
            title: str = 'Select session',
    ):
        self._listbox = None
        self._sessions = sessions
        self.result: str | None = None
        super().__init__(parent, title)

    def body(self, master):
        """Create the list-box and return initial focus widget."""
        self.minsize(400, 150)

        ttk.Label(master, text='Choose a recording session:').pack(
            padx=8, pady=(8, 4),
        )

        self._listbox = tk.Listbox(
            master,
            width=40,
            height=min(15, len(self._sessions)),
            exportselection=False,
        )
        for session in self._sessions:
            self._listbox.insert(tk.END, session)
        self._listbox.selection_set(0)
        self._listbox.pack(padx=8, pady=(0, 8), fill='both', expand=True)
        return self._listbox

    def apply(self):
        """Save the selected session name in ``self.result``."""
        selection = self._listbox.curselection()
        if selection:
            self.result = self._sessions[selection[0]]
