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
    """
    Dialog for letting the user choose one of the loaded recording sessions.

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
    title : str, default "Select session"
        Window title.

    Notes
    -----
    As with other `simpledialog.Dialog` classes, the window appears
    immediately during construction; once it closes you can read
    ``SessionSelectDialog(parent, sessions).result`` to obtain the choice.
    """

    def __init__(self, parent: tk.Misc, sessions: list[str], title: str = 'Select session'):
        self._listbox = None
        self._sessions = sessions
        self.result: str | None = None
        super().__init__(parent, title)

    def body(self, master):
        """Create the list-box UI and return the widget that gets initial focus."""
        self.minsize(400, 150)

        ttk.Label(master, text='Choose a recording session:').pack(
            padx=8, pady=(8, 4))

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
        """Save the selected session name in ``self.result`` when OK is pressed."""
        selection = self._listbox.curselection()
        if selection:
            self.result = self._sessions[selection[0]]
