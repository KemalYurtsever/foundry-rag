from __future__ import annotations

import tkinter as tk


class Tooltip:
    """Small keyboard-neutral tooltip for controls that need extra explanation."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event: tk.Event | None = None) -> None:
        if self.window is not None:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.attributes("-topmost", True)
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window.geometry(f"+{x}+{y}")
        tk.Label(
            self.window,
            text=self.text,
            justify=tk.LEFT,
            wraplength=320,
            background="#202a3a",
            foreground="#f1f5f9",
            relief=tk.SOLID,
            borderwidth=1,
            padx=8,
            pady=6,
            font=("Segoe UI", 9),
        ).pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


def add_tooltip(widget: tk.Widget, text: str) -> Tooltip:
    return Tooltip(widget, text)
