from __future__ import annotations

import os
import tkinter as tk


DARK_THEME = {
    "bg": "#090d14", "panel": "#111722", "panel_raised": "#192231",
    "field": "#0d131d", "border": "#263244", "text": "#f1f5f9",
    "muted": "#94a3b8", "accent": "#7c6df2", "accent_hover": "#9287f7",
    "accent_pressed": "#6658d9", "success": "#34d399", "warning": "#fbbf24",
    "danger": "#fb7185", "danger_hover": "#f43f5e", "selection": "#5b4fd1",
    "status_busy": "#252245", "status_ready": "#12352e", "status_error": "#421d2a",
}

LIGHT_THEME = {
    "bg": "#f5f7fb", "panel": "#ffffff", "panel_raised": "#e8edf5",
    "field": "#ffffff", "border": "#cbd5e1", "text": "#172033",
    "muted": "#64748b", "accent": "#6556d9", "accent_hover": "#7567e8",
    "accent_pressed": "#5145b8", "success": "#087f5b", "warning": "#a16207",
    "danger": "#be123c", "danger_hover": "#9f1239", "selection": "#6556d9",
    "status_busy": "#e8e5ff", "status_ready": "#d9f5e9", "status_error": "#ffe0e7",
}


def theme_palette(name: str) -> dict[str, str]:
    return dict(LIGHT_THEME if name == "Light" else DARK_THEME)


def apply_window_frame(
    window: tk.Misc, palette: dict[str, str], theme_name: str
) -> None:
    """Keep Tk's exposed background and the Windows frame on the same palette."""
    window.configure(background=palette["bg"])
    if os.name != "nt":
        return

    try:
        import ctypes

        window.update_idletasks()
        child_handle = window.winfo_id()
        get_parent = ctypes.windll.user32.GetParent
        get_parent.argtypes = [ctypes.c_void_p]
        get_parent.restype = ctypes.c_void_p
        handle = get_parent(ctypes.c_void_p(child_handle)) or child_handle
        native_handle = ctypes.c_void_p(handle)

        dark_mode = ctypes.c_int(1 if theme_name == "Dark" else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            native_handle, 20, ctypes.byref(dark_mode), ctypes.sizeof(dark_mode)
        )

        def colorref(value: str) -> ctypes.c_uint:
            red, green, blue = (
                int(value[index : index + 2], 16) for index in (1, 3, 5)
            )
            return ctypes.c_uint(red | (green << 8) | (blue << 16))

        for attribute, color in (
            (34, palette["border"]),
            (35, palette["bg"]),
            (36, palette["text"]),
        ):
            native_color = colorref(color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                native_handle,
                attribute,
                ctypes.byref(native_color),
                ctypes.sizeof(native_color),
            )
    except (AttributeError, OSError, tk.TclError):
        # Older Windows versions may not expose the newer DWM color attributes.
        pass
