from __future__ import annotations

import importlib.util
import logging
import os
import re
import shutil
import threading
from collections import Counter
from dataclasses import replace
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from tkinterdnd2 import DND_FILES, TkinterDnD

from .app_settings import AppSettings, load_app_settings, save_app_settings
from .documents import _SUPPORTED_SUFFIXES
from .embeddings import optional_foundry_embedder
from .pipeline import Answer, RAGPipeline, flexible_answer_text
from .theme import apply_window_frame, theme_palette
from .ui_support import add_tooltip

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS = PROJECT_ROOT / "data" / "documents"
DEFAULT_CACHE = PROJECT_ROOT / ".rag_cache"

RETRIEVAL_PRESETS: dict[str, tuple[str, int]] = {
    "Precise": ("2", 0),
    "Balanced": ("Auto", 2),
    "Thorough": ("6", 3),
}


def _style_scrolled_text(widget: ScrolledText) -> None:
    widget.vbar.configure(
        background=THEME["panel_raised"],
        activebackground=THEME["border"],
        troughcolor=THEME["field"],
        borderwidth=0,
        highlightthickness=0,
    )

THEME = theme_palette("Dark")

def import_documents(paths: tuple[str, ...], destination: Path) -> list[Path]:
    """Copy supported user files into the managed document directory."""
    destination.mkdir(parents=True, exist_ok=True)
    imported: list[Path] = []
    for raw_path in paths:
        source = Path(raw_path)
        if source.suffix.casefold() not in _SUPPORTED_SUFFIXES:
            continue
        target = destination / source.name
        counter = 2
        while target.exists() and not source.samefile(target):
            target = destination / f"{source.stem} ({counter}){source.suffix}"
            counter += 1
        if not target.exists():
            shutil.copy2(source, target)
        imported.append(target)
    return imported


class RAGDesktopApp(TkinterDnD.Tk):
    """Local document-import and chat interface for the strict RAG pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Foundry Local RAG")
        self.settings_path = DEFAULT_CACHE / "app-settings.json"
        saved = load_app_settings(self.settings_path)
        THEME.clear()
        THEME.update(theme_palette(saved.theme))
        try:
            self.geometry(saved.geometry)
        except tk.TclError:
            self.geometry("1100x720")
        self.minsize(820, 560)
        self.state("zoomed")
        self.documents_dir = DEFAULT_DOCUMENTS
        self.cache_dir = DEFAULT_CACHE
        self.pipeline: RAGPipeline | None = None
        self.pipeline_lock = threading.Lock()
        self.busy = False
        self.sidebar_visible = saved.sidebar_visible
        self.sidebar_width = 280
        self.sidebar_animation: str | None = None
        self.source_link_counter = 0
        self.source_link_payloads: dict[str, tuple[str, Answer]] = {}
        self.last_answer: Answer | None = None
        self.last_evidence_answer: Answer | None = None
        self.retrieval_preset_var = tk.StringVar(value=saved.retrieval_preset)
        self.top_k_var = tk.StringVar(value=saved.top_k)
        self.neighbor_window_var = tk.IntVar(value=saved.neighbors)
        self.use_embeddings_var = tk.BooleanVar(value=saved.use_embeddings)
        self.show_diagnostics_var = tk.BooleanVar(value=saved.show_diagnostics)
        self.answer_mode_var = tk.StringVar(value=saved.answer_mode)
        self.theme_var = tk.StringVar(value=saved.theme)
        self.document_filter_var = tk.StringVar()
        self.document_sort_var = tk.StringVar(value=saved.file_sort)
        self.document_paths: list[Path] = []
        self.indexed_file_signatures: dict[str, tuple[int, int]] = {}
        self.document_chunk_counts: Counter[str] = Counter()
        self.document_page_counts: dict[str, int] = {}
        self.index_cancel_event = threading.Event()
        self._configure_style()
        self._build_layout()
        self._add_tooltips()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_document_list()
        if self.sidebar_visible:
            self.sidebar.grid()
            self.sidebar.configure(width=self.sidebar_width)
            self.sidebar_toggle.configure(text="Hide knowledge files")
        self._rebuild_index("Loading documents…")

    def _configure_style(self) -> None:
        apply_window_frame(self, THEME, self.theme_var.get())
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        elif "vista" in style.theme_names():
            style.theme_use("vista")

        style.configure(".", background=THEME["bg"], foreground=THEME["text"], fieldbackground=THEME["field"])
        style.configure("TFrame", background=THEME["bg"])
        style.configure("Panel.TFrame", background=THEME["panel"])
        style.configure("TLabel", background=THEME["bg"], foreground=THEME["text"])
        style.configure("Title.TLabel", background=THEME["bg"], foreground=THEME["text"], font=("Segoe UI", 18, "bold"))
        style.configure("Subtle.TLabel", background=THEME["bg"], foreground=THEME["muted"])
        style.configure("TLabelframe", background=THEME["panel"], foreground=THEME["text"], bordercolor=THEME["border"], relief=tk.SOLID)
        style.configure("TLabelframe.Label", background=THEME["panel"], foreground=THEME["muted"], font=("Segoe UI", 9, "bold"))
        style.configure(
            "TButton",
            background=THEME["panel_raised"],
            foreground=THEME["text"],
            borderwidth=0,
            relief=tk.FLAT,
            focusthickness=1,
            focuscolor=THEME["accent"],
            padding=(14, 8),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "TButton",
            background=[("pressed", THEME["field"]), ("active", THEME["border"]), ("disabled", THEME["panel"])],
            foreground=[("active", THEME["text"]), ("disabled", "#637083")],
        )
        style.configure(
            "Accent.TButton",
            background=THEME["accent"],
            foreground="#ffffff",
            borderwidth=0,
            relief=tk.FLAT,
            focusthickness=1,
            focuscolor=THEME["accent_hover"],
            padding=(16, 9),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[
                ("pressed", THEME["accent_pressed"]),
                ("active", THEME["accent_hover"]),
                ("disabled", THEME["border"]),
            ],
            foreground=[("disabled", THEME["muted"])],
        )
        style.configure(
            "Danger.TButton",
            background="#3a1d29",
            foreground=THEME["danger"],
            borderwidth=0,
            relief=tk.FLAT,
            focusthickness=1,
            focuscolor=THEME["danger"],
            padding=(14, 8),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Danger.TButton",
            background=[("pressed", "#541d2d"), ("active", "#49202d"), ("disabled", THEME["panel"])],
            foreground=[("active", "#fda4af"), ("disabled", "#637083")],
        )
        style.configure(
            "Theme.TButton",
            background=THEME["panel_raised"],
            foreground=THEME["text"],
            borderwidth=0,
            relief=tk.FLAT,
            focusthickness=1,
            focuscolor=THEME["accent"],
            padding=(9, 5),
            font=("Segoe UI Symbol", 14),
        )
        style.map(
            "Theme.TButton",
            background=[("pressed", THEME["field"]), ("active", THEME["border"])],
        )
        style.configure("TEntry", fieldbackground=THEME["field"], foreground=THEME["text"], bordercolor=THEME["border"], insertcolor=THEME["text"])
        style.configure("TSpinbox", fieldbackground=THEME["field"], foreground=THEME["text"], bordercolor=THEME["border"], arrowsize=13)
        style.configure("TCombobox", fieldbackground=THEME["field"], background=THEME["panel_raised"], foreground=THEME["text"], bordercolor=THEME["border"], arrowcolor=THEME["muted"])
        style.map("TCombobox", fieldbackground=[("readonly", THEME["field"])], foreground=[("readonly", THEME["text"])])
        style.configure("TCheckbutton", background=THEME["panel"], foreground=THEME["text"], focuscolor=THEME["accent"])
        style.map("TCheckbutton", background=[("active", THEME["panel"])], foreground=[("active", THEME["text"])])
        style.configure("Vertical.TScrollbar", background=THEME["panel_raised"], troughcolor=THEME["field"], bordercolor=THEME["bg"], arrowcolor=THEME["muted"])
        style.configure("Treeview", background=THEME["field"], fieldbackground=THEME["field"], foreground=THEME["text"], rowheight=26, borderwidth=0)
        style.configure("Treeview.Heading", background=THEME["panel_raised"], foreground=THEME["text"], relief=tk.FLAT, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", THEME["selection"])], foreground=[("selected", "#ffffff")])

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=16, style="TFrame")
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        ttk.Label(root, text="Foundry Local RAG", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )
        header_actions = ttk.Frame(root)
        header_actions.grid(row=0, column=1, sticky="e", pady=(0, 12))
        self.theme_button = ttk.Button(
            header_actions,
            text="☀" if self.theme_var.get() == "Dark" else "☾",
            command=self._toggle_theme,
            style="Theme.TButton",
            width=2,
        )
        self.theme_button.grid(row=0, column=0, sticky="e", padx=(0, 8))
        self.sidebar_toggle = ttk.Button(
            header_actions, text="Show knowledge files", command=self._toggle_knowledge_panel
        )
        self.sidebar_toggle.grid(row=0, column=1, sticky="e")

        self.sidebar = ttk.LabelFrame(root, text="Knowledge files", padding=10)
        self.sidebar.configure(width=self.sidebar_width)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self.sidebar.rowconfigure(0, weight=1)
        self.sidebar.columnconfigure(0, weight=1)
        self.document_list = tk.Listbox(self.sidebar, width=34, selectmode=tk.EXTENDED, activestyle="none", background=THEME["field"], foreground=THEME["text"], selectbackground=THEME["selection"], selectforeground="#ffffff", highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"], relief=tk.FLAT, borderwidth=0)
        self.document_list.grid(row=0, column=0, sticky="nsew")
        self.document_list.bind("<Double-Button-1>", self._open_selected_document)
        self.document_list.bind("<<ListboxSelect>>", self._update_document_metadata)
        scrollbar = ttk.Scrollbar(
            self.sidebar, orient=tk.VERTICAL, command=self.document_list.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.document_list.configure(yscrollcommand=scrollbar.set)
        self.drop_hint = tk.Label(
            self.sidebar,
            text="Drop files here",
            background=THEME["panel_raised"],
            foreground=THEME["accent"],
            relief=tk.FLAT,
            padx=8,
            pady=10,
        )
        self.drop_hint.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        for drop_target in (self.document_list, self.drop_hint):
            drop_target.drop_target_register(DND_FILES)
            drop_target.dnd_bind("<<DropEnter>>", self._on_drop_enter)
            drop_target.dnd_bind("<<DropLeave>>", self._on_drop_leave)
            drop_target.dnd_bind("<<Drop>>", self._on_files_dropped)
        self.import_button = ttk.Button(
            self.sidebar, text="Import files", command=self._choose_documents, style="Accent.TButton"
        )
        self.import_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.remove_button = ttk.Button(
            self.sidebar,
            text="Remove selected file",
            command=self._remove_selected_document,
            style="Danger.TButton",
        )
        self.remove_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.refresh_selected_button = ttk.Button(
            self.sidebar,
            text="Refresh selected file",
            command=self._refresh_selected_document,
        )
        self.refresh_selected_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.rebuild_button = ttk.Button(
            self.sidebar, text="Rebuild all files", command=self._rebuild_index
        )
        self.rebuild_button.grid(row=5, column=0, columnspan=2, sticky="ew")
        library_tools = ttk.Frame(self.sidebar, style="Panel.TFrame")
        library_tools.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        library_tools.columnconfigure(0, weight=1)
        search_entry = ttk.Entry(library_tools, textvariable=self.document_filter_var)
        search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        sort_box = ttk.Combobox(
            library_tools,
            textvariable=self.document_sort_var,
            values=("Name", "Newest", "Largest", "Type"),
            state="readonly",
            width=8,
        )
        sort_box.grid(row=0, column=1, sticky="e")
        self.document_filter_var.trace_add("write", lambda *_args: self._refresh_document_list())
        sort_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_document_list())
        self.document_metadata = ttk.Label(
            self.sidebar,
            text="Search files or select one for details",
            style="Subtle.TLabel",
            wraplength=245,
        )
        self.document_metadata.grid(row=7, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Label(
            self.sidebar,
            text="Supported files\nTXT · MD · PDF · DOCX · DOC · PPTX · PPT",
            style="Subtle.TLabel",
            wraplength=230,
            justify=tk.LEFT,
        ).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        settings = ttk.LabelFrame(self.sidebar, text="Retrieval settings", padding=8)
        settings.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        settings.columnconfigure(1, weight=1)
        self.preset_label = ttk.Label(settings, text="Preset")
        self.preset_label.grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.retrieval_preset_box = ttk.Combobox(
            settings,
            textvariable=self.retrieval_preset_var,
            values=(*RETRIEVAL_PRESETS, "Custom"),
            state="readonly",
            width=10,
        )
        self.retrieval_preset_box.grid(row=0, column=1, sticky="ew", pady=(0, 4))
        self.retrieval_preset_box.bind("<<ComboboxSelected>>", self._apply_retrieval_preset)
        self.top_k_label = ttk.Label(settings, text="Top K")
        self.top_k_label.grid(row=1, column=0, sticky="w", pady=(0, 4))
        self.top_k_entry = ttk.Entry(settings, textvariable=self.top_k_var, width=8)
        self.top_k_entry.grid(row=1, column=1, sticky="ew", pady=(0, 4))
        self.neighbors_label = ttk.Label(settings, text="Neighbors")
        self.neighbors_label.grid(row=2, column=0, sticky="w", pady=(0, 4))
        self.neighbor_spinbox = ttk.Spinbox(settings, from_=0, to=5, textvariable=self.neighbor_window_var, width=6)
        self.neighbor_spinbox.grid(row=2, column=1, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(settings, text="Use embeddings", variable=self.use_embeddings_var).grid(row=3, column=0, columnspan=2, sticky="w")
        self.answer_mode_label = ttk.Label(settings, text="Answer mode")
        self.answer_mode_label.grid(row=4, column=0, sticky="w", pady=(4, 4))
        mode_box = ttk.Combobox(
            settings,
            textvariable=self.answer_mode_var,
            values=("Strict", "Flexible"),
            state="readonly",
            width=10,
        )
        mode_box.grid(row=4, column=1, sticky="ew", pady=(4, 4))
        ttk.Checkbutton(settings, text="Show diagnostics", variable=self.show_diagnostics_var).grid(row=5, column=0, columnspan=2, sticky="w")
        for combo_box in (self.retrieval_preset_box, sort_box, mode_box):
            combo_box.bind(
                "<<ComboboxSelected>>",
                self._clear_combobox_highlight,
                add="+",
            )
        ttk.Button(settings, text="Apply and rebuild", command=lambda: self._rebuild_index("Applying settings..."), style="Accent.TButton").grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._apply_retrieval_preset()
        self.sidebar.grid_remove()
        chat = ttk.Frame(root)
        chat.grid(row=1, column=1, sticky="nsew")
        chat.columnconfigure(0, weight=1)
        chat.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(chat)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(0, weight=1)
        self.clear_chat_button = ttk.Button(toolbar, text="Clear chat", command=self._clear_chat)
        self.clear_chat_button.grid(row=0, column=1, sticky="e", padx=(0, 6))
        self.copy_answer_button = ttk.Button(toolbar, text="Copy answer", command=self._copy_last_answer)
        self.copy_answer_button.grid(row=0, column=2, sticky="e")
        self.inspect_retrieval_button = ttk.Button(
            toolbar,
            text="Inspect retrieval",
            command=self._show_retrieval_inspector,
            state=tk.DISABLED,
        )
        self.inspect_retrieval_button.grid(row=0, column=3, sticky="e", padx=(6, 0))

        self.transcript = ScrolledText(
            chat,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 11),
            padx=14,
            pady=14,
            background=THEME["field"],
            foreground=THEME["text"],
            insertbackground=THEME["text"],
            selectbackground=THEME["selection"],
            selectforeground="#ffffff",
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
        )
        self.transcript.grid(row=1, column=0, columnspan=2, sticky="nsew")
        _style_scrolled_text(self.transcript)
        self.transcript.tag_configure("user", foreground=THEME["accent"], font=("Segoe UI", 11, "bold"))
        self.transcript.tag_configure("assistant", foreground=THEME["success"], font=("Segoe UI", 11, "bold"))
        self.transcript.tag_configure("sources", foreground=THEME["muted"], font=("Segoe UI", 9))
        self.transcript.tag_configure("confidence_high", foreground=THEME["success"], font=("Segoe UI", 9, "bold"))
        self.transcript.tag_configure("confidence_medium", foreground=THEME["warning"], font=("Segoe UI", 9, "bold"))
        self.transcript.tag_configure("confidence_low", foreground=THEME["danger"], font=("Segoe UI", 9, "bold"))
        self.transcript.tag_configure("source_link", foreground=THEME["accent_hover"], underline=True, font=("Segoe UI", 9))
        self.question = tk.Text(chat, height=3, wrap=tk.WORD, font=("Segoe UI", 11), background=THEME["field"], foreground=THEME["text"], insertbackground=THEME["text"], selectbackground=THEME["selection"], selectforeground="#ffffff", relief=tk.FLAT, borderwidth=0, highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"], padx=10, pady=8)
        self.question.grid(row=2, column=0, sticky="ew", pady=(10, 0), padx=(0, 8))
        self.question.bind("<Return>", self._send_from_event)
        self.send_button = ttk.Button(chat, text="Send", command=self._send_question, style="Accent.TButton")
        self.send_button.grid(row=2, column=1, sticky="nsew", pady=(10, 0))

        self.status = tk.StringVar(value="STARTING")
        self.status_label = tk.Label(
            root,
            textvariable=self.status,
            background=THEME["status_busy"],
            foreground=THEME["text"],
            font=("Segoe UI", 10),
            padx=14,
            pady=7,
            anchor="w",
        )
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.index_activity = ttk.Frame(root)
        self.index_activity.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.index_activity.columnconfigure(0, weight=1)
        self.index_progress = ttk.Progressbar(self.index_activity, mode="indeterminate")
        self.index_progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.cancel_index_button = ttk.Button(
            self.index_activity, text="Cancel indexing", command=self._cancel_indexing
        )
        self.cancel_index_button.grid(row=0, column=1, sticky="e")
        self.index_activity.grid_remove()

    def _add_tooltips(self) -> None:
        add_tooltip(self.preset_label, "Choose a safe retrieval profile. Custom unlocks the detailed controls.")
        add_tooltip(self.top_k_label, "Number of initial matching chunks. Auto adapts to the collection and model.")
        add_tooltip(self.neighbors_label, "Nearby chunks added around each direct hit. Lower values reduce unrelated context.")
        add_tooltip(self.answer_mode_label, "Strict keeps source wording. Flexible adapts grounded evidence to the question type.")
        add_tooltip(self.theme_button, "Switch between dark and light mode.")
        add_tooltip(self.inspect_retrieval_button, "Inspect ranking signals and the full text of every retrieved chunk.")

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-o>", lambda _event: self._choose_documents())
        self.bind_all("<Control-l>", lambda _event: self.question.focus_set())
        self.bind_all("<Control-k>", lambda _event: self._toggle_knowledge_panel())
        self.bind_all("<Control-Shift-I>", lambda _event: self._show_retrieval_inspector())
        self.bind_all("<F5>", lambda _event: self._rebuild_index())
        self.bind_all(
            "<Escape>",
            lambda _event: self._cancel_indexing() if self.index_activity.winfo_ismapped() else None,
        )
        self.document_list.bind("<Delete>", lambda _event: self._remove_selected_document())

    def _toggle_knowledge_panel(self) -> None:
        opening = not self.sidebar_visible
        self.sidebar_visible = opening
        if self.sidebar_animation is not None:
            self.after_cancel(self.sidebar_animation)
            self.sidebar_animation = None
        if opening:
            self.sidebar.grid()
            self.sidebar_toggle.configure(text="Hide knowledge files")
        else:
            self.sidebar_toggle.configure(text="Show knowledge files")
        self._animate_sidebar(opening)

    def _animate_sidebar(self, opening: bool, frame: int = 0) -> None:
        frames = 14
        progress = min(1.0, frame / frames)
        eased = 1 - (1 - progress) ** 3 if opening else (1 - progress) ** 3
        self.sidebar.configure(width=max(1, round(self.sidebar_width * eased)))
        if frame < frames:
            self.sidebar_animation = self.after(
                15, lambda: self._animate_sidebar(opening, frame + 1)
            )
            return
        self.sidebar_animation = None
        if not opening:
            self.sidebar.grid_remove()
    def _refresh_document_list(self) -> None:
        self.document_list.delete(0, tk.END)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        files = [
            path.relative_to(self.documents_dir) for path in self.documents_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in _SUPPORTED_SUFFIXES
        ]
        query = self.document_filter_var.get().strip().casefold()
        if query:
            files = [path for path in files if query in str(path).casefold()]
        sort_mode = self.document_sort_var.get()
        if sort_mode == "Newest":
            files.sort(key=lambda path: (self.documents_dir / path).stat().st_mtime, reverse=True)
        elif sort_mode == "Largest":
            files.sort(key=lambda path: (self.documents_dir / path).stat().st_size, reverse=True)
        elif sort_mode == "Type":
            files.sort(key=lambda path: (path.suffix.casefold(), str(path).casefold()))
        else:
            files.sort(key=lambda path: str(path).casefold())
        self.document_paths = files
        for path in files:
            self.document_list.insert(tk.END, str(path))
        if hasattr(self, "document_metadata"):
            self.document_metadata.configure(text=f"{len(files)} knowledge file(s)")

    def _update_document_metadata(self, _event: tk.Event | None = None) -> None:
        selection = self.document_list.curselection()
        if not selection:
            self.document_metadata.configure(text=f"{len(self.document_paths)} knowledge file(s)")
            return
        if len(selection) > 1:
            total = sum(
                (self.documents_dir / Path(self.document_list.get(index))).stat().st_size
                for index in selection
            )
            self.document_metadata.configure(
                text=f"{len(selection)} selected - {total / 1024:.1f} KB total"
            )
            return
        relative = Path(self.document_list.get(selection[0]))
        target = self.documents_dir / relative
        stat = target.stat()
        source = relative.as_posix()
        status = "Indexed" if source in self.indexed_file_signatures else "Pending index"
        chunks = self.document_chunk_counts.get(source, 0)
        pages = self.document_page_counts.get(source, 0)
        unit = "slides" if relative.suffix.casefold() in {".ppt", ".pptx"} else "pages"
        extraction = f"{chunks} chunks" + (f", {pages} {unit}" if pages else "")
        self.document_metadata.configure(
            text=f"{relative.suffix.upper().lstrip('.')} - {stat.st_size / 1024:.1f} KB - {status} - {extraction}"
        )

    def _choose_documents(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Import knowledge files",
            filetypes=[
                ("Supported documents", "*.txt *.md *.pdf *.docx *.doc *.pptx *.ppt"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self._import_selected_paths(paths)

    def _on_drop_enter(self, event: tk.Event) -> str:
        self.drop_hint.configure(
            text="Release to import", background=THEME["border"], relief=tk.FLAT
        )
        return event.action

    def _on_drop_leave(self, event: tk.Event) -> str:
        self._reset_drop_hint()
        return event.action

    def _on_files_dropped(self, event: tk.Event) -> str:
        self._reset_drop_hint()
        paths = tuple(self.tk.splitlist(event.data))
        self._import_selected_paths(paths)
        return event.action

    def _reset_drop_hint(self) -> None:
        self.drop_hint.configure(
            text="Drop files here", background=THEME["panel_raised"], relief=tk.FLAT
        )

    def _clear_chat(self) -> None:
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.delete("1.0", tk.END)
        self.transcript.configure(state=tk.DISABLED)
        self.source_link_payloads.clear()
        self.last_answer = None
        self.last_evidence_answer = None
        self.inspect_retrieval_button.configure(state=tk.DISABLED)

    def _copy_last_answer(self) -> None:
        if self.last_answer is None or not self.last_answer.text.strip():
            messagebox.showinfo("No answer", "There is no answer to copy yet.", parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(self.last_answer.text)
        self.status.set("ANSWER COPIED")

    def _show_retrieval_inspector(self) -> None:
        answer = self.last_evidence_answer
        if answer is None or not answer.results:
            messagebox.showinfo("No retrieval", "Ask a question before inspecting retrieval.", parent=self)
            return

        window = tk.Toplevel(self)
        apply_window_frame(window, THEME, self.theme_var.get())
        window.title("Retrieval inspector")
        window.geometry("980x640")
        window.minsize(720, 460)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=3)
        window.rowconfigure(2, weight=2)

        summary = ttk.Frame(window, padding=(12, 10, 12, 8))
        summary.grid(row=0, column=0, sticky="ew")
        summary.columnconfigure(0, weight=1)
        ttk.Label(summary, text="Retrieved evidence", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            summary,
            text=f"{len(answer.results)} chunks inspected - select a row to view its full text",
            style="Subtle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        table_frame = ttk.Frame(window, padding=(12, 0, 12, 8))
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("rank", "source", "page", "kind", "score", "confidence", "lexical", "semantic")
        table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        headings = {
            "rank": "#",
            "source": "Source",
            "page": "Page / slide",
            "kind": "Reason",
            "score": "Score",
            "confidence": "Confidence",
            "lexical": "Lexical",
            "semantic": "Semantic",
        }
        widths = {"rank": 42, "source": 230, "page": 55, "kind": 90, "score": 78, "confidence": 90, "lexical": 78, "semantic": 78}
        for column in columns:
            table.heading(column, text=headings[column])
            table.column(column, width=widths[column], anchor="w" if column == "source" else "center")
        table.grid(row=0, column=0, sticky="nsew")
        table_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
        table_scrollbar.grid(row=0, column=1, sticky="ns")
        table.configure(yscrollcommand=table_scrollbar.set)

        detail = ScrolledText(
            window,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 10),
            padx=12,
            pady=12,
            background=THEME["field"],
            foreground=THEME["text"],
            selectbackground=THEME["selection"],
            selectforeground="#ffffff",
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=THEME["border"],
        )
        detail.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        _style_scrolled_text(detail)
        detail.tag_configure("metadata", foreground=THEME["muted"], font=("Segoe UI", 9, "bold"))

        results_by_item: dict[str, object] = {}
        for rank, result in enumerate(answer.results, 1):
            kind = "Neighbor" if result.lexical_score == 0 and result.semantic_score == 0 else "Direct hit"
            item = table.insert(
                "",
                tk.END,
                values=(
                    rank,
                    result.chunk.source,
                    result.chunk.page or "-",
                    kind,
                    f"{result.score:.3f}",
                    f"{result.confidence:.3f}",
                    f"{result.lexical_score:.3f}",
                    f"{result.semantic_score:.3f}",
                ),
            )
            results_by_item[item] = result

        def show_selected(_event: tk.Event | None = None) -> None:
            selection = table.selection()
            if not selection:
                return
            result = results_by_item[selection[0]]
            metadata = [result.chunk.source]
            if result.chunk.page:
                location = "slide" if result.chunk.source.casefold().endswith((".ppt", ".pptx")) else "page"
                metadata.append(f"{location} {result.chunk.page}")
            if result.chunk.heading:
                metadata.append(result.chunk.heading)
            detail.configure(state=tk.NORMAL)
            detail.delete("1.0", tk.END)
            detail.insert(tk.END, " | ".join(metadata) + "\n\n", "metadata")
            detail.insert(tk.END, result.chunk.text)
            detail.configure(state=tk.DISABLED)

        table.bind("<<TreeviewSelect>>", show_selected)
        first_item = table.get_children()
        if first_item:
            table.selection_set(first_item[0])
            table.focus(first_item[0])
            show_selected()

    def _apply_retrieval_preset(self, _event: tk.Event | None = None) -> None:
        preset = self.retrieval_preset_var.get()
        values = RETRIEVAL_PRESETS.get(preset)
        if values is not None:
            top_k, neighbors = values
            self.top_k_var.set(top_k)
            self.neighbor_window_var.set(neighbors)
        custom_state = tk.NORMAL if preset == "Custom" else tk.DISABLED
        self.top_k_entry.configure(state=custom_state)
        self.neighbor_spinbox.configure(state=custom_state)

    def _clear_combobox_highlight(self, event: tk.Event) -> None:
        combo_box = event.widget

        def clear() -> None:
            try:
                combo_box.selection_clear()
                self.focus_set()
            except tk.TclError:
                pass

        self.after_idle(clear)

    def _apply_theme(self, _event: tk.Event | None = None) -> None:
        THEME.clear()
        THEME.update(theme_palette(self.theme_var.get()))
        self._configure_style()
        self.document_list.configure(
            background=THEME["field"], foreground=THEME["text"],
            selectbackground=THEME["selection"], highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
        )
        self.drop_hint.configure(background=THEME["panel_raised"], foreground=THEME["accent"])
        for text_widget in (self.transcript, self.question):
            text_widget.configure(
                background=THEME["field"], foreground=THEME["text"],
                insertbackground=THEME["text"], selectbackground=THEME["selection"],
                highlightbackground=THEME["border"], highlightcolor=THEME["accent"],
            )
        _style_scrolled_text(self.transcript)
        self.transcript.tag_configure("user", foreground=THEME["accent"])
        self.transcript.tag_configure("assistant", foreground=THEME["success"])
        self.transcript.tag_configure("sources", foreground=THEME["muted"])
        self.transcript.tag_configure("confidence_high", foreground=THEME["success"])
        self.transcript.tag_configure("confidence_medium", foreground=THEME["warning"])
        self.transcript.tag_configure("confidence_low", foreground=THEME["danger"])
        self.transcript.tag_configure("source_link", foreground=THEME["accent_hover"])
        self.status_label.configure(
            background=THEME["status_busy"] if self.busy else THEME["status_ready"],
            foreground=THEME["text"],
        )

    def _toggle_theme(self) -> None:
        self.theme_var.set("Light" if self.theme_var.get() == "Dark" else "Dark")
        self._apply_theme()
        self.theme_button.configure(
            text="☀" if self.theme_var.get() == "Dark" else "☾"
        )

    def _resolved_top_k(self) -> int | None:
        raw_value = self.top_k_var.get().strip()
        if not raw_value or raw_value.casefold() == "auto":
            return None
        try:
            value = int(raw_value)
        except ValueError:
            raise ValueError("Top K must be Auto or a positive number") from None
        if value < 1:
            raise ValueError("Top K must be Auto or a positive number")
        return value

    def _resolved_neighbor_window(self) -> int:
        try:
            value = int(self.neighbor_window_var.get())
        except (tk.TclError, ValueError):
            raise ValueError("Neighbors must be a number from 0 to 5") from None
        if not 0 <= value <= 5:
            raise ValueError("Neighbors must be a number from 0 to 5")
        return value

    def _dependency_health(self, document_paths: list[Path], using_embeddings: bool) -> list[str]:
        suffixes = {path.suffix.casefold() for path in document_paths}
        messages: list[str] = []

        if ".pdf" in suffixes:
            if importlib.util.find_spec("pypdf") is None:
                messages.append("PDF support missing: install pypdf to read PDF files.")
            else:
                messages.append("PDF support: ok")

        if any(suffix in suffixes for suffix in {".doc", ".docx"}):
            messages.append("DOCX support: ok")
        if ".doc" in suffixes:
            if os.name != "nt":
                messages.append("Legacy .doc support unavailable on this OS; convert .doc files to .docx.")
            else:
                messages.append("Legacy .doc support requires Microsoft Word installed locally.")

        if ".pptx" in suffixes:
            messages.append("PowerPoint PPTX support: ok")
        if ".ppt" in suffixes:
            if os.name != "nt":
                messages.append("Legacy .ppt support unavailable on this OS; convert .ppt files to .pptx.")
            else:
                messages.append("Legacy .ppt support requires Microsoft PowerPoint installed locally.")

        ocr_modules = all(
            importlib.util.find_spec(module) is not None
            for module in ("fitz", "pytesseract", "PIL")
        )
        tesseract_available = shutil.which("tesseract") is not None
        if ocr_modules and tesseract_available:
            messages.append("OCR support: ok")
        else:
            missing: list[str] = []
            if not ocr_modules:
                missing.append("Python OCR packages")
            if not tesseract_available:
                missing.append("Tesseract executable")
            messages.append(f"OCR support unavailable: {', '.join(missing)}.")

        foundry_sdk_available = importlib.util.find_spec("foundry_local_sdk") is not None
        if using_embeddings:
            messages.append("Embeddings: active")
        elif self.use_embeddings_var.get() and not foundry_sdk_available:
            messages.append("Embeddings unavailable: Foundry Local SDK is not installed.")
        elif self.use_embeddings_var.get():
            messages.append("Embeddings unavailable: model missing or could not be loaded; using lexical retrieval.")
        else:
            messages.append("Embeddings: disabled in settings")

        return messages
    def _index_summary(
        self,
        document_paths: list[Path],
        skipped_paths: list[Path],
        chunk_count: int,
        neighbor_window: int,
        using_embeddings: bool,
    ) -> str:
        file_types = Counter(path.suffix.lower() or "[none]" for path in document_paths)
        type_summary = ", ".join(
            f"{suffix}: {count}" for suffix, count in sorted(file_types.items())
        ) or "none"
        retrieval_mode = "hybrid" if using_embeddings else "lexical"
        lines = [
            "Index ready",
            f"Documents: {len(document_paths)}",
            f"Chunks: {chunk_count}",
            f"Types: {type_summary}",
            f"Retrieval: {retrieval_mode}",
            f"Neighbors: {neighbor_window}",
        ]
        if skipped_paths:
            skipped_names = ", ".join(path.name for path in skipped_paths[:5])
            if len(skipped_paths) > 5:
                skipped_names += f", +{len(skipped_paths) - 5} more"
            lines.append(f"Skipped unsupported: {len(skipped_paths)} ({skipped_names})")
        health_messages = self._dependency_health(document_paths, using_embeddings)
        if health_messages:
            lines.append("Health:")
            lines.extend(f"- {message}" for message in health_messages)
        return "\n".join(lines)
    def _append_status_message(self, message: str) -> None:
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, f"System\n{message}\n\n", "sources")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)
    def _import_selected_paths(self, paths: tuple[str, ...]) -> None:
        try:
            imported = import_documents(paths, self.documents_dir)
        except OSError as exc:
            messagebox.showerror("Import failed", str(exc), parent=self)
            return
        if not imported:
            messagebox.showwarning(
                "Unsupported files", "No supported documents were selected.", parent=self
            )
            return
        self._refresh_document_list()
        self._rebuild_index(f"Imported {len(imported)} file(s). Rebuilding index...")
    def _selected_document_path(self) -> Path | None:
        selection = self.document_list.curselection()
        if not selection:
            messagebox.showinfo("Select a file", "Choose a knowledge file first.", parent=self)
            return None
        relative_path = Path(self.document_list.get(selection[0]))
        target = (self.documents_dir / relative_path).resolve()
        documents_root = self.documents_dir.resolve()
        if target != documents_root and documents_root not in target.parents:
            messagebox.showerror("Invalid file", "The selected path is not safe.", parent=self)
            return None
        return target

    def _open_selected_document(self, _event: tk.Event | None = None) -> None:
        target = self._selected_document_path()
        if target is None:
            return
        if not target.is_file():
            messagebox.showwarning("Missing file", "The selected file no longer exists.", parent=self)
            self._refresh_document_list()
            return
        try:
            os.startfile(target)
        except OSError as exc:
            messagebox.showwarning("Open file failed", str(exc), parent=self)

    def _refresh_selected_document(self) -> None:
        target = self._selected_document_path()
        if target is None:
            return
        if not target.exists():
            messagebox.showwarning("Missing file", "The selected file no longer exists.", parent=self)
            self._refresh_document_list()
            return
        relative_path = target.relative_to(self.documents_dir.resolve())
        self._rebuild_index(f"Refreshing {relative_path}...")
    def _remove_selected_document(self) -> None:
        selection = self.document_list.curselection()
        if not selection:
            messagebox.showinfo(
                "Select a file", "Choose a knowledge file to remove first.", parent=self
            )
            return
        documents_root = self.documents_dir.resolve()
        targets = []
        for index in selection:
            relative_path = Path(self.document_list.get(index))
            target = (self.documents_dir / relative_path).resolve()
            if documents_root not in target.parents:
                messagebox.showerror("Invalid file", "A selected path is not safe.", parent=self)
                return
            targets.append((relative_path, target))
        if not messagebox.askyesno(
            "Remove knowledge files",
            f"Remove {len(targets)} selected file(s) from the knowledge base?",
            parent=self,
        ):
            return
        try:
            for _relative_path, target in targets:
                target.unlink()
        except OSError as exc:
            messagebox.showerror("Removal failed", str(exc), parent=self)
            return
        self._refresh_document_list()
        self._rebuild_index(f"Removed {len(targets)} file(s). Updating index...")
    def _set_busy(self, busy: bool, message: str) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.import_button.configure(state=state)
        self.remove_button.configure(state=state)
        self.rebuild_button.configure(state=state)
        self.refresh_selected_button.configure(state=state)
        self.send_button.configure(state=state)
        self.status.set(message.upper())
        self.status_label.configure(
            background=THEME["status_busy"] if busy else THEME["status_ready"],
            foreground=THEME["text"],
        )

    def _cancel_indexing(self) -> None:
        self.index_cancel_event.set()
        self.cancel_index_button.configure(state=tk.DISABLED)
        self.status.set("CANCELLING INDEX UPDATE...")
    def _rebuild_index(self, message: str = "Rebuilding index...") -> None:
        if self.busy:
            return
        try:
            neighbor_window = self._resolved_neighbor_window()
        except ValueError as exc:
            messagebox.showwarning("Invalid settings", str(exc), parent=self)
            return
        all_file_paths = [path for path in self.documents_dir.rglob("*") if path.is_file()]
        document_paths = [
            path
            for path in all_file_paths
            if path.suffix.casefold() in _SUPPORTED_SUFFIXES
        ]
        signatures = {
            path.relative_to(self.documents_dir).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in document_paths
        }
        changed_sources = {
            source for source, signature in signatures.items()
            if self.indexed_file_signatures.get(source) != signature
        }
        removed_sources = set(self.indexed_file_signatures) - set(signatures)
        skipped_paths = [
            path
            for path in all_file_paths
            if path.suffix.casefold() not in _SUPPORTED_SUFFIXES
        ]
        has_documents = bool(document_paths)
        if not has_documents:
            with self.pipeline_lock:
                if self.pipeline is not None:
                    self.pipeline.close()
                    self.pipeline = None
            self._set_busy(False, "No knowledge files - import a document")
            self.send_button.configure(state=tk.DISABLED)
            self.status_label.configure(background=THEME["status_busy"], foreground=THEME["text"])
            return
        self._set_busy(True, message)
        self.index_cancel_event.clear()
        self.cancel_index_button.configure(state=tk.NORMAL)
        self.index_activity.grid()
        self.index_progress.start(12)

        def worker() -> None:
            new_pipeline = None
            embedder = None
            try:
                # Foundry returns handles to a shared model instance. Close the old
                # owner before creating the replacement; otherwise closing it after
                # construction can unload the model used by the new client.
                with self.pipeline_lock:
                    old_pipeline, self.pipeline = self.pipeline, None
                    if old_pipeline is not None:
                        old_pipeline.close()
                def report_embedding_progress(current: int, total: int) -> None:
                    self.after(
                        0,
                        lambda current=current, total=total: self.status.set(
                            f"EMBEDDING {current}/{total} CHUNKS" if current < total else "FINALIZING INDEX..."
                        ),
                    )

                embedder = optional_foundry_embedder(
                    "qwen3-embedding-0.6b",
                    cache_dir=self.cache_dir,
                    progress_callback=report_embedding_progress,
                    cancel_check=self.index_cancel_event.is_set,
                ) if self.use_embeddings_var.get() else None
                using_embeddings = embedder is not None
                def report_progress(source: str, current: int, total: int) -> None:
                    self.after(
                        0,
                        lambda source=source, current=current, total=total: self.status.set(
                            f"INDEXING {current}/{total}: {source}".upper()
                        ),
                    )
                new_pipeline = RAGPipeline.from_directory(
                    self.documents_dir,
                    embedder=embedder,
                    cache_dir=self.cache_dir,
                    neighbor_window=neighbor_window,
                    progress_callback=report_progress,
                    cancel_check=self.index_cancel_event.is_set,
                )
                if self.index_cancel_event.is_set():
                    new_pipeline.close()
                    new_pipeline = None
                    self.after(0, self._finish_cancelled_index)
                    return
                embedder = None
                chunk_count = len(new_pipeline.retriever.chunks)
                chunk_counts = Counter(
                    chunk.source for chunk in new_pipeline.retriever.chunks
                )
                page_sets: dict[str, set[int]] = {}
                for chunk in new_pipeline.retriever.chunks:
                    if chunk.page:
                        page_sets.setdefault(chunk.source, set()).add(chunk.page)
                page_counts = {source: len(pages) for source, pages in page_sets.items()}
                with self.pipeline_lock:
                    self.pipeline = new_pipeline
                    new_pipeline = None
                summary = self._index_summary(
                    document_paths,
                    skipped_paths,
                    chunk_count,
                    neighbor_window,
                    using_embeddings,
                )
                summary += (
                    f"\nIncremental update: {len(changed_sources)} changed, "
                    f"{len(signatures) - len(changed_sources)} reused, {len(removed_sources)} removed"
                )
                self.after(
                    0,
                    lambda summary=summary, signatures=signatures, chunk_counts=chunk_counts, page_counts=page_counts: self._finish_index_update(
                        summary, signatures, chunk_counts, page_counts
                    ),
                )
            except Exception as exc:
                if new_pipeline is not None:
                    new_pipeline.close()
                elif embedder is not None:
                    embedder.close()
                if self.index_cancel_event.is_set() and str(exc) == "Indexing cancelled":
                    self.after(0, self._finish_cancelled_index)
                    return
                LOGGER.exception("Could not build the RAG index")
                message = str(exc)
                self.after(0, lambda message=message: self._show_error("Indexing failed", message))

        threading.Thread(target=worker, daemon=True, name="rag-indexer").start()

    def _finish_index_update(
        self,
        summary: str,
        signatures: dict[str, tuple[int, int]],
        chunk_counts: Counter[str],
        page_counts: dict[str, int],
    ) -> None:
        self.index_progress.stop()
        self.index_activity.grid_remove()
        self.indexed_file_signatures = signatures
        self.document_chunk_counts = chunk_counts
        self.document_page_counts = page_counts
        self._set_busy(False, "Ready")
        self._refresh_document_list()
        self._append_status_message(summary)

    def _finish_cancelled_index(self) -> None:
        self.index_progress.stop()
        self.index_activity.grid_remove()
        self._set_busy(False, "Index update cancelled")

    def _send_from_event(self, event: tk.Event) -> str | None:
        if event.state & 0x0001:
            return None
        self._send_question()
        return "break"
    def _send_question(self) -> None:
        question = self.question.get("1.0", tk.END).strip()
        if not question or self.busy:
            return
        if self.pipeline is None:
            messagebox.showwarning("Not ready", "Import a document and build the index first.", parent=self)
            return
        try:
            top_k = self._resolved_top_k()
        except ValueError as exc:
            messagebox.showwarning("Invalid settings", str(exc), parent=self)
            return
        self.question.delete("1.0", tk.END)
        self._append_message("You", question, "user")
        self._set_busy(True, "Searching documents…")

        def worker() -> None:
            try:
                with self.pipeline_lock:
                    if self.pipeline is None:
                        raise RuntimeError("The document index is not available")
                    answer = self.pipeline.ask(question, top_k)
                self.after(0, lambda: self._display_answer(answer, question))
            except Exception as exc:
                LOGGER.exception("RAG request failed")
                message = str(exc)
                self.after(0, lambda message=message: self._show_error("Request failed", message))

        threading.Thread(target=worker, daemon=True, name="rag-query").start()

    def _append_message(self, speaker: str, body: str, tag: str) -> None:
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, f"{speaker}\n", tag)
        self.transcript.insert(tk.END, f"{body}\n\n")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)

    def _answer_for_mode(self, answer: Answer, question: str = "") -> Answer:
        if self.answer_mode_var.get() != "Flexible" or not answer.sources:
            return answer
        flexible_text = flexible_answer_text(question, answer.text)
        return replace(answer, text=flexible_text)

    def _confidence_summary(self, answer: Answer) -> tuple[str, str, str]:
        if not answer.sources:
            reason = "; ".join(answer.diagnostics) if answer.diagnostics else "no cited source"
            return "Low", "confidence_low", f"No supported answer was found ({reason})."

        best_confidence = max((result.confidence for result in answer.results), default=0.0)
        retrieved_text = "\n".join(result.chunk.text for result in answer.results)
        answer_terms = {
            token.casefold()
            for token in re.findall(r"[^\W\d_]+|\d+", answer.text, flags=re.UNICODE)
            if len(token) > 3
        }
        retrieved_terms = {
            token.casefold()
            for token in re.findall(r"[^\W\d_]+|\d+", retrieved_text, flags=re.UNICODE)
        }
        term_overlap = len(answer_terms & retrieved_terms) / max(1, len(answer_terms))
        exact_answer_seen = bool(answer.text and answer.text in retrieved_text)

        reasons: list[str] = []
        if exact_answer_seen:
            reasons.append("answer text appears verbatim in retrieved evidence")
        elif term_overlap >= 0.75:
            reasons.append("answer terms are well covered by retrieved evidence")
        else:
            reasons.append("answer has weak overlap with retrieved evidence")
        reasons.append(f"top retrieval confidence {best_confidence:.2f}")
        if answer.diagnostics:
            reasons.append("diagnostics present")

        if exact_answer_seen or (best_confidence >= 0.55 and term_overlap >= 0.75):
            return "High", "confidence_high", "; ".join(reasons)
        if best_confidence >= 0.30 and term_overlap >= 0.50:
            return "Medium", "confidence_medium", "; ".join(reasons)
        return "Low", "confidence_low", "; ".join(reasons)
    def _diagnostic_text(self, answer: Answer) -> str:
        lines: list[str] = []
        if answer.diagnostics:
            lines.append(f"Diagnostics: {'; '.join(answer.diagnostics)}")
        if answer.results:
            lines.append("Retrieved snippets:")
            for index, result in enumerate(answer.results[:5], 1):
                snippet = re.sub(r"\s+", " ", result.chunk.text).strip()
                if len(snippet) > 220:
                    snippet = snippet[:217].rstrip() + "..."
                score = f"score={result.score:.3f}, confidence={result.confidence:.3f}"
                lines.append(f"{index}. {result.chunk.source} ({score}): {snippet}")
        return "\n".join(lines)

    def _display_answer(self, answer: Answer, question: str = "") -> None:
        evidence_answer = answer
        display_answer = self._answer_for_mode(answer, question)
        self.last_answer = display_answer
        self.last_evidence_answer = evidence_answer
        self.inspect_retrieval_button.configure(
            state=tk.NORMAL if evidence_answer.results else tk.DISABLED
        )
        self._append_message("Assistant", display_answer.text, "assistant")
        level, confidence_tag, confidence_reason = self._confidence_summary(evidence_answer)
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, "Confidence: ", "sources")
        self.transcript.insert(tk.END, level, confidence_tag)
        self.transcript.insert(tk.END, f" - {confidence_reason}\n", "sources")
        if evidence_answer.sources or evidence_answer.diagnostics:
            if evidence_answer.sources:
                self.transcript.insert(tk.END, "Sources: ", "sources")
                for index, source in enumerate(evidence_answer.sources):
                    if index:
                        self.transcript.insert(tk.END, ", ", "sources")
                    self._insert_source_link(source, evidence_answer)
                self.transcript.insert(tk.END, "\n", "sources")
            if self.show_diagnostics_var.get():
                diagnostic_text = self._diagnostic_text(evidence_answer)
                if diagnostic_text:
                    self.transcript.insert(tk.END, diagnostic_text + "\n", "sources")
        self.transcript.insert(tk.END, "\n")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)
        self._set_busy(False, "Ready")
        self.question.focus_set()

    def _insert_source_link(self, source: str, answer: Answer) -> None:
        self.source_link_counter += 1
        tag = f"source_link_{self.source_link_counter}"
        self.source_link_payloads[tag] = (source, answer)
        self.transcript.insert(tk.END, source, ("sources", "source_link", tag))
        source_name = re.sub(r"\s+\((?:page|slide)\s+\d+\)$", "", source).strip()
        heading = next(
            (
                result.chunk.heading
                for result in answer.results
                if result.chunk.source == source_name and result.chunk.heading
            ),
            None,
        )
        if heading:
            self.transcript.insert(tk.END, f" - {heading}", "sources")
        self.transcript.tag_bind(tag, "<Button-1>", lambda _event, tag=tag: self._open_source_link(tag))
        self.transcript.tag_bind(tag, "<Enter>", lambda _event: self.transcript.configure(cursor="hand2"))
        self.transcript.tag_bind(tag, "<Leave>", lambda _event: self.transcript.configure(cursor=""))

    def _open_source_link(self, tag: str) -> None:
        payload = self.source_link_payloads.get(tag)
        if payload is None:
            return
        source, answer = payload
        source_path = self._source_file_path(source)
        if not source_path.exists():
            messagebox.showwarning("Source not found", f"Could not find {source_path}", parent=self)
            return
        self._show_source_context(source, answer, source_path)

    def _source_file_path(self, source: str) -> Path:
        source_name = re.sub(r"\s+\((?:page|slide)\s+\d+\)$", "", source).strip()
        return self.documents_dir / Path(source_name)

    def _source_context_chunks(self, source: str, answer: Answer) -> list[dict[str, object]]:
        source_name = re.sub(r"\s+\((?:page|slide)\s+\d+\)$", "", source).strip()
        answer_terms = {
            token.casefold()
            for token in re.findall(r"[^\W\d_]+|\d+", answer.text, flags=re.UNICODE)
            if len(token) > 3
        }
        ranked: list[tuple[float, int, dict[str, object]]] = []
        for result in answer.results:
            if result.chunk.source != source_name:
                continue
            text = result.chunk.text.strip()
            if not text:
                continue
            text_terms = {
                token.casefold()
                for token in re.findall(r"[^\W\d_]+|\d+", text, flags=re.UNICODE)
            }
            matched_terms = sorted(answer_terms & text_terms)
            exact_bonus = 5 if answer.text and answer.text in text else 0
            score = exact_bonus + len(matched_terms) + result.score
            ranked.append(
                (
                    score,
                    result.position,
                    {
                        "text": text,
                        "source": result.chunk.source,
                        "page": result.chunk.page,
                        "heading": result.chunk.heading,
                        "score": result.score,
                        "confidence": result.confidence,
                        "lexical_score": result.lexical_score,
                        "semantic_score": result.semantic_score,
                        "matched_terms": matched_terms,
                        "position": result.position,
                    },
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [payload for _score, _position, payload in ranked[:8]]

    def _highlight_text_span(self, viewer: ScrolledText, tag: str, start: str, haystack: str, needle: str) -> None:
        if not needle:
            return
        offset = haystack.find(needle)
        if offset < 0:
            return
        viewer.tag_add(tag, f"{start}+{offset}c", f"{start}+{offset + len(needle)}c")

    def _highlight_answer_terms(self, viewer: ScrolledText, start: str, text: str, answer: Answer) -> None:
        terms = sorted(
            {
                token
                for token in re.findall(r"[^\W\d_]+|\d+", answer.text, flags=re.UNICODE)
                if len(token) > 3
            },
            key=len,
            reverse=True,
        )
        for term in terms:
            for match in re.finditer(rf"(?i)(?<!\w){re.escape(term)}(?!\w)", text):
                viewer.tag_add("term", f"{start}+{match.start()}c", f"{start}+{match.end()}c")

    def _show_source_context(self, source: str, answer: Answer, source_path: Path) -> None:
        chunks = self._source_context_chunks(source, answer)
        if not chunks:
            chunks = [
                {
                    "text": "No retrieved snippet is available for this source.",
                    "source": source,
                    "page": None,
                    "heading": None,
                    "score": 0.0,
                    "confidence": 0.0,
                    "lexical_score": 0.0,
                    "semantic_score": 0.0,
                    "matched_terms": [],
                    "position": 0,
                }
            ]

        window = tk.Toplevel(self)
        apply_window_frame(window, THEME, self.theme_var.get())
        window.title(f"Evidence - {source}")
        window.geometry("880x560")
        window.minsize(620, 400)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(2, weight=1)

        header = ttk.Frame(window, padding=(12, 10, 12, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=source, font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=str(source_path), style="Subtle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Button(header, text="Open document", command=lambda: os.startfile(source_path) if source_path.exists() else None).grid(row=0, column=1, rowspan=2, sticky="e")

        controls = ttk.Frame(window, padding=(12, 0, 12, 8))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        current_index = tk.IntVar(value=0)
        counter = ttk.Label(controls)
        counter.grid(row=0, column=1, sticky="ew", padx=8)
        previous_button = ttk.Button(controls, text="Previous")
        previous_button.grid(row=0, column=0, sticky="w")
        next_button = ttk.Button(controls, text="Next")
        next_button.grid(row=0, column=2, sticky="e")

        viewer = ScrolledText(window, wrap=tk.WORD, font=("Segoe UI", 10), padx=12, pady=12, background=THEME["field"], foreground=THEME["text"], insertbackground=THEME["text"], selectbackground=THEME["selection"], selectforeground="#ffffff", relief=tk.FLAT, borderwidth=0, highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"])
        _style_scrolled_text(viewer)
        viewer.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        viewer.tag_configure("metadata", foreground=THEME["muted"], font=("Segoe UI", 9, "bold"))
        viewer.tag_configure("evidence", background="#1f2a36")
        viewer.tag_configure("answer", background="#21412d", foreground=THEME["text"], font=("Segoe UI", 10, "bold"))
        viewer.tag_configure("term", background="#223b5c", foreground=THEME["text"])
        viewer.tag_configure("label", foreground=THEME["muted"], font=("Segoe UI", 9, "bold"))

        def render() -> None:
            index = current_index.get()
            payload = chunks[index]
            chunk_text = str(payload["text"])
            page = payload.get("page")
            heading = payload.get("heading")
            matched_terms = payload.get("matched_terms") or []
            source_bits = [f"Chunk {index + 1} of {len(chunks)}"]
            if page:
                location = "slide" if source_path.suffix.casefold() in {".ppt", ".pptx"} else "page"
                source_bits.append(f"{location} {page}")
            if heading:
                source_bits.append(str(heading))
            score_line = (
                f"score={float(payload['score']):.3f}, confidence={float(payload['confidence']):.3f}, "
                f"lexical={float(payload['lexical_score']):.3f}, semantic={float(payload['semantic_score']):.3f}"
            )
            terms_line = ", ".join(str(term) for term in matched_terms) or "no direct term overlap"

            viewer.configure(state=tk.NORMAL)
            viewer.delete("1.0", tk.END)
            viewer.insert(tk.END, " | ".join(source_bits) + "\n", "metadata")
            viewer.insert(tk.END, score_line + "\n", "metadata")
            viewer.insert(tk.END, f"Matched terms: {terms_line}\n\n", "metadata")
            start = viewer.index(tk.END)
            viewer.insert(tk.END, chunk_text)
            end = viewer.index(tk.END)
            viewer.tag_add("evidence", start, end)
            self._highlight_text_span(viewer, "answer", start, chunk_text, answer.text)
            self._highlight_answer_terms(viewer, start, chunk_text, answer)
            viewer.configure(state=tk.DISABLED)
            counter.configure(text=f"Retrieved section {index + 1} of {len(chunks)}")
            previous_button.configure(state=tk.NORMAL if index > 0 else tk.DISABLED)
            next_button.configure(state=tk.NORMAL if index + 1 < len(chunks) else tk.DISABLED)

        def move(delta: int) -> None:
            current_index.set(max(0, min(len(chunks) - 1, current_index.get() + delta)))
            render()

        previous_button.configure(command=lambda: move(-1))
        next_button.configure(command=lambda: move(1))
        render()

    def _show_error(self, title: str, message: str) -> None:
        self.index_progress.stop()
        self.index_activity.grid_remove()
        self._set_busy(False, "Error")
        self.status_label.configure(background=THEME["status_error"], foreground=THEME["text"])
        messagebox.showerror(title, message, parent=self)

    def _on_close(self) -> None:
        try:
            save_app_settings(
                self.settings_path,
                AppSettings(
                    retrieval_preset=self.retrieval_preset_var.get(),
                    top_k=self.top_k_var.get(),
                    neighbors=self._resolved_neighbor_window(),
                    use_embeddings=self.use_embeddings_var.get(),
                    show_diagnostics=self.show_diagnostics_var.get(),
                    answer_mode=self.answer_mode_var.get(),
                    file_sort=self.document_sort_var.get(),
                    geometry=self.geometry(),
                    sidebar_visible=self.sidebar_visible,
                    theme=self.theme_var.get(),
                ),
            )
        except (OSError, ValueError, tk.TclError):
            LOGGER.exception("Could not save application settings")
        with self.pipeline_lock:
            if self.pipeline is not None:
                self.pipeline.close()
                self.pipeline = None
        self.destroy()


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    app = RAGDesktopApp()
    app.mainloop()
    return 0
