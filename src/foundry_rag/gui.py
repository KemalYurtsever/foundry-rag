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

from .documents import _SUPPORTED_SUFFIXES
from .embeddings import optional_foundry_embedder
from .pipeline import Answer, RAGPipeline

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS = PROJECT_ROOT / "data" / "documents"
DEFAULT_CACHE = PROJECT_ROOT / ".rag_cache"

THEME = {
    "bg": "#0f141a",
    "panel": "#151b23",
    "panel_raised": "#1b2430",
    "field": "#0b1117",
    "border": "#2b3645",
    "text": "#e6edf3",
    "muted": "#9aa7b4",
    "accent": "#6aa6ff",
    "accent_hover": "#8ab8ff",
    "success": "#7ee787",
    "warning": "#d29922",
    "danger": "#ff7b72",
    "selection": "#1f6feb",
    "status_busy": "#24364d",
    "status_ready": "#183322",
    "status_error": "#3d1f24",
}

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
        self.geometry("1100x720")
        self.minsize(820, 560)
        self.documents_dir = DEFAULT_DOCUMENTS
        self.cache_dir = DEFAULT_CACHE
        self.pipeline: RAGPipeline | None = None
        self.pipeline_lock = threading.Lock()
        self.busy = False
        self.sidebar_visible = False
        self.sidebar_width = 280
        self.sidebar_animation: str | None = None
        self.source_link_counter = 0
        self.source_link_payloads: dict[str, tuple[str, Answer]] = {}
        self.last_answer: Answer | None = None
        self.top_k_var = tk.StringVar(value="Auto")
        self.neighbor_window_var = tk.IntVar(value=2)
        self.use_embeddings_var = tk.BooleanVar(value=True)
        self.show_diagnostics_var = tk.BooleanVar(value=False)
        self.answer_mode_var = tk.StringVar(value="Strict")
        self._configure_style()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_document_list()
        self._rebuild_index("Loading documents…")

    def _configure_style(self) -> None:
        self.configure(background=THEME["bg"])
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
        style.configure("TButton", background=THEME["panel_raised"], foreground=THEME["text"], bordercolor=THEME["border"], focusthickness=1, focuscolor=THEME["accent"], padding=(10, 6))
        style.map(
            "TButton",
            background=[("active", THEME["border"]), ("disabled", THEME["panel"])],
            foreground=[("active", THEME["text"]), ("disabled", "#637083")],
        )
        style.configure("TEntry", fieldbackground=THEME["field"], foreground=THEME["text"], bordercolor=THEME["border"], insertcolor=THEME["text"])
        style.configure("TSpinbox", fieldbackground=THEME["field"], foreground=THEME["text"], bordercolor=THEME["border"], arrowsize=13)
        style.configure("TCombobox", fieldbackground=THEME["field"], background=THEME["panel_raised"], foreground=THEME["text"], bordercolor=THEME["border"], arrowcolor=THEME["muted"])
        style.map("TCombobox", fieldbackground=[("readonly", THEME["field"])], foreground=[("readonly", THEME["text"])])
        style.configure("TCheckbutton", background=THEME["panel"], foreground=THEME["text"], focuscolor=THEME["accent"])
        style.map("TCheckbutton", background=[("active", THEME["panel"])], foreground=[("active", THEME["text"])])
        style.configure("Vertical.TScrollbar", background=THEME["panel_raised"], troughcolor=THEME["field"], bordercolor=THEME["bg"], arrowcolor=THEME["muted"])

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=16, style="TFrame")
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        ttk.Label(root, text="Foundry Local RAG", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )
        self.sidebar_toggle = ttk.Button(
            root, text="Show knowledge files", command=self._toggle_knowledge_panel
        )
        self.sidebar_toggle.grid(row=0, column=1, sticky="e", pady=(0, 12))

        self.sidebar = ttk.LabelFrame(root, text="Knowledge files", padding=10)
        self.sidebar.configure(width=self.sidebar_width)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self.sidebar.rowconfigure(0, weight=1)
        self.sidebar.columnconfigure(0, weight=1)
        self.document_list = tk.Listbox(self.sidebar, width=34, activestyle="none", background=THEME["field"], foreground=THEME["text"], selectbackground=THEME["selection"], selectforeground="#ffffff", highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"], relief=tk.FLAT, borderwidth=0)
        self.document_list.grid(row=0, column=0, sticky="nsew")
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
            self.sidebar, text="Import files", command=self._choose_documents
        )
        self.import_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.remove_button = ttk.Button(
            self.sidebar,
            text="Remove selected file",
            command=self._remove_selected_document,
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
        ttk.Label(
            self.sidebar,
            text="Supported: TXT, MD, PDF, DOCX, DOC",
            style="Subtle.TLabel",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        settings = ttk.LabelFrame(self.sidebar, text="Retrieval settings", padding=8)
        settings.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="Top K").grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Entry(settings, textvariable=self.top_k_var, width=8).grid(row=0, column=1, sticky="ew", pady=(0, 4))
        ttk.Label(settings, text="Neighbors").grid(row=1, column=0, sticky="w", pady=(0, 4))
        ttk.Spinbox(settings, from_=0, to=5, textvariable=self.neighbor_window_var, width=6).grid(row=1, column=1, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(settings, text="Use embeddings", variable=self.use_embeddings_var).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(settings, text="Answer mode").grid(row=3, column=0, sticky="w", pady=(4, 4))
        mode_box = ttk.Combobox(
            settings,
            textvariable=self.answer_mode_var,
            values=("Strict", "Flexible"),
            state="readonly",
            width=10,
        )
        mode_box.grid(row=3, column=1, sticky="ew", pady=(4, 4))
        ttk.Checkbutton(settings, text="Show diagnostics", variable=self.show_diagnostics_var).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(settings, text="Apply and rebuild", command=lambda: self._rebuild_index("Applying settings...")).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
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
        self.send_button = ttk.Button(chat, text="Send", command=self._send_question)
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
        files = sorted(
            path.relative_to(self.documents_dir) for path in self.documents_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in _SUPPORTED_SUFFIXES
        )
        for path in files:
            self.document_list.insert(tk.END, str(path))

    def _choose_documents(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Import knowledge files",
            filetypes=[
                ("Supported documents", "*.txt *.md *.pdf *.docx *.doc"),
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

    def _copy_last_answer(self) -> None:
        if self.last_answer is None or not self.last_answer.text.strip():
            messagebox.showinfo("No answer", "There is no answer to copy yet.", parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(self.last_answer.text)
        self.status.set("ANSWER COPIED")

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
        relative_path = Path(self.document_list.get(selection[0]))
        target = (self.documents_dir / relative_path).resolve()
        documents_root = self.documents_dir.resolve()
        if documents_root not in target.parents:
            messagebox.showerror("Invalid file", "The selected path is not safe.", parent=self)
            return
        if not messagebox.askyesno(
            "Remove knowledge file",
            f"Remove '{relative_path}' from the knowledge base?",
            parent=self,
        ):
            return
        try:
            target.unlink()
        except OSError as exc:
            messagebox.showerror("Removal failed", str(exc), parent=self)
            return
        self._refresh_document_list()
        self._rebuild_index(f"Removed {relative_path.name}. Rebuilding index...")
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
                embedder = optional_foundry_embedder("qwen3-embedding-0.6b", cache_dir=self.cache_dir) if self.use_embeddings_var.get() else None
                using_embeddings = embedder is not None
                new_pipeline = RAGPipeline.from_directory(
                    self.documents_dir, embedder=embedder, cache_dir=self.cache_dir, neighbor_window=neighbor_window
                )
                embedder = None
                chunk_count = len(new_pipeline.retriever.chunks)
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
                self.after(
                    0,
                    lambda summary=summary: (
                        self._set_busy(False, "Ready"),
                        self._append_status_message(summary),
                    ),
                )
            except Exception as exc:
                if new_pipeline is not None:
                    new_pipeline.close()
                elif embedder is not None:
                    embedder.close()
                LOGGER.exception("Could not build the RAG index")
                message = str(exc)
                self.after(0, lambda message=message: self._show_error("Indexing failed", message))

        threading.Thread(target=worker, daemon=True, name="rag-indexer").start()

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
                self.after(0, lambda: self._display_answer(answer))
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

    def _answer_for_mode(self, answer: Answer) -> Answer:
        if self.answer_mode_var.get() != "Flexible" or not answer.sources:
            return answer
        compact = re.sub(r"\s+", " ", answer.text).strip()
        if not compact:
            return answer
        if compact.startswith("The provided documents do not contain enough information"):
            return answer
        if compact.startswith(("- ", "* ")) or "\n- " in answer.text:
            flexible_text = f"Based on the cited source:\n{answer.text.strip()}"
        else:
            flexible_text = f"Based on the cited source, {compact}"
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

    def _display_answer(self, answer: Answer) -> None:
        evidence_answer = answer
        display_answer = self._answer_for_mode(answer)
        self.last_answer = display_answer
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
        self.transcript.tag_bind(tag, "<Button-1>", lambda _event, tag=tag: self._open_source_link(tag))
        self.transcript.tag_bind(tag, "<Enter>", lambda _event: self.transcript.configure(cursor="hand2"))
        self.transcript.tag_bind(tag, "<Leave>", lambda _event: self.transcript.configure(cursor=""))

    def _open_source_link(self, tag: str) -> None:
        payload = self.source_link_payloads.get(tag)
        if payload is None:
            return
        source, answer = payload
        source_path = self._source_file_path(source)
        if source_path.exists():
            try:
                os.startfile(source_path)
            except OSError as exc:
                messagebox.showwarning("Open source failed", str(exc), parent=self)
        else:
            messagebox.showwarning("Source not found", f"Could not find {source_path}", parent=self)
        self._show_source_context(source, answer, source_path)

    def _source_file_path(self, source: str) -> Path:
        source_name = re.sub(r"\s+\(page\s+\d+\)$", "", source).strip()
        return self.documents_dir / Path(source_name)

    def _source_context_chunks(self, source: str, answer: Answer) -> list[dict[str, object]]:
        source_name = re.sub(r"\s+\(page\s+\d+\)$", "", source).strip()
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
                source_bits.append(f"page {page}")
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
        self._set_busy(False, "Error")
        self.status_label.configure(background=THEME["status_error"], foreground=THEME["text"])
        messagebox.showerror(title, message, parent=self)

    def _on_close(self) -> None:
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
