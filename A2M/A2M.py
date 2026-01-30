#Check out my other projects! https://justagwas.com
#You can find the OFFICIAL CODE of this program here - https://github.com/Justagwas/a2m
import os
import re
import sys
import io
import json
import contextlib
import threading
import queue
import webbrowser
import ctypes
import importlib
from pathlib import Path
from urllib.request import urlopen
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

THEME = {
    "bg": "#0f1115",
    "surface": "#171a22",
    "surface_alt": "#202637",
    "surface_alt_hover": "#2a3246",
    "border": "#2c354a",
    "text": "#e8ecf1",
    "muted": "#9aa4b2",
    "accent": "#4ac3ff",
    "accent_hover": "#2f9bd4",
    "success": "#22c55e",
    "disabled_bg": "#242a38",
    "disabled_fg": "#6b7280",
}

def enable_high_dpi():
    if not sys.platform.startswith("win"):
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

try:
    from pathvalidate import sanitize_filename
except ImportError:
    def sanitize_filename(name):
        return re.sub(r'[<>:"/\\|?*]', '', name)

import librosa
import pretty_midi

SEGMENT_RE = re.compile(r"Segment\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)

MODEL_URL = "https://zenodo.org/records/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth?download=1"
MODEL_FILENAME = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
MODEL_LEGACY_FILENAMES = {
    "note_F1=0.9677_pedal_F1=0.9186.pth",
}
MODEL_MIN_BYTES = 160_000_000
MODEL_DIR = Path.home() / "piano_transcription_inference_data"
CONFIG_PATH = Path.home() / ".a2m_config.json"
DEFAULT_CONFIG = {
    "device_preference": "cpu",
    "gpu_batch_size": 4,
}

_PTI_MODULE = None

def get_pti_module():
    global _PTI_MODULE
    if _PTI_MODULE is None:
        _PTI_MODULE = importlib.import_module("piano_transcription_inference")
    return _PTI_MODULE

def get_app_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

def set_app_icon(window):
    icon_path = get_app_dir() / "icon.ico"
    if not icon_path.exists():
        return
    try:
        window.iconbitmap(default=str(icon_path))
    except Exception:
        pass

def download_file(url, dest_path, progress_callback=None):
    dest_path = Path(dest_path)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with urlopen(url) as response, open(tmp_path, "wb") as out_file:
            total_header = response.headers.get("Content-Length")
            total_size = int(total_header) if total_header and total_header.isdigit() else None
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size and progress_callback:
                    percent = int(downloaded * 100 / total_size)
                    progress_callback(min(percent, 100))
        os.replace(tmp_path, dest_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download model: {exc}") from exc

def _find_existing_model(base_dir):
    for name in (MODEL_FILENAME, *MODEL_LEGACY_FILENAMES):
        candidate = base_dir / name
        if candidate.exists() and candidate.stat().st_size >= MODEL_MIN_BYTES:
            return candidate
    return None

def _can_write_dir(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_path = path / ".a2m_write_test"
        with open(test_path, "wb") as handle:
            handle.write(b"")
        test_path.unlink()
        return True
    except Exception:
        return False

def ensure_model_file(progress_callback=None, log_callback=None):
    app_dir = get_app_dir()
    existing_local = _find_existing_model(app_dir)
    if existing_local:
        return existing_local
    existing_cache = _find_existing_model(MODEL_DIR)
    if existing_cache:
        return existing_cache

    if _can_write_dir(app_dir):
        target_dir = app_dir
    else:
        target_dir = MODEL_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        if log_callback:
            log_callback(f"App folder not writable. Using cache:\n{target_dir}")

    model_path = target_dir / MODEL_FILENAME
    if model_path.exists():
        model_path.unlink()
    if log_callback:
        log_callback(f"Downloading model (~165 MB) to:\n{model_path}")
    download_file(MODEL_URL, model_path, progress_callback)
    if not model_path.exists() or model_path.stat().st_size < MODEL_MIN_BYTES:
        raise RuntimeError("Downloaded model file is incomplete or corrupted.")
    return model_path

def get_existing_model_path():
    app_model = _find_existing_model(get_app_dir())
    if app_model:
        return app_model
    cache_model = _find_existing_model(MODEL_DIR)
    if cache_model:
        return cache_model
    return None

class StdoutTee(io.TextIOBase):
    def __init__(self, original, line_handler):
        self.original = original
        self.line_handler = line_handler
        self._buffer = ""

    def write(self, text):
        if self.original:
            self.original.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.line_handler(line)
        return len(text)

    def flush(self):
        if self.original:
            self.original.flush()
        if self._buffer.strip():
            self.line_handler(self._buffer)
        self._buffer = ""

def safe_filename(name, fallback="audio", max_length=180):
    cleaned = sanitize_filename(name or "").strip().strip(".")
    if not cleaned or cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = fallback
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned or fallback

def ensure_unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for i in range(2, 1000):
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"Unable to find a unique filename for {path}.")

def _import_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch runtime is missing from this build. Please reinstall A2M."
        ) from exc
    return torch

_DEVICE_PREFERENCE = "cpu"
_GPU_BATCH_SIZE = 4

def set_device_preference(preference):
    global _DEVICE_PREFERENCE
    pref = (preference or "cpu").lower()
    if pref not in ("cpu", "gpu"):
        pref = "cpu"
    if pref == _DEVICE_PREFERENCE:
        return
    _DEVICE_PREFERENCE = pref
    os.environ["A2M_DEVICE"] = pref
    reset_transcriptor()

def set_gpu_batch_size(value):
    global _GPU_BATCH_SIZE
    try:
        batch = int(value)
    except Exception:
        batch = 1
    batch = max(1, min(64, batch))
    if batch == _GPU_BATCH_SIZE:
        return
    _GPU_BATCH_SIZE = batch
    os.environ["A2M_BATCH_SIZE"] = str(batch)
    reset_transcriptor()

def get_gpu_batch_size():
    return _GPU_BATCH_SIZE

def is_torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False

def is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def get_device(preference=None):
    torch = _import_torch()
    pref = (preference or _DEVICE_PREFERENCE or "cpu").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError(
            "GPU mode selected, but CUDA is not available. Install a CUDA-enabled PyTorch build or switch to CPU."
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config():
    config = dict(DEFAULT_CONFIG)
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                config.update(data)
    except Exception:
        pass
    return config

def save_config(config):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
    except Exception:
        pass

DOWNLOADS_DIR = Path.home() / "Downloads"
OUTPUT_MIDI_DIR = str(DOWNLOADS_DIR / "A2M")

os.makedirs(OUTPUT_MIDI_DIR, exist_ok=True)

_TRANSCRIPTOR = None
_TRANSCRIPTOR_ERROR = None

def reset_transcriptor():
    global _TRANSCRIPTOR, _TRANSCRIPTOR_ERROR
    _TRANSCRIPTOR = None
    _TRANSCRIPTOR_ERROR = None

def get_transcriptor():
    global _TRANSCRIPTOR, _TRANSCRIPTOR_ERROR
    if _TRANSCRIPTOR is None and _TRANSCRIPTOR_ERROR is None:
        try:
            pti = get_pti_module()
            model_path = get_existing_model_path()
            if not model_path:
                raise RuntimeError("Model file is missing. Please download it first.")
            device = get_device()
            batch_size = get_gpu_batch_size() if device.type == "cuda" else 1
            _TRANSCRIPTOR = pti.PianoTranscription(device=device, checkpoint_path=str(model_path), batch_size=batch_size)
        except Exception as exc:
            _TRANSCRIPTOR_ERROR = exc
    if _TRANSCRIPTOR is None:
        raise RuntimeError(f"PianoTranscription model failed to load: {_TRANSCRIPTOR_ERROR}")
    return _TRANSCRIPTOR

def post_process_midi(midi_file_path, min_duration=0.05, min_velocity=20):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    for instrument in midi_data.instruments:

        notes = [note for note in instrument.notes if (note.end - note.start) >= min_duration and note.velocity >= min_velocity]

        notes.sort(key=lambda n: (n.pitch, n.start))
        filtered_notes = []
        last_end_by_pitch = {}
        for note in notes:

            last_end = last_end_by_pitch.get(note.pitch, -float('inf'))
            if note.start < last_end:
                continue
            filtered_notes.append(note)
            last_end_by_pitch[note.pitch] = note.end
        instrument.notes = filtered_notes
    midi_data.write(midi_file_path)

def convert_audio_to_midi(audio_file_path, custom_name=None, min_duration=0.02, min_velocity=20, progress_callback=None):
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    transcriptor = get_transcriptor()
    sample_rate = get_pti_module().sample_rate
    audio, _ = librosa.load(audio_file_path, sr=sample_rate)
    midi_dir = OUTPUT_MIDI_DIR
    os.makedirs(midi_dir, exist_ok=True)
    if custom_name:
        midi_base = safe_filename(custom_name, fallback="midi")
    else:
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        midi_base = safe_filename(base_name, fallback="midi")
    midi_file_name = ensure_unique_path(os.path.join(midi_dir, f"{midi_base}.mid"))
    if progress_callback:
        def handle_line(line):
            match = SEGMENT_RE.search(line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                if total > 0:
                    progress_callback(current, total)
        stdout_tee = StdoutTee(sys.stdout, handle_line)
        with contextlib.redirect_stdout(stdout_tee):
            transcriptor.transcribe(audio, midi_file_name)
        stdout_tee.flush()
    else:
        transcriptor.transcribe(audio, midi_file_name)
    post_process_midi(midi_file_name, min_duration, min_velocity)
    return midi_file_name

class AudioToMidiApp:
    def __init__(self, root):
        self.root = root
        self.theme = THEME
        self.fonts = {
            "title": ("Bahnschrift", 18, "bold"),
            "subtitle": ("Bahnschrift", 11),
            "label": ("Bahnschrift", 10, "bold"),
            "body": ("Bahnschrift", 12),
            "caption": ("Bahnschrift", 9),
            "button": ("Bahnschrift", 11, "bold"),
            "mono": ("Cascadia Mono", 10),
        }
        self.root.title("A2M (Audio To Midi)")
        self.root.configure(bg=self.theme["bg"])
        set_app_icon(self.root)
        self.root.resizable(True, True)
        self.set_initial_geometry()
        self.selected_file = None
        self.file_display_name = ""
        self.file_path_full = ""
        self.controls_enabled = True
        self.ui_queue = queue.Queue()
        self.config = load_config()
        self.model_download_in_progress = False
        set_device_preference(self.config.get("device_preference", "cpu"))
        set_gpu_batch_size(self.config.get("gpu_batch_size", 4))
        self.setup_widgets()
        self.update_model_status_label()
        self.set_controls_state(True)
        self.root.after(100, self.process_ui_queue)
        self.root.after(400, self.check_model_on_start)

    def setup_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        theme = self.theme
        fonts = self.fonts
        outer = tk.Frame(self.root, bg=theme["bg"])
        outer.pack(fill="both", expand=True)

        header = tk.Frame(outer, bg=theme["bg"], padx=24, pady=10)
        header.pack(fill="x", pady=(20, 0))
        title = tk.Label(header, text="A2M (Audio To Midi)", font=fonts["title"], fg=theme["text"], bg=theme["bg"])
        subtitle = tk.Label(
            header,
            text="Transcribe a local audio file into MIDI.",
            font=fonts["subtitle"],
            fg=theme["muted"],
            bg=theme["bg"],
        )
        title.pack(anchor="w")
        subtitle.pack(anchor="w", pady=(4, 0))

        separator = ttk.Separator(outer, orient="horizontal")
        separator.pack(fill="x", padx=24)

        main_frame = tk.Frame(outer, bg=theme["bg"], padx=24, pady=12)
        main_frame.pack(fill="both", expand=True)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        card = tk.Frame(main_frame, bg=theme["surface"], highlightthickness=1, highlightbackground=theme["border"])
        card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)
        card_left = tk.Frame(card, bg=theme["surface"])
        card_left.grid(row=0, column=0, sticky="nsew", padx=16, pady=14)
        self.file_card_left = card_left

        card_title = tk.Label(card_left, text="Audio file", font=fonts["label"], fg=theme["muted"], bg=theme["surface"])
        card_title.pack(anchor="w")
        self.file_label = tk.Label(
            card_left,
            text="No file chosen",
            font=fonts["body"],
            fg=theme["text"],
            bg=theme["surface"],
            anchor="w",
            justify="left",
        )
        self.file_label.pack(anchor="w", fill="x", pady=(4, 0))
        self.file_path_label = tk.Label(
            card_left,
            text="Supported: mp3, wav, flac, ogg, m4a, aac, wma, aiff",
            font=fonts["caption"],
            fg=theme["muted"],
            bg=theme["surface"],
            anchor="w",
            justify="left",
            wraplength=440,
        )
        self.file_path_label.pack(anchor="w", fill="x", pady=(4, 0))
        self.model_status_label = tk.Label(
            card_left,
            text="Transcription model: REQUIRED (download once)",
            font=fonts["caption"],
            fg=theme["muted"],
            bg=theme["surface"],
            anchor="w",
            justify="left",
            wraplength=440,
        )
        self.model_status_label.pack(anchor="w", fill="x", pady=(4, 0))

        self.file_button = tk.Button(
            card,
            text="Choose Audio",
            font=fonts["button"],
            bg=theme["surface_alt"],
            fg=theme["text"],
            activebackground=theme["surface_alt_hover"],
            activeforeground=theme["text"],
            relief="flat",
            bd=0,
            padx=14,
            pady=6,
            command=self.choose_file,
            cursor="hand2",
        )
        self.file_button.grid(row=0, column=1, sticky="e", padx=16, pady=14)

        self.convert_button = tk.Button(
            main_frame,
            text="Convert to MIDI",
            command=self.convert,
            font=fonts["button"],
            bg=theme["accent"],
            fg=theme["bg"],
            activebackground=theme["accent_hover"],
            activeforeground=theme["bg"],
            relief="flat",
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
        )
        self.convert_button.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.progress = tk.Canvas(
            main_frame,
            height=26,
            bg=theme["surface_alt"],
            highlightthickness=1,
            highlightbackground=theme["border"],
            highlightcolor=theme["border"],
            bd=0,
            relief="flat",
        )
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 12), ipady=4)
        self._progress_value = 0.0
        self._progress_fill_color = theme["accent"]
        self.progress_fill_id = self.progress.create_rectangle(0, 0, 0, 0, fill=self._progress_fill_color, width=0)
        self.progress_text_id = self.progress.create_text(
            0,
            0,
            text="0%",
            font=(fonts["caption"][0], fonts["caption"][1], "bold"),
            fill=theme["text"],
        )
        self.progress.bind("<Configure>", lambda _e: self._update_progress_canvas())
        self._last_progress_text = "0%"
        self._update_progress_canvas()

        console_frame = tk.Frame(main_frame, bg=theme["surface"], highlightthickness=1, highlightbackground=theme["border"])
        console_frame.grid(row=3, column=0, sticky="nsew")
        self.console_scrollbar = tk.Scrollbar(console_frame, orient="vertical")
        self.console_scrollbar.pack(side="right", fill="y")
        self.console = tk.Text(
            console_frame,
            height=6,
            bg=theme["surface"],
            fg=theme["text"],
            insertbackground=theme["text"],
            state="disabled",
            wrap="word",
            font=fonts["mono"],
            yscrollcommand=self.console_scrollbar.set,
        )
        self.console.pack(side="left", fill="both", expand=True)
        self.console_scrollbar.config(command=self.console.yview)

        bottom_bar = tk.Frame(outer, bg=theme["bg"], padx=24, pady=8)
        bottom_bar.pack(side="bottom", fill="x")
        bottom_left = tk.Frame(bottom_bar, bg=theme["bg"])
        bottom_left.pack(side="left", anchor="sw")
        settings_btn = tk.Label(bottom_left, text="Settings", font=fonts["caption"], fg=theme["accent"], bg=theme["bg"], cursor="hand2")
        settings_btn.pack(side="left", anchor="sw", padx=(0, 12))
        settings_btn.bind("<Button-1>", lambda e: self.open_settings())
        self.add_hover_effect(settings_btn, bg_normal=theme["bg"], fg_normal=theme["accent"], bg_hover=theme["bg"], fg_hover=theme["text"])
        bottom_right = tk.Frame(bottom_bar, bg=theme["bg"])
        bottom_right.pack(side="right", anchor="se")
        official_link = tk.Label(bottom_right, text="Official page", font=fonts["caption"], fg=theme["accent"], bg=theme["bg"], cursor="hand2")
        official_link.pack(side="right", anchor="se", padx=(0, 8))
        official_link.bind("<Button-1>", lambda e: webbrowser.open("https://justagwas.com/projects/a2m"))
        version_label = tk.Label(bottom_right, text="v1.0.0", font=fonts["caption"], fg=theme["muted"], bg=theme["bg"])
        version_label.pack(side="right", anchor="se", padx=(0, 8))
        self.add_hover_effect(self.file_button, bg_normal=theme["surface_alt"], fg_normal=theme["text"], bg_hover=theme["surface_alt_hover"], fg_hover=theme["accent"])
        self.add_hover_effect(self.convert_button, bg_normal=theme["accent"], fg_normal=theme["bg"], bg_hover=theme["accent_hover"], fg_hover=theme["bg"])
        self.add_hover_effect(official_link, bg_normal=theme["bg"], fg_normal=theme["accent"], bg_hover=theme["bg"], fg_hover=theme["text"])

        def update_wrap(event):
            wrap = max(event.width - 32, 220)
            self.file_path_label.config(wraplength=wrap)
            self.model_status_label.config(wraplength=wrap)
            self.update_file_label(event.width)
            self.update_file_path_label(event.width)
        card_left.bind("<Configure>", update_wrap)
        self.root.bind("<Configure>", lambda e: (self.update_file_label(), self.update_file_path_label()))

    def set_initial_geometry(self):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        min_w = int(max(520, screen_w * 0.3))
        min_h = int(max(660, screen_h * 0.32))
        self.root.minsize(min_w, min_h)
        x = max(0, int((screen_w - min_w) / 2))
        y = max(0, int((screen_h - min_h) / 2))
        self.root.geometry(f"{min_w}x{min_h}+{x}+{y}")

    def add_hover_effect(self, btn, bg_normal, fg_normal, bg_hover, fg_hover):
        def is_enabled():
            try:
                return btn["state"] == tk.NORMAL
            except tk.TclError:
                return True

        def on_enter(e):
            if is_enabled():
                btn.config(bg=bg_hover, fg=fg_hover)
        def on_leave(e):
            if is_enabled():
                btn.config(bg=bg_normal, fg=fg_normal)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def update_file_label(self, available_width=None):
        if not self.file_display_name:
            return
        if available_width is None:
            available_width = self.file_card_left.winfo_width()
        available = available_width - 8
        if available <= 1:
            return
        font = tkfont.Font(font=self.file_label.cget("font"))
        truncated = self.truncate_right(self.file_display_name, available, font)
        self.file_label.config(text=truncated)

    def update_file_path_label(self, available_width=None):
        if not self.file_path_full:
            return
        if available_width is None:
            available_width = self.file_card_left.winfo_width()
        available = available_width - 120
        if available <= 1:
            return
        font = tkfont.Font(font=self.file_path_label.cget("font"))
        truncated = self.truncate_right(self.file_path_full, available, font)
        self.file_path_label.config(text=truncated)

    def update_model_status_label(self):
        if not hasattr(self, "model_status_label"):
            return
        if get_existing_model_path():
            self.model_status_label.config(text="Transcription model: Ready")
        else:
            self.model_status_label.config(text="Transcription model: REQUIRED (download once)")

    @staticmethod
    def truncate_right(text, max_width, font):
        if font.measure(text) <= max_width:
            return text
        ellipsis = "..."
        available = max(0, max_width - font.measure(ellipsis))
        if available <= 0:
            return ellipsis
        trimmed = text
        while trimmed and font.measure(trimmed) > available:
            trimmed = trimmed[:-1]
        return f"{trimmed}{ellipsis}" if trimmed else ellipsis

    def _can_convert(self):
        if not self.selected_file:
            return False
        if not is_torch_available():
            return False
        preference = str(self.config.get("device_preference", "cpu")).lower()
        if preference == "gpu" and not is_cuda_available():
            return False
        return True

    def set_controls_state(self, enabled):
        self.controls_enabled = enabled
        theme = self.theme
        file_state = tk.NORMAL if enabled else tk.DISABLED
        allow_convert = enabled and self._can_convert()
        convert_state = tk.NORMAL if allow_convert else tk.DISABLED
        self.convert_button.config(state=convert_state)
        self.file_button.config(state=file_state)
        self.file_label.config(fg=theme["text"] if enabled else theme["disabled_fg"])
        self.file_path_label.config(fg=theme["muted"] if enabled else theme["disabled_fg"])
        if not enabled:
            self.file_button.unbind("<Enter>")
            self.file_button.unbind("<Leave>")
            self.convert_button.unbind("<Enter>")
            self.convert_button.unbind("<Leave>")
            self.file_button.config(bg=theme["disabled_bg"], fg=theme["disabled_fg"], activebackground=theme["disabled_bg"], activeforeground=theme["disabled_fg"], cursor="")
            self.convert_button.config(bg=theme["disabled_bg"], fg=theme["disabled_fg"], activebackground=theme["disabled_bg"], activeforeground=theme["disabled_fg"], cursor="")
        else:
            self.file_button.config(bg=theme["surface_alt"], fg=theme["text"], activebackground=theme["surface_alt_hover"], activeforeground=theme["text"], cursor="hand2")
            self.add_hover_effect(self.file_button, bg_normal=theme["surface_alt"], fg_normal=theme["text"], bg_hover=theme["surface_alt_hover"], fg_hover=theme["accent"])
            if allow_convert:
                self.convert_button.config(bg=theme["accent"], fg=theme["bg"], activebackground=theme["accent_hover"], activeforeground=theme["bg"], cursor="hand2")
                self.add_hover_effect(self.convert_button, bg_normal=theme["accent"], fg_normal=theme["bg"], bg_hover=theme["accent_hover"], fg_hover=theme["bg"])
            else:
                self.convert_button.unbind("<Enter>")
                self.convert_button.unbind("<Leave>")
                self.convert_button.config(bg=theme["disabled_bg"], fg=theme["disabled_fg"], activebackground=theme["disabled_bg"], activeforeground=theme["disabled_fg"], cursor="")

    def choose_file(self):
        if not self.controls_enabled:
            return
        filetypes = [
            ("Audio Files", "*.mp3;*.wav;*.flac;*.ogg;*.m4a;*.aac;*.wma;*.aiff;*.aif"),
            ("All Files", "*.*"),
        ]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.selected_file = file_path
            self.file_display_name = os.path.basename(file_path)
            self.file_path_full = file_path
            self.file_label.config(text=self.file_display_name)
            self.file_path_label.config(text=file_path)
            self.root.after_idle(self.update_file_label)
            self.root.after_idle(self.update_file_path_label)
        else:
            self.file_display_name = ""
            self.file_path_full = ""
            self.file_label.config(text="No file chosen")
            self.file_path_label.config(text="Supported: mp3, wav, flac, ogg, m4a, aac, wma, aiff")
        self.set_controls_state(self.controls_enabled)

    def open_settings(self):
        if getattr(self, "_settings_window", None) and self._settings_window.winfo_exists():
            self._settings_window.lift()
            self._settings_window.focus_set()
            return
        theme = self.theme
        fonts = self.fonts
        window = tk.Toplevel(self.root)
        self._settings_window = window
        window.title("Settings")
        window.configure(bg=theme["bg"])
        set_app_icon(window)
        window.resizable(False, False)
        window.transient(self.root)
        window.withdraw()
        try:
            self.root.attributes("-disabled", True)
        except tk.TclError:
            pass

        container = tk.Frame(window, bg=theme["bg"], padx=20, pady=16)
        container.pack(fill="both", expand=True)

        title = tk.Label(container, text="Device", font=fonts["label"], fg=theme["text"], bg=theme["bg"])
        title.pack(anchor="w", pady=(0, 6))

        device_var = tk.StringVar(value=str(self.config.get("device_preference", "cpu")).lower())
        batch_var = tk.IntVar(value=int(self.config.get("gpu_batch_size", 4)))
        radio_frame = tk.Frame(container, bg=theme["bg"])
        radio_frame.pack(anchor="w", fill="x")
        cpu_radio = tk.Radiobutton(
            radio_frame,
            text="CPU (default)",
            variable=device_var,
            value="cpu",
            font=fonts["body"],
            fg=theme["text"],
            bg=theme["bg"],
            selectcolor=theme["surface_alt"],
            activebackground=theme["bg"],
            activeforeground=theme["text"],
            highlightthickness=0,
        )
        gpu_radio = tk.Radiobutton(
            radio_frame,
            text="GPU (faster, NVIDIA only)",
            variable=device_var,
            value="gpu",
            font=fonts["body"],
            fg=theme["text"],
            bg=theme["bg"],
            selectcolor=theme["surface_alt"],
            activebackground=theme["bg"],
            activeforeground=theme["text"],
            highlightthickness=0,
        )
        cpu_radio.pack(anchor="w")
        gpu_radio.pack(anchor="w")

        torch_available = is_torch_available()
        cuda_available = is_cuda_available() if torch_available else False

        info_label = tk.Label(
            container,
            text="CPU mode uses the bundled PyTorch runtime. GPU mode needs a CUDA-enabled PyTorch build and NVIDIA drivers.",
            font=fonts["caption"],
            fg=theme["muted"],
            bg=theme["bg"],
            wraplength=420,
            justify="left",
        )
        info_label.pack(anchor="w", pady=(8, 0))

        def update_info(*_args):
            if not torch_available:
                info_label.config(text="PyTorch is missing. CPU/GPU requires it. Select GPU to see setup help.")
            elif device_var.get() == "gpu":
                if not cuda_available:
                    info_label.config(text="GPU mode is unavailable because NVIDIA/CUDA was not detected.")
                else:
                    info_label.config(text="GPU mode is enabled. CUDA-enabled PyTorch is installed.")
            else:
                info_label.config(text="CPU mode uses the bundled PyTorch runtime.")
        device_var.trace_add("write", update_info)
        update_info()

        batch_frame = tk.Frame(container, bg=theme["bg"])
        batch_frame.pack(anchor="w", pady=(12, 0), fill="x")
        batch_label = tk.Label(batch_frame, text="GPU batch size", font=fonts["label"], fg=theme["text"], bg=theme["bg"])
        batch_label.pack(anchor="w")
        batch_hint = tk.Label(
            batch_frame,
            text="Higher values can be faster on GPU but use more VRAM. Default: 4.",
            font=fonts["caption"],
            fg=theme["muted"],
            bg=theme["bg"],
            wraplength=420,
            justify="left",
        )
        batch_hint.pack(anchor="w", pady=(2, 6))
        batch_value_label = tk.Label(
            batch_frame,
            text=f"Value: {batch_var.get()}",
            font=fonts["caption"],
            fg=theme["muted"],
            bg=theme["bg"],
        )
        batch_value_label.pack(anchor="w", pady=(0, 4))
        batch_slider = tk.Scale(
            batch_frame,
            from_=1,
            to=32,
            orient="horizontal",
            variable=batch_var,
            resolution=1,
            font=fonts["body"],
            bg=theme["surface_alt"],
            fg=theme["text"],
            relief="flat",
            bd=0,
            length=360,
            sliderlength=48,
            width=20,
            highlightthickness=0,
            troughcolor=theme["surface_alt"],
            activebackground=theme["surface_alt_hover"],
        )
        batch_slider.pack(anchor="w")
        batch_slider.config(showvalue=0)

        def update_batch_label(*_args):
            batch_value_label.config(text=f"Value: {batch_var.get()}")

        batch_var.trace_add("write", update_batch_label)
        update_batch_label()

        def update_batch_state(*_args):
            if not torch_available or device_var.get() != "gpu" or not cuda_available:
                batch_slider.config(state="disabled")
            else:
                batch_slider.config(state="normal")
        device_var.trace_add("write", update_batch_state)
        update_batch_state()
        button_row = tk.Frame(container, bg=theme["bg"])
        button_row.pack(anchor="e", pady=(16, 0), fill="x")

        def on_back():
            try:
                try:
                    self.root.attributes("-disabled", False)
                except tk.TclError:
                    pass
                self.root.focus_set()
                window.grab_release()
                window.destroy()
            except Exception:
                pass

        back_btn = tk.Label(button_row, text="Back", font=fonts["caption"], fg=theme["accent"], bg=theme["bg"], cursor="hand2")
        back_btn.pack(side="left")
        back_btn.bind("<Button-1>", lambda _e: on_back())
        self.add_hover_effect(back_btn, bg_normal=theme["bg"], fg_normal=theme["accent"], bg_hover=theme["bg"], fg_hover=theme["text"])

        def persist_settings(preference=None, batch=None):
            if preference is not None:
                self.config["device_preference"] = preference
                set_device_preference(preference)
            if batch is not None:
                self.config["gpu_batch_size"] = batch
                set_gpu_batch_size(batch)
            save_config(self.config)

        self._device_change_guard = False

        def handle_device_change(*_args):
            if self._device_change_guard:
                return
            preference = device_var.get()
            if preference == "gpu":
                if not torch_available:
                    messagebox.showwarning(
                        "PyTorch required",
                        "PyTorch is not installed. Please install it first, then select GPU again.",
                    )
                    self.open_pytorch_guide()
                    self._device_change_guard = True
                    device_var.set("cpu")
                    self._device_change_guard = False
                    preference = "cpu"
                elif not cuda_available:
                    self.open_gpu_guide()
                    self._device_change_guard = True
                    device_var.set("cpu")
                    self._device_change_guard = False
                    preference = "cpu"
                else:
                    self.open_gpu_guide()
            persist_settings(preference=preference)
            update_info()
            update_batch_state()

        device_var.trace_add("write", handle_device_change)

        def on_slider_release(_event=None):
            if device_var.get() == "gpu" and torch_available and cuda_available:
                persist_settings(batch=int(batch_var.get()))

        batch_slider.bind("<ButtonRelease-1>", on_slider_release)

        window.protocol("WM_DELETE_WINDOW", on_back)
        window.update_idletasks()
        self.root.update_idletasks()
        win_w = max(window.winfo_reqwidth(), 460)
        win_h = max(window.winfo_reqheight(), 260)
        window.minsize(win_w, win_h)
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        x = root_x + root_w - win_w - 20
        y = root_y + 40
        if x < 20:
            x = root_x + 20
        if y < 20:
            y = root_y + 20
        window.geometry(f"{win_w}x{win_h}+{x}+{y}")
        window.deiconify()
        window.grab_set()
        window.focus_set()

    def open_gpu_guide(self):
        if getattr(self, "_gpu_guide_window", None) and self._gpu_guide_window.winfo_exists():
            self._gpu_guide_window.lift()
            self._gpu_guide_window.focus_set()
            return
        theme = self.theme
        fonts = self.fonts
        parent = self._settings_window if getattr(self, "_settings_window", None) and self._settings_window.winfo_exists() else self.root
        window = tk.Toplevel(self.root)
        self._gpu_guide_window = window
        window.title("GPU Setup Guide")
        window.configure(bg=theme["bg"])
        set_app_icon(window)
        window.resizable(False, False)
        window.transient(parent)

        container = tk.Frame(window, bg=theme["bg"], padx=20, pady=16)
        container.pack(fill="both", expand=True)

        title = tk.Label(container, text="Install GPU support (NVIDIA only)", font=fonts["label"], fg=theme["text"], bg=theme["bg"])
        title.pack(anchor="w", pady=(0, 6))

        steps = tk.Label(
            container,
            text=(
                "This enables faster transcription using your NVIDIA GPU.\n\n"
                "1) Make sure your computer has an NVIDIA graphics card.\n"
                "2) Install the latest NVIDIA graphics driver.\n"
                "3) Install the CUDA runtime (GPU computing support).\n"
                "4) Install the GPU-enabled PyTorch build.\n"
                "5) Restart A2M and select GPU mode in Settings."
            ),
            font=fonts["body"],
            fg=theme["text"],
            bg=theme["bg"],
            justify="left",
            wraplength=520,
        )
        steps.pack(anchor="w", pady=(0, 12))

        links_title = tk.Label(container, text="Helpful links:", font=fonts["caption"], fg=theme["muted"], bg=theme["bg"])
        links_title.pack(anchor="w")

        def link_label(text, url):
            lbl = tk.Label(container, text=text, font=fonts["caption"], fg=theme["accent"], bg=theme["bg"], cursor="hand2")
            lbl.pack(anchor="w", pady=(2, 0))
            lbl.bind("<Button-1>", lambda _e: webbrowser.open(url))
            self.add_hover_effect(lbl, bg_normal=theme["bg"], fg_normal=theme["accent"], bg_hover=theme["bg"], fg_hover=theme["text"])

        link_label("NVIDIA Drivers (official)", "https://www.nvidia.com/Download/index.aspx")
        link_label("CUDA Toolkit (official)", "https://developer.nvidia.com/cuda-downloads")
        link_label("PyTorch Install Guide", "https://pytorch.org/get-started/locally/")

        close_row = tk.Frame(container, bg=theme["bg"])
        close_row.pack(anchor="w", pady=(12, 0), fill="x")
        close_btn = tk.Label(close_row, text="Back", font=fonts["caption"], fg=theme["accent"], bg=theme["bg"], cursor="hand2")
        close_btn.pack(side="left")
        close_btn.bind("<Button-1>", lambda _e: window.destroy())
        self.add_hover_effect(close_btn, bg_normal=theme["bg"], fg_normal=theme["accent"], bg_hover=theme["bg"], fg_hover=theme["text"])

        window.update_idletasks()
        win_w = max(window.winfo_reqwidth(), 560)
        win_h = max(window.winfo_reqheight(), 360)
        window.minsize(win_w, win_h)
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        x = root_x + root_w - win_w - 20
        y = root_y + 40
        if x < 20:
            x = root_x + 20
        if y < 20:
            y = root_y + 20
        window.geometry(f"{win_w}x{win_h}+{x}+{y}")
        window.grab_set()
        window.focus_set()

    def open_pytorch_guide(self):
        if getattr(self, "_pytorch_guide_window", None) and self._pytorch_guide_window.winfo_exists():
            self._pytorch_guide_window.lift()
            self._pytorch_guide_window.focus_set()
            return
        theme = self.theme
        fonts = self.fonts
        parent = self._settings_window if getattr(self, "_settings_window", None) and self._settings_window.winfo_exists() else self.root
        window = tk.Toplevel(self.root)
        self._pytorch_guide_window = window
        window.title("PyTorch Setup Guide")
        window.configure(bg=theme["bg"])
        set_app_icon(window)
        window.resizable(False, False)
        window.transient(parent)

        container = tk.Frame(window, bg=theme["bg"], padx=20, pady=16)
        container.pack(fill="both", expand=True)

        title = tk.Label(container, text="Install PyTorch (required)", font=fonts["label"], fg=theme["text"], bg=theme["bg"])
        title.pack(anchor="w", pady=(0, 6))

        steps = tk.Label(
            container,
            text=(
                "A2M needs PyTorch to run the transcription model.\n\n"
                "1) Install Python (64-bit) from python.org.\n"
                "2) During install, check “Add Python to PATH”.\n"
                "3) Open the PyTorch website and select your system.\n"
                "4) Follow the instructions on the PyTorch page.\n"
                "5) Restart A2M when installation finishes."
            ),
            font=fonts["body"],
            fg=theme["text"],
            bg=theme["bg"],
            justify="left",
            wraplength=520,
        )
        steps.pack(anchor="w", pady=(0, 12))

        links_title = tk.Label(container, text="Helpful links:", font=fonts["caption"], fg=theme["muted"], bg=theme["bg"])
        links_title.pack(anchor="w")

        def link_label(text, url):
            lbl = tk.Label(container, text=text, font=fonts["caption"], fg=theme["accent"], bg=theme["bg"], cursor="hand2")
            lbl.pack(anchor="w", pady=(2, 0))
            lbl.bind("<Button-1>", lambda _e: webbrowser.open(url))
            self.add_hover_effect(lbl, bg_normal=theme["bg"], fg_normal=theme["accent"], bg_hover=theme["bg"], fg_hover=theme["text"])

        link_label("Python Downloads", "https://www.python.org/downloads/")
        link_label("PyTorch Install Guide", "https://pytorch.org/get-started/locally/")

        close_row = tk.Frame(container, bg=theme["bg"])
        close_row.pack(anchor="w", pady=(12, 0), fill="x")
        close_btn = tk.Label(close_row, text="Back", font=fonts["caption"], fg=theme["accent"], bg=theme["bg"], cursor="hand2")
        close_btn.pack(side="left")
        close_btn.bind("<Button-1>", lambda _e: window.destroy())
        self.add_hover_effect(close_btn, bg_normal=theme["bg"], fg_normal=theme["accent"], bg_hover=theme["bg"], fg_hover=theme["text"])

        window.update_idletasks()
        win_w = max(window.winfo_reqwidth(), 560)
        win_h = max(window.winfo_reqheight(), 320)
        window.minsize(win_w, win_h)
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        x = root_x + root_w - win_w - 20
        y = root_y + 40
        if x < 20:
            x = root_x + 20
        if y < 20:
            y = root_y + 20
        window.geometry(f"{win_w}x{win_h}+{x}+{y}")
        window.grab_set()
        window.focus_set()

    def log_console(self, text):
        self.console.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def _set_progress_text(self, text):
        if getattr(self, "_last_progress_text", None) == text:
            return
        self._last_progress_text = text
        self._update_progress_canvas()

    def _set_progress_value(self, value):
        try:
            val = float(value)
        except Exception:
            val = 0.0
        val = max(0.0, min(100.0, val))
        if getattr(self, "_progress_value", None) == val:
            return
        self._progress_value = val
        self._update_progress_canvas()

    def _set_progress_style(self, style_name):
        if not self.progress or not isinstance(self.progress, tk.Canvas):
            return
        theme = self.theme
        if style_name == "complete":
            self._progress_fill_color = theme["success"]
        elif style_name == "disabled":
            self._progress_fill_color = theme["disabled_fg"]
        else:
            self._progress_fill_color = theme["accent"]
        self._update_progress_canvas()

    def _update_progress_canvas(self):
        canvas = self.progress
        if not canvas or not isinstance(canvas, tk.Canvas) or not canvas.winfo_exists():
            return
        if not getattr(self, "progress_fill_id", None) or not getattr(self, "progress_text_id", None):
            return
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        value = getattr(self, "_progress_value", 0.0)
        fill_width = int(width * (value / 100.0))
        canvas.coords(self.progress_fill_id, 0, 0, fill_width, height)
        canvas.itemconfig(self.progress_fill_id, fill=self._progress_fill_color)
        canvas.coords(self.progress_text_id, width / 2, height / 2)
        text = self._last_progress_text
        if text is None:
            text = f"{int(value)}%"
        canvas.itemconfig(self.progress_text_id, text=text)
        text_color = self.theme["bg"] if value >= 50 else self.theme["text"]
        canvas.itemconfig(self.progress_text_id, fill=text_color)

    def update_progress(self, percent, text=None):
        try:
            value = float(percent)
        except Exception:
            value = 0.0
        value = max(0.0, min(100.0, value))
        if text is None:
            text = f"{int(value)}%"
        self._set_progress_style("complete" if value >= 100 else "normal")
        self._set_progress_value(value)
        self._set_progress_text(text)

    def enqueue_ui(self, action, payload=None):
        self.ui_queue.put((action, payload))

    def process_ui_queue(self):
        try:
            while True:
                action, payload = self.ui_queue.get_nowait()
                if action == "log":
                    self.log_console(payload)
                elif action == "progress":
                    if isinstance(payload, tuple) and len(payload) == 2:
                        value, text = payload
                    elif isinstance(payload, dict):
                        value = payload.get("value", 0)
                        text = payload.get("text")
                    else:
                        value, text = payload, None
                    self.update_progress(value, text)
                elif action == "controls":
                    self.set_controls_state(payload)
                elif action == "error":
                    messagebox.showerror("Error", payload)
                elif action == "info":
                    messagebox.showinfo("Info", payload)
                elif action == "model_status":
                    self.update_model_status_label()
        except queue.Empty:
            pass
        self.root.after(100, self.process_ui_queue)

    def prompt_model_download(self, message):
        return messagebox.askyesno("Download model", message)

    def check_model_on_start(self):
        if get_existing_model_path():
            return
        if self.model_download_in_progress:
            return
        if self.prompt_model_download("The transcription model is REQUIRED. Download now (~165 MB)?"):
            self.start_model_download()

    def start_model_download(self):
        if self.model_download_in_progress:
            return
        self.model_download_in_progress = True
        self.set_controls_state(False)
        self.update_progress(0, "Downloading 0%")
        self.log_console("Downloading model...\nPlease wait...")
        threading.Thread(target=self._download_model_thread, daemon=True).start()

    def _download_model_thread(self):
        try:
            def on_model_progress(percent):
                percent = max(0, min(int(percent), 100))
                self.enqueue_ui("progress", (percent, f"Downloading {percent}%"))
            ensure_model_file(progress_callback=on_model_progress, log_callback=lambda msg: self.enqueue_ui("log", msg))
            self.enqueue_ui("log", "Model download complete.")
            self.enqueue_ui("progress", (100, "Downloaded"))
            self.enqueue_ui("model_status", None)
        except Exception as exc:
            self.enqueue_ui("log", f"Model download failed: {exc}")
            self.enqueue_ui("error", f"Model download failed: {exc}")
            self.enqueue_ui("progress", (0, "0%"))
        finally:
            self.model_download_in_progress = False
            self.enqueue_ui("controls", True)

    def convert(self):
        if self.model_download_in_progress:
            messagebox.showinfo("Download in progress", "Model is downloading. Please wait for it to finish.")
            return
        if not get_existing_model_path():
            if self.prompt_model_download("The transcription model is REQUIRED. Download now (~165 MB)?"):
                self.start_model_download()
            return

        self.set_controls_state(False)
        self.update_progress(0, "0%")
        try:
            device_str = f"Device: {'GPU' if get_device().type == 'cuda' else 'CPU'}"
        except Exception as exc:
            self.log_console(f"Error: {exc}")
            self.enqueue_ui("error", str(exc))
            self.set_controls_state(True)
            self.update_progress(0, "0%")
            return
        self.log_console(f"Starting...\n{device_str}\nPlease wait, processing...")
        selected_file = self.selected_file
        threading.Thread(
            target=self.convert_thread,
            args=(selected_file,),
            daemon=True,
        ).start()

    def convert_thread(self, selected_file):
        try:
            self.enqueue_ui("log", "Loading audio file...\nPlease wait...")
            if not selected_file:
                raise ValueError("Please choose an audio file.")
            audio_path = selected_file
            self.enqueue_ui("log", f"Loaded audio file:\n{audio_path}\nConverting to MIDI...\nPlease wait...")
            self.enqueue_ui("progress", (0, "Transcribing 0%"))

            def on_segment(current, total):
                if total <= 0:
                    percent = 0
                else:
                    percent = int((current / total) * 100)
                percent = max(0, min(percent, 100))
                self.enqueue_ui("progress", (percent, f"Transcribing {percent}%"))

            midi_path = convert_audio_to_midi(audio_path, progress_callback=on_segment)
            self.enqueue_ui("progress", (100, "Done"))
            self.enqueue_ui("log", f"Done! Saved at:\n{midi_path}")
        except Exception as e:
            self.enqueue_ui("log", f"Error: {e}")
            self.enqueue_ui("progress", (0, "0%"))
            self.enqueue_ui("error", str(e))
        finally:
            self.enqueue_ui("controls", True)

if __name__ == "__main__":
    enable_high_dpi()
    root = tk.Tk()
    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=THEME["bg"])
    style.configure("TLabel", background=THEME["bg"], foreground=THEME["text"])
    style.configure(
        "Modern.Horizontal.TProgressbar",
        troughcolor=THEME["surface_alt"],
        bordercolor=THEME["border"],
        background=THEME["accent"],
        lightcolor=THEME["accent"],
        darkcolor=THEME["accent"],
        thickness=14,
    )
    style.configure(
        "Modern.Complete.Horizontal.TProgressbar",
        troughcolor=THEME["surface_alt"],
        bordercolor=THEME["border"],
        background=THEME["success"],
        lightcolor=THEME["success"],
        darkcolor=THEME["success"],
        thickness=14,
    )
    style.layout("Modern.Horizontal.TProgressbar",
        [('Horizontal.Progressbar.trough',
          {'children': [('Horizontal.Progressbar.pbar',
                         {'side': 'left', 'sticky': 'ns'})],
           'sticky': 'nswe'})]
    )
    style.layout("Modern.Complete.Horizontal.TProgressbar",
        [('Horizontal.Progressbar.trough',
          {'children': [('Horizontal.Progressbar.pbar',
                         {'side': 'left', 'sticky': 'ns'})],
           'sticky': 'nswe'})]
    )
    app = AudioToMidiApp(root)
    root.mainloop()
