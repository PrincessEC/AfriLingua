"""
ui/desktop_app.py
AfriLingua — PyQt6 Desktop Application

Features
--------
* Simultaneous interpretation  — microphone → translated audio + live transcript
* Text translation              — type or paste, instant result with optional TTS
* Document translation         — upload TXT / PDF / DOCX, save translated file
* System-tray integration      — minimise to tray, runs in background as a process

Launch
------
    python ui/desktop_app.py

Package as a standalone executable
-----------------------------------
    pip install pyinstaller
    # Windows
    pyinstaller --onefile --windowed --name AfriLingua ui/desktop_app.py
    # macOS / Linux
    pyinstaller --onefile --name AfriLingua ui/desktop_app.py
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

# ── Qt availability check ─────────────────────────────────────────────────────
try:
    from PyQt6.QtCore import (
        Qt, QThread, QObject, pyqtSignal, pyqtSlot,
    )
    from PyQt6.QtGui import (
        QAction, QCloseEvent, QColor, QFont, QIcon, QPainter, QPixmap, QTextCursor,
    )
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox,
        QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
        QSizePolicy, QSlider, QStatusBar, QSystemTrayIcon,
        QTabWidget, QTextEdit, QVBoxLayout, QWidget, QMenu, QSplitter,
        QProgressBar,
    )
except ImportError:
    print(
        "PyQt6 is not installed.\n"
        "Install it with:  pip install PyQt6\n"
        "Then re-run:      python ui/desktop_app.py"
    )
    sys.exit(1)

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEVICE, TORCH_DTYPE, DEPLOYMENT_MODE,
    GPU_INFO, HARDWARE_PROFILE,
    MODEL_NLLB, MODEL_WHISPER, MODEL_MMS_ASR,
)
from src.detection.lang_detector import LanguageDetector
from src.document.translator import DocumentTranslator
from src.model_loader import load_all, TOTAL_STEPS
from src.pipeline.realtime_pipeline import RealtimePipeline, TranslationResult
from src.translation.engine import TranslationEngine
from src.asr.whisper_asr import WhisperASR
from src.asr.mms_asr import MMSASR
from src.tts.english_tts import EnglishTTS
from src.tts.mms_tts import MMSTTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Shared singletons — pre-loaded at startup before the window appears ─────────
_engine    = TranslationEngine()
_whisper   = WhisperASR()
_mms_asr   = MMSASR()
_detector  = LanguageDetector()
_doc_trans = DocumentTranslator()
_mms_tts   = MMSTTS()
_eng_tts   = EnglishTTS()

_LANGUAGES = ["english", "igbo", "yoruba", "hausa"]


# ─────────────────────────────────────────────────────────────────────────────
# Application stylesheet
# ─────────────────────────────────────────────────────────────────────────────

_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #2d2d4a;
    border-radius: 6px;
    background: #16213e;
}
QTabBar::tab {
    background: #0f3460;
    color: #a0a0c0;
    padding: 8px 20px;
    border-radius: 4px 4px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #e94560;
    color: #ffffff;
    font-weight: bold;
}
QPushButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
}
QPushButton:hover  { background-color: #e94560; }
QPushButton:pressed { background-color: #c73652; }
QPushButton:disabled { background-color: #2d2d4a; color: #555; }
QPushButton#danger { background-color: #7f1d1d; }
QPushButton#danger:hover { background-color: #dc2626; }
QComboBox {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #2d2d4a;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 120px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #e0e0e0;
    selection-background-color: #e94560;
}
QTextEdit {
    background-color: #0d1b2a;
    color: #e0e0e0;
    border: 1px solid #2d2d4a;
    border-radius: 6px;
    padding: 8px;
    font-family: 'Consolas', 'Menlo', monospace;
    font-size: 13px;
    line-height: 1.5;
}
QLabel#section {
    color: #a0c4ff;
    font-weight: bold;
    font-size: 12px;
    letter-spacing: 0.5px;
}
QLabel#status_running { color: #4ade80; font-weight: bold; }
QLabel#status_stopped { color: #f87171; font-weight: bold; }
QProgressBar {
    border: 1px solid #2d2d4a;
    border-radius: 4px;
    background-color: #0d1b2a;
    text-align: center;
    color: #e0e0e0;
}
QProgressBar::chunk { background-color: #e94560; border-radius: 4px; }
QSlider::groove:horizontal {
    height: 6px;
    background: #2d2d4a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e94560;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QScrollBar:vertical {
    background: #0d1b2a;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2d2d4a;
    border-radius: 4px;
    min-height: 30px;
}
QGroupBox {
    border: 1px solid #2d2d4a;
    border-radius: 6px;
    margin-top: 12px;
    padding: 8px;
    font-weight: bold;
    color: #a0c4ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #2d2d4a;
    border-radius: 3px;
    background: #0d1b2a;
}
QCheckBox::indicator:checked {
    background: #e94560;
    border-color: #e94560;
}
QStatusBar { background: #0d1b2a; color: #94a3b8; border-top: 1px solid #2d2d4a; }
QSplitter::handle { background: #2d2d4a; width: 2px; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Tray icon (rendered programmatically — no external image file needed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_tray_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#16a34a"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, 60, 60)
    painter.setPen(QColor("white"))
    painter.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "AL")
    painter.end()
    return QIcon(pixmap)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader — runs on a background QThread with progress signals
# ─────────────────────────────────────────────────────────────────────────────

class _ModelLoaderWorker(QThread):
    """
    Loads all AI models on a background thread.
    Emits step_done(step, total, label) after each model finishes loading.
    Emits download_progress(filename, bytes_done, bytes_total, speed_bps) per chunk.
    Emits finished_ok() when everything is ready.
    Emits error(message) on failure.
    """
    step_done         = pyqtSignal(int, int, str)         # step, total, label
    download_progress = pyqtSignal(str, int, int, float)  # filename, done, total, speed_bps
    finished_ok       = pyqtSignal()
    error             = pyqtSignal(str)

    def run(self) -> None:
        try:
            load_all(
                on_progress=lambda step, total, label: self.step_done.emit(step, total, label),
                on_download_progress=lambda f, d, t, s: self.download_progress.emit(f, d, t, s),
                engine=_engine,
                whisper=_whisper,
                mms_asr=_mms_asr,
                eng_tts=_eng_tts,
                mms_tts=_mms_tts,
            )
            self.finished_ok.emit()
        except Exception as exc:
            logger.error("Model loading failed: %s", exc, exc_info=True)
            self.error.emit(str(exc))


class _LoadingDialog(QDialog):
    """
    Modal splash dialog shown while models are downloading / loading.
    Displays a progress bar and the current loading step label.
    """

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowTitle("AfriLingua — Starting")
        self.setWindowIcon(_make_tray_icon())
        self.setFixedSize(500, 300)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(24, 20, 24, 20)

        # Title
        title = QLabel("🌍  AfriLingua")
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#e0e0e0;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # Sub-title
        sub = QLabel("Loading AI models — please wait")
        sub.setStyleSheet("color:#94a3b8; font-size:12px;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(sub)

        # Model progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, TOTAL_STEPS)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%v / %m models loaded")
        root.addWidget(self._bar)

        # Step label
        self._label = QLabel("Initialising …")
        self._label.setStyleSheet("color:#94a3b8; font-size:12px;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        root.addWidget(self._label)

        # ── Download sub-progress (hidden until a file starts downloading) ──────
        self._dl_info = QLabel("")
        self._dl_info.setStyleSheet("color:#60a5fa; font-size:11px;")
        self._dl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dl_info.setWordWrap(True)
        self._dl_info.setVisible(False)
        root.addWidget(self._dl_info)

        self._dl_bar = QProgressBar()
        self._dl_bar.setRange(0, 100)
        self._dl_bar.setValue(0)
        self._dl_bar.setTextVisible(False)
        self._dl_bar.setFixedHeight(6)
        self._dl_bar.setStyleSheet(
            "QProgressBar { border:none; background:#0d1b2a; border-radius:3px; }"
            "QProgressBar::chunk { background:#3b82f6; border-radius:3px; }"
        )
        self._dl_bar.setVisible(False)
        root.addWidget(self._dl_bar)

        # Note
        note = QLabel(
            "First-time setup downloads ~3 GB of model weights. "
            "This only happens once — subsequent starts are fast."
        )
        note.setStyleSheet("color:#64748b; font-size:11px;")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setWordWrap(True)
        root.addWidget(note)

    @pyqtSlot(int, int, str)
    def on_step(self, step: int, total: int, label: str) -> None:
        self._bar.setValue(step)
        self._label.setText(label)
        # Reset download sub-bar at the start of each new model step
        self._dl_bar.setValue(0)
        self._dl_bar.setVisible(False)
        self._dl_info.setVisible(False)

    @pyqtSlot(str, int, int, float)
    def on_download_progress(self, filename: str, done: int, total: int, speed: float) -> None:
        if total <= 0:
            return
        pct      = min(100, int(done / total * 100))
        done_mb  = done  / 1_048_576
        total_mb = total / 1_048_576
        speed_mb = speed / 1_048_576
        name     = filename.split("/")[-1] or filename or "file"
        self._dl_bar.setValue(pct)
        self._dl_info.setText(
            f"⬇  {name}  ·  {done_mb:.1f} / {total_mb:.1f} MB  ·  {speed_mb:.1f} MB/s"
        )
        self._dl_bar.setVisible(True)
        self._dl_info.setVisible(True)


# ─────────────────────────────────────────────────────────────────────────────
# Signal bridge — safely passes pipeline results from worker thread → Qt UI
# ─────────────────────────────────────────────────────────────────────────────

class _ResultBridge(QObject):
    result_ready = pyqtSignal(str, str, str, str)   # original, translated, src, tgt


# ─────────────────────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────────────────────

class _TranslateWorker(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, text: str, src: str, tgt: str, play_tts: bool) -> None:
        super().__init__()
        self._text = text
        self._src  = src
        self._tgt  = tgt
        self._play = play_tts

    def run(self) -> None:
        try:
            translated = _engine.translate(self._text, self._src, self._tgt)
            if self._play and translated.strip():
                import sounddevice as sd
                if self._tgt == "english":
                    waveform, sr = _eng_tts.synthesize(translated)
                else:
                    waveform, sr = _mms_tts.synthesize(translated, lang=self._tgt)
                sd.play(waveform, sr)
            self.finished.emit(translated)
        except Exception as exc:
            self.error.emit(str(exc))


class _DocumentWorker(QThread):
    finished = pyqtSignal(str, str)   # preview, output_path
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)        # 0–100

    def __init__(self, file_path: str, src: Optional[str], tgt: str) -> None:
        super().__init__()
        self._file = file_path
        self._src  = src
        self._tgt  = tgt

    def run(self) -> None:
        try:
            self.progress.emit(20)
            out_path = _doc_trans.translate_file(self._file, self._tgt, src_lang=self._src)
            self.progress.emit(90)
            preview = out_path.read_text(encoding="utf-8", errors="replace")[:4000]
            if len(preview) == 4000:
                preview += "\n\n… (truncated — use Save button for full file)"
            self.progress.emit(100)
            self.finished.emit(preview, str(out_path))
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Live Simultaneous Interpretation
# ─────────────────────────────────────────────────────────────────────────────

class LiveTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Optional[RealtimePipeline] = None
        self._bridge   = _ResultBridge()
        self._bridge.result_ready.connect(self._on_result)
        self._entry_count = 0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(14, 14, 14, 14)

        desc = QLabel(
            "Speak continuously — translation begins every few seconds while you are "
            "still talking and plays back in the target language in near real-time, "
            "like a live conference interpreter."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
        root.addWidget(desc)

        # Language selectors
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Your Language"))
        self._src_combo = QComboBox()
        self._src_combo.addItems([l.capitalize() for l in _LANGUAGES])
        lang_row.addWidget(self._src_combo)
        lang_row.addSpacing(16)
        arrow = QLabel("→")
        arrow.setStyleSheet("font-size: 20px; color: #e94560; font-weight: bold;")
        lang_row.addWidget(arrow)
        lang_row.addSpacing(16)
        lang_row.addWidget(QLabel("Interpret Into"))
        self._tgt_combo = QComboBox()
        self._tgt_combo.addItems([l.capitalize() for l in _LANGUAGES])
        self._tgt_combo.setCurrentText("Igbo")
        lang_row.addWidget(self._tgt_combo)
        lang_row.addStretch()
        root.addLayout(lang_row)

        # Controls
        ctrl_row = QHBoxLayout()
        self._status_label = QLabel("● Stopped")
        self._status_label.setObjectName("status_stopped")
        ctrl_row.addWidget(self._status_label)
        ctrl_row.addStretch()

        self._start_btn = QPushButton("▶  Start Interpretation")
        self._start_btn.clicked.connect(self.start_interpretation)
        ctrl_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setObjectName("danger")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_interpretation)
        ctrl_row.addWidget(self._stop_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear)
        ctrl_row.addWidget(self._clear_btn)
        root.addLayout(ctrl_row)

        # Transcript
        section = QLabel("Live Interpretation Transcript")
        section.setObjectName("section")
        root.addWidget(section)

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText("🎙️  Start interpretation and speak …")
        root.addWidget(self._transcript)

    # ── public methods (also called from tray menu) ───────────────────────────

    def start_interpretation(self) -> None:
        src = self._src_combo.currentText().lower()
        tgt = self._tgt_combo.currentText().lower()
        if src == tgt:
            QMessageBox.warning(self, "AfriLingua", "Source and target languages must be different.")
            return

        bridge = self._bridge

        def _on_result(r: TranslationResult) -> None:
            bridge.result_ready.emit(r.original_text, r.translated_text, r.src_lang, r.tgt_lang)

        self._pipeline = RealtimePipeline(
            src_lang=src, tgt_lang=tgt,
            on_result=_on_result,
            audio_output=True,
            whisper=_whisper,
            mms_asr=_mms_asr,
            engine=_engine,
            mms_tts=_mms_tts,
            eng_tts=_eng_tts,
        )
        self._pipeline.start_microphone()
        self._set_running(True)

    def stop_interpretation(self) -> None:
        if self._pipeline:
            self._pipeline.stop_microphone()
            self._pipeline = None
        self._set_running(False)

    def cleanup(self) -> None:
        if self._pipeline:
            self._pipeline.stop_microphone()

    # ── private helpers ───────────────────────────────────────────────────────

    def _set_running(self, running: bool) -> None:
        self._status_label.setText("● Interpreting …" if running else "● Stopped")
        self._status_label.setStyleSheet(
            "color: #4ade80; font-weight: bold;" if running else "color: #f87171; font-weight: bold;"
        )
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._src_combo.setEnabled(not running)
        self._tgt_combo.setEnabled(not running)

    def _clear(self) -> None:
        self._transcript.clear()
        self._entry_count = 0

    @pyqtSlot(str, str, str, str)
    def _on_result(self, original: str, translated: str, src: str, tgt: str) -> None:
        self._entry_count += 1
        html = (
            f'<div style="margin-bottom:10px; padding:10px; '
            f'background:#0f3460; border-radius:6px; border-left:3px solid #e94560;">'
            f'<span style="color:#94a3b8; font-size:11px;">'
            f'#{self._entry_count} &nbsp;·&nbsp; {src.capitalize()} → {tgt.capitalize()}</span><br>'
            f'<span style="color:#e0e0e0;">{original}</span><br>'
            f'<span style="color:#4ade80; font-weight:bold;">↳ {translated}</span>'
            f'</div>'
        )
        self._transcript.append(html)
        self._transcript.moveCursor(QTextCursor.MoveOperation.End)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Text Translation
# ─────────────────────────────────────────────────────────────────────────────

class TextTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[_TranslateWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(14, 14, 14, 14)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("From"))
        self._src_combo = QComboBox()
        self._src_combo.addItem("Auto-detect")
        self._src_combo.addItems([l.capitalize() for l in _LANGUAGES])
        lang_row.addWidget(self._src_combo)
        lang_row.addSpacing(12)
        lang_row.addWidget(QLabel("To"))
        self._tgt_combo = QComboBox()
        self._tgt_combo.addItems([l.capitalize() for l in _LANGUAGES])
        self._tgt_combo.setCurrentText("Igbo")
        lang_row.addWidget(self._tgt_combo)
        lang_row.addStretch()
        self._tts_check = QCheckBox("Play audio")
        self._tts_check.setChecked(True)
        lang_row.addWidget(self._tts_check)
        root.addLayout(lang_row)

        # Side-by-side input / output
        splitter = QSplitter(Qt.Orientation.Horizontal)

        input_group = QGroupBox("Input Text")
        input_layout = QVBoxLayout(input_group)
        self._input_edit = QTextEdit()
        self._input_edit.setPlaceholderText("Type or paste text here …")
        input_layout.addWidget(self._input_edit)
        splitter.addWidget(input_group)

        output_group = QGroupBox("Translation")
        output_layout = QVBoxLayout(output_group)
        self._output_edit = QTextEdit()
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("Translation will appear here …")
        output_layout.addWidget(self._output_edit)
        splitter.addWidget(output_group)

        splitter.setSizes([500, 500])
        root.addWidget(splitter)

        ctrl_row = QHBoxLayout()
        self._translate_btn = QPushButton("Translate")
        self._translate_btn.clicked.connect(self._translate)
        ctrl_row.addWidget(self._translate_btn)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #94a3b8;")
        ctrl_row.addWidget(self._status_lbl)
        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

    def _translate(self) -> None:
        text = self._input_edit.toPlainText().strip()
        if not text:
            return
        src_raw = self._src_combo.currentText()
        src = _detector.detect_or_default(text, "english") if src_raw == "Auto-detect" else src_raw.lower()
        tgt = self._tgt_combo.currentText().lower()

        self._translate_btn.setEnabled(False)
        self._status_lbl.setText("Translating …")

        self._worker = _TranslateWorker(text, src, tgt, self._tts_check.isChecked())
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @pyqtSlot(str)
    def _on_done(self, translated: str) -> None:
        self._output_edit.setPlainText(translated)
        self._status_lbl.setText("Done.")
        self._translate_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._status_lbl.setText(f"Error: {msg}")
        self._translate_btn.setEnabled(True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Document Translation
# ─────────────────────────────────────────────────────────────────────────────

class DocumentTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[_DocumentWorker] = None
        self._file_path: Optional[str] = None
        self._output_path: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(14, 14, 14, 14)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("From"))
        self._src_combo = QComboBox()
        self._src_combo.addItem("Auto-detect")
        self._src_combo.addItems([l.capitalize() for l in _LANGUAGES])
        lang_row.addWidget(self._src_combo)
        lang_row.addSpacing(12)
        lang_row.addWidget(QLabel("To"))
        self._tgt_combo = QComboBox()
        self._tgt_combo.addItems([l.capitalize() for l in _LANGUAGES])
        self._tgt_combo.setCurrentText("Igbo")
        lang_row.addWidget(self._tgt_combo)
        lang_row.addStretch()
        root.addLayout(lang_row)

        file_row = QHBoxLayout()
        self._file_label = QLabel("No file selected")
        self._file_label.setStyleSheet("color: #94a3b8;")
        file_row.addWidget(self._file_label, stretch=1)
        browse_btn = QPushButton("Browse …")
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(browse_btn)
        root.addLayout(file_row)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        ctrl_row = QHBoxLayout()
        self._translate_btn = QPushButton("Translate Document")
        self._translate_btn.setEnabled(False)
        self._translate_btn.clicked.connect(self._translate)
        ctrl_row.addWidget(self._translate_btn)
        self._save_btn = QPushButton("Save Translated File …")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        ctrl_row.addWidget(self._save_btn)
        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

        section = QLabel("Translation Preview")
        section.setObjectName("section")
        root.addWidget(section)

        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setPlaceholderText("Upload a .txt, .pdf, or .docx file to begin …")
        root.addWidget(self._preview)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Document", "",
            "Documents (*.txt *.pdf *.docx);;All Files (*)"
        )
        if path:
            self._file_path = path
            self._file_label.setText(Path(path).name)
            self._file_label.setStyleSheet("color: #e0e0e0;")
            self._translate_btn.setEnabled(True)

    def _translate(self) -> None:
        if not self._file_path:
            return
        src_raw = self._src_combo.currentText()
        src = None if src_raw == "Auto-detect" else src_raw.lower()
        tgt = self._tgt_combo.currentText().lower()

        self._translate_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)

        self._worker = _DocumentWorker(self._file_path, src, tgt)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.start()

    @pyqtSlot(str, str)
    def _on_done(self, preview: str, out_path: str) -> None:
        self._preview.setPlainText(preview)
        self._output_path = out_path
        self._progress.setVisible(False)
        self._translate_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._preview.setPlainText(f"Error: {msg}")
        self._progress.setVisible(False)
        self._translate_btn.setEnabled(True)

    def _save(self) -> None:
        if not self._output_path:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Translated File", self._output_path, "All Files (*)"
        )
        if dest:
            shutil.copy2(self._output_path, dest)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Settings
# ─────────────────────────────────────────────────────────────────────────────

class SettingsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(14, 14, 14, 14)

        # ── Hardware & Device ───────────────────────────────────────────
        hw_group = QGroupBox("Hardware & Device")
        hw_layout = QVBoxLayout(hw_group)
        hw_layout.setSpacing(6)

        if GPU_INFO["available"] and DEVICE == "cuda":
            gpu_text = (
                f"🖥️  GPU: {GPU_INFO['name']}  ·  "
                f"{GPU_INFO['vram_gb']:.1f} GB VRAM  ·  "
                f"Compute {GPU_INFO['compute']}"
            )
            gpu_lbl = QLabel(gpu_text)
            gpu_lbl.setStyleSheet("color: #4ade80; font-weight: bold;")
        else:
            gpu_lbl = QLabel("💻  GPU: Not available — running on CPU")
            gpu_lbl.setStyleSheet("color: #f97316; font-weight: bold;")
        hw_layout.addWidget(gpu_lbl)

        hw_layout.addWidget(QLabel(f"Active device :  {DEVICE.upper()}"))
        hw_layout.addWidget(QLabel(f"Torch dtype   :  {TORCH_DTYPE}"))
        hw_layout.addWidget(QLabel(f"Deployment    :  {DEPLOYMENT_MODE.capitalize()}"))
        root.addWidget(hw_group)

        # ── Hardware profile + active models ────────────────────────────
        profile_group = QGroupBox("Hardware Profile & Active Models")
        profile_layout = QVBoxLayout(profile_group)
        profile_layout.setSpacing(6)

        _profile_desc = {
            "high": "High quality  —  CUDA GPU with ≥ 8 GB VRAM  ·  largest distilled models",
            "mid":  "Balanced     —  CUDA GPU with 4–7 GB VRAM  ·  standard distilled models",
            "low":  "Lite / fast  —  CPU only or < 4 GB VRAM   ·  smallest footprint models",
        }
        _profile_colors = {"high": "#4ade80", "mid": "#fbbf24", "low": "#fb923c"}

        profile_lbl = QLabel(
            f"⚡  Profile: {HARDWARE_PROFILE.upper()}  —  "
            f"{_profile_desc.get(HARDWARE_PROFILE, '')}"
        )
        profile_lbl.setStyleSheet(
            f"color: {_profile_colors.get(HARDWARE_PROFILE, '#e0e0e0')}; font-weight: bold;"
        )
        profile_lbl.setWordWrap(True)
        profile_layout.addWidget(profile_lbl)

        profile_layout.addWidget(QLabel(f"🔤  Translation : {MODEL_NLLB}"))
        profile_layout.addWidget(QLabel(f"🎤  English ASR : {MODEL_WHISPER}"))
        profile_layout.addWidget(QLabel(f"🔊  African ASR : {MODEL_MMS_ASR}"))

        note = QLabel(
            "Models are chosen automatically based on detected hardware.\n"
            "Override any model in your .env file (MODEL_NLLB, MODEL_WHISPER, MODEL_MMS_ASR)."
        )
        note.setStyleSheet("color: #64748b; font-size: 11px;")
        note.setWordWrap(True)
        profile_layout.addWidget(note)
        root.addWidget(profile_group)

        # ── VAD settings ────────────────────────────────────────────────
        vad_group = QGroupBox("Voice Activity Detection (takes effect on next session)")
        vad_layout = QVBoxLayout(vad_group)

        vad_row = QHBoxLayout()
        vad_row.addWidget(QLabel("Max chunk interval:"))
        self._chunk_slider = QSlider(Qt.Orientation.Horizontal)
        self._chunk_slider.setRange(2, 8)
        self._chunk_slider.setValue(4)
        self._chunk_slider.setTickInterval(1)
        self._chunk_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        vad_row.addWidget(self._chunk_slider)
        self._chunk_val = QLabel("4 s")
        vad_row.addWidget(self._chunk_val)
        self._chunk_slider.valueChanged.connect(lambda v: self._chunk_val.setText(f"{v} s"))
        vad_layout.addLayout(vad_row)

        root.addWidget(vad_group)

        # ── About ───────────────────────────────────────────────────────────
        about_group = QGroupBox("About AfriLingua")
        about_layout = QVBoxLayout(about_group)
        about_layout.addWidget(QLabel(
            "AfriLingua — Real-time AI Translation for African Indigenous Languages\n"
            "English ↔ Igbo · Yoruba · Hausa\n\n"
            "Powered by:\n"
            "  • Meta NLLB-200 (translation)\n"
            "  • Meta MMS-300M (African language speech recognition)\n"
            "  • OpenAI Whisper (English speech recognition)\n"
            "  • Meta MMS-VITS (text-to-speech)\n"
            "  • Silero VAD (voice activity detection)\n\n"
            "Open-source · MIT License\n"
            "github.com/your-username/afrilingua"
        ))
        root.addWidget(about_group)

        root.addStretch()


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AfriLingua")
        self.setWindowIcon(_make_tray_icon())
        self.setMinimumSize(860, 620)
        self.resize(1040, 740)

        self._setup_tabs()
        self._setup_status_bar()
        self._setup_tray()

    def _setup_tabs(self) -> None:
        self._live_tab     = LiveTab()
        self._text_tab     = TextTab()
        self._doc_tab      = DocumentTab()
        self._settings_tab = SettingsTab()

        tabs = QTabWidget()
        tabs.addTab(self._live_tab,     "🎙️  Live Interpretation")
        tabs.addTab(self._text_tab,     "✍️  Text Translation")
        tabs.addTab(self._doc_tab,      "📄  Document Translation")
        tabs.addTab(self._settings_tab, "⚙️  Settings")
        self.setCentralWidget(tabs)

    def _setup_status_bar(self) -> None:
        bar = QStatusBar()
        _profile_colors = {"high": "#4ade80", "mid": "#fbbf24", "low": "#fb923c"}
        _pc = _profile_colors.get(HARDWARE_PROFILE, "#e0e0e0")
        profile_lbl = QLabel(
            f"Profile: <b style='color:{_pc}'>{HARDWARE_PROFILE.upper()}</b>"
        )
        if GPU_INFO["available"] and DEVICE == "cuda":
            hw_text = f"{GPU_INFO['name']}  ·  {GPU_INFO['vram_gb']:.1f} GB VRAM"
        else:
            hw_text = "CPU only"
        bar.showMessage(f"AfriLingua ready  ·  {hw_text}")
        bar.addPermanentWidget(QLabel(f"{HARDWARE_PROFILE.upper()} profile  ·  {DEVICE.upper()}  ·  {TORCH_DTYPE}"))
        self.setStatusBar(bar)

    def _setup_tray(self) -> None:
        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(_make_tray_icon())
        self._tray.setToolTip("AfriLingua — Real-time African Language Interpreter")

        menu = QMenu()

        show_action = QAction("Show Window", menu)
        show_action.triggered.connect(self._show_window)
        menu.addAction(show_action)

        menu.addSeparator()

        start_action = QAction("▶  Start Interpretation", menu)
        start_action.triggered.connect(self._live_tab.start_interpretation)
        menu.addAction(start_action)

        stop_action = QAction("■  Stop Interpretation", menu)
        stop_action.triggered.connect(self._live_tab.stop_interpretation)
        menu.addAction(stop_action)

        menu.addSeparator()

        quit_action = QAction("Quit AfriLingua", menu)
        quit_action.triggered.connect(self._quit_app)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._tray_activated)
        self._tray.show()

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._show_window()

    def _show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, a0: QCloseEvent) -> None:
        """Intercept the X button — hide to tray instead of quitting."""
        a0.ignore()
        self.hide()
        self._tray.showMessage(
            "AfriLingua",
            "Still running in the background.\nRight-click the tray icon to restore or quit.",
            QSystemTrayIcon.MessageIcon.Information,
            3000,
        )

    def _quit_app(self) -> None:
        self._live_tab.cleanup()
        self._tray.hide()
        QApplication.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("AfriLingua")
    app.setOrganizationName("AfriLingua")
    app.setQuitOnLastWindowClosed(False)   # keep alive in tray when window is closed
    app.setStyleSheet(_STYLE)

    # ── Show loading dialog and pre-load all models before the main window ────
    loading = _LoadingDialog()
    loader  = _ModelLoaderWorker()
    loader.step_done.connect(loading.on_step)
    loader.download_progress.connect(loading.on_download_progress)

    def _on_loaded() -> None:
        loading.accept()          # close dialog → main() continues below

    def _on_error(msg: str) -> None:
        from PyQt6.QtWidgets import QMessageBox
        loading.reject()
        if "No internet connection" in msg:
            QMessageBox.critical(
                None,
                "AfriLingua — Internet Connection Required",
                "⚠  No internet connection detected.\n\n"
                "AfriLingua needs to download AI model weights on first run (~3 GB).\n"
                "Please connect to the internet and restart the app.\n\n"
                "Once downloaded, the app works fully offline.",
            )
        else:
            QMessageBox.critical(
                None,
                "AfriLingua — Startup Error",
                f"Failed to load models:\n\n{msg}\n\n"
                "Check your internet connection and try again.",
            )
        sys.exit(1)

    loader.finished_ok.connect(_on_loaded)
    loader.error.connect(_on_error)
    loader.start()

    loading.exec()   # blocks until _on_loaded() calls accept() or error exits

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
