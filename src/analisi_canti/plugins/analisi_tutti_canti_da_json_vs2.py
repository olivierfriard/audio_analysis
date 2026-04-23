import json
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile
from scipy.signal import find_peaks

add_noise = False
WINDOW_SIZE = 70
OVERLAP = 30
MIN_AMPLITUDE = 0.1
MIN_DISTANCE = 0.003
MAX_DISTANCE = 0.3
PROMINENCE = 0.08
FFT_LENGTH = 1024
FFT_OVERLAP = 512
SIGNAL_TO_NOISE_RATIO = 0.1


def iter_song_jobs(parameters: dict):
    for wav_key, wav_block in parameters.items():
        songs = wav_block.get("songs", {})
        if not isinstance(songs, dict):
            continue
        for song_id, sp in songs.items():
            if isinstance(sp, dict):
                yield wav_key, str(song_id), sp


def wav_path_same_folder(json_path: Path, sp: dict, wav_key: str) -> Path:
    name = sp.get("file") or wav_key
    return (json_path.parent / Path(name).name).resolve()


def wav_path_different_folder(json_path: Path, sp: dict, wav_key: str) -> Path:
    name = sp.get("file") or wav_key
    wav_name = Path(name).name
    stem = Path(wav_name).stem
    folder_name = stem.split("_sample_")[0]
    wav_path = json_path.parent / folder_name / wav_name
    return wav_path.resolve()


def find_wav_next_to_json(json_path: Path, wav_key: str, block: dict) -> Path | None:
    names = [wav_key, block.get("file")]
    for name in names:
        if not name:
            continue
        wav_name = Path(str(name)).name
        candidates = [json_path.parent / wav_name]
        if Path(wav_name).suffix.lower() != ".wav":
            candidates.append(json_path.parent / f"{Path(wav_name).stem}.wav")
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def add_noise_padding(data, sr, duration=0.1, noise_db=-40):
    pad_samples = int(duration * sr)

    if data.ndim == 1:
        signal_rms = np.sqrt(np.mean(data.astype(float) ** 2))
    else:
        signal_rms = np.sqrt(np.mean(data.astype(float) ** 2))

    if signal_rms == 0:
        signal_rms = 1e-12

    noise_rms = signal_rms * 10 ** (noise_db / 20)

    def generate_noise(shape):
        noise = np.random.randn(*shape)
        noise = noise / np.sqrt(np.mean(noise**2))
        noise = noise * noise_rms
        return noise

    if data.ndim == 1:
        noise_pre = generate_noise((pad_samples,))
        noise_post = generate_noise((pad_samples,))
    else:
        noise_pre = generate_noise((pad_samples, data.shape[1]))
        noise_post = generate_noise((pad_samples, data.shape[1]))

    original_dtype = data.dtype
    if np.issubdtype(original_dtype, np.integer):
        data = data.astype(float)

    new_data = np.concatenate([noise_pre, data, noise_post])

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        new_data = np.clip(new_data, info.min, info.max)
        new_data = new_data.astype(original_dtype)

    return new_data


def find_project_json_for_wav(wav_path: Path) -> Path | None:

    wav_path = Path(wav_path).expanduser().resolve()

    # fallback: cerca un json nella cartella corrente
    json_files = sorted(wav_path.parent.glob('*.json'))
    if len(json_files) == 1:
        return json_files[0].resolve()

    return None


def find_song_block_from_wav(parameters: dict, wav_name: str):
    """
    Cerca nel JSON il blocco song corrispondente al nome file WAV selezionato.
    Ritorna (chunk_name, song_dict) oppure (None, None).
    """
    chunks = parameters.get('chunks', {})
    for chunk_name, chunk_data in chunks.items():
        songs = chunk_data.get('songs', {})
        if wav_name in songs:
            return chunk_name, songs[wav_name]
    return None, None


class SpectrumWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(None)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Power Spectrum")
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.resize(900, 450)

    def update_spectrum(
        self, freqs, avg_power_db, peak_freqs, peak_power_db, max_freq=None
    ):
        self.ax.cla()
        self.ax.plot(freqs, avg_power_db, color="blue")
        if len(peak_freqs) > 0:
            self.ax.plot(peak_freqs, peak_power_db, "or")
        self.ax.set_title("Power Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Log Power (dB)")
        if max_freq is not None:
            self.ax.set_xlim(0, max_freq)
        self.canvas.draw_idle()


class Main(QWidget):
    def __init__(self, wav_file_list: list):
        super().__init__()

        self.init_values()
        self.rows = []
        self.df_results = None
        self.spectrum_window = None
        self.span_region = None
        self.selected_times = [0.0, 0.0]
        self.selected_peak_times = []
        self.cid_click = None

        self.wav_file_list = [str(Path(w).expanduser().resolve()) for w in wav_file_list]

        print(f"main init {self.wav_file_list=}")  # remove before release

        self.wav_file = None
        self.project_json_path = None
        self.project_parameters = None

        if self.wav_file_list:
            self.wav_file = self.wav_file_list[0]
            self.project_json_path = find_project_json_for_wav(Path(self.wav_file))
            if self.project_json_path and self.project_json_path.is_file():
                with open(self.project_json_path, 'r', encoding='utf-8') as f:
                    self.project_parameters = json.load(f)
            print(f"{self.wav_file=}")  # remove before release
            self.load_wav(self.wav_file)
        else:
            self.data = np.array([])
            self.sampling_rate = None
            self.duration = 0
            self.time = np.array([])
            self.xmin, self.xmax = 0, 1
            self.zmin, self.zmax = 0, 1
            self.id_xmin, self.id_xmax = 0, 0
            self.canto = np.array([])
            self.rms = np.array([])
            self.rms_times = np.array([])
            self.peaks_times = np.array([])
            self.selected_peak_times = []
            self.setWindowTitle(
                f"{Path(__file__).stem.replace('_', ' ')} - (no file loaded)"
            )

        self.figure, self.ax = plt.subplots(figsize=(12, 5))
        self.figure.subplots_adjust(bottom=0.15)
        self.canvas = FigureCanvas(self.figure)

        self.span_selector = SpanSelector(
            self.ax,
            self.on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="yellow"),
        )

        self.canvas.mpl_connect("button_press_event", self.on_double_click)

        self.control_panel = ControlPanel(self)

        layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        self.previous_file_btn = QPushButton("Previous file")
        self.previous_file_btn.clicked.connect(self.previous_file_clicked)
        top_layout.addWidget(self.previous_file_btn)

        self.next_file_btn = QPushButton("Next file")
        self.next_file_btn.clicked.connect(self.next_file_clicked)
        top_layout.addWidget(self.next_file_btn)

        self.save_results_btn = QPushButton("Save results")
        self.save_results_btn.clicked.connect(self.save_results_clicked)
        top_layout.addWidget(self.save_results_btn)

        self.auto_btn = QPushButton("Auto")
        self.auto_btn.clicked.connect(self.auto_btn_clicked)
        top_layout.addWidget(self.auto_btn)

        top_layout.addSpacing(16)

        self.begin_end_btn = QPushButton("Start/End call")
        self.begin_end_btn.clicked.connect(self.begin_end)
        top_layout.addWidget(self.begin_end_btn)

        self.spectrum_btn = QPushButton("Spectrum")
        self.spectrum_btn.clicked.connect(self.compute_spectrum_clicked)
        top_layout.addWidget(self.spectrum_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        top_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stopplaying)
        top_layout.addWidget(self.stop_btn)

        self.delete_selected_peaks_btn = QPushButton("Delete selected peaks")
        self.delete_selected_peaks_btn.clicked.connect(self.delete_selected_peaks)
        top_layout.addWidget(self.delete_selected_peaks_btn)

        top_layout.addSpacing(16)

        self.toggle_params_btn = QPushButton("Show/Hide Parameters")
        self.toggle_params_btn.clicked.connect(self.toggle_control_panel)
        top_layout.addWidget(self.toggle_params_btn)

        top_layout.addStretch()
        layout.addLayout(top_layout)

        layout.addWidget(self.canvas)

        self.slider = QSlider()
        self.slider.setOrientation(self.slider.orientation().Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.on_slider)
        layout.addWidget(self.slider)

        self.setLayout(layout)
        self.resize(1450, 720)

        if self.wav_file_list:
            self.load_selected_song_from_project()
            self.run_analysis()
            self.zoomOut_wav()

    def show(self):
        super().show()
        self.control_panel.show()
        self.control_panel.raise_()

    def closeEvent(self, event):
        try:
            sd.stop()
        except Exception:
            pass
        if hasattr(self, "control_panel") and self.control_panel is not None:
            self.control_panel.close()
        if self.spectrum_window is not None:
            self.spectrum_window.close()
        super().closeEvent(event)

    def toggle_control_panel(self):
        if self.control_panel.isVisible():
            self.control_panel.hide()
        else:
            self.control_panel.show()
            self.control_panel.raise_()
            self.control_panel.activateWindow()

    def compute_envelope_clicked(self):
        self.envelope()

    def find_peaks_clicked(self):
        self.trova_picchi()

    def compute_spectrum_clicked(self):
        self.plot_spectrum()

    def init_values(self):
        self.window_size = WINDOW_SIZE
        self.overlap = OVERLAP
        self.min_amplitude = MIN_AMPLITUDE
        self.min_distance = MIN_DISTANCE
        self.max_distance = MAX_DISTANCE
        self.prominence = PROMINENCE
        self.signal_to_noise_ratio = SIGNAL_TO_NOISE_RATIO
        self.fft_length = FFT_LENGTH
        self.fft_overlap = FFT_OVERLAP
        self.max_freq = 12000

        self.results_dict = {
            "file": None,
            "sampling_rate": None,
            "call_duration": None,
            "pulse_number": None,
            "spectrum": None,
            "spectrum_peaks": None,
        }

    def run_analysis(self):
        self.envelope()
        self.trova_picchi()
        if len(self.peaks_times) > 0:
            self.trova_ini_fin()

    def load_selected_song_from_project(self):
        """
        Se il plugin è stato aperto dal main con uno o più WAV selezionati,
        prova a leggere automaticamente il JSON del progetto corrispondente
        e ad applicare i parametri del song selezionato.
        """
        if not self.wav_file:
            return
        if self.project_parameters is None:
            return

        chunk_name, sp = find_song_block_from_wav(
            self.project_parameters, Path(self.wav_file).name
        )
        if not isinstance(sp, dict):
            return

        self.selected_times = []
        self.apply_song_params(sp)

        self.peaks_times = np.array(sp.get('peaks_times', []), dtype=float)
        self.envelope(reset_manual=False)
        self.trova_intensita_picchi()

        ini = sp.get('call_start')
        dur = sp.get('call_duration')
        if ini is not None and dur is not None:
            self.selected_times = [float(ini), float(ini) + float(dur)]
            self.update_canto_from_selected_times()
            self.peaks_times = self.peaks_times[
                (self.peaks_times >= self.selected_times[0])
                & (self.peaks_times <= self.selected_times[1])
            ]

        self.current_chunk_name = chunk_name

    def load_job(self, index: int):
        if not hasattr(self, "song_jobs") or not self.song_jobs:
            return

        index = max(0, min(index, len(self.song_jobs) - 1))
        self.current_job_index = index
        wav_key, song_id, sp = self.song_jobs[index]

        if (
            hasattr(self, "wav_file_list")
            and self.wav_file_list
            and index < len(self.wav_file_list)
        ):
            self.wav_file = self.wav_file_list[index]
        else:
            self.wav_file = str(wav_path_same_folder(self.input_json_path, sp, wav_key))

        self.load_wav(self.wav_file)
        self.apply_song_params(sp)
        self.peaks_times = np.array(sp.get("peaks_times", []), dtype=float)
        self.envelope(reset_manual=False)
        self.trova_intensita_picchi()

        ini = sp.get("call_start")
        dur = sp.get("call_duration")
        if ini is not None and dur is not None:
            self.selected_times = [float(ini), float(ini) + float(dur)]
            self.update_canto_from_selected_times()
        else:
            self.trova_ini_fin()

        self.plot_wav(self.xmin, self.xmax)

    def load_wav_from_json(self, json_path: Path):
        with open(json_path, "r", encoding="utf-8") as f:
            parameters = json.load(f)

        wav_key = None
        block = None
        for candidate_key, candidate_block in parameters.items():
            if isinstance(candidate_block, dict):
                wav_key = candidate_key
                block = candidate_block
                break

        if wav_key is None or block is None:
            QMessageBox.warning(self, "", "Nessun blocco WAV trovato nel JSON")
            return

        wav_path = find_wav_next_to_json(json_path, wav_key, block)
        if wav_path is None:
            QMessageBox.critical(
                self,
                "",
                f"WAV non trovato nella cartella del JSON:\n{json_path.parent}",
            )
            return

        self.input_json_path = json_path
        self.output_json_path = json_path.with_name(json_path.stem + "_out2.json")
        if not self.output_json_path.exists():
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(parameters, f, indent=0, ensure_ascii=False)

        self.song_jobs = []
        self.wav_file_list = [str(wav_path)]
        self.current_job_index = 0
        self.wav_file = str(wav_path)

        self.load_wav(self.wav_file)
        self.selected_times = []
        self.apply_song_params(block)
        self.envelope(reset_manual=False)

        self.peaks_times = np.array(block.get("peaks_times", []), dtype=float)
        self.trova_intensita_picchi()
        if len(self.peaks_times) > 0:
            self.trova_ini_fin()
        else:
            self.plot_wav(self.xmin, self.xmax)

    def load_wav(self, wav_file):
        self.sampling_rate, data = wavfile.read(wav_file)
        if add_noise:
            data = add_noise_padding(
                data, self.sampling_rate, duration=0.025, noise_db=-20
            )
        self.data = data

        if len(self.data.shape) > 1:
            self.data = self.data[:, 0]

        self.data = self.data / np.max(np.abs(self.data))
        self.time = np.linspace(
            0, len(self.data) / self.sampling_rate, num=len(self.data)
        )
        self.duration = len(self.data) / self.sampling_rate
        self.xmin = 0
        self.xmax = self.duration
        self.zmin = 0
        self.zmax = self.duration
        self.id_xmin = 0
        self.id_xmax = len(self.data)

        self.canto = np.zeros(len(self.data))
        self.rms = np.zeros(max(1, len(self.data) // max(1, self.overlap)))
        self.rms_times = np.array([])
        self.peaks_times = np.array([])
        self.peaks_int = np.array([])
        self.selected_times = [0.0, 0.0]
        self.selected_peak_times = []
        self.span_region = None

        self.setWindowTitle(
            f"{Path(__file__).stem.replace('_', ' ')} - {Path(wav_file).stem}"
        )

    def plot_wav(self, xmin=None, xmax=None):
        if self.sampling_rate is None or self.data is None or len(self.data) == 0:
            return

        if xmin is not None:
            self.xmin = xmin
        if xmax is not None:
            self.xmax = xmax

        self.ax.cla()
        self.ax.plot(self.time, self.data, linewidth=0.5, color="black", alpha=0.35)

        if len(self.canto) > 0:
            canto_plot = self.canto[: len(self.time)]
            self.ax.plot(self.time, canto_plot, "-", color="blue", linewidth=1)

        if len(self.rms) > 0 and len(self.rms_times) > 0:
            self.ax.plot(self.rms_times, self.rms, linewidth=1, color="red")

        if len(self.peaks_times) > 0:
            ymax = np.max(self.rms) if len(self.rms) > 0 else 1
            for peak in self.peaks_times:
                is_selected = any(
                    np.isclose(peak, selected_peak)
                    for selected_peak in self.selected_peak_times
                )
                color = "blue" if is_selected else "green"
                self.ax.plot([peak, peak], [0, ymax], color=color, linewidth=2)

        self.ax.set_xlim(self.zmin, self.zmax)
        # self.update_manual_selection_overlay(draw=False)
        self.canvas.draw_idle()

    def update_canto_from_selected_times(self):
        if (
            not self.selected_times
            or len(self.selected_times) != 2
            or self.sampling_rate is None
            or len(self.rms) == 0
        ):
            return None, None

        inizio, fine = sorted(self.selected_times)
        inizio = max(0.0, min(float(inizio), self.duration))
        fine = max(0.0, min(float(fine), self.duration))
        self.selected_times = [inizio, fine]

        canto_len = (
            len(self.data)
            if self.data is not None and len(self.data) > 0
            else len(self.rms) * self.overlap
        )
        self.canto = np.zeros(canto_len)
        inizio_fr = int(inizio * self.sampling_rate)
        fine_fr = int(fine * self.sampling_rate)
        inizio_fr = max(0, min(inizio_fr, canto_len))
        fine_fr = max(inizio_fr, min(fine_fr, canto_len))
        self.canto[inizio_fr:fine_fr] = np.max(self.rms)

        return inizio_fr, fine_fr

    def update_manual_selection_overlay(self, draw=True):
        if (
            self.selected_times
            and len(self.selected_times) == 2
            and self.selected_times[1] > self.selected_times[0]
        ):
            self.ax.axvspan(
                self.selected_times[0],
                self.selected_times[1],
                color="yellow",
                alpha=0.25,
            )
        if draw:
            self.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        if abs(xmax - xmin) < 0.01:
            return
        self.xmin, self.xmax = sorted([max(0, xmin), min(self.duration, xmax)])
        self.selected_times = [0.0, 0.0]
        self.zoomIn_wav()
        self.trova_picchi()

    def on_slider(self, value):
        if self.duration <= 0:
            return

        range_view = self.zmax - self.zmin
        if range_view <= 0:
            return

        if range_view >= self.duration:
            self.zmin = 0
            self.zmax = self.duration
            if value != 0:
                self.slider.blockSignals(True)
                self.slider.setValue(0)
                self.slider.blockSignals(False)
            self.ax.set_xlim(self.zmin, self.zmax)
            self.canvas.draw_idle()
            return

        pos = value / 100 * max(0.0, self.duration - range_view)
        self.zmin = pos
        self.zmax = min(self.duration, pos + range_view)
        self.ax.set_xlim(self.zmin, self.zmax)
        self.canvas.draw_idle()

    def on_double_click(self, event):
        if event.button == 3:
            self.select_nearest_peak(event)
            return
        if event.dblclick:
            self.zoomOut_wav()

    def select_nearest_peak(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.peaks_times is None or len(self.peaks_times) == 0:
            QMessageBox.warning(self, "", "Nessun picco disponibile")
            return

        peaks = np.asarray(self.peaks_times, dtype=float)
        nearest_index = int(np.argmin(np.abs(peaks - float(event.xdata))))
        selected_peak = float(peaks[nearest_index])
        if any(np.isclose(selected_peak, peak) for peak in self.selected_peak_times):
            self.selected_peak_times = [
                peak
                for peak in self.selected_peak_times
                if not np.isclose(selected_peak, peak)
            ]
        else:
            self.selected_peak_times.append(selected_peak)

        self.plot_wav()

    def delete_selected_peaks(self):
        if not self.selected_peak_times:
            QMessageBox.information(self, "", "Nessun picco blu da cancellare")
            return
        if self.peaks_times is None or len(self.peaks_times) == 0:
            self.selected_peak_times = []
            QMessageBox.warning(self, "", "Nessun picco disponibile")
            return

        peaks = np.asarray(self.peaks_times, dtype=float)
        selected_peaks = np.asarray(self.selected_peak_times, dtype=float)
        keep_mask = np.ones(len(peaks), dtype=bool)
        for selected_peak in selected_peaks:
            keep_mask &= ~np.isclose(peaks, selected_peak)

        removed_count = int(np.sum(~keep_mask))
        self.peaks_times = peaks[keep_mask]
        self.selected_peak_times = []
        self.trova_intensita_picchi()
        self.plot_wav()

        if removed_count == 0:
            QMessageBox.information(self, "", "Nessun picco cancellato")

    def zoomIn_wav(self):
        if hasattr(self, "xmin") and hasattr(self, "xmax") and self.xmax > self.xmin:
            self.zmin, self.zmax = self.xmin, self.xmax
            width = self.zmax - self.zmin
            if self.duration > width:
                self.slider.setValue(int((self.zmin / (self.duration - width)) * 100))
            self.ax.set_xlim(self.zmin, self.zmax)
            self.canvas.draw_idle()

    def zoomOut_wav(self):
        self.xmin = 0
        self.xmax = self.duration
        self.zmin = 0
        self.zmax = self.duration
        self.slider.setValue(0)
        self.plot_wav()

    def begin_end(self):
        self.selected_times = []

        def onclick(event):
            if event.inaxes != self.ax or event.xdata is None or event.button != 1:
                return
            x_clicked = float(event.xdata)
            print(x_clicked)
            self.selected_times.append(x_clicked)
            if len(self.selected_times) == 2:
                inizio, fine = sorted(self.selected_times)
                self.selected_times = [inizio, fine]
                self.canvas.mpl_disconnect(self.cid_click)
                self.cid_click = None
                inizio_fr, fine_fr = self.update_canto_from_selected_times()
                self.durata_canto = (fine_fr - inizio_fr) / self.sampling_rate
                self.rms_canto = self.rms[
                    int(inizio_fr / self.overlap) : int(fine_fr / self.overlap)
                ]
                self.peaks_times = self.peaks_times[
                    (self.peaks_times >= inizio) & (self.peaks_times <= fine)
                ]
                self.trova_picchi()
                self.plot_wav()

        if self.cid_click is not None:
            self.canvas.mpl_disconnect(self.cid_click)

        self.cid_click = self.canvas.mpl_connect("button_press_event", onclick)

        QMessageBox.information(
            self,
            "Selezione in corso",
            "Clicca due volte sul grafico per selezionare inizio e fine del canto.",
        )

    def envelope(self, reset_manual=True):
        try:
            if self.window_size <= 0 or self.overlap <= 0:
                return
            self.rms = librosa.feature.rms(
                y=self.data, frame_length=self.window_size, hop_length=self.overlap
            )[0]
            self.rms_times = librosa.frames_to_time(
                np.arange(len(self.rms)), sr=self.sampling_rate, hop_length=self.overlap
            )
            self.canto = np.zeros(len(self.rms) * self.overlap)
            if reset_manual:
                self.selected_times = [0.0, 0.0]
            if (
                hasattr(self, "selected_times")
                and len(self.selected_times) == 2
                and self.selected_times[1] > self.selected_times[0]
            ):
                self.trova_picchi()
                self.trova_ini_fin()
            self.plot_wav()
        except Exception as e:
            print("Errore in envelope:", e)

    def plot_spectrum(self):
        try:
            if (
                self.fft_length <= 0
                or self.fft_overlap < 0
                or self.fft_overlap >= self.fft_length
            ):
                return

            xmin = (
                self.selected_times[0]
                if len(self.selected_times) == 2
                and self.selected_times[1] > self.selected_times[0]
                else self.xmin
            )
            xmax = (
                self.selected_times[1]
                if len(self.selected_times) == 2
                and self.selected_times[1] > self.selected_times[0]
                else self.xmax
            )

            self.id_xmin = int(xmin * self.sampling_rate)
            self.id_xmax = int(xmax * self.sampling_rate)
            segment = self.data[self.id_xmin : self.id_xmax]
            if len(segment) == 0:
                return

            step = self.fft_length - self.fft_overlap
            n_segments = (len(segment) - self.fft_overlap) // step
            if n_segments <= 0:
                padded = np.zeros(self.fft_length)
                padded[: len(segment)] = segment
                fft_vals = np.fft.fft(padded)
                power = np.abs(fft_vals) ** 2
            else:
                spectra = []
                for i in range(n_segments):
                    start = i * step
                    end = start + self.fft_length
                    if end > len(segment):
                        break
                    windowed = segment[start:end] * np.hamming(self.fft_length)
                    fft_vals = np.fft.fft(windowed, n=self.fft_length)
                    power = np.abs(fft_vals) ** 2
                    spectra.append(power)
                power = np.mean(np.array(spectra), axis=0)

            freqs = np.fft.fftfreq(self.fft_length, d=1 / self.sampling_rate)
            mask_positive = freqs >= 0
            freqs = freqs[mask_positive]
            power = power[mask_positive]
            avg_power_db = np.log10(power / np.max(power) + 1e-10)

            self.spectrum_peaks, _ = find_peaks(avg_power_db, height=-5, distance=1000)
            self.spectrum_peaks_Hz = freqs[self.spectrum_peaks]
            spectrum_peaks_db = avg_power_db[self.spectrum_peaks]

            self.results_dict["spectrum"] = np.concatenate(
                ([freqs], [power]), axis=0
            ).tolist()
            self.results_dict["spectrum_peaks"] = np.concatenate(
                ([self.spectrum_peaks_Hz], [spectrum_peaks_db]), axis=0
            ).tolist()

            if self.spectrum_window is None:
                self.spectrum_window = SpectrumWindow()
            self.spectrum_window.update_spectrum(
                freqs,
                avg_power_db,
                self.spectrum_peaks_Hz,
                spectrum_peaks_db,
                max_freq=self.max_freq,
            )
            self.spectrum_window.show()
            self.spectrum_window.raise_()
            self.spectrum_window.activateWindow()
        except Exception as e:
            print("Errore in plot_spectrum:", e)

    def trova_picchi(self):
        try:
            if len(self.rms) == 0:
                return

            if (
                hasattr(self, "selected_times")
                and len(self.selected_times) == 2
                and self.selected_times[1] > self.selected_times[0]
            ):
                print("passo di qui")
                xmin, xmax = self.selected_times
            else:
                xmin, xmax = (
                    (self.xmin, self.xmax)
                    if (self.xmin > 0 or self.xmax < self.duration)
                    else (0, self.duration)
                )

            print(f"selected_times = {xmin}; {xmax}")
            existing = np.asarray(self.peaks_times, dtype=float)
            keep_mask = ~((existing >= xmin) & (existing <= xmax))
            base_peaks = existing[keep_mask] if len(existing) else np.array([])

            min_distance_samples = int(
                self.min_distance * (self.sampling_rate / self.overlap)
            )
            max_distance_samples = int(
                self.max_distance * (self.sampling_rate / self.overlap)
            )
            mask_rms = (self.rms_times >= xmin) & (self.rms_times <= xmax)
            rms_selected = self.rms[mask_rms]
            rms_times_selected = self.rms_times[mask_rms]
            if len(rms_selected) == 0:
                return

            peaks, _ = find_peaks(
                rms_selected,
                height=self.min_amplitude,
                distance=max(1, min_distance_samples),
                prominence=self.prominence,
            )

            if peaks.size == 0:
                self.peaks_times = np.sort(base_peaks)
                self.plot_wav()
                return

            peaks_filtered = [peaks[0]]
            for i in np.arange(1, len(peaks)):
                if peaks[i] - peaks_filtered[-1] < max_distance_samples:
                    peaks_filtered.append(peaks[i])

            peaks_filtered = np.array(peaks_filtered)
            if len(peaks_filtered) > 1:
                mean_distance_between_peaks = np.mean(np.diff(peaks_filtered))
                sdt_distance_between_peaks = np.std(np.diff(peaks_filtered))
                refined = [peaks_filtered[0]]
                for i in np.arange(1, len(peaks_filtered)):
                    if (
                        peaks_filtered[i] - refined[-1]
                    ) < mean_distance_between_peaks + 3 * sdt_distance_between_peaks:
                        refined.append(peaks_filtered[i])
                peaks_filtered = np.array(refined)

            new_peaks_times = rms_times_selected[peaks_filtered]
            self.peaks_times = np.sort(np.concatenate((base_peaks, new_peaks_times)))
            self.trova_intensita_picchi()
            if len(self.peaks_times) > 0 and not (
                len(self.selected_times) == 2
                and self.selected_times[1] > self.selected_times[0]
            ):
                self.trova_ini_fin()
            else:
                self.plot_wav()
        except Exception as e:
            QMessageBox.critical(
                self,
                "",
                f"Funzione Trova picchi\n\nError on file {self.wav_file}\n\n{e}",
            )

    def trova_intensita_picchi(self):
        if self.rms is None or len(self.rms) == 0:
            self.peaks_int = np.array([])
            return
        if self.peaks_times is None or len(self.peaks_times) == 0:
            self.peaks_int = np.array([])
            return

        pt = np.asarray(self.peaks_times, dtype=float)
        idx = np.searchsorted(self.rms_times, pt)
        idx = np.clip(idx, 0, len(self.rms) - 1)
        self.peaks_int = self.rms[idx]

    def trova_ini_fin(self):
        peaks = self.peaks_times * self.sampling_rate / self.overlap
        peaks = np.asarray(peaks, dtype=np.float64)

        if peaks.size == 0:
            anchor = int(np.argmax(self.rms))
            peaks = np.array([anchor, anchor], dtype=float)
        elif peaks.size == 1:
            peaks = np.array([peaks[0], peaks[0]], dtype=float)

        p0 = int(np.clip(peaks[0], 0, len(self.rms) - 1))
        p1 = int(np.clip(peaks[-1], 0, len(self.rms) - 1))

        w = 11
        if len(self.rms) >= w:
            kernel = np.ones(w) / w
            rms_s = np.convolve(self.rms, kernel, mode="same")
        else:
            rms_s = self.rms

        ini_end = max(1, int(p0 * 0.9))
        fin_start = min(len(rms_s) - 1, p1)
        rms_ini_noise = rms_s[:ini_end]
        rms_fin_noise = rms_s[fin_start:] if fin_start < len(rms_s) else rms_s[-1:]
        if len(rms_ini_noise) < 10:
            rms_ini_noise = rms_s
        if len(rms_fin_noise) < 10:
            rms_fin_noise = rms_s

        q = 0.6
        noise_ini = np.quantile(rms_ini_noise, q)
        noise_fin = np.quantile(rms_fin_noise, q)

        def mad(x):
            x = np.asarray(x)
            m = np.median(x)
            return np.median(np.abs(x - m)) + 1e-12

        mad_ini = mad(rms_ini_noise)
        mad_fin = mad(rms_fin_noise)

        k_on = self.signal_to_noise_ratio
        k_off = max(1.0, 2 * k_on)
        thr_off_ini = noise_ini + k_off * mad_ini
        thr_off_fin = noise_fin + k_off * mad_fin

        max_back_ms = 100
        start_i = max(
            0, p0 - int((max_back_ms / 1000) * self.sampling_rate / self.overlap)
        )
        inizio_frame = None
        count_off = 0
        for i in range(p0, start_i - 1, -1):
            if rms_s[i] <= thr_off_ini:
                count_off += 1
                if count_off >= 1:
                    inizio_frame = i
                    break
            else:
                count_off = 0

        if inizio_frame is not None:
            inizio = int((inizio_frame + 3) * self.overlap)
        else:
            inizio = int(start_i * self.overlap)

        inizio = min(p0 * self.overlap - 1, inizio)
        self.ini_canto = inizio

        fine_frame = len(rms_s) - 1
        for i in range(p1, len(rms_s)):
            if rms_s[i] <= thr_off_fin:
                fine_frame = i
                break

        fine = int((fine_frame + 1) * self.overlap)
        fine = max(fine, inizio + 1)
        fine = min(fine, len(rms_s) * self.overlap)

        self.canto = np.zeros(len(self.rms) * self.overlap)
        self.canto[inizio:fine] = np.max(self.rms)
        self.selected_times = [inizio, fine]
        self.durata_canto = (fine - inizio) / self.sampling_rate
        self.rms_canto = self.rms[int(inizio / self.overlap) : int(fine / self.overlap)]
        self.selected_times = [inizio / self.sampling_rate, fine / self.sampling_rate]
        self.plot_wav()

    def save_results_clicked(self):
        """ """

        json_file_path = find_project_json_for_wav(Path(self.wav_file))
        if json_file_path is None or not json_file_path.is_file():
            QMessageBox.warning(self, "", "JSON di progetto non trovato")
            return

        with open(json_file_path, "r", encoding="utf-8") as f:
            parameters = json.load(f)

        """
        if not hasattr(self, "output_json_path"):
            QMessageBox.warning(self, "", "Carica prima un JSON")
            return
        if (
            len(self.selected_times) != 2
            or self.selected_times[1] <= self.selected_times[0]
        ):
            QMessageBox.warning(self, "", "Definisci prima inizio e fine del canto")
            return

        data_file_path = self.output_json_path
        with open(data_file_path, "r", encoding="utf-8") as f:
            parameters = json.load(f)
        """

        sample = int(Path(self.wav_file).stem.split("_")[-1])

        wav_file_name = Path(self.wav_file).name

        chunk_file_name, _song_block = find_song_block_from_wav(parameters, wav_file_name)
        if chunk_file_name is None:
            # fallback per vecchia convenzione basata su _sample_
            chunk_file_name = Path(self.wav_file.split("_sample_")[0] + ".wav").name

        if chunk_file_name not in parameters.get("chunks", {}):
            print(f"{chunk_file_name} NOT FOUND!")
            QMessageBox.warning(self, "", f"{chunk_file_name} NOT FOUND!")
            return

        parameters["chunks"].setdefault(chunk_file_name, {})
        parameters["chunks"][chunk_file_name].setdefault("songs", {})
        parameters["chunks"][chunk_file_name]["songs"].setdefault(wav_file_name, {})

        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["window_size"] = (
            self.window_size
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["overlap"] = (
            self.overlap
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
            "min_amplitude"
        ] = self.min_amplitude
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
            "min_distance"
        ] = self.min_distance
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
            "max_distance"
        ] = self.max_distance
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["prominence"] = (
            self.prominence
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
            "signal_to_noise_ratio"
        ] = self.signal_to_noise_ratio
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["fft_length"] = (
            self.fft_length
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["fft_overlap"] = (
            self.fft_overlap
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
            "sampling rate"
        ] = self.sampling_rate
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["call_start"] = (
            float(self.selected_times[0])
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
            "call_duration"
        ] = float(self.selected_times[1] - self.selected_times[0])
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
            "pulse_number"
        ] = len(self.peaks_times)
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["peaks_times"] = (
            self.peaks_times.tolist()
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["peaks_int"] = (
            self.peaks_int.tolist() if len(self.peaks_int) else []
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name]["spectrum"] = (
            self.results_dict["spectrum"]
        )
        parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
            "spectrum peaks"
        ] = self.results_dict["spectrum_peaks"]

        """
        if hasattr(self, "song_jobs") and self.song_jobs:
            wav_key, song_id, sp = self.song_jobs[self.current_job_index]
            file_name = wav_key
        else:
            file_name = Path(self.wav_file).name
        parameters.setdefault(file_name, {})
        parameters[file_name].setdefault("songs", {})
        parameters[file_name]["songs"][str(sample)] = {}
        block = parameters[file_name]["songs"][str(sample)]
        block["file"] = Path(self.wav_file).name
        block["window_size"] = self.window_size
        block["overlap"] = self.overlap
        block["min_amplitude"] = self.min_amplitude
        block["min_distance"] = self.min_distance
        block["max_distance"] = self.max_distance
        block["prominence"] = self.prominence
        block["signal_to_noise_ratio"] = self.signal_to_noise_ratio
        block["fft_length"] = self.fft_length
        block["fft_overlap"] = self.fft_overlap
        block["sampling rate"] = self.sampling_rate
        block["call_start"] = float(self.selected_times[0])
        block["call_duration"] = float(self.selected_times[1] - self.selected_times[0])
        block["pulse_number"] = len(self.peaks_times)
        block["peaks_times"] = self.peaks_times.tolist()
        block["peaks_int"] = self.peaks_int.tolist() if len(self.peaks_int) else []
        block["spectrum"] = self.results_dict["spectrum"]
        block["spectrum peaks"] = self.results_dict["spectrum_peaks"]
        """

        if len(self.rms) > 0:
            sel_mask = (self.rms_times >= self.selected_times[0]) & (
                self.rms_times <= self.selected_times[1]
            )
            self.rms_canto = self.rms[sel_mask]
            envelope_max = np.max(self.rms) if len(self.rms) else 0
            soglia20 = 0.2 * envelope_max
            soglia50 = 0.5 * envelope_max
            soglia80 = 0.8 * envelope_max
            if len(self.rms_canto) > 0:
                parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
                    "envelope20"
                ] = float(np.mean(self.rms_canto >= soglia20))
                parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
                    "envelope50"
                ] = float(np.mean(self.rms_canto >= soglia50))
                parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
                    "envelope80"
                ] = float(np.mean(self.rms_canto >= soglia80))
            else:
                parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
                    "envelope20"
                ] = np.nan
                parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
                    "envelope50"
                ] = np.nan
                parameters["chunks"][chunk_file_name]["songs"][wav_file_name][
                    "envelope80"
                ] = np.nan

        with open(json_file_path, "w", encoding="utf-8") as f_out:
            json.dump(parameters, f_out, indent=0, ensure_ascii=False)

        # with open(data_file_path.with_suffix(".pkl"), "wb") as f_out:
        #    pickle.dump(parameters, f_out)

        row = {
            "file_wav": wav_file_name,
            "specie": wav_file_name.split("_")[0],
            "data": wav_file_name.split("_")[1]
            if len(wav_file_name.split("_")) > 1
            else None,
            "id": wav_file_name.split("_")[2]
            if len(wav_file_name.split("_")) > 2
            else None,
            "note_peak": sample / self.sampling_rate,
            "inter_note": None,
            "bout": None,
            "note_period": None,
            "durata_nota": float(self.selected_times[1] - self.selected_times[0]),
            "pulse_number": len(self.peaks_times),
            "pulse_rate": (len(self.peaks_times) - 1)
            / max(1e-12, float(self.selected_times[1] - self.selected_times[0])),
            "fq_peak": self.results_dict["spectrum_peaks"][0][0]
            if self.results_dict["spectrum_peaks"]
            else None,
        }

        self.rows = [r for r in self.rows if r.get("note_peak") != row["note_peak"]]
        self.rows.append(row)
        self.df_results = pd.DataFrame(getattr(self, "rows", []))
        name_outfile = json_file_path.with_suffix(".csv")
        self.df_results.to_csv(name_outfile, sep=";", encoding="utf-8", index=False)
        QMessageBox.information(self, "", f"Risultati salvati in {json_file_path}")

    def next_file_clicked(self):
        if hasattr(self, "song_jobs") and self.song_jobs:
            if self.current_job_index >= len(self.song_jobs) - 1:
                QMessageBox.critical(self, "", "Last file")
                return
            self.load_job(self.current_job_index + 1)
            return

        if not self.wav_file_list:
            return
        current_wav_index = self.wav_file_list.index(self.wav_file)
        if current_wav_index == len(self.wav_file_list) - 1:
            QMessageBox.critical(self, "", "Last file")
            return

        self.wav_file = self.wav_file_list[current_wav_index + 1]
        self.load_wav(self.wav_file) 
        self.run_analysis()

    def previous_file_clicked(self):
        if hasattr(self, "song_jobs") and self.song_jobs:
            if self.current_job_index <= 0:
                QMessageBox.critical(self, "", "First file of directory")
                return
            self.load_job(self.current_job_index - 1)
            return

        if not self.wav_file_list:
            return
        current_wav_index = self.wav_file_list.index(self.wav_file)
        if current_wav_index == 0:
            QMessageBox.critical(self, "", "First file of directory")
            return

        self.wav_file = self.wav_file_list[current_wav_index - 1]
        self.load_wav(self.wav_file)
        self.run_analysis()

    def auto_btn_clicked(self):
        if hasattr(self, "song_jobs") and self.song_jobs:
            start = getattr(self, "current_job_index", 0)
            for idx in range(start, len(self.song_jobs)):
                self.load_job(idx)
                self.save_results_clicked()
        else:
            for wav_file in self.wav_file_list[
                self.wav_file_list.index(self.wav_file) :
            ]:
                self.wav_file = wav_file
                self.load_wav(self.wav_file)
                self.run_analysis()
                self.save_results_clicked()

        self.df_results = pd.DataFrame(getattr(self, "rows", []))
        if hasattr(self, "output_json_path"):
            name_outfile = self.output_json_path.with_suffix(".csv")
            self.df_results.to_csv(name_outfile, sep=";", encoding="utf-8", index=False)

    def apply_song_params(self, sp: dict):
        self.window_size = int(sp.get("window_size", self.window_size))
        self.overlap = int(sp.get("overlap", self.overlap))
        self.min_amplitude = float(sp.get("min_amplitude", self.min_amplitude))
        self.min_distance = float(sp.get("min_distance", self.min_distance))
        self.max_distance = float(sp.get("max_distance", self.max_distance))
        self.prominence = float(sp.get("prominence", self.prominence))
        self.signal_to_noise_ratio = float(
            sp.get("signal_to_noise_ratio", self.signal_to_noise_ratio)
        )
        self.fft_length = int(sp.get("fft_length", self.fft_length))
        self.fft_overlap = int(sp.get("fft_overlap", self.fft_overlap))

        cp = self.control_panel
        cp.window_size_input.setValue(self.window_size)
        cp.overlap_input.setValue(self.overlap)
        cp.amp_threshold_input.setValue(self.min_amplitude)
        cp.min_distance_input.setValue(self.min_distance)
        cp.max_distance_input.setValue(self.max_distance)
        cp.prominence_input.setValue(self.prominence)
        cp.signal_noise_ration_input.setValue(self.signal_to_noise_ratio)
        cp.fft_length_combo.setCurrentText(str(self.fft_length))
        cp.fft_overlap_combo.setCurrentText(
            str(int(round(100 * self.fft_overlap / max(1, self.fft_length))))
        )

    def play_audio(self):
        if self.sampling_rate is None or self.data is None or len(self.data) == 0:
            QMessageBox.warning(self, "", "Carica prima un file WAV")
            return

        xmin = self.xmin
        xmax = self.xmax
        if (
            len(self.selected_times) == 2
            and self.selected_times[1] > self.selected_times[0]
        ):
            xmin, xmax = self.selected_times

        start = max(0, min(int(xmin * self.sampling_rate), len(self.data)))
        end = max(start, min(int(xmax * self.sampling_rate), len(self.data)))
        if end <= start:
            QMessageBox.warning(self, "", "Nessun segmento audio da riprodurre")
            return

        sd.stop()
        sd.play(self.data[start:end], samplerate=self.sampling_rate)

    def stopplaying(self):
        sd.stop()


class ControlPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setWindowTitle("Parameters")
        self.setGeometry(1150, 100, 390, 560)

        main_layout = QVBoxLayout()

        envelope_layout = QVBoxLayout()
        envelope_layout.addWidget(QLabel("Envelope Parameters"))
        h_layout = QHBoxLayout()
        self.window_size_input = QSpinBox()
        self.window_size_input.setMinimum(10)
        self.window_size_input.setMaximum(1000)
        self.window_size_input.setValue(WINDOW_SIZE)
        self.window_size_input.setSingleStep(10)
        self.window_size_input.valueChanged.connect(self.window_size_changed)

        self.overlap_input = QSpinBox()
        self.overlap_input.setMinimum(10)
        self.overlap_input.setMaximum(1000)
        self.overlap_input.setValue(OVERLAP)
        self.overlap_input.setSingleStep(10)
        self.overlap_input.valueChanged.connect(self.overlap_changed)

        h_layout.addWidget(QLabel("Window size"))
        h_layout.addWidget(self.window_size_input)
        h_layout.addWidget(QLabel("Overlap"))
        h_layout.addWidget(self.overlap_input)
        envelope_layout.addLayout(h_layout)
        main_layout.addLayout(envelope_layout)

        peak_layout = QVBoxLayout()
        peak_layout.addWidget(QLabel("Peak Finder Parameters"))

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Amplitude threshold"))
        self.amp_threshold_input = QDoubleSpinBox()
        self.amp_threshold_input.setDecimals(3)
        self.amp_threshold_input.setSingleStep(0.005)
        self.amp_threshold_input.setMinimum(0.0)
        self.amp_threshold_input.setMaximum(1)
        self.amp_threshold_input.setValue(MIN_AMPLITUDE)
        self.amp_threshold_input.valueChanged.connect(self.min_amplitude_changed)
        h_layout.addWidget(self.amp_threshold_input)
        peak_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Min distance"))
        self.min_distance_input = QDoubleSpinBox()
        self.min_distance_input.setDecimals(3)
        self.min_distance_input.setSingleStep(0.0005)
        self.min_distance_input.setMinimum(0.0005)
        self.min_distance_input.setMaximum(0.3)
        self.min_distance_input.setValue(MIN_DISTANCE)
        self.min_distance_input.valueChanged.connect(self.min_distance_changed)
        h_layout.addWidget(self.min_distance_input)

        h_layout.addWidget(QLabel("Max distance"))
        self.max_distance_input = QDoubleSpinBox()
        self.max_distance_input.setDecimals(3)
        self.max_distance_input.setSingleStep(0.001)
        self.max_distance_input.setMinimum(0.0)
        self.max_distance_input.setMaximum(2)
        self.max_distance_input.setValue(MAX_DISTANCE)
        self.max_distance_input.valueChanged.connect(self.max_distance_changed)
        h_layout.addWidget(self.max_distance_input)
        peak_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Prominence"))
        self.prominence_input = QDoubleSpinBox()
        self.prominence_input.setDecimals(3)
        self.prominence_input.setSingleStep(0.01)
        self.prominence_input.setMinimum(0.01)
        self.prominence_input.setMaximum(0.9)
        self.prominence_input.setValue(PROMINENCE)
        self.prominence_input.valueChanged.connect(self.prominence_changed)
        h_layout.addWidget(self.prominence_input)
        peak_layout.addLayout(h_layout)
        main_layout.addLayout(peak_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("S/N"))
        self.signal_noise_ration_input = QDoubleSpinBox()
        self.signal_noise_ration_input.setDecimals(1)
        self.signal_noise_ration_input.setSingleStep(0.1)
        self.signal_noise_ration_input.setMinimum(0.1)
        self.signal_noise_ration_input.setMaximum(10)
        self.signal_noise_ration_input.setValue(SIGNAL_TO_NOISE_RATIO)
        self.signal_noise_ration_input.valueChanged.connect(
            self.signal_to_noise_ratio_changed
        )
        h_layout.addWidget(self.signal_noise_ration_input)
        main_layout.addLayout(h_layout)

        spectrum_layout = QVBoxLayout()
        spectrum_layout.addWidget(QLabel("Spectrum Parameters"))
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("FFT"))
        self.fft_length_combo = QComboBox()
        self.fft_length_combo.addItems(["256", "512", "1024", "2048", "4096"])
        self.fft_length_combo.setCurrentText(str(FFT_LENGTH))
        self.fft_length_combo.currentTextChanged.connect(self.fft_length_changed)
        h_layout.addWidget(self.fft_length_combo)

        h_layout.addWidget(QLabel("Overlap %"))
        self.fft_overlap_combo = QComboBox()
        self.fft_overlap_combo.addItems(["0", "25", "50", "75"])
        self.fft_overlap_combo.setCurrentText("50")
        self.fft_overlap_combo.currentTextChanged.connect(self.fft_overlap_changed)
        h_layout.addWidget(self.fft_overlap_combo)
        spectrum_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Max Fq (Hz)"))
        self.max_freq_input = QLineEdit("12000")
        self.max_freq_input.editingFinished.connect(self.max_freq_changed)
        h_layout.addWidget(self.max_freq_input)
        spectrum_layout.addLayout(h_layout)
        main_layout.addLayout(spectrum_layout)

        h_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset values")
        self.reset_btn.clicked.connect(self.reset_values)
        h_layout.addWidget(self.reset_btn)

        self.json_btn = QPushButton("Carica da Json")
        self.json_btn.clicked.connect(self.leggi_json)
        h_layout.addWidget(self.json_btn)

        self.wav_from_json_btn = QPushButton("Carica WAV da Json")
        self.wav_from_json_btn.clicked.connect(self.carica_wav_da_json)
        h_layout.addWidget(self.wav_from_json_btn)
        main_layout.addLayout(h_layout)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def leggi_json(self):
        start_dir = str(getattr(self.main, "json_root", ""))
        json_file, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona file JSON",
            start_dir,
            "JSON files (*.json);All files (*.*)",
        )
        if not json_file:
            return

        json_path = Path(json_file)
        self.main.input_json_path = json_path
        self.main.output_json_path = json_path.with_name(json_path.stem + "_out2.json")

        if not self.main.output_json_path.exists():
            with open(self.main.input_json_path, "r", encoding="utf-8") as f:
                base = json.load(f)
            with open(self.main.output_json_path, "w", encoding="utf-8") as f:
                json.dump(base, f, indent=0, ensure_ascii=False)

        with open(json_path, "r", encoding="utf-8") as f:
            parameters = json.load(f)

        wav_level_peaks_map = {}
        for wav_key, wav_block in parameters.items():
            if not isinstance(wav_block, dict):
                continue
            wav_level_peaks_map[wav_key] = np.array(
                wav_block.get("peaks_times", []), dtype=float
            )

        jobs = list(iter_song_jobs(parameters))
        if not jobs:
            QMessageBox.warning(self, "", "Nessun 'song' trovato nel JSON")
            return

        wav_list = []
        song_jobs = []
        for wav_key, song_id, sp in jobs:
            wav_path = wav_path_different_folder(json_path, sp, wav_key)
            if not wav_path.is_file():
                print(f"[WARN] WAV non trovato: {wav_path}")
                continue
            wav_list.append(str(wav_path))
            song_jobs.append((wav_key, song_id, sp))

        if not wav_list:
            QMessageBox.critical(self, "", "Nessun WAV valido trovato")
            return

        self.main.song_jobs = song_jobs
        self.main.wav_file_list = wav_list
        self.main.current_job_index = 0
        self.main.wav_level_peaks_map = wav_level_peaks_map
        self.main.load_job(0)

    def carica_wav_da_json(self):
        start_dir = str(getattr(self.main, "json_root", ""))
        json_file, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona file JSON con WAV",
            start_dir,
            "JSON files (*.json);All files (*.*)",
        )
        if not json_file:
            return

        self.main.load_wav_from_json(Path(json_file))

    def reset_values(self):
        self.window_size_input.setValue(WINDOW_SIZE)
        self.overlap_input.setValue(OVERLAP)
        self.amp_threshold_input.setValue(MIN_AMPLITUDE)
        self.min_distance_input.setValue(MIN_DISTANCE)
        self.max_distance_input.setValue(MAX_DISTANCE)
        self.prominence_input.setValue(PROMINENCE)
        self.signal_noise_ration_input.setValue(SIGNAL_TO_NOISE_RATIO)
        self.fft_length_combo.setCurrentText(str(FFT_LENGTH))
        self.fft_overlap_combo.setCurrentText("50")
        self.max_freq_input.setText("12000")
        self.main.max_freq = 12000

    def window_size_changed(self, new_value):
        self.main.window_size = new_value
        self.main.envelope(reset_manual=False)
        self.main.trova_picchi()

    def overlap_changed(self, new_value):
        self.main.overlap = new_value
        self.main.envelope(reset_manual=False)
        self.main.trova_picchi()

    def min_amplitude_changed(self, new_value):
        self.main.min_amplitude = new_value
        self.main.trova_picchi()

    def min_distance_changed(self, new_value):
        self.main.min_distance = new_value
        self.main.trova_picchi()

    def max_distance_changed(self, new_value):
        self.main.max_distance = new_value
        self.main.trova_picchi()

    def prominence_changed(self, new_value):
        self.main.prominence = new_value
        self.main.trova_picchi()

    def signal_to_noise_ratio_changed(self, new_value):
        self.main.signal_to_noise_ratio = new_value
        if len(getattr(self.main, "peaks_times", [])) > 0 and not (
            len(self.main.selected_times) == 2
            and self.main.selected_times[1] > self.main.selected_times[0]
        ):
            self.main.trova_ini_fin()
        self.main.trova_picchi()

    def fft_length_changed(self, value):
        self.main.fft_length = int(value)
        overlap_percent = int(self.fft_overlap_combo.currentText())
        self.main.fft_overlap = int(overlap_percent / 100 * self.main.fft_length)

    def fft_overlap_changed(self, value):
        self.main.fft_overlap = int(int(value) / 100 * self.main.fft_length)

    def max_freq_changed(self):
        try:
            self.main.max_freq = float(self.max_freq_input.text())
        except ValueError:
            self.max_freq_input.setText(str(self.main.max_freq))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = Main(wav_file_list=[])
    main.show()
    QTimer.singleShot(0, main.control_panel.leggi_json)
    sys.exit(app.exec())
