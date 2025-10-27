"""
Plugin analisi canti 
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import shutil
import json
from scipy.io import wavfile
from scipy.signal import find_peaks
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QDoubleSpinBox,
    QMessageBox,
    QGridLayout,
    QSlider,
    QComboBox,
    QVBoxLayout
    )
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector
import librosa
import sounddevice as sd


class Main(QWidget):
    def __init__(self, wav_file_list: list):
        super().__init__()

        if wav_file_list:
            self.wav_file_list = wav_file_list
            self.wav_file = wav_file_list[0]
        else:
            QMessageBox.critical(
                self,
                "",
                "No file WAV!",
            )
            return

        self.start_times = []
        self.end_times = []
        self.span_region = []
        self.fft_length = 512
        self.fft_overlap = self.fft_length//2
            # Crea il dizionario dei risultati
        self.results_dict = {
            "file": None,
            "sampling_rate": None,
            "call_duration": None,
            "pulse_number": None,
            "spectrum": None,
            "spectrum_peaks": None,
        }


        self.setWindowTitle(
            f"{Path(__file__).stem.replace('_', ' ')} - {Path(self.wav_file).stem}"
        )

        # Layout principale a griglia (10 colonne)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.widgets_riga1_2 = {
            "zoom_in": {
                "type": "QPushButton",
                "label": "Zoom IN",
                "height_label": 10,
                "row": 1,
                "col": 0,
                "row_span": 1,
                "col_span": 1,
                "default": None,
                "linked_fnc": "zoomIn_wav",
                "widget": None,
            },
            "zoom_out": {
                "type": "QPushButton",
                "label": "Zoom OUT",
                "row": 1,
                "col": 1,
                "row_span": 1,
                "col_span": 1,
                "default": None,
                "linked_fnc": "zoomOut_wav",
                "widget": None,
            },
                        "begin_end": {
                "type": "QPushButton",
                "label": "Start/End call",
                "row": 1,
                "col": 2,
                "row_span": 1,
                "col_span": 1,
                "default": None,
                "linked_fnc": "begin_end",
                "widget": None,
            },
            "Envelope": {
                "type": "QPushButton",
                "label": "Envelope",
                "row": 0,
                "col": 3,
                "row_span": 2,
                "col_span": 1,
                "default": None,
                "linked_fnc": "envelope",
                "widget": None,
            },
            "w_size": {
                "type": "QLineEdit",
                "label": "W Size (ms)",
                "row": 1,
                "col": 4,
                "row_span": 1,
                "col_span": 1,
                "default": "1",
                "widget": None,
            },
            "w_overlap": {
                "type": "QLineEdit",
                "label": "W step (ms)",
                "row": 1,
                "col": 5,
                "row_span": 1,
                "col_span": 1,
                "default": "0.1",
                "widget": None,
            },
            "Peaks": {
                "type": "QPushButton",
                "label": "Pulses",
                "row": 0,
                "col": 7,
                "row_span": 2,
                "col_span": 1,
                "default": None,
                "linked_fnc": "fd_peaks_button",
                "widget": None,
            },
            "min_amp": {
                "type": "QDoubleSpinBox",
                "label": "Min Amp",
                "row": 1,
                "col": 8,
                "row_span": 1,
                "col_span": 1,
                "default": ["0.1", "3", "0.005", "0.0", "1"],
                "linked_fnc": "fd_peaks_spinbox",
                "widget": None,
            },
            "min_dist": {
                "type": "QLineEdit",
                "label": "Min distance (s)",
                "row": 1,
                "col": 9,
                "row_span": 1,
                "col_span": 1,
                "default": "0.001",
                "widget": None,
            },
        }

        self.widgets_rigaFinale = {
            "Power spectrum": {
                "type": "QPushButton",
                "label": "Spectrum",
                "height_label": 40,
                "row": 5,
                "col": 0,
                "row_span": 1,
                "col_span": 2,
                "linked_fnc": "plot_spectrum",
                "widget": None,
                "default": None,
            },
            "FFT": {
                "type": "QComboBox",
                "label": "FFT",
                "row": 5,
                "col": 2,
                "row_span": 1,
                "col_span": 1,
                "default": ["256", "512", "1024", "2048"],
                "linked_fnc": "fft_changed",  
                "widget": None,
            },
            "FFT overlap": {
                "type": "QComboBox",
                "label": "Overlap %",
                "row": 5,
                "col": 3,
                "row_span": 1,
                "col_span": 1,
                "default": ["0", "25", "50", "75"],
                "linked_fnc": "overlap_changed",  
                "widget": None,
            },
             "Max_freq": {
                "type": "QLineEdit",
                "label": "Max Fq (Hz)",
                "row": 5,
                "col": 4,
                "row_span": 1,
                "col_span": 1,
                "default": "12000",
                "widget": None,
            },
            "save_results": {
                "type": "QPushButton",
                "label": "Save Results",
                "height_label": 40,
                "row": 5,
                "col": 9,
                "row_span": 1,
                "col_span": 3,
                "linked_fnc": "save_results",
                "widget": None,
                "default": None,
            },
            "Play": {
                "type": "QPushButton",
                "label": "PLAY",
                "height_label": 40,
                "row": 5,
                "col": 6,
                "row_span": 1,
                "col_span": 1,
                "linked_fnc": "play_audio",
                "widget": None,
                "default": None,
            },
            "Stop": {
                "type": "QPushButton",
                "label": "STOP",
                "height_label": 40,
                "row": 5,
                "col": 7,
                "row_span": 1,
                "col_span": 1,
                "linked_fnc": "stopplaying",
                "widget": None,
                "default": None,
            },
        }

        for key, props in self.widgets_riga1_2.items():
            widget = None
            # Creazione del widget corrispondente
            if props["type"] == "QPushButton":
                widget = QPushButton(props["label"])
                self.layout.addWidget(
                    widget,
                    props["row"],
                    props["col"],
                    props["row_span"],
                    props["col_span"],
                )
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(
                        self, props["linked_fnc"]
                    )  # Recupera la funzione
                    widget.clicked.connect(
                        linked_function
                    )  # Connetti il pulsante alla funzione

            elif props["type"] == "QLineEdit":
                widget = QLineEdit()
                label = QLabel(props["label"])
                self.layout.addWidget(
                    label, props["row"] - 1, props["col"], 1, props["col_span"]
                )
                label.setFixedHeight(10)
                self.layout.addWidget(
                    widget, props["row"], props["col"], 1, props["col_span"]
                )
                if props["default"] is not None:
                    widget.setText(props["default"])

            elif props["type"] == "QDoubleSpinBox":
                label = QLabel(props["label"])
                self.layout.addWidget(
                    label, props["row"] - 1, props["col"], 1, props["col_span"]
                )

                widget = QDoubleSpinBox()

                if props["default"] is not None:
                    widget.setMinimum(float(props["default"][3]))
                    widget.setMaximum(float(props["default"][4]))
                    widget.setDecimals(float(props["default"][1]))
                    widget.setSingleStep(float(props["default"][2]))

                    widget.setValue(float(props["default"][0]))
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(
                        self, props["linked_fnc"]
                    )  # Recupera la funzione
                    widget.valueChanged.connect(
                        linked_function
                    )  # Connetti il pulsante alla funzione

                self.layout.addWidget(
                    widget, props["row"], props["col"], 1, props["col_span"]
                )
            elif props["type"] == "QComboBox":
                widget = QComboBox()
                label = QLabel(props["label"])
                self.layout.addWidget(label, props["row"] - 1, props["col"], 1, props["col_span"])
                self.layout.addWidget(widget, props["row"], props["col"], 1, props["col_span"])
                if props["default"]:
                    widget.addItems(props["default"])
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(self, props["linked_fnc"])
                    widget.currentTextChanged.connect(linked_function)
            
            else:
                continue  # Ignora eventuali tipi non definiti
            if widget:
                self.widgets_riga1_2[key]["widget"] = widget

        for key, props in self.widgets_rigaFinale.items():
            widget = None
            # Creazione del widget corrispondente
            if props["type"] == "QPushButton":
                widget = QPushButton(props["label"])
                self.layout.addWidget(
                    widget,
                    props["row"],
                    props["col"],
                    props["row_span"],
                    props["col_span"],
                )
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(
                        self, props["linked_fnc"]
                    )  # Recupera la funzione
                    widget.clicked.connect(
                        linked_function
                    )  # Connetti il pulsante alla funzione
                self.widgets_rigaFinale[key]["widget"] = widget
            elif props["type"] == "QLineEdit":
                widget = QLineEdit()
                label = QLabel(props["label"])
                self.layout.addWidget(
                    label, props["row"] - 1, props["col"], 1, props["col_span"]
                )
                label.setFixedHeight(10)
                self.layout.addWidget(
                    widget, props["row"], props["col"], 1, props["col_span"]
                )
                if props["default"] is not None:
                    widget.setText(props["default"])

            elif props["type"] == "QDoubleSpinBox":
                widget = QDoubleSpinBox()
                label = QLabel(props["label"])
                self.layout.addWidget(
                    label, props["row"] - 1, props["col"], 1, props["col_span"]
                )
                self.layout.addWidget(
                    widget, props["row"], props["col"], 1, props["col_span"]
                )
                if props["default"] is not None:
                    widget.setValue(float(props["default"][0]))
                    widget.setDecimals(float(props["default"][1]))
                    widget.setSingleStep(float(props["default"][2]))
                    widget.setMinimum(float(props["default"][3]))
                    widget.setMaximum(float(props["default"][4]))
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(
                        self, props["linked_fnc"]
                    )  # Recupera la funzione
                    widget.valueChanged.connect(
                        linked_function
                    )  # Connetti il pulsante alla funzione

            elif props["type"] == "QComboBox":
                widget = QComboBox()
                label = QLabel(props["label"])
                self.layout.addWidget(label, props["row"] - 1, props["col"], 1, props["col_span"])
                self.layout.addWidget(widget, props["row"], props["col"], 1, props["col_span"])
                if props["default"]:
                    widget.addItems(props["default"])
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(self, props["linked_fnc"])
                    widget.currentTextChanged.connect(linked_function)
                self.widgets_rigaFinale[key]["widget"] = widget

            else:
                continue
            if widget:
                self.widgets_rigaFinale[key]["widget"] = widget

        self.layout.setRowStretch(0, 1)  # Etichette
        self.layout.setRowStretch(1, 2)  # Pulsanti e input
        self.layout.setRowStretch(2, 5)  # Grafico Matplotlib
        self.layout.setRowStretch(3, 1)  # Slider
        self.layout.setRowStretch(4, 0)  # Etichette
        self.layout.setRowStretch(5, 1)  # Pulsanti
        for col in range(10):  # Supponiamo 10 colonne
            self.layout.setColumnStretch(
                col, 1
            )  # Tutte le colonne hanno lo stesso peso

        # 🔹 Creazione della figura matplotlib (grafico)
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.figure.subplots_adjust(bottom=0.2)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(
            self.canvas, 2, 0, 1, 10
        )  # Riga 3, occupa tutte le colonne

        # 🔹 Slider per la navigazione nel tempo (Riga 3)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.on_slider)
        self.layout.addWidget(
            self.slider, 3, 0, 1, 10
        )  # Riga 4, occupa tutte le colonne

        # 🔹 Attivazione di SpanSelector per la selezione
        self.selected_region = []
        self.span_selector = SpanSelector(
            self.ax,
            self.on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="red"),
        )

        self.load_wav(self.wav_file)
        self.plot_wav(self.xmin, self.xmax)
    
    def begin_end(self):
        """
        Attiva la modalità di selezione con clic su grafico: salva xmin e xmax.
        """
        
        if self.span_region:
            self.ax.axvspan(self.selected_times[0], self.selected_times[1], color="yellow", alpha=0.3)
            self.canvas.draw_idle()
        else:
            self.span_region = []
        self.selected_times = []
        
        def onclick(event):
            if event.inaxes != self.ax:
                return
            self.span_region = []
            self.plot_wav(self.xmin, self.xmax)
            x_clicked = event.xdata
            print(f"Click rilevato a x = {x_clicked:.3f} s")
            self.selected_times.append(x_clicked)

            # Se abbiamo una coppia (inizio + fine)
            print(self.selected_times)
            if len(self.selected_times) == 2:
                xbegin, xend = sorted(self.selected_times)
                print(self.selected_times)
                print(f"Intervallo selezionato: {xbegin:.3f} - {xend:.3f} s")
                self.span_region = self.ax.axvspan(xbegin, xend, color="yellow", alpha=0.3)
                self.canvas.draw_idle()
                self.span_selector.set_active(False)
                self.span_selector.set_active(True)
                self.canvas.mpl_disconnect(self.cid_click)
            

        # Connetti l'evento alla figura
        self.cid_click = self.canvas.mpl_connect("button_press_event", onclick)

        QMessageBox.information(
            self,
            "Selezione in corso",
            "Clicca due volte sul grafico per selezionare inizio e fine di un canto.\n"
        )



    def load_wav(self, wav_file):
        """
        Load WAV file and extract data
        """

        self.start = 0
        # check if wav was cut
        data_file_path = Path(self.wav_file).parent / Path(
            Path(self.wav_file).parent.name
        ).with_suffix(".json")

        print(f"parameters file: {data_file_path}")

        if data_file_path.is_file():
            # read file content
            with open(data_file_path, "r", encoding="utf-8") as f_in:
                parameters = json.load(f_in)
            if Path(self.wav_file).name in parameters:
                self.start = parameters[Path(self.wav_file).name].get("start", 0)
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"The {Path(self.wav_file).name} was not found in {data_file_path} file.\nCannot load WAV file",
                )
                return

        print(f"start position of file: {self.start}")

        self.amp_rms = 1.5  # fattore di moltiplicazione dell'inviluppo, per accentuare variazione in ampiezza
        self.sampling_rate, self.data = wavfile.read(wav_file)
        self.xmin = 0
        self.duration = len(self.data) / self.sampling_rate
        self.xmax = self.duration
        self.peaks_times = np.array([])
        self.rms_times = np.array([])
        self.rms = np.array([])
        self.peaks = np.array([])

        # Se il file è stereo, usa solo un canale
        if len(self.data.shape) > 1:
            print("File stereo rilevato. Prendo solo il primo canale.")
            self.data = self.data[:, 0]
        # Normalizza il segnale
        self.data = self.data / np.max(np.abs(self.data))
        # Crea variable tempo
        self.time = np.linspace(
            0, len(self.data) / self.sampling_rate, num=len(self.data)
        )
        self.binary_song = np.zeros(len(self.time))
        self.xmin = 0
        self.xmax = self.time[-1]
        self.zmin = 0
        self.zmax = self.time[-1]
        self.id_xmin = 0
        self.id_xmax = len(self.data)
        self.binary_song = np.zeros_like(self.time)

    def plot_wav(self, xmin, xmax):
        self.ax.cla()  # Cancella il grafico precedente
        self.ax.plot(self.time, self.data, linewidth=0.5, color="black", alpha=0.5)
        self.ax.plot(
            self.time, self.binary_song, linewidth=0.5, color="black", alpha=0.5
        )

        if len(self.rms) > 0:
            self.ax.plot(self.rms_times, self.rms, linewidth=1, color="red")

        if len(self.peaks_times) > 0:
            # print("         ", self.peaks_times)
            for i in np.arange(len(self.peaks_times)):
                self.ax.plot(
                    [self.peaks_times[i], self.peaks_times[i]],
                    [0, np.max(self.rms)],
                    "-y",
                    linewidth=2,
                )
        self.ax.set_xlim(self.zmin, self.zmax)
        if self.xmin > 0 or self.xmax < self.duration:
            self.ax.axvspan(self.xmin, self.xmax, color="yellow", alpha=0.3)
        self.canvas.draw()

    def show_message(self, button_name):
        """
        Mostra un messaggio di avviso con il nome del pulsante premuto.
        """
        QMessageBox.warning(
            self,
            "Non Implementato",
            f"La funzione {button_name} non è ancora implementata.",
        )

    def on_select(self, xmin, xmax):
        """
        Evidenzia l'area selezionata con il mouse e aggiorna xmin e xmax.
        """
        if xmin < 0:
            xmin = 0
            xmax = 0
        if abs(xmax - xmin) < 0.01:  # Evita selezioni troppo piccole
            return
        self.xmin = xmin
        self.xmax = xmax
        for patch in self.ax.patches[:]:
            patch.remove()
        self.ax.axvspan(self.xmin, self.xmax, color="yellow", alpha=0.3)
        self.canvas.draw_idle()

    def on_slider(self, value):
        """Aggiorna la vista dell'oscillogramma in base allo slider."""
        range_view = self.duration / 5  # Mostra 1/5 del segnale
        pos = value / 100 * (self.duration - range_view)
        self.ax.set_xlim(pos, pos + range_view)
        self.canvas.draw_idle()

    def zoomIn_wav(self):
        if hasattr(self, "xmin") and hasattr(self, "xmax"):
            self.zmin, self.zmax = self.xmin, self.xmax
            if self.ax.patches:
                for patch in self.ax.patches:
                    patch.remove()
            range = self.xmax - self.xmin
            self.slider.setValue(int((self.xmin / (self.duration - range)) * 100))
            self.ax.set_xlim(self.zmin, self.zmax)
            self.canvas.draw_idle()

    def zoomOut_wav(self):
        self.xmin = 0
        self.xmax = self.duration
        self.zmin, self.zmax = self.xmin, self.xmax
        self.plot_wav(self.xmin, self.xmax)

    def envelope(self, event=None):
        """
        Calcola l'inviluppo RMS usando i parametri aggiornati da Window Size e Overlap.
        """
        try:
            # azzera la eventuale selezione
            self.xmin = 0
            self.xmax = self.duration

            # azzera canti e picchi trovati
            self.binary_song = np.zeros_like(self.time)
            self.peaks_times = np.array([])
            self.peaks = np.array([])

            self.window_size = float(self.widgets_riga1_2["w_size"]["widget"].text())
            self.window_size = int(self.window_size * self.sampling_rate / 1000)
            self.overlap = float(self.widgets_riga1_2["w_overlap"]["widget"].text())
            self.overlap = int(self.overlap * self.sampling_rate / 1000)

            print(f"{self.window_size=}")
            print(f"{self.overlap=}")

            # Verifica che i valori siano validi
            if self.window_size <= 0 or self.overlap < 0:
                print("Errore: Window size deve essere > 0 e Overlap >= 0")
                return

            # Calcola l'inviluppo RMS con i nuovi valori
            self.rms = librosa.feature.rms(
                y=self.data, frame_length=self.window_size, hop_length=self.overlap
            )[0]
            self.rms_times = librosa.frames_to_time(
                np.arange(len(self.rms)), sr=self.sampling_rate, hop_length=self.overlap
            )
            self.rms = self.amp_rms * self.rms
            self.xmin = 0
            self.xmax = self.duration
            self.plot_wav(self.xmin, self.xmax)

        except ValueError:
            print(
                "Errore: Assicurati che Window Size e Overlap siano numeri interi validi."
            )

    def fd_peaks_spinbox(self, value):
        if self.xmin > 0 or self.xmax < self.duration:
            xmin, xmax = self.xmin, self.xmax
        else:
            xmin, xmax = 0, self.duration
        mask_in = (self.peaks_times >= xmin) & (self.peaks_times <= xmax)
        self.peaks_times = self.peaks_times[~mask_in]
        self.fd_peaks()

    def fd_peaks_button(self):
        if self.xmin > 0 or self.xmax < self.duration:
            xmin, xmax = self.xmin, self.xmax
        else:
            xmin, xmax = 0, self.duration
        mask_in = (self.peaks_times >= xmin) & (self.peaks_times <= xmax)
        self.peaks_times = self.peaks_times[~mask_in]
        if np.any(mask_in):  # Se ci sono picchi nell'intervallo, li rimuove
            self.plot_wav(self.zmin, self.zmax)
        else:
            self.fd_peaks()

    def fd_peaks(self):
        """
        Find (f) or delete (d) peaks
        """
        if self.xmin > 0 or self.xmax < self.duration:
            xmin, xmax = self.xmin, self.xmax
        else:
            xmin, xmax = 0, self.duration
        mask_in = (self.peaks_times >= xmin) & (self.peaks_times <= xmax)

        self.amp_threshold = self.widgets_riga1_2["min_amp"]["widget"].value()
        self.min_distance_sec = np.float64(
            self.widgets_riga1_2["min_dist"]["widget"].text()
        )
        self.min_distance_samples = int(
            self.min_distance_sec * (self.sampling_rate / self.overlap)
        )  # Converti in campioni

        mask_rms = (self.rms_times >= xmin) & (self.rms_times <= xmax)
        rms_selected = self.rms[mask_rms]
        rms_times_selected = self.rms_times[mask_rms]

        # Trova i picchi nell'inviluppo RMS
        peaks, _ = find_peaks(
            rms_selected,
            height=self.amp_threshold,
            distance=self.min_distance_samples,
            prominence=0.01,
        )
        new_peaks_times = rms_times_selected[peaks]
        self.peaks_times = np.sort(np.concatenate((self.peaks_times, new_peaks_times)))
        self.plot_wav(self.zmin, self.zmax)

    def plot_spectrum(self):
        """
        Calcola e visualizza lo spettro di potenza del segmento selezionato.
        """
        print(self.fft_length)
        try:
            if (
                self.fft_length <= 0
                or self.fft_overlap < 0
                or self.fft_overlap >= self.fft_length
                ):
                print("Errore: Parametri FFT non validi.")
                return
            self.id_xmin = int(self.xmin * self.sampling_rate)
            self.id_xmax = int(self.xmax * self.sampling_rate)
            segment = self.data[self.id_xmin : self.id_xmax]
            if len(segment) == 0:
                print("Segmento vuoto nella selezione.")
                return
            # Suddivide il segmento in finestre con sovrapposizione e calcola la FFT per ciascuna finestra
            step = self.fft_length - self.fft_overlap
            n_segments = (len(segment) - self.fft_overlap) // step
            print(f"numero di segmenti: {n_segments}; xmin: {self.xmin}; xmax: {self.xmax}")
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

                self.spectrum_peaks, properties = find_peaks(
                    avg_power_db, height=-5, distance=1000
                )
                self.spectrum_peaks_Hz = freqs[self.spectrum_peaks]
                spectrum_peaks_db = avg_power_db[self.spectrum_peaks]

                # print(f"spectrum_peaks: {self.spectrum_peaks_Hz}")

                self.results_dict["spectrum"] = np.concatenate(
                    ([freqs], [power]), axis=0
                ).tolist()

                self.results_dict["spectrum_peaks"] = np.concatenate(
                    ([self.spectrum_peaks_Hz], [spectrum_peaks_db]), axis=0
                ).tolist()

                max_freq = np.float64(
                            self.widgets_rigaFinale["Max_freq"]["widget"].text()
                            )

                print("ho calcolato lo spettro")
                spectrum_fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(freqs, avg_power_db, label="Power Spectrum", color="blue")
                ax.plot(self.spectrum_peaks_Hz, spectrum_peaks_db, "or", label=f"Peak at {self.spectrum_peaks_Hz}")
                ax.set_title("Power Spectrum")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Log Power (dB)")
                ax.set_xlim(0,max_freq)
                ax.legend()

                canvas = FigureCanvas(spectrum_fig)
                self.spectrum_window = QWidget()
                self.spectrum_window.setWindowTitle("Power Spectrum")
                layout = QVBoxLayout()
                layout.addWidget(canvas)
                self.spectrum_window.setLayout(layout)
                self.spectrum_window.resize(800, 400)
                self.spectrum_window.show()
        except Exception as e:
            print("Errore in plot_spectrum:", e)

    def fft_changed(self, value):
        self.fft_length = int(value)
        print(f"FFT length set to: {self.fft_length}")
        # eventualmente aggiorna il grafico o altri parametri
    
    def overlap_changed(self, value):
        self.fft_overlap = int(int(value)/100 * self.fft_length)
        print(f"FFT overlap set to: {self.fft_overlap}")
        # eventualmente aggiorna il grafico o altri parametri

    import pickle  # Assicurati che sia in cima al file

    def save_results(self):
        """
        Salva i risultati delle analisi nel file .json
        """
        # Path del file JSON di output
        json_path = Path(self.wav_file).with_suffix(".json")

        # Crea dizionario con i risultati
        self.results_dict["file"] = Path(self.wav_file).stem
        self.results_dict["sampling_rate"] = self.sampling_rate
        self.results_dict["call_duration"] = self.selected_times[1] - self.selected_times[0]
        self.results_dict["pulse_number"] = len(self.peaks_times)
        self.results_dict["peaks_times"] = self.peaks_times.tolist()
        self.results_dict["spectrum"] = self.results_dict.get("spectrum", [])
        self.results_dict["spectrum_peaks"] = self.results_dict.get("spectrum_peaks", [])

        # Carica contenuto esistente se il file c'è
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f_in:
                try:
                    parameters = json.load(f_in)
                except json.JSONDecodeError:
                    parameters = {}
        else:
            parameters = {}

        # Inserisce i dati sotto la chiave del nome file
        parameters[Path(self.wav_file).name] = self.results_dict

        # Scrive su file
        try:
            with open(json_path, "w", encoding="utf-8") as f_out:
                json.dump(parameters, f_out, indent=2, ensure_ascii=False)
            print(f"Risultati salvati in {json_path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile salvare il file: {e}")
            return




    def save_parameters(self, file_path):
        """
        save parameters in json file
        """

        if file_path.is_file():
            # read file content
            with open(file_path, "r", encoding="utf-8") as f_in:
                parameters = json.load(f_in)

            # add parameters for current file
            if Path(self.wav_file).name not in parameters:
                parameters[Path(self.wav_file).name] = {}

            parameters[Path(self.wav_file).name]["window size"] = self.window_size
            parameters[Path(self.wav_file).name]["overlap"] = self.overlap
            parameters[Path(self.wav_file).name]["amplitude threshold"] = (
                self.amp_threshold
            )
            parameters[Path(self.wav_file).name]["minimum distance"] = (
                self.min_distance_sec
            )
            parameters[Path(self.wav_file).name]["peaks_times"] = (
                self.peaks_times.tolist()
            )

        else:
            # parameters for current file
            parameters = {
                Path(self.wav_file).name: {
                    "window size": self.window_size,
                    "overlap": self.overlap,
                    "amplitude threshold": self.amp_threshold,
                    "minimum distance": self.min_distance_sec,
                    "peaks_times": self.peaks_times.tolist(),
                }
            }
        # save file
        try:
            with open(file_path, "w", encoding="utf-8") as f_out:
                json.dump(parameters, f_out, indent=0, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "", f"The parameters file cannot be saved. {e}")

    def play_audio(self):
        """
        play selection
        """

        try:
            if sd.get_stream().active:
                print("stop playing")
                sd.stop()
            else:
                print(f"Play audio from {self.xmin} s to {self.xmax} s")
                segment = self.data[
                    int(self.xmin * self.sampling_rate) : int(
                        self.xmax * self.sampling_rate
                    )
                ]
                sd.play(segment, samplerate=self.sampling_rate)

        except Exception:
            print(f"Play audio from {self.xmin} s to {self.xmax} s")
            segment = self.data[
                int(self.xmin * self.sampling_rate) : int(
                    self.xmax * self.sampling_rate
                )
            ]
            sd.play(segment, samplerate=self.sampling_rate)

    def stopplaying(self):
        sd.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
    else:
        wav_file, _ = QFileDialog.getOpenFileName(None, "Open WAV File", "", "WAV Files (*.wav)")
    
    if wav_file:
        main_widget = Main(wav_file_list=[wav_file])
        main_widget.show()
        sys.exit(app.exec())
    else:
        QMessageBox.critical(None, "Errore", "Nessun file WAV selezionato.")

    sys.exit(app.exec())
