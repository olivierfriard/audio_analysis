from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector
import numpy as np
import sounddevice as sd  # sudo apt install libportaudio2

from PySide6.QtWidgets import (
    QWidget,
    QGridLayout,
    QSlider,
    QPushButton,
    QMessageBox,
    QInputDialog,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut

from .wav_cutting import Wav_cutting


"""
class AmplifyDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Amplify Signal")

        layout = QVBoxLayout()

        self.label = QLabel("Enter amplification factor:")
        layout.addWidget(self.label)

        self.textbox = QTextEdit()
        self.textbox.setFixedHeight(30)
        layout.addWidget(self.textbox)

        self.button = QPushButton("Apply")
        self.button.clicked.connect(self.apply_amplification)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def apply_amplification(self):
        try:
            factor = float(self.textbox.toPlainText())
            self.parent().apply_amplification(factor)
            self.accept()  # Chiude la finestra in modo corretto
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid amplification factor")
"""


class OscillogramWindow(QWidget):
    load_wav_signal = Signal(list)

    def __init__(self, wav_file: str):
        super().__init__()

        self.wav_file = wav_file

        self.mem_data = None
        self.mem_amplif_factor = None

        self.setWindowTitle(f"Oscillogram for {Path(wav_file).stem}")
        # self.setGeometry(200, 200, 800, 500)

        self.sampling_rate, self.data = wavfile.read(wav_file)
        self.duration = len(self.data) / self.sampling_rate

        self.time = np.linspace(0, self.duration, num=len(self.data))

        # Create a shortcut for undo amplification: Ctrl+Z
        shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        shortcut.activated.connect(self.undo_amplification)

        # Create a shortcut for redoing amplification: Ctrl+R
        shortcut2 = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut2.activated.connect(self.redo_amplification)

        # Layout principale a griglia
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Creazione della figura matplotlib (plot)
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)

        # Slider Qt
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)  # Inizialmente a 0
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.on_slider)
        self.layout.addWidget(self.slider, 2, 1)

        # double click to reset xmin e xmax
        self.canvas.mpl_connect("button_press_event", self.on_double_click)

        # Pulsante zoomIn
        self.zoomIn_button = QPushButton("Zoom In")
        self.zoomIn_button.setEnabled(False)
        self.zoomIn_button.clicked.connect(self.zoomIn_wav)

        # Pulsante zoomOut
        self.zoomOut_button = QPushButton("Zoom Out")
        self.zoomOut_button.setEnabled(False)
        self.zoomOut_button.clicked.connect(self.zoomOut_wav)

        # Pulsante "Amplify"
        self.amplify_button = QPushButton("Amplify")
        self.amplify_button.clicked.connect(self.open_amplify_dialog)

        # Pulsante "Riproduci"
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_dialog)

        """
        # Pulsante STOP
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stopplaying)
        """

        # cut and save
        self.cut_save_button = QPushButton("Cut and save")
        self.cut_save_button.clicked.connect(self.cut_save)

        # Disegna il grafico iniziale
        self.ax.plot(self.time, self.data, linewidth=0.5, color="black")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title(f"Oscillogram for {Path(wav_file).stem}")
        # self.ax.grid()

        # Imposta il range iniziale della finestra
        self.xmin = 0
        self.xmax = self.duration  # Inizialmente mostra tutta la registrazione
        self.xrange = self.xmax - self.xmin
        self.ax.set_xlim(self.xmin, self.xmax)

        self.plot_xmin = self.xmin
        self.plot_xmax = self.xmax

        # Aggiunta della selezione interattiva
        self.selected_region = None
        self.span_selector = SpanSelector(
            self.ax,
            self.on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="red"),
        )

        # **Organizzazione della griglia**
        self.layout.addWidget(self.zoomIn_button, 0, 0, 1, 1)
        self.layout.addWidget(self.zoomOut_button, 0, 1, 1, 1)
        self.layout.addWidget(self.play_button, 0, 2, 1, 1)
        # self.layout.addWidget(self.stop_button, 0, 3, 1, 1)
        self.layout.addWidget(self.amplify_button, 0, 4, 1, 1)
        self.layout.addWidget(self.cut_save_button, 0, 5, 1, 1)
        self.layout.addWidget(self.canvas, 1, 0, 1, 6)
        self.layout.addWidget(self.slider, 2, 0, 1, 6)

        # Configurazione del comportamento delle colonne
        for i in range(5 + 1):
            self.layout.setColumnStretch(i, 1)

        # Configurazione del comportamento delle righe
        self.layout.setRowStretch(0, 1)  # Pulsanti (meno spazio)
        self.layout.setRowStretch(1, 5)  # Plot (più spazio)
        self.layout.setRowStretch(2, 1)  # Slider (meno spazio)

        self.canvas.draw()

    def undo_amplification(self):
        if self.mem_data is not None:
            # restore previous data
            self.data = self.mem_data.copy()

            self.mem_data = None

            """self.ax.clear()
            self.ax.plot(self.time, self.data, linewidth=0.5, color="black")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")

            self.ax.set_xlim(self.plot_xmin, self.plot_xmax)

            self.canvas.draw()"""

            self.apply_amplification(1)  # used to update plot
            print("undo")
        else:
            print("undo not possible")

    def redo_amplification(self):
        """
        redo amplif with memorized amplification factor
        """
        if self.mem_amplif_factor is None:
            return
        self.apply_amplification(self.mem_amplif_factor)

    def on_select(self, xmin, xmax):
        """
        Evidenzia l'area selezionata con il mouse.
        """
        min_width = 0.01
        if abs(xmax - xmin) < min_width:
            return
        print(f"{self.selected_region=}")
        if self.selected_region:
            self.selected_region.remove()  # Rimuove l'area precedente
        self.selected_region = self.ax.axvspan(xmin, xmax, color="red", alpha=0.3)
        self.xmin = xmin
        self.xmax = xmax
        self.canvas.draw_idle()
        self.zoomIn_button.setEnabled(True)

    def zoomIn_wav(self):
        if hasattr(self, "xmin") and hasattr(self, "xmax"):
            range = self.xmax - self.xmin
            self.slider.setValue(int((self.xmin / (self.duration - range)) * 100))
            self.ax.set_xlim(self.xmin, self.xmax)

            self.plot_xmin = self.xmin
            self.plot_xmax = self.xmax

        # Rimuove la selezione rossa se esiste
        if self.selected_region:
            self.selected_region.remove()
            self.selected_region = None  # Reset della variabile

        self.zoomIn_button.setEnabled(False)
        self.zoomOut_button.setEnabled(True)
        self.canvas.draw_idle()

    def zoomOut_wav(self):
        self.xmin = 0
        self.xmax = self.duration
        self.plot_xmin = self.xmin
        self.plot_xmax = self.xmax

        self.ax.set_xlim(self.xmin, self.xmax)  # Applica il reset
        self.slider.setValue(100)  # Imposta lo slider al massimo (100%)
        self.canvas.draw_idle()  # Aggiorna il grafico
        self.zoomIn_button.setEnabled(False)
        self.zoomOut_button.setEnabled(False)

    def stopplaying(self):
        sd.stop()

    def on_double_click(self, event):
        if event.dblclick:  # Controlla se è un doppio clic
            # Resetta i limiti
            self.xmin = 0
            self.xmax = self.duration
            self.ax.set_xlim(self.xmin, self.xmax)  # Applica il reset
            self.slider.setValue(100)  # Imposta lo slider al massimo (100%)
            self.canvas.draw_idle()  # Aggiorna il grafico

    def on_slider(self, value):
        """
        Aggiorna la vista dell'oscillogramma in base alla posizione dello slider mantenendo la durata selezionata.


        """
        range = self.xmax - self.xmin
        pos = value / 100 * (self.duration - range)

        if pos + range < self.duration:
            self.xmin = pos
            self.xmax = pos + range
        else:
            self.xmin = self.duration - range
            self.xmax = self.duration
            self.slider.setValue(100)

        self.ax.set_xlim(self.xmin, self.xmax)
        self.canvas.draw_idle()

    def open_amplify_dialog(self):
        """
        ask user to select an amplification factor

        """
        value, ok = QInputDialog.getDouble(
            None,
            "Enter Value",
            "Enter a floating point number:",
            value=1,  # default value
            minValue=0.0,  # minimum allowed
            maxValue=1000.0,  # maximum allowed
            decimals=2,  # number of decimals
        )

        if not ok:
            return

        """
        self.amplify_dialog = AmplifyDialog(self)
        self.amplify_dialog.exec()
        """

        self.mem_amplif_factor = value
        self.apply_amplification(value)

    def apply_amplification(self, factor):
        """
        amplify the region between xmin and xmax
        """

        if factor != 1:
            ini = int(self.xmin * self.sampling_rate)
            fin = int(self.xmax * self.sampling_rate)
            # memorize data before amplification
            self.mem_data = self.data.copy()
            segnale = self.data[ini:fin]
            segnale = np.clip(segnale * factor, -32768, 32767).astype(np.int16)
            self.data[ini:fin] = segnale

        self.ax.clear()
        self.ax.plot(self.time, self.data, linewidth=0.5, color="black")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

        self.ax.set_xlim(self.plot_xmin, self.plot_xmax)

        self.canvas.draw()

        print(f"{self.selected_region=}")
        self.selected_region = None

    def play_dialog(self):
        """
        Riproduce il segmento selezionato dell'audio.
        """
        if self.play_button.text() == "Play":
            # self.stop_button.setEnabled(True)
            ini = int(self.xmin * self.sampling_rate)
            fin = int(self.xmax * self.sampling_rate)
            segment = self.data[ini:fin]  # Estrarre il segmento selezionato
            sd.play(segment, samplerate=self.sampling_rate)  # Riprodurre il suono
            self.play_button.setText("Stop")
        else:
            sd.stop()
            self.play_button.setText("Play")

    def cut_save(self):
        """
        cut and save WAV files
        """

        # save data in a wav temporary file (.tmp)
        try:
            wavfile.write(
                Path(self.wav_file).with_suffix(".tmp"), self.sampling_rate, self.data
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"The file {Path(self.wav_file).with_suffix('.tmp')} cannot be saved.\n{e}",
            )

        self.wav_cutting_widget = Wav_cutting(
            str(Path(self.wav_file).with_suffix(".tmp"))
        )
        self.wav_cutting_widget.cut_ended_signal.connect(self.cut_ended)
        self.wav_cutting_widget.show()

    @Slot(list)
    def cut_ended(self, file_list: list):
        """
        receive signal from Wav_cutting
        """
        self.wav_cutting_widget.close()

        self.load_wav_signal.emit(file_list)

        self.close()
