import importlib.util
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QTextEdit, QFileDialog, QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider, SpanSelector, Button, TextBox
import librosa


class Main(QWidget):
    def __init__(self, wav_file=None):
        super().__init__()

        self.wav_file = wav_file

        self.setWindowTitle("Selezione Interattiva")
        self.setGeometry(100, 100, 800, 500)
        self.window_size = 1024
        self.overlap = 512

        # Creazione layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Creazione della figura matplotlib
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(bottom=0.2)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)  # IMPORTANTE: Aggiungere il canvas alla finestra

        # Dati di test
        self.x = []
        self.y = []
        self.rms = []
        self.rms_times = []
        self.ax.plot(self.x, self.y, color="black")

        # Attivazione di SpanSelector
        self.span_selector = SpanSelector(self.ax, self.on_select, "horizontal", useblit=True, props=dict(alpha=0.5, facecolor="red"))

        self.canvas.draw_idle()

        if self.wav_file:
            print(f"DEBUG: Plugin ricevuto file WAV: {self.wav_file}")
            self.load_wav(self.wav_file)
        else:
            # Carica il file .wav
            self.wav_file = "Blommersia_blommersae.wav"
            print(self.wav_file)
            self.load_wav(self.wav_file)

        # Creazione delle caselle di testo
        axbox1 = self.figure.add_axes([0.1, 0.01, 0.10, 0.05])
        self.textbox_window_size = TextBox(axbox1, "Finestra", initial=str(self.window_size))

        axbox3 = self.figure.add_axes([0.45, 0.01, 0.10, 0.05])
        self.textbox_overlap = TextBox(axbox3, "Overlap", initial=str(self.overlap))

    def on_select(self, xmin, xmax):
        print(xmin, xmax)
        if self.xmax - self.xmin < 0.01:
            self.xmin = 0
            self.id_xmin = 0
            self.xmax = len(self.data) / self.sampling_rate
            self.id_xmax = len(self.data)
        else:
            self.xmin = xmin
            self.id_xmin = int(self.xmin * self.sampling_rate)
            self.xmax = xmax
            self.id_xmax = int(self.xmax * self.sampling_rate)
        self.envelope()

    def load_wav(self, wav_file):
        """Carica il file WAV e ne estrae i dati."""
        self.sampling_rate, self.data = wavfile.read(wav_file)
        self.xmin = 0
        self.xmax = len(self.data) / self.sampling_rate
        # Se il file è stereo, usa solo un canale
        if len(self.data.shape) > 1:
            print("File stereo rilevato. Prendo solo il primo canale.")
            self.data = self.data[:, 0]
        # Normalizza il segnale
        self.data = self.data / np.max(np.abs(self.data))
        # Crea variable tempo
        self.time = np.linspace(0, len(self.data) / self.sampling_rate, num=len(self.data))
        self.xmin = 0
        self.xmax = self.time[-1]
        self.id_xmin = 0
        self.id_xmax = len(self.data)

        self.plot_wav()

    def plot_wav(self):
        # Plot dell'Oscillogramma
        print(self.id_xmin, self.id_xmax)
        time = self.time[self.id_xmin : self.id_xmax]
        data = self.data[self.id_xmin : self.id_xmax]
        self.ax.cla()
        self.ax.plot(time, data, linewidth=0.5, color="black")
        self.ax.set_ylabel("Ampiezza")
        self.ax.set_xlabel("Tempo (s)")
        # self.ax.set_title(self.wav_file)
        if len(self.rms) > 0:
            self.ax.plot(self.rms_times, self.rms, linewidth=0.5, color="red")

        self.canvas.draw()  # Ridisegna il grafico

    def envelope(self):
        # calcolo inviluppo con media mobile
        frame_length = 1024
        hop_length = 512
        self.rms = librosa.feature.rms(y=self.data[self.id_xmin : self.id_xmax], frame_length=frame_length, hop_length=hop_length)[0]
        self.rms_times = self.xmin + librosa.frames_to_time(np.arange(len(self.rms)), sr=self.sampling_rate, hop_length=hop_length)
        print(self.id_xmin, self.id_xmax)
        self.plot_wav()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_widget = Main(wav_file="")
    main_widget.show()

    sys.exit(app.exec())
