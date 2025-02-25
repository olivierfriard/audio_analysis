import importlib.util
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenuBar,
    QTextEdit,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QSizePolicy,
    QPushButton,
)

from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider, SpanSelector, Button, TextBox
import librosa


class Main(QWidget):
    def __init__(self, wav_file=None):
        super().__init__()

        if not wav_file:
            self.wav_file = "/media/olivier/6780-F4DB/20250125/GeCorn_2025-01-25_09.wav"
        else:
            self.wav_file = wav_file
        print(self.wav_file)
        self.load_wav(self.wav_file)

        self.window_size = 1024
        self.overlap = 512
        self.amp_threshold = 0
        self.min_dist = 0
        self.rms = np.zeros(self.window_size // self.overlap)
        n_frames = np.arange(len(self.rms))
        self.rms_times = librosa.frames_to_time(n_frames, sr=self.sampling_rate, hop_length=self.overlap)
        self.peaks_times = np.array([])

        self.setWindowTitle("Selezione Interattiva")
        self.setGeometry(100, 100, 800, 500)

        # 🔹 Layout principale
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # 🔹 Layout superiore per i parametri (disposizione orizzontale)
        self.params_layout = QHBoxLayout()

        # Crea i parametri con etichetta sopra la casella di testo
        window_size_layout, self.window_size_input = self.create_label_input_pair("Window size", self.window_size)
        window_overlap_layout, self.window_overlap_input = self.create_label_input_pair("Overlap", self.overlap)
        window_amp_threshold_layout, self.amp_threshold_input = self.create_label_input_pair("Amplitude threshold", self.amp_threshold)
        window_min_distance_layout, self.min_distance_input = self.create_label_input_pair("Minimum distance (points)", self.min_dist)

        # Aggiunge i layout orizzontalmente
        self.params_layout.addLayout(window_size_layout)
        self.params_layout.addLayout(window_overlap_layout)
        self.params_layout.addLayout(window_amp_threshold_layout)
        self.params_layout.addLayout(window_min_distance_layout)

        # 🔹 Layout inferiore per l'oscillogramma e i pulsanti di analisi
        self.analysis_layout = QVBoxLayout()

        # Creazione della figura matplotlib
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(bottom=0.2)
        self.canvas = FigureCanvas(self.figure)
        self.analysis_layout.addWidget(self.canvas)

        # Creazione dei pulsanti di analisi
        buttons_layout = QHBoxLayout()

        self.run_envelope = QPushButton("Run Envelope")
        self.run_envelope.setFixedSize(120, 30)
        self.run_envelope.clicked.connect(self.envelope)

        self.find_peaks_btn = QPushButton("Find Peaks")
        self.find_peaks_btn.setFixedSize(120, 30)
        self.find_peaks_btn.setEnabled(False)  # 🔹 Disattivato all'inizio
        self.find_peaks_btn.clicked.connect(self.trova_picchi)

        self.edit_peaks_btn = QPushButton("Edit Peaks")
        self.edit_peaks_btn.setFixedSize(120, 30)
        self.edit_peaks_btn.setEnabled(False)  # 🔹 Disattivato all'inizio
        self.edit_peaks_btn.clicked.connect(self.edita_picchi)

        self.save_calls_btn = QPushButton("Save Calls")
        self.save_calls_btn.setFixedSize(120, 30)
        self.save_calls_btn.setEnabled(False)  # 🔹 Disattivato all'inizio
        self.save_calls_btn.clicked.connect(self.salva_canti)

        # Aggiunge i pulsanti in una riga orizzontale
        buttons_layout.addWidget(self.run_envelope)
        buttons_layout.addWidget(self.find_peaks_btn)
        buttons_layout.addWidget(self.edit_peaks_btn)
        buttons_layout.addWidget(self.save_calls_btn)

        # Aggiunge i pulsanti al layout inferiore
        self.analysis_layout.addLayout(buttons_layout)

        # 🔹 Aggiunge i due layout al layout principale
        self.main_layout.addLayout(self.params_layout)  # Parametri in alto
        self.main_layout.addLayout(self.analysis_layout)  # Oscillogramma e pulsanti in basso

        # Crea lo slider<
        self.slider_ax = self.figure.add_axes([0.2, 0.05, 0.65, 0.03])
        self.slider = Slider(self.slider_ax, "Time", 0, 1, valinit=self.xmax / self.duration)
        self.slider.on_changed(self.on_slider)

        # Attivazione di SpanSelector click left
        self.span_selector = SpanSelector(self.ax, self.zoom, "horizontal", button=1, useblit=True, props=dict(alpha=0.5, facecolor="red"))

        # Attivazione di SpanSelector click right
        self.span_selector2 = SpanSelector(
            self.ax, self.delete_peaks, "horizontal", button=3, useblit=True, props=dict(alpha=0.5, facecolor="green")
        )
        # Attivazione di SpanSelector click middle
        self.span_selector3 = SpanSelector(
            self.ax, self.trova_picchi, "horizontal", button=2, useblit=True, props=dict(alpha=0.5, facecolor="yellow")
        )

        self.canvas.draw_idle()
        self.plot_wav(self.xmin, self.xmax)

    def create_label_input_pair(self, label_text, default_value):
        vbox = QVBoxLayout()  # Layout verticale per allineare etichetta e input
        label = QLabel(label_text)
        input_box = QLineEdit(str(default_value))
        input_box.setFixedSize(100, 30)  # Imposta dimensioni
        vbox.addWidget(label)
        vbox.addWidget(input_box)
        return vbox, input_box

    def load_wav(self, wav_file):
        """Carica il file WAV e ne estrae i dati."""
        self.sampling_rate, self.data = wavfile.read(wav_file)
        self.xmin = 0
        self.duration = len(self.data) / self.sampling_rate
        self.xmax = self.duration
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

    def plot_wav(self, xmin, xmax):
        self.ax.cla()  # Cancella il grafico precedente
        self.xmin = xmin
        self.xmax = xmax

        self.id_xmin = int(self.xmin * self.sampling_rate)
        self.id_xmax = int(self.xmax * self.sampling_rate)

        time = self.time[self.id_xmin : self.id_xmax]
        data = self.data[self.id_xmin : self.id_xmax]

        # seleziono i valori di rms che ricadono nell'intervallo xmin-xmax
        mask_rms = (self.rms_times >= self.xmin) & (self.rms_times <= self.xmax)
        rms_times_selected = self.rms_times[mask_rms]
        rms_selected = self.rms[mask_rms]

        # seleziono i picchi che ricadono nell'intervallo xmin-xmax
        print(len(self.peaks_times))
        if len(self.peaks_times) > 0:
            mask_peaks = (self.peaks_times >= self.xmin) & (self.peaks_times <= self.xmax)
            print(mask_peaks)
            peaks_selected = self.peaks_times[mask_peaks]
        else:
            peaks_selected = np.array([])

        # Disegno l'oscillogramma
        self.ax.plot(time, data, linewidth=0.5, color="black", alpha=0.25)
        self.ax.plot(rms_times_selected, rms_selected, linewidth=1, color="red")
        if len(peaks_selected) > 0:
            for i in np.arange(len(peaks_selected)):
                self.ax.plot([peaks_selected[i], peaks_selected[i]], [0, np.max(rms_selected)], "-g", linewidth=1)
        self.ax.plot()

        # Aggiorna il grafico
        self.canvas.draw()

    def on_slider(self, val):
        """Aggiorna la vista dell'oscillogramma in base alla posizione dello slider mantenendo la durata selezionata."""
        self.plot_wav(0, self.duration)
        self.xmax = max(self.range, val * self.duration)
        self.xmin = self.xmax - self.range
        self.ax.set_xlim(self.xmin, self.xmax)
        self.canvas.draw_idle()

    def delete_peaks(self, xmin, xmax):
        print("NUMERO PICCHI", len(self.peaks_times))
        # trova i picchi tra self.xmin e self.xmax
        """Elimina i picchi compresi tra xmin e xmax dalla lista self.peaks."""
        if len(self.peaks_times) == 0:
            return
        # Converti gli indici dei picchi in tempi reali
        num_peaks_before = len(self.peaks_times)
        # Filtra i picchi che NON sono compresi tra xmin e xmax
        mask = (self.peaks_times < xmin) | (self.peaks_times > xmax)
        self.peaks_times = self.peaks_times[mask]  # Mantiene solo i picchi fuori dall'intervallo
        num_peaks_after = len(self.peaks_times)
        # Aggiorna il grafico per riflettere i cambiamenti
        print("HO ELIMINATO I PICCHI", num_peaks_before - num_peaks_after)
        self.plot_wav(self.xmin, self.xmax)

    def zoom(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

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

        self.ax.set_xlim(self.xmin, self.xmax)
        self.range = self.xmax - self.xmin
        self.slider.set_val(self.xmax / self.duration)

    def envelope(self, event=None):
        """Calcola l'inviluppo RMS usando i parametri aggiornati da Window Size e Overlap."""
        try:
            # Leggi i valori dalle caselle di testo
            self.window_size = int(self.window_size_input.text())
            self.overlap = int(self.window_overlap_input.text())
            print(self.window_size, self.overlap)
            # Verifica che i valori siano validi
            if self.window_size <= 0 or self.overlap < 0:
                print("Errore: Window size deve essere > 0 e Overlap >= 0")
                return

            # Calcola l'inviluppo RMS con i nuovi valori
            self.rms = librosa.feature.rms(y=self.data, frame_length=self.window_size, hop_length=self.overlap)[0]
            self.rms_times = librosa.frames_to_time(np.arange(len(self.rms)), sr=self.sampling_rate, hop_length=self.overlap)
            print("----------------", len(self.rms))
            self.find_peaks_btn.setEnabled(True)  # 🔹 Attiva il pulsante
            self.save_calls_btn.setEnabled(True)  # 🔹 Attiva il pulsante
            self.plot_wav(self.xmin, self.xmax)

        except ValueError:
            print("Errore: Assicurati che Window Size e Overlap siano numeri interi validi.")

    def trova_picchi(self, xmin=0, xmax=0):
        """Trova i picchi dell'inviluppo RMS e li converte nei campioni della registrazione originale."""

        print(self.sender())
        print(f"{xmin=}")
        print(f"{xmax=}")

        if self.sender() is not None:
            rms = self.rms
            self.peaks_times = np.array([])
        else:
            rms = self.rms[int(xmin * self.sampling_rate / self.overlap) : int(xmax * self.sampling_rate / self.overlap)]
            self.delete_peaks(xmin, xmax)

        print(f"{rms=}")

        try:
            min_distance_sec = float(self.min_distance_input.text())  # Distanza in secondi
            print(min_distance_sec)
            min_distance_samples = int(min_distance_sec * (self.sampling_rate / self.overlap))  # Converti in campioni
            print(min_distance_samples, self.sampling_rate, self.overlap)
            amp_threshold = float(self.amp_threshold_input.text())  # Soglia di ampiezza

            print(f" Cercando picchi con:")
            print(f"   - Soglia di ampiezza: {amp_threshold:.5f}")
            print(f"   - Distanza minima tra picchi: {min_distance_sec:.5f} sec ({min_distance_samples} campioni)")

            # Trova i picchi nell'inviluppo RMS
            peaks, properties = find_peaks(rms, height=amp_threshold, distance=min_distance_samples, prominence=0.01)

            # Converti gli indici nei campioni effettivi dell'audio originale
            peaks_original = peaks * self.overlap  # Campioni effettivi

            print(f"{peaks=}")

            peaks_times = peaks * self.overlap / self.sampling_rate + xmin
            print(f"{peaks_times=}")

            # self.peaks_times = peaks * self.overlap / self.sampling_rate  # In secondi
            self.peaks_times = np.sort(np.concatenate((self.peaks_times, peaks_times)))

            print(f" {len(peaks)} picchi trovati")
            print(f"   - Indici nell'inviluppo: {peaks}")
            print(f"   - Campioni originali: {peaks_original}")
            print(f"   - Posizioni in secondi: {self.peaks_times}")

            self.edit_peaks_btn.setEnabled(True)  # 🔹 Attiva il pulsante

            self.plot_wav(self.xmin, self.xmax)

        except ValueError:
            print(" Errore: Inserisci valori numerici validi per la distanza e la soglia.")

    def edita_picchi(self):
        print("NUMERO PICCHI", len(self.peaks_times))
        # trova i picchi tra self.xmin e self.xmax
        """Elimina i picchi compresi tra xmin e xmax dalla lista self.peaks."""
        if len(self.peaks_times) == 0:
            print("in costruzione")
            return
        # Converti gli indici dei picchi in tempi reali
        num_peaks_before = len(self.peaks_times)
        # Filtra i picchi che NON sono compresi tra xmin e xmax
        mask = (self.peaks_times < self.xmin) | (self.peaks_times > self.xmax)
        self.peaks_times = self.peaks_times[mask]  # Mantiene solo i picchi fuori dall'intervallo
        num_peaks_after = len(self.peaks_times)
        # Aggiorna il grafico per riflettere i cambiamenti
        print("HO ELIMINATO I PICCHI", num_peaks_before - num_peaks_after)
        self.plot_wav(self.xmin, self.xmax)

    def salva_canti(self):
        """Salva i segmenti audio attorno ai picchi selezionati in file separati."""

        # 🔹 Seleziona i picchi all'interno dell'intervallo xmin - xmax
        mask = (self.peaks_times > self.xmin) & (self.peaks_times < self.xmax)
        peaks_selected = self.peaks_times[mask]

        if len(peaks_selected) != 1:
            print("Seleziona un solo picco nella finestra!")
            return

        # 🔹 Calcola i margini prima e dopo il picco selezionato
        peak_time = peaks_selected[0]
        before = peak_time - self.xmin
        after = self.xmax - peak_time

        # 🔹 Crea una cartella con il nome del file di origine (senza estensione)
        original_filename = Path(self.wav_file).stem  # Nome del file senza estensione
        save_folder = Path(original_filename)
        save_folder.mkdir(exist_ok=True)  # Crea la cartella se non esiste

        print(f"Salvando i canti nella cartella: {save_folder}")

        for i, peak_time in enumerate(self.peaks_times):
            # 🔹 Calcola l'inizio e la fine del ritaglio
            ini = int((peak_time - before) * self.sampling_rate)
            fine = int((peak_time + after) * self.sampling_rate)

            # 🔹 Verifica che l'intervallo sia valido
            if ini < 0 or fine > len(self.data):
                print(f"Il picco {peak_time:.5f}s supera i limiti del file audio! Saltato.")
                continue

            ritaglio = self.data[ini:fine]

            # Crea il nome del file con il numero di campione
            sample_number = int(peak_time * self.sampling_rate)
            nome_ritaglio = save_folder / f"{original_filename}_sample{sample_number}.wav"

            # Salva il file ritagliato
            wavfile.write(nome_ritaglio, self.sampling_rate, ritaglio)
            print(f"Salvato: {nome_ritaglio}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_widget = Main(wav_file="")
    main_widget.show()

    sys.exit(app.exec())
