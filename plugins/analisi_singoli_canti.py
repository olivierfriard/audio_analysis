import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider, SpanSelector
import librosa
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton


# Finestra che ospita i grafici (plot dell'oscillogramma e dello spettro)
class Main(QWidget):
    def __init__(self, wav_file):
        super().__init__()
        self.wav_file = wav_file

        print("Carico il file:", self.wav_file)
        self.load_wav(self.wav_file)

        self.window_size = 50
        self.overlap = 50
        self.min_amplitude = 0.1
        self.min_distance = 0.003
        self.canto = np.zeros(len(self.data))
        self.rms = np.zeros(len(self.data) // self.overlap)
        n_frames = np.arange(len(self.rms))
        self.rms_times = librosa.frames_to_time(n_frames, sr=self.sampling_rate, hop_length=self.overlap)
        self.peaks_times = np.array([])

        # Crea la figura con 2 subplot affiancati
        self.figure, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plt.subplots_adjust(bottom=0.25)

        self.canvas = FigureCanvas(self.figure)

        # Layout principale
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        save_results_layout = QVBoxLayout()
        self.save_results_btn = QPushButton("Save results")
        save_results_layout.addWidget(self.save_results_btn)
        self.save_results_btn.clicked.connect(self.save_results_clicked)

        # Aggiunge il layout del pulsante al layout principale
        layout.addLayout(save_results_layout)
        self.setLayout(layout)

        self.range = self.xmax - self.xmin
        self.canvas.draw_idle()
        self.plot_wav(self.xmin, self.xmax)

        # load control panel and show it
        self.control_panel = ControlPanel(self)
        self.control_panel.show()

    def load_wav(self, wav_file):
        """Carica il file WAV e ne estrae i dati."""
        self.sampling_rate, self.data = wavfile.read(wav_file)
        self.xmin = 0
        self.duration = len(self.data) / self.sampling_rate
        self.xmax = self.duration
        if len(self.data.shape) > 1:
            print("File stereo rilevato. Uso il primo canale.")
            self.data = self.data[:, 0]
        self.data = self.data / np.max(np.abs(self.data))
        self.time = np.linspace(0, len(self.data) / self.sampling_rate, num=len(self.data))
        self.id_xmin = 0
        self.id_xmax = len(self.data)

    def plot_wav(self, xmin, xmax):
        """Aggiorna l'oscillogramma (sinistro) con il segnale e l'envelope."""
        self.ax.cla()
        self.xmin = xmin
        self.xmax = xmax
        self.id_xmin = int(self.xmin * self.sampling_rate)
        self.id_xmax = int(self.xmax * self.sampling_rate)
        time_segment = self.time[self.id_xmin : self.id_xmax]
        data_segment = self.data[self.id_xmin : self.id_xmax]
        canto_segment = self.canto[self.id_xmin : self.id_xmax]
        print("xmin", self.xmin, "xmax", self.xmax)
        self.ax.plot(time_segment, data_segment, linewidth=0.5, color="black", alpha=0.25)
        self.ax.plot(time_segment, canto_segment, "-", color="blue")
        mask_rms = (self.rms_times >= self.xmin) & (self.rms_times <= self.xmax)
        rms_times_sel = self.rms_times[mask_rms]
        rms_sel = self.rms[mask_rms]
        if len(rms_sel) > 0:
            self.ax.plot(rms_times_sel, rms_sel, linewidth=1, color="red")

        # seleziono i picchi che ricadono nell'intervallo xmin-xmax
        print(len(self.peaks_times))
        if len(self.peaks_times) > 0:
            mask_peaks = (self.peaks_times >= self.xmin) & (self.peaks_times <= self.xmax)
            print(mask_peaks)
            peaks_selected = self.peaks_times[mask_peaks]
        else:
            peaks_selected = np.array([])

        if len(peaks_selected) > 0:
            for i in np.arange(len(peaks_selected)):
                self.ax.plot([peaks_selected[i], peaks_selected[i]], [0, 1], "-g", linewidth=1)

        self.ax.plot()
        self.ax.plot(rms_times_sel, rms_sel, linewidth=1, color="red")
        self.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        """Aggiorna il range in base alla selezione (SpanSelector)."""
        if xmax - xmin < 0.01:
            self.xmin = 0
            self.xmax = self.duration
        else:
            self.xmin, self.xmax = xmin, xmax
        if self.xmax - self.xmin < 0.01:
            self.xmin = 0
            self.xmax = self.duration
        self.ax.set_xlim(self.xmin, self.xmax)
        self.range = self.xmax - self.xmin
        self.slider.set_val(self.xmax / self.duration)

    def envelope(self):
        """Calcola l'envelope (RMS) usando i parametri correnti e aggiorna l'oscillogramma."""
        try:
            if self.window_size <= 0 or self.overlap < 0:
                print("Errore: Window size deve essere > 0 e Overlap >= 0")
                return
            self.rms = librosa.feature.rms(y=self.data, frame_length=self.window_size, hop_length=self.overlap)[0]
            self.rms_times = librosa.frames_to_time(np.arange(len(self.rms)), sr=self.sampling_rate, hop_length=self.overlap)
            print("Envelope calcolato, lunghezza:", len(self.rms))
            self.plot_wav(self.xmin, self.xmax)
        except Exception as e:
            print("Errore in envelope:", e)

    def plot_spectrum(self, fft_length, fft_overlap):
        """Calcola e visualizza lo spettro di potenza del segmento selezionato."""
        try:
            if fft_length <= 0 or fft_overlap < 0 or fft_overlap >= fft_length:
                print("Errore: Parametri FFT non validi.")
                return
            self.id_xmin = int(self.xmin * self.sampling_rate)
            self.id_xmax = int(self.xmax * self.sampling_rate)
            segment = self.data[self.id_xmin : self.id_xmax]
            if len(segment) == 0:
                print("Segmento vuoto nella selezione.")
                return
            # Suddivide il segmento in finestre con sovrapposizione e calcola la FFT per ciascuna finestra
            step = fft_length - fft_overlap
            n_segments = (len(segment) - fft_overlap) // step
            if n_segments <= 0:
                padded = np.zeros(fft_length)
                padded[: len(segment)] = segment
                fft_vals = np.fft.fft(padded)
                power = np.abs(fft_vals) ** 2
            else:
                spectra = []
                for i in range(n_segments):
                    start = i * step
                    end = start + fft_length
                    if end > len(segment):
                        break
                    windowed = segment[start:end] * np.hamming(fft_length)
                    fft_vals = np.fft.fft(windowed, n=fft_length)
                    power = np.abs(fft_vals) ** 2
                    spectra.append(power)
                power = np.mean(np.array(spectra), axis=0)
            freqs = np.fft.fftfreq(fft_length, d=1 / self.sampling_rate)
            mask_positive = freqs >= 0
            freqs = freqs[mask_positive]
            power = power[mask_positive]
            avg_power_db = np.log10(power + 1e-10)
            self.ax2.cla()
            self.ax2.plot(freqs, avg_power_db, color="blue")
            self.ax2.set_title("Power Spectrum")
            self.ax2.set_xlabel("Frequency (Hz)")
            self.ax2.set_ylabel("Power")
            self.canvas.draw_idle()
        except Exception as e:
            print("Errore in plot_spectrum:", e)

    def trova_picchi(self, min_amplitude, min_distance):
        """Trova i picchi dell'inviluppo RMS e li converte nei campioni della registrazione originale."""
        try:
            # min_distance = float(self.min_distance_input.text())  # Distanza in secondi
            min_distance_samples = int(min_distance * (self.sampling_rate / self.overlap))  # Converti in campioni

            # min_amplitude = float(self.amp_threshold_input.text())  # Soglia di ampiezza

            # Trova i picchi nell'inviluppo RMS
            peaks, properties = find_peaks(self.rms, height=min_amplitude, distance=min_distance_samples, prominence=0.01)

            # Converti gli indici nei campioni effettivi dell'audio originale
            peaks_original = peaks * self.overlap  # Campioni effettivi
            self.peaks_times = peaks * self.overlap / self.sampling_rate  # In secondi

            # self.plot_wav(self.xmin, self.xmax)
            self.trova_ini_fin()
        except ValueError:
            print(" Errore: Inserisci valori numerici validi per la distanza e la soglia.")

    def trova_ini_fin(self):
        # trova inizio
        peaks = self.peaks_times * self.sampling_rate / self.overlap
        diff_ini = np.concatenate(([-1], np.diff(self.rms[: int(peaks[0])])))
        ini = np.where(diff_ini < 0)[0]
        if np.size(ini) > 0:
            inizio = int(ini[-1] * self.overlap)
        else:
            inizio = 0
        # trova fine
        diff_fin = np.concatenate((np.diff(self.rms[int(peaks[-1]) :]), [1]))
        fin = np.where(diff_fin > 0)[0]

        if np.size(fin) > 0:
            fine = int((peaks[-1] + fin[0]) * self.overlap)
        else:
            fine = len(self.rms) * self.overlap

        self.canto = np.zeros(len(self.rms) * self.overlap)
        self.canto[inizio:fine] = np.max(self.rms)
        self.plot_wav(self.xmin, self.xmax)

    def save_results_clicked(self):
        print("salvo")


class ControlPanel(QWidget):
    def __init__(self, plot_panel):
        super().__init__()
        self.plot_panel = plot_panel
        self.setWindowTitle("Control Panel")
        self.setGeometry(1100, 100, 300, 400)  # Posiziona la finestra dei controlli separata

        # Layout per i parametri dell'envelope
        envelope_layout = QVBoxLayout()
        envelope_layout.addWidget(QLabel("Envelope Parameters"))
        self.window_size_input = QLineEdit("50")
        self.overlap_input = QLineEdit("50")
        envelope_layout.addWidget(QLabel("Window size"))
        envelope_layout.addWidget(self.window_size_input)
        envelope_layout.addWidget(QLabel("Overlap"))
        envelope_layout.addWidget(self.overlap_input)
        self.envelope_btn = QPushButton("Compute Envelope")
        envelope_layout.addWidget(self.envelope_btn)
        self.envelope_btn.clicked.connect(self.envelope_clicked)

        # Layout per i parametri del peak finder
        peak_finder_layout = QVBoxLayout()
        peak_finder_layout.addWidget(QLabel("Peak Finder Parameters"))
        self.amp_threshold_input = QLineEdit("0.1")
        self.min_distance_input = QLineEdit("0.003")
        peak_finder_layout.addWidget(QLabel("Amplitude Threshold"))
        peak_finder_layout.addWidget(self.amp_threshold_input)
        peak_finder_layout.addWidget(QLabel("Minimum Distance"))
        peak_finder_layout.addWidget(self.min_distance_input)
        self.peaks_btn = QPushButton("Find Peaks")
        peak_finder_layout.addWidget(self.peaks_btn)
        self.peaks_btn.clicked.connect(self.peaks_clicked)

        # Layout per i parametri dello spettro
        spectrum_layout = QVBoxLayout()
        spectrum_layout.addWidget(QLabel("Spectrum Parameters"))
        self.fft_length_input = QLineEdit("1024")
        self.fft_overlap_input = QLineEdit("512")
        spectrum_layout.addWidget(QLabel("FFT Length"))
        spectrum_layout.addWidget(self.fft_length_input)
        spectrum_layout.addWidget(QLabel("Overlap"))
        spectrum_layout.addWidget(self.fft_overlap_input)
        self.spectrum_btn = QPushButton("Compute Spectrum")
        spectrum_layout.addWidget(self.spectrum_btn)
        self.spectrum_btn.clicked.connect(self.spectrum_clicked)

        # Layout principale
        main_layout = QVBoxLayout()
        main_layout.addLayout(envelope_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(peak_finder_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(spectrum_layout)
        self.setLayout(main_layout)

    def envelope_clicked(self):
        try:
            ws = int(self.window_size_input.text())
            ov = int(self.overlap_input.text())
            self.plot_panel.window_size = ws
            self.plot_panel.overlap = ov
            self.plot_panel.envelope()
        except Exception as e:
            print("Errore nei parametri envelope:", e)

    def peaks_clicked(self):
        try:
            min_amplitude = float(self.amp_threshold_input.text())
            min_distance = float(self.min_distance_input.text())
            self.plot_panel.amp_threshold = min_amplitude
            self.plot_panel.min_distance = min_distance
            self.plot_panel.trova_picchi(min_amplitude, min_distance)
        except Exception as e:
            print("Errore nei parametri peak finder:", e)

    def spectrum_clicked(self):
        try:
            fft_len = int(self.fft_length_input.text())
            fft_ov = int(self.fft_overlap_input.text())
            self.plot_panel.plot_spectrum(fft_len, fft_ov)
        except Exception as e:
            print("Errore nei parametri spectrum:", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Crea la finestra dei plots e quella dei controlli
    plot_panel = Main(wav_file="GeCorn_2025-01-25_09/GeCorn_2025-01-25_09_sample17408.wav")
    plot_panel.show()
    sys.exit(app.exec())
