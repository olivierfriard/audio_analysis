import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import librosa
import pickle
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QMessageBox,
    QSpinBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt
from pathlib import Path

WINDOW_SIZE = 50
OVERLAP = 50
MIN_AMPLITUDE = 0.1
MIN_DISTANCE = 0.003
FFT_LENGTH = 1024
FFT_OVERLAP = 512


class Main(QWidget):
    def __init__(self, wav_file):
        super().__init__()

        self.init_values()

        self.wav_file = wav_file

        print("Carico il file:", self.wav_file)
        self.load_wav(self.wav_file)

        # Crea la figura con 2 subplot affiancati
        self.figure, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plt.subplots_adjust(bottom=0.25)

        # Layout principale
        layout = QHBoxLayout()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        vlayout = QVBoxLayout()
        # load control panel
        self.control_panel = ControlPanel(self)
        vlayout.addWidget(self.control_panel)

        hh_layout = QHBoxLayout()

        # previous file button
        self.previous_file_btn = QPushButton("Previous file")
        self.previous_file_btn.clicked.connect(self.previous_file_clicked)
        hh_layout.addWidget(self.previous_file_btn)

        # next file button
        self.next_file_btn = QPushButton("Next file")
        self.next_file_btn.clicked.connect(self.next_file_clicked)
        hh_layout.addWidget(self.next_file_btn)

        hh_layout.addStretch()

        vlayout.addLayout(hh_layout)

        hhh_layout = QHBoxLayout()

        # save data button
        self.save_results_btn = QPushButton("Save results")
        self.save_results_btn.clicked.connect(self.save_results_clicked)
        hhh_layout.addWidget(self.save_results_btn)

        # auto button
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.clicked.connect(self.auto_btn_clicked)
        hhh_layout.addWidget(self.auto_btn)

        hhh_layout.addStretch()

        vlayout.addLayout(hhh_layout)

        left_layout.addLayout(vlayout)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        r_layout = QHBoxLayout()

        self.canvas = FigureCanvas(self.figure)
        r_layout.addWidget(self.canvas)

        right_layout.addLayout(r_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        layout.addWidget(splitter)

        self.setLayout(layout)

        self.range = self.xmax - self.xmin
        self.canvas.draw_idle()
        self.plot_wav(self.xmin, self.xmax)

        self.run_analysis()

    def init_values(self):
        self.window_size = WINDOW_SIZE
        self.overlap = OVERLAP
        self.min_amplitude = MIN_AMPLITUDE
        self.min_distance = MIN_DISTANCE
        self.fft_length = FFT_LENGTH
        self.fft_overlap = FFT_OVERLAP

        # Crea il dizionario dei risultati
        self.results_dict = {
            "file": None,
            "sampling_rate": None,
            "call_duration": None,
            "pulse_number": None,
            "spectrum": None,
            "spectrum_peaks": None,
        }

    def run_analysis(self):
        # compute envelope
        self.envelope()

        # find peaks
        self.trova_picchi()

        # spectrum
        self.plot_spectrum()

    def load_wav(self, wav_file):
        """
        Carica il file WAV e ne estrae i dati.
        """

        self.sampling_rate, self.data = wavfile.read(wav_file)

        self.xmin = 0
        self.duration = len(self.data) / self.sampling_rate
        self.xmax = self.duration
        if len(self.data.shape) > 1:
            print("File stereo rilevato. Uso il primo canale.")
            self.data = self.data[:, 0]
        self.data = self.data / np.max(np.abs(self.data))
        self.time = np.linspace(
            0, len(self.data) / self.sampling_rate, num=len(self.data)
        )
        self.id_xmin = 0
        self.id_xmax = len(self.data)

        self.canto = np.zeros(len(self.data))
        self.rms = np.zeros(len(self.data) // self.overlap)
        n_frames = np.arange(len(self.rms))
        self.rms_times = librosa.frames_to_time(
            n_frames, sr=self.sampling_rate, hop_length=self.overlap
        )
        self.peaks_times = np.array([])

        self.setWindowTitle(
            f"{Path(__file__).stem.replace('_', ' ')} - {Path(wav_file).stem}"
        )

    def plot_wav(self, xmin, xmax):
        """
        Aggiorna l'oscillogramma (sinistro) con il segnale e l'envelope.
        """
        self.ax.cla()
        self.xmin = xmin
        self.xmax = xmax
        self.id_xmin = int(self.xmin * self.sampling_rate)
        self.id_xmax = int(self.xmax * self.sampling_rate)
        time_segment = self.time[self.id_xmin : self.id_xmax]
        data_segment = self.data[self.id_xmin : self.id_xmax]
        canto_segment = self.canto[self.id_xmin : self.id_xmax]
        print("xmin", self.xmin, "xmax", self.xmax)
        self.ax.plot(
            time_segment, data_segment, linewidth=0.5, color="black", alpha=0.25
        )
        self.ax.plot(time_segment, canto_segment, "-", color="blue")
        mask_rms = (self.rms_times >= self.xmin) & (self.rms_times <= self.xmax)
        rms_times_sel = self.rms_times[mask_rms]
        rms_sel = self.rms[mask_rms]
        if len(rms_sel) > 0:
            self.ax.plot(rms_times_sel, rms_sel, linewidth=1, color="red")

        # seleziono i picchi che ricadono nell'intervallo xmin-xmax

        print(len(self.peaks_times))

        if len(self.peaks_times) > 0:
            mask_peaks = (self.peaks_times >= self.xmin) & (
                self.peaks_times <= self.xmax
            )
            print(mask_peaks)

            peaks_selected = self.peaks_times[mask_peaks]
        else:
            peaks_selected = np.array([])

        if len(peaks_selected) > 0:
            for i in np.arange(len(peaks_selected)):
                self.ax.plot(
                    [peaks_selected[i], peaks_selected[i]], [0, 1], "-g", linewidth=1
                )

        self.ax.plot()
        self.ax.plot(rms_times_sel, rms_sel, linewidth=1, color="red")
        self.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        """
        Aggiorna il range in base alla selezione (SpanSelector).
        """
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
        """
        Calcola l'envelope (RMS) usando i parametri correnti e aggiorna l'oscillogramma.
        """

        try:
            if self.window_size <= 0 or self.overlap < 0:
                print("Errore: Window size deve essere > 0 e Overlap >= 0")
                return
            self.rms = librosa.feature.rms(
                y=self.data, frame_length=self.window_size, hop_length=self.overlap
            )[0]
            self.rms_times = librosa.frames_to_time(
                np.arange(len(self.rms)), sr=self.sampling_rate, hop_length=self.overlap
            )
            print("Envelope calcolato, lunghezza:", len(self.rms))
            self.plot_wav(self.xmin, self.xmax)
        except Exception as e:
            print("Errore in envelope:", e)

    def plot_spectrum(self):
        """
        Calcola e visualizza lo spettro di potenza del segmento selezionato.
        """

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

            print(f"spectrum_peaks: {self.spectrum_peaks_Hz}")

            self.results_dict["spectrum"] = np.concatenate(
                ([freqs], [power]), axis=1
            ).tolist()
            self.results_dict["spectrum_peaks"] = np.concatenate(
                ([self.spectrum_peaks_Hz], [spectrum_peaks_db]), axis=1
            ).tolist()

            self.ax2.cla()
            self.ax2.plot(freqs, avg_power_db, color="blue")
            self.ax2.plot(
                self.spectrum_peaks_Hz, avg_power_db[self.spectrum_peaks], "or"
            )

            self.ax2.set_title("Power Spectrum")
            self.ax2.set_xlabel("Frequency (Hz)")
            self.ax2.set_ylabel("Power")
            self.canvas.draw_idle()
        except Exception as e:
            print("Errore in plot_spectrum:", e)

    def trova_picchi(self):
        """
        Trova i picchi dell'inviluppo RMS e li converte nei campioni della registrazione originale.
        """
        try:
            min_distance_samples = int(
                self.min_distance * (self.sampling_rate / self.overlap)
            )  # Converti in campioni

            # Trova i picchi nell'inviluppo RMS
            peaks, properties = find_peaks(
                self.rms,
                height=self.min_amplitude,
                distance=min_distance_samples,
                prominence=0.01,
            )

            # Converti gli indici nei campioni effettivi dell'audio originale
            self.peaks_times = peaks * self.overlap / self.sampling_rate  # In secondi

            self.trova_ini_fin()
        except ValueError:
            print(
                " Errore: Inserisci valori numerici validi per la distanza e la soglia."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "",
                f"Funzione Trova picchi\n\nError on file {self.wav_file}\n\n{e}",
            )

    def trova_ini_fin(self):
        # trova inizio
        peaks = self.peaks_times * self.sampling_rate / self.overlap
        rms_noise = np.mean(self.rms[: int(peaks[0] // 2)])

        rms_ini = self.rms[: int(peaks[0])]
        trova_ini = np.where(rms_ini > rms_noise * 2)[0]

        if np.size(trova_ini) > 0:
            inizio = int(trova_ini[-1] * self.overlap)
        else:
            inizio = 0
        # trova fine
        rms_fine = self.rms[int(peaks[-1]) :]
        trova_fine = np.where(rms_fine <= rms_noise * 2)[0][0]
        fine = (int(peaks[-1]) + trova_fine) * self.overlap
        self.canto = np.zeros(len(self.rms) * self.overlap)
        self.canto[inizio:fine] = np.max(self.rms)

        self.plot_wav(self.xmin, self.xmax)

    def save_results_clicked(self):
        """
        Salva i risultati delle analisi nel file data.json
        """

        sample = int(Path(self.wav_file).stem.split("_")[-1])

        # self.run_analysis()

        # test if data.json exists
        if not (Path(self.wav_file).parent / "data.json").is_file():
            QMessageBox.warning(self, "", "The data.json file does not exist")
            return

        self.results_dict["file"] = Path(self.wav_file).stem
        self.results_dict["sampling_rate"] = self.sampling_rate
        self.results_dict["call_duration"] = len(self.canto) / self.sampling_rate
        self.results_dict["pulse_number"] = len(self.peaks_times)

        save_path = Path(self.wav_file).parent / "data.json"

        # read json content
        with open(save_path, "r", encoding="utf-8") as f_in:
            parameters = json.load(f_in)

        if "songs" not in parameters:
            parameters["songs"] = {}

        parameters["songs"][str(sample)] = {}
        parameters["songs"][str(sample)]["file"] = Path(self.wav_file).stem
        parameters["songs"][str(sample)]["window_size"] = self.window_size
        parameters["songs"][str(sample)]["overlap"] = self.overlap
        parameters["songs"][str(sample)]["min_amplitude"] = self.min_amplitude
        parameters["songs"][str(sample)]["min_distance"] = self.min_distance
        parameters["songs"][str(sample)]["fft_length"] = self.fft_length
        parameters["songs"][str(sample)]["fft_overlap"] = self.fft_overlap
        parameters["songs"][str(sample)]["sampling rate"] = self.sampling_rate
        parameters["songs"][str(sample)]["call_duration"] = (
            len(self.canto) / self.sampling_rate
        )
        parameters["songs"][str(sample)]["pulse_number"] = len(self.peaks_times)
        parameters["songs"][str(sample)]["peaks_times"] = self.peaks_times.tolist()
        parameters["songs"][str(sample)]["spectrum"] = self.results_dict["spectrum"]
        parameters["songs"][str(sample)]["spectrum peaks"] = self.results_dict[
            "spectrum_peaks"
        ]

        # save in data.json
        try:
            with open(save_path, "w", encoding="utf-8") as f_out:
                json.dump(parameters, f_out, indent=0, ensure_ascii=False)

            print(f"Risultati salvati in {save_path}")
        except Exception as e:
            print(f"Errore nel salvataggio dei risultati: {e}")

        # Save the dictionary to a pickle file
        with open("data.pkl", "wb") as f_out:
            pickle.dump(parameters, f_out)

    def next_file_clicked(self):
        """
        load next file
        """
        wav_list = sorted(Path(self.wav_file).parent.glob("*.wav"))
        for idx, wav_file in enumerate(wav_list):
            if Path(self.wav_file).name == wav_file.name:
                break
        else:
            QMessageBox.critical(self, "", "Current file not found")
            return

        if idx >= len(wav_list):
            QMessageBox.critical(self, "", "Last file")
            return

        self.wav_file = wav_list[idx + 1]

        self.load_wav(self.wav_file)
        self.plot_wav(self.xmin, self.xmax)
        self.run_analysis()

    def previous_file_clicked(self):
        """
        load previous file
        """
        wav_list = sorted(Path(self.wav_file).parent.glob("*.wav"))
        for idx, wav_file in enumerate(wav_list):
            if Path(self.wav_file).name == wav_file.name:
                break
        else:
            QMessageBox.critical(self, "", "Current file not found")
            return

        if idx == 0:
            QMessageBox.critical(self, "", "First file of directory")
            return

        self.wav_file = wav_list[idx - 1]

        self.load_wav(self.wav_file)
        self.plot_wav(self.xmin, self.xmax)
        self.run_analysis()

    def auto_btn_clicked(self):
        """
        automatically process current file and next files
        """
        wav_list = sorted(Path(self.wav_file).parent.glob("*.wav"))
        for idx, wav_file in enumerate(wav_list):
            if wav_file.name >= Path(self.wav_file).name:
                self.wav_file = wav_file
                self.load_wav(self.wav_file)

                self.plot_wav(self.xmin, self.xmax)

                self.run_analysis()

                self.save_results_clicked()


class ControlPanel(QWidget):
    def __init__(self, plot_panel):
        super().__init__()
        self.plot_panel = plot_panel
        self.setWindowTitle("Control Panel")
        self.setGeometry(
            1100, 100, 300, 400
        )  # Posiziona la finestra dei controlli separata

        # Layout per i parametri dell'envelope
        envelope_layout = QHBoxLayout()

        # window size
        self.window_size_input = QSpinBox()
        self.window_size_input.setMinimum(10)
        self.window_size_input.setMaximum(1000)
        self.window_size_input.setValue(WINDOW_SIZE)
        self.window_size_input.setSingleStep(10)
        self.window_size_input.valueChanged.connect(self.window_size_changed)

        # overlap
        self.overlap_input = QSpinBox()
        self.overlap_input.setMinimum(10)
        self.overlap_input.setMaximum(1000)
        self.overlap_input.setValue(OVERLAP)
        self.overlap_input.setSingleStep(10)
        self.overlap_input.valueChanged.connect(self.overlap_changed)

        envelope_layout.addWidget(QLabel("Window size"))
        envelope_layout.addWidget(self.window_size_input)
        envelope_layout.addWidget(QLabel("Overlap"))
        envelope_layout.addWidget(self.overlap_input)
        self.envelope_btn = QPushButton("Compute Envelope")
        envelope_layout.addWidget(self.envelope_btn)
        self.envelope_btn.clicked.connect(self.envelope_clicked)

        # Layout per i parametri del peak finder
        peak_finder_layout = QHBoxLayout()

        # MIN_AMPLITUDE
        self.amp_threshold_input = QDoubleSpinBox()
        self.amp_threshold_input.setDecimals(3)  # Set to 3 decimal places
        self.amp_threshold_input.setSingleStep(0.005)  # Step size of 0.1
        self.amp_threshold_input.setMinimum(0.0)  # Set minimum value
        self.amp_threshold_input.setMaximum(1)  # Set maximum value
        self.amp_threshold_input.setValue(MIN_AMPLITUDE)
        self.amp_threshold_input.valueChanged.connect(self.min_amplitude_changed)

        # MIN_DISTANCE
        self.min_distance_input = QDoubleSpinBox()
        self.min_distance_input.setDecimals(3)  # Set to 3 decimal places
        self.min_distance_input.setSingleStep(0.005)  # Step size of 0.1
        self.min_distance_input.setMinimum(0.0)  # Set minimum value
        self.min_distance_input.setMaximum(1)  # Set maximum value
        self.min_distance_input.setValue(MIN_DISTANCE)
        self.min_distance_input.valueChanged.connect(self.min_distance_changed)

        peak_finder_layout.addWidget(QLabel("Amplitude Threshold"))
        peak_finder_layout.addWidget(self.amp_threshold_input)
        peak_finder_layout.addWidget(QLabel("Minimum Distance"))
        peak_finder_layout.addWidget(self.min_distance_input)
        self.peaks_btn = QPushButton("Find Peaks")
        peak_finder_layout.addWidget(self.peaks_btn)
        self.peaks_btn.clicked.connect(self.peaks_clicked)

        # Layout per i parametri dello spettro
        spectrum_layout = QHBoxLayout()

        # FFT LENGTH
        self.fft_length_input = QSpinBox()
        self.fft_length_input.setMinimum(10)
        self.fft_length_input.setMaximum(10000)
        self.fft_length_input.setValue(FFT_LENGTH)
        self.fft_length_input.setSingleStep(10)
        self.fft_length_input.valueChanged.connect(self.fft_length_changed)

        # FFT OVERLAP
        self.fft_overlap_input = QSpinBox()
        self.fft_overlap_input.setMinimum(10)
        self.fft_overlap_input.setMaximum(10000)
        self.fft_overlap_input.setValue(FFT_OVERLAP)
        self.fft_overlap_input.setSingleStep(10)
        self.fft_overlap_input.valueChanged.connect(self.fft_overlap_changed)

        spectrum_layout.addWidget(QLabel("FFT Length"))
        spectrum_layout.addWidget(self.fft_length_input)
        spectrum_layout.addWidget(QLabel("Overlap"))
        spectrum_layout.addWidget(self.fft_overlap_input)
        self.spectrum_btn = QPushButton("Compute Spectrum")
        spectrum_layout.addWidget(self.spectrum_btn)
        self.spectrum_btn.clicked.connect(self.spectrum_clicked)

        # Layout principale
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Envelope Parameters"))
        main_layout.addLayout(envelope_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(QLabel("Peak Finder Parameters"))
        main_layout.addLayout(peak_finder_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(QLabel("Spectrum Parameters"))
        main_layout.addLayout(spectrum_layout)
        main_layout.addStretch()

        # reset values
        h_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset values")
        self.reset_btn.clicked.connect(self.reset_values)
        h_layout.addWidget(self.reset_btn)
        h_layout.addStretch()
        main_layout.addLayout(h_layout)
        self.setLayout(main_layout)

    def reset_values(self):
        self.window_size_input.setValue(WINDOW_SIZE)
        self.overlap_input.setValue(OVERLAP)
        self.amp_threshold_input.setValue(MIN_AMPLITUDE)
        self.min_distance_input.setValue(MIN_DISTANCE)
        self.fft_length_input.setValue(FFT_LENGTH)
        self.fft_overlap_input.setValue(FFT_OVERLAP)

    def window_size_changed(self, new_value):
        self.plot_panel.window_size = new_value
        self.plot_panel.envelope()

    def overlap_changed(self, new_value):
        self.plot_panel.overlap = new_value
        self.plot_panel.envelope()

    def min_amplitude_changed(self, new_value):
        self.plot_panel.min_amplitude = new_value
        self.peaks_clicked()

    def min_distance_changed(self, new_value):
        self.plot_panel.min_distance = new_value
        self.peaks_clicked()

    def fft_length_changed(self, new_value):
        self.plot_panel.fft_length = new_value
        self.plot_panel.plot_spectrum()

    def fft_overlap_changed(self, new_value):
        self.plot_panel.fft_overlap = new_value
        self.plot_panel.plot_spectrum()

    def envelope_clicked(self):
        try:
            self.plot_panel.envelope()
        except Exception as e:
            print("Errore nei parametri envelope:", e)

    def peaks_clicked(self):
        try:
            self.plot_panel.trova_picchi(min_amplitude, min_distance)
        except Exception as e:
            print("Errore nei parametri peak finder:", e)

    def spectrum_clicked(self):
        try:
            self.plot_panel.plot_spectrum()
        except Exception as e:
            print("Errore nei parametri spectrum:", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Crea la finestra dei plots e quella dei controlli
    plot_panel = Main(
        wav_file="/tmp/ramdisk/GeCorn_2025-01-25_09/GeCorn_2025-01-25_09_sample_000096256.wav"
    )
    # plot_panel = Main(wav_file="/tmp/ramdisk/GeCorn_2025-01-25_09/GeCorn_2025-01-25_09_sample_000017408.wav")

    # plot_panel = Main(        wav_file="GeCorn_2025-01-25_09/GeCorn_2025-01-25_09_sample17408.wav"    )
    plot_panel.show()
    sys.exit(app.exec())
