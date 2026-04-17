"""
Plugin for analysis of single song
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import librosa
import pickle
from PySide6.QtCore import QTimer
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
    QFileDialog,
)
from PySide6.QtCore import Qt
from pathlib import Path


WINDOW_SIZE = 10
OVERLAP = 10
MIN_AMPLITUDE = 0.1
MIN_DISTANCE = 0.003
MAX_DISTANCE = 0.01
PROMINENCE = 0.01
FFT_LENGTH = 1024
FFT_OVERLAP = 512
SIGNAL_TO_NOISE_RATIO = 2

def iter_song_jobs(parameters: dict):
    for wav_key, wav_block in parameters.items():
        songs = wav_block.get("songs", {})
        #print(f"songs = {songs}")
        if not isinstance(songs, dict):
            continue
        for song_id, sp in songs.items():
            if isinstance(sp, dict):
                yield wav_key, str(song_id), sp


def wav_path_same_folder(json_path: Path, sp: dict, wav_key: str) -> Path:
    name = sp.get("file") or wav_key
    return (json_path.parent / Path(name).name).resolve()



def wav_path_different_folder(json_path: Path, sp: dict, wav_key: str) -> Path:
    """
    Cerca il WAV in una sottocartella il cui nome è il nome del file
    senza il suffisso '_sample_...'.

    Esempio:
      file:    SpiAglav_..._005824626_sample_001213191.wav
      folder:  SpiAglav_..._005824626
    """
    name = sp.get("file") or wav_key
    wav_name = Path(name).name          # filename completo
    stem = Path(wav_name).stem          # senza estensione

    # split sul token '_sample_'
    folder_name = stem.split("_sample_")[0]

    wav_path = json_path.parent / folder_name / wav_name
    return wav_path.resolve()



class Main(QWidget):
    def __init__(self, wav_file_list: list):
        super().__init__()

        self.init_values()
        
        self.json_root = Path(r"C:\Users\scast\Desktop\REGISTRAZIONI_per_specie\GuTorn")

        self.wav_file_list = wav_file_list or []
        self.wav_file = None

        if self.wav_file_list:
            self.wav_file = self.wav_file_list[0]
            print("Carico il file:", self.wav_file)
            self.load_wav(self.wav_file)
        else:
            # avvio "idle": niente wav caricato
            self.data = np.array([])
            self.sampling_rate = None
            self.duration = 0
            self.time = np.array([])
            self.xmin, self.xmax = 0, 1
            self.id_xmin, self.id_xmax = 0, 0
            self.canto = np.array([])
            self.rms = np.array([])
            self.rms_times = np.array([])
            self.peaks_times = np.array([])
            self.setWindowTitle(f"{Path(__file__).stem.replace('_',' ')} - (no file loaded)")

        

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

        if self.wav_file_list:
            self.run_analysis()


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
        # self.close()

    def load_job(self, index: int):
        """
        Carica il song 'index' dal JSON (song_jobs), applica i parametri salvati
        e rilancia l'analisi.
        """
        if not hasattr(self, "song_jobs") or not self.song_jobs:
            return

        # clamp index
        index = max(0, min(index, len(self.song_jobs) - 1))
        self.current_job_index = index

        wav_key, song_id, sp = self.song_jobs[index]

        # 1) Applica parametri del song (window_size, overlap, soglie, FFT, ecc.)
        self.apply_song_params(sp)

        # 2) Carica WAV associato (dalla lista già costruita in leggi_json)
        if hasattr(self, "wav_file_list") and self.wav_file_list and index < len(self.wav_file_list):
            self.wav_file = self.wav_file_list[index]
        else:
            # fallback (non dovrebbe servire se wav_file_list è allineata)
            self.wav_file = str(wav_path_same_folder(self.input_json_path, sp, wav_key))

        self.load_wav(self.wav_file)
        self.plot_wav(self.xmin, self.xmax)

        # 3) carica i picchi salvati nel JSON
        self.peaks_times = np.array(sp.get("peaks_times", []), dtype=float)

        # 4) ricalcola solo l’envelope
        self.envelope()

        # 5) calcola l’intensità dei picchi sui tempi già esistenti
        self.trova_intensita_picchi()
        self.trova_ini_fin()
        # 6) ricalcola lo spettro
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
        if self.sampling_rate is None or self.data is None or len(self.data) == 0:
            return
        self.ax.cla()
        self.xmin = xmin
        self.xmax = xmax
        self.id_xmin = int(self.xmin * self.sampling_rate)
        self.id_xmax = int(self.xmax * self.sampling_rate)
        time_segment = self.time[self.id_xmin : self.id_xmax]
        data_segment = self.data[self.id_xmin : self.id_xmax]
        canto_segment = self.canto[self.id_xmin : self.id_xmax]
        #print("xmin", self.xmin, "xmax", self.xmax)
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

        #print(len(self.peaks_times))

        if len(self.peaks_times) > 0:
            mask_peaks = (self.peaks_times >= self.xmin) & (
                self.peaks_times <= self.xmax
            )
            # print(mask_peaks)

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
            

            #self.peaks_times = []
            self.canto = np.zeros(len(self.rms) * self.overlap)
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

            # print(f"spectrum_peaks: {self.spectrum_peaks_Hz}")

            self.results_dict["spectrum"] = np.concatenate(
                ([freqs], [power]), axis=0
            ).tolist()

            self.results_dict["spectrum_peaks"] = np.concatenate(
                ([self.spectrum_peaks_Hz], [spectrum_peaks_db]), axis=0
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
            max_distance_samples = int(
                self.max_distance * (self.sampling_rate / self.overlap)
            )  # Converti in campioni

            # Trova i picchi nell'inviluppo RMS
            peaks, properties = find_peaks(
                self.rms,
                height=self.min_amplitude,
                distance=min_distance_samples,
                prominence=self.prominence,
            )
            peaks_filtered = [peaks[0]]
            # Elimino i picchi troppo distanti dal precedente
            for i in np.arange(1, len(peaks)):
                if peaks[i] - peaks_filtered[-1] < max_distance_samples:
                    peaks_filtered.append(peaks[i])

            peaks_filtered = np.array(peaks_filtered)
            mean_distance_between_peaks = np.mean(np.diff(peaks_filtered))
            sdt_distance_between_peaks = np.std(np.diff(peaks_filtered))
            #print("STD=", sdt_distance_between_peaks)
            peaks = [peaks_filtered[0]]
            # ultimo check
            for i in np.arange(1, len(peaks_filtered)):
                #print(peaks_filtered[i] - peaks[-1])
                if (
                    peaks_filtered[i] - peaks[-1]
                ) < mean_distance_between_peaks + 3 * sdt_distance_between_peaks:
                    peaks.append(peaks_filtered[i])

            peaks_filtered = np.array(peaks)
            # Converti gli indici nei campioni effettivi dell'audio originale
            
            self.peaks_times = (
                np.array(peaks_filtered) * self.overlap / self.sampling_rate
            )  # In secondi
            self.peaks_int = self.rms[peaks_filtered]
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

    def trova_intensita_picchi(self):
        """
        Calcola int_picchi usando peaks_times già presenti nel json
        
        """
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
        # trova inizio
        peaks = self.peaks_times * self.sampling_rate / self.overlap
        peaks = np.asarray(peaks, dtype = np.float64)

        # Se peaks vuoto o poco affidabile, usa il massimo RMS come ancora
        if peaks.size == 0:
            anchor = int(np.argmax(self.rms))
            peaks = np.array([anchor, anchor], dtype=float)
        elif peaks.size == 1:
            peaks = np.array([peaks[0], peaks[0]], dtype=float)
            rms_noise_ini = np.mean(self.rms[: int(peaks[0] // 2)])
            rms_noise_fin = np.mean(self.rms[int(peaks[-1]) :])

        p0 = int(np.clip(peaks[0], 0, len(self.rms)-1))
        p1 = int(np.clip(peaks[-1], 0, len(self.rms)-1))
       
    # -----------------------------
    # 1) Smoothing leggero dell'RMS (anti-buchi)
    # -----------------------------
        w = 11  # prova 5, 7, 9 (deve essere piccolo)
        if len(self.rms) >= w:
            kernel = np.ones(w) / w
            rms_s = np.convolve(self.rms, kernel, mode="same")
        else:
            rms_s = self.rms
            rms_ini = self.rms[: int(peaks[0])]

    # -----------------------------
    # 2) Stima rumore robusta (percentile + MAD)
    # -----------------------------  

        ini_end = max(1, int(p0 * 0.9))
        fin_start = min(len(rms_s)-1, p1)

        rms_ini_noise = rms_s[:ini_end]
        rms_fin_noise = rms_s[fin_start:] if fin_start < len(rms_s) else rms_s[-1:]

        # Se una delle due porzioni è troppo corta, fallback su tutta la traccia
        if len(rms_ini_noise) < 10:
            rms_ini_noise = rms_s
            
        if len(rms_fin_noise) < 10:
            rms_fin_noise = rms_s

        # livello rumore = percentile basso (più robusto della media)
        q = 0.6  #  percentile
        noise_ini = np.quantile(rms_ini_noise, q)
        noise_fin = np.quantile(rms_fin_noise, q)

        # MAD (Median absolute deviation) 
        def mad(x):
            x = np.asarray(x)
            m = np.median(x)
            return np.median(np.abs(x - m)) + 1e-12

        mad_ini = mad(rms_ini_noise)
        mad_fin = mad(rms_fin_noise)
    
    # -----------------------------
    # 3) Isteresi: soglia ON / OFF
    # ---
        k_on = self.signal_to_noise_ratio        # usa il parametro da input
        k_off = max(1.0, 2 * k_on)             # OFF più bassa (60% della ON)

        thr_on_ini  = noise_ini + k_on  * mad_ini
        thr_off_ini = noise_ini + k_off * mad_ini

        thr_on_fin  = noise_fin + k_on  * mad_fin
        thr_off_fin = noise_fin + k_off * mad_fin

        
        print(f"thr_on_ini  {thr_off_ini} noise_ini  {noise_ini}  k_on {k_on}   mad_ini {mad_ini}")
        print(f"Soglia_IN = {thr_off_ini}, mentre rms totale = {noise_ini}, nframes = {rms_s[:p0]}")
        print(f"RMS_INI_NOIS = {rms_ini_noise}")
        inizio_frame = 0
        in_call = True  

        # 
        max_back_ms = 100  # scala biologicamente sensata
        start_i = max(0, p0 - int((max_back_ms / 1000) * self.sampling_rate / self.overlap))

        need_off = 1
        count_off = 0
        inizio_frame = None

        for i in range(p0, start_i - 1, -1):
            if rms_s[i] <= thr_off_ini:
                print(f"condizione soddisfatta:{rms_s[i]}  ")
                count_off += 1
                if count_off >= need_off:
                    inizio_frame = i
                    break
            else:
                print(f"condizione NON soddisfatta:{rms_s[i]} {thr_off_ini}") 
                count_off = 0

        if inizio_frame is not None:
            inizio = int((inizio_frame + 3) * self.overlap)
        else:
            # fallback molto vicino al picco (pochi ms)
            inizio = int(start_i * self.overlap)
        
        print(f"inizio {inizio} p0  {p0}")

        inizio = np.min([p0 * self.overlap - 1, inizio])
        self.ini_canto = inizio

        # -----------------------------
        # 5) Trova FINE con isteresi (scorrendo in avanti da p1)
        # -----------------------------
        fine_frame = len(rms_s) - 1
        in_call = True

        for i in range(p1, len(rms_s)):
            v = rms_s[i]
            if in_call:
                if v <= thr_off_fin:
                    in_call = False
                    fine_frame = i
                    break
            else:
                if v >= thr_on_fin:
                    in_call = True

        fine = int((fine_frame + 1) * self.overlap)  # +1 per includere il frame

        fine = max(fine, inizio + 1)
        fine = min(fine, len(rms_s) * self.overlap)

        self.canto = np.zeros(len(self.rms) * self.overlap)
        self.canto[inizio:fine] = np.max(self.rms)
        self.durata_canto = (fine - inizio)/self.sampling_rate
        #print(f"durata_canto: inizio ={inizio}, fine = {fine}, durata = fine - inizio, durata_sec = {(fine-inizio)/self.sampling_rate} ")
        #introduco una variabile globale che mi serve per calcolare correttamente envelope50, envelope80, envelope90
        self.rms_canto = self.rms[int(inizio/self.overlap):int(fine/self.overlap)]
        print(f"inizio {inizio}    fine {fine}   len {len(self.rms_canto)}")
        
        
        self.plot_wav(self.xmin, self.xmax)

    def save_results_clicked(self):
        """
        Salva i risultati delle analisi nel file .json
        """

        # read sample position in file name
        sample = int(Path(self.wav_file).stem.split("_")[-1])
        
        data_file_path = self.output_json_path
        with open(data_file_path, "r", encoding="utf-8") as f:
            parameters = json.load(f)

        # carica output json
        with open(self.output_json_path, "r", encoding="utf-8") as f:
            parameters = json.load(f)

        # recupera il wav_key associato al song corrente (chiave principale nel json)
        if hasattr(self, "song_jobs") and self.song_jobs:
            wav_key, song_id, sp = self.song_jobs[self.current_job_index]
            file_name = wav_key  # <-- QUESTO deve combaciare con la chiave nel json
        else:
            # fallback: usa solo il nome del wav corrente
            file_name = Path(self.wav_file).name

        
        
        # CREA la chiave se manca (evita KeyError)
        parameters.setdefault(file_name, {})
        parameters[file_name].setdefault("songs", {})
        parameters[file_name]["songs"][str(sample)] = {}
        parameters[file_name]["songs"][str(sample)]["file"] = Path(self.wav_file).name
        parameters[file_name]["songs"][str(sample)]["window_size"] = self.window_size
        parameters[file_name]["songs"][str(sample)]["overlap"] = self.overlap
        parameters[file_name]["songs"][str(sample)]["min_amplitude"] = (
            self.min_amplitude
        )  # amp threshold
        parameters[file_name]["songs"][str(sample)]["min_distance"] = self.min_distance
        parameters[file_name]["songs"][str(sample)]["max_distance"] = self.max_distance
        parameters[file_name]["songs"][str(sample)]["prominence"] = self.prominence
        parameters[file_name]["songs"][str(sample)]["signal_to_noise_ratio"] = (
            self.signal_to_noise_ratio
        )
        parameters[file_name]["songs"][str(sample)]["fft_length"] = self.fft_length
        parameters[file_name]["songs"][str(sample)]["fft_overlap"] = self.fft_overlap
        parameters[file_name]["songs"][str(sample)]["sampling rate"] = (
            self.sampling_rate
        )

        envelope_max = np.max(self.rms)
        soglia50 = 0.5 * envelope_max   
        soglia80 = 0.8 * envelope_max
        soglia90 = 0.9 * envelope_max
        self.envelope50 = np.sum(np.where(self.rms_canto >= soglia50)) / len(self.rms_canto)
        self.envelope80 = np.sum(np.where(self.rms_canto >= soglia80)) / len(self.rms_canto)

        #CHECK: 
        print(f"ENVELOPE50: {self.envelope50} RMS canto {len(self.rms_canto)}")

        if envelope_max <= 0:
            self.fase_ascendente = np.nan
            self.fase_discendente = np.nan
        else:
            
            idx90 = np.where(self.rms >= soglia90)[0]
            if idx90.size == 0:
                self.fase_ascendente = np.nan
                self.fase_discendente = np.nan
            else:
                dt = self.overlap / self.sampling_rate
                self.fase_ascendente = idx90[0] * dt - self.ini_canto/self.sampling_rate
                self.fase_discendente = self.durata_canto - (idx90[-1] * dt - self.ini_canto/self.sampling_rate)
                

        


        parameters[file_name]["songs"][str(sample)]["call_duration"] = (
            self.durata_canto
        )
        parameters[file_name]["songs"][str(sample)]["pulse_number"] = len(
            self.peaks_times
        )
        parameters[file_name]["songs"][str(sample)]["envelope50"] = (
                    self.envelope50
                )
        parameters[file_name]["songs"][str(sample)]["envelope80"] = (
            self.envelope80
        )
        parameters[file_name]["songs"][str(sample)]["fase_ascendente"] = (
            self.fase_ascendente
        )
        parameters[file_name]["songs"][str(sample)]["fase_discendente"] = (
            self.fase_discendente
        )

        parameters[file_name]["songs"][str(sample)]["peaks_times"] = (
            self.peaks_times.tolist()
        )
        parameters[file_name]["songs"][str(sample)]["peaks_int"] = (
            self.peaks_int.tolist()
        )
        
        parameters[file_name]["songs"][str(sample)]["FFT_length"] = self.control_panel.fft_length_input.value()
        parameters[file_name]["songs"][str(sample)]["FFT_overlap"] = self.control_panel.fft_overlap_input.value()

        parameters[file_name]["songs"][str(sample)]["spectrum"] = self.results_dict[
            "spectrum"
        ]
        parameters[file_name]["songs"][str(sample)]["spectrum peaks"] = (
            self.results_dict["spectrum_peaks"]
        )

        # save in .json
        try:
            with open(data_file_path, "w", encoding="utf-8") as f_out:
                json.dump(parameters, f_out, indent=0, ensure_ascii=False)

            print(f"Risultati salvati in {data_file_path}")
        except Exception as e:
            print(f"Errore nel salvataggio dei risultati: {e}")

        # Save the dictionary to a pickle file
        with open(data_file_path.with_suffix(".pkl"), "wb") as f_out:
            pickle.dump(parameters, f_out)

    def next_file_clicked(self):
        # modalità JSON: usa i job
        if hasattr(self, "song_jobs") and self.song_jobs:
            if self.current_job_index >= len(self.song_jobs) - 1:
                QMessageBox.critical(self, "", "Last file")
                return
            self.load_job(self.current_job_index + 1)
            return

        # modalità “normale”: scorre la wav_file_list
        current_wav_index = self.wav_file_list.index(self.wav_file)
        if current_wav_index == len(self.wav_file_list) - 1:
            QMessageBox.critical(self, "", "Last file")
            return

        self.wav_file = self.wav_file_list[current_wav_index + 1]
        self.load_wav(self.wav_file)
        self.plot_wav(self.xmin, self.xmax)
        self.run_analysis()


    def previous_file_clicked(self):
        # modalità JSON: usa i job
        if hasattr(self, "song_jobs") and self.song_jobs:
            if self.current_job_index <= 0:
                QMessageBox.critical(self, "", "First file of directory")
                return
            self.load_job(self.current_job_index - 1)
            return

        # modalità “normale”
        current_wav_index = self.wav_file_list.index(self.wav_file)
        if current_wav_index == 0:
            QMessageBox.critical(self, "", "First file of directory")
            return

        self.wav_file = self.wav_file_list[current_wav_index - 1]
        self.load_wav(self.wav_file)
        self.plot_wav(self.xmin, self.xmax)
        self.run_analysis()

 
    def auto_btn_clicked(self):
        """
        automatically process current file and next files
        """

        # modalità JSON
        if hasattr(self, "song_jobs") and self.song_jobs:
            start = getattr(self, "current_job_index", 0)
            for idx in range(start, len(self.song_jobs)):
                self.load_job(idx)
                self.save_results_clicked()
            return

        # modalità “normale”
        for wav_file in self.wav_file_list[self.wav_file_list.index(self.wav_file) :]:
            self.wav_file = wav_file
            self.load_wav(self.wav_file)
            self.plot_wav(self.xmin, self.xmax)
            self.run_analysis()
            self.save_results_clicked()


    def apply_song_params(self, sp: dict):
        """Applica a Main i parametri letti dal json (se presenti)."""
        # envelope/peaks
        self.window_size = int(sp.get("window_size", self.window_size))
        self.overlap = int(sp.get("overlap", self.overlap))
        self.min_amplitude = float(sp.get("min_amplitude", self.min_amplitude))
        self.min_distance = float(sp.get("min_distance", self.min_distance))
        self.max_distance = float(sp.get("max_distance", self.max_distance))
        self.prominence = float(sp.get("prominence", self.prominence))
        self.signal_to_noise_ratio = float(
            sp.get("signal_to_noise_ratio", self.signal_to_noise_ratio)
        )

        # spectrum
        self.fft_length = int(sp.get("fft_length", self.fft_length))
        self.fft_overlap = int(sp.get("fft_overlap", self.fft_overlap))

        # opzionale: aggiorna anche i widget del pannello (se esistono)
        if hasattr(self, "control_panel"):
            cp = self.control_panel
            cp.window_size_input.setValue(self.window_size)
            cp.overlap_input.setValue(self.overlap)
            cp.amp_threshold_input.setValue(self.min_amplitude)
            cp.min_distance_input.setValue(self.min_distance)
            cp.max_distance_input.setValue(self.max_distance)
            cp.prominence_input.setValue(self.prominence)
            cp.signal_noise_ration_input.setValue(self.signal_to_noise_ratio)
            cp.fft_length_input.setValue(self.fft_length)
            cp.fft_overlap_input.setValue(self.fft_overlap)


class ControlPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setWindowTitle("Control Panel")
        self.setGeometry(
            1100, 100, 300, 400
        )  # Posiziona la finestra dei controlli separata

        # Layout per i parametri dell'envelope
        envelope_layout = QVBoxLayout()

        envelope_layout.addWidget(QLabel("Envelope Parameters"))

        h_layout = QHBoxLayout()

        # envelope window size
        self.window_size_input = QSpinBox()
        self.window_size_input.setMinimum(10)
        self.window_size_input.setMaximum(1000)
        self.window_size_input.setValue(WINDOW_SIZE)
        self.window_size_input.setSingleStep(10)
        self.window_size_input.valueChanged.connect(self.window_size_changed)

        #  envelope overlap
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
        h_layout.addStretch()

        envelope_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        self.envelope_btn = QPushButton("Compute Envelope")
        self.envelope_btn.clicked.connect(self.envelope_clicked)
        h_layout.addWidget(self.envelope_btn)
        h_layout.addStretch()
        envelope_layout.addLayout(h_layout)

        # Layout per i parametri del peak finder
        peak_finder_layout = QVBoxLayout()

        peak_finder_layout.addWidget(QLabel("Peak Finder Parameters"))

        h_layout = QHBoxLayout()

        # Amp threshold
        h_layout.addWidget(QLabel("Amplitude threshold"))

        self.amp_threshold_input = QDoubleSpinBox()
        self.amp_threshold_input.setDecimals(3)  # Set to 3 decimal places
        self.amp_threshold_input.setSingleStep(0.005)  # Step size of 0.1
        self.amp_threshold_input.setMinimum(0.0)  # Set minimum value
        self.amp_threshold_input.setMaximum(1)  # Set maximum value
        self.amp_threshold_input.setValue(MIN_AMPLITUDE)
        self.amp_threshold_input.valueChanged.connect(self.min_amplitude_changed)

        h_layout.addWidget(self.amp_threshold_input)

        h_layout.addStretch()

        peak_finder_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        # MIN_DISTANCE
        h_layout.addWidget(QLabel("Min"))

        self.min_distance_input = QDoubleSpinBox()
        self.min_distance_input.setDecimals(3)  # Set to 3 decimal places
        self.min_distance_input.setSingleStep(0.0005)  # Step size of 0.1
        self.min_distance_input.setMinimum(0.0005)  # Set minimum value
        self.min_distance_input.setMaximum(0.01)  # Set maximum value
        self.min_distance_input.setValue(MIN_DISTANCE)
        self.min_distance_input.valueChanged.connect(self.min_distance_changed)

        h_layout.addWidget(self.min_distance_input)

        # MAX_DISTANCE
        h_layout.addWidget(QLabel("Max"))

        self.max_distance_input = QDoubleSpinBox()
        self.max_distance_input.setDecimals(3)  # Set to 3 decimal places
        self.max_distance_input.setSingleStep(0.001)  # Step size of 0.1
        self.max_distance_input.setMinimum(0.0)  # Set minimum value
        self.max_distance_input.setMaximum(1)  # Set maximum value
        self.max_distance_input.setValue(MAX_DISTANCE)
        self.max_distance_input.valueChanged.connect(self.max_distance_changed)

        h_layout.addWidget(self.max_distance_input)
        h_layout.addStretch()

        peak_finder_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        # PROMINENCE
        h_layout.addWidget(QLabel("Prominence"))

        self.prominence_input = QDoubleSpinBox()
        self.prominence_input.setDecimals(3)  # Set to 3 decimal places
        self.prominence_input.setSingleStep(0.01)  # Step size of 0.1
        self.prominence_input.setMinimum(0.01)  # Set minimum value
        self.prominence_input.setMaximum(0.5)  # Set maximum value
        self.prominence_input.setValue(PROMINENCE)
        self.prominence_input.valueChanged.connect(self.prominence_changed)

        h_layout.addWidget(self.prominence_input)
        h_layout.addStretch()

        peak_finder_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        self.peaks_btn = QPushButton("Find Peaks")
        self.peaks_btn.clicked.connect(self.peaks_clicked)
        h_layout.addWidget(self.peaks_btn)
        h_layout.addStretch()

        peak_finder_layout.addLayout(h_layout)

        # Layout per i parametri del call duration
        call_duration_layout = QHBoxLayout()

        call_duration_layout.addWidget(QLabel("Call duration Parameters"))

        call_duration_layout.addWidget(QLabel("S/N"))

        self.signal_noise_ration_input = QDoubleSpinBox()
        self.signal_noise_ration_input.setDecimals(1)
        self.signal_noise_ration_input.setSingleStep(0.1)
        self.signal_noise_ration_input.setMinimum(0.1)
        self.signal_noise_ration_input.setMaximum(10)
        self.signal_noise_ration_input.setValue(SIGNAL_TO_NOISE_RATIO)
        self.signal_noise_ration_input.valueChanged.connect(
            self.signal_to_noise_ratio_changed
        )

        call_duration_layout.addWidget(self.signal_noise_ration_input)
        call_duration_layout.addStretch()

        # Layout per i parametri dello spettro
        spectrum_layout = QVBoxLayout()

        spectrum_layout.addWidget(QLabel("Spectrum Parameters"))

        h_layout = QHBoxLayout()

        # FFT LENGTH
        h_layout.addWidget(QLabel("FFT Length"))

        self.fft_length_input = QSpinBox()
        self.fft_length_input.setMinimum(10)
        self.fft_length_input.setMaximum(10000)
        self.fft_length_input.setValue(FFT_LENGTH)
        self.fft_length_input.setSingleStep(10)
        self.fft_length_input.valueChanged.connect(self.fft_length_changed)

        h_layout.addWidget(self.fft_length_input)

        # FFT OVERLAP
        h_layout.addWidget(QLabel("Overlap"))

        self.fft_overlap_input = QSpinBox()
        self.fft_overlap_input.setMinimum(10)
        self.fft_overlap_input.setMaximum(10000)
        self.fft_overlap_input.setValue(FFT_OVERLAP)
        self.fft_overlap_input.setSingleStep(10)
        self.fft_overlap_input.valueChanged.connect(self.fft_overlap_changed)

        h_layout.addWidget(self.fft_overlap_input)
        h_layout.addStretch()

        spectrum_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        self.spectrum_btn = QPushButton("Compute Spectrum")
        self.spectrum_btn.clicked.connect(self.spectrum_clicked)
        h_layout.addWidget(self.spectrum_btn)
        h_layout.addStretch()

        spectrum_layout.addLayout(h_layout)

        # Layout principale
        main_layout = QVBoxLayout()

        main_layout.addLayout(envelope_layout)
        main_layout.addSpacing(10)

        main_layout.addLayout(peak_finder_layout)
        main_layout.addSpacing(10)

        main_layout.addLayout(call_duration_layout)
        main_layout.addSpacing(10)

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

        # carica dati da JSON
        h_layout = QHBoxLayout()
        self.json_btn = QPushButton("Carica da Json")
        self.json_btn.clicked.connect(self.leggi_json)
        h_layout.addWidget(self.json_btn)
        h_layout.addStretch()
        main_layout.addLayout(h_layout)
        self.setLayout(main_layout)
    
    def leggi_json(self):
        print("DEBUG leggi_json file:", __file__, flush=True)
        start_dir = str(getattr(self.main, "json_root", ""))
        json_file, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona file JSON",
            start_dir,         
            "JSON files (*.json);All files (*.*)"
        )

        if not json_file:
            return

        json_path = Path(json_file)

        self.main.input_json_path = json_path
        self.main.output_json_path = json_path.with_name(json_path.stem + "_out2.json")

        # crea output json se manca
        if not self.main.output_json_path.exists():
            with open(self.main.input_json_path, "r", encoding="utf-8") as f:
                base = json.load(f)
            with open(self.main.output_json_path, "w", encoding="utf-8") as f:
                json.dump(base, f, indent=0, ensure_ascii=False)

        with open(json_path, "r", encoding="utf-8") as f:
            parameters = json.load(f)

        # === PRE-PASS: calcola intensità dei picchi di 1° livello per ogni WAV lungo ===
        for wav_key, wav_block in parameters.items():
            if not isinstance(wav_block, dict):
                continue

            # se non ci sono picchi di livello 1, salta
            if "peaks_times" not in wav_block:
                continue

            # se già calcolato, salta (togli questo if se vuoi forzare ricalcolo)
            if "peaks_int_level1" in wav_block:
                continue

            # path del WAV lungo: stessa cartella dei WAV corti (e del JSON)
            wav_path = json_path.parent / wav_key
            if not wav_path.is_file():
                print(f"[WARN] WAV lungo non trovato: {wav_path}")
                continue

            # 1) carica wav lungo
            self.main.load_wav(str(wav_path))

            # 2) applica parametri envelope del livello 1
            if "window size" in wav_block:
                self.main.window_size = int(wav_block["window size"])
            if "overlap" in wav_block:
                self.main.overlap = int(wav_block["overlap"])

            # 3) imposta i picchi di 1° livello nei campi che le tue funzioni usano
            self.main.peaks_times = np.array(wav_block.get("peaks_times", []), dtype=float)

            # 4) calcola envelope + intensità

            #<------------------------------         cambiamento provvisorio
            self.window_size = WINDOW_SIZE
            self.overlap = OVERLAP
            
            self.main.envelope()
            self.main.trova_intensita_picchi()

            # 5) salva nel JSON (chiave non ambigua)
            wav_block["peaks_int_level1"] = self.main.peaks_int.tolist()
            # ricostruisci il wav_block con ordine desiderato
            new_block = {}

            for k, v in wav_block.items():
                new_block[k] = v
                if k == "peaks_times":
                    new_block["peaks_int_level1"] = self.main.peaks_int.tolist()

            # sostituisci il blocco originale
            parameters[wav_key] = new_block


        # salva subito su output json le intensità di livello 1 calcolate
        with open(self.main.output_json_path, "w", encoding="utf-8") as f_out:
            json.dump(parameters, f_out, indent=0, ensure_ascii=False)


        
        # 1) Mappa dei picchi a livello WAV (livello 1)
        wav_level_peaks_map = {}
        for wav_key, wav_block in parameters.items():
            if not isinstance(wav_block, dict):
                continue
            wav_level_peaks_map[wav_key] = np.array(wav_block.get("peaks_times", []), dtype=float)

        # 2) Costruisci i job dei SONG (livello 2)
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

        # 3) Inizializza Main
        self.main.song_jobs = song_jobs
        self.main.wav_file_list = wav_list
        self.main.current_job_index = 0

        # 4) Salva la mappa “livello WAV” dentro Main
        self.main.wav_level_peaks_map = wav_level_peaks_map

        # carica primo song
        self.main.load_job(0)


    def reset_values(self):
        """
        reset with default values
        """

        self.window_size_input.setValue(WINDOW_SIZE)
        self.overlap_input.setValue(OVERLAP)
        self.amp_threshold_input.setValue(MIN_AMPLITUDE)  # Amplitude threshold
        self.min_distance_input.setValue(MIN_DISTANCE)
        self.max_distance_input.setValue(MAX_DISTANCE)
        self.prominence_input.setValue(PROMINENCE)
        self.signal_noise_ration_input.setValue(SIGNAL_TO_NOISE_RATIO)
        self.fft_length_input.setValue(FFT_LENGTH)
        self.fft_overlap_input.setValue(FFT_OVERLAP)

    def window_size_changed(self, new_value):
        self.main.window_size = new_value
        self.main.envelope()

    def overlap_changed(self, new_value):
        self.main.overlap = new_value
        self.main.envelope()

    def min_amplitude_changed(self, new_value):
        self.main.min_amplitude = new_value
        self.peaks_clicked()

    def min_distance_changed(self, new_value):
        self.main.min_distance = new_value
        self.peaks_clicked()

    def max_distance_changed(self, new_value):
        self.main.max_distance = new_value
        self.peaks_clicked()

    def prominence_changed(self, new_value):
        self.main.prominence = new_value
        self.peaks_clicked()

    def signal_to_noise_ratio_changed(self, new_value):
        self.main.signal_to_noise_ratio = new_value
        self.main.trova_ini_fin()

    def fft_length_changed(self, new_value):
        self.main.fft_length = new_value
        self.main.plot_spectrum()

    def fft_overlap_changed(self, new_value):
        self.main.fft_overlap = new_value
        self.main.plot_spectrum()

    def envelope_clicked(self):
        try:
            self.main.envelope()
        except Exception as e:
            print("Errore nei parametri envelope:", e)

    def peaks_clicked(self):
        try:
            self.main.trova_picchi()
        except Exception as e:
            print("Errore nei parametri peak finder:", e)

    def spectrum_clicked(self):
        try:
            self.main.plot_spectrum()
        except Exception as e:
            print("Errore nei parametri spectrum:", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Crea la finestra dei plots e quella dei controlli

    main = Main(wav_file_list = []
           )
    #wav_file_list=[r"C:\Users\Sergio\audio_analysis\GeCorn_2025-01-25_09\GeCorn_2025-01-25_09_000000000_002328479_sample_000915075.wav"]

    main.show()
    
    QTimer.singleShot(0, main.control_panel.leggi_json)

    sys.exit(app.exec())
