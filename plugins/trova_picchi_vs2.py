from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil
import json
from scipy.io import wavfile
from scipy.signal import find_peaks
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QPushButton,
    QDoubleSpinBox,
    QSplitter,
    QMessageBox,
    QGridLayout,
    QSlider
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider, SpanSelector
import librosa
import sounddevice as sd

class Main(QWidget):
    def __init__(self, wav_file_list: list):
        super().__init__()

        if wav_file_list:
            self.wav_file = wav_file_list[0]
        else:
            QMessageBox.critical(
                self,
                "",
                "No file WAV!",
            )
            return

        self.setWindowTitle(
            f"{Path(__file__).stem.replace('_', ' ')} - {Path(self.wav_file).stem}"
        )

    

        # Layout principale a griglia (10 colonne)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        
        
        self.widgets_riga1_2 = {
                "zoom_in": {
                    "type": "QPushButton",
                    "label": "Zoom +",
                    "height_label": 10,
                    "row": 1, "col": 0,
                    "row_span": 1, "col_span": 1,
                    "default": None,
                    "linked_fnc": "zoomIn_wav",
                    "widget": None
                },
                "zoom_out": {
                    "type": "QPushButton",
                    "label": "Zoom OUT",
                    "row": 1, "col": 1,
                    "row_span": 1, "col_span": 1,
                    "default": None,
                    "linked_fnc": "zoomOut_wav",
                    "widget": None
                },
                 "Envelope": {
                    "type": "QPushButton",
                    "label": "Envelope",
                    "row": 0, "col": 3,
                    "row_span": 2, "col_span": 1,
                    "default": None,
                    "linked_fnc": "envelope",
                    "widget": None
                        },
                "w_size": {
                    "type": "QLineEdit",
                    "label": "W Size (ms)",
                    "row": 1, "col": 4,
                    "row_span": 1, "col_span": 1,
                    "default": "100",
                    "widget": None
                },
                "w_overlap": {
                    "type": "QLineEdit",
                    "label": "W step (ms)",
                    "row": 1, "col": 5,
                    "row_span": 1, "col_span": 1,
                    "default": "10",
                    "widget": None
                },
                "Peaks": {
                    "type": "QPushButton",
                    "label": "Peaks",
                    "row": 0, "col": 7,
                    "row_span": 2, "col_span": 1,
                    "default": None,
                    "linked_fnc": "fd_peaks_button",
                    "widget": None
                },
                "min_amp": {
                    "type": "QDoubleSpinBox",
                    "label": "Min Amp",
                    "row": 1, "col": 8,
                    "row_span": 1, "col_span": 1,
                    "default": ["0.1", "3", "0.005", "0.0", "1"],
                    "linked_fnc": "fd_peaks_spinbox",
                    "widget": None
                },
                "min_dist": {
                    "type": "QLineEdit",
                    "label": "Min distance (s)",
                    "row": 1, "col": 9,
                    "row_span": 1, "col_span": 1,
                    "default": "0.300",
                    "widget": None
                },
            }
       
        self.widgets_rigaFinale = {
                "detect_call": {
                    "type": "QPushButton",
                    "label": "Detect Calls",
                    "height_label": 40,
                    "row": 5, "col": 0,
                    "row_span": 1, "col_span": 2,
                    "linked_fnc": "detect_calls",
                    "widget": None,
                    "default": None
                },
                "from_peak_to_end": {
                    "type": "QDoubleSpinBox",
                    "label": "Max dist (ms)",
                    "row": 5, "col": 2,
                    "row_span": 1, "col_span": 1,
                    "default": ["600", "0", "50", "10", "2000"],
                    "linked_fnc": "detect_calls",
                    "widget": None
                },
                "save_calls": {
                    "type": "QPushButton",
                    "label": "Save Calls",
                    "height_label": 40,
                    "row": 5, "col": 9,
                    "row_span": 1, "col_span": 3,
                    "linked_fnc": "save_calls",
                    "widget": None,
                    "default": None
                },
                "Play": {
                    "type": "QPushButton",
                    "label": "PLAY",
                    "height_label": 40,
                    "row": 5, "col": 6,
                    "row_span": 1, "col_span": 1,
                    "linked_fnc": "play_audio",
                    "widget": None,
                    "default": None
                },
                "Stop": {
                    "type": "QPushButton",
                    "label": "STOP",
                    "height_label": 40,
                    "row": 5, "col": 7,
                    "row_span": 1, "col_span": 1,
                    "linked_fnc": "stopplaying",
                    "widget": None,
                    "default": None
                }
            }
       

        for key, props in self.widgets_riga1_2.items():
            widget = None 
           # Creazione del widget corrispondente
            if props["type"] == "QPushButton":
                widget = QPushButton(props["label"])
                self.layout.addWidget(widget, props["row"], props["col"], props["row_span"], props["col_span"])
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(self, props["linked_fnc"])  # Recupera la funzione
                    widget.clicked.connect(linked_function)  # Connetti il pulsante alla funzione
                    
            elif props["type"] == "QLineEdit":
                widget = QLineEdit()
                label = QLabel(props["label"])
                self.layout.addWidget(label, props["row"] - 1, props["col"], 1, props["col_span"])
                label.setFixedHeight(10)
                self.layout.addWidget(widget, props["row"], props["col"], 1, props["col_span"])
                if props["default"] is not None:  
                    widget.setText(props["default"])
                
            elif props["type"] == "QDoubleSpinBox":
                widget = QDoubleSpinBox()
                label = QLabel(props["label"])
                self.layout.addWidget(label, props["row"] - 1, props["col"], 1, props["col_span"])
                self.layout.addWidget(widget, props["row"], props["col"], 1, props["col_span"])
                if props["default"] is not None:  
                    widget.setValue(float(props["default"][0]))
                    widget.setDecimals(float(props["default"][1]))
                    widget.setSingleStep(float(props["default"][2]))
                    widget.setMinimum(float(props["default"][3]))
                    widget.setMaximum(float(props["default"][4]))
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(self, props["linked_fnc"])  # Recupera la funzione
                    widget.valueChanged.connect(linked_function)  # Connetti il pulsante alla funzione
            else:
                continue  # Ignora eventuali tipi non definiti
            if widget:
                self.widgets_riga1_2[key]["widget"] = widget
  

        
        
        for key, props in self.widgets_rigaFinale.items():
            widget = None 
           # Creazione del widget corrispondente
            if props["type"] == "QPushButton":
                widget = QPushButton(props["label"])
                self.layout.addWidget(widget, props["row"], props["col"], props["row_span"], props["col_span"])
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(self, props["linked_fnc"])  # Recupera la funzione
                    widget.clicked.connect(linked_function)  # Connetti il pulsante alla funzione
                self.widgets_rigaFinale[key]["widget"] = widget
            elif props["type"] == "QLineEdit":
                widget = QLineEdit()
                label = QLabel(props["label"])
                self.layout.addWidget(label, props["row"] - 1, props["col"], 1, props["col_span"])
                label.setFixedHeight(10)
                self.layout.addWidget(widget, props["row"], props["col"], 1, props["col_span"])
                if props["default"] is not None:  
                    widget.setText(props["default"])
            
            elif props["type"] == "QDoubleSpinBox":
                widget = QDoubleSpinBox()
                label = QLabel(props["label"])
                self.layout.addWidget(label, props["row"] - 1, props["col"], 1, props["col_span"])
                self.layout.addWidget(widget, props["row"], props["col"], 1, props["col_span"])
                if props["default"] is not None:  
                    widget.setValue(float(props["default"][0]))
                    widget.setDecimals(float(props["default"][1]))
                    widget.setSingleStep(float(props["default"][2]))
                    widget.setMinimum(float(props["default"][3]))
                    widget.setMaximum(float(props["default"][4]))
                if "linked_fnc" in props and hasattr(self, props["linked_fnc"]):
                    linked_function = getattr(self, props["linked_fnc"])  # Recupera la funzione
                    widget.valueChanged.connect(linked_function)  # Connetti il pulsante alla funzione

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
            self.layout.setColumnStretch(col, 1)  # Tutte le colonne hanno lo stesso peso

    
        
        
        # 🔹 Creazione della figura matplotlib (grafico)
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.figure.subplots_adjust(bottom=0.2)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas, 2, 0, 1, 10)  # Riga 3, occupa tutte le colonne

        # 🔹 Slider per la navigazione nel tempo (Riga 3)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.on_slider)
        self.layout.addWidget(self.slider, 3, 0, 1, 10)  # Riga 4, occupa tutte le colonne
        

        # 🔹 Attivazione di SpanSelector per la selezione
        self.selected_region = []
        self.span_selector = SpanSelector(
            self.ax, self.on_select, "horizontal",
            useblit=True, props=dict(alpha=0.5, facecolor="red")
        )


        self.load_wav(self.wav_file)
        self.plot_wav(self.xmin, self.xmax)
        
    
    def load_wav(self, wav_file):
        """
        Load WAV file and extract data
        """
        self.amp_rms = 1.5 # fattore di moltiplicazione dell'inviluppo, per accentuare variazione in ampiezza
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
        self.ax.plot(self.time, self.binary_song, linewidth=0.5, color="black", alpha=0.5)
        
        if len(self.rms) > 0:
            self.ax.plot(self.rms_times, self.rms, linewidth=1, color="red")
            
        if len(self.peaks_times) > 0:
            #print("         ", self.peaks_times)
            for i in np.arange(len(self.peaks_times)):
                self.ax.plot(
                    [self.peaks_times[i], self.peaks_times[i]],
                    [0, np.max(self.rms)],
                    "-y",
                    linewidth=2,
                    )
        self.ax.set_xlim(self.zmin, self.zmax)
        if self.xmin > 0 or self.xmax < self.duration:
            self.ax.axvspan(self.xmin, self.xmax, color = "yellow", alpha = 0.3)
        self.canvas.draw()

    def show_message(self, button_name):
        """Mostra un messaggio di avviso con il nome del pulsante premuto."""
        QMessageBox.warning(self, "Non Implementato", f"La funzione {button_name} non è ancora implementata.")

    def on_select(self, xmin, xmax):
        """Evidenzia l'area selezionata con il mouse e aggiorna xmin e xmax."""
        if xmin < 0:
            xmin = 0
            xmax = 0
        if abs(xmax - xmin) < 0.01:  # Evita selezioni troppo piccole
            return  
        self.xmin = xmin
        self.xmax = xmax
        for patch in self.ax.patches[:]:
            patch.remove()
        self.ax.axvspan(self.xmin, self.xmax, color = "yellow", alpha = 0.3)
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
            
            self.window_size = int(self.widgets_riga1_2["w_size"]["widget"].text()) 
            self.window_size = int(self.window_size * self.sampling_rate/1000)
            self.overlap = int(self.widgets_riga1_2["w_overlap"]["widget"].text())
            self.overlap = int(self.overlap * self.sampling_rate/1000)

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
        if np.any(mask_in):  #Se ci sono picchi nell'intervallo, li rimuove
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

        amp_threshold = self.widgets_riga1_2["min_amp"]["widget"].value()
        min_distance_sec = np.float64(self.widgets_riga1_2["min_dist"]["widget"].text())
        min_distance_samples = int(
                                min_distance_sec * (self.sampling_rate / self.overlap)
                                )  # Converti in campioni

        mask_rms = (self.rms_times >= xmin) & (self.rms_times <= xmax)
        rms_selected = self.rms[mask_rms]
        rms_times_selected = self.rms_times[mask_rms]
            
        # Trova i picchi nell'inviluppo RMS
        peaks, _ = find_peaks(
                rms_selected,
                height=amp_threshold,
                distance=min_distance_samples,
                prominence=0.01,
                )
        new_peaks_times = rms_times_selected[peaks]
        self.peaks_times = np.sort(np.concatenate((self.peaks_times, new_peaks_times)))
        self.plot_wav(self.zmin, self.zmax)
    
    def detect_calls(self):
        """Identifica i canti basandosi su picchi e inviluppo"""
        if len(self.peaks_times) == 0:
            return
        start_times = []
        end_times = []
        before = int(self.widgets_rigaFinale["from_peak_to_end"]["widget"].text()) 
        before = int(before * self.sampling_rate / (self.overlap * 1000))
        after = before 
        print("after & Before", after)
        print("len(rms)", len(self.rms))
        # Trova la posizione di ogni picco in `rms_times`
        rms_peaks = np.searchsorted(self.rms_times, self.peaks_times)
        print("rms",rms_peaks)
        for ic in range(len(rms_peaks)):
            s_min = 1  
            e_min = 1
            id0 = max(0, rms_peaks[ic] - before)
            s_idmin = id0

            for ss in range(before):
                c = id0 + ss
                if c >= len(self.rms_times) - 2:
                    break
                x = np.sum(self.rms[c] - self.rms[c + 1 : min(c + before, len(self.rms))])
                if x < 0:
                    s_min = x
                    s_idmin = c

            start_times.append(self.rms_times[s_idmin])

            id1 = rms_peaks[ic] + 1
            e_idmin = id1
            
            for ee in range(after):
                e = id1 + ee
                x = np.sum(self.rms[e] - self.rms[e + 1 : e + after])
                
                if x < e_min:
                    e_min = x
                    e_idmin = e
                    print("e_min", e_min)
                    break

            end_times.append(self.rms_times[e_idmin])
        #print("ss", start_times)
        #print("ee", end_times)
        self.start_times = start_times
        self.end_times = end_times
        self.binary_song = np.zeros_like(self.time)
        for start, end in zip(start_times, end_times):
            mask = (self.time >= start) & (self.time <= end)
            self.binary_song[mask] = 1  

        self.plot_wav(self.xmin, self.xmax)

    def save_calls(self):
        """
        Salva i segmenti audio attorno ai picchi selezionati in file separati.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        parent_directory = Path(directory)

        data_directory = parent_directory / Path(self.wav_file).stem

        if data_directory.is_dir():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"The directory {data_directory} already exists!")
            msg.setWindowTitle("Warning")

            msg.addButton("Erase files", QMessageBox.YesRole)
            msg.addButton("Cancel", QMessageBox.YesRole)

            msg.exec()

            match msg.clickedButton().text():
                case "Erase files":
                    shutil.rmtree(data_directory)
                case "Cancel":
                    return

        data_directory.mkdir(exist_ok=True)  # Crea la cartella se non esiste

        # save data
        #self.save_parameters(data_directory / "data.json")

        print(f"Salvando i canti nella cartella: {data_directory}")

        for i in np.arange(len(self.peaks_times)):
            # Calcola l'inizio e la fine del ritaglio
            ini = int(self.start_times[i] * self.sampling_rate)
            fine = int(self.end_times[i] * self.sampling_rate)

            # 🔹 Verifica che l'intervallo sia valido
            if ini < 0 or fine > len(self.data):
                print(
                    f"Il picco {self.peaks_times[i]:.5f}s supera i limiti del file audio! Saltato."
                )
                continue

            ritaglio = self.data[ini:fine]

            # Crea il nome del file con il numero di campione
            sample_number = int(self.peaks_times[i] * self.sampling_rate)
            nome_ritaglio = (
                data_directory
                / f"{Path(self.wav_file).stem}_sample_{sample_number:09d}.wav"
            )

            # Salva il file ritagliato
            wavfile.write(nome_ritaglio, self.sampling_rate, ritaglio)
            print(f"Salvato: {nome_ritaglio}")
    
 
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
    main_widget = Main(wav_file_list=["GeCorn_2025-01-25_09.wav"])
    #main_widget = Main(wav_file_list=["Blommersia_blommersae.wav"])
    main_widget.show()

    sys.exit(app.exec())