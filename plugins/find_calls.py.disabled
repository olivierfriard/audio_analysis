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
    QPushButton
    )

from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider, SpanSelector, Button, TextBox
import librosa

class OscillogramWindow(QWidget):
    def __init__(self, wav_file):
        super().__init__()
        self.setWindowTitle("Oscillogram")
        self.setGeometry(200, 200, 800, 500)

        self.wav_file = wav_file
        self.sampling_rate, self.data = wavfile.read(self.wav_file)
        self.duration = len(self.data) / self.sampling_rate
        
        self.time = np.linspace(0, self.duration, num=len(self.data))
        
        # Layout principale
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Creazione della figura matplotlib
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Disegna il grafico iniziale
        self.ax.plot(self.time, self.data, linewidth=0.5, color="black")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Oscillogram")
        #self.ax.grid()

        # Imposta il range iniziale della finestra
        self.xmin, self.xmax = 0, self.duration  # Inizialmente mostra massimo 3 secondi
        self.xrange = self.xmax - self.xmin
        self.ax.set_xlim(self.xmin, self.xmax)
        
        # Crea lo slider
        self.slider_ax = self.figure.add_axes([0.2, 0.05, 0.65, 0.03])
        self.slider = Slider(self.slider_ax, "Time", 0, 1, valinit=self.xmax / self.duration)
        self.slider.on_changed(self.on_slider)
        
        self.canvas.draw()

        # Aggiunta della selezione interattiva
        self.span_selector = SpanSelector(
            self.ax, self.on_select, "horizontal", useblit=True, props=dict(alpha=0.5, facecolor='red')
        )

    def on_select(self, xmin, xmax):
        """Aggiorna il plot con la selezione dell'utente e sincronizza lo slider."""
        if xmax - xmin < 0.01:
            self.xmin = 0
            self.xmax = self.duration
        else:
            self.xmin, self.xmax = xmin, xmax
        self.ax.set_xlim(self.xmin, self.xmax)
        self.range = self.xmax - self.xmin

        self.slider.set_val(self.xmax/self.duration)
        self.canvas.draw_idle()

    def on_slider(self, val):
        """Aggiorna la vista dell'oscillogramma in base alla posizione dello slider mantenendo la durata selezionata."""
        
        self.xmax = max(self.range,val * self.duration) 
        self.xmin = self.xmax - self.range
        self.ax.set_xlim(self.xmin, self.xmax)
        self.canvas.draw_idle()
        
        
class Main(QWidget):
    def __init__(self, wav_file = None):
        super().__init__()

        if not wav_file:
            print("SONO QUI")
            self.wav_file = "C:\\Users\\Sergio\\audio_analysis\\GeCorn_2025-01-19_01.wav"
        else:
            self.wav_file = wav_file
        print(self.wav_file)
        self.load_wav(self.wav_file)
        
        self.window_size = 1024
        self.overlap = 512
        self.amp_threshold = 0
        self.min_dist = 0
        self.rms = np.zeros(self.window_size//self.overlap)
        n_frames = np.arange(len(self.rms))
        self.rms_times = librosa.frames_to_time(n_frames, sr=self.sampling_rate, hop_length=self.overlap)
        self.peaks_times = np.array([])
        
        self.setWindowTitle("Selezione Interattiva")
        self.setGeometry(100, 100, 800, 500)

        # Creazione layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Crea layout orizzontale
        self.hbox_layout = QHBoxLayout()

        # Creazione della figura matplotlib
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(bottom=0.2)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)  # IMPORTANTE: Aggiungere il canvas alla finestra
        
        # Attivazione di SpanSelector
        self.span_selector = SpanSelector(self.ax, self.on_select, "horizontal", useblit=True, props=dict(alpha=0.5, facecolor='red'))

        self.canvas.draw_idle()

        # Creazione delle caselle Window_size
        window_size_layout, self.window_size_input = self.create_label_input_pair("Window size", self.window_size)
        window_overlap_layout, self.window_overlap_input = self.create_label_input_pair("Overlap", self.overlap)
        window_amp_threshold_layout, self.amp_threshold_input = self.create_label_input_pair("Amplitude threshold", self.amp_threshold)
        
        window_min_distance_layout, self.min_distance_input = self.create_label_input_pair("Minimum distance (points)", self.min_dist)
        
        # Crea il pulsante calcola_envelope
        self.run_envelope = QPushButton("Run evelope")
        self.run_envelope.setFixedSize(100, 30)  # Imposta una dimensione fissa
        self.run_envelope.clicked.connect(self.envelope)

        # Crea il pulsante amp_threshold
        self.manual_amp_threshold = QPushButton("Setting")
        self.manual_amp_threshold.setFixedSize(100, 30)  # Imposta una dimensione fissa
        self.manual_amp_threshold.clicked.connect(self.set_amp_threshold)

         # Crea il pulsante min distance
        self.manual_min_dist = QPushButton("Setting")
        self.manual_min_dist.setFixedSize(100, 30)  # Imposta una dimensione fissa
        self.manual_min_dist.clicked.connect(self.set_min_distance)

        # Aggiungo le coppie al layout orizzontale
        self.hbox_layout.addLayout(window_size_layout)
        self.hbox_layout.addLayout(window_overlap_layout)
        self.hbox_layout.addWidget(self.run_envelope)
        self.hbox_layout.addLayout(window_amp_threshold_layout)
        self.hbox_layout.addWidget(self.manual_amp_threshold)
        self.hbox_layout.addLayout(window_min_distance_layout)
        self.hbox_layout.addWidget(self.manual_min_dist)
        
        self.layout.addLayout(self.hbox_layout)
        self.plot_wav(self.xmin, self.xmax)
                
    
    def create_label_input_pair(self,label_text, default_value):
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
        self.xmax = len(self.data)/self.sampling_rate
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

        

    def plot_wav(self,xmin,xmax):
        self.ax.cla()  # Cancella il grafico precedente
        self.xmin = xmin
        self.xmax = xmax
        
        self.id_xmin = int(self.xmin * self.sampling_rate)
        self.id_xmax = int(self.xmax * self.sampling_rate)
        
        time = self.time[self.id_xmin:self.id_xmax]
        data = self.data[self.id_xmin:self.id_xmax]

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
        self.ax.plot(time, data, linewidth=0.5, color="black", alpha = 0.25)
        self.ax.plot(rms_times_selected, rms_selected, linewidth=1, color="red")
        if len(peaks_selected) > 0:
            for i in np.arange(len(peaks_selected)):
                self.ax.plot([peaks_selected[i], peaks_selected[i]], [0, np.max(rms_selected)], '-g', linewidth = 1)
        self.ax.plot()
        # Aggiorna il grafico
        self.canvas.draw()

    def on_select(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        if self.xmax - self.xmin < 0.01:
            self.xmin = 0
            self.id_xmin = 0
            self.xmax = len(self.data)/self.sampling_rate
            self.id_xmax = len(self.data)
        else:
            self.xmin = xmin
            self.id_xmin = int(self.xmin * self.sampling_rate)
            self.xmax = xmax
            self.id_xmax = int(self.xmax * self.sampling_rate)
        self.plot_wav(self.xmin,self.xmax)

       
    def envelope(self, event = None):
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
            self.rms = librosa.feature.rms(y=self.data, 
                                        frame_length=self.window_size, 
                                        hop_length=self.overlap)[0]
            self.rms_times = librosa.frames_to_time(np.arange(len(self.rms)), 
                                                                sr=self.sampling_rate, 
                                                                hop_length=self.overlap)
            
            self.plot_wav(self.xmin,self.xmax)
            

        except ValueError:
            print("Errore: Assicurati che Window Size e Overlap siano numeri interi validi.")


    def set_amp_threshold(self):
        """Attiva la modalità di selezione di un punto sul grafico."""
        print("📌 Modalità selezione attivata: fai clic sul grafico per scegliere un punto.")
        if hasattr(self, "mouse_event_id"):
            self.canvas.mpl_disconnect(self.mouse_event_id)
        self.mouse_event_id = self.canvas.mpl_connect("button_press_event", self.get_point_coordinates)


    def get_point_coordinates(self, event):
        """Recupera le coordinate del punto cliccato sul grafico."""
        if event.inaxes is not None:  # Controlla che il clic sia all'interno del grafico
            y_coord = event.ydata
            self.amp_threshold = y_coord
            self.amp_threshold_input.setText(f"{self.amp_threshold:.5f}")  # Scrive nella casella di testo
        if hasattr(self, "mouse_event_id"):
            self.canvas.mpl_disconnect(self.mouse_event_id)  
            del self.mouse_event_id  # Rimuove l'ID dell'evento per evitare problemi
            
            

           
    def set_min_distance(self):
        """Imposta la distanza trascinando il mouse"""
        #self.span_selector = SpanSelector(self.ax, self.on_select, "horizontal", useblit=True, props=dict(alpha=0.5, facecolor='red'))
        self.min_distance_input.text()
        self.trova_picchi()
    
    def trova_picchi(self):
        """Trova i picchi dell'inviluppo RMS e li converte nei campioni della registrazione originale."""
        try:
            min_distance_sec = float(self.min_distance_input.text())  # Distanza in secondi
            print(min_distance_sec)
            min_distance_samples = int(min_distance_sec * (self.sampling_rate / self.overlap))  # Converti in campioni
            print(min_distance_samples, self.sampling_rate, self.overlap )
            amp_threshold = float(self.amp_threshold_input.text())  # Soglia di ampiezza

            print(f"🔍 Cercando picchi con:")
            print(f"   - Soglia di ampiezza: {amp_threshold:.5f}")
            print(f"   - Distanza minima tra picchi: {min_distance_sec:.5f} sec ({min_distance_samples} campioni)")

            # Trova i picchi nell'inviluppo RMS
            peaks, properties = find_peaks(self.rms, height=amp_threshold, distance=min_distance_samples, prominence=0.01)

            # 🔹 Converti gli indici nei campioni effettivi dell'audio originale
            peaks_original = peaks * self.overlap  # Campioni effettivi
            self.peaks_times = peaks * self.overlap / self.sampling_rate  # In secondi

            print(f" {len(peaks)} picchi trovati")
            print(f"   - Indici nell'inviluppo: {peaks}")
            print(f"   - Campioni originali: {peaks_original}")
            print(f"   - Posizioni in secondi: {self.peaks_times}")

            self.plot_wav(self.xmin, self.xmax)

        except ValueError:
            print("❌ Errore: Inserisci valori numerici validi per la distanza e la soglia.")




if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_widget = Main(wav_file='')
    main_widget.show()

    sys.exit(app.exec())
