import importlib.util
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QTextEdit, QFileDialog, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider, SpanSelector
import librosa

__version__ = "0.0.1"
__version_date__ = "2025-02-14"

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
        
        

class ResamplingWindow(QWidget):
    def __init__(self, wav_file):
        super().__init__()

        self.setWindowTitle("Resampling Audio")
        self.setGeometry(200, 200, 400, 250)
        
        self.wav_file = wav_file  # Salva il file WAV passato dal MainWindow
        
        # Carica il file WAV e ottiene le informazioni
        self.sampling_rate, self.data = wavfile.read(self.wav_file)
        self.duration = len(self.data) / self.sampling_rate

        # Layout principale
        layout = QVBoxLayout()

        # Etichetta con informazioni sul file
        self.label_info = QLabel(f"File WAV selezionato: {self.wav_file}\n"
                                 f"Durata: {self.duration:.2f} sec\n"
                                 f"Frequenza di campionamento: {self.sampling_rate} Hz")
        layout.addWidget(self.label_info)

        # Menu a tendina per selezionare il nuovo sampling rate
        self.combo_sampling_rate = QComboBox()
        self.combo_sampling_rate.addItems(["11000", "22050", "44100", "48000", "96000"])
        layout.addWidget(QLabel("Seleziona nuova frequenza di campionamento:"))
        layout.addWidget(self.combo_sampling_rate)

        # Pulsante per applicare il resampling
        self.button_resample = QPushButton("Applica Resampling")
        self.button_resample.clicked.connect(self.apply_resampling)
        layout.addWidget(self.button_resample)

        # Pulsante per salvare il file resamplato
        self.button_save = QPushButton("Salva WAV")
        self.button_save.setEnabled(False)  # Inizialmente disabilitato fino a quando il resampling non è stato applicato
        self.button_save.clicked.connect(self.save_wav)
        layout.addWidget(self.button_save)

        self.setLayout(layout)

    def apply_resampling(self):
        """Applica il resampling ai dati audio"""
        new_sampling_rate = int(self.combo_sampling_rate.currentText())

        print(f"DEBUG: Resampling da {self.sampling_rate} Hz a {new_sampling_rate} Hz...")
        self.data_resampled = librosa.resample(self.data, orig_sr = self.sampling_rate, target_sr = new_sampling_rate)
        plt.plot(self.data_resampled)
        # Converte il risultato in formato int16 per la scrittura su WAV
        self.resampled_data = (self.data_resampled * 32767).astype(np.int16)

        self.new_sampling_rate = new_sampling_rate
        self.button_save.setEnabled(True)  # Abilita il salvataggio

        print(f"DEBUG: Resampling completato. Nuova lunghezza: {len(self.data_resampled)} campioni.")

    def save_wav(self):
        """Salva il file WAV dopo il resampling"""
        save_path, _ = QFileDialog.getSaveFileName(self, "Salva file WAV", "", "WAV Files (*.wav)")
        if save_path:
            wavfile.write(save_path, self.new_sampling_rate, self.data_resampled)
            print(f"DEBUG: File salvato correttamente in {save_path}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Analysis")
        self.setGeometry(100, 100, 600, 400)
        self.wav_file = None  # Attributo per memorizzare il file WAV caricato
        
        # Load modules from plugins directory
        self.modules: dict = {}
        for file_ in sorted(Path("plugins").glob("*.py")):
            module_name = file_.stem  # python file name without '.py'
            spec = importlib.util.spec_from_file_location(module_name, file_)
            self.modules[module_name] = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = self.modules[module_name]
            spec.loader.exec_module(self.modules[module_name])

        # Editor di testo per output
        self.text_edit = QTextEdit(self)
        self.setCentralWidget(self.text_edit)

        # Creazione del menu
        menubar = self.menuBar()
        

        # Menu File
        file_menu = menubar.addMenu("File")
        open_wav_action = QAction("Open wav", self)
        open_wav_action.triggered.connect(self.open_wav)
        file_menu.addAction(open_wav_action)

        close_action = QAction("Close", self)
        close_action.triggered.connect(self.close_program)
        file_menu.addAction(close_action)

        # Menu Edit
        edit_menu = menubar.addMenu("Edit")
        show_oscillogram_action = QAction("Show Oscillogram", self)
        show_oscillogram_action.triggered.connect(self.show_oscillogram)
        edit_menu.addAction(show_oscillogram_action)

        resampling_action = QAction("Resampling", self)
        resampling_action.triggered.connect(self.resampling)
        edit_menu.addAction(resampling_action)
        
        # Menu Process
        process_menu = menubar.addMenu("Process")

        # Menu Analyse
        analyse_menu = menubar.addMenu("Analyse")
        
        
        # Aggiunta dei plugin al menu Analyse
        self.actions = []
        for module_name in self.modules:
            action = QAction(module_name, self)
            action.triggered.connect(self.run_option)
            analyse_menu.addAction(action)
            self.actions.append(action)
    
    def open_wav(self):
        print("DEBUG: La funzione open_wav() è stata chiamata.")  # Controllo immediato
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV Files (*.wav)")
        if file_path:
            self.wav_file = file_path
            self.text_edit.append(f"Selected file: {file_path}")
            print(f"DEBUG: File WAV caricato correttamente: {self.wav_file}")
            self.show_oscillogram()
        else:
            print("DEBUG: Nessun file WAV selezionato.")

    def close_program(self):
        print("ho chiamato la funzione close")
        self.close()
    
    def show_oscillogram(self):
        if self.wav_file:
            self.oscillogram_window = OscillogramWindow(self.wav_file)
            self.oscillogram_window.show()
        else:
            self.text_edit.append("No WAV file loaded. Please open a file first.")
    
    def resampling(self):
        """Apre la finestra di resampling"""
        if self.wav_file:
            self.resampling_window = ResamplingWindow(self.wav_file)  # Inizializza la finestra
            self.resampling_window.show()
        else:
            self.text_edit.append(" Nessun file WAV caricato! Apri un file prima di eseguire il resampling.")


    def run_option(self, module_name):
        """
        Carica il plugin e passa il file WAV se disponibile.
        """
        module_name = self.sender().text()
        print(f"running {module_name=}")

        self.text_edit.append(f"Running {module_name} plugin")
        self.w = self.modules[module_name].Main(self.wav_file)
        self.w.show()
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
