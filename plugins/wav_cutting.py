import importlib.util
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QWidget
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider, SpanSelector, Button, TextBox
import librosa

class Main(QWidget):
    def __init__(self, wav_file=None):
        super().__init__()
        if not wav_file:
            self.wav_file = "C:\\Users\\Sergio\\audio_analysis\\GeCorn_2025-01-19_01.wav"
        else:
            self.wav_file = wav_file
        
                        
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
        
        # Pulsante seleziona cartella madre
        self.button_select = QPushButton("Scegli Cartella Madre", self)
        self.button_select.clicked.connect(self.select_folder)
        layout.addWidget(self.button_select)

        # **Aggiunta di un titolo alla casella di testo**
        self.label_durata = QLabel("Durata ritaglio (secondi):")
        layout.addWidget(self.label_durata)

        # **Casella di testo per inserire la durata del ritaglio**
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("60")  # Valore predefinito
        self.text_input.textChanged.connect(self.update_label)
        layout.addWidget(self.text_input)

        # **Etichetta per mostrare il valore della durata**
        self.label_selected_duration = QLabel("Durata selezionata: 60 sec")
        layout.addWidget(self.label_selected_duration)

        # Pulsante per salvare i file ritagliati
        self.button_save = QPushButton("Salva i files ritagliati", self)
        self.button_save.clicked.connect(self.save_files)
        layout.addWidget(self.button_save)
        self.button_save.setEnabled(False)  # Disabilitato se sottocartella non è stata ancora selezionata


        # Variabile per salvare la cartella selezionata
        self.selected_folder = None
        self.durata_ritaglio = 60  # Durata predefinita

        self.setLayout(layout)
    
    def update_label(self, text):
        """Aggiorna l'etichetta con la durata scelta"""
        try:
            self.durata_ritaglio = float(text)  
            self.label_selected_duration.setText(f"Durata selezionata: {self.durata_ritaglio} sec")
        except ValueError:
            self.label_selected_duration.setText("⚠️ Inserire un numero valido")

    def select_folder(self):
        """Apre il file dialog per selezionare una cartella e la salva in self.selected_folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Seleziona la cartella di origine")
        if folder_path:  # Controlla che l'utente non abbia annullato la selezione
            self.selected_folder = Path(folder_path)
            print(f"DEBUG: Cartella selezionata -> {self.selected_folder}")

            # Creo la sottocartella basata sul nome del file WAV
            self.nome_subcartella = self.selected_folder / Path(self.wav_file).stem
            self.nome_subcartella.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Sottocartella creata -> {self.nome_subcartella}")
            self.button_save.setEnabled(True)  # Abilita il pulsante solo dopo la creazione della sottocartella
    
    def save_files(self):
        """Salva i ritagli assicurandosi che il taglio avvenga dove il segnale è minimo"""
        original_name = f"{Path(self.nome_subcartella) / Path(self.wav_file).stem}"
        ini = 0
        counter = 0  # per tenere traccia del numero di ritagli salvati
        while ini < len(self.data):
            # Calcolo della fine teorica del segmento di durata self.durata_ritaglio
            fin = int(ini + self.sampling_rate * self.durata_ritaglio)
            if fin > len(self.data):
                fin = len(self.data)
            
            # Definisco l'intervallo ±0.1 secondi attorno al punto fin
            offset = int(self.sampling_rate * 0.1)
            start_range = max(fin - offset, 0)
            end_range = min(fin + offset, len(self.data))
            fin_range = np.arange(start_range, end_range)
            
            # Calcolo del RMS nel range definito
            frame_length = int(self.sampling_rate / 100)
            hop_length = int(self.sampling_rate / 100)
            rms = librosa.feature.rms(y=self.data[fin_range],
                                    frame_length=frame_length,
                                    hop_length=hop_length)[0]
            
            # Individuo l'indice in cui il valore RMS è minimo
            min_index = np.argmin(rms)
            fin_best = fin_range[min_index]
            
            # Costruisco il nome del file per il ritaglio corrente
            nome_ritaglio = f"{original_name}_{ini}_{fin_best - 1}.wav"
            print(nome_ritaglio)
            
            # Evito un eventuale loop infinito: se il nuovo punto di taglio
            # non fa avanzare l'indice, interrompo il ciclo
            if fin_best <= ini:
                break
            
            # Ritaglio la porzione dal segnale e la salvo
            ritaglio = self.data[ini:fin_best]
            wavfile.write(nome_ritaglio, self.sampling_rate, ritaglio)
                        
            # Aggiorno ini per il prossimo ritaglio
            ini = fin_best
            counter += 1

    
if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_widget = Main(wav_file='')
    main_widget.show()

    sys.exit(app.exec())
