import importlib.util
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave
import sounddevice as sd
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTextEdit,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QCheckBox,
    QLabel,
    QComboBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
    QDialog,
    QGridLayout,
    QSlider
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider, SpanSelector
import librosa


__version__ = "0.0.4"
__version_date__ = "2025-03-11"


class OscillogramWindow(QWidget):
    def __init__(self, wav_file: str):
        super().__init__()
        self.setWindowTitle(f"Oscillogram for {Path(wav_file).stem}")
        # self.setGeometry(200, 200, 800, 500)

        self.sampling_rate, self.data = wavfile.read(wav_file)
        self.duration = len(self.data) / self.sampling_rate

        self.time = np.linspace(0, self.duration, num=len(self.data))

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
        self.layout.addWidget(self.slider,2,1)

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


        # Pulsante STOP
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stopplaying)

        # Pulsante "Amplify"
        self.amplify_button = QPushButton("Amplify")
        self.amplify_button.clicked.connect(self.open_amplify_dialog)

        # Pulsante "Riproduci"
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_dialog)

   
        # Disegna il grafico iniziale
        self.ax.plot(self.time, self.data, linewidth=0.5, color="black")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title(f"Oscillogram for {Path(wav_file).stem}")
        # self.ax.grid()

        # Imposta il range iniziale della finestra
        self.xmin, self.xmax = (
            0,
            self.duration,
        )  # Inizialmente mostra tutta la registrazione
        self.xrange = self.xmax - self.xmin
        self.ax.set_xlim(self.xmin, self.xmax)


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
        self.layout.addWidget(self.stop_button, 0, 3, 1, 1)
        self.layout.addWidget(self.amplify_button, 0, 4, 1, 1)  
        self.layout.addWidget(self.canvas, 1, 0, 1, 5) 
        self.layout.addWidget(self.slider, 2, 0, 1, 5) 

        # Configurazione del comportamento delle colonne
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
        self.layout.setColumnStretch(3, 1)
        self.layout.setColumnStretch(4, 1)

        # Configurazione del comportamento delle righe
        self.layout.setRowStretch(0, 1)  # Pulsanti (meno spazio)
        self.layout.setRowStretch(1, 5)  # Plot (più spazio)
        self.layout.setRowStretch(2, 1)  # Slider (meno spazio)

        self.canvas.draw()

    def on_select(self, xmin, xmax):
        """Evidenzia l'area selezionata con il mouse."""
        min_width = 0.01
        if abs(xmax - xmin) < min_width:
            return 
        if self.selected_region:
            self.selected_region.remove()  # Rimuove l'area precedente
        self.selected_region = self.ax.axvspan(xmin, xmax, color="red", alpha=0.3)
        self.xmin, self.xmax = xmin, xmax
        self.canvas.draw_idle()
        self.zoomIn_button.setEnabled(True)

    
    def zoomIn_wav(self):
        if hasattr(self, "xmin") and hasattr(self, "xmax"):
            range = self.xmax - self.xmin
            self.slider.setValue(int((self.xmin / (self.duration - range)) * 100))
            self.ax.set_xlim(self.xmin, self.xmax)
        
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
        self.ax.set_xlim(self.xmin, self.xmax)  # Applica il reset
        self.slider.setValue(100)  # Imposta lo slider al massimo (100%)
        self.canvas.draw_idle()  # Aggiorna il grafico
        self.zoomIn_button.setEnabled(False)
        self.zoomOut_button.setEnabled(False)
            
    def stopplaying(self):
        sd.stop()
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Errore")
        msg.setText("Questa funzione non è ancora implementata.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()
        """
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
        self.amplify_dialog = AmplifyDialog(self)
        self.amplify_dialog.exec()

    def apply_amplification(self, factor):
        ini = int(self.xmin * self.sampling_rate)
        fin = int(self.xmax * self.sampling_rate)
        segnale = self.data[ini:fin]
        segnale = np.clip(segnale * factor, -32768, 32767).astype(np.int16)  
        self.data[ini:fin] = segnale
        self.ax.clear()
        self.ax.plot(self.time, self.data, linewidth=0.5, color="black")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlim(self.xmin, self.xmax)
        self.canvas.draw()

    def play_dialog(self):
        """Riproduce il segmento selezionato dell'audio."""
        self.stop_button.setEnabled(True)
        ini = int(self.xmin * self.sampling_rate)
        fin = int(self.xmax * self.sampling_rate)
        segment = self.data[ini:fin]  # Estrarre il segmento selezionato
        sd.play(segment, samplerate=self.sampling_rate)  # Riprodurre il suono

from PySide6.QtWidgets import QDialog

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


class ResamplingWindow(QWidget):
    def __init__(self, wav_file: str, wav_list: list = []):
        super().__init__()

        self.setWindowTitle("Resampling Audio")
        # self.setGeometry(200, 200, 400, 250)

        self.wav_file = wav_file
        self.wav_list = wav_list

        # Carica il file WAV e ottiene le informazioni
        self.sampling_rate, self.data = wavfile.read(self.wav_file)
        self.duration = len(self.data) / self.sampling_rate

        # Layout principale
        layout = QVBoxLayout()

        # Etichetta con informazioni sul file
        self.label_info = QLabel(
            f"File WAV selezionato: {self.wav_file}\nDurata: {self.duration:.2f} sec\nFrequenza di campionamento: {self.sampling_rate} Hz"
        )
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
        self.button_save.setEnabled(
            False
        )  # Inizialmente disabilitato fino a quando il resampling non è stato applicato
        self.button_save.clicked.connect(self.save_wav)
        layout.addWidget(self.button_save)

        self.bt_resample_all = QPushButton("Resample all WAV")
        self.bt_resample_all.setEnabled(self.wav_list != [])
        self.bt_resample_all.clicked.connect(self.resample_all)
        layout.addWidget(self.bt_resample_all)

        self.setLayout(layout)

    def apply_resampling(self):
        """
        Applica il resampling ai dati audio
        """
        new_sampling_rate = int(self.combo_sampling_rate.currentText())

        print(
            f"DEBUG: Resampling da {self.sampling_rate} Hz a {new_sampling_rate} Hz..."
        )
        self.data_resampled = librosa.resample(
            self.data, orig_sr=self.sampling_rate, target_sr=new_sampling_rate
        )
        plt.plot(self.data_resampled)
        # Converte il risultato in formato int16 per la scrittura su WAV
        self.resampled_data = (self.data_resampled * 32767).astype(np.int16)

        self.new_sampling_rate = new_sampling_rate
        self.button_save.setEnabled(True)  # Abilita il salvataggio

        print(
            f"DEBUG: Resampling completato. Nuova lunghezza: {len(self.data_resampled)} campioni."
        )

    def save_wav(self):
        """
        Salva il file WAV dopo il resampling
        """
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Salva file WAV", "", "WAV Files (*.wav)"
        )
        if save_path:
            wavfile.write(save_path, self.new_sampling_rate, self.data_resampled)
            print(f"DEBUG: File salvato correttamente in {save_path}")

    def resample_all(self):
        """
        resample all files in dir
        """
        # QMessageBox.information(self, "", "To be continued...")
        print(f"{self.wav_list=}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Analysis")
        # self.setGeometry(100, 100, 600, 400)
        self.wav_file = None

        self.wav_list: dict = {}

        self.plugin_widgets: list = []

        # Layout
        central_widget = QWidget()
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        # select/deselect all checkbox
        self.check_all_checkbox = QCheckBox("Check/Uncheck all")
        self.check_all_checkbox.stateChanged.connect(self.toggle_all_items)
        hlayout.addWidget(self.check_all_checkbox)  # Add "Check All" checkbox on top
        # remove pushbutton
        self.pb_remove = QPushButton("Remove checked WAV from list")
        self.pb_remove.clicked.connect(self.remove_from_list)
        hlayout.addWidget(self.pb_remove)  # Add "Check All" checkbox on top
        # spacer for left grouping
        hlayout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        layout.addLayout(hlayout)

        # list widget for WAV file paths
        self.wav_list_widget = QTreeWidget()
        self.wav_list_widget.setColumnCount(2)  # Number of columns
        self.wav_list_widget.setHeaderLabels(
            ["WAV file path", "duration (s)", "Sample rate (Hz)"]
        )  # Column headers

        self.wav_list_widget.header().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )  # Resize to fit content

        # Editor di testo per output
        self.text_edit = QTextEdit(self)

        splitter = QSplitter(Qt.Vertical)

        splitter.addWidget(self.wav_list_widget)
        splitter.addWidget(self.text_edit)
        splitter.setSizes([200, 400])

        layout.addWidget(splitter)

        central_widget.setLayout(layout)

        # Set central widget
        self.setCentralWidget(central_widget)

        self.resize(1400, 500)

        # Creazione del menu
        menubar = self.menuBar()

        # Menu File
        file_menu = menubar.addMenu("File")
        open_wav_action = QAction("Open wav", self)
        open_wav_action.triggered.connect(self.open_wav)
        file_menu.addAction(open_wav_action)

        open_wav_dir_action = QAction("Open wav directory", self)
        open_wav_dir_action.triggered.connect(self.open_wav_dir)
        file_menu.addAction(open_wav_dir_action)

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

        # Load modules from plugins directory
        self.modules: dict = {}
        for file_ in sorted(Path("plugins").glob("*.py")):
            module_name = file_.stem  # python file name without '.py'
            spec = importlib.util.spec_from_file_location(module_name, file_)
            self.modules[module_name] = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = self.modules[module_name]
            spec.loader.exec_module(self.modules[module_name])

        # Aggiunta dei plugin al menu Analyse
        self.actions = []
        for module_name in self.modules:
            action = QAction(module_name, self)
            action.triggered.connect(self.run_option)
            analyse_menu.addAction(action)
            self.actions.append(action)

    def toggle_all_items(self, state):
        """
        Toggles all items in the list based on the "Check/Uncheck all" checkbox state.
        """
        check_state = Qt.Checked if state == 2 else Qt.Unchecked
        for i in range(self.wav_list_widget.topLevelItemCount()):
            self.wav_list_widget.topLevelItem(i).setCheckState(0, check_state)

    def remove_from_list(self):
        """
        remove selected files from list
        """
        # Remove checked child items first
        for i in reversed(range(self.wav_list_widget.topLevelItemCount())):
            item = self.wav_list_widget.topLevelItem(i)

            if item.checkState(0) == Qt.Checked:  # Remove top-level item if checked
                # self.wav_list_widget.takeTopLevelItem(i)
                del self.wav_list[item.text(0)]

        self.update_wav_list()

        if not self.wav_list_widget.topLevelItemCount():
            self.check_all_checkbox.setChecked(False)

    def get_rate_duration(self, wav_file_path):
        """
        Carica il file WAV e ottiene le informazioni
        """
        sample_rate, data = wavfile.read(str(wav_file_path))
        duration = round(len(data) / sample_rate, 3)
        return sample_rate, duration

        # get sample_rate with wave module
        # disabled because give some errors
        """
        try:
            with wave.open(wav_file_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                duration = round(frames / float(sample_rate), 3)
            return sample_rate, duration
        except Exception:
            return "Not found", "Not found"
        """

    def update_wav_list(self):
        """
        update wav treewidget with list of wav
        """
        self.wav_list_widget.clear()
        for wav_file_path in self.wav_list:
            item = QTreeWidgetItem(
                [
                    wav_file_path,
                    str(self.wav_list[wav_file_path]["duration"]),
                    str(self.wav_list[wav_file_path]["sample rate"]),
                ]
            )
            item.setCheckState(0, Qt.Unchecked)
            self.wav_list_widget.addTopLevelItem(item)

    def open_wav(self):
        print("DEBUG: La funzione open_wav() è stata chiamata.")  # Controllo immediato
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open WAV File", "", "WAV Files (*.wav)"
        )
        if not file_paths:
            print("DEBUG: Nessun file WAV selezionato.")
            return

        for file_path in file_paths:
            if file_path in self.wav_list:
                self.text_edit.append(f"file {file_path} already loaded")
                continue

            sample_rate, duration = self.get_rate_duration(str(file_path))

            self.wav_list[file_path] = {
                "sample rate": sample_rate,
                "duration": duration,
            }

        self.update_wav_list()

        if len(file_paths) == 1:
            self.show_oscillogram(wav_file_path=file_paths[0])

    def open_wav_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if not directory:
            return
        for file_path in Path(directory).glob("*.wav"):
            # get sample_rate
            sample_rate, duration = self.get_rate_duration(str(file_path))
            self.wav_list[str(file_path)] = {
                "sample rate": sample_rate,
                "duration": duration,
            }
            # self.text_edit.append(f"file {file_path} added to list")

        self.update_wav_list()

    def close_program(self):
        print("ho chiamato la funzione close")
        self.close()

    def show_oscillogram(self, wav_file_path: str = ""):
        if wav_file_path:
            self.oscillogram_window = OscillogramWindow(wav_file_path)
            self.oscillogram_window.show()
        else:
            # check if wav checked in treewidget
            checked_wav_files = [
                self.wav_list_widget.topLevelItem(i).text(0)
                for i in range(self.wav_list_widget.topLevelItemCount())
                if self.wav_list_widget.topLevelItem(i).checkState(0) == Qt.Checked
            ]
            if checked_wav_files:
                self.oscillogram_window_list = []
                for wav_file_path in checked_wav_files:
                    osc_win = OscillogramWindow(wav_file_path)
                    self.oscillogram_window_list.append(osc_win)
                    osc_win.show()
            else:
                self.text_edit.append("No WAV file selected!")

    def resampling(self):
        """
        Apre la finestra di resampling
        """

        # check if wav checked in treewidget
        checked_wav_files = [
            self.wav_list_widget.topLevelItem(i).text(0)
            for i in range(self.wav_list_widget.topLevelItemCount())
            if self.wav_list_widget.topLevelItem(i).checkState(0) == Qt.Checked
        ]
        if checked_wav_files:
            self.resampling_window = ResamplingWindow(
                checked_wav_files[0], checked_wav_files
            )
            self.resampling_window.show()
        else:
            self.text_edit.append("No WAV file selected!")

    def run_option(self, module_name):
        """
        Carica il plugin e passa l'elenco dei file WAV selezionati
        """
        module_name = self.sender().text()
        print(f"running {module_name=}")

        self.text_edit.append(f"Running {module_name} plugin")

        # check if wav checked in treewidget
        checked_wav_files = [
            self.wav_list_widget.topLevelItem(i).text(0)
            for i in range(self.wav_list_widget.topLevelItemCount())
            if self.wav_list_widget.topLevelItem(i).checkState(0) == Qt.Checked
        ]
        if checked_wav_files:
            self.plugin_widgets.append(
                self.modules[module_name].Main(checked_wav_files)
            )
            self.plugin_widgets[-1].show()
        else:
            QMessageBox.warning(self, "", "No WAV file selected")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
