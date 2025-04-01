from pathlib import Path
from scipy.io import wavfile
import shutil
import numpy as np
import json
import librosa

from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QLabel,
    QMessageBox,
    QHBoxLayout,
    QSpinBox,
    QFileDialog,
)
from PySide6.QtCore import Signal


class Wav_cutting(QWidget):
    cut_ended_signal = Signal(list)

    def __init__(self, wav_file: str):
        super().__init__()

        self.durata_ritaglio = 60  # Durata predefinita

        self.wav_file = wav_file

        self.setWindowTitle(
            f"{Path(__file__).stem.replace('_', ' ')} - {Path(self.wav_file).stem}"
        )

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

        # Pulsante seleziona cartella madre
        """
        hlayout = QHBoxLayout()
        self.button_select = QPushButton("Scegli Cartella Madre", self)
        self.button_select.clicked.connect(self.select_folder)
        hlayout.addWidget(self.button_select)
        hlayout.addStretch()
        layout.addLayout(hlayout)
        """

        """
        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Durata ritaglio (secondi):"))
        self.duration = QSpinBox()
        self.duration.setMinimum(1)
        self.duration.setMaximum(1000)
        self.duration.setValue(self.durata_ritaglio)
        self.duration.setSingleStep(1)
        self.duration.valueChanged.connect(self.update_label)
        self.duration.setEnabled(False)
        hlayout.addWidget(self.duration)
        hlayout.addStretch()
        layout.addLayout(hlayout)
        """

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Number of chunk(s)"))
        self.n_chunks_sb = QSpinBox()
        self.n_chunks_sb.setMinimum(1)
        self.n_chunks_sb.setMaximum(100)
        self.n_chunks_sb.setValue(1)
        self.n_chunks_sb.setSingleStep(1)
        # self.n_chunks_sb.valueChanged.connect(self.update_label)
        hlayout.addWidget(self.n_chunks_sb)
        hlayout.addStretch()
        layout.addLayout(hlayout)


        hlayout.addWidget(QLabel("offset (s)"))
        self.offset = QLineEdit()
        self.offset.setText("0.4")
        hlayout.addWidget(self.offset)
        hlayout.addStretch()
        layout.addLayout(hlayout)

        # Pulsante per salvare i file ritagliati
        hlayout = QHBoxLayout()
        self.button_save = QPushButton("Salva i files ritagliati", self)
        self.button_save.clicked.connect(self.save_files)
        hlayout.addWidget(self.button_save)

        """
        self.button_save.setEnabled(
            False
        )  # Disabilitato se sottocartella non è stata ancora selezionata
        """
        hlayout.addStretch()
        layout.addLayout(hlayout)

        # Variabile per salvare la cartella selezionata
        self.selected_folder = None

        self.setLayout(layout)

    def update_label(self, text):
        """
        Aggiorna l'etichetta con la durata scelta
        """
        self.durata_ritaglio = self.duration.value()

    def select_folder(self):
        """
        Apre il file dialog per selezionare una cartella e la salva in self.selected_folder
        """

        folder_path = QFileDialog.getExistingDirectory(
            self, "Seleziona la cartella di origine"
        )
        if folder_path:  # Controlla che l'utente non abbia annullato la selezione
            self.selected_folder = Path(folder_path)
            print(f"DEBUG: Cartella selezionata -> {self.selected_folder}")

            # Creo la sottocartella basata sul nome del file WAV
            self.nome_subcartella = self.selected_folder / Path(self.wav_file).stem

            # check if folder already exists
            if self.nome_subcartella.is_dir():
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(f"The directory {self.nome_subcartella} already exists!")
                msg.setWindowTitle("Warning")

                msg.addButton("Erase files", QMessageBox.YesRole)
                msg.addButton("Cancel", QMessageBox.YesRole)

                msg.exec()

                match msg.clickedButton().text():
                    case "Erase files":
                        shutil.rmtree(self.nome_subcartella)
                    case "Cancel":
                        return

            self.nome_subcartella.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Sottocartella creata -> {self.nome_subcartella}")

    def save_files(self):
        """
        Salva i ritagli assicurandosi che il taglio avvenga dove il segnale è minimo
        """
        self.select_folder()

        # create the json file
        data_file_path = Path(self.nome_subcartella) / Path(
            Path(self.wav_file).name
        ).with_suffix(".json")
        # test if .json exists
        if data_file_path.is_file():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"The {data_file_path} file already exists!")
            msg.setWindowTitle("Warning")

            msg.addButton("Overwrite file", QMessageBox.YesRole)
            msg.addButton("Cancel", QMessageBox.YesRole)

            msg.exec()

            match msg.clickedButton().text():
                case "Cancel":
                    return

        parameters: dict = {}

        original_name = f"{Path(self.nome_subcartella) / Path(self.wav_file).stem}"

        # set duration in base of number of chunks
        n_chunks = self.n_chunks_sb.value()
        intervallo = float(self.offset.text())
        self.durata_ritaglio = round(len(self.data) / self.sampling_rate / n_chunks)
        print(self.durata_ritaglio)
         
        cut_file_list: list = []
        ini = 0
        counter = 0  # per tenere traccia del numero di ritagli salvati
        while ini < len(self.data):
            # Calcolo della fine teorica del segmento di durata self.durata_ritaglio
            if counter == n_chunks - 1 or int(
                ini + self.sampling_rate * self.durata_ritaglio
            ) > len(self.data):
                fin = len(self.data)
            else:
                fin = int(ini + self.sampling_rate * self.durata_ritaglio)

            # Definisco l'intervallo ±0.1 secondi attorno al punto fin
            offset = int(self.sampling_rate * intervallo/2)
            start_range = max(fin - offset, 0)
            end_range = min(fin + offset, len(self.data))
            fin_range = np.arange(start_range, end_range)
            print("offset", offset, "start", start_range,"end",end_range)
            # Calcolo del RMS nel range definito
            frame_length = int(self.sampling_rate /100)
            hop_length = 1 #int(self.sampling_rate)
            rms = librosa.feature.rms(
                y=self.data[fin_range], frame_length=frame_length, hop_length=hop_length
            )[0]

            # Individuo l'indice in cui il valore RMS è minimo
            min_index = np.argmin(rms)
            fin_best = fin_range[min_index]
            print("rms", len(rms),"fin_best", fin_best)

            # Costruisco il nome del file per il ritaglio corrente
            nome_ritaglio = f"{original_name}_{ini:09d}_{fin_best - 1:09d}.wav"
            print(nome_ritaglio)

            # Evito un eventuale loop infinito: se il nuovo punto di taglio
            # non fa avanzare l'indice, interrompo il ciclo
            if fin_best <= ini:
                break

            # Ritaglio la porzione dal segnale e la salvo
            ritaglio = self.data[ini:fin_best]
            wavfile.write(nome_ritaglio, self.sampling_rate, ritaglio)

            # add file to list of files
            cut_file_list.append(nome_ritaglio)

            parameters[Path(nome_ritaglio).name] = {
                "start": int(ini),
                "end": int(fin_best - 1),
                "cut_from": self.wav_file,
            }

            # Aggiorno ini per il prossimo ritaglio
            ini = fin_best
            counter += 1

        # write file
        try:
            with open(data_file_path, "w", encoding="utf-8") as f_out:
                json.dump(parameters, f_out, indent=0, ensure_ascii=False)

            print(f"Risultati salvati in {data_file_path}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "",
                f"Error saving the {data_file_path} file: {e}",
            )

        QMessageBox.information(
            self,
            "",
            f"{counter} file{'s' if counter > 1 else ''} saved",
        )

        # delete file .tmp
        if Path(self.wav_file).exists() and Path(self.wav_file).suffix == ".tmp":
            Path(self.wav_file).unlink()

        self.cut_ended_signal.emit(cut_file_list)
