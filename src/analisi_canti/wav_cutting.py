import json
import shutil
from pathlib import Path

import librosa
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile


class Wav_cutting(QWidget):
    cut_ended_signal = Signal(list)

    def __init__(self, wav_file: str):
        super().__init__()

        self.durata_ritaglio = 60  # Default duration

        self.wav_file = wav_file

        self.setWindowTitle(
            f"{Path(__file__).stem.replace('_', ' ')} - {Path(self.wav_file).stem}"
        )

        # Load the WAV file and get information
        self.sampling_rate, self.data = wavfile.read(self.wav_file)
        self.duration = len(self.data) / self.sampling_rate

        # Main layout
        layout = QVBoxLayout()

        # Label with file information
        self.label_info = QLabel(
            f"Selected WAV file: {self.wav_file}\nDuration: {self.duration:.2f} sec\nSampling rate: {self.sampling_rate} Hz"
        )
        layout.addWidget(self.label_info)

        # Parent folder selection button
        """
        hlayout = QHBoxLayout()
        self.button_select = QPushButton("Choose Parent Folder", self)
        self.button_select.clicked.connect(self.select_folder)
        hlayout.addWidget(self.button_select)
        hlayout.addStretch()
        layout.addLayout(hlayout)
        """

        """
        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Cut duration (seconds):"))
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

        # Button to save cut files
        hlayout = QHBoxLayout()
        self.button_save = QPushButton("Save cut files", self)
        self.button_save.clicked.connect(self.save_files)
        hlayout.addWidget(self.button_save)

        """
        self.button_save.setEnabled(
            False
        )  # Disabled if the subfolder has not been selected yet
        """
        hlayout.addStretch()
        layout.addLayout(hlayout)

        # Variable used to store the selected folder
        self.selected_folder = None

        self.setLayout(layout)

    def update_label(self, text):
        """
        Update the label with the selected duration.
        """
        self.durata_ritaglio = self.duration.value()

    def select_folder(self):
        """
        Open the file dialog to select a folder and store it in self.selected_folder.
        """

        folder_path = QFileDialog.getExistingDirectory(
            self, "Select the source folder"
        )
        if folder_path:  # Check that the user did not cancel the selection
            self.selected_folder = Path(folder_path)
            print(f"DEBUG: Selected folder -> {self.selected_folder}")

            # Create the subfolder based on the WAV file name
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
            print(f"DEBUG: Subfolder created -> {self.nome_subcartella}")

    def save_files(self):
        """
        Save cuts while ensuring the cut happens where the signal is minimal.
        """

        """
        self.select_folder()
        # create the json file
        data_file_path = Path(self.nome_subcartella) / Path(
            Path(self.wav_file).name
        ).with_suffix(".json")
        """

        json_file_path = (
            Path(self.wav_file).with_suffix("") / f"{Path(self.wav_file).stem}.json"
        )

        # test if .json exists

        if not json_file_path.is_file():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"The {json_file_path} file does not exist!")
            msg.setWindowTitle("Warning")
            msg.addButton("OK", QMessageBox.YesRole)
            msg.exec()
            match msg.clickedButton().text():
                case "OK":
                    return

        with open(json_file_path, "r") as f_in:
            parameters: dict = json.load(f_in)

        # delete old chunks
        for chunk_file in parameters["chunks"]:
            (Path(self.wav_file).with_suffix("") / Path(chunk_file)).unlink(
                missing_ok=True
            )
        parameters["chunks"] = {}

        original_name = Path(self.wav_file).stem

        # Set duration based on the number of chunks.
        n_chunks = self.n_chunks_sb.value()
        intervallo = float(self.offset.text())
        self.durata_ritaglio = round(len(self.data) / self.sampling_rate / n_chunks)
        print(self.durata_ritaglio)

        cut_file_list: list = []
        ini = 0
        counter = 0  # Track the number of saved cuts
        while ini < len(self.data):
            # Compute the theoretical end of the segment with duration self.durata_ritaglio.
            if counter == n_chunks - 1 or int(
                ini + self.sampling_rate * self.durata_ritaglio
            ) > len(self.data):
                fin = len(self.data)
            else:
                fin = int(ini + self.sampling_rate * self.durata_ritaglio)

            # Define the interval around the fin point.
            offset = int(self.sampling_rate * intervallo / 2)
            start_range = max(fin - offset, 0)
            end_range = min(fin + offset, len(self.data))
            fin_range = np.arange(start_range, end_range)
            print("offset", offset, "start", start_range, "end", end_range)
            # Compute RMS in the defined range.
            frame_length = int(self.sampling_rate / 100)
            hop_length = 1  # int(self.sampling_rate)
            rms = librosa.feature.rms(
                y=self.data[fin_range], frame_length=frame_length, hop_length=hop_length
            )[0]

            # Find the index where the RMS value is minimal.
            min_index = np.argmin(rms)
            fin_best = fin_range[min_index]
            print("rms", len(rms), "fin_best", fin_best)

            # Build the file name for the current cut.
            nome_ritaglio = f"{original_name}_{ini:09d}_{fin_best - 1:09d}.wav"
            print(nome_ritaglio)

            # Avoid a possible infinite loop: stop if the new cut point
            # does not advance the index.
            if fin_best <= ini:
                break

            # Cut the signal segment and save it.
            ritaglio = self.data[ini:fin_best]
            wavfile.write(
                Path(self.wav_file).with_suffix("") / nome_ritaglio,
                self.sampling_rate,
                ritaglio,
            )

            # add file to list of files
            cut_file_list.append(
                str(Path(self.wav_file).with_suffix("") / nome_ritaglio)
            )

            parameters["chunks"][Path(nome_ritaglio).name] = {
                "start": int(ini),
                "end": int(fin_best - 1),
                "cut_from": str(Path(self.wav_file).with_suffix(".wav")),
            }

            # Update ini for the next cut.
            ini = fin_best
            counter += 1

        # write file
        try:
            with open(json_file_path, "w", encoding="utf-8") as f_out:
                json.dump(parameters, f_out, indent=0, ensure_ascii=False)

            print(f"Results saved in {json_file_path}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "",
                f"Error saving the {json_file_path} file: {e}",
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
