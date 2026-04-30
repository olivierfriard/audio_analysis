"""
main program of 'Analisi canti' package

"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QStyle,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile

__version__ = "0.2.0"
__version_date__ = "2026-04-23"


from .oscillogram import OscillogramWindow
from .wav_cutting import Wav_cutting


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

        self.setWindowTitle(f"Audio Analysis - v.{__version__}")
        self.wav_file = None

        self.wav_list: dict = {}

        self.project_path = None

        self.plugin_widgets: list = []

        # Layout
        central_widget = QWidget()
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        # select/deselect all checkbox
        self.check_all_checkbox = QCheckBox("Check/Uncheck all")
        self.check_all_checkbox.stateChanged.connect(self.toggle_all_items)
        hlayout.addWidget(self.check_all_checkbox)  # Add "Check All" checkbox on top
        # spacer for left grouping
        hlayout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        layout.addLayout(hlayout)

        # list widget for WAV file paths
        self.wav_list_widget = QTreeWidget()
        self.wav_list_widget.setColumnCount(2)  # Number of columns
        self.wav_list_widget.setHeaderLabels(
            ["WAV file path", "duration (s)", "Sample rate (Hz)"]
        )  # Column headers
        self.wav_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.wav_list_widget.customContextMenuRequested.connect(
            self.show_wav_context_menu
        )

        header = self.wav_list_widget.header()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(False)
        self.wav_list_widget.setColumnWidth(0, 520)
        self.wav_list_widget.setColumnWidth(1, 110)
        self.wav_list_widget.setColumnWidth(2, 130)

        # Editor di testo per output
        self.text_edit = QTextEdit(self)

        splitter = QSplitter(Qt.Vertical)

        splitter.addWidget(self.wav_list_widget)
        splitter.addWidget(self.text_edit)
        splitter.setSizes([400, 100])

        layout.addWidget(splitter)

        central_widget.setLayout(layout)

        # Set central widget
        self.setCentralWidget(central_widget)

        self.resize(1200, 500)

        # Creazione del menu
        menubar = self.menuBar()

        # Menu File

        file_menu = menubar.addMenu("File")

        open_wav_action = QAction("New project", self)
        open_wav_action.triggered.connect(self.new_project)
        file_menu.addAction(open_wav_action)

        open_project_action = QAction("Open project", self)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        """
        open_wav_action = QAction("Open wav", self)
        open_wav_action.triggered.connect(self.open_wav)
        file_menu.addAction(open_wav_action)
        """

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

        wavcutting_action = QAction("WAV cutting", self)
        wavcutting_action.triggered.connect(self.wav_cutting)
        edit_menu.addAction(wavcutting_action)

        # Menu Process
        # process_menu = menubar.addMenu("Process")

        # Menu Analyse
        analyse_menu = menubar.addMenu("Analyse")

        # Load modules from plugins directory
        self.modules: dict = {}

        for file_ in sorted((Path(__file__).parent / Path("plugins")).glob("*.py")):
            module_name = file_.stem  # python file name without '.py'
            spec = importlib.util.spec_from_file_location(module_name, file_)
            self.modules[module_name] = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = self.modules[module_name]
            spec.loader.exec_module(self.modules[module_name])

        # Aggiunta dei plugin al menu Analyse
        self.actions = []
        for module_name in self.modules:
            action = QAction(module_name, self)
            action.triggered.connect(self.run_plugin)
            analyse_menu.addAction(action)
            self.actions.append(action)

    def closeEvent(self, event):
        """
        Close all child windows explicitly
        """
        for widget in QApplication.topLevelWidgets():
            if widget is not self:
                widget.close()
        event.accept()

    def toggle_all_items(self, state):
        """
        Toggles all items in the list based on the "Check/Uncheck all" checkbox state.
        """
        check_state = Qt.Checked if state == 2 else Qt.Unchecked
        for i in range(self.wav_list_widget.topLevelItemCount()):
            top_level_item = self.wav_list_widget.topLevelItem(i)
            top_level_item.setCheckState(0, check_state)
            self.set_chunks_check_state(top_level_item, check_state)

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

    def set_chunks_check_state(self, parent_item, check_state):
        """
        Set the check state for all chunks of a top-level WAV item.
        """
        for i in range(parent_item.childCount()):
            parent_item.child(i).setCheckState(0, check_state)

    def set_songs_check_state(self, chunk_item, check_state):
        """
        Set the check state for all songs of a chunk item.
        """
        for i in range(chunk_item.childCount()):
            chunk_item.child(i).setCheckState(0, check_state)

    def set_songs_without_icon_check_state(self, chunk_item, check_state):
        """
        Set the check state only for songs without an icon.
        """
        for i in range(chunk_item.childCount()):
            song_item = chunk_item.child(i)
            if song_item.icon(0).isNull():
                song_item.setCheckState(0, check_state)

    def show_wav_context_menu(self, pos):
        """
        Show a context menu for WAV and chunk items.
        """
        item = self.wav_list_widget.itemAt(pos)
        if item is None:
            return

        menu = QMenu(self)

        if item.parent() is None:
            show_oscillogram_action = menu.addAction("Show oscillogram")
            select_chunks_action = menu.addAction("Select all chunks")
            deselect_chunks_action = menu.addAction("Deselect all chunks")

            selected_action = menu.exec(
                self.wav_list_widget.viewport().mapToGlobal(pos)
            )

            if selected_action == show_oscillogram_action:
                self.show_oscillogram(
                    wav_file_path=item.data(0, Qt.ItemDataRole.UserRole)
                )
            elif selected_action == select_chunks_action:
                self.set_chunks_check_state(item, Qt.CheckState.Checked)
            elif selected_action == deselect_chunks_action:
                self.set_chunks_check_state(item, Qt.CheckState.Unchecked)
            return

        if item.parent().parent() is None:
            select_songs_action = menu.addAction("Select all songs")
            select_songs_without_data_action = menu.addAction(
                "Select songs without data"
            )
            deselect_songs_action = menu.addAction("Deselect all songs")

            selected_action = menu.exec(
                self.wav_list_widget.viewport().mapToGlobal(pos)
            )

            if selected_action == select_songs_action:
                self.set_songs_check_state(item, Qt.CheckState.Checked)
            elif selected_action == select_songs_without_data_action:
                self.set_songs_without_icon_check_state(item, Qt.CheckState.Checked)
            elif selected_action == deselect_songs_action:
                self.set_songs_check_state(item, Qt.CheckState.Unchecked)

    def get_selected_files(self) -> list:
        """
        Return the file paths of all checked items in the tree.
        """
        selected_levels = set()
        selected_files = []
        seen_files = set()

        for i in range(self.wav_list_widget.topLevelItemCount()):
            parent_item = self.wav_list_widget.topLevelItem(i)
            parent_wav_path = Path(parent_item.data(0, Qt.ItemDataRole.UserRole))

            if parent_item.checkState(0) == Qt.CheckState.Checked:
                selected_levels.add(0)
                wav_path = str(parent_wav_path)
                if wav_path not in seen_files:
                    selected_files.append(wav_path)
                    seen_files.add(wav_path)

            for j in range(parent_item.childCount()):
                chunk_item = parent_item.child(j)
                chunk_file_path = parent_wav_path.with_suffix("") / chunk_item.text(0)

                if chunk_item.checkState(0) == Qt.CheckState.Checked:
                    selected_levels.add(1)
                    chunk_path = str(chunk_file_path)
                    if chunk_path not in seen_files:
                        selected_files.append(chunk_path)
                        seen_files.add(chunk_path)

                for k in range(chunk_item.childCount()):
                    song_item = chunk_item.child(k)
                    if song_item.checkState(0) != Qt.CheckState.Checked:
                        continue

                    selected_levels.add(2)
                    song_file_path = parent_wav_path.with_suffix("") / song_item.text(0)
                    song_path = str(song_file_path)
                    if song_path not in seen_files:
                        selected_files.append(song_path)
                        seen_files.add(song_path)

        if len(selected_levels) > 1:
            QMessageBox.warning(
                self,
                "",
                "Please select object of the same type.",
            )
            return []

        return selected_files

    def get_rate_duration(self, wav_file_path):
        """
        Carica il file WAV e ottiene le informazioni
        """
        sample_rate, data = wavfile.read(str(wav_file_path))
        duration = round(len(data) / sample_rate, 3)
        return sample_rate, duration

    def update_wav_list(self):
        """
        Update wav treewidget with wav files, chunks and songs.
        """
        completed_song_icon = self.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogApplyButton
        )

        print("update treeview")

        r = self.read_json_file(self.json_file_path)
        wav_file_path = r["wav_file_name"]
        self.wav_list[wav_file_path] = r

        self.wav_list_widget.clear()

        for wav_file_path, wav_data in self.wav_list.items():
            parent_item = QTreeWidgetItem(
                [
                    Path(wav_file_path).name,
                    str(wav_data["duration"]),
                    str(wav_data["sample rate"]),
                ]
            )
            parent_item.setCheckState(0, Qt.CheckState.Unchecked)
            parent_item.setData(0, Qt.ItemDataRole.UserRole, wav_file_path)
            self.wav_list_widget.addTopLevelItem(parent_item)

            for chunk_name, chunk_data in wav_data.get("chunks", {}).items():
                child_item = QTreeWidgetItem(
                    [
                        chunk_name,
                        str(chunk_data["start"]),
                        str(chunk_data["end"]),
                    ]
                )
                child_item.setCheckState(0, Qt.CheckState.Unchecked)
                child_item.setData(0, Qt.ItemDataRole.UserRole, chunk_data)
                parent_item.addChild(child_item)

                for song_name, song_data in chunk_data.get("songs", {}).items():
                    song_item = QTreeWidgetItem(
                        [
                            song_name,
                            str(song_data.get("call_duration", "")),
                            str(song_data.get("pulse_number", "")),
                        ]
                    )
                    song_item.setCheckState(0, Qt.CheckState.Unchecked)
                    song_item.setData(0, Qt.ItemDataRole.UserRole, song_data)
                    if song_data:
                        song_item.setIcon(0, completed_song_icon)
                    child_item.addChild(song_item)

                if child_item.childCount():
                    child_item.setExpanded(True)

            parent_item.setExpanded(True)

    def create_json_file(self, path) -> int:
        """
        create json file corresponding to wav file
        Args:
            path: path of main wav file

        """
        print("create json file")

        sample_rate, duration = self.get_rate_duration(path)

        self.json_file_path = path.with_suffix("") / f"{path.stem}.json"
        try:
            with open(self.json_file_path, "w") as f_out:
                json.dump(
                    {
                        "wav_file_name": str(path),
                        "sample rate": sample_rate,
                        "duration": duration,
                        "chunks": {
                            path.name: {
                                "start": 0,
                                "end": int(sample_rate * duration),
                            }
                        },
                    },
                    f_out,
                )
        except Exception:
            return 1

        # copy fake chunk
        print("create fake chunk")
        shutil.copy(path, path.with_suffix("") / path.name)

        return 0

    def read_json_file(self, json_file_path) -> dict:
        """
        read content of JSON file
        """
        with open(json_file_path, "r") as f_in:
            return json.load(f_in)

    def open_project(self, project_path: str | Path | None = None):
        """
        open a json or wav file
        """
        if isinstance(project_path, bool):
            project_path = None

        if project_path is None:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open file",
                "",
                "Supported Files (*.wav *.json);;WAV Files (*.wav);;JSON Files (*.json)",
            )
            if not file_path:
                print("DEBUG: Nessun file WAV selezionato.")
                return
        else:
            file_path = str(Path(project_path).expanduser().resolve())
            if not Path(file_path).exists():
                QMessageBox.warning(self, "", f"Project not found: {file_path}")
                return

        # Reset the current project tree before loading the selected project.
        self.wav_list = {}
        self.wav_list_widget.clear()

        if Path(file_path).suffix in (".wav", ".WAV"):
            # check if directory exists
            if not Path(file_path).with_suffix("").is_dir():
                QMessageBox.warning(self, "", "No project found")
                return
            else:
                self.json_file_path = Path(file_path).with_suffix("") / Path(
                    Path(file_path).name
                ).with_suffix(".json")
                wav_file_path = file_path
                self.update_wav_list()
                r = self.read_json_file(self.json_file_path)
                if len(r.get("chunks", {})) == 1:
                    self.show_oscillogram(wav_file_path=wav_file_path)

        if Path(file_path).suffix in (".json"):
            self.json_file_path = Path(file_path)
            r = self.read_json_file(file_path)
            wav_file_path = r["wav_file_name"]
            self.wav_list[wav_file_path] = r
            self.update_wav_list()

            if len(r.get("chunks", {})) == 1:
                self.show_oscillogram(wav_file_path=wav_file_path)

    def new_project(self):
        """
        apre il file wav indicato dall'utente
        crea la directory e il file json con sample rate e durata
        """

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV File", "", "WAV Files (*.wav)"
        )
        if not file_path:
            print("DEBUG: Nessun file WAV selezionato.")
            return

        path = Path(file_path)
        json_file_path = path.with_suffix("") / f"{path.stem}.json"
        create_json_file_flag = False
        # check if directory exists
        if Path(file_path).with_suffix("").is_dir():
            # check if json file exists
            if not json_file_path.is_file():
                create_json_file_flag = True
        else:
            # create directory
            print("create directory {Path(file_path).with_suffix('')}")
            Path(file_path).with_suffix("").mkdir()
            create_json_file_flag = True

        if create_json_file_flag:
            r = self.create_json_file(path)
            if r:
                print("Creation of JSON file failed")
                return
        else:
            self.json_file_path = path.with_suffix("") / f"{path.stem}.json"

        # Reset the current project tree before loading the newly selected one.
        self.wav_list = {}
        self.wav_list_widget.clear()

        self.update_wav_list()

        self.show_oscillogram(wav_file_path=file_path)

    def close_program(self):
        """
        close
        """
        self.close()

    def show_oscillogram(self, wav_file_path: str = ""):
        if wav_file_path:
            self.oscillogram_window = OscillogramWindow(wav_file_path)
            self.oscillogram_window.load_wav_signal.connect(self.load_wav)
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

    def load_wav(self, file_list: list):
        """
        load wav file in wav_list dict
        """
        r = self.read_json_file(self.json_file_path)
        wav_file_path = r["wav_file_name"]
        self.wav_list[wav_file_path] = r
        self.update_wav_list()

        return

        """
        self.wav_list = {}
        for file_path in file_list:
            sample_rate, duration = self.get_rate_duration(str(file_path))
            self.wav_list[str(file_path)] = {
                "sample rate": sample_rate,
                "duration": duration,
            }
        """

        self.update_wav_list()

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

    def wav_cutting(self):
        """
        Open the wav cutting window
        """
        # check if wav checked in treewidget
        checked_wav_files = [
            self.wav_list_widget.topLevelItem(i).text(0)
            for i in range(self.wav_list_widget.topLevelItemCount())
            if self.wav_list_widget.topLevelItem(i).checkState(0) == Qt.Checked
        ]
        if checked_wav_files:
            self.wav_cutting_widget = Wav_cutting(checked_wav_files[0])
            self.wav_cutting_widget.show()
        else:
            self.text_edit.append("No WAV file selected!")

    def run_plugin(self, module_name):
        """
        Carica il plugin e passa l'elenco dei file selezionati
        """
        module_name = self.sender().text()
        print(f"running {module_name=}")

        self.text_edit.append(f"Running {module_name} plugin")

        selected_files = self.get_selected_files()
        if selected_files:
            self.plugin_widgets.append(self.modules[module_name].Main(selected_files))
            if hasattr(self.plugin_widgets[-1], "results_saved_signal"):
                self.plugin_widgets[-1].results_saved_signal.connect(
                    self.update_wav_list
                )
            if module_name != "trova_picchi_vs2" and hasattr(
                self.plugin_widgets[-1], "plugin_closed_signal"
            ):
                self.plugin_widgets[-1].plugin_closed_signal.connect(
                    self.update_wav_list
                )
            self.plugin_widgets[-1].show()
        else:
            QMessageBox.warning(self, "", "No file selected")


def parse_cli_args(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-p",
        "--project",
        dest="project",
        help="WAV or JSON project to open automatically at startup",
    )
    return parser.parse_known_args(argv)


def run(argv: list[str] | None = None):
    args, remaining_args = parse_cli_args(argv)
    qt_argv = [sys.argv[0], *remaining_args]
    app = QApplication(qt_argv)
    window = MainWindow()
    window.show()
    if args.project:
        window.open_project(args.project)
    else:
        window.project_path = None
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
