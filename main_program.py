"""
Framework for the management and startup of analysis plugins.

"""

import importlib.util
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QTextEdit
from PySide6.QtGui import QAction


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio analysis")
        self.setGeometry(100, 100, 600, 400)

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
        run_menu = menubar.addMenu("Plugins")

        # Creazione delle azioni
        for module_name in self.modules:
            action = QAction(module_name, self)
            # Collegamento delle azioni alle funzioni
            action.triggered.connect(lambda: self.run_option(module_name))
            # Aggiunta delle azioni al menu
            run_menu.addAction(action)

    def run_option(self, module_name):
        """ """
        self.text_edit.append(f"Running {module_name} plugin")
        # self.modules[module_name].main()

        self.w = self.modules[module_name].Main()
        self.w.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
