from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QGridLayout,
    QPushButton,
    QSlider,
    QMessageBox,
    QLineEdit,
    QLabel,
    QDoubleSpinBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector
import numpy as np
import matplotlib.pyplot as plt
import sys


class OscillogramWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generic Oscillogram")
        self.setGeometry(300, 300, 800, 500)

        # Simulazione dati per il grafico
        self.duration = 10  # Simuliamo 10 secondi di segnale
        self.time = np.linspace(0, self.duration, num=1000)
        self.data = np.sin(2 * np.pi * 5 * self.time)  # Segnale fittizio

        # Layout principale a griglia (10 colonne)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        
        self.widgets_riga1 = {
                "zoom_in": {
                    "type": "QPushButton",
                    "label": "Zoom +",
                    "height_label": 10,
                    "row": 1, "col": 0,
                    "row_span": 1, "col_span": 1,
                    "default": None
                },
                "zoom_out": {
                    "type": "QPushButton",
                    "label": "Zoom -",
                    "row": 1, "col": 1,
                    "row_span": 1, "col_span": 1,
                    "default": None
                },
                 "Envelope": {
                    "type": "QPushButton",
                    "label": "Envelope",
                    "row": 0, "col": 3,
                    "row_span": 2, "col_span": 1,
                    "default": None
                        },
                "w_size": {
                    "type": "QLineEdit",
                    "label": "W Size",
                    "row": 1, "col": 4,
                    "row_span": 1, "col_span": 1,
                    "default": "1024"
                },
                "w_overlap": {
                    "type": "QLineEdit",
                    "label": "W Overlap",
                    "row": 1, "col": 5,
                    "row_span": 1, "col_span": 1,
                    "default": "512"
                },
                "Peaks": {
                    "type": "QPushButton",
                    "label": "Peaks",
                    "row": 0, "col": 7,
                    "row_span": 2, "col_span": 1,
                    "default": None
                },
                "min_amp": {
                    "type": "QDoubleSpinBox",
                    "label": "Min Amp",
                    "row": 1, "col": 8,
                    "row_span": 1, "col_span": 1,
                    "default": ["0.1", "3", "0.005", "0.0", "1"] 
                },
                "min_dist": {
                    "type": "QDoubleSpinBox",
                    "label": "Min Dist",
                    "row": 1, "col": 9,
                    "row_span": 1, "col_span": 1,
                    "default": ["0.3", "3", "0.05", "0.0", "10"] 
                },
            }
       
        self.widgets_rigaFinale = {
                "save": {
                    "type": "QPushButton",
                    "label": "Save Calls",
                    "height_label": 10,
                    "row": 4, "col": 0,
                    "row_span": 1, "col_span": 3,
                    "default": None
                }
            }
       

        for key, props in self.widgets_riga1.items():
            widget = None 
           # Creazione del widget corrispondente
            if props["type"] == "QPushButton":
                widget = QPushButton(props["label"])
                if widget:
                    self.layout.addWidget(widget, props["row"], props["col"], props["row_span"], props["col_span"])
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
            else:
                continue  # Ignora eventuali tipi non definiti
  

        
        
        for key, props in self.widgets_rigaFinale.items():
            widget = None 
           # Creazione del widget corrispondente
            if props["type"] == "QPushButton":
                widget = QPushButton(props["label"])
                if widget:
                    self.layout.addWidget(widget, props["row"], props["col"], props["row_span"], props["col_span"])                
            else:
                continue

        self.layout.setRowStretch(0, 1)  # Etichette
        self.layout.setRowStretch(1, 1)  # Pulsanti e input
        self.layout.setRowStretch(2, 5)  # Grafico Matplotlib
        self.layout.setRowStretch(3, 1)  # Slider
        self.layout.setRowStretch(4, 1)  # Pulsanti
    
    
        
        
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
        self.selected_region = None
        self.span_selector = SpanSelector(
            self.ax, self.on_select, "horizontal",
            useblit=True, props=dict(alpha=0.5, facecolor="red")
        )

        # 🔹 Disegna il grafico iniziale
        self.ax.plot(self.time, self.data, linewidth=0.5, color="black")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlim(0, self.duration)
        self.canvas.draw()

    def show_message(self, button_name):
        """Mostra un messaggio di avviso con il nome del pulsante premuto."""
        QMessageBox.warning(self, "Non Implementato", f"La funzione {button_name} non è ancora implementata.")

    def on_select(self, xmin, xmax):
        """Evidenzia l'area selezionata con il mouse e aggiorna xmin e xmax."""
        if abs(xmax - xmin) < 0.01:  # Evita selezioni troppo piccole
            return  

        # Aggiorna xmin e xmax con la selezione
        self.xmin = xmin
        self.xmax = xmax
        print(f"Selezione aggiornata: xmin={self.xmin}, xmax={self.xmax}")  # Debug

        if self.selected_region:
            self.selected_region.remove()  # Rimuove la selezione precedente
        self.selected_region = self.ax.axvspan(xmin, xmax, color="red", alpha=0.3)

        self.canvas.draw_idle()

    def on_slider(self, value):
        """Aggiorna la vista dell'oscillogramma in base allo slider."""
        range_view = self.duration / 5  # Mostra 1/5 del segnale
        pos = value / 100 * (self.duration - range_view)
        self.ax.set_xlim(pos, pos + range_view)
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OscillogramWindow()
    window.show()
    sys.exit(app.exec())
