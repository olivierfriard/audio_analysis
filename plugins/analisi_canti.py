"""
Plugin analisi canti

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button, TextBox
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QFileDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import get_window
import sounddevice as sd


class Main(QWidget):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        self.window_size = 1024
        self.window_type = "hamming"
        self.overlap = 50

        self.figure, self.ax = plt.subplots(2, 1, figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.draw_plot()

    def draw_plot(self):
        """
        draw initial plot
        """

        # Plot dell'Oscillogramma
        self.ax[0].plot([], [], linewidth=0.5, color="black")
        self.ax[0].set_ylabel("Ampiezza")
        self.ax[0].set_xlabel("Tempo (s)")
        self.ax[0].set_title("", fontsize=20)

        # Creazione dell'annotazione (inizialmente nascosta)
        self.annot = self.ax[1].annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), fontsize=10, visible=False
        )

        # Variabile per tenere traccia dello stato dell'annotazione
        self.show_coords = False  # 🟢 Inizia disattivato
        self.cursor_event_id = None  # Memorizza l'evento per disattivarlo

        # Collega l'evento del tasto destro alla funzione toggle_coords
        self.figure.canvas.mpl_connect("button_press_event", self.toggle_coords)

        # Creazione delle caselle di testo
        axbox1 = self.figure.add_axes([0.1, 0.01, 0.10, 0.05])
        self.textbox_window_size = TextBox(axbox1, "Finestra", initial=str(self.window_size))

        axbox2 = self.figure.add_axes([0.25, 0.01, 0.10, 0.05])
        self.textbox_window_type = TextBox(axbox2, "Tipo", initial=self.window_type)

        axbox3 = self.figure.add_axes([0.45, 0.01, 0.10, 0.05])
        self.textbox_overlap = TextBox(axbox3, "Overlap (%)", initial=str(self.overlap))

        # Creazione del pulsante di aggiornamento
        ax_button_aggiorna = self.figure.add_axes([0.6, 0.01, 0.1, 0.05])
        self.button_update = Button(ax_button_aggiorna, "Aggiorna")
        self.button_update.on_clicked(self.update_spectrum)

        ax_button_stampa = self.figure.add_axes([0.75, 0.01, 0.1, 0.05])
        self.button_stampa = Button(ax_button_stampa, "Stampa")
        self.button_stampa.on_clicked(self.stampa_fig)

        ax_button_apri = self.figure.add_axes([0.85, 0.01, 0.1, 0.05])
        self.button_apri = Button(ax_button_apri, "Apri wav")
        self.button_apri.on_clicked(self.open_file)

        ax_button_play = self.figure.add_axes([0.1, 0.9, 0.1, 0.05])  # Definizione posizione pulsante
        self.button_play = Button(ax_button_play, "Play")  # Creazione pulsante
        self.button_play.on_clicked(self.play_audio)  # Collegare il pulsante alla funzione

        # Aggiunge l'interattività sull'oscillogramma
        self.span_time = SpanSelector(self.ax[0], self.onselect_time, "horizontal", useblit=True)
        self.span_time.set_props(alpha=0.5, facecolor="red")

        # Aggiunge l'interattività sullo spettro
        self.span_freq = SpanSelector(self.ax[1], self.onselect_freq, "horizontal", useblit=True)
        self.span_freq.set_props(alpha=0.5, facecolor="blue")

        """
        # Ajout du SpanSelector
        self.span = SpanSelector(self.ax[0], self.on_select, "horizontal", useblit=True, props=dict(facecolor="blue", alpha=0.5))

        
        # Bouton Reset
        reset_button_ax = self.figure.add_axes([0.7, 0.01, 0.1, 0.05])
        self.reset_button = Button(reset_button_ax, "Reset")
        self.reset_button.on_clicked(self.reset_plot)
        """

        self.canvas.draw()

    def open_file(self, event):
        """
        Apre una finestra per selezionare un file .wav e lo carica in NumPy.
        Restituisce il tasso di campionamento (sampling_rate) e i dati audio (data).
        """

        self.file_path, _ = QFileDialog.getOpenFileName(None, "Seleziona un file WAV", "", "File audio WAV (*.wav);;All files (*)")
        if not self.file_path:  # Se l'utente chiude la finestra senza scegliere un file
            print("Nessun file selezionato.")
            return None, None

        self.file_wav = str(Path(self.file_path).stem)

        # Carica il file .wav
        self.sampling_rate, self.data = wavfile.read(self.file_path)

        # Se il file è stereo, usa solo un canale
        if len(self.data.shape) > 1:
            print("File stereo rilevato. Prendo solo il primo canale.")
            self.data = self.data[:, 0]

        print(f"File caricato: {self.file_path}")
        print(f"Frequenza di campionamento: {self.sampling_rate} Hz")
        print(f"Durata: {len(self.data) / self.sampling_rate:.2f} secondi")

        """
        # Se il file è stereo, usa solo un canale
        if len(self.data.shape) > 1:
            self.data = data[:, 0]
        """

        # Normalizza il segnale
        self.data = self.data / np.max(np.abs(self.data))

        # Crea variable tempo
        self.time = np.linspace(0, len(self.data) / self.sampling_rate, num=len(self.data))

        # Variabili globali valori di default
        self.selected_range = [0, len(self.data)]
        self.selected_time = [0, np.max(self.time)]
        self.freq_range = [0, self.sampling_rate / 2]  # Range massimo (fino a Nyquist)

        # Plot dell'Oscillogramma
        self.ax[0].cla()
        self.ax[0].plot(self.time, self.data, linewidth=0.5, color="black")
        self.ax[0].set_ylabel("Ampiezza")
        self.ax[0].set_xlabel("Tempo (s)")
        self.ax[0].set_title(self.file_wav)

        # Secondo grafico preparazione
        self.ax[1].cla()
        self.ax[1].set_xlabel("Frequenza (Hz)")
        self.ax[1].set_ylabel("Ampiezza")

        self.canvas.draw()

    def spectrum(self):
        """
        Funzione per aggiornare lo spettro medio
        """

        # Memorizziamo la posizione dell'annotazione prima di cancellare il grafico
        x_annot, y_annot = self.annot.xy if self.annot.get_visible() else (None, None)

        self.ax[1].cla()  # Cancella il vecchio spettro

        selected_data = self.data[self.selected_range[0] : self.selected_range[1]]

        if len(selected_data) > 1:
            try:
                # global window_size, window_type, overlap

                # Converte i parametri delle caselle di testo
                self.window_size = int(self.textbox_window_size.text)
                self.window_type = self.textbox_window_type.text.strip().lower()
                self.overlap = int(self.textbox_overlap.text) * self.window_size // 100

                # Controlla i parametri
                if self.window_size <= 0 or self.overlap < 0 or self.overlap >= self.window_size:
                    print("Errore: Overlap deve essere positivo e minore della finestra.")
                    return

                num_windows = (len(selected_data) - self.window_size) // self.overlap
                if num_windows <= 0:
                    print("Errore: Dimensione finestra troppo grande.")
                    return

                # Crea la finestra
                window = get_window(self.window_type, self.window_size)

                # Inizializza lo spettro medio
                self.avg_spectrum = np.zeros(self.window_size // 2 + 1)

                for i in range(num_windows):
                    start = i * self.overlap
                    end = start + self.window_size
                    segment = selected_data[start:end] * window
                    fft_segment = np.abs(np.fft.rfft(segment))
                    self.avg_spectrum += fft_segment

                self.avg_spectrum /= num_windows  # Normalizza

                # Crea l'asse delle frequenze
                self.freqs = np.fft.rfftfreq(self.window_size, d=1 / self.sampling_rate)

                # Plotta lo spettro aggiornato
                self.ax[1].plot(self.freqs, self.avg_spectrum, color="k")
                self.ax[1].set_xlabel("Frequenza (Hz)")
                self.ax[1].set_ylabel("Ampiezza")
                self.ax[1].set_xlim(self.freq_range)

                # Ricrea l'annotazione dopo il disegno dello spettro
                self.annot = self.ax[1].annotate(
                    "",
                    xy=(0, 0),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    fontsize=10,
                    visible=False,
                )

                # Se l'annotazione era visibile, la ripristiniamo
                if x_annot is not None:
                    self.annot.xy = (x_annot, y_annot)
                    self.annot.set_text(f"x={x_annot:.3f}, y={y_annot:.3f}")
                    self.annot.set_visible(self.show_coords)

                # plt.draw()
                self.canvas.draw()

            except ValueError:
                print("Errore: Inserire numeri validi per finestra e overlap.")

    def update_spectrum(self, event):
        print(f"update spectrum {event}")
        self.spectrum()

    def stampa_fig(self, event):
        """
        Crea la figura con due subplot (Oscillogramma + Spettro)
        """

        fig_st, ax_st = plt.subplots(2, 1, figsize=(6, 8))
        plt.subplots_adjust(bottom=0.1)

        ax_st[0].plot(self.time, self.data, linewidth=0.5, color="black", alpha=0.3)
        ax_st[0].plot(
            self.time[self.selected_range[0] : self.selected_range[1]],
            self.data[self.selected_range[0] : self.selected_range[1]],
            linewidth=0.5,
            color="black",
            alpha=1,
        )
        ax_st[0].set_ylabel("Ampiezza")
        ax_st[0].set_xlabel("Tempo (s)")
        ax_st[0].set_title(self.file_wav, fontsize=14)

        # Plotta lo spettro aggiornato
        ax_st[1].plot(self.freqs, self.avg_spectrum, color="k")
        ax_st[1].set_xlabel("Frequenza (Hz)")
        ax_st[1].set_ylabel("Ampiezza")
        ax_st[1].set_xlim(self.freq_range)

        plt.show()

    def play_audio(self, event):
        """Riproduce il segmento selezionato dell'audio."""
        if self.selected_range:
            start, end = self.selected_range
            segment = self.data[start:end]  # Estrarre il segmento selezionato
            sd.play(segment, samplerate=self.sampling_rate)  # Riprodurre il suono

    def update_cursor(self, event):
        """
        Funzione per aggiornare le coordinate del mouse
        """

        def find_nearest(array, value):
            """
            Funzione per trovare l'indice più vicino in un array
            """
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]  # Restituisce l'indice e il valore più vicino

        if self.show_coords and event.inaxes in [self.ax[0], self.ax[1]]:  # Controlla se il mouse è nei grafici
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                idx, nearest_freq = find_nearest(self.freqs, x)

                if idx > 5:
                    yy = self.avg_spectrum[idx - 1 : idx + 1]
                    max_yy = np.argmax(yy)
                    nearest_freq = self.freqs[idx - 1 + max_yy]

                self.annot.xy = (nearest_freq, y)
                self.annot.set_text(f"fq={nearest_freq:.3f}")
                self.annot.set_visible(True)
                self.figure.canvas.draw_idle()
        else:
            self.annot.set_visible(False)
            self.figure.canvas.draw_idle()

    def toggle_coords(self, event):
        # global show_coords, cursor_event_id, freqs
        print(f"toggle coords {event}")

        if event.button == 3:  # Tasto destro del mouse
            self.show_coords = not self.show_coords  # Cambia stato

            if self.show_coords:
                # Attiva l'evento solo se è disattivato
                # self.cursor_event_id = self.figure.canvas.mpl_connect("motion_notify_event", update_cursor)
                print("Coordinate ATTIVATE")
            else:
                # Disattiva l'evento e nasconde il tooltip
                self.figure.canvas.mpl_disconnect(self.cursor_event_id)
                self.annot.set_visible(False)
                self.figure.canvas.draw_idle()
                print("Coordinate DISATTIVATE")
            self.spectrum()

    def onselect_time(self, xmin, xmax):
        # Funzione per selezionare una porzione dell'audio
        idx_min = np.searchsorted(self.time, xmin)
        idx_max = np.searchsorted(self.time, xmax)
        self.selected_range = [idx_min, idx_max]
        self.selected_time = [xmin, xmax]

        self.ax[0].cla()
        self.ax[0].plot(self.time, self.data, linewidth=0.5, color="black", alpha=0.3)
        self.ax[0].plot(
            self.time[self.selected_range[0] : self.selected_range[1]],
            self.data[self.selected_range[0] : self.selected_range[1]],
            linewidth=0.5,
            color="black",
            alpha=1,
        )
        self.ax[0].set_ylabel("Ampiezza")
        self.ax[0].set_xlabel("Tempo (s)")
        self.ax[0].set_title(self.file_wav, fontsize=20)

        # Aggiorna lo spettro
        self.spectrum()

    def onselect_freq(self, xmin, xmax):
        self.freq_range = [xmin, xmax]
        if self.freq_range[1] - self.freq_range[0] < 100:
            self.freq_range = [0, self.sampling_rate / 2]
        self.ax[1].set_xlim(self.freq_range)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_widget = Main()
    main_widget.show()

    sys.exit(app.exec())
