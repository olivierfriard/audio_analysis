import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window
from matplotlib.widgets import SpanSelector, Button, TextBox
import tkinter as tk
from tkinter import filedialog
import sounddevice as sd

def open_file(event):
    global file_wav, sampling_rate, time, data, selected_range, selected_time, freq_range
    """
    Apre una finestra per selezionare un file .wav e lo carica in NumPy.
    Restituisce il tasso di campionamento (sampling_rate) e i dati audio (data).
    """
    # Crea la finestra per la selezione del file
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale

    # Finestra di dialogo per scegliere il file .wav
    file_path = filedialog.askopenfilename(title="Seleziona un file WAV",
                                           filetypes=[("File audio WAV", "*.wav")])

    file_wav = file_path.split("/")[-1][:-4]
    if not file_path:  # Se l'utente chiude la finestra senza scegliere un file
        print("Nessun file selezionato.")
        return None, None

    # Carica il file .wav
    sampling_rate, data = wavfile.read(file_path)

    # Se il file è stereo, usa solo un canale
    if len(data.shape) > 1:
        print("File stereo rilevato. Prendo solo il primo canale.")
        data = data[:, 0]

    print(f"File caricato: {file_path}")
    print(f"Frequenza di campionamento: {sampling_rate} Hz")
    print(f"Durata: {len(data) / sampling_rate:.2f} secondi")
    
    # Se il file è stereo, usa solo un canale
    if len(data.shape) > 1:
        data = data[:, 0]

    # Normalizza il segnale
    data = data / np.max(np.abs(data))

    # Crea variable tempo 
    time = np.linspace(0, len(data) / sampling_rate, num=len(data))

    # Variabili globali valori di default
    selected_range = [0, len(data)]
    selected_time = [0, np.max(time)]
    freq_range = [0, sampling_rate / 2]  # Range massimo (fino a Nyquist)
    
    # Plot dell'Oscillogramma
    ax[0].cla()
    ax[0].plot(time, data, linewidth=0.5, color='black')
    ax[0].set_ylabel("Ampiezza")
    ax[0].set_xlabel("Tempo (s)")
    ax[0].set_title(file_wav)
    

    # Secondo grafico preparazione
    ax[1].cla()
    ax[1].set_xlabel("Frequenza (Hz)")
    ax[1].set_ylabel("Ampiezza")
    
    plt.draw()


def play_audio(event):
    """Riproduce il segmento selezionato dell'audio."""
    if selected_range:
        start, end = selected_range
        segment = data[start:end]  # Estrarre il segmento selezionato
        sd.play(segment, samplerate=sampling_rate)  # Riprodurre il suono

def onselect_time(xmin, xmax):
    # Funzione per selezionare una porzione dell'audio
    global selected_range, selected_time
    idx_min = np.searchsorted(time, xmin)
    idx_max = np.searchsorted(time, xmax)
    selected_range = [idx_min, idx_max]
    selected_time = [xmin, xmax]

    ax[0].cla()
    ax[0].plot(time, data, linewidth=0.5, color='black', alpha=0.3)
    ax[0].plot(time[selected_range[0]:selected_range[1]], data[selected_range[0]:selected_range[1]], 
               linewidth=0.5, color='black', alpha=1)
    ax[0].set_ylabel("Ampiezza")
    ax[0].set_xlabel("Tempo (s)")
    ax[0].set_title(file_wav, fontsize = 20)

    # Aggiorna lo spettro
    spectrum()

    
def onselect_freq(xmin, xmax):
    global freq_range
    freq_range = [xmin, xmax]
    if freq_range[1] - freq_range[0] < 100: 
        freq_range = [0, sampling_rate/2]
    ax[1].set_xlim(freq_range)
    plt.draw()

# Funzione per trovare l'indice più vicino in un array
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]  # Restituisce l'indice e il valore più vicino

# Funzione per aggiornare lo spettro medio
def spectrum():
    global freqs, avg_spectrum, annot  # Manteniamo l'annotazione e le frequenze
    
    # Memorizziamo la posizione dell'annotazione prima di cancellare il grafico
    x_annot, y_annot = annot.xy if annot.get_visible() else (None, None)

    ax[1].cla()  # Cancella il vecchio spettro

    selected_data = data[selected_range[0]:selected_range[1]]
    
    if len(selected_data) > 1:
        try:
            global window_size, window_type, overlap
            
            # Converte i parametri delle caselle di testo
            window_size = int(textbox_window_size.text)
            window_type = textbox_window_type.text.strip().lower()
            overlap = int(textbox_overlap.text) * window_size // 100 

            # Controlla i parametri
            if window_size <= 0 or overlap < 0 or overlap >= window_size:
                print("Errore: Overlap deve essere positivo e minore della finestra.")
                return

            num_windows = (len(selected_data) - window_size) // overlap
            if num_windows <= 0:
                print("Errore: Dimensione finestra troppo grande.")
                return

            # Crea la finestra
            window = get_window(window_type, window_size)

            # Inizializza lo spettro medio
            avg_spectrum = np.zeros(window_size // 2 + 1)

            for i in range(num_windows):
                start = i * overlap
                end = start + window_size
                segment = selected_data[start:end] * window
                fft_segment = np.abs(np.fft.rfft(segment))
                avg_spectrum += fft_segment

            avg_spectrum /= num_windows  # Normalizza

            # Crea l'asse delle frequenze
            freqs = np.fft.rfftfreq(window_size, d=1 / sampling_rate)

            # Plotta lo spettro aggiornato
            ax[1].plot(freqs, avg_spectrum, color='k')
            ax[1].set_xlabel("Frequenza (Hz)")
            ax[1].set_ylabel("Ampiezza")
            ax[1].set_xlim(freq_range)

            # Ricrea l'annotazione dopo il disegno dello spettro
            annot = ax[1].annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="w"), fontsize=10, visible=False)

            # Se l'annotazione era visibile, la ripristiniamo
            if x_annot is not None:
                annot.xy = (x_annot, y_annot)
                annot.set_text(f"x={x_annot:.3f}, y={y_annot:.3f}")
                annot.set_visible(show_coords)

            plt.draw()

        except ValueError:
            print("Errore: Inserire numeri validi per finestra e overlap.")

def update_spectrum(event):
    spectrum()

def stampa_fig(event):
    # Crea la figura con due subplot (Oscillogramma + Spettro)
    fig_st, ax_st = plt.subplots(2, 1, figsize=(6, 8))
    plt.subplots_adjust(bottom=0.1)  
    
    ax_st[0].plot(time, data, linewidth=0.5, color='black', alpha=0.3)
    ax_st[0].plot(time[selected_range[0]:selected_range[1]], data[selected_range[0]:selected_range[1]], 
               linewidth=0.5, color='black', alpha=1)
    ax_st[0].set_ylabel("Ampiezza")
    ax_st[0].set_xlabel("Tempo (s)")
    ax_st[0].set_title(file_wav, fontsize = 14)

# Plotta lo spettro aggiornato
    ax_st[1].plot(freqs, avg_spectrum, color='k')
    ax_st[1].set_xlabel("Frequenza (Hz)")
    ax_st[1].set_ylabel("Ampiezza")
    ax_st[1].set_xlim(freq_range)


# Funzione per aggiornare le coordinate del mouse
def update_cursor(event):
    
    if show_coords and event.inaxes in [ax[0], ax[1]]:  # Controlla se il mouse è nei grafici
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            idx, nearest_freq = find_nearest(freqs, x)
            
            if idx > 5:
                yy = avg_spectrum[idx-1:idx+1]
                max_yy = np.argmax(yy)
                nearest_freq = freqs[idx - 1 + max_yy]
                
            annot.xy = (nearest_freq, y)
            annot.set_text(f"fq={nearest_freq:.3f}")
            annot.set_visible(True)
            fig.canvas.draw_idle()
    else:
        annot.set_visible(False)
        fig.canvas.draw_idle()

# Funzione per attivare/disattivare le coordinate con il tasto destro
def toggle_coords(event):
    global show_coords, cursor_event_id, freqs

    if event.button == 3:  # Tasto destro del mouse
        show_coords = not show_coords  # Cambia stato

        if show_coords:
            # Attiva l'evento solo se è disattivato
            cursor_event_id = fig.canvas.mpl_connect("motion_notify_event", update_cursor)
            print("Coordinate ATTIVATE")
        else:
            # Disattiva l'evento e nasconde il tooltip
            fig.canvas.mpl_disconnect(cursor_event_id)
            annot.set_visible(False)
            fig.canvas.draw_idle()
            print("Coordinate DISATTIVATE")
        spectrum()
        

# Parametri iniziali della FFT
window_size = 1024
window_type = "hamming"
overlap = 50

# Crea la figura con due subplot (Oscillogramma + Spettro)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(bottom=0.3)  # Spazio per i widget

# Variabile per tenere traccia dello stato dell'annotazione
show_coords = False  # Inizia disattivato

# Plot dell'Oscillogramma
ax[0].plot([], [], linewidth=0.5, color='black')
ax[0].set_ylabel("Ampiezza")
ax[0].set_xlabel("Tempo (s)")
ax[0].set_title("", fontsize = 20)

# Creazione dell'annotazione (inizialmente nascosta)
annot = ax[1].annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"), fontsize=10, visible=False)

# Variabile per tenere traccia dello stato dell'annotazione
show_coords = False  # 🟢 Inizia disattivato
cursor_event_id = None  # Memorizza l'evento per disattivarlo

# Collega l'evento del tasto destro alla funzione toggle_coords
fig.canvas.mpl_connect("button_press_event", toggle_coords)



# Creazione delle caselle di testo
axbox1 = plt.axes([0.1, 0.1, 0.10, 0.05])
textbox_window_size = TextBox(axbox1, "Finestra", initial=str(window_size))

axbox2 = plt.axes([0.25, 0.1, 0.10, 0.05])
textbox_window_type = TextBox(axbox2, "Tipo", initial=window_type)

axbox3 = plt.axes([0.45, 0.1, 0.10, 0.05])
textbox_overlap = TextBox(axbox3, "Overlap (%)", initial=str(overlap))

# Creazione del pulsante di aggiornamento
ax_button_aggiorna = plt.axes([0.6, 0.1, 0.1, 0.05])
button_update = Button(ax_button_aggiorna, "Aggiorna")
button_update.on_clicked(update_spectrum)

ax_button_stampa = plt.axes([0.75, 0.1, 0.1, 0.05])
button_stampa = Button(ax_button_stampa, "Stampa")
button_stampa.on_clicked(stampa_fig)

ax_button_apri = plt.axes([0.85, 0.1, 0.1, 0.05])
button_apri = Button(ax_button_apri, "Apri wav")
button_apri.on_clicked(open_file)

ax_button_play = plt.axes([0.1, 0.9, 0.1, 0.05])  # Definizione posizione pulsante
button_play = Button(ax_button_play, "Play")  # Creazione pulsante
button_play.on_clicked(play_audio)  # Collegare il pulsante alla funzione


# Aggiunge l'interattività sull'oscillogramma
span_time = SpanSelector(ax[0], onselect_time, "horizontal", useblit=True)
span_time.set_props(alpha=0.5, facecolor="red")

# Aggiunge l'interattività sullo spettro
span_freq = SpanSelector(ax[1], onselect_freq, "horizontal", useblit=True)
span_freq.set_props(alpha=0.5, facecolor="blue")

plt.show()
