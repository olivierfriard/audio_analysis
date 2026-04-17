import json
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

PROMINENCE = 0.09

directory = sys.argv[1]


def load_wav(wav_file):
    """
    Carica il file WAV e ne estrae i dati.
    """

    sampling_rate, data = wavfile.read(wav_file)

    xmin = 0
    duration = len(data) / sampling_rate
    xmax = duration
    if len(data.shape) > 1:
        print("File stereo rilevato. Uso il primo canale.")
        data = data[:, 0]
    data = data / np.max(np.abs(data))
    time = np.linspace(0, len(data) / sampling_rate, num=len(data))
    id_xmin = 0
    id_xmax = len(data)

    return data, sampling_rate


def envelope(data, window_size, overlap, sampling_rate):
    """
    Calcola l'envelope (RMS) usando i parametri correnti e aggiorna l'oscillogramma.
    """

    try:
        if window_size <= 0 or overlap < 0:
            print("Errore: Window size deve essere > 0 e Overlap >= 0")
            return
        rms = librosa.feature.rms(y=data, frame_length=window_size, hop_length=overlap)[
            0
        ]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sampling_rate, hop_length=overlap
        )

        # peaks_times = []
        # canto = np.zeros(len(rms) * overlap)
        return rms, rms_times

    except Exception as e:
        print("Errore in envelope:", e)


def trova_picchi(rms, min_distance, max_distance, sampling_rate, overlap, prominence):
    """
    Trova i picchi dell'inviluppo RMS e li converte nei campioni della registrazione originale.
    """
    try:
        min_distance_samples = int(
            min_distance * (sampling_rate / overlap)
        )  # Converti in campioni
        max_distance_samples = int(
            max_distance * (sampling_rate / overlap)
        )  # Converti in campioni

        # Trova i picchi nell'inviluppo RMS
        peaks, properties = find_peaks(
            rms,
            height=min_amplitude,
            distance=min_distance_samples,
            prominence=prominence,
        )
        peaks_filtered = [peaks[0]]
        # Elimino i picchi troppo distanti dal precedente
        for i in np.arange(1, len(peaks)):
            if peaks[i] - peaks_filtered[-1] < max_distance_samples:
                peaks_filtered.append(peaks[i])

        peaks_filtered = np.array(peaks_filtered)
        mean_distance_between_peaks = np.mean(np.diff(peaks_filtered))
        sdt_distance_between_peaks = np.std(np.diff(peaks_filtered))
        # print("STD=", sdt_distance_between_peaks)
        peaks = [peaks_filtered[0]]
        # ultimo check
        for i in np.arange(1, len(peaks_filtered)):
            # print(peaks_filtered[i] - peaks[-1])
            if (
                peaks_filtered[i] - peaks[-1]
            ) < mean_distance_between_peaks + 3 * sdt_distance_between_peaks:
                peaks.append(peaks_filtered[i])

        peaks_filtered = np.array(peaks)
        # Converti gli indici nei campioni effettivi dell'audio originale

        peaks_times = np.array(peaks_filtered) * overlap / sampling_rate  # In secondi
        peaks_int = rms[peaks_filtered]

    except ValueError:
        print(" Errore: Inserisci valori numerici validi per la distanza e la soglia.")
    except Exception as e:
        print(f"Funzione Trova picchi\n\nError on file {wav_file}\n\n{e}")

    return peaks_times


def trova_ini_fin(
    data,
    rms,
    rms_times,
    peaks_times,
    sampling_rate,
    overlap,
    prominence,
    signal_to_noise_ratio,
    min_amplitude,
):
    # trova inizio
    peaks = peaks_times * sampling_rate / overlap
    peaks = np.asarray(peaks, dtype=np.float64)

    # Se peaks vuoto o poco affidabile, usa il massimo RMS come ancora
    if peaks.size == 0:
        anchor = int(np.argmax(rms))
        peaks = np.array([anchor, anchor], dtype=float)
    elif peaks.size == 1:
        peaks = np.array([peaks[0], peaks[0]], dtype=float)
        rms_noise_ini = np.mean(rms[: int(peaks[0] // 2)])
        rms_noise_fin = np.mean(rms[int(peaks[-1]) :])

    p0 = int(np.clip(peaks[0], 0, len(rms) - 1))
    p1 = int(np.clip(peaks[-1], 0, len(rms) - 1))
    if p1 == p0:  # un solo picco
        mask = rms[p0:] <= min_amplitude
        idx = np.where(mask)[0]
        if len(idx) > 0:
            p1 = p0 + idx[0]
        else:
            p1 = len(rms) - 1

    # -----------------------------
    # 1) Smoothing leggero dell'RMS (anti-buchi)
    # -----------------------------
    w = 11  # prova 5, 7, 9 (deve essere piccolo)
    if len(rms) >= w:
        kernel = np.ones(w) / w
        rms_s = np.convolve(rms, kernel, mode="same")
    else:
        rms_s = rms
        rms_ini = rms[: int(peaks[0])]

    # -----------------------------
    # 2) Stima rumore robusta (percentile + MAD)
    # -----------------------------

    ini_end = max(1, int(p0 * 0.9))
    fin_start = min(len(rms_s) - 1, p1)

    rms_ini_noise = rms_s[:ini_end]
    rms_fin_noise = rms_s[fin_start:] if fin_start < len(rms_s) else rms_s[-1:]

    # Se una delle due porzioni è troppo corta, fallback su tutta la traccia
    if len(rms_ini_noise) < 10:
        rms_ini_noise = rms_s

    if len(rms_fin_noise) < 10:
        rms_fin_noise = rms_s

    # livello rumore = percentile basso (più robusto della media)
    q = 0.6  #  percentile
    noise_ini = np.quantile(rms_ini_noise, q)
    noise_fin = np.quantile(rms_fin_noise, q)

    # MAD (Median absolute deviation)
    def mad(x):
        x = np.asarray(x)
        m = np.median(x)
        return np.median(np.abs(x - m)) + 1e-12

    mad_ini = mad(rms_ini_noise)
    mad_fin = mad(rms_fin_noise)

    # -----------------------------
    # 3) Isteresi: soglia ON / OFF
    # ---
    k_on = signal_to_noise_ratio  # usa il parametro da input
    k_off = max(1.0, 2 * k_on)  # OFF più bassa (60% della ON)

    thr_on_ini = noise_ini + k_on * mad_ini
    thr_off_ini = noise_ini + k_off * mad_ini

    thr_on_fin = noise_fin + k_on * mad_fin
    thr_off_fin = noise_fin + k_off * mad_fin

    print(
        f"thr_on_ini  {thr_off_ini} noise_ini  {noise_ini}  k_on {k_on}   mad_ini {mad_ini}"
    )
    print(
        f"Soglia_IN = {thr_off_ini}, mentre rms totale = {noise_ini}, nframes = {rms_s[:p0]}"
    )
    print(f"RMS_INI_NOIS = {rms_ini_noise}")
    inizio_frame = 0
    in_call = True

    #
    max_back_ms = 100  #
    start_i = max(0, p0 - int((max_back_ms / 1000) * sampling_rate / overlap))

    need_off = 1
    count_off = 0
    inizio_frame = None

    for i in range(p0, start_i - 1, -1):
        if rms_s[i] <= thr_off_ini:
            print(f"condizione soddisfatta:{rms_s[i]}  ")
            count_off += 1
            if count_off >= need_off:
                inizio_frame = i
                break
        else:
            print(f"condizione NON soddisfatta:{rms_s[i]} {thr_off_ini}")
            count_off = 0

    if inizio_frame is not None:
        inizio = int((inizio_frame + 3) * overlap)
    else:
        # fallback molto vicino al picco (pochi ms)
        inizio = int(start_i * overlap)

    print(f"inizio {inizio} p0  {p0}")

    inizio = np.min([p0 * overlap - 1, inizio])
    ini_canto = inizio

    # -----------------------------
    # 5) Trova FINE con isteresi (scorrendo in avanti da p1)
    # -----------------------------
    fine_frame = len(rms_s) - 1
    in_call = True

    for i in range(p1, len(rms_s)):
        v = rms_s[i]
        if in_call:
            if v <= thr_off_fin:
                in_call = False
                fine_frame = i
                break
        else:
            if v >= thr_on_fin:
                in_call = True

    fine = int((fine_frame + 1) * overlap)  # +1 per includere il frame

    fine = max(fine, inizio + 1)
    fine = min(fine, len(rms_s) * overlap)

    canto = np.zeros(len(rms) * overlap)
    canto[inizio:fine] = np.max(rms)
    durata_canto = (fine - inizio) / sampling_rate
    # print(f"durata_canto: inizio ={inizio}, fine = {fine}, durata = fine - inizio, durata_sec = {(fine-inizio)/sampling_rate} ")
    # introduco una variabile globale che mi serve per calcolare correttamente envelope50, envelope80, envelope90
    rms_canto = rms[int(inizio / overlap) : int(fine / overlap)]
    print(f"inizio {inizio}    fine {fine}   len {len(rms_canto)}")

    return inizio, fine, durata_canto


for dir in sorted(list(Path(directory).glob("*"))):
    json_file = dir / Path(dir.name).with_suffix(".json")
    # print(json_file)

    with open(json_file, "r") as f_in:
        parameters = json.load(f_in)

    new_json = dict(parameters)

    for wav_key, wav_block in new_json.items():
        # print(wav_key)
        if "songs" not in wav_block:
            print(wav_key, "no songs found")
            continue
        # change PROMINENCE value
        for song in wav_block["songs"]:
            wav_block["songs"][song]["prominence"] = PROMINENCE
            prominence = wav_block["songs"][song]["prominence"]
            overlap = wav_block["songs"][song]["overlap"]
            window_size = wav_block["songs"][song]["window_size"]
            peaks_times = np.array(wav_block["songs"][song]["peaks_times"])
            signal_to_noise_ratio = wav_block["songs"][song]["signal_to_noise_ratio"]
            min_amplitude = wav_block["songs"][song]["min_amplitude"]
            min_distance = wav_block["songs"][song]["min_distance"]
            max_distance = wav_block["songs"][song]["max_distance"]

            # load wav
            wav_file = (
                json_file.parent
                / Path(wav_key).stem
                / Path(wav_block["songs"][song]["file"])
            )
            if not wav_file.is_file():
                print(f"[WARN] WAV lungo non trovato: {wav_file}")
                continue
            data, sampling_rate = load_wav(wav_file)

            rms, rms_times = envelope(data, window_size, overlap, sampling_rate)

            peaks_times = trova_picchi(
                rms, min_distance, max_distance, sampling_rate, overlap, prominence
            )

            inizio, fine, durata_canto = trova_ini_fin(
                data,
                rms,
                rms_times,
                peaks_times,
                sampling_rate,
                overlap,
                prominence,
                signal_to_noise_ratio,
                min_amplitude,
            )
            print("-" * 20)
            wav_block["songs"][song]["call_duration"] = durata_canto
            wav_block["songs"][song]["peaks_times"] = list(peaks_times)
            wav_block["songs"][song]["pulse_number"] = len(list(peaks_times))

    with open(json_file.with_name(json_file.stem + "_auto.json"), "w") as f_out:
        json.dump(new_json, f_out, indent=2)
