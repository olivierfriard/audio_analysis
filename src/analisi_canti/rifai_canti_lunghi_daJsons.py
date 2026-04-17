

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from scipy.io import wavfile
import librosa


@dataclass
class EnvelopeConfig:
    window_ms: float = 1.0
    step_ms: float = 0.1
    amp_rms: float = 1.5


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_wav_mono_normalized(wav_path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    maxabs = np.max(np.abs(data)) if data.size else 0.0
    if maxabs > 0:
        data = data / maxabs
    return sr, data


def _compute_rms_envelope(y: np.ndarray, sr: int, cfg: EnvelopeConfig) -> tuple[np.ndarray, np.ndarray]:
    frame_length = int(cfg.window_ms * sr / 1000.0)
    hop_length = int(cfg.step_ms * sr / 1000.0)
    frame_length = max(frame_length, 1)
    hop_length = max(hop_length, 1)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms = cfg.amp_rms * rms
    return rms_times, rms


def _nearest_rms_at_times(rms_times: np.ndarray, rms: np.ndarray, t: np.ndarray) -> np.ndarray:
    if rms_times.size == 0:
        return np.array([], dtype=float)

    idx = np.searchsorted(rms_times, t)
    idx = np.clip(idx, 0, len(rms_times) - 1)

    prev_idx = np.clip(idx - 1, 0, len(rms_times) - 1)
    choose_prev = np.abs(rms_times[prev_idx] - t) < np.abs(rms_times[idx] - t)
    idx = np.where(choose_prev, prev_idx, idx)

    return rms[idx]


def process_pair(
    json_in: Path,
    wav_forced: Optional[Path] = None,
    out_suffix: str = "_out2",
    cfg: EnvelopeConfig = EnvelopeConfig(),
) -> Path:
    parameters = _read_json(json_in)
    if not isinstance(parameters, dict) or len(parameters) == 0:
        raise ValueError(f"JSON vuoto o non-dizionario: {json_in}")

    wav_key = next(iter(parameters.keys()))
    entry = parameters[wav_key]

    wav_path = wav_forced if wav_forced is not None else (json_in.parent / wav_key)
    if not wav_path.is_file():
        # prova anche se wav_key è già un path
        wk = Path(wav_key)
        if wk.is_file():
            wav_path = wk

    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV non trovato per {json_in.name}: cercato {wav_path}")

    sr, y = _load_wav_mono_normalized(wav_path)
    rms_times, rms = _compute_rms_envelope(y, sr, cfg)

    pt_list = entry.get("peaks_times", [])
    if pt_list:
        pt = np.asarray(pt_list, dtype=float)
        entry["peaks_int"] = _nearest_rms_at_times(rms_times, rms, pt).tolist()
    else:
        entry["peaks_int"] = []

    if rms.size:
        envelope_max = float(np.max(rms))
        entry["envelope50"] = float(np.sum(rms >= 0.5 * envelope_max)) / len(rms)
        entry["envelope80"] = float(np.sum(rms >= 0.8 * envelope_max)) / len(rms)
        entry["envelope90"] = float(np.sum(rms >= 0.9 * envelope_max)) / len(rms)
    else:
        entry["envelope50"] = entry["envelope80"] = entry["envelope90"] = 0.0

    entry["rms_window_ms"] = cfg.window_ms
    entry["rms_step_ms"] = cfg.step_ms
    entry["rms_amp_factor"] = cfg.amp_rms
    entry["sampling_rate_loaded"] = int(sr)
    entry["wav_path_used"] = str(wav_path)

    json_out = json_in.with_name(json_in.stem + out_suffix + json_in.suffix)
    _write_json(json_out, parameters)
    return json_out


from pathlib import Path
from typing import List, Tuple

def read_pairs_from_file(path: Path) -> List[Tuple[Path, Path]]:
    """
    Legge un TXT con:
      - una colonna: json_path
        -> wav ricavato togliendo l'ultimo blocco dopo '_'
      - oppure due colonne con TAB o virgola: json_path  wav_path
    """
    pairs: List[Tuple[Path, Path]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # --- caso 2 colonne ---
        if "\t" in line:
            a, b = line.split("\t", 1)
            pairs.append((Path(a).expanduser(), Path(b).expanduser()))
            continue

        if "," in line:
            a, b = line.split(",", 1)
            pairs.append((Path(a).expanduser(), Path(b).expanduser()))
            continue

        # --- caso 1 colonna: solo JSON ---
        j = Path(line).expanduser()

        parts = j.stem.split("_")
        if len(parts) > 1:
            wav_stem = "_".join(parts[:-1])   # togli l'ultimo pezzo
        else:
            wav_stem = j.stem                 # fallback ultra-sicuro

        w = j.with_name(wav_stem + ".wav")
        pairs.append((j, w))

    return pairs


def choose_file_dialog(title: str, filetypes: list[tuple[str, str]]) -> Optional[Path]:
    # Dialog minimale con tkinter (nessuna GUI persistente)
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filename = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()

    if not filename:
        return None
    return Path(filename)


def main():
    # 1) scegli il txt
    txt_path = choose_file_dialog(
        title="Seleziona il file TXT con la lista di JSON (o coppie JSON/WAV)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if txt_path is None:
        print("Nessun file selezionato. Esco.")
        return

    # 2) (opzionale) config: qui puoi fissare valori globali
    cfg = EnvelopeConfig(window_ms=1.0, step_ms=0.1, amp_rms=1.5)
    out_suffix = "_out2"

    pairs = read_pairs_from_file(txt_path)
    if not pairs:
        print("Il file non contiene righe valide.")
        return

    ok = 0
    skip = 0

    for json_in, wav_in in pairs:
        json_in = json_in.expanduser()
        wav_in = wav_in.expanduser()

        if not json_in.is_file():
            print(f"SKIP (json mancante): {json_in}")
            skip += 1
            continue
        if not wav_in.is_file():
            print(f"SKIP (wav mancante):  {wav_in}")
            skip += 1
            continue

        try:
            out = process_pair(json_in=json_in, wav_forced=wav_in, out_suffix=out_suffix, cfg=cfg)
            print(f"OK: {json_in.name} -> {out.name}")
            ok += 1
        except Exception as e:
            print(f"ERRORE su {json_in}: {e}")
            skip += 1

    print(f"\nFatto. OK={ok}, SKIP/ERRORI={skip}")


if __name__ == "__main__":
    main()
