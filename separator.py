"""
Moteur de séparation vocale et d'analyse audio pour Atelier Voix.
Intègre l'algorithme exact et les paramètres officiels d'Ultimate Vocal Remover (audio-separator MDX-Net) :
- UVR-MDX-NET-Inst_HQ_3.onnx (n_fft=6144, dim_f=3072, compensate=1.022) : Isolation Haute Précision Musique / Toutes les Voix
- UVR_MDXNET_KARA_2.onnx     (n_fft=5120, dim_f=2048, compensate=1.065) : Séparation Voix Principale (Lead) / Chœurs & Harmonies
- Supporte également la séparation harmonique (Voix Haute / Voix Basse) sur les voix isolées par l'IA pour les duos harmonisés.
- Accélération matérielle automatique (GPU NVIDIA CUDA si disponible sur le PC Linux, sinon CPU multi-cœurs).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres officiels issus de mdx_model_data.json d'Ultimate Vocal Remover (audio-separator)
MDX_MODELS = {
    "inst_hq": {
        "filename": "UVR-MDX-NET-Inst_HQ_3.onnx",
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx",
        "n_fft": 6144,
        "dim_f": 3072,
        "dim_t": 256,
        "hop": 1024,
        "compensate": 1.022,
        "description": "Isolation Musique / Toutes les voix (UVR-MDX-NET-Inst_HQ_3)",
    },
    "karaoke": {
        "filename": "UVR_MDXNET_KARA_2.onnx",
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx",
        "n_fft": 5120,
        "dim_f": 2048,
        "dim_t": 256,
        "hop": 1024,
        "compensate": 1.065,
        "description": "Séparation Voix Principale / Chœurs & Harmonies (UVR_MDXNET_KARA_2)",
    },
}

ProgressCallback = Optional[Callable[[int, str], None]]


def get_hardware_info() -> Dict[str, object]:
    """Détecte automatiquement les capacités du PC Linux (CPU, RAM, GPU NVIDIA)."""
    cpu_count = os.cpu_count() or 2
    cpu_model = "Processeur Linux"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    gpu_name = None
    has_gpu = False
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_name = res.stdout.strip().splitlines()[0]
                has_gpu = True
        except Exception:
            pass

    onnx_providers = []
    try:
        import onnxruntime as ort

        onnx_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in onnx_providers:
            has_gpu = True
    except Exception:
        pass

    models_downloaded = all(
        (MODELS_DIR / info["filename"]).exists() for info in MDX_MODELS.values()
    )

    return {
        "cpu_model": cpu_model,
        "cpu_threads": cpu_count,
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "onnx_providers": onnx_providers,
        "models_downloaded": models_downloaded,
        "models_dir": str(MODELS_DIR),
    }


def decode_audio_to_stereo(input_path: Path, sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Convertit n'importe quel fichier audio (MP3, WAV, M4A, OGG, FLAC)
    en signal stéréo float32 (shape: [2, N]) à 44100 Hz via FFmpeg.
    """
    tmp_wav = input_path.parent / f"_tmp_decode_{os.getpid()}.wav"
    tmp_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "2",
            "-ar",
            str(sr),
            "-vn",
            "-f",
            "wav",
            str(tmp_wav),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        data, file_sr = sf.read(str(tmp_wav), dtype="float32", always_2d=True)
        return data.T, file_sr
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)


def save_audio_files(
    audio: np.ndarray, sr: int, wav_path: Path, mp3_path: Path
) -> None:
    """Sauvegarde une piste audio (shape [2, N]) en WAV 16-bit et MP3 192 kbps sans écrêtage."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = audio * (0.98 / peak)
    sf.write(str(wav_path), audio.T, sr, subtype="PCM_16")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def compute_waveform_peaks(audio: np.ndarray, num_points: int = 400) -> list[float]:
    """Calcule l'enveloppe d'amplitude (0..1) pour l'affichage visuel dans le studio."""
    mono = np.mean(np.abs(audio), axis=0)
    if len(mono) < num_points:
        return [round(float(x), 4) for x in mono]
    chunk_size = len(mono) // num_points
    trimmed = mono[: chunk_size * num_points].reshape(num_points, chunk_size)
    peaks = np.max(trimmed, axis=1)
    max_p = float(np.max(peaks)) if np.max(peaks) > 1e-6 else 1.0
    return [round(float(p / max_p), 4) for p in peaks]


def ensure_mdx_model(
    model_key: str, progress_cb: ProgressCallback = None
) -> Optional[Path]:
    """Télécharge le modèle ONNX officiel UVR s'il n'est pas encore présent dans ./models/."""
    info = MDX_MODELS[model_key]
    target_path = MODELS_DIR / info["filename"]
    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        return target_path

    tmp_path = target_path.with_suffix(".onnx.part")
    try:
        if progress_cb:
            progress_cb(
                8,
                f"Téléchargement unique du modèle IA {info['filename']} (~55 Mo)...",
            )

        req = urllib.request.Request(
            info["url"], headers={"User-Agent": "Mozilla/5.0 AtelierVoix/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(
            tmp_path, "wb"
        ) as out_file:
            total = int(response.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            block_size = 256 * 1024
            while True:
                buf = response.read(block_size)
                if not buf:
                    break
                out_file.write(buf)
                downloaded += len(buf)
                if total > 0 and progress_cb:
                    pct = int(8 + 10 * (downloaded / total))
                    mb_done = downloaded / (1024 * 1024)
                    mb_tot = total / (1024 * 1024)
                    progress_cb(
                        pct,
                        f"Téléchargement modèle IA ({mb_done:.1f} / {mb_tot:.1f} Mo)...",
                    )
        tmp_path.replace(target_path)
        return target_path
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        print(f"[AtelierVoix] Avertissement téléchargement ONNX ({model_key}): {exc}")
        return None


def run_mdx_onnx(
    mix: np.ndarray,
    model_key: str,
    progress_cb: ProgressCallback = None,
    progress_range: Tuple[int, int] = (15, 55),
    stage_label: str = "Analyse IA UVR MDX-Net",
) -> Optional[np.ndarray]:
    """
    Exécute un modèle ONNX UVR MDX-Net en suivant exactement l'algorithme officiel
    MDXSeparator d'Ultimate Vocal Remover (audio-separator) :
    - Fenêtre de Hann périodique np.hanning(n_fft + 1)[:-1]
    - STFT / ISTFT non normalisée (center=True) identique à torch.stft(..., center=True)
    - Mise à zéro des 3 premières bandes basses fréquences (spek[:, :, :3, :] = 0)
    - Découpage avec trim = n_fft // 2, gen_size = chunk_size - 2 * trim et facteur compensate officiel.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    model_path = ensure_mdx_model(model_key, progress_cb)
    if model_path is None or not model_path.exists():
        return None

    info = MDX_MODELS[model_key]
    n_fft = int(info["n_fft"])
    hop = int(info["hop"])
    dim_f = int(info["dim_f"])
    dim_t = int(info["dim_t"])
    compensate = float(info["compensate"])

    sess_options = ort.SessionOptions()
    cpu_threads = os.cpu_count() or 4
    sess_options.intra_op_num_threads = cpu_threads
    sess_options.inter_op_num_threads = max(1, cpu_threads // 2)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if has_cuda
        else ["CPUExecutionProvider"]
    )

    session = ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers
    )
    input_name = session.get_inputs()[0].name

    # Normalisation par rapport au pic (comme dans MDXSeparator.separate)
    peak = float(np.max(np.abs(mix)))
    if peak < 1e-6:
        return np.zeros_like(mix, dtype=np.float32)
    norm_mix = mix / peak

    n_bins = n_fft // 2 + 1
    trim = n_fft // 2
    chunk_size = hop * (dim_t - 1)
    gen_size = chunk_size - 2 * trim

    orig_len = norm_mix.shape[1]
    pad = gen_size + trim - (orig_len % gen_size)
    mixture = np.concatenate(
        (
            np.zeros((2, trim), dtype=np.float32),
            norm_mix.astype(np.float32),
            np.zeros((2, pad), dtype=np.float32),
        ),
        axis=1,
    )

    # Overlap adaptatif : 0.25 sur GPU, 0.10 sur CPU pour accélérer tout en gardant des transitions douces
    overlap = 0.25 if has_cuda else 0.10
    step = int((1.0 - overlap) * chunk_size)

    result = np.zeros((2, mixture.shape[1]), dtype=np.float32)
    divider = np.zeros((2, mixture.shape[1]), dtype=np.float32)

    stft_win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    starts = list(range(0, mixture.shape[1], step))
    total_chunks = max(1, len(starts))

    for idx, start in enumerate(starts):
        end = min(start + chunk_size, mixture.shape[1])
        chunk_actual = end - start

        mix_part = mixture[:, start:end]
        if chunk_actual < chunk_size:
            mix_part = np.pad(
                mix_part, ((0, 0), (0, chunk_size - chunk_actual)), mode="constant"
            )

        # STFT identique à torch.stft(..., center=True, return_complex=False)
        Zl = librosa.stft(
            mix_part[0], n_fft=n_fft, hop_length=hop, window=stft_win, center=True
        )[:dim_f, :dim_t]
        Zr = librosa.stft(
            mix_part[1], n_fft=n_fft, hop_length=hop, window=stft_win, center=True
        )[:dim_f, :dim_t]

        spek = np.stack([Zl.real, Zl.imag, Zr.real, Zr.imag], axis=0)[
            np.newaxis, ...
        ].astype(np.float32)

        # Règle officielle UVR MDXSeparator : mise à zéro des 3 premiers bins DC/sub-basses
        spek[:, :, :3, :] = 0.0

        spec_pred = session.run(None, {input_name: spek})[0][0]

        Zout_l = np.zeros((n_bins, dim_t), dtype=np.complex64)
        Zout_r = np.zeros((n_bins, dim_t), dtype=np.complex64)
        Zout_l[:dim_f] = spec_pred[0] + 1j * spec_pred[1]
        Zout_r[:dim_f] = spec_pred[2] + 1j * spec_pred[3]

        rec_l = librosa.istft(
            Zout_l,
            hop_length=hop,
            window=stft_win,
            center=True,
            length=chunk_size,
        )
        rec_r = librosa.istft(
            Zout_r,
            hop_length=hop,
            window=stft_win,
            center=True,
            length=chunk_size,
        )
        tar_wave = np.stack([rec_l, rec_r], axis=0)

        win_chunk = np.hanning(chunk_actual).astype(np.float32)[np.newaxis, :]
        # Évite la division par zéro aux extrémités exactes
        win_chunk = np.maximum(win_chunk, 1e-4)

        result[:, start:end] += tar_wave[:, :chunk_actual] * win_chunk
        divider[:, start:end] += win_chunk

        if progress_cb:
            p0, p1 = progress_range
            pct = int(p0 + (p1 - p0) * ((idx + 1) / total_chunks))
            progress_cb(pct, f"{stage_label} ({idx + 1}/{total_chunks})...")

    divider = np.maximum(divider, 1e-6)
    demixed = (result / divider)[:, trim:-trim][:, :orig_len]
    return (demixed * peak * compensate).astype(np.float32)


# ============================================================================
# MOTEUR SPECTRAL / HARMONIQUE & SPATIAL (Complément Duo et Mode Rapide)
# ============================================================================


def isolate_vocals_dsp(
    audio: np.ndarray, sr: int = 44100, progress_cb: ProgressCallback = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sépare toutes les voix (vocals) de l'accompagnement musical (instrumental)
    par filtrage temps-fréquence harmonique/percussif et cohérence vocale (120 Hz - 9 kHz).
    """
    if progress_cb:
        progress_cb(20, "Isolation spectrale Voix / Musique en cours...")

    n_fft = 4096
    hop = 1024
    win = scipy.signal.windows.hann(n_fft, sym=False)

    freqs, _, Z_l = scipy.signal.stft(
        audio[0], fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )
    _, _, Z_r = scipy.signal.stft(
        audio[1], fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )

    mid = 0.5 * (Z_l + Z_r)
    side = 0.5 * (Z_l - Z_r)

    mag_mid = np.abs(mid)
    mag_side = np.abs(side)

    harmonic_mid = scipy.ndimage.median_filter(mag_mid, size=(1, 15))
    percussive_mid = scipy.ndimage.median_filter(mag_mid, size=(31, 1))
    harmonic_mask = (harmonic_mid**2) / (harmonic_mid**2 + percussive_mid**2 + 1e-8)

    freq_weight = np.zeros_like(freqs, dtype=np.float32)
    in_band = (freqs >= 95.0) & (freqs <= 9000.0)
    freq_weight[in_band] = 1.0
    low_trans = (freqs >= 80.0) & (freqs < 150.0)
    freq_weight[low_trans] = (freqs[low_trans] - 80.0) / 70.0
    high_trans = (freqs > 5500.0) & (freqs <= 9500.0)
    freq_weight[high_trans] = 1.0 - 0.6 * ((freqs[high_trans] - 5500.0) / 4000.0)

    center_affinity = (mag_mid**1.5) / (mag_mid**1.5 + 0.65 * (mag_side**1.5) + 1e-8)
    vocal_mask = np.clip(
        harmonic_mask * (0.35 + 0.65 * center_affinity) * freq_weight[:, np.newaxis],
        0.0,
        1.0,
    )
    vocal_mask = scipy.ndimage.gaussian_filter(vocal_mask, sigma=(1.2, 1.5))

    Z_voc_l = Z_l * vocal_mask
    Z_voc_r = Z_r * vocal_mask
    Z_inst_l = Z_l * (1.0 - vocal_mask)
    Z_inst_r = Z_r * (1.0 - vocal_mask)

    orig_len = audio.shape[1]
    _, voc_l = scipy.signal.istft(Z_voc_l, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, voc_r = scipy.signal.istft(Z_voc_r, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, inst_l = scipy.signal.istft(Z_inst_l, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, inst_r = scipy.signal.istft(Z_inst_r, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)

    vocals = np.stack([voc_l[:orig_len], voc_r[:orig_len]], axis=0).astype(np.float32)
    instrumental = np.stack([inst_l[:orig_len], inst_r[:orig_len]], axis=0).astype(np.float32)
    return vocals, instrumental


def split_two_voices_pitch(
    vocals: np.ndarray, sr: int = 44100, progress_cb: ProgressCallback = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sépare un flux vocal contenant 2 voix simultanées (ex: 2 personnes dans les chœurs ou un duo) en :
    - Voix Haute (Aiguë / Soprano / Ténor / Harmonie haute)
    - Voix Basse (Grave / Alto / Baryton / Harmonie basse)
    Utilise un suivi dynamique vectorisé des fondamentales (F0) et un peigne harmonique doux (Wiener).
    """
    if progress_cb:
        progress_cb(82, "Séparation harmonique Voix Haute / Voix Basse...")

    n_fft = 4096
    hop = 1024
    win = scipy.signal.windows.hann(n_fft, sym=False)
    orig_len = vocals.shape[1]

    freqs, _, Z_l = scipy.signal.stft(
        vocals[0], fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )
    _, _, Z_r = scipy.signal.stft(
        vocals[1], fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )

    mag = np.hypot(np.abs(Z_l), np.abs(Z_r)) + 1e-8
    n_bins, n_frames = mag.shape

    f0_mask = (freqs >= 100.0) & (freqs <= 850.0)
    f0_idx = np.where(f0_mask)[0]
    f0_freqs = freqs[f0_idx]

    hps = mag[f0_idx, :].copy()
    for harm in (2, 3):
        harm_idx = np.minimum(f0_idx * harm, n_bins - 1)
        hps *= (mag[harm_idx, :] ** 0.5)

    centroid = np.sum(f0_freqs[:, np.newaxis] * hps, axis=0) / (np.sum(hps, axis=0) + 1e-8)
    centroid = scipy.ndimage.median_filter(centroid, size=25)
    split_f = np.clip(centroid, 230.0, 440.0)[np.newaxis, :]

    f_grid = f0_freqs[:, np.newaxis]
    low_hps = np.where(f_grid < split_f, hps, 0.0)
    high_hps = np.where(f_grid >= split_f, hps, 0.0)

    f_low = scipy.ndimage.median_filter(f0_freqs[np.argmax(low_hps, axis=0)], size=9)
    f_high = scipy.ndimage.median_filter(f0_freqs[np.argmax(high_hps, axis=0)], size=9)

    # Limite le calcul du peigne aux fréquences vocales utiles (< 10 kHz) pour aller 3x plus vite
    max_bin = int(np.searchsorted(freqs, 10000.0))
    freqs_col = freqs[:max_bin, np.newaxis]
    comb_low = np.zeros((max_bin, n_frames), dtype=np.float32)
    comb_high = np.zeros((max_bin, n_frames), dtype=np.float32)

    for k in range(1, 9):
        bw = 24.0 * (1.0 + 0.1 * k)
        comb_low += (1.0 / (k**0.4)) * np.exp(-0.5 * ((freqs_col - k * f_low[np.newaxis, :]) / bw) ** 2)
        comb_high += (1.0 / (k**0.4)) * np.exp(-0.5 * ((freqs_col - k * f_high[np.newaxis, :]) / bw) ** 2)

    reg_high = 1.0 / (1.0 + np.exp(-(freqs_col - split_f * 1.15) / 68.0))
    score_high = comb_high * (0.4 + 0.6 * reg_high) + 0.15 * reg_high
    score_low = comb_low * (0.4 + 0.6 * (1.0 - reg_high)) + 0.15 * (1.0 - reg_high)

    mask_high = np.ones((n_bins, n_frames), dtype=np.float32) * 0.65
    mask_high[:max_bin, :] = (score_high**1.6) / (score_high**1.6 + score_low**1.6 + 1e-8)
    mask_high = scipy.ndimage.gaussian_filter(mask_high, sigma=(1.5, 2.0))
    mask_high = np.clip(mask_high, 0.0, 1.0)
    # Voix 3 (mask_low) reste 100% identique à la version 949ac66
    mask_low = 1.0 - mask_high

    # Dé-étouffement de la Voix 2 (mask_high_clear) :
    # - conserve 100% des formants et consonnes (> 1.8 kHz) sans les trous creusés par comb_low
    # - relève la clarté vocale (+4 dB entre 2 kHz et 10 kHz) et coupe la boue grave (< 150 Hz)
    f_col_all = freqs[:, np.newaxis]
    high_formant_keep = np.clip((f_col_all - 1600.0) / 1400.0, 0.0, 1.0)
    clarity_lift = (
        (1.0 - 0.35 * np.exp(-0.5 * ((f_col_all - 180.0) / 80.0) ** 2))
        * (1.0 + 0.55 * (1.0 / (1.0 + np.exp(-(f_col_all - 2200.0) / 600.0))))
    ).astype(np.float32)
    mask_high_clear = np.maximum(mask_high, 0.88 * high_formant_keep) * clarity_lift

    _, v_high_l = scipy.signal.istft(Z_l * mask_high_clear, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, v_high_r = scipy.signal.istft(Z_r * mask_high_clear, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, v_low_l = scipy.signal.istft(Z_l * mask_low, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, v_low_r = scipy.signal.istft(Z_r * mask_low, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)

    voice_high = np.stack([v_high_l[:orig_len], v_high_r[:orig_len]], axis=0).astype(np.float32)
    voice_low = np.stack([v_low_l[:orig_len], v_low_r[:orig_len]], axis=0).astype(np.float32)
    return voice_high, voice_low


def split_lead_and_backing_dsp(
    vocals: np.ndarray, sr: int = 44100, progress_cb: ProgressCallback = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mode 1 (en mode rapide) : Sépare la Voix Principale (Lead)
    des Chœurs / Harmonies (Backing Vocals).
    """
    if progress_cb:
        progress_cb(72, "Séparation Voix Principale (Lead) / Chœurs & Harmonies...")

    n_fft = 4096
    hop = 1024
    win = scipy.signal.windows.hann(n_fft, sym=False)
    orig_len = vocals.shape[1]

    freqs, _, Z_l = scipy.signal.stft(
        vocals[0], fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )
    _, _, Z_r = scipy.signal.stft(
        vocals[1], fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )

    mid = 0.5 * (Z_l + Z_r)
    side = 0.5 * (Z_l - Z_r)
    mag_mid = np.abs(mid)
    mag_side = np.abs(side)

    stereo_ratio = float(np.sum(mag_side) / (np.sum(mag_mid) + 1e-8))

    if stereo_ratio > 0.12:
        lead_pan_mask = (mag_mid**2) / (mag_mid**2 + (1.8 * mag_side) ** 2 + 1e-8)
        local_max = scipy.ndimage.maximum_filter(mag_mid, size=(9, 5)) + 1e-8
        dominance = (mag_mid / local_max) ** 1.2
        lead_mask = np.clip(0.65 * lead_pan_mask + 0.35 * dominance, 0.0, 1.0)
        lead_mask = scipy.ndimage.gaussian_filter(lead_mask, sigma=(1.2, 2.0))

        _, lead_l = scipy.signal.istft(Z_l * lead_mask, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, lead_r = scipy.signal.istft(Z_r * lead_mask, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, back_l = scipy.signal.istft(Z_l * (1.0 - lead_mask), fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, back_r = scipy.signal.istft(Z_r * (1.0 - lead_mask), fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)

        lead = np.stack([lead_l[:orig_len], lead_r[:orig_len]], axis=0).astype(np.float32)
        backing = np.stack([back_l[:orig_len], back_r[:orig_len]], axis=0).astype(np.float32)
        return lead, backing
    else:
        return split_two_voices_pitch(vocals, sr=sr, progress_cb=progress_cb)


def split_two_voices_stereo(
    vocals: np.ndarray, sr: int = 44100, progress_cb: ProgressCallback = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mode 3 (Duo Spatial) : Sépare 2 voix selon leur position panoramique dans le champ stéréo.
    """
    if progress_cb:
        progress_cb(72, "Séparation spatiale stéréo Voix A / Voix B...")

    n_fft = 4096
    hop = 1024
    win = scipy.signal.windows.hann(n_fft, sym=False)
    orig_len = vocals.shape[1]

    _, _, Z_l = scipy.signal.stft(
        vocals[0], fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )
    _, _, Z_r = scipy.signal.stft(
        vocals[1], fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop
    )

    mag_l = np.abs(Z_l) + 1e-8
    mag_r = np.abs(Z_r) + 1e-8

    pan = (mag_r - mag_l) / (mag_r + mag_l)
    pan_spread = float(np.std(pan))

    if pan_spread > 0.05:
        median_pan = float(np.median(pan))
        mask_a = 1.0 / (1.0 + np.exp((pan - median_pan) / 0.12))
        mask_a = scipy.ndimage.gaussian_filter(mask_a, sigma=(1.5, 2.0))
        mask_b = 1.0 - mask_a

        _, va_l = scipy.signal.istft(Z_l * mask_a, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, va_r = scipy.signal.istft(Z_r * mask_a, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, vb_l = scipy.signal.istft(Z_l * mask_b, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)
        _, vb_r = scipy.signal.istft(Z_r * mask_b, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop)

        voice_a = np.stack([va_l[:orig_len], va_r[:orig_len]], axis=0).astype(np.float32)
        voice_b = np.stack([vb_l[:orig_len], vb_r[:orig_len]], axis=0).astype(np.float32)
        return voice_a, voice_b
    else:
        return split_two_voices_pitch(vocals, sr=sr, progress_cb=progress_cb)


# ============================================================================
# PIPELINE COMPLET DE TRAITEMENT D'UNE CHANSON (3, 4 OU 5 PISTES)
# ============================================================================


def process_song(
    input_file: Path,
    output_dir: Path,
    mode: str = "lead_backing",
    engine: str = "ai_hq",
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    """
    Découpe un fichier audio :
    - Mode lead_2choirs :
      * voix_1 : Voix Principale (Lead)
      * choeurs : Chœurs — Les 2 voix ensemble
      * voix_2 : Chœur 1 — Voix Haute (Aiguë)
      * voix_3 : Chœur 2 — Voix Basse (Grave)
      * musique : Accompagnement instrumental
    - Autres modes (3 pistes) : voix_1, voix_2, musique.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(5, "Lecture et décodage du fichier audio (44.1 kHz Stéréo)...")

    audio, sr = decode_audio_to_stereo(input_file, sr=44100)
    duration_sec = round(float(audio.shape[1] / sr), 2)

    used_ai = False
    vocals = None
    instrumental = None

    # Étape 1 : Séparer toutes les voix de l'accompagnement musical via UVR-MDX-NET-Inst_HQ_3
    if engine == "ai_hq":
        if progress_cb:
            progress_cb(12, "Étape 1/2 : Isolation IA Musique / Voix (UVR-MDX-NET-Inst_HQ_3)...")
        inst_pred = run_mdx_onnx(
            audio,
            model_key="inst_hq",
            progress_cb=progress_cb,
            progress_range=(15, 55),
            stage_label="Étape 1/2 : Isolation IA Musique / Voix",
        )
        if inst_pred is not None:
            instrumental = inst_pred
            vocals = audio - instrumental
            used_ai = True

    if vocals is None or instrumental is None:
        vocals, instrumental = isolate_vocals_dsp(audio, sr=sr, progress_cb=progress_cb)

    # Étape 2 : Séparer les différentes voix humaines
    voice_1 = None
    voice_2 = None
    voice_3 = None
    choir_combined = None

    if mode in ("lead_backing", "lead_2choirs"):
        v1_label = "Voix 1 — Voix Principale (Lead)"
        v2_label = "Voix 2 — Chœurs / Harmonies"
        if engine == "ai_hq":
            if progress_cb:
                progress_cb(58, "Étape 2/2 : Séparation IA Voix Principale / Chœurs (UVR_MDXNET_KARA_2)...")
            backing_pred = run_mdx_onnx(
                vocals,
                model_key="karaoke",
                progress_cb=progress_cb,
                progress_range=(60, 82 if mode == "lead_2choirs" else 88),
                stage_label="Étape 2/2 : Séparation IA Voix Principale / Chœurs",
            )
            if backing_pred is not None:
                voice_2 = backing_pred
                voice_1 = vocals - voice_2

        if voice_1 is None or voice_2 is None:
            voice_1, voice_2 = split_lead_and_backing_dsp(vocals, sr=sr, progress_cb=progress_cb)

        if mode == "lead_2choirs":
            choir_combined = voice_2.copy()
            v2_label = "Voix 2 — Chœur 1 (Voix Haute / Aiguë)"
            v3_label = "Voix 3 — Chœur 2 (Voix Basse / Grave)"
            if progress_cb:
                progress_cb(84, "Étape 3 : Division des Chœurs en 2 voix (Chœur Haute / Chœur Basse)...")
            voice_2, voice_3 = split_two_voices_pitch(voice_2, sr=sr, progress_cb=None)

    elif mode == "duet_pitch":
        v1_label = "Voix 1 — Voix Haute / Principale"
        v2_label = "Voix 2 — Voix Basse / Harmonie"
        if engine == "ai_hq":
            if progress_cb:
                progress_cb(58, "Étape 2/2 : Séparation IA Duo & Harmonies (UVR_MDXNET_KARA_2 + Harmonique)...")
            backing_pred = run_mdx_onnx(
                vocals,
                model_key="karaoke",
                progress_cb=progress_cb,
                progress_range=(60, 80),
                stage_label="Étape 2/2 : Séparation IA Duo & Harmonies",
            )
            if backing_pred is not None:
                lead_ai = vocals - backing_pred
                v_high, v_low = split_two_voices_pitch(lead_ai, sr=sr, progress_cb=progress_cb)
                voice_1 = v_high
                voice_2 = v_low + backing_pred
        if voice_1 is None or voice_2 is None:
            voice_1, voice_2 = split_two_voices_pitch(vocals, sr=sr, progress_cb=progress_cb)

    else:  # duet_stereo
        v1_label = "Voix 1 — Voix A (Gauche / Centre)"
        v2_label = "Voix 2 — Voix B (Droite / Côtés)"
        voice_1, voice_2 = split_two_voices_stereo(vocals, sr=sr, progress_cb=progress_cb)

    if progress_cb:
        progress_cb(90, "Encodage des pistes séparées en MP3 et WAV...")

    stems_data = [
        ("voix_1", v1_label, "#6366f1", voice_1),
    ]
    if choir_combined is not None:
        stems_data.append(("choeurs_combines", "Chœurs — Les 2 voix ensemble", "#a855f7", choir_combined))
    stems_data.append(("voix_2", v2_label, "#ec4899", voice_2))
    if voice_3 is not None:
        stems_data.append(("voix_3", v3_label, "#f59e0b", voice_3))
    stems_data.append(("musique", "Musique / Accompagnement", "#10b981", instrumental))

    stems_meta = []
    for stem_id, label, color, stem_audio in stems_data:
        wav_path = output_dir / f"{stem_id}.wav"
        mp3_path = output_dir / f"{stem_id}.mp3"
        save_audio_files(stem_audio, sr, wav_path, mp3_path)
        peaks = compute_waveform_peaks(stem_audio, num_points=360)
        stems_meta.append(
            {
                "id": stem_id,
                "label": label,
                "color": color,
                "wav_file": f"{stem_id}.wav",
                "mp3_file": f"{stem_id}.mp3",
                "peaks": peaks,
            }
        )

    orig_wav = output_dir / "original.wav"
    orig_mp3 = output_dir / "original.mp3"
    if not orig_wav.exists() or input_file.resolve() != orig_wav.resolve():
        save_audio_files(audio, sr, orig_wav, orig_mp3)

    if progress_cb:
        progress_cb(100, "Séparation terminée avec succès !")

    return {
        "duration": duration_sec,
        "sample_rate": sr,
        "mode": mode,
        "engine_used": "ai_hq" if used_ai else "fast_dsp",
        "stems": stems_meta,
    }


def split_choir_in_song_dir(song_dir: Path) -> Dict[str, object]:
    """
    Prend un morceau déjà découpé (possédant voix_1, voix_2 [Chœurs], musique)
    et conserve la piste Chœurs complète (les 2 voix ensemble) tout en créant :
    - choeurs_combines : Chœurs — Les 2 voix ensemble
    - voix_2 : Voix 2 — Chœur 1 (Voix Haute / Aiguë, dé-étouffée)
    - voix_3 : Voix 3 — Chœur 2 (Voix Basse / Grave)
    sans avoir besoin de recalculer l'IA depuis le début !
    """
    import json
    import time

    meta_path = song_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Sauvegarde une copie de la piste chœurs combinée si pas déjà fait
    choir_combined_wav = song_dir / "choeurs_combines.wav"
    choir_combined_mp3 = song_dir / "choeurs_combines.mp3"
    v2_wav = song_dir / "voix_2.wav"
    if not choir_combined_wav.exists() and v2_wav.exists():
        shutil.copyfile(v2_wav, choir_combined_wav)

    source_wav = choir_combined_wav if choir_combined_wav.exists() else v2_wav
    data, sr = sf.read(str(source_wav), dtype="float32", always_2d=True)
    choir_audio = data.T

    # S'assure que le MP3 de la piste Chœurs (les 2 voix ensemble) existe aussi
    if not choir_combined_mp3.exists():
        save_audio_files(choir_audio, sr, choir_combined_wav, choir_combined_mp3)
    peaks_choir = compute_waveform_peaks(choir_audio, num_points=360)

    choir_high, choir_low = split_two_voices_pitch(choir_audio, sr=sr)

    v2_mp3 = song_dir / "voix_2.mp3"
    v3_wav = song_dir / "voix_3.wav"
    v3_mp3 = song_dir / "voix_3.mp3"

    save_audio_files(choir_high, sr, v2_wav, v2_mp3)
    save_audio_files(choir_low, sr, v3_wav, v3_mp3)

    peaks_high = compute_waveform_peaks(choir_high, num_points=360)
    peaks_low = compute_waveform_peaks(choir_low, num_points=360)

    new_stems = []
    for st in meta["stems"]:
        if st["id"] == "voix_1":
            new_stems.append(st)
        elif st["id"] == "choeurs_combines":
            continue
        elif st["id"] == "voix_2":
            new_stems.append(
                {
                    "id": "choeurs_combines",
                    "label": "Chœurs — Les 2 voix ensemble",
                    "color": "#a855f7",
                    "wav_file": "choeurs_combines.wav",
                    "mp3_file": "choeurs_combines.mp3",
                    "peaks": peaks_choir,
                }
            )
            new_stems.append(
                {
                    "id": "voix_2",
                    "label": "Voix 2 — Chœur 1 (Voix Haute / Aiguë)",
                    "color": "#ec4899",
                    "wav_file": "voix_2.wav",
                    "mp3_file": "voix_2.mp3",
                    "peaks": peaks_high,
                }
            )
            new_stems.append(
                {
                    "id": "voix_3",
                    "label": "Voix 3 — Chœur 2 (Voix Basse / Grave)",
                    "color": "#f59e0b",
                    "wav_file": "voix_3.wav",
                    "mp3_file": "voix_3.mp3",
                    "peaks": peaks_low,
                }
            )
        elif st["id"] == "voix_3":
            continue
        else:
            new_stems.append(st)

    meta["mode"] = "lead_2choirs"
    meta["version"] = int(time.time())
    meta["stems"] = new_stems
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def generate_demo_duet_file(target_path: Path, sr: int = 44100) -> Path:
    """
    Génère un morceau de démonstration de 12 secondes contenant :
    - Une Voix 1 (Soprano/Ténor aiguë avec vibrato et formants vocaux)
    - Une Voix 2 (Alto/Baryton en harmonie à la tierce/sixte avec vibrato)
    - Un accompagnement musical doux (accords + pulsation rythmique)
    """
    duration = 12.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    notes_v1 = [523.25, 587.33, 659.25, 698.46, 659.25, 587.33, 523.25, 659.25]
    notes_v2 = [329.63, 392.00, 261.63, 440.00, 392.00, 349.23, 329.63, 261.63]

    note_len = duration / len(notes_v1)
    v1_sig = np.zeros_like(t)
    v2_sig = np.zeros_like(t)

    for idx, (f1, f2) in enumerate(zip(notes_v1, notes_v2)):
        mask = (t >= idx * note_len) & (t < (idx + 1) * note_len)
        lt = t[mask] - idx * note_len
        env = np.sin(np.pi * lt / note_len) ** 0.45
        env *= np.minimum(1.0, lt / 0.06)

        phase1 = 2 * np.pi * f1 * lt + 0.012 * f1 * np.sin(2 * np.pi * 5.5 * lt)
        phase2 = 2 * np.pi * f2 * lt + 0.012 * f2 * np.sin(2 * np.pi * 5.0 * lt)

        note1 = (
            0.55 * np.sin(phase1)
            + 0.35 * np.sin(2 * phase1)
            + 0.18 * np.sin(3 * phase1)
            + 0.09 * np.sin(4 * phase1)
        )
        note2 = (
            0.60 * np.sin(phase2)
            + 0.40 * np.sin(2 * phase2)
            + 0.22 * np.sin(3 * phase2)
            + 0.12 * np.sin(4 * phase2)
            + 0.06 * np.sin(5 * phase2)
        )
        v1_sig[mask] = note1 * env
        v2_sig[mask] = note2 * env

    chord = (
        0.12 * np.sin(2 * np.pi * 130.81 * t)
        + 0.08 * np.sin(2 * np.pi * 196.00 * t)
        + 0.08 * np.sin(2 * np.pi * 261.63 * t)
    )
    beat_env = (np.sin(2 * np.pi * 2.0 * t) > 0.85).astype(np.float32) * np.exp(
        -18.0 * (t % 0.5)
    )
    rng = np.random.default_rng(42)
    hihat = 0.06 * rng.standard_normal(len(t)).astype(np.float32) * beat_env

    left = 0.42 * v1_sig + 0.28 * v2_sig + chord + hihat
    right = 0.30 * v1_sig + 0.42 * v2_sig + chord + hihat
    stereo = np.stack([left, right], axis=0)

    tmp_wav = target_path.with_suffix(".wav")
    mp3_path = target_path.with_suffix(".mp3")
    save_audio_files(stereo, sr, tmp_wav, mp3_path)
    tmp_wav.unlink(missing_ok=True)
    return mp3_path
