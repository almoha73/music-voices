"""
Serveur Web local pour Atelier Voix — Application de séparation vocale et d'entraînement au chant.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from separator import (
    BASE_DIR,
    generate_demo_duet_file,
    get_hardware_info,
    process_song,
    save_audio_files,
    split_choir_in_song_dir,
)

LIBRARY_DIR = BASE_DIR / "library"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Atelier Voix — Séparation Vocale & Studio de Chant")

# Suivi en mémoire des tâches de découpage en cours
JOBS: Dict[str, Dict[str, object]] = {}


class CustomMixRequest(BaseModel):
    volumes: Dict[str, float]  # ex: {"voix_1": 1.0, "voix_2": 0.25, "musique": 0.7}
    pans: Dict[str, float]  # ex: {"voix_1": -0.5, "voix_2": 0.5, "musique": 0.0} dans [-1, 1]
    format: str = "mp3"  # "mp3" ou "wav"


def load_all_songs() -> list[Dict[str, object]]:
    """Charge la liste des morceaux déjà séparés dans ./library/."""
    songs = []
    for meta_file in sorted(
        LIBRARY_DIR.glob("*/metadata.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                songs.append(json.load(f))
        except Exception:
            continue
    return songs


def run_separation_job(
    job_id: str,
    song_id: str,
    title: str,
    input_path: Path,
    mode: str,
    engine: str,
) -> None:
    """Exécute la séparation vocale en arrière-plan et met à jour la progression."""
    song_dir = LIBRARY_DIR / song_id

    def update_progress(pct: int, msg: str) -> None:
        JOBS[job_id]["progress"] = pct
        JOBS[job_id]["message"] = msg

    try:
        result = process_song(
            input_file=input_path,
            output_dir=song_dir,
            mode=mode,
            engine=engine,
            progress_cb=update_progress,
        )
        metadata = {
            "id": song_id,
            "title": title,
            "created_at": time.strftime("%d/%m/%Y %H:%M"),
            "duration": result["duration"],
            "sample_rate": result["sample_rate"],
            "mode": result["mode"],
            "engine_used": result["engine_used"],
            "stems": result["stems"],
        }
        with open(song_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Découpage terminé !"
        JOBS[job_id]["song"] = metadata
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["message"] = f"Erreur lors du traitement : {exc}"
    finally:
        if input_path.exists() and input_path.parent == UPLOADS_DIR:
            input_path.unlink(missing_ok=True)


@app.get("/api/hardware")
def api_hardware() -> Dict[str, object]:
    return get_hardware_info()


@app.get("/api/songs")
def api_list_songs() -> Dict[str, object]:
    return {"songs": load_all_songs()}


@app.post("/api/separate")
async def api_separate_audio(
    file: UploadFile = File(...),
    mode: str = Form("lead_backing"),
    engine: str = Form("fast_dsp"),
) -> Dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni.")

    song_id = uuid.uuid4().hex[:10]
    job_id = uuid.uuid4().hex[:10]
    orig_name = Path(file.filename).stem
    ext = Path(file.filename).suffix or ".mp3"
    upload_path = UPLOADS_DIR / f"{song_id}{ext}"

    with open(upload_path, "wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    JOBS[job_id] = {
        "id": job_id,
        "song_id": song_id,
        "title": orig_name,
        "status": "running",
        "progress": 2,
        "message": "Préparation du fichier audio...",
    }

    thread = threading.Thread(
        target=run_separation_job,
        args=(job_id, song_id, orig_name, upload_path, mode, engine),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "song_id": song_id}


@app.post("/api/demo")
def api_create_demo(
    mode: str = Form("duet_pitch"),
    engine: str = Form("fast_dsp"),
) -> Dict[str, object]:
    """Génère et sépare un morceau de démonstration (Duo Vocal en Harmonie)."""
    song_id = uuid.uuid4().hex[:10]
    job_id = uuid.uuid4().hex[:10]
    demo_base = UPLOADS_DIR / f"demo_{song_id}"
    demo_mp3 = generate_demo_duet_file(demo_base)

    title = "Démo — Duo Vocal Harmonisé (Voix 1 & Voix 2)"
    JOBS[job_id] = {
        "id": job_id,
        "song_id": song_id,
        "title": title,
        "status": "running",
        "progress": 5,
        "message": "Génération du duo de démonstration...",
    }

    thread = threading.Thread(
        target=run_separation_job,
        args=(job_id, song_id, title, demo_mp3, mode, engine),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "song_id": song_id}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str) -> Dict[str, object]:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Tâche introuvable.")
    return JOBS[job_id]


@app.delete("/api/songs/{song_id}")
def api_delete_song(song_id: str) -> Dict[str, object]:
    song_dir = LIBRARY_DIR / song_id
    if song_dir.exists() and song_dir.is_dir():
        shutil.rmtree(song_dir)
    return {"ok": True}


@app.post("/api/songs/{song_id}/split-choir")
def api_split_song_choir(song_id: str) -> Dict[str, object]:
    """
    Divise la piste Chœurs (voix_2) d'un morceau déjà découpé en 2 pistes :
    Voix 2 (Chœur 1 — Voix Haute / Aiguë) et Voix 3 (Chœur 2 — Voix Basse / Grave),
    sans avoir besoin de relancer les 10 minutes d'IA !
    """
    song_dir = LIBRARY_DIR / song_id
    if not (song_dir / "metadata.json").exists():
        raise HTTPException(status_code=404, detail="Morceau introuvable.")
    try:
        updated_meta = split_choir_in_song_dir(song_dir)
        return {"song": updated_meta}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@app.post("/api/songs/{song_id}/mix")
def api_export_custom_mix(song_id: str, req: CustomMixRequest) -> FileResponse:
    """
    Crée à la volée un fichier MP3 ou WAV correspondant aux réglages exacts
    de la table de mixage de l'utilisateur (volumes + panoramique gauche/droite).
    Idéal pour s'exporter un MP3 d'entraînement personnalisé !
    """
    song_dir = LIBRARY_DIR / song_id
    meta_path = song_dir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Morceau introuvable.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mix_audio = None
    sr = 44100

    for stem in meta["stems"]:
        stem_id = stem["id"]
        wav_path = song_dir / stem["wav_file"]
        if not wav_path.exists():
            continue
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        audio = data.T  # [2, N]

        vol = float(req.volumes.get(stem_id, 1.0))
        pan = float(req.pans.get(stem_id, 0.0))  # -1 (Gauche) à +1 (Droite)

        # Loi de panoramique à puissance constante
        angle = (pan + 1.0) * (np.pi / 4.0)
        gain_l = np.cos(angle) * np.sqrt(2.0) / 2.0 * 1.414
        gain_r = np.sin(angle) * np.sqrt(2.0) / 2.0 * 1.414

        styled = np.stack([audio[0] * vol * gain_l, audio[1] * vol * gain_r], axis=0)
        if mix_audio is None:
            mix_audio = styled
        else:
            min_len = min(mix_audio.shape[1], styled.shape[1])
            mix_audio = mix_audio[:, :min_len] + styled[:, :min_len]

    if mix_audio is None:
        raise HTTPException(status_code=400, detail="Aucune piste à mixer.")

    out_wav = song_dir / "mix_entrainement.wav"
    out_mp3 = song_dir / "mix_entrainement.mp3"
    save_audio_files(mix_audio, sr, out_wav, out_mp3)

    safe_title = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in meta["title"]
    )
    if req.format.lower() == "wav":
        return FileResponse(
            str(out_wav),
            media_type="audio/wav",
            filename=f"{safe_title}_Mix_Entrainement.wav",
        )
    return FileResponse(
        str(out_mp3),
        media_type="audio/mpeg",
        filename=f"{safe_title}_Mix_Entrainement.mp3",
    )


app.mount("/library", StaticFiles(directory=str(LIBRARY_DIR)), name="library")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
