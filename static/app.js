/**
 * Atelier Voix — Logique client du Studio de Répétition Multipiste
 */

const MODE_DESCRIPTIONS = {
  lead_2choirs:
    "Idéal quand il y a un chanteur principal + 2 personnes qui chantent les chœurs en harmonie (ex: Wagon Wheel) : crée la piste Chœurs (les 2 voix) + Chœur 1 + Chœur 2 !",
  lead_backing:
    "Idéal pour séparer le chanteur principal (Voix 1), les harmonies/chœurs groupés (Voix 2) et la musique.",
  duet_pitch:
    "Idéal quand 2 personnes chantent en même temps en harmonie : sépare la Voix Haute (Aiguë) et la Voix Basse (Grave) + Musique.",
  duet_stereo:
    "Idéal quand les deux chanteurs sont légèrement répartis à gauche et à droite dans l'enregistrement stéréo.",
};

const MODE_BADGES = {
  lead_2choirs: "Mode Lead + Chœurs (2 voix + séparées)",
  lead_backing: "Mode Lead & Chœurs",
  duet_pitch: "Mode Duo Harmonisé (Haute / Basse)",
  duet_stereo: "Mode Duo Spatial (Gauche / Droite)",
};

// État global du studio
let songsLibrary = [];
let activeSong = null;
let selectedFile = null;

// Web Audio API & Pistes synchronisées
let audioCtx = null;
let trackNodes = {}; // { stemId: { audio, source, gainNode, panNode, volume, pan, muted, solo, peaks, color, canvas } }
let isPlaying = false;
let playbackRate = 1.0;
let loopA = null;
let loopB = null;
let loopEnabled = false;
let animFrameId = null;

/**
 * Synchronise les propriétés CSS d'un <input type="range"> (conforme modern-web-guidance)
 */
function syncSliderCSS(slider) {
  if (!slider) return;
  slider.style.setProperty("--attr-min", slider.min || 0);
  slider.style.setProperty("--attr-max", slider.max || 100);
  slider.style.setProperty("--control-value", slider.value);
}

function attachSliderSync(slider, fillColor = null) {
  if (!slider) return;
  if (fillColor) {
    slider.style.setProperty("--slider-fill-color", fillColor);
  }
  syncSliderCSS(slider);
  slider.addEventListener("input", () => syncSliderCSS(slider));
}

function formatTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

function formatPan(panVal) {
  const pct = Math.round(panVal * 100);
  if (Math.abs(pct) <= 3) return "Centre";
  return pct < 0 ? `${Math.abs(pct)}% G` : `${pct}% D`;
}

// ============================================================================
// INITIALISATION & MATÉRIEL LINUX
// ============================================================================

async function loadHardwareInfo() {
  const hwText = document.getElementById("hw-text");
  try {
    const res = await fetch("/api/hardware");
    const info = await res.json();
    const gpuPart = info.has_gpu
      ? `GPU NVIDIA (${info.gpu_name || "CUDA actif"})`
      : `Mode CPU (${info.cpu_threads} threads)`;
    hwText.textContent = `${info.cpu_model} • ${gpuPart}`;
  } catch {
    hwText.textContent = "Serveur local Linux actif";
  }
}

async function loadLibrary(autoSelectId = null) {
  try {
    const res = await fetch("/api/songs");
    const data = await res.json();
    songsLibrary = data.songs || [];
    renderLibraryList();

    if (autoSelectId) {
      const found = songsLibrary.find((s) => s.id === autoSelectId);
      if (found) selectSong(found);
    } else if (!activeSong && songsLibrary.length > 0) {
      selectSong(songsLibrary[0]);
    }
  } catch (err) {
    console.error("Erreur chargement bibliothèque:", err);
  }
}

function renderLibraryList() {
  const listEl = document.getElementById("song-list");
  const countEl = document.getElementById("song-count");
  countEl.textContent = String(songsLibrary.length);

  if (songsLibrary.length === 0) {
    listEl.innerHTML = `<li class="empty-library">Aucun morceau pour l'instant. Importe un MP3 ou lance la Démo !</li>`;
    return;
  }

  listEl.innerHTML = "";
  for (const song of songsLibrary) {
    const li = document.createElement("li");
    li.className = `song-item ${activeSong && activeSong.id === song.id ? "active" : ""}`;
    const modeLabel = MODE_BADGES[song.mode] || "Multipiste";
    li.innerHTML = `
      <div class="song-item-info">
        <div class="song-item-title">${escapeHtml(song.title)}</div>
        <div class="song-item-meta">${formatTime(song.duration)} • ${modeLabel}</div>
      </div>
      <button type="button" class="btn-delete-song" title="Supprimer ce morceau" aria-label="Supprimer ${escapeHtml(song.title)}">🗑</button>
    `;

    li.addEventListener("click", (e) => {
      if (e.target.closest(".btn-delete-song")) return;
      selectSong(song);
    });

    li.querySelector(".btn-delete-song").addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/api/songs/${song.id}`, { method: "DELETE" });
      if (activeSong && activeSong.id === song.id) {
        stopPlayback();
        activeSong = null;
        document.getElementById("studio-workspace").hidden = true;
        document.getElementById("empty-studio").hidden = false;
      }
      await loadLibrary();
    });

    listEl.appendChild(li);
  }
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// ============================================================================
// IMPORTATION DE FICHIER & SUIVI DU DÉCOUPAGE
// ============================================================================

function setupUploadControls() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const fileLabel = document.getElementById("selected-file-label");
  const btnSeparate = document.getElementById("btn-separate");
  const btnDemo = document.getElementById("btn-demo");
  const modeSelect = document.getElementById("mode-select");
  const modeHelp = document.getElementById("mode-help");

  modeSelect.addEventListener("change", () => {
    modeHelp.textContent = MODE_DESCRIPTIONS[modeSelect.value] || "";
  });

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setChosenFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      setChosenFile(fileInput.files[0]);
    }
  });

  function setChosenFile(file) {
    selectedFile = file;
    fileLabel.innerHTML = `Fichier prêt : <strong>${escapeHtml(file.name)}</strong>`;
    btnSeparate.disabled = false;
  }

  btnSeparate.addEventListener("click", async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("mode", modeSelect.value);
    formData.append("engine", document.getElementById("engine-select").value);

    await startJobRequest("/api/separate", formData);
  });

  btnDemo.addEventListener("click", async () => {
    const formData = new FormData();
    formData.append("mode", "duet_pitch");
    formData.append("engine", "fast_dsp");
    await startJobRequest("/api/demo", formData);
  });
}

async function startJobRequest(url, formData) {
  const btnSeparate = document.getElementById("btn-separate");
  const btnDemo = document.getElementById("btn-demo");
  const progressBox = document.getElementById("progress-box");
  const progressStatus = document.getElementById("progress-status");
  const progressPct = document.getElementById("progress-pct");
  const progressFill = document.getElementById("progress-bar-fill");

  btnSeparate.disabled = true;
  btnDemo.disabled = true;
  progressBox.hidden = false;
  progressStatus.textContent = "Envoi et préparation...";
  progressPct.textContent = "2%";
  progressFill.style.width = "2%";

  try {
    const res = await fetch(url, { method: "POST", body: formData });
    if (!res.ok) throw new Error("Erreur serveur");
    const { job_id } = await res.json();

    const poll = setInterval(async () => {
      try {
        const stRes = await fetch(`/api/jobs/${job_id}`);
        const job = await stRes.json();
        progressStatus.textContent = job.message || "Traitement...";
        progressPct.textContent = `${job.progress || 0}%`;
        progressFill.style.width = `${job.progress || 0}%`;

        if (job.status === "completed") {
          clearInterval(poll);
          btnSeparate.disabled = !selectedFile;
          btnDemo.disabled = false;
          setTimeout(() => {
            progressBox.hidden = true;
          }, 1200);
          await loadLibrary(job.song_id);
        } else if (job.status === "error") {
          clearInterval(poll);
          btnSeparate.disabled = !selectedFile;
          btnDemo.disabled = false;
          progressStatus.textContent = job.message || "Erreur pendant le découpage.";
        }
      } catch (e) {
        clearInterval(poll);
        btnSeparate.disabled = !selectedFile;
        btnDemo.disabled = false;
      }
    }, 450);
  } catch (err) {
    progressStatus.textContent = `Erreur : ${err.message}`;
    btnSeparate.disabled = !selectedFile;
    btnDemo.disabled = false;
  }
}

// ============================================================================
// STUDIO MULTIPISTE & WEB AUDIO API
// ============================================================================

function ensureAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
}

function selectSong(song) {
  stopPlayback();
  activeSong = song;
  renderLibraryList();

  document.getElementById("empty-studio").hidden = true;
  document.getElementById("studio-workspace").hidden = false;

  document.getElementById("active-song-title").textContent = song.title;
  document.getElementById("active-song-mode").textContent =
    MODE_BADGES[song.mode] || "Multipiste";
  document.getElementById("total-time").textContent = formatTime(song.duration);
  document.getElementById("current-time").textContent = "0:00";

  const hasVoix3 = (song.stems || []).some((s) => s.id === "voix_3");
  const hasChoirCombined = (song.stems || []).some((s) => s.id === "choeurs_combines");
  const btnSplitChoir = document.getElementById("btn-split-choir");
  if (btnSplitChoir) {
    btnSplitChoir.hidden = hasVoix3;
  }
  const presetLearnChoir = document.getElementById("preset-learn-choir");
  const presetLearnV3 = document.getElementById("preset-learn-v3");
  const presetKaraokeV3 = document.getElementById("preset-karaoke-v3");
  if (presetLearnChoir) presetLearnChoir.hidden = !hasChoirCombined;
  if (presetLearnV3) presetLearnV3.hidden = !hasVoix3;
  if (presetKaraokeV3) presetKaraokeV3.hidden = !hasVoix3;

  const masterSeek = document.getElementById("master-seek");
  masterSeek.max = String(song.duration || 100);
  masterSeek.value = "0";
  syncSliderCSS(masterSeek);

  // Réinitialise boucle A-B
  clearLoop();

  // Construit les pistes Web Audio
  buildTracksUI(song);
}

function buildTracksUI(song) {
  // Nettoyage des pistes précédentes
  for (const key of Object.keys(trackNodes)) {
    const t = trackNodes[key];
    try {
      t.audio.pause();
      t.audio.src = "";
    } catch {}
  }
  trackNodes = {};

  const container = document.getElementById("tracks-container");
  container.innerHTML = "";

  const ver = song.version || song.created_at || 1;
  const hasSeparateChoirs = (song.stems || []).some((s) => s.id === "voix_3");
  for (const stem of song.stems) {
    const wavUrl = `/library/${song.id}/${stem.wav_file}?v=${encodeURIComponent(ver)}`;
    const mp3Url = `/library/${song.id}/${stem.mp3_file}?v=${encodeURIComponent(ver)}`;

    const audioEl = new Audio(wavUrl);
    audioEl.preload = "auto";
    audioEl.crossOrigin = "anonymous";
    audioEl.preservesPitch = true;
    audioEl.mozPreservesPitch = true;
    audioEl.webkitPreservesPitch = true;
    audioEl.playbackRate = playbackRate;

    const card = document.createElement("div");
    card.className = "track-card";
    card.id = `track-card-${stem.id}`;

    card.innerHTML = `
      <div class="track-info-col">
        <div class="track-title-row">
          <div class="track-name">
            <span class="track-dot" style="background:${stem.color}"></span>
            <span>${escapeHtml(stem.label)}</span>
          </div>
          <div class="mute-solo-group">
            <button type="button" class="btn-ms solo" data-stem="${stem.id}" title="Solo : écouter uniquement cette piste">S</button>
            <button type="button" class="btn-ms mute" data-stem="${stem.id}" title="Muet : couper cette piste">M</button>
          </div>
        </div>

        <div class="slider-row">
          <label class="slider-label" for="vol-${stem.id}">Volume</label>
          <input type="range" id="vol-${stem.id}" min="0" max="150" step="1" value="100" />
          <span class="slider-val" id="vol-val-${stem.id}">100%</span>
        </div>

        <div class="slider-row">
          <label class="slider-label" for="pan-${stem.id}">Oreille G/D</label>
          <input type="range" id="pan-${stem.id}" min="-1" max="1" step="0.05" value="0" />
          <span class="slider-val" id="pan-val-${stem.id}">Centre</span>
        </div>
      </div>

      <div class="waveform-wrap" data-stem="${stem.id}" title="Clique sur l'onde sonore pour aller directement à ce passage">
        <canvas class="waveform-canvas" id="canvas-${stem.id}"></canvas>
      </div>

      <div class="track-downloads">
        <span class="track-dl-title">Télécharger cette piste seule :</span>
        <div class="track-dl-buttons">
          <a class="btn-dl" href="${mp3Url}" download="${escapeHtml(song.title)}_${stem.id}.mp3">⬇ MP3</a>
          <a class="btn-dl" href="${wavUrl}" download="${escapeHtml(song.title)}_${stem.id}.wav">⬇ WAV</a>
        </div>
      </div>
    `;

    container.appendChild(card);

    const volSlider = card.querySelector(`#vol-${stem.id}`);
    const panSlider = card.querySelector(`#pan-${stem.id}`);
    attachSliderSync(volSlider, stem.color);
    attachSliderSync(panSlider, "#94a3b8");

    const canvas = card.querySelector(`#canvas-${stem.id}`);
    const initialMuted = stem.id === "choeurs_combines" && hasSeparateChoirs;
    if (initialMuted) {
      card.querySelector(".btn-ms.mute").classList.add("active");
    }

    trackNodes[stem.id] = {
      id: stem.id,
      audio: audioEl,
      source: null,
      gainNode: null,
      panNode: null,
      volume: 1.0,
      pan: 0.0,
      muted: initialMuted,
      solo: false,
      peaks: stem.peaks || [],
      color: stem.color,
      canvas,
      card,
    };

    // Événements Volume, Pan, Solo, Mute
    volSlider.addEventListener("input", () => {
      const v = Number(volSlider.value) / 100;
      const node = trackNodes[stem.id];
      node.volume = v;
      if (v > 0 && node.muted) {
        node.muted = false;
        card.querySelector(".btn-ms.mute").classList.remove("active");
      }
      card.querySelector(`#vol-val-${stem.id}`).textContent = `${Math.round(v * 100)}%`;
      updateAudioGains();
    });

    panSlider.addEventListener("input", () => {
      const p = Number(panSlider.value);
      trackNodes[stem.id].pan = p;
      card.querySelector(`#pan-val-${stem.id}`).textContent = formatPan(p);
      updateAudioGains();
    });

    card.querySelector(".btn-ms.mute").addEventListener("click", (e) => {
      const node = trackNodes[stem.id];
      node.muted = !node.muted;
      e.currentTarget.classList.toggle("active", node.muted);
      if (node.muted && node.solo) {
        node.solo = false;
        card.querySelector(".btn-ms.solo").classList.remove("active");
      }
      updateAudioGains();
    });

    card.querySelector(".btn-ms.solo").addEventListener("click", () => {
      const node = trackNodes[stem.id];
      const willSolo = !node.solo;
      // Mode Solo exclusif pratique : cliquer sur S d'une autre piste bascule directement l'écoute dessus
      for (const other of Object.values(trackNodes)) {
        other.solo = false;
        other.card.querySelector(".btn-ms.solo").classList.remove("active");
      }
      node.solo = willSolo;
      if (willSolo) {
        node.muted = false;
        card.querySelector(".btn-ms.mute").classList.remove("active");
        if (node.volume === 0) {
          node.volume = 1.0;
          volSlider.value = "100";
          syncSliderCSS(volSlider);
          card.querySelector(`#vol-val-${stem.id}`).textContent = "100%";
        }
      }
      card.querySelector(".btn-ms.solo").classList.toggle("active", node.solo);
      updateAudioGains();
    });

    // Clic sur la forme d'onde pour naviguer dans le chant
    card.querySelector(".waveform-wrap").addEventListener("click", (e) => {
      if (!activeSong) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      seekAllTracks(ratio * activeSong.duration);
    });
  }

  // Fin de morceau sur la première piste
  const firstTrack = Object.values(trackNodes)[0];
  if (firstTrack) {
    firstTrack.audio.addEventListener("ended", () => {
      if (loopEnabled && loopA !== null && loopB !== null) {
        seekAllTracks(loopA);
        playAllTracks();
      } else {
        stopPlayback();
      }
    });
  }

  updateAudioGains();
  requestAnimationFrame(() => drawAllWaveforms(0));
}

function connectWebAudioGraphIfNeeded() {
  ensureAudioContext();
  for (const node of Object.values(trackNodes)) {
    // Important : toujours garder le volume HTML5 à 1.0 pour que le GainNode Web Audio ait le contrôle total
    node.audio.volume = 1.0;
    node.audio.muted = false;
    if (!node.source) {
      node.source = audioCtx.createMediaElementSource(node.audio);
      node.gainNode = audioCtx.createGain();
      node.panNode = audioCtx.createStereoPanner();
      node.source
        .connect(node.gainNode)
        .connect(node.panNode)
        .connect(audioCtx.destination);
    }
  }
  updateAudioGains();
}

function getEffectiveVolumes() {
  const nodes = Object.values(trackNodes);
  const anySolo = nodes.some((n) => n.solo);
  const eff = {};
  for (const n of nodes) {
    const audible = anySolo ? n.solo && !n.muted : !n.muted;
    eff[n.id] = audible ? n.volume : 0.0;
  }
  return eff;
}

function updateAudioGains() {
  const eff = getEffectiveVolumes();
  for (const n of Object.values(trackNodes)) {
    const gain = eff[n.id];
    n.card.classList.toggle("muted-track", gain === 0);
    // Toujours laisser l'élément <audio> à 1.0 pour éviter qu'il reste bloqué à 0 après création du GainNode
    n.audio.volume = 1.0;
    n.audio.muted = false;
    if (n.gainNode && audioCtx) {
      n.gainNode.gain.cancelScheduledValues(audioCtx.currentTime);
      n.gainNode.gain.setValueAtTime(gain, audioCtx.currentTime);
    }
    if (n.panNode && audioCtx) {
      n.panNode.pan.cancelScheduledValues(audioCtx.currentTime);
      n.panNode.pan.setValueAtTime(n.pan, audioCtx.currentTime);
    }
  }
}

function getMasterCurrentTime() {
  const first = Object.values(trackNodes)[0];
  return first ? first.audio.currentTime : 0;
}

function seekAllTracks(targetSec) {
  if (!activeSong) return;
  const clamped = Math.max(0, Math.min(activeSong.duration, targetSec));
  for (const n of Object.values(trackNodes)) {
    try {
      n.audio.currentTime = clamped;
    } catch {}
  }
  updateTransportUI(clamped);
}

async function playAllTracks() {
  if (!activeSong) return;
  connectWebAudioGraphIfNeeded();

  const masterTime = getMasterCurrentTime();
  for (const n of Object.values(trackNodes)) {
    n.audio.playbackRate = playbackRate;
    if (Math.abs(n.audio.currentTime - masterTime) > 0.015) {
      n.audio.currentTime = masterTime;
    }
  }

  await Promise.all(
    Object.values(trackNodes).map((n) => n.audio.play().catch(() => {}))
  );
  isPlaying = true;
  const btnPlay = document.getElementById("btn-play");
  btnPlay.textContent = "⏸ Pause";
  btnPlay.classList.add("playing");

  // Alignement fin post-démarrage (élimine tout décalage/écho de 20-80ms entre les pistes au lancement)
  setTimeout(() => {
    if (!isPlaying) return;
    const t = getMasterCurrentTime();
    for (const n of Object.values(trackNodes)) {
      if (!n.audio.seeking && Math.abs(n.audio.currentTime - t) > 0.012) {
        n.audio.currentTime = t;
      }
    }
  }, 90);

  startRenderLoop();
}

function pauseAllTracks() {
  for (const n of Object.values(trackNodes)) {
    n.audio.pause();
  }
  isPlaying = false;
  const btnPlay = document.getElementById("btn-play");
  btnPlay.textContent = "▶ Lecture";
  btnPlay.classList.remove("playing");
  if (animFrameId) cancelAnimationFrame(animFrameId);
}

function stopPlayback() {
  pauseAllTracks();
  seekAllTracks(0);
}

let lastSyncTimestamp = 0;

function startRenderLoop() {
  if (animFrameId) cancelAnimationFrame(animFrameId);

  function tick() {
    if (!isPlaying || !activeSong) return;
    const t = getMasterCurrentTime();
    const now = performance.now();

    // Gestion de la boucle A-B
    if (loopEnabled && loopA !== null && loopB !== null && t >= loopB) {
      seekAllTracks(loopA);
    } else {
      // Resynchronisation anti-écho (toutes les 1.2s si l'écart dépasse 30 ms)
      if (now - lastSyncTimestamp > 1200) {
        lastSyncTimestamp = now;
        for (const n of Object.values(trackNodes)) {
          if (n.audio.paused && !n.audio.ended && !n.audio.seeking) {
            n.audio.play().catch(() => {});
          }
          if (!n.audio.seeking && Math.abs(n.audio.currentTime - t) > 0.03) {
            n.audio.currentTime = t;
          }
        }
      }
      updateTransportUI(t);
    }

    animFrameId = requestAnimationFrame(tick);
  }

  animFrameId = requestAnimationFrame(tick);
}

function updateTransportUI(currentSec) {
  document.getElementById("current-time").textContent = formatTime(currentSec);
  const masterSeek = document.getElementById("master-seek");
  masterSeek.value = String(currentSec);
  syncSliderCSS(masterSeek);
  drawAllWaveforms(currentSec);
}

// ============================================================================
// DESSIN DES FORMES D'ONDE (WAVEFORMS) + ZONE BOUCLE A-B
// ============================================================================

function drawAllWaveforms(currentSec) {
  if (!activeSong) return;
  const duration = activeSong.duration || 1;
  const progressRatio = Math.max(0, Math.min(1, currentSec / duration));

  for (const n of Object.values(trackNodes)) {
    const canvas = n.canvas;
    if (!canvas) continue;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;

    if (canvas.width !== Math.floor(rect.width * dpr)) {
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);

    // Zone de boucle A-B si active
    if (loopA !== null) {
      const xA = (loopA / duration) * W;
      const xB = loopB !== null ? (loopB / duration) * W : xA + 2;
      ctx.fillStyle = loopEnabled
        ? "rgba(245, 158, 11, 0.22)"
        : "rgba(245, 158, 11, 0.1)";
      ctx.fillRect(xA, 0, Math.max(2, xB - xA), H);
    }

    const peaks = n.peaks;
    const count = peaks.length || 1;
    const barW = W / count;
    const midY = H / 2;

    for (let i = 0; i < count; i++) {
      const x = i * barW;
      const ratio = i / count;
      const amp = Math.max(0.04, peaks[i] || 0.04);
      const barH = amp * (H * 0.82);

      ctx.fillStyle = ratio <= progressRatio ? n.color : "rgba(148, 163, 184, 0.32)";
      ctx.fillRect(x, midY - barH / 2, Math.max(1.2, barW - 1), barH);
    }

    // Curseur de lecture
    const playX = progressRatio * W;
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(playX - 1, 0, 2, H);

    ctx.restore();
  }
}

// ============================================================================
// CONTRÔLES DE TRANSPORT, BOUCLE A-B, VITESSE & PRÉRÉGLAGES CHANTEUR
// ============================================================================

function clearLoop() {
  loopA = null;
  loopB = null;
  loopEnabled = false;
  document.getElementById("btn-loop-a").textContent = "Début [A : --:--]";
  document.getElementById("btn-loop-b").textContent = "Fin [B : --:--]";
  const toggle = document.getElementById("btn-loop-toggle");
  toggle.textContent = "Boucle inactive";
  toggle.classList.remove("active");
  toggle.disabled = true;
  document.getElementById("btn-loop-clear").hidden = true;
  drawAllWaveforms(getMasterCurrentTime());
}

function applyPreset(presetName) {
  const hasV3 = Boolean(trackNodes.voix_3);
  const configs = {
    learn_v1: {
      voix_1: { vol: 125, pan: 0, mute: false, solo: false },
      choeurs_combines: { vol: 100, pan: 0, mute: true, solo: false },
      voix_2: { vol: 10, pan: 0, mute: false, solo: false },
      voix_3: { vol: 10, pan: 0, mute: false, solo: false },
      musique: { vol: 65, pan: 0, mute: false, solo: false },
    },
    learn_choir: {
      voix_1: { vol: 10, pan: 0, mute: false, solo: false },
      choeurs_combines: { vol: 125, pan: 0, mute: false, solo: false },
      voix_2: { vol: 100, pan: 0, mute: true, solo: false },
      voix_3: { vol: 100, pan: 0, mute: true, solo: false },
      musique: { vol: 65, pan: 0, mute: false, solo: false },
    },
    learn_v2: {
      voix_1: { vol: 10, pan: 0, mute: false, solo: false },
      choeurs_combines: { vol: 100, pan: 0, mute: true, solo: false },
      voix_2: { vol: 125, pan: 0, mute: false, solo: false },
      voix_3: { vol: 10, pan: 0, mute: false, solo: false },
      musique: { vol: 65, pan: 0, mute: false, solo: false },
    },
    learn_v3: {
      voix_1: { vol: 10, pan: 0, mute: false, solo: false },
      choeurs_combines: { vol: 100, pan: 0, mute: true, solo: false },
      voix_2: { vol: 10, pan: 0, mute: false, solo: false },
      voix_3: { vol: 125, pan: 0, mute: false, solo: false },
      musique: { vol: 65, pan: 0, mute: false, solo: false },
    },
    split_ears: hasV3
      ? {
          voix_1: { vol: 100, pan: 0, mute: false, solo: false },
          choeurs_combines: { vol: 100, pan: 0, mute: true, solo: false },
          voix_2: { vol: 115, pan: -0.85, mute: false, solo: false },
          voix_3: { vol: 115, pan: 0.85, mute: false, solo: false },
          musique: { vol: 60, pan: 0, mute: false, solo: false },
        }
      : {
          voix_1: { vol: 110, pan: -0.8, mute: false, solo: false },
          voix_2: { vol: 110, pan: 0.8, mute: false, solo: false },
          musique: { vol: 65, pan: 0, mute: false, solo: false },
        },
    karaoke_no_v1: {
      voix_1: { vol: 0, pan: 0, mute: true, solo: false },
      choeurs_combines: { vol: 100, pan: 0, mute: true, solo: false },
      voix_2: { vol: 100, pan: 0, mute: false, solo: false },
      voix_3: { vol: 100, pan: 0, mute: false, solo: false },
      musique: { vol: 100, pan: 0, mute: false, solo: false },
    },
    karaoke_no_v2: {
      voix_1: { vol: 100, pan: 0, mute: false, solo: false },
      choeurs_combines: { vol: 100, pan: 0, mute: true, solo: false },
      voix_2: { vol: 0, pan: 0, mute: true, solo: false },
      voix_3: { vol: 100, pan: 0, mute: false, solo: false },
      musique: { vol: 100, pan: 0, mute: false, solo: false },
    },
    karaoke_no_v3: {
      voix_1: { vol: 100, pan: 0, mute: false, solo: false },
      choeurs_combines: { vol: 100, pan: 0, mute: true, solo: false },
      voix_2: { vol: 100, pan: 0, mute: false, solo: false },
      voix_3: { vol: 0, pan: 0, mute: true, solo: false },
      musique: { vol: 100, pan: 0, mute: false, solo: false },
    },
    reset: {
      voix_1: { vol: 100, pan: 0, mute: false, solo: false },
      choeurs_combines: { vol: 100, pan: 0, mute: hasV3, solo: false },
      voix_2: { vol: 100, pan: 0, mute: false, solo: false },
      voix_3: { vol: 100, pan: 0, mute: false, solo: false },
      musique: { vol: 100, pan: 0, mute: false, solo: false },
    },
  };

  const cfg = configs[presetName];
  if (!cfg) return;

  for (const [stemId, settings] of Object.entries(cfg)) {
    const node = trackNodes[stemId];
    if (!node) continue;
    node.volume = settings.vol / 100;
    node.pan = settings.pan;
    node.muted = settings.mute;
    node.solo = settings.solo;

    const volSlider = document.getElementById(`vol-${stemId}`);
    const panSlider = document.getElementById(`pan-${stemId}`);
    if (volSlider) {
      volSlider.value = String(settings.vol);
      syncSliderCSS(volSlider);
      document.getElementById(`vol-val-${stemId}`).textContent = `${settings.vol}%`;
    }
    if (panSlider) {
      panSlider.value = String(settings.pan);
      syncSliderCSS(panSlider);
      document.getElementById(`pan-val-${stemId}`).textContent = formatPan(settings.pan);
    }

    node.card.querySelector(".btn-ms.mute").classList.toggle("active", node.muted);
    node.card.querySelector(".btn-ms.solo").classList.toggle("active", node.solo);
  }

  updateAudioGains();
}

async function splitActiveSongChoir() {
  if (!activeSong) return;
  const btn = document.getElementById("btn-split-choir");
  if (!btn) return;
  const prevText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "⏳ Séparation des 2 voix de chœurs (~25s)...";

  try {
    const res = await fetch(`/api/songs/${activeSong.id}/split-choir`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Échec de la séparation des chœurs");
    await loadLibrary(activeSong.id);
  } catch (err) {
    alert(`Erreur : ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

async function exportCustomMix(format = "mp3") {
  if (!activeSong) return;
  const effVolumes = getEffectiveVolumes();
  const pans = {};
  for (const [id, node] of Object.entries(trackNodes)) {
    pans[id] = node.pan;
  }

  const btn =
    format === "wav"
      ? document.getElementById("btn-export-mix-wav")
      : document.getElementById("btn-export-mix-mp3");
  const prevText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "⏳ Mixage...";

  try {
    const res = await fetch(`/api/songs/${activeSong.id}/mix`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ volumes: effVolumes, pans, format }),
    });
    if (!res.ok) throw new Error("Échec export mix");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeSong.title}_Mix_Entrainement.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Erreur lors de l'export : ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

function setupStudioControls() {
  const masterSeek = document.getElementById("master-seek");
  const speedSlider = document.getElementById("speed-slider");
  attachSliderSync(masterSeek, "#6366f1");
  attachSliderSync(speedSlider, "#10b981");

  document.getElementById("btn-play").addEventListener("click", () => {
    if (isPlaying) pauseAllTracks();
    else playAllTracks();
  });

  document.getElementById("btn-restart").addEventListener("click", () => {
    seekAllTracks(0);
  });

  document.getElementById("btn-rw").addEventListener("click", () => {
    seekAllTracks(getMasterCurrentTime() - 5);
  });

  document.getElementById("btn-ff").addEventListener("click", () => {
    seekAllTracks(getMasterCurrentTime() + 5);
  });

  masterSeek.addEventListener("input", () => {
    seekAllTracks(Number(masterSeek.value));
  });

  // Barre d'espace pour Lecture / Pause
  window.addEventListener("keydown", (e) => {
    if (
      e.code === "Space" &&
      !["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement?.tagName)
    ) {
      e.preventDefault();
      if (isPlaying) pauseAllTracks();
      else playAllTracks();
    }
  });

  // Vitesse de lecture
  function setSpeed(rate) {
    playbackRate = Math.max(0.5, Math.min(1.25, Number(rate)));
    speedSlider.value = String(playbackRate);
    syncSliderCSS(speedSlider);
    document.getElementById("speed-value").textContent = `${playbackRate.toFixed(2)}x`;

    for (const btn of document.querySelectorAll(".btn-speed")) {
      btn.classList.toggle(
        "active",
        Math.abs(Number(btn.dataset.speed) - playbackRate) < 0.02
      );
    }

    for (const n of Object.values(trackNodes)) {
      n.audio.playbackRate = playbackRate;
    }
  }

  speedSlider.addEventListener("input", () => setSpeed(speedSlider.value));
  for (const btn of document.querySelectorAll(".btn-speed")) {
    btn.addEventListener("click", () => setSpeed(btn.dataset.speed));
  }

  // Boucle A-B
  document.getElementById("btn-loop-a").addEventListener("click", () => {
    loopA = getMasterCurrentTime();
    if (loopB !== null && loopB <= loopA) loopB = null;
    document.getElementById("btn-loop-a").textContent = `Début [A : ${formatTime(loopA)}]`;
    document.getElementById("btn-loop-clear").hidden = false;
    drawAllWaveforms(getMasterCurrentTime());
  });

  document.getElementById("btn-loop-b").addEventListener("click", () => {
    const t = getMasterCurrentTime();
    if (loopA === null) loopA = Math.max(0, t - 5);
    if (t <= loopA + 0.5) return;
    loopB = t;
    loopEnabled = true;
    document.getElementById("btn-loop-a").textContent = `Début [A : ${formatTime(loopA)}]`;
    document.getElementById("btn-loop-b").textContent = `Fin [B : ${formatTime(loopB)}]`;
    const toggle = document.getElementById("btn-loop-toggle");
    toggle.disabled = false;
    toggle.textContent = "Boucle active";
    toggle.classList.add("active");
    document.getElementById("btn-loop-clear").hidden = false;
    drawAllWaveforms(getMasterCurrentTime());
  });

  document.getElementById("btn-loop-toggle").addEventListener("click", (e) => {
    if (loopA === null || loopB === null) return;
    loopEnabled = !loopEnabled;
    e.currentTarget.textContent = loopEnabled ? "Boucle active" : "Boucle en pause";
    e.currentTarget.classList.toggle("active", loopEnabled);
    drawAllWaveforms(getMasterCurrentTime());
  });

  document.getElementById("btn-loop-clear").addEventListener("click", clearLoop);

  // Préréglages Chanteur
  for (const btn of document.querySelectorAll(".btn-chip[data-preset]")) {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  }

  // Bouton Diviser les Chœurs en 2 voix (sur un morceau déjà séparé)
  const btnSplitChoir = document.getElementById("btn-split-choir");
  if (btnSplitChoir) {
    btnSplitChoir.addEventListener("click", splitActiveSongChoir);
  }

  // Export Mix personnalisé
  document
    .getElementById("btn-export-mix-mp3")
    .addEventListener("click", () => exportCustomMix("mp3"));
  document
    .getElementById("btn-export-mix-wav")
    .addEventListener("click", () => exportCustomMix("wav"));

  window.addEventListener("resize", () =>
    drawAllWaveforms(getMasterCurrentTime())
  );
}

// Démarrage
document.addEventListener("DOMContentLoaded", () => {
  loadHardwareInfo();
  setupUploadControls();
  setupStudioControls();
  loadLibrary();
});
