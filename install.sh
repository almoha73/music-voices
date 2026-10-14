#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
APP_DIR="$(pwd)"

echo "============================================================"
echo " 🎙️  Installation de Atelier Voix (Linux)"
echo " Dossier : $APP_DIR"
echo "============================================================"

# 1. Vérification de Python 3 et FFmpeg
MISSING_PKGS=""
if ! command -v python3 >/dev/null 2>&1; then
    MISSING_PKGS="$MISSING_PKGS python3"
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
    MISSING_PKGS="$MISSING_PKGS python3-venv"
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    MISSING_PKGS="$MISSING_PKGS ffmpeg"
fi

if [ -n "$MISSING_PKGS" ]; then
    echo "📦 Installation des outils système requis :$MISSING_PKGS"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y $MISSING_PKGS
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y $MISSING_PKGS
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm $MISSING_PKGS
    else
        echo "⚠️ Veuillez installer manuellement :$MISSING_PKGS"
        exit 1
    fi
fi

# 2. Création de l'environnement virtuel Python (.venv)
if [ ! -d ".venv" ]; then
    echo "🐍 Création de l'environnement Python isolé (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

# 3. Installation des bibliothèques Python
echo "📥 Installation des bibliothèques audio et IA..."
pip install -r requirements.txt

# 4. Détection automatique d'une carte graphique NVIDIA sur ce PC Linux
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "🚀 Carte graphique NVIDIA détectée ! Activation de l'accélération GPU (onnxruntime-gpu)..."
    pip install onnxruntime-gpu || true
else
    echo "💻 Mode Processeur (CPU) configuré."
fi

chmod +x "$APP_DIR/lancer.sh"

# 5. Création d'un raccourci sur le Bureau Linux si disponible
DESKTOP_DIR=""
if [ -d "$HOME/Bureau" ]; then
    DESKTOP_DIR="$HOME/Bureau"
elif [ -d "$HOME/Desktop" ]; then
    DESKTOP_DIR="$HOME/Desktop"
fi

if [ -n "$DESKTOP_DIR" ]; then
    SHORTCUT="$DESKTOP_DIR/Atelier-Voix.desktop"
    cat > "$SHORTCUT" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Atelier Voix
Comment=Séparation des voix et studio de répétition
Exec=$APP_DIR/lancer.sh
Icon=audio-x-generic
Terminal=true
Categories=AudioVideo;Audio;Music;
EOF
    chmod +x "$SHORTCUT"
    echo "✅ Raccourci créé sur le Bureau : $SHORTCUT"
fi

echo ""
echo "============================================================"
echo " ✅ Installation terminée avec succès !"
echo " Pour démarrer l'application, tapez : ./lancer.sh"
echo "============================================================"
