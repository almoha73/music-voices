#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "⚙️ Première exécution sur cet ordinateur : lancement de l'installation automatique..."
    bash ./install.sh
fi

source .venv/bin/activate

PORT=8080
URL="http://127.0.0.1:$PORT"

echo "============================================================"
echo " 🎙️  Atelier Voix est en cours de démarrage..."
echo " 👉 Interface accessible sur : $URL"
echo " (Appuyez sur Ctrl+C dans cette fenêtre pour fermer l'application)"
echo "============================================================"

# Ouvre automatiquement le navigateur après 1.5 seconde
(sleep 1.5 && (xdg-open "$URL" >/dev/null 2>&1 || sensible-browser "$URL" >/dev/null 2>&1 || true)) &

exec python3 -m uvicorn app:app --host 127.0.0.1 --port "$PORT"
