# 🎙️ Atelier Voix — Séparation Vocale & Studio de Répétition

Application locale pour **Linux** permettant de séparer les différentes voix humaines d'un fichier audio (`MP3`, `WAV`, `FLAC`, `M4A`, `OGG`) afin de travailler facilement la voix que l'on doit chanter.

---

## 🚀 1. Comment installer ou transférer sur un autre ordinateur sous Linux ?

Cette application a été conçue pour être **100 % portable** d'un ordinateur Linux à un autre :

1. **Copiez le dossier `music-voices`** sur une clé USB (si vous avez déjà un dossier `.venv` dedans, inutile de copier `.venv`, il sera recréé proprement sur le nouvel ordinateur).
2. Collez le dossier `music-voices` sur le nouvel ordinateur Linux (par exemple sur le Bureau).
3. Ouvrez un terminal dans ce dossier et lancez :
   ```bash
   chmod +x install.sh lancer.sh
   ./install.sh
   ```
   *(Astuce : `install.sh` détecte automatiquement si le nouvel ordinateur possède une carte graphique NVIDIA pour activer l'accélération matérielle IA, ou configure le mode Processeur multi-cœurs sinon, et crée un raccourci sur le Bureau).*
4. Pour démarrer l'application à tout moment :
   ```bash
   ./lancer.sh
   ```
   Le navigateur s'ouvre automatiquement sur **http://127.0.0.1:8080**.

---

## 🎶 2. Les 3 modes de séparation disponibles

* **🎤 Mode 1 — Voix Principale (Lead) + Chœurs / 2e Voix** :
  Sépare le chanteur principal (**Voix 1**), les chœurs / harmonies (**Voix 2**) et l'accompagnement musical (**Musique**).
* **🎶 Mode 2 — Duo Harmonisé (Voix Haute / Voix Basse)** :
  Idéal quand 2 personnes chantent en même temps en harmonie ou en chorale. Sépare la **Voix Haute (Aiguë / Soprano-Ténor)**, la **Voix Basse (Grave / Alto-Baryton)** et la **Musique**.
* **🎧 Mode 3 — Duo Spatial (Gauche-Centre / Droite-Côtés)** :
  Sépare deux voix réparties différemment dans l'espace stéréo.

---

## 🎚️ 3. Outils d'entraînement pour les chanteurs

* **Préréglages en 1 clic** :
  * *Apprendre Voix 1* (Voix 1 à 120 %, Voix 2 baissée à 25 % en repère, Musique à 65 %)
  * *Apprendre Voix 2* (Voix 2 à 120 %, Voix 1 baissée à 25 % en repère, Musique à 65 %)
  * *Une voix par oreille* (Voix 1 dans l'oreille gauche, Voix 2 dans l'oreille droite au casque)
  * *Chanter seul (Karaoké)* (coupe votre voix tout en gardant l'autre voix et la musique)
* **Boucle A–B** : sélectionnez le début `[A]` et la fin `[B]` d'un passage difficile pour le répéter en boucle.
* **Vitesse sans changer la tonalité** (`0.50x` à `1.25x`) : ralentissez un passage rapide en gardant exactement les mêmes notes.
* **Export MP3 / WAV** : téléchargez chaque voix séparément ou exportez votre mix d'entraînement personnalisé.
