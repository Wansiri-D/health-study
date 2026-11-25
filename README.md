# Hälsostudie – Individuell uppgift

Analys av `health_study_dataset.csv` (800 deltagare) i Jupyter Notebook.

## Struktur
- `data/health_study_dataset.csv` – datasetet
- `src/health_analyzer.py` – funktioner och klassen `HealthAnalyzer` (Del 2)
- `src/__init__.py` – gör `src` till ett paket så att det kan importeras
- `health_study_del1.ipynb` – Del 1: beskrivande analys och statistik (branch `del1`)
- `health_study_del2.ipynb` – Del 2: funktioner, klass, linjär regression
  och fördjupad analys (branch `del2`)

## Python-version
Python 3.14.7

## Kom igång
1. `python -m venv .venv`
2. `source .venv/Scripts/activate` (Windows, Git Bash)
   eller `source .venv/bin/activate` (Mac/Linux)
3. `python -m pip install -r requirements.txt`
4. Öppna en notebook och kör **Restart & Run All**

Notebookarna ska köras från projektets rotmapp, eftersom datan läses med
en relativ sökväg och `src` importeras därifrån.

**OBS:** `requirements.txt` skapades på Windows. Paketet `pywinpty`
fungerar bara på Windows och kan tas bort på Mac/Linux.