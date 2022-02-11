# Herzenssache_Abgabe
Dieses Repository enthält den Code für die automatische Erkennung von Vorhofflimmern aus kurzen EKG-Segmenten mittels Deep Learning. Diese Arbeit wurde im Rahmen des Wettbewerbs "Wettbewerb künstliche Intelligenz in der Medizin" an der TU Darmstadt (KIS*MED, Prof. Hoog Antink) durchgeführt.

## Erste Schritte
Die erforderlichen packages können aus der requirements.txt Datei entnommen werden.
### Installation
abc
### Abhängigkeiten

Die erforderlichen Abhängigkeiten können mit dem folgenden Befehl installiert werden:
```
git clone https://github.com/ChristophReich1996/ECG_Classification
cd ECG_Classification
pip install --no-deps -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
cd ecg_classification/pade_activation_unit/cuda
python setup.py install
cd ../../../
```
## Verwendung

1. Repository klonen `git clone https://...`
2. README bearbeiten `subl README.md`
3. Änderungen einchecken `git add README.md` und committen `git commit -m "Update README"`

## Funktionen

Binäres Problem:
- python predict_pretrained.py --model_name CNN_bin

Multi-Class Problem:
- python predict_pretrained.py --model_name CNN_multi

## Daten

- Beispiel-Readme auf Englisch hinzufügen
- Einstellungen im Text-Editor ausführlicher beschreiben

## Verweise

- Philosophie: [Art of README](https://github.com/noffle/art-of-readme)
- Markdown-Beispiele von [Github Markdown](https://guides.github.com/features/mastering-markdown/) und [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
