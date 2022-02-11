# Herzenssache_Abgabe
Dieses Repository enthält den Code für die automatische Erkennung von Vorhofflimmern aus kurzen EKG-Segmenten mittels Deep Learning. Diese Arbeit wurde im Rahmen des Wettbewerbs "Wettbewerb künstliche Intelligenz in der Medizin" an der TU Darmstadt (KIS*MED, Prof. Hoog Antink) durchgeführt.

## Erste Schritte
Die erforderlichen packages können aus der [`requirements.txt`](https://github.com/Orbitery/Herzenssache_Abgabe/blob/main/Files/requirements.txt) Datei entnommen werden.
### Installation
Die erforderlichen Abhängigkeiten können mit dem folgenden Befehl installiert werden:
```
git clone xyz
cd ECG_Classification
pip install --no-deps -r requirements.txt -f 
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


Für ein erfolgreiches benutzerdefiniertes Training wird die Verwendung des Trainingsskripts train.py empfohlen. Hierfür werden folgende Befehle benötigt:

| Argument | Default  Value | Info |
| --- | --- | --- |
| `--modelname` | Resnet | ---. |
| `--bin` | True | Binäre Darstellung. ---. |
| `--pca_active` | False | Binäre Darstellung. ---. |
| `--epochs` | 10 | Ein Vorwärtsdurchlauf und ein Rückwärtsdurchlauf aller Trainingsbeispiele. |
| `--batch_size` | 512 | Die Anzahl der Trainingsbeispiele in einem Vorwärts-/Rückwärtsdurchlauf. |


Die Dateien
 - [`predict_pretrained.pyy`](predict_pretrained.py)
 - [`wettbewerb.py`](wettbewerb.py)
 - [`score.py`](score.py)

stammen aus dem Repository [18-ha-2010-pj](https://github.com/KISMED-TUDa/18-ha-2010-pj) von [Maurice Rohr](https://github.com/MauriceRohr) und [Prof. Hoog Antink](https://github.com/hogius). Die Funktion `predict_labels` in [`predict.py`](https://github.com/Orbitery/Herzenssache_Abgabe/blob/main/Files/predict.py) beinhaltet das folgende Interface, welches für die Evaluierung verwendet wird.

`predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]`

In `model_name` sind die Modelle CNN, LSTM, Random Forest & ResNet enthalten. 

## Daten

Die Daten für das Training so wie die Auswertung der Modelle wurden aus dem Repository [18-ha-2010-pj](https://github.com/KISMED-TUDa/18-ha-2010-pj) von 
[Maurice Rohr](https://github.com/MauriceRohr) und [Prof. Hoog Antink](https://github.com/hogius) verwendet. Weitere Trainingsdaten stammen aus dem PTB-XL-EKG-Datensatz, welche von Wissenschaftlerinnen und Wissenschaftler des Fraunhofer Heinrich-Hertz-Instituts (HHI) und der Physikalisch-Technischen Bundesanstalt (PTB) [hier](https://www.physionet.org/content/ptb-xl/1.0.1/) veröffentlich wurden.

## Verweise

- Resnet Quelle (Idee: https://towardsdatascience.com/using-resnet-for-time-series-data-4ced1f5395e3 , Architecture / Codeursprung: https://github.com/spdrnl/ecg/blob/master/ECG.ipynb
- Paper CNN Ansatz ("ECG Heartbeat Classification Using Convolutional Neural Networks" von Xu und Liu, 2020)
- Philosophie: [Art of README](https://github.com/noffle/art-of-readme)
- Markdown-Beispiele von [Github Markdown](https://guides.github.com/features/mastering-markdown/) und [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
