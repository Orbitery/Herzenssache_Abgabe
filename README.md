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


Für ein erfolgreiches benutzerdefiniertes Training wird die Verwendung des Trainingsskripts train.py empfohlen. Hierfür werden folgende Befehle benötigt:

| Argument | Default Value | Info |
| --- | --- | --- |
| `--cuda_devices` | "0" | String of cuda device indexes to be used. Indexes must be separated by a comma. |
| `--no_data_aug` | False | Binary flag. If set no data augmentation is utilized. |
| `--data_parallel` | False | Binary flag. If set data parallel is utilized. |
| `--epochs` | 100 | Number of epochs to perform while training. |
| `--lr` | 1e-03 | Learning rate to be employed. |
| `--physio_net` | False | Binary flag. Utilized PhysioNet dataset instead of default one. |
| `--batch_size` | 24 | Number of epochs to perform while training. |
| `--dataset_path` | False | Path to dataset. |
| `--network_config` | "ECGCNN_M" | Type of network configuration to be utilized. |
| `--load_network` | None | If set given network (state dict) is loaded. |
| `--no_signal_encoder` | False | Binary flag. If set no signal encoder is utilized. |
| `--no_spectrogram_encoder` | False | Binary flag. If set no spectrogram encoder is utilized. |
| `--icentia11k` | False | Binary flag. If set icentia11k dataset is utilized. |
| `--challange` | False | Binary flag. If set challange split is utilized. |
| `--two_classes` | False | Binary flag. If set two classes are utilized. Can only used with PhysioNet dataset and challange flag. |

Die Dateien
 - predict_pretrained.py
 - wettbewerb.py
 - score.py

stammen aus dem Repository [18-ha-2010-pj](https://github.com/KISMED-TUDa/18-ha-2010-pj) von [Maurice Rohr](https://github.com/MauriceRohr) und [Prof. Hoog Antink](https://github.com/hogius). Die Funktion `predict_labels` in [`predict.py`](predict.py) beinhaltet das folgende Interface, welches für die Evaluierung verwendet wird.

`predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]`

In `model_name` sind die Modelle CNN, LSTM, Random Forest & ResTNet enthalten. 

## Daten

Die Daten für das Training so wie die Auswertung der Modelle wurden aus dem Repository [18-ha-2010-pj](https://github.com/KISMED-TUDa/18-ha-2010-pj) von 
[Maurice Rohr](https://github.com/MauriceRohr) und [Prof. Hoog Antink](https://github.com/hogius) verwendet. Weitere Trainingsdaten stammen aus dem PTB-XL-EKG-Datensatz, welche von Wissenschaftlerinnen und Wissenschaftler des Fraunhofer Heinrich-Hertz-Instituts (HHI) und der Physikalisch-Technischen Bundesanstalt (PTB) [hier](https://www.physionet.org/content/ptb-xl/1.0.1/) veröffentlich wurden.

## Verweise

- Philosophie: [Art of README](https://github.com/noffle/art-of-readme)
- Markdown-Beispiele von [Github Markdown](https://guides.github.com/features/mastering-markdown/) und [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

