# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
from re import A, L
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
import os
from scipy.signal.spectral import periodogram

import tensorflow as tf
import tensorflow.keras as keras
from ecgdetectors import Detectors
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from keras.models import load_model

from decide import *
from modelloader import *


###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

    #------------------------------------------------------------------------------
    # Euer Code ab hier  

    model = modelload(is_binary_classifier,model_name)
    




    if (model_name == "CNN"):
        if (is_binary_classifier==True):  #Beginn des binären Klassifizierers
            cnn_model = load_model("./CNN_Model/model_bin.hdf5")
            print('Model Loaded!')

            data_names = []
            data_samples = []
            r_peaks_list = []

            detectors = Detectors(fs)  
            for idx, ecg_lead in enumerate(ecg_leads):
                ecg_lead = ecg_lead.astype('float')  # Wandel der Daten von Int in Float32 Format für CNN später
                ecg_lead = (ecg_lead - ecg_lead.mean()) 
                ecg_lead = ecg_lead / (ecg_lead.std() + 1e-08)  
                r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
                sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
                for r_peak in r_peaks:
                    if r_peak > 150 and r_peak + 150 <= len(ecg_lead):
                        data_samples.append(ecg_lead[r_peak - 150:r_peak + 150])
                        data_names.append(ecg_names[idx])


            predicted = model.predict(data_samples)
            predictions = decider(predicted, ecg_names,data_samples, is_binary_classifier)
           



        elif(is_binary_classifier==False):   #Beginn des Multilabel-Klassifizierers
            cnn_model = load_model("./CNN_Model/model_multi.hdf5")
            print('Model Loaded!')
            data_names = []
            data_samples = []
            r_peaks_list = []

            detectors = Detectors(fs)  
            for idx, ecg_lead in enumerate(ecg_leads):
                ecg_lead = ecg_lead.astype('float')  # Wandel der Daten von Int in Float32 Format für CNN später
                r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
                sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
                for r_peak in r_peaks:
                    if r_peak > 150 and r_peak + 150 <= len(ecg_lead):
                        data_samples.append(ecg_lead[r_peak - 150:r_peak + 150])
                        data_names.append(ecg_names[idx])


            predicted = model.predict(data_samples) # Hier auch
            predictions = decider(predicted, ecg_names,data_samples, is_binary_classifier)
                
    #------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               