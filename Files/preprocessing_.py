import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv
import os
from statistics import mode
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras as keras
from ecgdetectors import Detectors
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import pickle
from PCA_ import *
from save_models import *
from plotter import *
from Models_ import *

from sklearn.decomposition import PCA


def BP_Filter(ecg_lead): 

    '''
    Parameters
    ----------
    ecg_leads : list of numpy-Arrays
        EKG-Signale.

    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''


    fs = 300  # Sampling frequency

    fc = 30  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency

    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, ecg_lead)
    d, c = signal.butter(5, w, 'high')
    ecg_lead = signal.filtfilt(d, c, output)
    return ecg_lead

def Scaler(ecg_lead):

    '''
    Parameters
    ----------
    ecg_leads : list of numpy-Arrays
        EKG-Signale.

    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''


    number = np.random.uniform(low=0.0, high=1.0, size=None)
    ecg_lead = ecg_lead * number
    return ecg_lead



def Preprocessing(ecg_leads,ecg_labels,fs,ecg_names,modelname,bin):
    """[summary]

    Args:
        ecg_leads ([type]): [description]
        ecg_labels ([type]): [description]
        fs ([type]): [description]
        ecg_names ([type]): [description]
        modelname ([type]): [description]
        bin ([type]): [description]

    Returns:
        [type]: [description]
    """


    detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors

    train_labels = []
    train_samples = []
    r_peaks_list = []
    bin = bin

    line_count = 0
    if (bin=="True"):
        print("Start Binary Data Preprocessing")
        for idx, ecg_lead in enumerate(ecg_leads):
            ecg_lead = ecg_lead.astype('float')  # Wandel der Daten von Int in Float32 Format für CNN später
            ecg_lead = (ecg_lead - ecg_lead.mean()) 
            ecg_lead = ecg_lead / (ecg_lead.std() + 1e-08)  
            r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
            if ecg_labels[idx] == 'N' or ecg_labels[idx] == 'A':
                for r_peak in r_peaks:
                    if r_peak > 150 and r_peak + 450 <= len(ecg_lead): 
                        train_samples.append(ecg_lead[r_peak - 150:r_peak + 450]) #Einzelne Herzschläge werden separiert und als Trainingsdaten der Länge 300 abgespeichert
                        train_labels.append(ecg_labels[idx])

            line_count = line_count + 1
            if (line_count % 100)==0:
                print(f"{line_count} Dateien wurden verarbeitet.")
            if line_count == 200:  #Für Testzwecke kann hier mit weniger Daten gearbeitet werden.
                break
                #pass

        tf.keras.layers.Softmax(axis=-1)
        if(modelname=="CNN" or modelname=="LSTM" or modelname=="Resnet"):
            # Klassen in one-hot-encoding konvertieren
            # 'N' --> Klasse 0
            # 'A' --> Klasse 1
            train_labels = [0 if train_label == 'N' else train_label for train_label in train_labels]
            train_labels = [1 if train_label == 'A' else train_label for train_label in train_labels] 
            train_labels = keras.utils.to_categorical(train_labels)
        elif(modelname=="RandomForrest"):
            pass

    elif (bin=="False"):
        print("Start Multilabel Data Preprocessing")
        for idx, ecg_lead in enumerate(ecg_leads):
            ecg_lead = ecg_lead.astype('float')  # Wandel der Daten von Int in Float32 Format für CNN später
            ecg_lead = (ecg_lead - ecg_lead.mean()) 
            ecg_lead = ecg_lead / (ecg_lead.std() + 1e-08) 
            r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
            for r_peak in r_peaks:
                if r_peak > 150 and r_peak + 450 <= len(ecg_lead):
                    train_samples.append(ecg_lead[r_peak - 150:r_peak + 450]) #Einzelne Herzschläge werden separiert und als Trainingsdaten der Länge 300 abgespeichert
                    train_labels.append(ecg_labels[idx])

            line_count = line_count + 1
            if (line_count % 100)==0:
                print(f"{line_count} Dateien wurden verarbeitet.")
            
            if line_count == 100:    #Für Testzwecke kann hier mit weniger Daten gearbeitet werden.
                break
                #pass


        if(modelname=="CNN" or modelname=="LSTM"  or modelname=="Resnet"):
            # Klassen in one-hot-encoding konvertieren
            # 'N' --> Klasse 0
            # 'A' --> Klasse 1
            # 'O' --> Klasse 2
            # '~' --> Klasse 3
            train_labels = [0 if train_label == 'N' else train_label for train_label in train_labels]
            train_labels = [1 if train_label == 'A' else train_label for train_label in train_labels]
            train_labels = [2 if train_label == 'O' else train_label for train_label in train_labels]
            train_labels = [3 if train_label == '~' else train_label for train_label in train_labels]
            train_labels = keras.utils.to_categorical(train_labels)
        elif(modelname=="RandomForrest"):
            pass

    X_train, X_test, y_train, y_test = train_test_split(train_samples, train_labels, test_size=0.2)

    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    X_train = X_train.reshape((*X_train.shape, 1))
    X_test = X_test.reshape((*X_test.shape, 1))


    return (train_labels,train_samples,r_peaks_list,X_train, y_train, X_test, y_test)

