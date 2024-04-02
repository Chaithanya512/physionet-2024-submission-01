#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################


import numpy as np
import os
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

from itertools import chain
import joblib
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from scipy.signal import resample
from scipy.signal import detrend
from scipy.stats import entropy


class Preprocessing_Without_Contours():
    def __init__(self,image):
        self.image = np.array(image)
        # self.image.shape)
        # self.img1 = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
    def check(self):
       
        # img1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if len(self.image.shape) == 3:
            img1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img1 = self.image
        _,thre = cv2.threshold(img1,30,255,cv2.THRESH_BINARY)
        thre = thre[600:800, 50:550]
   
   
        b = False
        if np.count_nonzero(thre == 0) > 250:
            b = True
        else:
            b = False
        return b
    def grid_removal(self,b):
        if len(self.image.shape) == 3:

            img1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img1 = self.image

        if b:
            _, img1 = cv2.threshold(img1, 30 , 255, cv2.THRESH_BINARY)
        else:
            kernel = np.ones((1, 1), np.uint8)
            eroded_image = cv2.erode(img1, kernel, iterations=1)
            ret, img1 = cv2.threshold(eroded_image, 50, 255, cv2.THRESH_BINARY)
            
    
        img1 = cv2.resize(img1,(2200,1700))  
        
        return img1
    


    def Divide_leads(self,img):
        image = img
        Lead_1 = image[600:800, 50:550]  # Lead 1
        Lead_4 = image[600:800, 600:1100]  # Lead aVR
        Lead_7 = image[600:800, 1060:1560]  # Lead V1
        Lead_10 = image[600:800, 1550:2050]  # Lead V4
        Lead_2 = image[800:1200, 100:600]  # Lead 2
        Lead_5 = image[800:1200, 600:1100]  # Lead aVL
        Lead_8 = image[800:1200, 1060:1560]  # Lead V2
        Lead_11 = image[800:1200, 1550:2050]  # Lead V5
        Lead_3 = image[1200:1400, 100:600]  # Lead 3
        Lead_6 = image[1200:1400, 600:1100]  # Lead aVF
        Lead_9 = image[1200:1400, 1060:1560]  # Lead V3
        Lead_12 = image[1200:1400, 1550:2050]  # Lead V6
        # Lead_13 = image[1400:1600, 100:600]  # Long Lead

        ll = [Lead_1, Lead_2, Lead_3, Lead_4,Lead_5, Lead_6, Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12]

        return ll
    



    def image_extraction(self , img,resample = False):

        ll = Preprocessing_Without_Contours.Divide_leads(self ,img = img)
        signals = []
        for i in ll:

            recreated = 180 - i.argmin(axis=0)
            # ecg_signal_scaled= 2 * ((recreated) - np.min(recreated)) / (np.max(recreated) - np.min(recreated)) - 1

             
            
            signals.append(recreated.reshape(500,)) 

        return np.array(signals)

    def repeat_signals(self , signals):
        final = []
        
        original_sampling_rate = 500
        new_sampling_rate = 1300

        # Calculate the ratio of new sampling rate to original sampling rate
        resample_ratio = new_sampling_rate / original_sampling_rate
        resample_ratio1 = 1250 / 1215

        # Calculate the new length of the resampled signal
        
        for i in signals:
            
            new_length = int(len(i) * resample_ratio)

            # Resample the signal
            i = resample(i, new_length)
            i = i[55:1270]
            new_length = int(len(i) * resample_ratio1)

            # Resample the signal
            i = resample(i, new_length)
            repeated_signal = np.tile(i, 4)[:5000]
            final.append(repeated_signal.reshape(5000,))

        return np.array(final)



# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    if verbose:
        print('Done.')
        print()

# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()
    dxs = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = load_dx(record)
        if dx:
            current_features = extract_features(record)
            # print("hi")
            current_features = get_eeg_features2(current_features)

            features.append(current_features)
            dxs.append(dx)

    if not dxs:
        raise Exception('There are no labels for the data.')

    features = np.vstack(features)
    classes = sorted(set.union(*map(set, dxs)))
    dxs = compute_one_hot_encoding(dxs, classes)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

   
    model = XGBClassifier()

    # Defining the hyperparameters grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.001]
    }
   
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

    grid_search.fit(features, dxs)

    model = grid_search.best_estimator_

    # model.fit(features,dxs)

    

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_dx_model(model_folder, model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.


def load_digitization_model(model_folder, verbose):

    return 0

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.


def load_dx_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'dx_model.sav')
    return joblib.load(filename)

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.

def run_digitization_model(digitization_model, record, verbose):


    # Extract features.
    signal = extract_features(record)

    signal = signal.T

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.


def run_dx_model(dx_model, record, signal, verbose):
    model = dx_model['model']
    classes = dx_model['classes']

    # Extract features.
    
    features = extract_features(record)
    features = get_eeg_features2(features)
    features = features.reshape(1, -1)

    # Get model probabilities.
    probabilities = model.predict_proba(features)
 

    probabilities = np.asarray(probabilities, dtype=np.float32)[0]

    # Choose the class(es) with the highest probability as the label(s).
    max_probability = np.nanmax(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

    return labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):

    image = load_image(record)

    image = image[0].convert("L")

    obj = Preprocessing_Without_Contours(image)
    vr = obj.check()
    image = obj.grid_removal(b=vr)

    signals = obj.image_extraction(image)
    signals = obj.repeat_signals(signals)

 
    
    return signals
  



def zero_crossing_rate(signal):
    KK = []
    for i in range(12):
        crossings = np.where(np.diff(np.sign(signal[i])))[0]
        zcr = len(crossings) / (2 * len(signal[i]))
        KK.append(zcr)
    return np.array(KK)
        
    

def energy(signal):
    KK = []
    for i in range(12):
        f = np.sum(np.square(signal[i]))
        KK.append(f)
    return np.array(KK)



def entropy_feature(signal):
    KK = []
    for i in range(12):
        hist, _ = np.histogram(signal[i], bins=50)
        hist = hist / hist.sum()  # Normalize histogram
        KK.append(entropy(hist))
    return np.array(KK)


def dominant_frequency(signal, fs):
    kk = []
    for i in range(12):
        fft_result = np.fft.fft(signal[i])
        freqs = np.fft.fftfreq(len(fft_result), d=1/fs)
        magnitude = np.abs(fft_result)
        dominant_freq_index = np.argmax(magnitude)
        dominant_freq = freqs[dominant_freq_index]
        kk.append(dominant_freq)
    return np.array(kk)

def spectral_entropy(signal):
    kk = []
    for i in range(12):
        fft_result = np.fft.fft(signal[i])
        magnitudes = np.abs(fft_result)
        normalized_magnitudes = magnitudes / np.sum(magnitudes)
        kk.append(entropy(normalized_magnitudes))
    return np.array(kk)

def power_distribution(signal):
    kk = []
    for i in range(12):
        
        fft_result = np.fft.fft(signal[i])
        power = np.square(np.abs(fft_result))
        normalized_power = power / np.sum(power)
        kk.append(power)
    return np.array(kk)

def get_eeg_features2(data):
    if data is None:
        return float("nan")*np.ones(108)
    
    features = np.hstack(  (zero_crossing_rate(data).ravel() , energy(data).ravel() , entropy_feature(data).ravel()   ,  dominant_frequency(data,100).ravel(), spectral_entropy(data).ravel() ,  np.mean(power_distribution(data),axis = 1).ravel()  )  )
    return features


def save_digitization_model(model_folder, model):
	return 0


# Save your trained dx classification model.

def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    joblib.dump(d, filename, protocol=0)
