import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import numpy as np
import soundfile
import glob
import os

# Classes in the dataset
activities ={
  'I':0,
  'E':1
}

#################### Audio Signal Feature Extractions ############################
def feature_spectrogram(waveform, sample_rate):
    # power spoectrogram (2D full spectrogram)
    stft_spectrum_matrix = librosa.stft(waveform)
    spectrogram = librosa.amplitude_to_db(np.abs(stft_spectrum_matrix), ref=np.max)
    return spectrogram

def feature_spectrogram_mean(waveform, sample_rate):
    # power spoectrogram (get the mean of each column)
    stft_spectrum_matrix = librosa.stft(waveform)
    spectrogram_mean = np.mean(librosa.amplitude_to_db(np.abs(stft_spectrum_matrix), ref=np.max).T,axis=0)
    return spectrogram_mean

def feature_centroid(waveform, sample_rate):
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    return centroid

def feature_bandwidth(waveform, sample_rate):
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)
    return bandwidth

def feature_melspectrogram(waveform, sample_rate):
    # Mel spoectrogram (2D full spectrogram)
    melspectrogram=librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000)
    return melspectrogram

def feature_melspectrogram_mean(waveform, sample_rate):
    # Mel spoectrogram (get the mean of each column)
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram=np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000).T,axis=0)
    return melspectrogram

def feature_mfcc(waveform, sample_rate):
    # Compute the MFCCs for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # 40 filterbanks = 40 coefficients
    mfc_coefficients=np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T, axis=0) 
    return mfc_coefficients

def feature_chromagram(waveform, sample_rate):
    # Chromagram (2D full spectrogram)
    stft_spectrogram=np.abs(librosa.stft(waveform))
    chromagram=librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate)
    return chromagram

def feature_chromagram_mean(waveform, sample_rate):
    # Chromagram (get the mean of each column)
    stft_spectrogram=np.abs(librosa.stft(waveform))
    # Produce the chromagram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    chromagram_mean=np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T,axis=0)
    return chromagram_mean

########################## Feature sets creation ######################
def create_feature_set_1(file):
    # feature set 1: Power Spectrogram
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype='float32')
        sample_rate = audio.samplerate

        # compute features
        spectrogram = feature_spectrogram(waveform, sample_rate)

        return spectrogram

def create_feature_set_2(file):
    # feature set 2: Mel Spectrogram
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype='float32')
        sample_rate = audio.samplerate

        # compute features
        melspectrogram = feature_melspectrogram(waveform, sample_rate)

        return melspectrogram

def create_feature_set_3(file):
    # feature set 3: MFCC
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype='float32')
        sample_rate = audio.samplerate

        # compute features
        mfcc = feature_mfcc(waveform, sample_rate)

        return mfcc

def create_feature_set_4(file):
    # feature set 4: Chromagram
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype='float32')
        sample_rate = audio.samplerate

        # compute features
        chromagram = feature_chromagram(waveform, sample_rate)

        return chromagram

def create_feature_set_5(file):
    # feature set 5: Combination 1: spectrogram with spectral centroid and bandwidth
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype='float32')
        sample_rate = audio.samplerate

        # compute features
        spectrogram_mean = feature_spectrogram_mean(waveform, sample_rate)
        centroid = feature_centroid(waveform, sample_rate)
        bandwidth = feature_bandwidth(waveform, sample_rate)
        # upper_bound = centroid[0] + bandwidth[0]
        # lower_bound = centroid[0] - bandwidth[0]
        
        feature_matrix=np.array([])
        # use np.hstack to stack our feature arrays horizontally to create a feature matrix
        # feature_matrix = np.hstack((spectrogram, upper_bound, lower_bound))
        feature_matrix = np.hstack((spectrogram_mean, centroid.flatten(), bandwidth.flatten()))
        
        return feature_matrix

def create_feature_set_6(file):
    # feature set 6: All
    with soundfile.SoundFile(file) as audio:
        waveform = audio.read(dtype='float32')
        sample_rate = audio.samplerate

        # compute features
        spectrogram_mean = feature_spectrogram_mean(waveform, sample_rate)
        centroid = feature_centroid(waveform, sample_rate)
        bandwidth = feature_bandwidth(waveform, sample_rate)
        # upper_bound = centroid[0] + bandwidth[0]
        # lower_bound = centroid[0] - bandwidth[0]

        melspectrogram_mean = feature_melspectrogram_mean(waveform, sample_rate)
        mfcc = feature_mfcc(waveform, sample_rate)
        chromagram_mean = feature_chromagram_mean(waveform, sample_rate)

        feature_matrix=np.array([])
        # use np.hstack to stack our feature arrays horizontally to create a feature matrix
        feature_matrix = np.hstack((spectrogram_mean, centroid.flatten(), bandwidth.flatten(), melspectrogram_mean, mfcc, chromagram_mean))

        return feature_matrix

