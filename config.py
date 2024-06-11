import numpy as np
import math
import warnings
import torch
import time
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import librosa
import time
import logging
import os

notice=''

data_name = 'mdb_zi'  #choose: mdb,ptdb,ikala,mir1k,mdb_zi,mdb_song

epochs=100
save_model_path = '/ssdhome/lixuefei/code/pyin_ikala_all/model_cmndf_ptdb'

data_dict={

    'ptdb_train_path' : '/ssdhome/.../PTDB/SPEECH_DATA/label/hz_class_2/train_lable.ark',
    'ptdb_validation_path' : '/ssdhome/.../data/PTDB/SPEECH_DATA/label/hz_class_2/cv_lable.ark',
    'ptdb_validation_csv_path' : '/ssdhome/.../data/PTDB/SPEECH_DATA/f0',
    'ptdb_test_path' : '/ssdhome/.../data/PTDB/SPEECH_DATA_2/label/hz_class_2/test_lable.ark',
    'ptdb_wav_dir_path' : '/ssdhome/.../data/PTDB/SPEECH_DATA_2/wav',
    'ptdb_noise_wav_dir_path' : '/ssdhome/.../CREPE/crepe_GRL/ptdb_noise_wav',
    # 'ptdb_all_label_ark_path':'/ssdhome/.../data/PTDB/SPEECH_DATA_2/label/hz_class_2/all_label.ark',
    'ptdb_all_label_ark_path': '/ssdhome/.../data/PTDB/SPEECH_DATA_2/label/hz_class_2/all_label_F.ark',
    'ptdb_hop_size' : 0.01,
    'ptdb_hop_size_dot' : 0.01*16000,
    'ptdb_out_class' : 360,
    'ptdb_sr' : 16000,

    'mir1k_train_path' : '/ssdhome/.../data/MIR-1K/data/label/bin_class/train_lable.ark',
    'mir1k_validation_path' : '/ssdhome/.../data/MIR-1K/data/label/bin_class/cv_lable.ark',
    'mir1k_validation_csv_path' : '/ssdhome/.../data/MIR-1K/MIR-1K/PitchLabel',
    'mir1k_test_path' : '/ssdhome/.../data/MIR-1K/data/label/bin_class/test_lable.ark',
    'mir1k_wav_dir_path' : '/ssdhome/.../data/MIR-1K/data/mono_voice',
    'mir1k_noise_wav_dir_path' : '/ssdhome/.../data/MIR-1K/data/mono_voice',
    # 'mir1k_wav_dir_path' : '/ssdhome/.../data/MIR-1K/data/monso_voice_noise/mono_voice5dB',
    'mir1k_all_label_ark_path':'/ssdhome/.../data/MIR-1K/data/label/bin_class/all_lable.ark',
    'mir1k_hop_size' : 0.02,
    'mir1k_hop_size_dot' : 0.02*16000,
    'mir1k_out_class' : 360,
    'mir1k_sr' : 16000,


    'mdb_zi_train_path' : '/ssdhome/.../data/mdb/MDB-stem-synth/data/data_zi_del_all_zreo/label_bin/train_label.ark',
    'mdb_zi_validation_path' : '/ssdhome/.../data/mdb/MDB-stem-synth/data/data_zi_del_all_zreo/label_bin/cv_label.ark',
    'mdb_zi_validation_csv_path' : '/ssdhome/.../data/mdb/MDB-stem-synth/data/data_zi_del_all_zreo/zi_csv_del_all_zero_div_3',
    'mdb_zi_test_path' : '/ssdhome/.../data/mdb/MDB-stem-synth/data/data_zi_del_all_zreo/label_bin/test_label.ark',
    'mdb_zi_wav_dir_path' : '/ssdhome/.../data/mdb/MDB-stem-synth/data/data_zi_del_all_zreo/zi_sound_del_all_zero',
    'mdb_zi_noise_wav_dir_path' : '/ssdhome/.../code/CREPE/crepe_GRL/mdb_zi_noise_wav',
    # 'mdb_zi_wav_dir_path':'/ssdhome/.../code/CREPE/crepe_GRL/mdb_zi_noise_test/-10dB',
    # 'mdb_zi_wav_dir_path':'/ssdhome/.../code/CREPE/crepe_GRL/mdb_zi_noise_cv/-10dB',
    # 'mdb_zi_all_label_ark_path':'/ssdhome/.../data/mdb/MDB-stem-synth/data/data_zi_del_all_zreo/label_bin/all_label.ark',
    'mdb_zi_all_label_ark_path':'/ssdhome/lixuefei/data/mdb/MDB-stem-synth/data/data_zi_nan_zero/label_bin/all_label_new.ark',
    'mdb_zi_hop_size' : 128/44100*3,
    'mdb_zi_hop_size_dot' : 128*3,
    'mdb_zi_out_class' : 360,
    'mdb_zi_sr' : 44100,

    
    }

data_train_path = data_name+'_train_path'
data_validation_path = data_name+'_validation_path'
data_validation_csv_path = data_name+'_validation_csv_path'
data_test_path = data_name+'_test_path'
data_wav_dir_path = data_name+'_wav_dir_path'
data_all_label_ark_path = data_name +'_all_label_ark_path'
data_hop_size = data_name+'_hop_size'
data_hop_size_dot = data_name +'_hop_size_dot'
data_out_class = data_name+'_out_class'
data_sr = data_name+'_sr'
data_noise_wav_dir_path = data_name+'_noise_wav_dir_path'

train_path = data_dict[data_train_path]         # train_label_ark_path
validation_path = data_dict[data_validation_path]       # validation_label_ark_path
validation_csv_path = data_dict[data_validation_csv_path]       # ground_truth_path
test_path = data_dict[data_test_path]       # test_label_ark_path
wav_dir_path = data_dict[data_wav_dir_path]       # wav_dir_path
all_label_ark_path = data_dict[data_all_label_ark_path]        # all_label_ark_path
hop_size = data_dict[data_hop_size]       # hop_size, such as 10 ms
hop_size_dot = data_dict[data_hop_size_dot]       # hop_size_dot, such as 10ms * 16000 (sr) = 160
out_class = data_dict[data_out_class]       # out_class
sr = data_dict[data_sr]       # Sampling Rate 
noise_wav_dir_path = data_dict[data_noise_wav_dir_path]       # noise_wav_dir_path





























