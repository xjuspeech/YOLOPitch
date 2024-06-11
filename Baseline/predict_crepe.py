import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from dataset_org import Net_DataSet
from cnn_bin import Crepe
# from crepe import Crepe
import librosa
import os
import re
import sys
from numpy.lib.stride_tricks import as_strided
from formula_all import *
from scipy.io import wavfile
import config

log = feature.get_logger()

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'


def get_frame_1(wav_file,step_size = 32):
    audio, sr = librosa.load(wav_file,sr=16000)
    hop_length = int(sr * step_size / 1000)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]
    frames = torch.from_numpy(frames)
    return frames

# def frames(wav_path,model_srate = 16000,step_size = 128*3 / 44100 * 1000):
#     sample_rate, audio = wavfile.read(wav_path)  # 读取音频文件，返回采样率 和 信号
#     audio = audio.astype(np.float32)

#     #使输入帧的跳长为10ms
#     hop_length = int(sample_rate * step_size / 1000)  #160
#     wlen = int(sample_rate * 0.064)
#     n_frames = 1 + int((len(audio) - wlen) / hop_length)
#     frames = as_strided(audio, shape=(wlen, n_frames),
#                         strides=(audio.itemsize, hop_length * audio.itemsize))
#     frames = frames.transpose().copy()
#     #标准化每个帧（这是模型期望的）
#     # print(frames.shape)
#     #z-score归一化
#     frames -= np.mean(frames, axis=1)[:, np.newaxis]
#     frames /= np.std(frames, axis=1)[:, np.newaxis]
#     frames[np.isnan(frames)] = 0

#     ret = librosa.resample(y=frames, res_type='linear', orig_sr=sample_rate, target_sr=model_srate)
#     ret = torch.tensor(ret)
#     return ret

def frames(wav_path,model_srate = 16000,step_size = 128*3 / 44100):
    sample_rate, audio = wavfile.read(wav_path)  # 读取音频文件，返回采样率 和 信号
    audio = audio.astype(np.float32)

    #使输入帧的跳长为10ms
    hop_length = int(sample_rate * step_size)  #160
    wlen = int(sample_rate * 0.064)
    n_frames = 1 + int((len(audio) - wlen) / hop_length)
    frames = as_strided(audio, shape=(wlen, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    #标准化每个帧（这是模型期望的）
    # print(frames.shape)
    #z-score归一化
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]
    frames[np.isnan(frames)] = 0

    ret = librosa.resample(y=frames, res_type='linear', orig_sr=sample_rate, target_sr=model_srate)
    ret = torch.tensor(ret)
    return ret

def get_cents(model_output, model,center=None):

    # confidence = model_output.max(axis=1)
    all_cents = np.linspace(0, 7100, 360) + 2051.14876287
    if model_output.ndim == 1:
        if center is None:
            center = int(np.argmax(model_output))
        if center == 360:
            return -np.inf
        start = max(0, center - 4)
        end = min(len(model_output), center + 5)
        net_data = model_output[start:end]
        product_sum = np.sum(
            net_data * all_cents[start:end])
        weight_sum = np.sum(net_data)
        return product_sum / weight_sum
    if model_output.ndim == 2:
        return np.array([get_cents(model_output[i, :],model) for i in
                         range(model_output.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")

def get_freq(net_input,model):
     # = Sound("/student/home/lixuefei/Data/PUTD/SPEECH DATA")
     model_output = model(net_input)
     model_output = model_output.detach().numpy()
     cents = get_cents(model_output,model)

     frequency = 10 * 2 ** (cents / 1200)
     frequency[np.isnan(frequency)] = 0
    # model_output = model(net_input)
    #  import pdb
    #  pdb.set_trace()
     # confidence = cents.max(axis=1)
     time = np.arange(len(cents)) * 10 / 1000.0
     return time, frequency


def get_label(path):
    pitch = []
    ref_cent = []
    # pitch.append(f0)
    with open(path, mode="r") as file:
        for line in file.readlines():
            x = float(line.split(" ")[0])
            x = Convert.convert_semitone_to_hz(x)
            if x >= 10:
                hz = x
                cent = Convert.convert_hz_to_cent(x)
            else:
                hz = 0
                cent = 0
            pitch.append(hz)
            ref_cent.append(cent) 

    return pitch,ref_cent

def get_label_mdb(path):
    pitch = []
    ref_cent = []
    # pitch.append(f0)
    with open(path, mode="r") as file:
        for line in file.readlines():
            hz = float(line.split(",")[1].split("\n")[0])
            if hz > 0:
                cent = Convert.convert_hz_to_cent(hz)
            else:
                cent = 0
            pitch.append(hz)
            ref_cent.append(cent) 

    return pitch,ref_cent

def to_local_average_cents(salience, center=None):
    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        # 如果"to_local_average_cents"函数不包含属性"cents_mapping"
        # to_local_average_cents.cents_mapping = (
        #         np.linspace(0, 7180, 360) + 2041.1487628680297)
        to_local_average_cents.cents_mapping = (
                np.linspace(0, 7180, 360) + 2051.1487628680297)
        to_local_average_cents.cents_mapping = np.insert(to_local_average_cents.cents_mapping,0,0)
    
    if salience.ndim == 1:

        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")

# process_file(test_sound_path)
if __name__ == "__main__":
    log.info(f"predict_cmndf_net_{config.data_name}_test_clean")


    # model_path = '/ssdhome/lixuefei/code/CREPE/crepe_GRL/model_mdb_zi_crepe_noise/model_100.pth'
    model_path = '/ssdhome/lixuefei/code/CREPE/crepe_bin_mdb_735/model_full/model_99.pth'


    wav_dir_path = config.wav_dir_path
    label_dir_path = config.validation_csv_path
    cv_ark_path = config.validation_path
    # cv_ark_path = config.all_label_ark_path
    cv_ark_path = config.test_path

    sound_score_list=[]
    music_score_list=[]
    

    with open(cv_ark_path,mode='r') as file:
        a = file.readlines()
        for i in a:
            filename = i.split(' ')[0]
            label = i.split(' ')[1:]
            if 'ptdb' in config.data_name:
                wav_name = filename.replace('f0','wav').replace('ref','mic')
            elif 'ikala' in config.data_name or 'mir1k' in config.data_name:
                wav_name = filename.replace('pv','wav')
            elif 'mdb' in config.data_name:
                wav_name = filename.replace('csv','wav')
            f0_path = label_dir_path + '/' + filename
            wav_path = wav_dir_path + '/' + wav_name
            print(wav_name)
            log.info(wav_name)

            for wav in os.listdir(wav_dir_path):
                if wav == wav_name:
                    
                    # sound_data = frames(wav_path,step_size=config.hop_size*1000)
                    sound_data = frames(wav_path,step_size=config.hop_size)
                    # sound_data = frames(wav_path,step_size=384/44100*1000)
                    # sound_data = sound_data.T
                    # sound_data = sound_data.unsqueeze(0).unsqueeze(0)
                    sound_data = sound_data.unsqueeze(0).to(device)
                    # sound_data = sound_data.T.unsqueeze(0).to(device)
                    # ds = Sound(None, True)
                    model = Crepe().to(device)
                    checkpoint = torch.load(model_path)
                    model.load_state_dict(checkpoint)
                    model.eval()
                    model_output = model(sound_data.transpose(-1,-2))
                    model_output = model_output.cpu()
                    pitch_data = torch.max(model_output,dim=1)
                    # print(pitch_data)
                    pitch_data = np.array(pitch_data[1])
                    pitch_cent = []
                    pitch=[]
                    for i in pitch_data:
                        if i !=0:
                            j = Convert.convert_bin_to_cent(i)    
                            k = Convert.convert_cent_to_hz(j) 
                        else:
                            j = 0
                            k = 0 
                        pitch_cent.append(j)
                        pitch.append(k)

                    # ref_cent_list = get_label(f0_path)[1]
                    # label_data = get_label(f0_path)[0] 
                    label_data,ref_cent_list = feature.get_label(f0_path)
                    # label_data,ref_cent_list = get_label_mdb(f0_path)
                    # label_data,ref_cent_list = get_label(f0_path)

                    len_pitch = len(pitch)
                    len_label = len(label_data)
                    print("len_pitch:",len_pitch,"len_label:",len_label)
                    pitch = pitch[:min(len_pitch,len_label)]
                    label = label_data[:min(len_pitch,len_label)]
                    pitch_cent = pitch_cent[:min(len_pitch,len_label)]
                    label_cent = ref_cent_list[:min(len_pitch,len_label)]

                    pitch = np.array(pitch)
                    label = np.array(label)
                    pitch_cent = np.array(pitch_cent)
                    label_cent = np.array(label_cent)
                    # print(pitch)
                    # print(label)
                    # print(pitch_cent.shape)
                    # print(label_cent.shape)
                    sound_score = Sound.all_mir_eval(pitch,label,threshold = 0.2)
                    music_score = Music.all_mir_eval(label, label_cent, pitch, pitch_cent,cent_tolerance=50)
                    sound_score_list.append(sound_score)
                    music_score_list.append(music_score)

                    print("sound_score:",sound_score)
                    print("music_score",music_score)
                    log.info(f"sound_score:{sound_score}")
                    log.info(f"music_score:{music_score}")

    # 预测实验结果
    # sound_score = Sound.all_mir_eval(pitch,label,threshold = 0.2)
    # music_score = Music.all_mir_eval(label, label_cent, pitch, pitch_cent,cent_tolerance=50)
    # sound_score_list.append(sound_score)
    # music_score_list.append(music_score)

    # sound_score_numpy = np.array(sound_score_list)
    # music_score_numpy = np.array(music_score_list)
    # # sound_score_avg = np.sum(sound_score_numpy,axis = 0) / sound_score_numpy.shape[0]
    # sound_score_avg = np.nanmean(sound_score_numpy,axis = 0)
    # music_score_avg = np.sum(music_score_numpy,axis = 0) / music_score_numpy.shape[0]
    
    # print("score结果:")
    # print("sound_score_avg:",sound_score_avg)
    # print('music_score_avg:',music_score_avg)
    # log.info(f"score结果:")
    # log.info(f"sound_score_avg:{sound_score_avg}")
    # log.info(f"music_score_avg:{music_score_avg}")

    sound_score_numpy = np.array(sound_score_list)
    music_score_numpy = np.array(music_score_list)

    std_sound = np.std(sound_score_numpy,axis=0)
    std_music = np.std(music_score_numpy,axis=0)
    std_ddof_sound = np.std(sound_score_numpy, axis=0,ddof = 1)
    std_ddof_music = np.std(music_score_numpy, axis=0,ddof = 1)

    # sound_score_avg = np.sum(sound_score_numpy,axis = 0) / sound_score_numpy.shape[0]
    sound_score_avg = np.nanmean(sound_score_numpy,axis = 0)
    music_score_avg = np.sum(music_score_numpy,axis = 0) / music_score_numpy.shape[0]


    print("score结果:")
    print("sound_score_avg:",sound_score_avg)
    print('music_score_avg:',music_score_avg)
    print('std_sound:',std_sound)
    print('std_music:',std_music)
    print('std_ddof_sound:',std_ddof_sound)
    print('std_ddof_music:',std_ddof_music)
    log.info(f"score结果:")
    log.info(f"sound_score_avg:{sound_score_avg}")
    log.info(f"music_score_avg:{music_score_avg}")
    log.info(f"std_sound:{std_sound}")
    log.info(f"std_music:{std_music}")
    log.info(f"std_ddof_sound:{std_ddof_sound}")
    log.info(f"std_ddof_music:{std_ddof_music}")


  



