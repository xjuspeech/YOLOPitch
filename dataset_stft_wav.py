import os
import numpy as np
import torch
import librosa
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import Dataset
import pickle
import librosa
from scipy.io import wavfile
import config
import librosa.core as lc  
# import matplotlib.pyplot as plt  
# import librosa.display  

def get_frames(abs_wav_path,model_srate=16000,step_size= 0.02,len_frame_time=0.064):
    sample_rate, audio = wavfile.read(abs_wav_path)  # 读取音频文件，返回采样率 和 信号
    audio = audio.astype(np.float32)
    
    #分帧
    hop_length = int(sample_rate * step_size)  
    wlen = int(sample_rate * len_frame_time)
    n_frames = 1 + int((len(audio) - wlen) / hop_length)
    frames = as_strided(audio, shape=(wlen, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    #z-score归一化
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]
    frames[np.isnan(frames)] = 0
    ret = librosa.resample(y=frames, res_type='linear', orig_sr=sample_rate, target_sr=model_srate)
    return ret

def get_stft(abs_wav_path,model_srate=16000,step_size= 0.02,n_fft=2047):
    y, sr = librosa.load(abs_wav_path, sr=model_srate)

    stft = librosa.stft(y,n_fft=n_fft, hop_length=int(step_size*model_srate), win_length=1024, window='hamming', center=True)
    # log_stft = np.log(stft)
    # 将振幅谱转换为对数尺度
    log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return log_stft


class Net_DataSet(Dataset):
    def __init__(self,path):
        super(Net_DataSet,self).__init__()
        self.label = self.real_label(path)
    
    def real_label(self,path):
        with open(path,mode="r",encoding="gbk") as file:
            return file.readlines()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        every_file = self.label[index]

        if 'f0' in every_file.strip().split()[0]:
            filename = every_file.strip().split()[0].replace("ref","mic").replace("f0","txt")
            pv_name = filename.replace("txt","f0").replace("mic","ref")
        elif 'pv' in every_file.strip().split()[0]:
            filename = every_file.strip().split()[0].replace("pv","txt")
            pv_name = filename.replace("txt","pv")
        elif 'csv' in every_file.strip().split()[0]:
            filename = every_file.strip().split()[0].replace("csv","txt")
            pv_name = filename.replace("txt","csv")
        wavname = filename.replace("txt","wav")
        # pv_name = filename.replace("txt","pv")
        # print(wavname)
        wav_path = config.wav_dir_path + '/' + wavname
        frames = get_frames(wav_path,step_size=config.hop_size)
        stft = get_stft(wav_path,step_size=config.hop_size).T
        # print(filename)
        label = every_file.strip().split()[1:]
        label = label[:len(frames)]
        label1=[]
        for x in label:
            x = int(np.float64(x))
            label1.append(x)
        # min_len = min([len(label1),frames.size(0)])
        min_len = min([len(label1),len(frames)])
        # min_len = 2
        label1 = label1[:min_len]
        frames = frames[:min_len][:]
        stft = stft[:min_len][:]

        label1 = torch.tensor(label1).squeeze().long()
        # frames = frames.float().transpose(0,1)
        frames = torch.tensor(frames).float().transpose(0,1)
        stft = torch.tensor(stft).float().transpose(0,1)
        
        return [frames,stft],[label1,pv_name]


if __name__ == "__main__":

    path = config.test_path
    s =Net_DataSet(path)
    print("s[2]:",s[6])
    # label = s[6][1].numpy()
    # label = list(label)
    # print(label)
    print(s[6][0][0].shape,s[6][0][1].shape)
    print(len(s[6][1]))

 
