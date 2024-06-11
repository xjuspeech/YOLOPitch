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

def get_frames(abs_wav_path,model_srate=16000,step_size= 0.02,len_frame_time=0.064):
    sample_rate, audio = wavfile.read(abs_wav_path) 
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
        wav_path = config.wav_dir_path + '/' + wavname

        frames = get_frames(wav_path,step_size=config.hop_size)
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

        label1 = torch.tensor(label1).squeeze().long()
        # frames = frames.float().transpose(0,1)
        frames = torch.tensor(frames).float().transpose(0,1)
        
        return frames,[label1,pv_name]


if __name__ == "__main__":

    path = config.test_path
    s =Net_DataSet(path)
    print("s[2]:",s[6])
    # label = s[6][1].numpy()
    # label = list(label)
    # print(label)
    print(s[6][0].shape)
    print(len(s[6][0][0]))

 
