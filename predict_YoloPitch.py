import torch
import numpy as np
from yolo_wav_stft import YoloBody
import librosa
import os
from numpy.lib.stride_tricks import as_strided
from formula_all import *
from scipy.io import wavfile
import config

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_frames(abs_wav_path,model_srate=16000,step_size= 0.02,len_frame_time=0.064):
    sample_rate, audio = wavfile.read(abs_wav_path)  
    audio = audio.astype(np.float32)
    
    # frame
    hop_length = int(sample_rate * step_size)  
    wlen = int(sample_rate * len_frame_time)
    n_frames = 1 + int((len(audio) - wlen) / hop_length)
    frames = as_strided(audio, shape=(wlen, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    #z-score
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]
    frames[np.isnan(frames)] = 0
    ret = librosa.resample(y=frames, res_type='linear', orig_sr=sample_rate, target_sr=model_srate)
    return ret

def get_stft(abs_wav_path,model_srate=16000,step_size= 0.02,n_fft=2047):
    y, sr = librosa.load(abs_wav_path, sr=model_srate)

    stft = librosa.stft(y,n_fft=n_fft, hop_length=int(step_size*model_srate), win_length=1024, window='hamming', center=True)
    log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return log_stft

def get_YOLOPitch(model_path, wav_path):

    sound_data_wav = get_frames(wav_path,step_size=config.hop_size)
    sound_data_stft = get_stft(wav_path,step_size=config.hop_size).T
    min_num = min(sound_data_wav.shape[0],sound_data_stft.shape[0])
    sound_data_wav = sound_data_wav[:min_num,:]
    sound_data_stft = sound_data_stft[:min_num,:]
    sound_data_wav=torch.tensor(sound_data_wav)
    sound_data_stft=torch.tensor(sound_data_stft)
    sound_data_wav = sound_data_wav.unsqueeze(0).to(device)
    sound_data_stft = sound_data_stft.unsqueeze(0).to(device)

    model = YoloBody(phi='l', pretrained=False).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model_output = model(sound_data_wav.transpose(-1,-2),sound_data_stft.transpose(-1,-2))
    model_output = model_output.cpu()
    pitch_data = torch.max(model_output,dim=1)
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
    return pitch


if __name__ == "__main__":
    

    model_path = 'xxx/model_YOLOPitch.pth'
    wav_path = 'xxx/xxx.wav'
    pitch = get_YOLOPitch(model_path, wav_path)
    print(pitch)
    
