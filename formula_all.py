import numpy as np
import math
import warnings
import torch
import time
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import librosa
import config
import time
import logging
import os

class Convert():
    def convert_cent_to_hz(cent, f_ref=10.0):
        return f_ref * 2 ** (cent / 1200.0)

    def convert_hz_to_cent(hertz, f_ref=10.0):
        # return mir_eval.melody.hz2cents(hertz, f_ref)
        return 1200.0 * math.log(hertz / f_ref , 2)

    def convert_semitone_to_hz(midi):
        # return convert_cent_to_hz(100 * semi, f_ref)
        return 440 * math.pow(2, (midi - 69) / 12)

    def convert_hz_to_semitone(hz):
        return 69 + round(12 * math.log(hz / 440 , 2))

    def convert_semitone_to_cent(midi):
        return Convert.convert_hz_to_cent(Convert.convert_semitone_to_hz(midi))  

    def convert_cent_to_semitone(cent):
        return Convert.convert_hz_to_semitone(Convert.convert_cent_to_hz(cent))

    def convert_cent_to_bin(cent):
        return round((cent-2051.1487628680297) / 20 + 1)

    def convert_hz_to_bin(hz):
        return Convert.convert_cent_to_bin(Convert.convert_hz_to_cent(hz)) 

    def convert_semitone_to_bin(midi):
        return Convert.convert_cent_to_bin(Convert.convert_semitone_to_cent(midi))  

    def convert_bin_to_cent(bin):
        return (bin -1) * 20 + 2051.1487628680297

    def convert_bin_to_hz(bin):
        return Convert.convert_cent_to_hz(Convert.convert_bin_to_cent(bin))

class Music():
    def validate_voicing(ref_voicing, est_voicing):
   
        if ref_voicing.size == 0:
            warnings.warn("Reference voicing array is empty.")
        if est_voicing.size == 0:
            warnings.warn("Estimated voicing array is empty.")
        if ref_voicing.sum() == 0:
            warnings.warn("Reference melody has no voiced frames.")
        if est_voicing.sum() == 0:
            warnings.warn("Estimated melody has no voiced frames.")
        # Make sure they're the same length
        if ref_voicing.shape[0] != est_voicing.shape[0]:
            raise ValueError('Reference and estimated voicing arrays should '
                            'be the same length.')
        for voicing in [ref_voicing, est_voicing]:
            # Make sure voicing is between 0 and 1
            if np.logical_or(voicing < 0, voicing > 1).any():
                raise ValueError('Voicing arrays must be between 0 and 1.')


    def validate(ref_voicing, ref_cent, est_voicing, est_cent):
    
        if ref_cent.size == 0:
            warnings.warn("Reference frequency array is empty.")
        if est_cent.size == 0:
            warnings.warn("Estimated frequency array is empty.")
        # Make sure they're the same length
        if ref_voicing.shape[0] != ref_cent.shape[0] or \
            est_voicing.shape[0] != est_cent.shape[0] or \
            ref_cent.shape[0] != est_cent.shape[0]:
            raise ValueError('All voicing and frequency arrays must have the '
                            'same length.')


    def hz_to_vocing(hz):
        if hz == 0:
            voicing = 0
        else:
            voicing = (hz > 0).astype(float)
        return np.abs(hz),voicing

    #语音召回率VR：（预测正确的）浊音帧数/（（预测正确的）浊音帧数 + （预测错误的）清音帧帧数）-->真实的所有浊音帧数      
    # （预测正确）浊音帧数/真实总浊音帧数
    def voicing_recall(ref_voicing, est_voicing):
        if ref_voicing.size == 0 or est_voicing.size == 0:
            return 0.
        ref_indicator = (ref_voicing > 0).astype(float)
        est_indicator = (est_voicing>0).astype(float)
        if np.sum(ref_indicator) == 0:
            return 1
        return np.sum(est_indicator * ref_indicator) / np.sum(ref_indicator)

    #语音精确率VP：（预测正确）浊音帧数/（预测正确）浊音帧数 + （预测错误）浊音帧数 -->预测所有浊音帧数  
    # （预测正确）浊音帧数/预测总浊音帧数
    def voice_precision(ref_voicing, est_voicing):
        if ref_voicing.size == 0 or est_voicing.size == 0:
            return 0.
        ref_indicator = (ref_voicing > 0).astype(float)
        est_indicator = (est_voicing > 0).astype(float)
        if np.sum(ref_indicator) == 0:
            return 1
        return np.sum(est_indicator * ref_indicator) / np.sum(est_indicator)

    #语音报警率VFA：（预测错误）浊音帧数/（（预测正确）浊音帧数 + （预测错误）清音帧帧数）-->真实的所有浊音帧数
    # （预测错误）浊音帧数/真实无声帧数，无声帧误判为浊音帧
    def voicing_false_alarm(ref_voicing, est_voicing):

        if ref_voicing.size == 0 or est_voicing.size == 0:
            return 0.
        #若为无音帧，则记为1，浊音帧记为0
        ref_indicator = (ref_voicing == 0).astype(float)
        #浊音帧为1
        est_voicing = (est_voicing > 0).astype(float)
        if np.sum(ref_indicator) == 0:
            return 0
        return np.sum(est_voicing * ref_indicator) / np.sum(ref_indicator)


    #原始pitch准确度RPA：真实浊音帧中，基频值（误差50cent）估计正确的数目
    def raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                        cent_tolerance=50):
        
        # validate_voicing(ref_voicing, est_voicing)
        # validate(ref_voicing, ref_cent, est_voicing, est_cent)
        
        if ref_voicing.size == 0 or ref_voicing.sum() == 0 \
        or ref_cent.size == 0 or est_cent.size == 0:
            return 0.

        #取二者关系 est_cent & ref_cent都不为0时的值
        nonzero_freqs = np.logical_and(est_cent != 0, ref_cent != 0)

        if sum(nonzero_freqs) == 0:
            return 0.

        #只计算ref_cent与est_cent都不为0时的值差值
        freq_diff_cents = np.abs(ref_cent - est_cent)[nonzero_freqs]
    
        correct_frequencies = freq_diff_cents < cent_tolerance
        ref_voicing = (ref_voicing > 0).astype(float)
        est_voicing = (est_voicing>0).astype(float)

        #计算ref_cent与est_cent都不为0时浊音 * 误差<50cent的数目 / 真实总浊音数
        rpa = (
            np.sum(ref_voicing[nonzero_freqs] * correct_frequencies) /
            np.sum(ref_voicing)
        )
        return rpa                   

    #原始音色准确度RCA：真实浊音帧中，基频值（误差50cent，真实cent与预测cent之间是否含有八度误差）估计正确的数目
    def raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                            cent_tolerance=50):
        
        # validate_voicing(ref_voicing, est_voicing)
        # validate(ref_voicing, ref_cent, est_voicing, est_cent)
        if ref_voicing.size == 0 or ref_voicing.sum() == 0 \
        or ref_cent.size == 0 or est_cent.size == 0:
            return 0.

        # # Raw chroma = same as raw pitch except that octave errors are ignored.
        nonzero_freqs = np.logical_and(est_cent != 0, ref_cent != 0)

        if sum(nonzero_freqs) == 0:
            return 0.

        freq_diff_cents = np.abs(ref_cent - est_cent)[nonzero_freqs]
        octave = 1200.0 * np.floor(freq_diff_cents / 1200 + 0.5)
        correct_chroma = np.abs(freq_diff_cents - octave) < cent_tolerance
        ref_voicing = (ref_voicing > 0).astype(float)
        rca = (
            np.sum(ref_voicing[nonzero_freqs] * correct_chroma) /
            np.sum(ref_voicing)
        )
        return rca

    #总体精度OA：浊音帧+无声帧（正确）/总帧数
    def overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                        cent_tolerance=50):

        # validate_voicing(ref_voicing, est_voicing)
        # validate(ref_voicing, ref_cent, est_voicing, est_cent)

        if ref_voicing.size == 0 or est_voicing.size == 0 \
        or ref_cent.size == 0 or est_cent.size == 0:
            return 0.

        nonzero_freqs = np.logical_and(est_cent != 0, ref_cent != 0)
        freq_diff_cents = np.abs(ref_cent - est_cent)[nonzero_freqs]
        correct_frequencies = freq_diff_cents < cent_tolerance
        ref_binary = (ref_voicing > 0).astype(float)
        n_frames = float(len(ref_voicing))

        ref_voicing = (ref_voicing > 0).astype(float)
        est_voicing = (est_voicing > 0).astype(float)

        if np.sum(ref_voicing) == 0:
            ratio = 0.0
        else:
            ratio = (np.sum(ref_binary) / np.sum(ref_voicing))

        accuracy = (
            (
                ratio * np.sum(ref_voicing[nonzero_freqs] *
                            est_voicing[nonzero_freqs] *
                            correct_frequencies)
            ) +
            np.sum((1.0 - ref_binary) * (1.0 - est_voicing))
        ) / n_frames

        return accuracy


    def all_mir_eval(ref_voicing, ref_cent, est_voicing, est_cent,cent_tolerance=50):

        VR = Music.voicing_recall(ref_voicing, est_voicing)
        VFA = Music.voicing_false_alarm(ref_voicing, est_voicing)
        RPA = Music.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                        cent_tolerance=50)
        RCA = Music.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                            cent_tolerance=50)
        OA = Music.overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                        cent_tolerance=50)

        return VR, VFA, RPA, RCA, OA

def melody_eval(ref_freq, est_freq):
    import mir_eval
    
    ref_time = est_time = np.arange(len(ref_freq)+1)[1:]

    output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq)
    VR = output_eval['Voicing Recall']
    VFA = output_eval['Voicing False Alarm']
    RPA = output_eval['Raw Pitch Accuracy']
    RCA = output_eval['Raw Chroma Accuracy']
    OA = output_eval['Overall Accuracy']
    # eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    # eval_arr = eval_arr.tolist()
    # return eval_arr
    return VR, VFA, RPA, RCA, OA

class Sound():
    #清浊音检测错误率VDE：
    def voicing_decision_error(ref_voicing, est_voicing):
        if ref_voicing.size == 0 or est_voicing.size == 0:
            return 0.
        ref_indicator = (ref_voicing > 0).astype(float)
        est_voicing = (est_voicing>0).astype(float)
        ref_unvoice_indicator = (ref_voicing == 0).astype(float)
        est_unvoice_indicator = (est_voicing == 0).astype(float)
        n_frames = float(len(ref_voicing))
        if np.sum(ref_indicator) == 0:
            return 1
        return np.sum(est_voicing * ref_unvoice_indicator + est_unvoice_indicator * ref_indicator) / n_frames

    #总体pitch错误率GPE:
    def  gross_pitch_error(ref_voicing, est_voicing,threshold = 0.2):
        if ref_voicing.size == 0 or est_voicing.size == 0:
            return 0.

        nonzero_freqs = np.logical_and(est_voicing != 0, ref_voicing != 0)
        ref_indicator = (ref_voicing > 0).astype(float)
        #N_vv：真实和预测都是浊音
        N_vv = np.sum(ref_indicator[nonzero_freqs])
        # 计算预测浊音 / 真实浊音  > 20%
        freq_diff = np.abs((est_voicing[nonzero_freqs] / ref_voicing[nonzero_freqs]) - 1)
        correct_frequencies = freq_diff > threshold
        # 计算（预测浊音 / 真实浊音  > 20%）的总数目
        N_F0E = np.sum(ref_indicator[nonzero_freqs] * correct_frequencies)
    
        if np.sum(ref_indicator) == 0:
            return 1

        return N_F0E / N_vv

    #每帧数据集品错误率FFE：
    def F0_frame_error(ref_voicing, est_voicing,threshold = 0.2):
        nonzero_freqs = np.logical_and(est_voicing != 0, ref_voicing != 0)
        ref_indicator = (ref_voicing > 0).astype(float)
        N_vv = np.sum(ref_indicator[nonzero_freqs])
        n_frames = float(len(ref_voicing))
        GPE = Sound.gross_pitch_error(ref_voicing, est_voicing,threshold)
        VDE = Sound.voicing_decision_error(ref_voicing, est_voicing)

        return N_vv / n_frames * GPE + VDE

    #语音精确率VP：（预测正确）浊音帧数/（预测正确）浊音帧数 + （预测错误）浊音帧数 -->预测所有浊音帧数  
    # （预测正确）浊音帧数/预测总浊音帧数
    def voice_precision(ref_voicing, est_voicing):
        if ref_voicing.size == 0 or est_voicing.size == 0:
            return 0.
        ref_indicator = (ref_voicing > 0).astype(float)
        est_indicator = (est_voicing > 0).astype(float)
        if np.sum(ref_indicator) == 0:
            return 1
        return np.sum(est_indicator * ref_indicator) / np.sum(est_indicator)

    #语音召回率VR：（预测正确的）浊音帧数/（（预测正确的）浊音帧数 + （预测错误的）清音帧帧数）-->真实的所有浊音帧数      
    # （预测正确）浊音帧数/真实总浊音帧数
    def voice_recall(ref_voicing, est_voicing):
        if ref_voicing.size == 0 or est_voicing.size == 0:
            return 0.
        ref_indicator = (ref_voicing > 0).astype(float)
        est_indicator = (est_voicing>0).astype(float)
        if np.sum(ref_indicator) == 0:
            return 1
        return np.sum(est_indicator * ref_indicator) / np.sum(ref_indicator)

    def f1_measure(ref_voicing, est_voicing):
        if ref_voicing.size == 0 or est_voicing.size == 0:
            return 0.
        VP = Sound.voice_precision(ref_voicing, est_voicing)
        VR = Sound.voice_recall(ref_voicing, est_voicing)
        
        return (2 * VP * VR) / (VP + VR)

    def all_mir_eval(ref_voicing, est_voicing,threshold = 0.2):
        VDE = Sound.voicing_decision_error(ref_voicing, est_voicing)
        GPE = Sound.gross_pitch_error(ref_voicing, est_voicing,threshold = 0.2)
        FFE = Sound.F0_frame_error(ref_voicing, est_voicing,threshold = 0.2)
        VP = Sound.voice_precision(ref_voicing, est_voicing)
        VR = Sound.voice_recall(ref_voicing, est_voicing)
        F_score = Sound.f1_measure(ref_voicing, est_voicing)

        return VDE, GPE, FFE, VP, VR, F_score


class Smooth():

    def ZeroCount(pitch_list):
        count = 0
        for i in range(len(pitch_list)-1):
            if pitch_list[i+1]==0:
                count += 1
            else:
                break
        return count

    def Median(pitch_list,i,PD=4,FD=4):
        PD_index = max(0,i-PD)
        FD_index = min(i+FD,len(pitch_list))
        # print(PD,FD)
        localPitches = pitch_list[PD_index:FD_index]
        localPitches.sort()
        # print(localPitches)
        if len(localPitches) % 2 ==0:
            med = (localPitches[len(localPitches)//2] + localPitches[len(localPitches)//2-1]) /2
        else:
            med = localPitches[len(localPitches)//2]
        # print(med)
        return med        

    def SmartMedian(pitch_list,priorDistance=4,acceptableFrequencyDifference=100,followingDistance=4,noZero=4,maxFrequency=1200):
        for (i,pitch) in enumerate(pitch_list):
            # if i >0 and i<len(pitch_list)-3:
            if pitch_list[i-1] > 0 and abs(pitch_list[i] - pitch_list[i-1]) >acceptableFrequencyDifference and Smooth.ZeroCount(pitch_list[i:])<noZero:
                # print(i,pitch)
                followingDistance = 4
                Med = Smooth.Median(pitch_list,i,FD=followingDistance)
                for epoch in range(followingDistance):
                    # print(epoch)
                    followingDistance -= 1
                    # print("followingDistance:",followingDistance)
                    if abs(Med - pitch_list[i-1]) > acceptableFrequencyDifference:
                        Med = Smooth.Median(pitch_list,i,FD=followingDistance)
                        # print("********Med*********:",Med)
                    else:
                        if Med <= maxFrequency:
                            pitch_list[i] = Med
                        else:
                            pitch_list[i] = 0
                        break
            else:
                if pitch_list[i-1]==0 and pitch_list[i]>maxFrequency and abs(pitch_list[i] - pitch_list[i-1]) >acceptableFrequencyDifference:
                    pitch_list[i] = 0
                
                else:
                    # pitch_list[i] = pitch_list[i]
                    pass
        return pitch_list

    def to_local_average_cents(salience, center=None):
        if not hasattr(Smooth.to_local_average_cents, 'cents_mapping'):
            # the bin number-to-cents mapping
            Smooth.to_local_average_cents.cents_mapping = (
                    np.linspace(0, 7180, 360) + 2051.1487628680297)
            # to_local_average_cents.cents_mapping = np.insert(to_local_average_cents.cents_mapping,0,0)
        
        if salience.ndim == 1:

            if center is None:
                center = int(np.argmax(salience))
            if center != 0:
                salience = salience[1:]
                center = center-1
                start = max(0, center - 4)
                end = min(len(salience), center + 5)
                salience = salience[start:end]
                product_sum = np.sum(
                    salience * Smooth.to_local_average_cents.cents_mapping[start:end])
                weight_sum = np.sum(salience)
                return product_sum / weight_sum
            else:
                return 0
        if salience.ndim == 2:
            return np.array([Smooth.to_local_average_cents(salience[i, :]) for i in
                            range(salience.shape[0])])

        raise Exception("label should be either 1d or 2d ndarray")

    def to_local_average_cents_359(salience, center=None):
        if not hasattr(Smooth.to_local_average_cents, 'cents_mapping'):
            # the bin number-to-cents mapping
            Smooth.to_local_average_cents.cents_mapping = (
                    np.linspace(0, 7160, 359) + 2051.1487628680297)
            # to_local_average_cents.cents_mapping = np.insert(to_local_average_cents.cents_mapping,0,0)
        
        if salience.ndim == 1:

            if center is None:
                center = int(np.argmax(salience))
            if center != 0:
                salience = salience[1:]
                center = center-1
                start = max(0, center - 4)
                end = min(len(salience), center + 5)
                salience = salience[start:end]
                product_sum = np.sum(
                    salience * Smooth.to_local_average_cents.cents_mapping[start:end])
                weight_sum = np.sum(salience)
                return product_sum / weight_sum
            else:
                return 0
        if salience.ndim == 2:
            return np.array([Smooth.to_local_average_cents(salience[i, :]) for i in
                            range(salience.shape[0])])

        raise Exception("label should be either 1d or 2d ndarray")

    def to_local_average_cents_360(salience, center=None):
        """
        find the weighted average cents near the argmax bin
        """

        if not hasattr(Smooth.to_local_average_cents_360, 'cents_mapping'):
            # the bin number-to-cents mapping
            Smooth.to_local_average_cents_360.cents_mapping = (
                    np.linspace(0, 7180, 360) + 1997.3794084376191)

        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = np.sum(
                salience * Smooth.to_local_average_cents_360.cents_mapping[start:end])
            weight_sum = np.sum(salience)
            return product_sum / weight_sum
        if salience.ndim == 2:
            return np.array([Smooth.to_local_average_cents_360(salience[i, :]) for i in
                            range(salience.shape[0])])

        raise Exception("label should be either 1d or 2d ndarray")

class feature():
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

    def get_label(path):
        pitch = []
        ref_cent = []
        # pitch.append(f0)
        with open(path, mode="r") as file:
            for line in file.readlines():
                
                if 'ptdb' in config.data_name :
                    x = float(line.split(" ")[0])
                    if x != 0:
                        cent = Convert.convert_hz_to_cent(x)
                    else:
                        cent = 0
                    hz=x

                elif 'ikala'in config.data_name  or 'mir1k' in config.data_name :
                    x = float(line.split(" ")[0])
                    x = Convert.convert_semitone_to_hz(x)
                    if x >= 10:
                        hz = x
                        cent = Convert.convert_hz_to_cent(x)
                    else:
                        hz = 0
                        cent = 0

                elif 'mdb' in config.data_name :
                    hz = float(line.split(",")[1].split("\n")[0])
                    if hz > 0:
                        cent = Convert.convert_hz_to_cent(hz)
                    else:
                        cent = 0

                pitch.append(hz) 
                ref_cent.append(cent)               
        return pitch,ref_cent

    def get_logger(level=logging.INFO):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.path.split(os.path.abspath(__file__))[0] + '/Logs/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # print(log_path)
        log_name = log_path + rq + '.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        logger.addHandler(fh)
        return logger


class Note():
    def nsynth(label,pitch,cent_tolerance=50):
        # label = label.tolist()
        # pitch = pitch.tolist()
        label = label[:min(len(label),len(pitch))]
        pitch = pitch[:min(len(label),len(pitch))]

        ref_cent,pitch_cent = [],[]
        for i in label:
            if i >0:
                j = Convert.convert_hz_to_cent(i)
            else:
                j = 0
            ref_cent.append(j)
        for i in pitch:
            if i > 0:
                j = Convert.convert_hz_to_cent(i)
            else:
                j = 0
            pitch_cent.append(j)
        
        sum = 0
        for i in range(len(label)):
            if abs(ref_cent[i] - pitch_cent[i]) <= cent_tolerance:
                sum += 1
        
        return sum / len(label)

class pitch_signal():
    #计算每一个音频的正确率tensor
    def evaluation(outputs, labels):
        bias = 0
        correct = 0
        # pred = outputs.argmax(dim=1)
        for i in range(labels.shape[0]):
            if labels[i] - bias <= outputs[i] <= labels[i] + bias:
                correct += 1
        return correct

    def GPE(outputs, labels):
        bias = 537*0.2
        correct = 0
        # pred = outputs.argmax(dim=1)

        for i in range(labels.shape[0]):
            if labels[i] - bias <= outputs[i] <= labels[i] + bias:
                correct += 1
        return correct

    def get_abs(output, label):
        return torch.sum(torch.abs(output.argmax(dim=1).view(-1) - label))


    def MAE(total_abs, total_num):
        return float(total_abs.item()) / float(total_num)

class Time():
    def time_costing(func):
        def core(*args, **kwargs):
            start = time()
            func(*args, **kwargs)
            final = time()

            print('time costing:', final-start)
            return func
        return core

if __name__ == '__main__':

    # f0 = 45
    # b=1200
    # cent = Formula.convert_semitone_to_cent(f0)
    # print(cent)
    # c=Formula.convert_cent_to_semitone(b)
    # print(c)
    # d = Formula.convert_hz_to_bin(8373)
    # print(d)
    # e = Formula.convert_semitone_to_bin(46)
    # print(e)
    # print(Convert.convert_hz_to_bin(13))
    # print(Convert.convert_hz_to_cent(13))
    # print(Convert.convert_hz_to_semitone(11.6))
    # b= math.log(4 , 2)
    # print(b)

    ref_freq = np.array([100,122,123,111,122,110,0])
    est_freq = np.array([100,122,0,110,122,110,0])
    ref_cent = np.array([100,0,200,120,120,120,0])
    est_cent = np.array([110,90,249,110,150,180,0])
    c=Music.all_mir_eval(ref_freq, ref_cent, est_freq, est_cent,
                        cent_tolerance=50)

    print(c)
    print("")
    ref_voicing = np.array([1,2,3,4,5,0,1])
    est_voicing = np.array([0,0,3,4,5,1,0])
    a = Sound.all_mir_eval(ref_voicing, est_voicing)
    print(a)