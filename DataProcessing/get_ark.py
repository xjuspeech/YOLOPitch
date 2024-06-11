import numpy as np
import os
import random
from formula_all import *

#ground truth path
path ='.../MDB-stem-synth/annotation_stems'
#ark path 
label_path = '/home/lxf521/data/MDB-stem-synth/data/all_label'

if not os.path.exists(label_path):
    os.makedirs(label_path)
with open(os.path.join(label_path,"train_lable.ark"),mode= "w") as file:
    pass
with open(os.path.join(label_path,"test_lable.ark"),mode= "w") as file:
    pass
with open(os.path.join(label_path,"cv_lable.ark"),mode= "w") as file:
    pass

train_count,test_count,cv_count=0,0,0
for root, dirs,files in os.walk(path):
    for f0 in files:
        # if f0.endswith('f0'):
        if f0.endswith('csv'):
            pitch = []
            pitch.append(f0)
            with open(root + '/' + f0, mode="r") as file:
                for line in file.readlines():
                    # x = round(float(line.split(" ")[0]))
                    x = line.split(",")[1].split('\n')[0]
                    hz = float(x)
                    # hz = Convert.convert_semitone_to_hz(x)
                    if hz >=10:
                        bin = Convert.convert_hz_to_bin(hz)
                    else:
                        bin = 0
                    
                    pitch.append(str(round(bin)))

            massage = " ".join(pitch) + '\n'

            #%% 0.8 0.13 0.07
            a = random.uniform(0,1)

            if a<0.8:
                train_count+=1
                with open(label_path + '/' + "train_lable.ark", mode= "a+") as file:

                    file.write(massage)
            elif a >=0.8 and a<0.93:
                test_count+=1
                with open(label_path + '/' + "test_lable.ark", mode= "a+") as file:

                    file.write(massage)
            else:
                cv_count+=1
                with open(label_path + '/' + "cv_lable.ark", mode= "a+") as file:
                    
                    file.write(massage)

print('ok')
