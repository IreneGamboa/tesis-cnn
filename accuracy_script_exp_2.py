import os

import json
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read, UTCDateTime
from obspy.io.sac.util import get_sac_reftime
from obspy.signal.filter import lowpass, bandpass, highpass
import datetime


models_file_path = '/home/igamboa/thesis/data/models_exp_2.json'
sac_files_path = '/home/igamboa/thesis/data/data_ovsicori'
cnn_files_path = '/home/igamboa/thesis/results/exp_2'

def rolling_window(a, window, step_size, padding=True, copy=True):
    if copy:
        result = a.copy()
    else:
        result = a
    if padding:
        result = np.hstack((result, np.zeros(window)))
    shape = result.shape[:-1] + (result.shape[-1] - window + 1 - step_size, window)
    strides = result.strides + (result.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(result, shape=shape, strides=strides)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def picker_p_predict(file, scaler, normalizer, cnn, size):
    p_is_defined = False
    tr = read(file)[0]
    if tr.stats.npts >= size :
        tr.filter('bandpass', freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)
        tr.normalize()

        if 'a' in tr.stats.sac and tr.stats.sac.a is not None:
            picker_p = int(tr.stats.sac.a * tr.stats.sampling_rate)
            p_is_defined = True

        if not p_is_defined:
            return 1, 1

        data = tr.data

        #Escalar datos antes de hacer el window
        if scaler == 'min_max_scaler':
            scaler = MinMaxScaler()
        elif scaler == 'standard_scaler':
            scaler = StandardScaler()
        elif scaler == 'min_max_scaler_1':
            scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(tr.data.reshape(-1, 1))
        x = scaler.transform(tr.data.reshape(1, -1))[0]
        if normalizer:
            normalizer = Normalizer().fit(tr.data.reshape(-1, 1))
            x = normalizer.transform(x.reshape(1, -1))[0]

        input_dataset = rolling_window(x, size, 1)

        input_dataset = np.reshape(input_dataset, (len(input_dataset), size, 1))
        output = [y[1] for y in cnn.predict(input_dataset)]
        output_running_mean = running_mean(output, size)
        guess = np.argmax(output_running_mean)

        # plt.plot(tr.times(), data)
        #
        # plt.axvline(x=(guess*tr.stats.delta) , color='green', drawstyle ="steps-pre")
        print("guess:", guess)
        print("picker_p:", picker_p)

        return guess, picker_p

models = json.loads(open(models_file_path).read())
ape = {}
n = 0

for file_name in os.listdir(sac_files_path):
    file_name = os.path.join(sac_files_path, file_name)
    # fig = plt.figure(figsize=(15,50))
    # fig.subplots_adjust(hspace=2.5, wspace=0.4)
    for i, model in enumerate(models):
        cnn = load_model(os.path.join(cnn_files_path, '{}.h5'.format(model['name'])))
        size = model['input_window_size']
        scaler = model['scaler']
        normalizer = model['normalizer']
        # ax = fig.add_subplot(35, 1, i+1)
        # ax.title.set_text('{}'.format(model['name']))
        guess, correct = picker_p_predict(file_name, scaler, normalizer, cnn, size)
        ape_calc = np.abs(guess - correct)/float(correct)
        if model['name'] not in ape:
            ape[model['name']] = ape_calc
        else:
            ape[model['name']] = ape[model['name']] + ape_calc
    # name = os.path.splitext(os.path.basename(file_name))[0]
    # plt.savefig(output_path.format(name))
    # plt.close()
    n = n + 1

for key, value in ape.items():
    ape[key] = value/float(n)
with open('/home/igamboa/thesis/ape_result_total.json', 'w') as ape_result:
    ape_result.write(json.dumps(ape))
print(ape)
