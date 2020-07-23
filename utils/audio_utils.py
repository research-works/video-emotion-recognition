from PIL import Image
import numpy as np
import librosa
import sklearn.preprocessing as preprocessing
import scipy.signal as signal
import os


def resize(X, width, height):
    image = Image.fromarray(X)
    image = image.resize((width, height), resample = Image.BILINEAR)
    return np.array(image)

def preprocess_audio(input_wav_path, output_dir_path):
    FFT_WINDOW_SIZE = 25 # ms
    HOP_LENGTH = 10 # ms
    SAMPLE_RATE = 44100
    OFFSET = 0.8
    DURATION = 2
    F_MIN = 20
    F_MAX = 8000
    N_MELS = 64

    # RESIZED_WIDTH = 299
    # RESIZED_HEIGHT = 299

    data, sr = librosa.load(input_wav_path, sr = SAMPLE_RATE, offset = OFFSET, duration = DURATION)
    n_fft = int(sr * (FFT_WINDOW_SIZE/1000))
    hop_length = n_fft - int(sr * (HOP_LENGTH/1000))

    S = librosa.feature.melspectrogram(data, sr = sr, n_fft = n_fft, hop_length = hop_length, window = signal.hamming, fmin = F_MIN, fmax = F_MAX, n_mels = N_MELS)
    S_DB_static = librosa.power_to_db(S)
    S_DB_delta = librosa.feature.delta(S_DB_static)
    S_DB_delta2 = librosa.feature.delta(S_DB_delta)

    # S_DB_static = resize(S_DB_static, RESIZED_WIDTH, RESIZED_HEIGHT) 
    # S_DB_delta = resize(S_DB_delta, RESIZED_WIDTH, RESIZED_HEIGHT)
    # S_DB_delta2 = resize(S_DB_delta2, RESIZED_WIDTH, RESIZED_HEIGHT)

    S_DB_static = preprocessing.normalize(S_DB_static)
    S_DB_delta = preprocessing.normalize(S_DB_delta)
    S_DB_delta2 = preprocessing.normalize(S_DB_delta2)

    S_input = np.zeros((S_DB_static.shape[0], S_DB_static.shape[1], 3))
    S_input[:,:,0] = S_DB_static
    S_input[:,:,1] = S_DB_delta
    S_input[:,:,2] = S_DB_delta2


    if(not(os.path.exists(output_dir_path))):
        os.makedirs(output_dir_path)
    file_name = os.path.basename(input_wav_path).split('.')[0] + '.npy'
    print('Saving to file : ' + output_dir_path + '/' + file_name)
    with open(output_dir_path + '/' + file_name, "wb") as npy_file:
        np.save(npy_file, S_input)
    
    return S_input
